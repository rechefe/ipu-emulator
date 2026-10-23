"""Routing as a value: Registry.load / of, resolve, from_layer (E06-16)."""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness.expressions import ExpressionError
from ipu_harness.manifest import KernelManifest, ManifestError, load_manifest
from ipu_harness.registry import Registry, UnsupportedLayer, default_registry, kernel_tree

TOML = """\
[kernel]
op = "{op}"
formats = {formats}
{cost}
[params]
rows = {{ type = "int", min = {lo}, max = {hi} }}
s    = {{ type = "float", default = 1.0 }}

[inputs.x]
shape = ["rows", 128]

[outputs.y]
shape = ["rows", 128]

[abi]
x    = "CR2"
y    = "CR3"
rows = "CR4"
s    = "CR5"

[cases.default]
rows = {lo}
"""


def kernel(root: Path, name: str, *, family: str = "elementwise", **fields: object) -> Path:
    """Write kernel ``name`` under ``root/family``; ``fields`` override op, lo,
    hi, cost (absent: none) and formats in :data:`TOML`."""
    values = {"op": "scale", "lo": 1, "hi": 256, "formats": '["fp32", "e4"]', **fields}
    cost = values.pop("cost", None)
    values["cost"] = "" if cost is None else f"cost = {cost!r}\n".replace("'", '"')
    directory = root / family / name
    directory.mkdir(parents=True)
    (directory / f"{name}.asm").write_text("HALT ;;\n")
    (directory / "kernel.toml").write_text(TOML.format(**values))
    return directory


def manifests(tmp_path: Path) -> list[KernelManifest]:
    """small (rows 1..8, cost 1), wide (rows 1..256, cost 10), rows4 (exactly 4, cost 0)."""
    return [
        load_manifest(kernel(tmp_path, "small", hi=8, cost=1)),
        load_manifest(kernel(tmp_path, "wide", cost=10)),
        load_manifest(kernel(tmp_path, "rows4", lo=4, hi=4, cost=0)),
    ]


def test_resolve_picks_the_cheapest_accepting_kernel_whatever_the_order(tmp_path):
    found = manifests(tmp_path)
    for order in itertools.permutations(found):
        registry = Registry.of(order)
        assert registry.resolve("scale", rows=4).kernel_name == "rows4"
        assert registry.resolve("scale", rows=6).kernel_name == "small"
        verdict = registry.resolve("scale", rows=100)
        assert verdict.kernel_name == "wide" and verdict.alternatives == ()
    v = Registry.of(found).resolve("scale", rows=6)
    assert v.alternatives == ("wide",) and dict(v.params) == {"rows": 6, "s": 1.0}
    assert v.label == "//kernels/elementwise/small"


def test_a_query_no_kernel_takes_is_a_refusal_naming_each_reason(tmp_path):
    registry = Registry.of(manifests(tmp_path))
    verdict = registry.resolve("scale", rows=999)
    assert not verdict and "no scale kernel covers this query" in verdict.reason
    assert "rows = 999 is above its max 8" in verdict.reason
    assert "rows = 999 is above its max 256" in verdict.reason
    unknown = registry.resolve("scale", rows=4, cols=2)
    assert (
        not unknown
        and "every kernel (rows4, small, wide): cols = 2 is no parameter" in unknown.reason
    )
    other = registry.resolve("softmax", rows=4)
    assert not other and "no kernels are registered for operation 'softmax'" in other.reason


def test_a_format_in_the_query_must_be_one_the_kernel_takes_the_values_in(tmp_path):
    registry = Registry.of(manifests(tmp_path))
    # s = 1 + 2^-7 has no NX8 constant encoding (it needs 7 mantissa bits):
    # every kernel takes it in FP32 only.
    assert registry.resolve("scale", rows=4, s=1.0078125).formats == ("fp32",)
    refused = registry.resolve("scale", rows=4, s=1.0078125, format="e4")
    assert not refused and "does not take these parameters in e4" in refused.reason


def test_a_cost_expression_is_evaluated_and_a_crash_in_it_propagates(tmp_path):
    cheap_when_small = load_manifest(kernel(tmp_path, "tiled", cost="rows * 2"))
    flat = load_manifest(kernel(tmp_path, "flat", cost=20))
    registry = Registry.of([cheap_when_small, flat])
    assert registry.resolve("scale", rows=3).kernel_name == "tiled"  # 6 < 20
    assert registry.resolve("scale", rows=30).kernel_name == "flat"  # 60 > 20
    broken = load_manifest(kernel(tmp_path / "b", "broken", cost="100 // (rows - 4)"))
    with pytest.raises(ExpressionError, match="division by zero"):
        Registry.of([broken]).resolve("scale", rows=4)


def test_equal_lowest_costs_on_a_query_are_an_error_never_a_choice_by_name(tmp_path):
    # Equal constant costs are fine while no query is accepted by both.
    low = load_manifest(kernel(tmp_path, "low", hi=8))
    high = load_manifest(kernel(tmp_path, "high", lo=9))
    registry = Registry.of([low, high])
    assert registry.resolve("scale", rows=3).kernel_name == "low"
    assert registry.resolve("scale", rows=30).kernel_name == "high"
    both = load_manifest(kernel(tmp_path, "both"))
    with pytest.raises(
        ConfigurationError, match="both and low both accept this scale query at cost 0"
    ):
        Registry.of([low, high, both]).resolve("scale", rows=3)
    c = load_manifest(kernel(tmp_path / "e", "c", cost="rows"))
    d = load_manifest(kernel(tmp_path / "e", "d", cost=4))
    with pytest.raises(ConfigurationError, match="c and d both accept this scale query at cost 4"):
        Registry.of([c, d]).resolve("scale", rows=4)


def test_a_negative_cost_prefers_a_kernel(tmp_path):
    preferred = load_manifest(kernel(tmp_path, "preferred", cost=-1))
    plain = load_manifest(kernel(tmp_path, "plain"))
    assert Registry.of([plain, preferred]).resolve("scale", rows=2).kernel_name == "preferred"


def test_load_reads_the_tree_keeps_invalid_tomls_and_scopes_to_families(tmp_path):
    root = tmp_path / "kernels"
    kernel(root, "small", hi=8, cost=1)
    kernel(root, "softmax_rows", family="softmax", op="softmax")
    broken = kernel(root, "broken", family="softmax", op="softmax", cost=2)
    (broken / "kernel.toml").write_text("[kernel]\nop = 3\n")
    registry = Registry.load(root)
    assert sorted(registry.manifests) == ["small", "softmax_rows"]
    assert registry.operations() == ("scale", "softmax")
    assert list(registry.invalid) == [broken / "kernel.toml"]
    # The invalid kernel's own load error, not "unknown kernel".
    with pytest.raises(ManifestError, match="kernel.toml:2"):
        registry.kernel("broken")
    with pytest.raises(ConfigurationError, match="unknown kernel 'nope'"):
        registry.kernel("nope")
    assert sorted(Registry.load(root, families=["elementwise"]).manifests) == ["small"]


def test_two_kernels_of_one_name_are_an_error(tmp_path):
    root = tmp_path / "kernels"
    kernel(root, "same", cost=1)
    kernel(root, "same", family="other", cost=2)
    with pytest.raises(ConfigurationError, match="kernel same is also"):
        Registry.load(root)


class Conv2d:
    def __init__(self, rows: int) -> None:
        self.rows = rows


LAYERS = """\
from pathlib import Path

Path({marker!r}).write_text("imported")


def conv(layer, shape):
    return "scale", {{"rows": layer.rows * shape[0]}}


LAYER_ADAPTERS = {{"Conv2d": conv}}
"""


def adapter_package(tmp_path: Path, monkeypatch, name: str, files: dict[str, str]) -> None:
    """A package ``name`` on the import path, its ``<family>/layers.py`` files given."""
    base = tmp_path / "site" / name
    base.mkdir(parents=True)
    (base / "__init__.py").write_text("")
    for family, text in files.items():
        (base / family).mkdir()
        (base / family / "__init__.py").write_text("")
        (base / family / "layers.py").write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path / "site"))


def test_layer_adapters_are_imported_by_name_and_only_when_used(tmp_path, monkeypatch):
    root = tmp_path / "kernels"
    kernel(root, "small", hi=8, cost=1)
    kernel(root, "wide", cost=10)
    kernel(root, "rows4", family="softmax", lo=4, hi=4, cost=0)  # a family with no layers
    marker = tmp_path / "imported"
    adapter_package(
        tmp_path, monkeypatch, "kernels_a", {"elementwise": LAYERS.format(marker=str(marker))}
    )
    registry = Registry.load(root, package="kernels_a")
    registry.resolve("scale", rows=4)
    assert not marker.exists(), "routing a query imports no kernel Python"
    assert registry.from_layer(Conv2d(2), (3, 128)).kernel_name == "small"  # rows = 6
    assert marker.exists()
    assert registry.from_layer(Conv2d(20), [3, 128]).kernel_name == "wide"


def test_a_broken_layers_module_is_a_configuration_error_raised_once(tmp_path, monkeypatch):
    root = tmp_path / "kernels"
    kernel(root, "small", cost=1)
    marker = tmp_path / "runs"
    broken = f"from pathlib import Path\nPath({str(marker)!r}).open('a').write('x')\nraise RuntimeError('boom')\n"
    adapter_package(tmp_path, monkeypatch, "kernels_b", {"elementwise": broken})
    registry = Registry.load(root, package="kernels_b")
    for _ in range(2):
        with pytest.raises(
            ConfigurationError,
            match="kernels_b.elementwise.layers: cannot be imported: RuntimeError: boom",
        ):
            registry.from_layer(Conv2d(1), (1,))
    assert marker.read_text() == "x"


def test_one_layer_adapted_by_two_families_is_an_error_naming_both(tmp_path, monkeypatch):
    root = tmp_path / "kernels"
    kernel(root, "small", cost=1)
    kernel(root, "other", family="conv", op="conv", cost=1)
    adapter = "LAYER_ADAPTERS = {'Conv2d': lambda layer, shape: ('scale', {})}\n"
    adapter_package(tmp_path, monkeypatch, "kernels_c", {"conv": adapter, "elementwise": adapter})
    with pytest.raises(
        ConfigurationError,
        match="Conv2d has an adapter in both kernels_c.conv.layers and kernels_c.elementwise.layers",
    ):
        Registry.load(root, package="kernels_c").from_layer(Conv2d(1), (1,))


def test_a_layer_without_an_adapter_names_the_known_ones(tmp_path):
    registry = Registry.of(manifests(tmp_path), adapters={"Linear": lambda layer, shape: ("x", {})})
    with pytest.raises(UnsupportedLayer, match="'Conv2d' .*layers with one: Linear"):
        registry.from_layer(Conv2d(1), (1,))


def test_the_default_registry_reads_the_kernel_tree_once_and_imports_no_kernel_module():
    # The tree is in the test's runfiles (//kernels:tree); E06-13 adds kernels to it.
    before = {m for m in sys.modules if m.startswith("ipu_kernels")}
    assert (kernel_tree() / "BUILD.bazel").is_file()
    assert default_registry() is default_registry()
    assert default_registry().invalid == {}
    assert {m for m in sys.modules if m.startswith("ipu_kernels")} == before


@pytest.mark.parametrize("made", [("scale",), ["scale", {}], ("scale", [("rows", 1)])])
def test_an_adapter_that_returns_no_op_and_params_is_an_error(tmp_path, made):
    registry = Registry.of(manifests(tmp_path), adapters={"Conv2d": lambda layer, shape: made})
    with pytest.raises(ConfigurationError, match=r"not \(op, params\)"):
        registry.from_layer(Conv2d(1), (1,))


def test_a_missing_tree_is_an_error_not_an_empty_registry(tmp_path):
    with pytest.raises(ConfigurationError, match="no kernel tree there"):
        Registry.load(tmp_path / "nowhere")


# --- //kernels:query ----------------------------------------------------------------------


def test_query_answers_a_query_a_sweep_and_the_report(tmp_path, capsys):
    from ipu_harness import query

    registry = Registry.of(manifests(tmp_path))
    assert query.main(["scale", "rows=6"], registry) == 0
    out = capsys.readouterr().out
    assert out.startswith("SUPPORTED: small accepts it, at cost 1")
    assert "use:  //kernels/elementwise/small (rows=6, s=1.0)" in out and "also:  wide" in out
    assert query.main(["scale", "rows=999", "--json"], registry) == 1
    answer = json.loads(capsys.readouterr().out)
    assert answer["supported"] is False and set(answer) == {
        "supported",
        "reason",
        "app",
        "use",
        "notes",
        "alternatives",
    }
    assert query.main(["scale", "rows=n", "--sweep", "n=3..10"], registry) == 0
    assert capsys.readouterr().out.splitlines() == [
        f"{'n = 3':<28} small",
        f"{'n = 4':<28} rows4",
        f"{'n 5..8':<28} small",
        f"{'n 9..10':<28} wide",
    ]
    assert query.main([], registry) == 0
    report = capsys.readouterr().out
    assert (
        "Operations: scale" in report and "- **rows4** -- `//kernels/elementwise/rows4`" in report
    )


def test_query_reports_its_own_usage_errors_and_lets_kernel_bugs_through(tmp_path):
    from ipu_harness import query

    registry = Registry.of(manifests(tmp_path))
    with pytest.raises(SystemExit) as usage:
        query.main(["scale", "rows=4", "format=e9"], registry)
    assert usage.value.code == 2
    with pytest.raises(SystemExit):
        query.main(["scale", "rows=n", "--sweep", "n=5..2"], registry)
    broken = load_manifest(kernel(tmp_path / "b", "broken", cost="100 // (rows - 4)"))
    with pytest.raises(ExpressionError):
        query.main(["scale", "rows=4"], Registry.of([broken]))
