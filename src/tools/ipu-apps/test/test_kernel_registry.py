"""Registry mechanics, and checks every registered kernel inherits.

The registry's whole value is that its answers match reality, so most of these
tests are consistency checks against the kernels themselves rather than
assertions about hardcoded strings. The per-kernel checks (hygiene, every case
accepted by its own spec and routed) are written against *whatever is
registered*, so a newly added kernel is checked automatically.
"""

from __future__ import annotations

import contextlib
import importlib
import json
import os
import sys
import tempfile
import types
from pathlib import Path

import numpy as np
import pytest

import ipu_apps.kernel_registry.benchmarking as benchmarking
import ipu_apps.kernel_registry.manifest as manifest
import ipu_apps.kernel_registry.query as query
import ipu_apps.kernel_registry.registry as registry
import ipu_apps.kernel_registry.runner as runner
from ipu_apps.kernel_registry import (
    KernelSpec,
    ShapeBundle,
    SkippedModule,
    UnsupportedLayer,
    boundaries,
    discover,
    flatten_to_matrix,
    from_layer,
    kernel_folder,
    kernels,
    load,
    lookup_layer,
    no,
    operations,
    register_layer,
    report,
    resolve,
    yes,
)
from ipu_apps.kernel_registry.cases import case_options, load_cases, package_kernel
from ipu_apps.kernel_registry.layers import _ADAPTERS


@contextlib.contextmanager
def _registered(*specs: KernelSpec, package: str = "_synthetic"):
    """Expose synthetic specs under their own package name, then clean up.

    Lets a test pin registry *mechanics* (how a claim, a reason or a failure is
    reported) without inventing a real kernel for it.
    """
    registry._CACHE[package] = registry.Discovered(tuple(specs), ())
    try:
        yield package
    finally:
        registry._CACHE.pop(package, None)


# The mechanics below run on `identity`, the one old kernel kept until E07-25
# (E07-04), and on synthetic specs. Softmax, their example until E07-11, is
# routed by kernels/softmax.

#: The identity kernel's query, at 8 rows.
IDENTITY = ["identity", "shape=8,128"]


def _split(*, fits, cost, name):
    """A synthetic row kernel of ``split_op``: claims lengths where ``fits``."""

    def supports(**params):
        if params["n"] < 1:
            return no(f"n ({params['n']}) must be >= 1")
        return yes() if fits(params["n"]) else no(f"{name} does not take n={params['n']}")

    return KernelSpec(
        name=name,
        op="split_op",
        app_class=object,
        supports=supports,
        build=lambda **p: {},
        explain=lambda **p: name,
        requires=("n",),
        cost=lambda **p: cost(p["n"]),
    )


#: Three kernels split by row length, as the softmax row kernels were: the
#: packed one takes n <= 128 (dearer at 128), the exact one n == 128, the
#: chunked one n > 128.
SPLIT = (
    _split(fits=lambda n: n <= 128, cost=lambda n: 2.0 if n == 128 else 1.0, name="packed"),
    _split(fits=lambda n: n == 128, cost=lambda n: 0.0, name="exact"),
    _split(fits=lambda n: n > 128, cost=lambda n: 1.0, name="chunked"),
)


# -- discovery --------------------------------------------------------------


def test_discovery_finds_the_identity_kernel():
    assert {k.name for k in kernels("identity")} == {"identity"}


def test_discovery_reports_nothing_skipped():
    """A module that fails to import is a kernel silently missing from
    coverage, so discovery records it rather than swallowing it. The tree must
    currently be clean."""
    assert load().skipped == ()


def test_a_broken_family_package_is_reported_not_silently_dropped(monkeypatch):
    """A family whose __init__ cannot import hides every kernel beneath it, so
    it must appear as skipped -- a coverage hole nobody would otherwise see."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp) / "broken_family_apps"
        (root / "family" / "kernel").mkdir(parents=True)
        (root / "__init__.py").write_text("")
        (root / "family" / "__init__.py").write_text("import a_dependency_that_is_not_installed\n")
        (root / "family" / "kernel" / "__init__.py").write_text("")
        (root / "family" / "kernel" / "app.py").write_text("")
        monkeypatch.syspath_prepend(tmp)
        try:
            found = discover("broken_family_apps")
        finally:
            for name in [m for m in sys.modules if m.startswith("broken_family_apps")]:
                del sys.modules[name]
    assert [s.module for s in found.skipped] == ["broken_family_apps.family"]
    assert "a_dependency_that_is_not_installed" in found.skipped[0].error


def test_discovery_is_recursive():
    """Kernels sit one level below their family package (`reshape`), so
    discovery must not stop at the first level."""
    found = discover("ipu_apps.kernels.reshape")
    assert "identity" in {k.name for k in found.specs}


def test_every_spec_is_well_formed():
    for spec in kernels():
        assert spec.name and spec.op, spec
        assert isinstance(spec, KernelSpec)
        assert isinstance(spec.app_class, type), spec.name
        if spec.asm:
            from importlib.resources import files

            assert files(spec.resource_package).joinpath(spec.asm).is_file(), (
                f"{spec.name} declares a missing asm: {spec.asm}"
            )


def test_every_kernel_is_named_after_its_folder():
    """One kernel per folder: the folder, ``SPEC.name``, the ``.asm`` stem and
    the Bazel target are one word, which the Bazel entry point relies on."""
    for spec in kernels():
        assert spec.name == spec.resource_package.rpartition(".")[2], spec.name
        assert spec.asm == spec.name + ".asm", spec.name


def test_misplaced_harnesses_fail_with_a_reason():
    """A shared base class has no SPEC, and a class outside any package has no
    folder: both used to die with AttributeError on None."""
    from ipu_apps.kernel_registry.memory import MemoryApp

    with pytest.raises(TypeError, match="no SPEC"):
        MemoryApp(inst_path="x", input_path="y", params=dict(shape=(1, 128)))
    loose = type("Loose", (), {"__module__": "os"})  # a top-level, non-package module
    with pytest.raises(ValueError, match="kernel folder"):
        kernel_folder(loose)


def test_kernel_names_are_unique():
    names = [k.name for k in kernels()]
    assert len(names) == len(set(names))


# -- resolution -------------------------------------------------------------


def test_unknown_operation_is_refused_with_the_known_list():
    verdict = resolve("convolution_of_doom", shape=(8, 8), dim=1)
    assert not verdict
    assert "no kernels are registered" in verdict.reason
    for op in operations():
        assert op in verdict.reason


def test_refusal_aggregates_every_kernel_reason():
    """When nothing covers a query the user needs to know what was wrong, not
    merely that it failed."""
    with _registered(*SPLIT) as pkg:
        verdict = resolve("split_op", package=pkg, n=0)
    assert not verdict
    assert "must be >= 1" in verdict.reason
    assert not resolve("identity", shape=(0, 128))


def test_overlapping_claims_resolve_by_cost_not_discovery_order():
    """The packed kernel genuinely handles n == 128, so both it and the exact
    kernel claim that length. The specialised kernel must win, and the other
    must be reported as an alternative rather than silently dropped -- in
    either registration order."""
    for order in (SPLIT, SPLIT[::-1]):
        with _registered(*order) as pkg:
            verdict = resolve("split_op", package=pkg, n=128)
        assert verdict.app_name == "exact"
        assert "packed" in verdict.alternatives


def test_a_kernels_own_support_reason_reaches_the_verdict():
    """`yes("...")` says something specific about *this* query; a verdict that
    always fell back to `explain` would discard it."""
    spec = KernelSpec(
        name="reasoned",
        op="reasoned_op",
        app_class=object,
        supports=lambda **p: yes("claimed because the input is already aligned"),
        build=lambda **p: {},
        explain=lambda **p: "generic explanation",
    )
    with _registered(spec) as pkg:
        assert resolve("reasoned_op", package=pkg).reason == (
            "claimed because the input is already aligned"
        )


def test_a_kernel_that_claims_without_a_reason_falls_back_to_explain():
    spec = KernelSpec(
        name="silent",
        op="silent_op",
        app_class=object,
        supports=lambda **p: yes(),
        build=lambda **p: {},
        explain=lambda **p: "generic explanation",
    )
    with _registered(spec) as pkg:
        assert resolve("silent_op", package=pkg).reason == "generic explanation"


def test_a_missing_parameter_is_a_refusal_not_a_crash():
    """Callers branch on the verdict, so an omitted parameter must come back as
    one -- specs index ``**params``, so without a declared contract it would
    escape as a raw KeyError."""
    verdict = resolve("identity")
    assert not verdict
    assert "needs parameter" in verdict.reason
    assert "shape" in verdict.reason


def test_every_spec_declares_the_parameters_it_indexes():
    """Empty queries must either use defaults or name missing parameters."""
    for spec in kernels():
        verdict = spec.check()
        if spec.requires:
            assert not verdict.ok
            assert all(name in verdict.reason for name in spec.requires)
        elif verdict.ok:
            # Optional callbacks must also tolerate an empty query; a missing
            # required declaration would raise KeyError here.
            spec.build()
            spec.explain()
            spec.cost()
            spec.caveats()
            if spec.bundle is not None:
                spec.bundle()


def test_a_kernels_own_keyerror_is_not_reported_as_unsupported():
    """A KeyError raised *inside* a spec is a bug in that kernel. Folding it
    into a refusal would turn it into a silent routing miss."""

    def _buggy(**params):
        return {"a": 1}["typo"]

    spec = KernelSpec(
        name="buggy",
        op="buggy_op",
        app_class=object,
        supports=_buggy,
        build=lambda **p: {},
        explain=lambda **p: "ok",
        requires=("shape", "dim"),
    )
    with _registered(spec) as pkg:
        with pytest.raises(KeyError, match="typo"):
            resolve("buggy_op", package=pkg, shape=(8, 128), dim=1)


def test_identical_refusals_are_reported_once():
    """A query that fails to normalise at all is refused in the same words by
    every kernel; repeating the sentence five times buries it."""
    with _registered(*SPLIT) as pkg:
        verdict = resolve("split_op", package=pkg, n=-1)
    assert not verdict
    assert verdict.reason.count("must be >= 1") == 1
    # ...and the reader is still told it was unanimous, not one kernel's quirk.
    assert "every kernel" in verdict.reason


def test_a_broken_bundle_is_disclosed_rather_than_swallowed():
    """Everything else here refuses to hide a reinterpretation. A bundle helper
    that raises must not cost the caller that disclosure silently."""

    def _explode(**params):
        raise RuntimeError("bundle is broken")

    spec = KernelSpec(
        name="bad_bundle",
        op="bad_bundle_op",
        app_class=object,
        supports=lambda **p: yes(),
        build=lambda **p: {},
        explain=lambda **p: "ok",
        bundle=_explode,
    )
    with _registered(spec) as pkg:
        verdict = resolve("bad_bundle_op", package=pkg)
    assert verdict.supported
    assert any("NOT disclosed" in c for c in verdict.caveats)


@pytest.mark.parametrize(
    "n,expected",
    [(1, "packed"), (64, "packed"), (128, "exact"), (129, "chunked"), (300, "chunked")],
)
def test_routing(n, expected):
    with _registered(*SPLIT) as pkg:
        assert resolve("split_op", package=pkg, n=n).app_name == expected


# -- shapes: flattening and provenance --------------------------------------


def test_rank3_is_flattened_and_the_reshape_is_disclosed():
    """Flattening is done on the caller's behalf, so it must be stated in the
    verdict -- a silent reshape is the failure mode this guards."""

    def bundle(**params):
        _, _, note = flatten_to_matrix(params["shape"], params["dim"])
        return ShapeBundle.of(input=params["shape"]).with_shapes(
            derived={"output": params["shape"]}, notes=(note,) if note else ()
        )

    spec = KernelSpec(
        name="rows_kernel",
        op="rows_op",
        app_class=object,
        supports=lambda **p: yes(),
        build=lambda **p: {"rows": flatten_to_matrix(p["shape"], p["dim"])[0][0]},
        explain=lambda **p: "ok",
        bundle=bundle,
    )
    with _registered(spec) as pkg:
        verdict = resolve("rows_op", package=pkg, shape=(4, 2, 128), dim=-1)
    assert verdict.kwargs["rows"] == 8
    assert any("flattened" in n for n in verdict.shapes.notes)


def test_derived_shapes_are_marked_as_derived():
    """A shape the registry computed must not read as one the caller asserted."""
    verdict = resolve("identity", shape=(8, 128))
    assert "output" in verdict.shapes.derived_roles
    assert "input" not in verdict.shapes.derived_roles
    assert "output*" in verdict.shapes.describe()


def test_interior_axis_is_refused_rather_than_transposed():
    """Flattening around a middle axis would require re-ordering the other
    dims, silently reinterpreting the caller's memory layout."""
    with pytest.raises(ValueError, match="interior axis"):
        flatten_to_matrix((4, 8, 16), 1)


def test_1d_input_is_a_single_row():
    shape_2d, dim_2d, note = flatten_to_matrix((300,), 0)
    assert shape_2d == (1, 300) and dim_2d == 1 and note is None


def test_shape_bundle_rejects_degenerate_shapes():
    with pytest.raises(ValueError):
        ShapeBundle.of(input=())
    with pytest.raises(ValueError):
        ShapeBundle.of(input=(4, -1))


# -- layer adapters ---------------------------------------------------------


def test_layer_lookup_matches_the_direct_query():
    class CopyLayer:
        pass

    @register_layer("CopyLayer")
    def _adapt(layer, input_shape):
        return "identity", {"shape": input_shape}

    try:
        assert lookup_layer(CopyLayer(), (8, 128)).to_dict() == (
            resolve("identity", shape=(8, 128)).to_dict()
        )
    finally:
        _ADAPTERS.pop("CopyLayer", None)


def test_unknown_layer_names_the_registered_adapters():
    class NotALayer:
        pass

    with pytest.raises(UnsupportedLayer, match="no adapter for layer type"):
        from_layer(NotALayer(), (8, 8))


def test_adapter_registration_is_additive():
    """Supporting a new layer type must not require editing the core."""

    class MadeUpLayer:
        pass

    @register_layer("MadeUpLayer")
    def _adapt(layer, input_shape):
        return "identity", {"shape": input_shape}

    try:
        assert lookup_layer(MadeUpLayer(), (8, 128)).app_name == "identity"
    finally:
        # The adapter table is process-global; leaving this behind would let
        # one test change what `adapters()` reports to every later one.
        _ADAPTERS.pop("MadeUpLayer", None)


def test_adapter_declared_beside_its_kernel_is_found(monkeypatch):
    """The documented way to add an adapter must actually work.

    Adapters register as an import side effect of the package that declares
    them, so a lookup that does not discover first cannot see an adapter living
    beside its kernel -- which is exactly what `from_layer`'s own error message
    tells contributors to do.
    """
    with tempfile.TemporaryDirectory() as tmp:
        pkg = Path(tmp) / "adapter_probe_apps"
        (pkg / "probe_kernel").mkdir(parents=True)
        (pkg / "__init__.py").write_text("")
        (pkg / "probe_kernel" / "__init__.py").write_text("")
        (pkg / "probe_kernel" / "app.py").write_text(
            "from ipu_apps.kernel_registry import KernelSpec, register_layer, yes\n"
            "\n"
            "@register_layer('ProbeLayer')\n"
            "def _adapt(layer, input_shape):\n"
            "    return 'probe_op', {'shape': input_shape}\n"
            "\n"
            "SPEC = KernelSpec(name='probe_kernel', op='probe_op', app_class=object,\n"
            "                  supports=lambda **p: yes(), build=lambda **p: {},\n"
            "                  explain=lambda **p: 'probe')\n"
        )

        class ProbeLayer:
            pass

        monkeypatch.syspath_prepend(tmp)
        try:
            verdict = lookup_layer(ProbeLayer(), (8, 128), package="adapter_probe_apps")
            assert verdict.app_name == "probe_kernel"
        finally:
            _ADAPTERS.pop("ProbeLayer", None)
            for name in [m for m in sys.modules if m.startswith("adapter_probe_apps")]:
                del sys.modules[name]


# -- coverage reporting -----------------------------------------------------


def test_boundaries_report_the_real_routing_switchover():
    """Coverage tables are probed from the kernels, so they cannot describe
    behaviour the registry does not have."""
    with _registered(*SPLIT) as pkg:
        runs = boundaries("split_op", "n", range(120, 136), package=pkg)
    winner_at = {}
    for run in runs:
        for n in range(run.start, run.end + 1):
            winner_at[n] = run.kernel

    assert winner_at[127] == "packed"
    assert winner_at[128] == "exact"  # exact-length specialisation
    assert winner_at[129] == "chunked"
    # Three regimes over this range, collapsed into contiguous runs.
    assert [r.kernel for r in runs] == ["packed", "exact", "chunked"]


def test_report_lists_every_kernel():
    text = report()
    for spec in kernels():
        assert spec.name in text


def test_every_case_routes_to_a_kernel(tmp_path):
    """A kernel's cases are configurations it is known to compute, so its own
    spec must accept each one and the registry must route each one somewhere.
    It covers every kernel."""
    for spec in kernels():
        for name, case in load_cases(spec.name).items():
            workspace = tmp_path / spec.name / name
            workspace.mkdir(parents=True)
            params = case.prepare(workspace, **case.defaults).params
            assert spec.check(**params).ok, f"{spec.name}/{name}: {spec.check(**params).reason}"
            verdict = resolve(spec.op, **params)
            assert verdict.supported, f"{spec.name}/{name}: {verdict.reason}"


# -- benchmarks (`bazel run :benchmark_<package>`) ----------------------------
#
# `bazel test` builds the benchmark binaries but never runs them, so a config
# naming an option its case does not have would only fail when someone next
# benchmarks. Check every declared config against its kernel's cases instead.


def _benchmark_modules():
    import pkgutil

    import ipu_apps.kernels

    return [
        importlib.import_module(info.name)
        for info in pkgutil.walk_packages(ipu_apps.kernels.__path__, "ipu_apps.kernels.")
        if info.name.rpartition(".")[2] == "benchmark"
    ]


def test_benchmark_modules_are_found():
    assert len(_benchmark_modules()) >= 6


@pytest.mark.parametrize("module", _benchmark_modules(), ids=lambda m: m.__name__)
def test_benchmark_configs_name_real_case_options(module):
    per = getattr(module, "PER", None)
    kernel = package_kernel(module.__name__.rpartition(".")[0])
    defaults = load_cases(kernel)["default"].defaults
    assert module.CONFIGS, f"{kernel}: no benchmark configs"
    for options in module.CONFIGS:
        assert set(options) <= set(defaults), (kernel, options, sorted(defaults))
    assert per is None or per in defaults, (kernel, per)


def test_benchmark_runs_configs_through_the_cases():
    module = types.SimpleNamespace(
        __name__="ipu_apps.kernels.reshape.identity.benchmark",
        CONFIGS=[{"rows": 8}],
        PER="rows",
    )
    [row] = benchmarking.run_benchmark(module)
    assert row.label == "rows=8"
    assert row.cycles > 0 and row.per_unit == row.cycles / 8
    # MAC accounting comes from the run: identity multiplies are not MACs.
    assert 0 <= row.effective_mac_utilization <= row.lane_occupancy <= 1
    table = benchmarking.render_table([row], "rows")
    header = table.splitlines()[0]
    assert "cyc/rows" in header and "effMAC%" in header and "rows=8" in table
    # The run's alias profile reaches the row: identity's copy is A1 (MOV_RC).
    assert "A1_MOV_RC" in row.aliases
    assert "ISA aliases:" in table
    for alias, (verified, _) in row.aliases.items():
        assert alias.replace("_", " ", 1) in table and str(verified) in table


# -- query CLI (`bazel run :query`) -------------------------------------------


def test_query_without_arguments_prints_the_report(capsys):
    assert query.main([]) == 0
    assert capsys.readouterr().out.strip() == report().strip()


def test_query_resolves_name_value_parameters(capsys):
    assert query.main(IDENTITY) == 0
    assert "app:  identity" in capsys.readouterr().out


def test_query_parses_a_trailing_comma_as_a_one_d_shape(capsys):
    assert query.main(["identity", "shape=128,"]) == 1
    assert "got (128,)" in capsys.readouterr().out


def test_query_exits_nonzero_when_nothing_covers_it(capsys):
    assert query.main(["no_such_op", "shape=8,8"]) == 1
    assert "NOT SUPPORTED" in capsys.readouterr().out


def test_query_sweep_prints_the_probed_boundaries(capsys):
    assert query.main(["identity", "shape=r,128", "--sweep", "r=0..3"]) == 1  # r=0 is refused
    expected = boundaries("identity", "shape", range(4), build=lambda r: (r, 128))
    assert [run.kernel for run in expected] == [None, "identity"]
    assert capsys.readouterr().out.splitlines() == [run.render("r") for run in expected]


@pytest.mark.parametrize(
    "argv",
    [
        ["identity", "shape"],  # not NAME=VALUE
        ["identity", "shape=r,128", "rows=r", "--sweep", "r=1..5"],  # r in two parameters
        ["identity", "shape=r,128", "--sweep", "r=1-5"],  # malformed span
        ["identity", "shape=8,128", "--sweep", "r=1..5"],  # r not mentioned
        ["identity", "shape="],  # empty value
        ["identity", "shape=r,128", "--sweep", "r=5..1"],  # reversed span
    ],
)
def test_query_rejects_malformed_queries(argv):
    with pytest.raises(SystemExit) as exc:
        query.main(argv)
    assert exc.value.code == 2


def test_query_parses_booleans(capsys):
    """Left as the string "False", a flag would be truthy and flip the answer."""
    assert query._value("False") is False and query._value("true") is True
    assert query.main(["fully_connected", "shape=10,128", "dtype=fp8_e4m3", "wide_mode=false"]) == 0
    assert "SUPPORTED" in capsys.readouterr().out


# -- guards: a constructor must not accept what its spec refuses ------------
#
# The softmax conformance suite (resolve, run, compare) is
# kernels/softmax/test_conformance.py since E07-11.


def test_constructor_guards_reject_what_supports_refuses():
    """A kernel's ctor must not accept a query its spec refuses.

    Guards delegate to ``SPEC.guard``, so this pins that they cannot drift back
    apart -- the failure this design exists to prevent.
    """
    spec = registry.kernel_spec("identity")
    for shape in [(8, 128), (0, 128), (8, 64), (8,), (2, 3, 128)]:
        if spec.check(shape=shape).ok:
            continue
        with pytest.raises(ValueError):
            spec.app_class(inst_path="x", input_path="y", output_path=None, params={"shape": shape})


# -- tool output: `query --json` and the kernel manifest -----------------------


@pytest.mark.parametrize("argv", [IDENTITY, ["no_such_op", "shape=8,8"]])
def test_query_json_is_the_text_verdict(argv, capsys):
    code = query.main(argv)
    text = capsys.readouterr().out
    assert query.main([*argv, "--json"]) == code
    data = json.loads(capsys.readouterr().out)
    assert text.startswith(
        ("SUPPORTED: " if data["supported"] else "NOT SUPPORTED: ") + data["reason"]
    )
    if data["app"]:
        assert f"  app:  {data['app']}" in text
        assert f"  use:  {data['use']}(" in text
    for note in data["notes"]:
        assert f"  note: {note}" in text


@pytest.mark.parametrize(
    "argv", [["--json"], ["identity", "shape=r,128", "--sweep", "r=1..5", "--json"]]
)
def test_query_json_needs_exactly_one_query(argv):
    with pytest.raises(SystemExit) as exc:
        query.main(argv)
    assert exc.value.code == 2


def _kernel_targets():
    path = os.environ.get("IPU_KERNEL_TARGETS")
    if not path:
        pytest.skip("IPU_KERNEL_TARGETS is set by the Bazel target (it names :kernel_targets)")
    return json.loads(Path(path).read_text())


def test_manifest_joins_bazel_targets_and_the_registry():
    built = manifest.build(_kernel_targets())
    assert set(built["kernels"]) == {spec.name for spec in kernels()}
    for name, kernel in built["kernels"].items():
        assert kernel["targets"]["run"].endswith(":" + name)
        assert kernel["asm"].endswith(f"/{name}/{name}.asm")
        assert Path(kernel["asm"]).name == registry.kernel_spec(name).asm
        assert set(kernel["cases"]) == set(load_cases(name))
    assert set(built["operations"]) == set(operations())


@pytest.mark.parametrize(
    "edit, culprit",
    [
        (lambda kernels: {**kernels, "no_such_kernel": kernels["identity"]}, "no_such_kernel"),
        (lambda kernels: {k: v for k, v in kernels.items() if k != "identity"}, "identity"),
    ],
    ids=["unregistered", "untargeted"],
)
def test_manifest_refuses_targets_the_registry_does_not_know(edit, culprit):
    targets = _kernel_targets()
    with pytest.raises(ValueError, match=culprit):
        manifest.build(dict(targets, kernels=edit(targets["kernels"])))


def test_manifest_lists_a_kernel_that_failed_to_import_as_skipped(monkeypatch):
    # One broken app.py must not take every other kernel out of the manifest.
    targets = _kernel_targets()
    module = registry.kernel_spec("identity").app_class.__module__  # its package's `app`
    discovered = registry.load()
    broken = SkippedModule(module, "ImportError: broken")
    monkeypatch.setattr(
        registry,
        "load",
        lambda *a, **k: type(discovered)(
            tuple(s for s in discovered.specs if s.name != "identity"),
            (*discovered.skipped, broken),
        ),
    )
    monkeypatch.setattr(
        registry, "kernels", lambda *a, **k: tuple(s for s in kernels() if s.name != "identity")
    )
    built = manifest.build(targets)
    assert set(built["kernels"]) == {spec.name for spec in kernels()} - {"identity"}
    assert {"module": module, "error": "ImportError: broken"} in built["skipped"]


@pytest.mark.parametrize("kernel", sorted(spec.name for spec in kernels()))
def test_manifest_flags_are_the_runner_flags(kernel, capsys):
    # Every option the manifest (and so the editor) offers must parse.
    for case_name, case in load_cases(kernel).items():
        with pytest.raises(SystemExit) as exc:
            runner.main(["--kernel", kernel, "--case", case_name, "--help"])
        assert exc.value.code == 0
        help_text = capsys.readouterr().out
        for flag in [
            o[k] for o in case_options(case).values() for k in ("flag", "negated_flag") if k in o
        ]:
            assert flag in help_text


# --- Test data: one generator per case --------------------------------------

# The first six float32 values of seed 7, as little-endian bytes. Single-tensor
# cases draw with no generator and must keep exactly these inputs.
_SEED7_PREFIX = bytes.fromhex("193da13a2df5983ecc5b8cbed4fd63bf9ccae8be8cdc7dbf")


def test_random_values_without_a_generator_restarts_seed_7():
    from ipu_apps.kernel_registry.case_support import random_values

    first, again = random_values((6,)), random_values((2, 3))
    assert first.dtype == np.dtype("<f4")
    assert first.tobytes() == _SEED7_PREFIX
    assert again.tobytes() == _SEED7_PREFIX


def test_random_values_with_one_generator_draws_independent_tensors():
    from ipu_apps.kernel_registry.case_support import random_values

    rng = np.random.default_rng(7)
    draws = [random_values((6,), rng) for _ in range(3)]
    assert draws[0].tobytes() == _SEED7_PREFIX
    for i, a in enumerate(draws):
        for b in draws[i + 1 :]:
            assert not np.isin(a[:4], b).any()


# -- case_tests and the emulator-only ST.ACC warning ----------------

_TOY_PROGRAMS = {
    "uses_str_acc_reg": "SET lr0 cr8;;\nST.ACC lr0 cr0;;\nHALT;;",
    "hardware_only": "SET lr0 cr8;;\nHALT;;",
}


def _toy_case_test(monkeypatch, program, *, run_hook=None):
    """``case_tests`` for a one-case toy kernel assembled from ``program``.

    The case runs the real emulator on the assembled binary; ``run_hook``
    wraps that run to inject or suppress the warning.
    """
    import ipu_apps.kernel_registry.testing as testing
    from ipu_as.lark_tree import assemble_to_bin_file
    from ipu_emu.emulator import load_program_from_binary, run_until_complete
    from ipu_emu.ipu_state import IpuState

    def assemble(kernel, directory):
        path = Path(directory) / f"{kernel}.bin"
        assemble_to_bin_file(_TOY_PROGRAMS[program], str(path))
        return path

    def run(kernel, case, *, options, inst_path, recorder):
        state = IpuState()
        load_program_from_binary(state, inst_path)
        with run_hook() if run_hook else contextlib.nullcontext():
            run_until_complete(state)
        assert state.is_halted

    monkeypatch.setattr(testing, "package_kernel", lambda package: program)
    monkeypatch.setattr(
        testing, "load_cases", lambda kernel: {"default": types.SimpleNamespace(defaults={})}
    )
    monkeypatch.setattr(testing, "assemble_kernel", assemble)
    monkeypatch.setattr(testing, "run_case", run)
    # The toy kernel is not registered, so it has no baseline to compare with.
    monkeypatch.setattr(testing.baseline, "check", lambda *args: None)
    return testing.case_tests("toy")


#: A request whose config holds no update root: the case test compares.
_COMPARING = types.SimpleNamespace(config=types.SimpleNamespace(stash={}))


def test_case_tests_passes_a_kernel_that_uses_str_acc_reg_when_it_warns(
    monkeypatch, tmp_path_factory
):
    _toy_case_test(monkeypatch, "uses_str_acc_reg")("default", {}, tmp_path_factory, _COMPARING)


def test_case_tests_fails_a_kernel_that_uses_str_acc_reg_without_the_warning(
    monkeypatch, tmp_path_factory
):
    import warnings

    @contextlib.contextmanager
    def silence():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

    test = _toy_case_test(monkeypatch, "uses_str_acc_reg", run_hook=silence)
    with pytest.raises(pytest.fail.Exception, match="DID NOT WARN"):
        test("default", {}, tmp_path_factory, _COMPARING)


def test_case_tests_fails_a_hardware_only_kernel_if_the_warning_appears(
    monkeypatch, tmp_path_factory
):
    import warnings

    from ipu_emu.errors import DebugOnlyInstructionWarning

    @contextlib.contextmanager
    def inject():
        warnings.warn("[DEBUG ONLY] injected", DebugOnlyInstructionWarning, stacklevel=2)
        yield

    # Without the injection the hardware-only kernel passes ...
    _toy_case_test(monkeypatch, "hardware_only")("default", {}, tmp_path_factory, _COMPARING)
    # ... and with it, the warning is an error.
    test = _toy_case_test(monkeypatch, "hardware_only", run_hook=inject)
    with pytest.raises(DebugOnlyInstructionWarning, match="injected"):
        test("default", {}, tmp_path_factory, _COMPARING)


def test_debug_only_detection_reads_the_assembled_words(tmp_path):
    from ipu_apps.kernel_registry.testing import (
        kernel_uses_debug_only_instructions,
        uses_debug_only_instructions,
    )
    from ipu_as.lark_tree import assemble_to_bin_file

    for name, program in _TOY_PROGRAMS.items():
        path = tmp_path / f"{name}.bin"
        assemble_to_bin_file(program, str(path))
        assert uses_debug_only_instructions(path) is (name == "uses_str_acc_reg")
    assert kernel_uses_debug_only_instructions("fully_connected")
    assert not kernel_uses_debug_only_instructions("identity")


def test_runner_reports_a_broken_kernel_source(tmp_path, monkeypatch, capsys):
    """A kernel whose `.asm` does not assemble is reported by the runner's handler
    (E05-06: the assembler raises an AssemblerError instead of exiting)."""
    from ipu_apps.kernel_registry import cases as cases_module
    from ipu_apps.kernel_registry import runner
    from ipu_apps.kernel_registry.cases import kernel_spec, load_cases

    spec = kernel_spec("identity")
    (tmp_path / spec.asm).write_text("NOT_AN_INSTRUCTION LR0;;\n")
    monkeypatch.setattr(cases_module, "files", lambda package: tmp_path)
    case = next(iter(load_cases("identity")))

    with pytest.raises(SystemExit) as exc:
        runner.main(["--kernel", "identity", "--case", case])
    assert exc.value.code == 1
    assert capsys.readouterr().err == (
        "error: Opcode 'NOT_AN_INSTRUCTION' not found in any Inst subclass.\n"
    )
