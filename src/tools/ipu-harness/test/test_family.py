"""A family's registry.toml: entries over a template kernel (E06-17)."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ipu_emu.errors import ConfigurationError
from ipu_harness import kernel_bin, kernel_test, registry_bzl
from ipu_harness.cases import expand, inputs_for
from ipu_harness.family import (
    ENTRY_SEPARATOR,
    invalid_name,
    load_kernel,
    read_registry,
    render_bzl,
    split_ref,
)
from ipu_harness.manifest import ManifestError
from ipu_harness.params import plan_case
from ipu_harness.registry import Registry
from ipu_isa.formats import ElementFormat

TEMPLATE = """\
[kernel]
op = "proj"
formats = ["fp32", "int8"]

[params]
k     = { type = "int", min = 1, max = 512 }
n_out = { type = "int", min = 1, max = 512 }
act   = { type = "str", choices = ["none", "silu"], default = "none" }

[inputs.x]
shape = ["k", 128]

[outputs.y]
shape = ["n_out", 128]

[abi]
x = "CR2"
y = "CR3"
k = "CR4"

[cases.default]
k = 128
n_out = 128

[cases.edge]
k = 1
n_out = 1
"""

#: Three entries; line 1 is the first [[kernel]].
REGISTRY = """\
[[kernel]]
name = "proj_ffn1_192"
template = "proj"
params = { k = 192, n_out = 384, act = "silu" }
doc = "Layer 4 FFN1"

[[kernel]]
name = "proj_qkv_144"
template = "proj"
params = { k = 144, n_out = 432 }
formats = ["fp32"]

[[kernel]]
name = "proj_out_240"
template = "proj"
params = { k = 240, n_out = 240 }
cases = ["default"]
"""


def entry_ref(registry: Path, name: str) -> str:
    return f"{registry}{ENTRY_SEPARATOR}{name}"


def family(root: Path, registry: str = REGISTRY, name: str = "projections") -> Path:
    """``root/<name>/`` with the template ``proj`` and ``registry``; returns its registry.toml."""
    template = root / name / "proj"
    template.mkdir(parents=True)
    (template / "proj.asm").write_text("HALT ;;\n")
    (template / "kernel.toml").write_text(TEMPLATE)
    path = root / name / "registry.toml"
    path.write_text(registry)
    return path


def test_three_entries_are_three_kernels_with_fixed_domains(tmp_path):
    family(tmp_path)
    registry = Registry.load(tmp_path)
    assert not registry.invalid
    assert sorted(registry.manifests) == ["proj_ffn1_192", "proj_out_240", "proj_qkv_144"]
    assert list(registry.templates) == ["proj"]  # run and tested itself, not routed
    ffn1 = registry.manifests["proj_ffn1_192"]
    assert (ffn1.op, dict(ffn1.fixed), ffn1.doc) == (
        "proj",
        {"k": 192, "n_out": 384, "act": "silu"},
        "Layer 4 FFN1",
    )
    assert ffn1.params["k"].choices == (192,)
    assert ffn1.label == "//kernels/projections:proj_ffn1_192"
    assert ffn1.asm == tmp_path / "projections" / "proj" / "proj.asm"
    assert [f.value for f in registry.manifests["proj_qkv_144"].formats] == ["fp32"]
    # The template's cases run with the entry's values.
    assert ffn1.cases["edge"].values == {"k": 192, "n_out": 384, "act": "silu"}


def test_resolve_routes_each_exact_query_to_its_entry(tmp_path):
    family(tmp_path)
    registry = Registry.load(tmp_path)
    for name, query in [
        ("proj_ffn1_192", {"k": 192, "n_out": 384, "act": "silu"}),
        ("proj_qkv_144", {"k": 144, "n_out": 432}),
        ("proj_out_240", {"k": 240, "n_out": 240}),
    ]:
        verdict = registry.resolve("proj", **query)
        assert verdict.kernel_name == name, verdict.reason
        assert verdict.label == f"//kernels/projections:{name}"
    # A shape no entry fixes is refused, though the template's domain holds it.
    # The template's default stays: leaving act out means act = "none", which the
    # SiLU entry refuses (as the old exact-match supports did) and a "none" entry takes.
    refused = registry.resolve("proj", k=192, n_out=384)
    assert not refused and "act = 'none' is not one of ['silu']" in refused.reason
    assert registry.resolve("proj", k=144, n_out=432).kernel_name == "proj_qkv_144"
    assert not registry.resolve("proj", k=192, n_out=384, act="none")
    assert not registry.resolve("proj", k=100, n_out=100)


def test_a_query_leaving_out_what_entries_fix_differently_is_refused_for_it(tmp_path):
    # n_out has no template default: each entry's fixed value stands in, so a
    # query without it is accepted by both 192 entries; routing names what is missing.
    registry_toml = REGISTRY.replace("k = 144, n_out = 432", "k = 192, n_out = 432")
    family(tmp_path, registry_toml)
    registry = Registry.load(tmp_path)
    # act = "none" by the template's default: the SiLU entry refuses it.
    verdict = registry.resolve("proj", k=192)
    assert verdict.kernel_name == "proj_qkv_144"
    toml = registry_toml.replace(
        'params = { k = 192, n_out = 384, act = "silu" }', "params = { k = 192, n_out = 384 }"
    )
    (tmp_path / "projections" / "registry.toml").write_text(toml)
    verdict = Registry.load(tmp_path).resolve("proj", k=192)
    assert not verdict
    assert "each fix n_out, which the query leaves out; give n_out" in verdict.reason


def test_a_tie_with_a_kernel_that_is_no_entry_stays_a_configuration_error(tmp_path):
    family(tmp_path)
    free = tmp_path / "other" / "free_proj"  # a folder kernel with the template's free domain
    free.mkdir(parents=True)
    (free / "free_proj.asm").write_text("HALT ;;\n")
    (free / "kernel.toml").write_text(TEMPLATE)
    with pytest.raises(ConfigurationError, match="both accept this proj query at cost 0"):
        Registry.load(tmp_path).resolve("proj", k=144, n_out=432)


def test_a_query_required_parameter_must_be_given_whatever_its_default(tmp_path):
    family(tmp_path)
    toml = tmp_path / "projections" / "proj" / "kernel.toml"
    toml.write_text(
        toml.read_text().replace('default = "none" }', 'default = "none", query_required = true }')
    )
    registry = Registry.load(tmp_path)
    refused = registry.resolve("proj", k=144, n_out=432)
    assert not refused and "the query must give act" in refused.reason
    assert registry.resolve("proj", k=144, n_out=432, act="none").kernel_name == "proj_qkv_144"
    # Cases are unaffected: they take the default (or the entry's fixed value).
    (default, *_) = expand(registry.manifests["proj_qkv_144"])
    assert default.values["act"] == "none"


def test_an_entry_chooses_its_template_cases(tmp_path):
    family(tmp_path)
    registry = Registry.load(tmp_path)
    assert list(registry.manifests["proj_out_240"].cases) == ["default"]
    assert list(registry.manifests["proj_qkv_144"].cases) == ["default", "edge"]


def test_an_entrys_cases_may_use_the_familys_shared_generator(tmp_path):
    """``generator = "../gen.py:fn"`` from a template is the family's file
    (beside registry.toml), as E07-02's gen.py:seeded_uniform."""
    path = family(tmp_path)
    toml = TEMPLATE.replace(
        "[cases.edge]",
        '[cases.made]\nk = 2\nn_out = 2\ngenerator = "../gen.py:make"\n\n[cases.edge]',
    )
    (path.parent / "proj" / "kernel.toml").write_text(toml)
    (path.parent / "gen.py").write_text(
        "import numpy as np\n\n\ndef make(plan, rng):\n"
        "    return {'x': np.full(plan.shapes['x'], 0.25)}\n"
    )
    entry = load_kernel(entry_ref(path, "proj_qkv_144"))
    assert entry.directory.parent == path.parent  # the family's folder
    case = next(c for c in expand(entry) if c.name == "made")
    plan = plan_case(entry, case.values, ElementFormat.FP32, case=case.decl)
    assert np.array_equal(inputs_for(case, plan).tensors["x"], np.full((144, 128), 0.25))


def test_load_kernel_takes_a_folder_or_an_entry(tmp_path):
    path = family(tmp_path)
    assert load_kernel(path.parent / "proj").name == "proj"
    assert load_kernel(entry_ref(path, "proj_qkv_144")).fixed["k"] == 144
    with pytest.raises(ManifestError, match=r"registry.toml:1: no entry named 'nope'"):
        load_kernel(f"{path}{ENTRY_SEPARATOR}nope")


def test_a_reference_is_a_folder_its_toml_or_an_entry():
    assert split_ref("a/b") == (Path("a/b"), None)
    assert split_ref("a/b/kernel.toml") == (Path("a/b"), None)
    assert split_ref(f"a/registry.toml{ENTRY_SEPARATOR}x") == (Path("a/registry.toml"), "x")
    assert [invalid_name(Path(p)) for p in ("f/k/kernel.toml", "f/x", "f/registry.toml")] == [
        "k",
        "x",
        None,
    ]


def test_an_unreadable_registry_is_named_when_its_entry_is_asked_for(tmp_path):
    path = family(tmp_path, "[[kernel]]\nname = 3\n")
    registry = Registry.load(tmp_path)
    assert list(registry.invalid) == [path]
    with pytest.raises(
        ConfigurationError, match=f"registries that did not read: {re.escape(str(path))}"
    ):
        registry.kernel("proj_qkv_144")


def test_the_test_entry_spells_an_entry_as_the_family_module_does():
    ref, directory = kernel_test.kernel_ref(f"k/registry.toml{ENTRY_SEPARATOR}x")
    assert ref.endswith(f"/k/registry.toml{ENTRY_SEPARATOR}x") and directory is None
    ref, directory = kernel_test.kernel_ref("k/x/kernel.toml")
    assert directory is not None and ref == str(directory) and directory.name == "x"


@pytest.mark.parametrize(
    ("old", "new", "message", "line"),
    [
        # The third entry (line 13) reuses the first's name.
        ('name = "proj_out_240"', 'name = "proj_ffn1_192"', "is also declared at line 1", 13),
        (
            'template = "proj"\nparams = { k = 144',
            'template = "proj2"\nparams = { k = 144',
            "template 'proj2' is no kernel folder in projections/",
            9,
        ),
        ('doc = "Layer 4 FFN1"', 'docs = "x"', "docs", 5),
        ('name = "proj_out_240"\n', "", "#3 needs 'name'", 13),
    ],
)
def test_a_bad_registry_names_its_line(tmp_path, old, new, message, line):
    assert old in REGISTRY
    path = family(tmp_path, REGISTRY.replace(old, new))
    with pytest.raises(ManifestError) as error:
        read_registry(path)
    assert message in str(error.value)
    assert str(error.value).startswith(f"{path}:{line}: ")


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("params = { k = 144, n_out = 432 }", "params = { k = 144, m = 1 }", "params.m is no"),
        ("params = { k = 144, n_out = 432 }", "params = { k = 999 }", "outside its domain"),
        ('formats = ["fp32"]', 'formats = ["e4"]', "non-empty subset"),
        ('cases = ["default"]', 'cases = ["edge"]', "default among them"),
        (
            'cases = ["default"]',
            'cases = ["default", "big"]',
            "cases ['big'] are not the template's",
        ),
    ],
)
def test_an_override_the_template_refuses_names_the_entrys_line(tmp_path, old, new, message):
    text = REGISTRY.replace(old, new)
    path = family(tmp_path, text)
    registry = Registry.load(tmp_path)
    # The entries load one by one: only the bad one is invalid, by <family>/<name>.
    ((key, error),) = registry.invalid.items()
    assert key.parent == path.parent
    assert message in str(error)
    entry = next(e for e in read_registry(path) if e.name == key.name)
    assert str(error).startswith(f"{path}:{entry.line}: ")
    assert len(registry.manifests) == 2
    with pytest.raises(ManifestError, match=re.escape(message)):
        registry.kernel(key.name)


def test_an_entry_named_like_a_folder_kernel_is_an_error(tmp_path):
    path = family(tmp_path, REGISTRY.replace('name = "proj_qkv_144"', 'name = "other"'))
    other = tmp_path / "projections" / "other"
    other.mkdir()
    (other / "other.asm").write_text("HALT ;;\n")
    (other / "kernel.toml").write_text(TEMPLATE)
    with pytest.raises(
        ConfigurationError, match=f"{re.escape(str(path))} .entry other.: kernel other is also"
    ):
        Registry.load(tmp_path)


def test_entries_are_unique_across_families(tmp_path):
    family(tmp_path)
    family(tmp_path, REGISTRY, name="more")
    with pytest.raises(
        ConfigurationError, match="kernel proj is also|kernel proj_ffn1_192 is also"
    ):
        Registry.load(tmp_path)


def test_registry_bzl_is_generated_and_checked(tmp_path):
    path = family(tmp_path)
    assert render_bzl(path).splitlines()[2:] == [
        "ENTRIES = [",
        '    {"name": "proj_ffn1_192", "template": "proj"},',
        '    {"name": "proj_qkv_144", "template": "proj"},',
        '    {"name": "proj_out_240", "template": "proj"},',
        "]",
    ]
    bzl = path.with_name("registry.bzl")
    assert "stale" in registry_bzl.check(path, bzl)  # missing
    assert registry_bzl.update([tmp_path]) == [bzl]
    assert registry_bzl.check(path, bzl) is None
    assert registry_bzl.update([tmp_path]) == []  # nothing stale
    assert registry_bzl.main(["check", str(path), str(bzl)]) == 0
    path.write_text(REGISTRY.replace("proj_out_240", "proj_out_241"))
    assert registry_bzl.main(["check", str(path), str(bzl)]) == 1
    assert registry_bzl.main(["nonsense"]) == 2


def test_assemble_of_an_entry_is_its_template_assembled_with_its_values(tmp_path):
    path = family(tmp_path)
    template = tmp_path / "projections" / "proj"
    (template / "proj.asm").write_text("    INC LR0 {{ k // 16 }} ;;\n    HALT ;;\n")
    entry = kernel_bin.default_bin(entry_ref(path, "proj_qkv_144"))
    assert entry != kernel_bin.default_bin(str(template / "kernel.toml"))  # k = 128 there
    (template / "kernel.toml").write_text(TEMPLATE.replace("k = 128\n", "k = 144\n", 1))
    assert entry == kernel_bin.default_bin(str(template))
    out = tmp_path / "out.bin"
    assert kernel_bin.main([entry_ref(path, "proj_qkv_144"), str(out)]) == 0
    assert out.read_bytes() == entry
