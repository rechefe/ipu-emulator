"""Registry mechanics, and a generic conformance suite every kernel inherits.

The registry's whole value is that its answers match reality, so most of these
tests are consistency checks against the kernels themselves rather than
assertions about hardcoded strings. Crucially, the conformance tests are
written against *whatever is registered* -- a newly added kernel is verified
automatically, and a kernel whose ``supports`` over-claims fails here rather
than in a user's hands.
"""

from __future__ import annotations

import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from ipu_as.lark_tree import assemble_to_bin_file

import ipu_apps.kernel_registry.registry as registry
from ipu_apps.kernel_registry.layers import _ADAPTERS
from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_apps.kernel_registry import (
    KernelSpec,
    ShapeBundle,
    UnsupportedLayer,
    boundaries,
    discover,
    flatten_to_matrix,
    from_layer,
    kernels,
    load,
    lookup_layer,
    operations,
    register_layer,
    report,
    resolve,
    yes,
)

_APP_SRC = Path(__file__).resolve().parents[1] / "src/ipu_apps"


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


# -- discovery --------------------------------------------------------------


def test_discovery_finds_the_softmax_kernels():
    names = {k.name for k in kernels("softmax")}
    assert names == {
        "softmax_rows",
        "softmax_rows_partial",
        "softmax_rows_long",
        "softmax_columns",
        "softmax_columns_packed",
    }


def test_discovery_reports_nothing_skipped():
    """A module that fails to import is a kernel silently missing from
    coverage, so discovery records it rather than swallowing it. The tree must
    currently be clean."""
    assert load().skipped == ()


def test_discovery_is_recursive():
    """Kernels sit one level below `softmax`, and the convolution family nests
    three levels deep, so discovery must not stop at the first level."""
    found = discover("ipu_apps.softmax")
    assert {k.name for k in found.specs} == {k.name for k in kernels("softmax")}


def test_registry_identity_example_loads_runs_and_reads_memory():
    """The built-in boilerplate is executable, not only a discoverable spec."""
    verdict = resolve("identity", shape=(3, 128))
    assert verdict.supported and verdict.app_name == "identity"
    _, cycles = run_case(verdict.app_name, load_cases(verdict.app_name)["default"])
    assert cycles > 0


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
    verdict = resolve("softmax", shape=(0, 8), dim=1)
    assert not verdict
    assert "must be >= 1" in verdict.reason


def test_overlapping_claims_resolve_by_cost_not_discovery_order():
    """softmax_rows_partial genuinely handles n == 128 (P=1), so both it and
    softmax_rows claim that width. The specialised kernel must win, and the
    other must be reported as an alternative rather than silently dropped."""
    verdict = resolve("softmax", shape=(8, 128), dim=1)
    assert verdict.app_name == "softmax_rows"
    assert "softmax_rows_partial" in verdict.alternatives


def test_a_kernels_own_support_reason_reaches_the_verdict():
    """`yes("...")` says something specific about *this* query; a verdict that
    always fell back to `explain` would discard it."""
    spec = KernelSpec(
        name="reasoned", op="reasoned_op", app_class=object,
        supports=lambda **p: yes("claimed because the input is already aligned"),
        build=lambda **p: {}, explain=lambda **p: "generic explanation",
    )
    with _registered(spec) as pkg:
        assert resolve("reasoned_op", package=pkg).reason == (
            "claimed because the input is already aligned"
        )


def test_a_kernel_that_claims_without_a_reason_falls_back_to_explain():
    spec = KernelSpec(
        name="silent", op="silent_op", app_class=object,
        supports=lambda **p: yes(), build=lambda **p: {},
        explain=lambda **p: "generic explanation",
    )
    with _registered(spec) as pkg:
        assert resolve("silent_op", package=pkg).reason == "generic explanation"


def test_a_missing_parameter_is_a_refusal_not_a_crash():
    """Callers branch on the verdict, so an omitted parameter must come back as
    one -- specs index ``**params``, so without a declared contract it would
    escape as a raw KeyError."""
    verdict = resolve("softmax")
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
        name="buggy", op="buggy_op", app_class=object, supports=_buggy,
        build=lambda **p: {}, explain=lambda **p: "ok",
        requires=("shape", "dim"),
    )
    with _registered(spec) as pkg:
        with pytest.raises(KeyError, match="typo"):
            resolve("buggy_op", package=pkg, shape=(8, 128), dim=1)


def test_identical_refusals_are_reported_once():
    """A query that fails to normalise at all is refused in the same words by
    every kernel; repeating the sentence five times buries it."""
    verdict = resolve("softmax", shape=(4, 5, 6), dim=1)
    assert not verdict
    assert verdict.reason.count("interior axis") == 1
    # ...and the reader is still told it was unanimous, not one kernel's quirk.
    assert "every kernel" in verdict.reason


def test_a_broken_bundle_is_disclosed_rather_than_swallowed():
    """Everything else here refuses to hide a reinterpretation. A bundle helper
    that raises must not cost the caller that disclosure silently."""
    def _explode(**params):
        raise RuntimeError("bundle is broken")

    spec = KernelSpec(
        name="bad_bundle", op="bad_bundle_op", app_class=object,
        supports=lambda **p: yes(), build=lambda **p: {},
        explain=lambda **p: "ok", bundle=_explode,
    )
    with _registered(spec) as pkg:
        verdict = resolve("bad_bundle_op", package=pkg)
    assert verdict.supported
    assert any("NOT disclosed" in c for c in verdict.caveats)


@pytest.mark.parametrize("shape,dim,expected", [
    ((8, 64), 1, "softmax_rows_partial"),
    ((8, 128), 1, "softmax_rows"),
    ((8, 300), 1, "softmax_rows_long"),
    ((8, 32), 0, "softmax_columns_packed"),
    ((8, 300), 0, "softmax_columns"),
])
def test_routing(shape, dim, expected):
    assert resolve("softmax", shape=shape, dim=dim).app_name == expected


# -- shapes: flattening and provenance --------------------------------------


def test_rank3_is_flattened_and_the_reshape_is_disclosed():
    """Flattening is done on the caller's behalf, so it must be stated in the
    verdict -- a silent reshape is the failure mode this guards."""
    verdict = resolve("softmax", shape=(4, 2, 128), dim=-1)
    assert verdict.app_name == "softmax_rows"
    assert verdict.kwargs["rows"] == 8
    assert any("flattened" in n for n in verdict.shapes.notes)


def test_derived_shapes_are_marked_as_derived():
    """A shape the registry computed must not read as one the caller asserted."""
    verdict = resolve("softmax", shape=(8, 128), dim=1)
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
    torch = pytest.importorskip("torch")
    layer = torch.nn.Softmax(dim=1)
    assert lookup_layer(layer, (32, 300)).app_name == (
        resolve("softmax", shape=(32, 300), dim=1).app_name
    )


@pytest.mark.parametrize("layer_name", ["LogSoftmax", "Softmin"])
def test_softmax_neighbours_are_refused(layer_name):
    """These share Softmax's signature and sit beside it in torch, so a
    permissive adapter would route them to a softmax kernel and return
    confidently wrong numbers."""
    torch = pytest.importorskip("torch")
    layer = getattr(torch.nn, layer_name)(dim=1)
    with pytest.raises(UnsupportedLayer, match="not softmax"):
        lookup_layer(layer, (8, 64))


def test_softmax_without_an_explicit_dim_is_refused():
    torch = pytest.importorskip("torch")
    with pytest.raises(UnsupportedLayer, match="does not state which axis"):
        lookup_layer(torch.nn.Softmax(), (8, 64))


def test_unknown_layer_names_the_registered_adapters():
    class NotALayer:
        pass

    with pytest.raises(UnsupportedLayer, match="no adapter for layer type"):
        from_layer(NotALayer(), (8, 8))


def test_adapter_registration_is_additive():
    """Supporting a new layer type must not require editing the core."""

    class MadeUpLayer:
        dim = 1

    @register_layer("MadeUpLayer")
    def _adapt(layer, input_shape):
        return "softmax", {"dim": layer.dim, "shape": input_shape}

    try:
        assert lookup_layer(MadeUpLayer(), (8, 128)).app_name == "softmax_rows"
    finally:
        # The adapter table is process-global; leaving this behind would let
        # one test change what `adapters()` reports to every later one.
        _ADAPTERS.pop("MadeUpLayer", None)


def test_adapter_declared_beside_its_kernel_is_found():
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
        (pkg / "probe_kernel" / "__init__.py").write_text(
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

        sys.path.insert(0, tmp)
        try:
            verdict = lookup_layer(ProbeLayer(), (8, 128), package="adapter_probe_apps")
            assert verdict.app_name == "probe_kernel"
        finally:
            sys.path.remove(tmp)
            _ADAPTERS.pop("ProbeLayer", None)
            for name in [m for m in sys.modules if m.startswith("adapter_probe_apps")]:
                del sys.modules[name]


# -- coverage reporting -----------------------------------------------------


def test_boundaries_report_the_real_routing_switchover():
    """Coverage tables are probed from the kernels, so they cannot describe
    behaviour the registry does not have."""
    runs = boundaries(
        "softmax", "shape", range(120, 136),
        build=lambda n: (8, n), dim=1,
    )
    winner_at = {}
    for run in runs:
        for n in range(run.start, run.end + 1):
            winner_at[n] = run.kernel

    assert winner_at[127] == "softmax_rows_partial"
    assert winner_at[128] == "softmax_rows"      # exact-width specialisation
    assert winner_at[129] == "softmax_rows_long"
    # Three regimes over this range, collapsed into contiguous runs.
    assert [r.kernel for r in runs] == [
        "softmax_rows_partial", "softmax_rows", "softmax_rows_long",
    ]


def test_report_lists_every_kernel():
    text = report()
    for spec in kernels():
        assert spec.name in text


# -- conformance: a kernel must actually do what it claims ------------------
#
# Sampled across each kernel's own declared domain, so a new kernel inherits
# this verification by registering, and an over-claiming `supports` fails here.

_CONFORMANCE_SHAPES = [
    (6, 8), (6, 20), (6, 64), (5, 100), (4, 128),
    (3, 129), (3, 200), (2, 256), (7, 300),
    (9, 16), (12, 32), (33, 64), (17, 65), (10, 127),
]


def _reference(x: np.ndarray, axis: int) -> np.ndarray:
    z = np.exp(x - x.max(axis=axis, keepdims=True))
    return z / z.sum(axis=axis, keepdims=True)


@pytest.mark.parametrize("shape", _CONFORMANCE_SHAPES)
@pytest.mark.parametrize("dim", [0, 1])
def test_resolved_kernel_computes_the_operation(shape, dim):
    """End-to-end: whatever the registry picks must actually compute softmax.

    This is the check that keeps `supports` honest. It runs the kernel the
    registry chose, on the shape it was asked about, and compares against
    numpy -- so a kernel cannot claim a domain it mishandles.
    """
    verdict = resolve("softmax", shape=shape, dim=dim)
    assert verdict.supported, verdict.reason

    spec = verdict.kernel
    asm = next(_APP_SRC.rglob(spec.asm))
    x = (np.random.RandomState(sum(shape) + dim).randn(*shape) * 3.0).astype(np.float32)

    with tempfile.TemporaryDirectory() as tmp:
        inst = Path(tmp) / "prog.bin"
        assemble_to_bin_file(asm.read_text(), str(inst))
        inp = Path(tmp) / "in.bin"
        outp = Path(tmp) / "out.bin"
        inp.write_bytes(x.tobytes())

        spec.app_class(
            inst_path=inst, input_path=inp, output_path=outp, **verdict.kwargs
        ).run(max_cycles=20_000_000)

        out = np.frombuffer(outp.read_bytes(), dtype=np.float32)

    # Output layout must equal input layout -- see test_softmax_layout_roundtrip.
    assert out.size == x.size, f"{spec.name} wrote {out.size} elements for {x.size}"
    assert np.abs(out.reshape(shape) - _reference(x, dim)).max() < 1e-4


def test_constructor_guards_reject_what_supports_refuses():
    """A kernel's ctor must not accept a shape its spec refuses.

    Guards delegate to ``SPEC.guard``, so this pins that they cannot drift back
    apart -- the failure this design exists to prevent.

    The check is one-directional on purpose. Some kernels cannot express the
    refused case in their constructor at all: ``softmax_rows`` takes only
    ``rows`` because its 128-element width is fixed by the .asm, so there is no
    argument on which to reject a 300-wide query. Those are skipped rather than
    asserted, since "constructs successfully" carries no information there.
    """
    for spec in kernels("softmax"):
        for shape in [(8, 8), (8, 64), (8, 128), (8, 300), (200, 16)]:
            for dim in (0, 1):
                claimed = spec.check(shape=shape, dim=dim).ok
                kwargs = spec.build(shape=shape, dim=dim)
                if any(v is None for v in kwargs.values()):
                    # This query is on the other axis, so `build` has nothing
                    # meaningful to produce. `supports` must already have said
                    # no -- that agreement is the property under test.
                    assert not claimed, (
                        f"{spec.name} claims shape={shape} dim={dim} but cannot "
                        f"build ctor kwargs for it: {kwargs}"
                    )
                    continue
                # Only meaningful when the ctor actually receives the dimension
                # the spec refused on -- otherwise it has nothing to check.
                describes_shape = {"n", "width"} & set(kwargs)
                if claimed or not describes_shape:
                    continue
                with pytest.raises(ValueError):
                    spec.app_class(
                        inst_path="x", input_path="y", output_path=None, **kwargs
                    )
