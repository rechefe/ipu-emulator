"""Common file I/O for kernels whose callers supply a complete FP32 XMEM image.

The input file contains every row before the output region, including weights,
constants, padding and scratch. Layout preparation belongs to cases/callers;
the harness only validates file size, loads memory, sets CRs and dumps output.
"""
from dataclasses import dataclass
import inspect
from pathlib import Path

from ipu_emu.emulator import dump_xmem_to_binary, load_binary_to_xmem
from ipu_emu.ipu import LANES, R_ACC_SIZE
from ipu_emu.xmem import XMEM_SIZE_BYTES
from ipu_apps.kernel_registry.base import IpuApp
from ipu_apps.kernel_registry import ExecutionConfig, folder_spec, kernel_folder, no, yes


def positive_shape(shape, rank):
    shape = tuple(shape)
    if len(shape) != rank or any(type(n) is not int or n < 1 for n in shape):
        raise ValueError(f"expected {rank} positive integer dimensions; got {shape}")
    return shape


def ceildiv(n, d):
    return (n + d - 1) // d


@dataclass(frozen=True)
class MemoryLayout:
    input_rows: int
    output_rows: int
    crs: dict[int, int]

    def __post_init__(self):
        if (self.input_rows + self.output_rows) * R_ACC_SIZE > XMEM_SIZE_BYTES:
            raise ValueError("preformatted input and output exceed XMEM capacity")


class MemoryApp(IpuApp):
    """A harness that is only a memory layout; subclasses define ``memory_layout``."""

    @classmethod
    def memory_layout(cls, **params) -> MemoryLayout:
        """The kernel's rows and CR map for ``params``; raise ValueError if unsupported.

        Its keyword-only parameters without defaults are the ones a query must
        carry (``memory_spec`` derives ``requires`` from them).
        """
        raise NotImplementedError(f"{cls.__name__} must define memory_layout")

    def __init__(self, *, input_path, params, **kwargs):
        super().__init__(**kwargs)
        from ipu_apps.kernel_registry.registry import _harness_spec

        spec = _harness_spec(self)
        if spec is None:
            raise TypeError(
                f"{type(self).__name__} has no SPEC; construct a kernel's App (or use "
                f"create_harness), not a shared family base class"
            )
        spec.guard(**params)
        self.layout = self.memory_layout(**params)
        self.input_path = Path(input_path)
        expected = self.layout.input_rows * R_ACC_SIZE
        if self.input_path.stat().st_size != expected:
            raise ValueError(f"preformatted input must contain {expected} bytes")

    def setup(self, state):
        load_binary_to_xmem(state, self.input_path, 0, R_ACC_SIZE,
                            max_chunks=self.layout.input_rows)
        for register, value in self.layout.crs.items():
            state.regfile.set_cr(register, value)
        state.set_cr_dstructure(valid_elements=LANES)

    def teardown(self, state):
        if self.output_path is not None:
            dump_xmem_to_binary(state, self.output_path,
                                self.layout.input_rows * R_ACC_SIZE, R_ACC_SIZE,
                                num_chunks=self.layout.output_rows)


def memory_spec(op, app_class, *, cost=0, **fields):
    """KernelSpec for a MemoryApp; ``fields`` override any default KernelSpec field.

    The kernel is named after its folder (:func:`folder_spec`), its domain is
    whatever ``memory_layout`` accepts, and ``requires`` defaults to
    ``memory_layout``'s keyword-only parameters that have no default.
    """
    def supports(**params):
        try:
            app_class.memory_layout(**params)
        except (ValueError, TypeError) as exc:
            return no(str(exc))
        return yes()

    name = kernel_folder(app_class)
    layout_params = inspect.signature(app_class.memory_layout).parameters.values()
    defaults = dict(
        op=op, supports=supports,
        requires=tuple(p.name for p in layout_params
                       if p.kind is p.KEYWORD_ONLY and p.default is p.empty),
        build=lambda **params: {"params": params},
        explain=lambda **params: f"{name} with preformatted FP32 XMEM rows",
        caveats=lambda **params: ("input includes padding, constants and scratch; output is raw tiled FP32",),
        execution=ExecutionConfig(mode="fp32"), cost=lambda **params: cost,
    )
    return folder_spec(app_class, **(defaults | fields))
