# ipu-apps

IPU application test harnesses — Python ports of the C test harnesses.

## Framework

Subclass `IpuApp`, write `setup` and `teardown`, call `run`:

```python
from ipu_apps import IpuApp

class MyApp(IpuApp):
    def setup(self, state):
        load_binary_to_xmem(state, self.data_path, 0x0000, 128)
        state.regfile.set_cr(0, 0x0000)

    def teardown(self, state):
        if self.output_path:
            dump_xmem_to_binary(state, self.output_path, 0x1000, 128, 1)
```

Extra `__init__` kwargs are stored as attributes automatically:

```python
app = MyApp(inst_path="program.bin", data_path="data.bin", output_path="out.bin")
state, cycles = app.run()
```

## Existing apps

### Fully Connected

Port of `fully_connected.c` — loads inputs/weights, transposes weights,
runs the FC assembly, dumps output activations.

```python
from ipu_apps.fully_connected import FullyConnectedApp

app = FullyConnectedApp(
    inst_path="fc.bin",
    inputs_path="inputs.bin",
    weights_path="weights.bin",
    output_path="output.bin",
    dtype="INT8",
)
state, cycles = app.run()
```

```bash
bazel test //src/tools/ipu-apps:test_fully_connected
```

### MambaVision kernels

The data-movement and residual operations of `MambaVisionMixer` and of the MLP
that follows it inside `Block` (see `mamba_vision.py`), sized for the two
`mamba_vision_T` stages that contain mixer blocks — stage 3 (`d_model` 320,
`L` 196) and stage 4 (`d_model` 640, `L` 49).

| App | Reference line it implements |
|---|---|
| `reshape_token_view` | `rearrange(xz, "b l d -> b d l")` and its inverse |
| `split_xz` | `x, z = xz.chunk(2, dim=1)` |
| `concat_yz` | `y = torch.cat([y, z], dim=1)` |
| `residual_mixer` | `x = x + gamma_1 * mixer(norm1(x))` |
| `residual_mlp` | `x = x + gamma_2 * mlp(norm2(x))` |

Every tensor lives in XMEM as a stack of 128-element rows, with each logical
line (a token, or a channel) occupying `ceil(len / 128)` consecutive rows and
zero-padded in the tail. The kernels alternate between two layouts, matching
the two `rearrange` calls in the mixer: **token view** `(B, L, D)`, where a
line is one token, and **channel view** `(B, D, L)`, where a line is one
channel. `reshape_token_view` converts between them.

```python
from ipu_apps.reshape_token_view import ReshapeTokenViewApp

app = ReshapeTokenViewApp(
    inst_path="reshape.bin",
    inputs_path="t2c_in_int8.bin",
    output_path="channel_view.bin",
    stage="stage3",
    direction="t2c",
)
state, cycles = app.run()
```

```bash
bazel test //src/tools/ipu-apps:test_reshape_token_view
bazel test //src/tools/ipu-apps:test_split_xz
bazel test //src/tools/ipu-apps:test_concat_yz
bazel test //src/tools/ipu-apps:test_residual_mixer
bazel test //src/tools/ipu-apps:test_residual_mlp
bazel test //src/tools/ipu-apps:test_mamba_vision_block   # all five, chained
```

Full design notes, register maps, measured cycle counts and the remaining
optimisation opportunities are in [MAMBAVISION_KERNELS.md](MAMBAVISION_KERNELS.md).

Two implementation notes worth knowing before reading the assembly:

- **A copy is a multiply by one.** There is no move instruction; everything
  reaches `R_ACC` through the multiplier. `MULT.RC.VE`'s scalar operand is an
  `LcrIdx`, so naming `CR1` (permanently 1) supplies the scalar directly — no
  constants row in XMEM. The residuals use the same trick for `gamma`, which is
  why a layer-scaled variant costs no extra cycles.
- **A transpose is a masked broadcast.** With no shuffle network, moving
  element *i* of one line to element *j* of another means multiplying by the
  one-hot vector e_j and accumulating. `reshape_token_view` gets all 128 one-hot
  vectors from a single 512-element cyclic buffer that is zero except at
  element 128: the wrapping window starting at `128 - j` *is* e_j. Four
  prologue loads then replace 128 loads per output row.
