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
