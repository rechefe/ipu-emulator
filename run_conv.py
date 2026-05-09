import sys
sys.path.insert(0, r'c:\Users\shdoo\Documents\university\Enics_projects\ipu-emulator\src\tools\ipu-emu-py\src')
sys.path.insert(0, r'c:\Users\shdoo\Documents\university\Enics_projects\ipu-emulator\src\tools\ipu-common\src')
sys.path.insert(0, r'c:\Users\shdoo\Documents\university\Enics_projects\ipu-emulator\src\tools\ipu-apps\src')
sys.path.insert(0, r'c:\Users\shdoo\Documents\university\Enics_projects\ipu-emulator\src\tools\ipu-as-py\src')

import numpy as np, tempfile
from pathlib import Path
from ipu_apps.convolutions_universal.conv_universal import ConvUniversalApp

binp = r'c:\Users\shdoo\Documents\university\Enics_projects\ipu-emulator\conv_universal.bin'

# Parameters: 32x32 input, 128 in channels, 64 out channels
rows, cols, in_ch, out_ch = 32, 32, 128, 64
np.random.seed(0)
kernel = np.random.randint(-4,4,(out_ch,in_ch,3,3), dtype=np.int8)
num_chunks = rows*cols//128; rows_per_chunk = 128//cols
inp = np.random.randint(-8,8,(in_ch, rows, cols), dtype=np.int8)
buf = bytearray()
for c in range(num_chunks):
    r0 = c*rows_per_chunk
    for ch in range(in_ch):
        buf += inp[ch, r0:r0+rows_per_chunk, :].tobytes()
inp_path = tempfile.mktemp(suffix='.bin')
Path(inp_path).write_bytes(bytes(buf))
app = ConvUniversalApp(inst_path=binp, input_path=inp_path, kernel=kernel,
    output_path=None, dtype='INT8', rows=rows, cols=cols, in_channels=in_ch, out_channels=out_ch)
print('App created, running...')
state, cycles = app.run()
print(f'{rows}x{cols} in={in_ch} out={out_ch}: {cycles} total cycles')