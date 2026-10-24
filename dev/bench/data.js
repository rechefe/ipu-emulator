window.BENCHMARK_DATA = {
  "lastUpdate": 1792841167093,
  "repoUrl": "https://github.com/rechefe/ipu-emulator",
  "entries": {
    "Times and memory (smaller is better)": [
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792496721297,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.858933,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006589,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.911025,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.262956,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.044776,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.058478,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "emulator/ipu_state_rss_delta_mb",
            "value": 0,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "import/ipu_as",
            "value": 0.148049,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.179706,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.292446,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.921875,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 241.984375,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=34.666 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 301,
            "unit": "s",
            "extra": "ci run 39259698583; whole job 321 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792583377591,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.889154,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006633,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.940439,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.264406,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.045376,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.058478,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "emulator/ipu_state_rss_delta_mb",
            "value": 0,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "import/ipu_as",
            "value": 0.146741,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.179322,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.296011,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.511719,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 241.8125,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=36.059 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 301,
            "unit": "s",
            "extra": "ci run 39259698583; whole job 321 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792669522502,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.841559,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006416,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.917035,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.262218,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.044607,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.058478,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "emulator/ipu_state_rss_delta_mb",
            "value": 0,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "import/ipu_as",
            "value": 0.140149,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.172765,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.287062,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.8125,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 242.148438,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=34.557 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 301,
            "unit": "s",
            "extra": "ci run 39259698583; whole job 321 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792755734097,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.757635,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006455,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.813862,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.232015,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.045905,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.058478,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "emulator/ipu_state_rss_delta_mb",
            "value": 0,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "import/ipu_as",
            "value": 0.128089,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.167623,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.271841,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.410156,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 240.492188,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=29.467 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 301,
            "unit": "s",
            "extra": "ci run 39259698583; whole job 321 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4dba5355a3d5f3e975909726048e69bf1a251ac4",
          "message": "Merge pull request #263 from rechefe/claude/cool-pascal-xbj444\n\nOperational AaQ: NX8 quantization, row metadata and the spec's AaQ instructions",
          "timestamp": "2026-10-24T07:31:45Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4dba5355a3d5f3e975909726048e69bf1a251ac4"
        },
        "date": 1792841163271,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.635886,
            "unit": "s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005307,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.691851,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.176867,
            "unit": "s",
            "extra": "files=132 words=8282"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.032321,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.058478,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "emulator/ipu_state_rss_delta_mb",
            "value": 0,
            "unit": "MB",
            "extra": "the first IpuState() in a fresh interpreter"
          },
          {
            "name": "import/ipu_as",
            "value": 0.114282,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.136245,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.229368,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.484375,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 241.679688,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=20.085 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 1907,
            "unit": "s",
            "extra": "ci run 40074498836; whole job 1931 s"
          }
        ]
      }
    ],
    "Rates (bigger is better)": [
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792496724113,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9611.927385,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 5026.566441,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 15.355 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 9262.045936,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.289 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 8719.836897,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.307 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7508.860455,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 5171.643525,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=34.666 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792583380706,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9285.231098,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 4988.962558,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 15.471 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 9292.919868,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.288 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 8502.212369,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.315 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7577.503373,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 4971.913711,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=36.059 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792669525516,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9810.363406,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 5218.94965,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 14.789 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 9587.708492,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.279 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 8753.638059,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.305 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7559.005822,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 5187.977,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=34.557 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "c5c04f7cc1ee37599b165d287ec50178ef4b130d",
          "message": "Merge pull request #251 from yarden-carmi/master\n\nIPU vscode extension - leftovers settings",
          "timestamp": "2026-10-19T13:23:08Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/c5c04f7cc1ee37599b165d287ec50178ef4b130d"
        },
        "date": 1792755737106,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 10897.061535,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 6084.327495,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 12.686 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 11186.320828,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.239 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 10211.891829,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.262 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 8149.658409,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 6084.111897,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=29.467 s"
          }
        ]
      },
      {
        "commit": {
          "author": {
            "name": "Eyal Rechef",
            "username": "rechefe",
            "email": "103856782+rechefe@users.noreply.github.com"
          },
          "committer": {
            "name": "GitHub",
            "username": "web-flow",
            "email": "noreply@github.com"
          },
          "id": "4dba5355a3d5f3e975909726048e69bf1a251ac4",
          "message": "Merge pull request #263 from rechefe/claude/cool-pascal-xbj444\n\nOperational AaQ: NX8 quantization, row metadata and the spec's AaQ instructions",
          "timestamp": "2026-10-24T07:31:45Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4dba5355a3d5f3e975909726048e69bf1a251ac4"
        },
        "date": 1792841166507,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 12983.466772,
            "unit": "words/s",
            "extra": "files=129 words=8256"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 8951.17081,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 8.623 s"
          },
          {
            "name": "emulator/faithful/fully_connected.int8/cycles_per_s",
            "value": 16245.97559,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.165 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 14971.719631,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.179 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 10513.595316,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 8926.194689,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=20.085 s"
          }
        ]
      }
    ]
  }
}