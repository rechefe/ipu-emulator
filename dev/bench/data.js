window.BENCHMARK_DATA = {
  "lastUpdate": 1795348406790,
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794311708873,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.463968,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.004123,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.499266,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.147969,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.019812,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.092614,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.013 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.107058,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.013 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.18372,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.013 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.03125,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.001 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.210938,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=17.809 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 6852,
            "unit": "s",
            "extra": "nightly run 42795124590; whole job 6889 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794396525106,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.933587,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006636,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.946718,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.322972,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.045433,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.153832,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.183302,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.299263,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.824219,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.109375,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=38.917 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 6735,
            "unit": "s",
            "extra": "nightly run 42995956842; whole job 6782 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794484180460,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.895985,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006403,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.927241,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.319608,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.045245,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.148676,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.179545,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.293336,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.020 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.867188,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.253906,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=39.250 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 6617,
            "unit": "s",
            "extra": "nightly run 43195666023; whole job 6661 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794568037099,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.927144,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006723,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.97238,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.332722,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.046065,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.156944,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.189085,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.308176,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.021 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 234.851562,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.0625,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=39.329 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 4972,
            "unit": "s",
            "extra": "nightly run 43403151835; whole job 5034 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794655049382,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.663882,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005316,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.688906,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.222332,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.023939,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.10762,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.014 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.123215,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.014 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.20995,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.014 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.355469,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 243.238281,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=24.817 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 7856,
            "unit": "s",
            "extra": "nightly run 43578698426; whole job 7891 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794742062472,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.673298,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005582,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.738013,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.219199,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.027897,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.119973,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.138633,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.235164,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.035156,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.3125,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=23.064 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 6648,
            "unit": "s",
            "extra": "nightly run 43699834909; whole job 6686 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794830877516,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.510716,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.004553,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.530997,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.160022,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.020167,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.095633,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.013 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.108702,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.013 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.180929,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.013 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.152344,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.001 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.429688,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=18.976 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 7968,
            "unit": "s",
            "extra": "nightly run 43843603755; whole job 8025 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794920896551,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.689808,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005869,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.7315,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.219339,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.0281,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.125639,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.148749,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.243571,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.128906,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.308594,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=22.753 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 8194,
            "unit": "s",
            "extra": "nightly run 44068223131; whole job 8227 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795001651623,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.796037,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006652,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.829855,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.284188,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.046724,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.130522,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.166876,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.26724,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.039062,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 243.15625,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=33.526 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 4366,
            "unit": "s",
            "extra": "nightly run 44283989569; whole job 4406 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795092377738,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.706364,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005555,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.732096,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.215884,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.028139,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.123758,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.139161,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.235254,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.017 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.152344,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.386719,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=22.961 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 8121,
            "unit": "s",
            "extra": "nightly run 44505858343; whole job 8159 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795174555887,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.825987,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.00687,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.876986,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.288561,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.039508,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.13662,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.167144,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.274327,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.066406,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 243.277344,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=32.288 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 5035,
            "unit": "s",
            "extra": "nightly run 44716561959; whole job 5097 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795263014299,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.665951,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.005498,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.720418,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.217661,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.027958,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.117023,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.136344,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.227602,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.016 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.074219,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 244.257812,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=22.863 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 7882,
            "unit": "s",
            "extra": "nightly run 44891859166; whole job 7924 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795348405852,
        "tool": "customSmallerIsBetter",
        "benches": [
          {
            "name": "assemble/all_kernels",
            "value": 0.838765,
            "unit": "s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "assemble/render_s",
            "value": 0.006732,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/parse_s",
            "value": 0.873189,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "assemble/encode_s",
            "value": 0.294348,
            "unit": "s",
            "extra": "files=132 words=8266"
          },
          {
            "name": "emulator/ipu_state_construct_ms",
            "value": 0.037607,
            "unit": "ms",
            "extra": "IpuState(), mean of 20"
          },
          {
            "name": "emulator/ipu_state_tracemalloc_peak_mb",
            "value": 0.06123,
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
            "value": 0.138421,
            "unit": "s",
            "extra": "import ipu_as.lark_tree; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "import/ipu_emu",
            "value": 0.167879,
            "unit": "s",
            "extra": "import ipu_emu; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "import/ipu_harness",
            "value": 0.28036,
            "unit": "s",
            "extra": "import ipu_harness.runner; best of 5; interpreter startup 0.018 s"
          },
          {
            "name": "kernel/identity/peak_rss_mb",
            "value": 235.054688,
            "unit": "MB",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/peak_rss_mb",
            "value": 243.105469,
            "unit": "MB",
            "extra": "default case; cycles=179281 run=32.629 s"
          },
          {
            "name": "ci/full_suite_wall_s",
            "value": 7024,
            "unit": "s",
            "extra": "nightly run 45009651549; whole job 7057 s"
          },
          {
            "name": "ci/default_tier_wall_s",
            "value": 228,
            "unit": "s",
            "extra": "ci run 42634927582; whole job 245 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794311711423,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 17759.852894,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 10169.39663,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 7.590 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 14449.563911,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.185 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 14066.966642,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.190 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 12740.486042,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.001 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 10066.949387,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=17.809 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794396527647,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 8826.17557,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 4651.065444,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 16.595 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 6630.287021,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.403 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 6435.83352,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.415 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 6808.379754,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 4606.730111,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=38.917 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794484183059,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9196.582152,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 4620.936216,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 16.703 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 6775.251602,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.395 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 6534.333463,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.409 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7263.520081,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 4567.685457,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=39.250 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794568039718,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 8887.510126,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 4507.758993,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 17.123 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 6573.577144,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.407 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 6385.045564,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.419 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 6829.821734,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 4558.503122,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=39.329 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794655051873,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 12411.843376,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 7272.369607,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 10.613 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 10699.09024,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.250 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 10283.536918,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.260 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 10777.617804,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 7224.114867,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=24.817 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794742065321,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 12238.257371,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 7869.620405,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 9.808 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 11094.986545,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.241 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 10701.338084,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.250 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 9786.510159,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 7773.270096,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=23.064 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794830881074,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 16134.201879,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 9570.885513,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 8.065 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 14028.978434,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.191 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 13237.850313,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.202 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 12670.048824,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.001 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 9447.90421,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=18.976 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1794920899243,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 11945.357741,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 7902.660681,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 9.767 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 11012.204438,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.243 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 10527.503268,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.254 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 9687.851719,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 7879.395285,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=22.753 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795001654676,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 10351.283597,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 5443.154623,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 14.180 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 7838.737115,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.341 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 7562.229043,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.354 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 8098.014557,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 5347.594868,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=33.526 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795092381200,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 11665.365849,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 7874.946053,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 9.801 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 11007.536985,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.243 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 10500.031321,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.255 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 10186.234323,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 7808.18442,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=22.961 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795174558947,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 9975.943605,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 5576.457002,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 13.841 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 8161.975945,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.328 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 7836.378664,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.341 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 7992.500214,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 5552.49304,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=32.288 s"
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
          "id": "4ebd3918f89cf709047a0ccdfec9499262296ca6",
          "message": "Merge pull request #296 from rechefe/crm-cache\n\nCrm cache",
          "timestamp": "2026-11-09T14:27:50Z",
          "url": "https://github.com/rechefe/ipu-emulator/commit/4ebd3918f89cf709047a0ccdfec9499262296ca6"
        },
        "date": 1795263017955,
        "tool": "customBiggerIsBetter",
        "benches": [
          {
            "name": "assemble/words_per_s",
            "value": 12373.284466,
            "unit": "words/s",
            "extra": "files=129 words=8240"
          },
          {
            "name": "emulator/faithful/matmul_384x192_x128.default/cycles_per_s",
            "value": 7803.283512,
            "unit": "cycles/s",
            "extra": "cycles=77185 best of 5: 9.891 s"
          },
          {
            "name": "emulator/faithful/fully_connected.e4/cycles_per_s",
            "value": 11132.755035,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.240 s"
          },
          {
            "name": "emulator/faithful/fully_connected.dynamic/cycles_per_s",
            "value": 10588.240512,
            "unit": "cycles/s",
            "extra": "cycles=2674 best of 5: 0.253 s"
          },
          {
            "name": "kernel/identity/cycles_per_s",
            "value": 10383.664175,
            "unit": "cycles/s",
            "extra": "default case; cycles=17 run=0.002 s"
          },
          {
            "name": "kernel/matmul_720x240_x128/cycles_per_s",
            "value": 7841.663609,
            "unit": "cycles/s",
            "extra": "default case; cycles=179281 run=22.863 s"
          }
        ]
      }
    ]
  }
}