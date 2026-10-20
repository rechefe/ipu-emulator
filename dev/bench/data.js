window.BENCHMARK_DATA = {
  "lastUpdate": 1792496722273,
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
      }
    ]
  }
}