"""score_threshold's padding lanes (E07-15): the ``ipu_apps`` cases' second
check. The lanes past ``count`` in the selected rows hold ``tau - 800 / T``
on the way in; both outputs must suppress them, so the selected rows' padding
lanes read back 0 (``gate_and_count`` drops them, so this reads the raw
output region)."""

from __future__ import annotations

import math

import numpy as np

from ipu_isa.formats import ElementFormat


def test_padding_lanes_select_nothing(cases, run_case):
    for case in cases:
        result = run_case(case, ElementFormat.FP32)
        assert result.passed, result.describe()
        count = int(case.values["count"])
        raw = np.frombuffer(result.raw_outputs["y"], dtype="<f4")
        selected = raw[: math.ceil(count / 128) * 128]
        assert not selected[count:].any(), f"{case.name}: a padding lane was selected"
