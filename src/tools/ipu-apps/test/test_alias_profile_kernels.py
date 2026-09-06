"""All registered cases preserve numerical results and accounting under profiling."""
import json
import os
from pathlib import Path
import time

import pytest
from ipu_apps.kernel_registry import kernels, load
from ipu_apps.kernel_registry.cases import load_cases, run_case
from ipu_emu.alias_profile import AliasProfile
from ipu_emu.alias_detectors import default_registry
from ipu_common.isa_alias_spec import ISA_ALIAS_SPEC

load()
CASES = [(spec.name, name) for spec in kernels() for name in load_cases(spec.name)]


@pytest.mark.parametrize('kernel,name', CASES)
def test_profile_kernel(kernel, name, options=None, label=None):
    case = load_cases(kernel)[name]
    start = time.perf_counter()
    baseline, baseline_cycles = run_case(kernel, case, options=options)
    baseline_seconds = time.perf_counter() - start
    profile = AliasProfile(metadata={'case': label or name})
    start = time.perf_counter()
    state, cycles = run_case(kernel, case, alias_profile=profile, options=options)
    profile_seconds = time.perf_counter() - start
    assert cycles == baseline_cycles == profile.cycles
    for key, value in vars(baseline.stats).items():
        if key != 'alias_profile':
            assert getattr(state.stats, key) == value, key
    assert state.regfile.get_r_acc_bytes() == baseline.regfile.get_r_acc_bytes()
    report = json.loads(profile.to_json())
    if kernel.startswith('softmax'):
        exponential = report['aliases']['A9_EXP']['measurements']
        assert exponential and all(row['status'] == 'verified' for row in exponential), exponential
    for alias in ISA_ALIAS_SPEC:
        observed = sum(row['count'] for row in report['aliases'][alias]['measurements'])
        assert observed == state.stats.alias_hits[alias]
    assert set(report['aliases']) == set(default_registry().entries)
    assert report['totals']['unique_read_bytes'] <= report['totals']['read_bytes']
    if output := os.environ.get('TEST_UNDECLARED_OUTPUTS_DIR'):
        report['timing'] = {'baseline_seconds': baseline_seconds, 'profile_seconds': profile_seconds,
                            'scope': 'whole case including preparation, assembly, execution and output checks'}
        Path(output, kernel + '--' + (label or name) + '.json').write_text(json.dumps(report, indent=2))


@pytest.mark.parametrize('width', [17, 50])
def test_profile_packed_tail_mask(width):
    test_profile_kernel('softmax_columns_packed', 'default',
                        options={'rows': 17, 'width': width}, label=f'tail_width_{width}')
