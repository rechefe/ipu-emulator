"""attn_scores_km_64x48: every case in cases.py, plus head selection.

A different ``head`` must score a different block of the same input: the
sweep runs the first and last head on one input and checks each against its
own block's reference.
"""
from ipu_apps.kernel_registry.testing import case_tests
from ipu_apps.kernels.attention.attn_scores_km_64x48.app import N_HEAD

test_case = case_tests(__package__, sweep=[
    dict(head=0, seed=0xD49),
    dict(head=N_HEAD - 1, seed=0xD49),
])
