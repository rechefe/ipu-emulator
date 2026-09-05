"""Runtime cases for softmax_rows, independent of pytest."""
from ipu_apps.softmax.test_support import random_case


CASES = {
    "default": random_case(axis=1, fixed_width=128, defaults={'rows': 128, 'scale': 5.0, 'seed': 0}, max_cycles=2000000),
}
