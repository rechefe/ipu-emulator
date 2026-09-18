"""Runtime cases for softmax_rows_long, independent of pytest."""
from ipu_apps.softmax.test_support import random_case


CASES = {
    "default": random_case(axis=1, width_option="n", defaults={'rows': 8, 'n': 300, 'scale': 5.0, 'seed': 0}, max_cycles=8000000),
}
