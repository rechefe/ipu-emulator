"""Runtime cases for softmax_rows_partial, independent of pytest."""
from ipu_apps.softmax.test_support import random_case


CASES = {
    "default": random_case(axis=1, width_option="n", defaults={'n': 32, 'rows': 8, 'seed': 0, 'scale': 5.0}, max_cycles=4000000),
}
