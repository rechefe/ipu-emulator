"""Runtime cases for softmax_columns_packed, independent of pytest."""
from ipu_apps.softmax.test_support import random_case


CASES = {
    "default": random_case(axis=0, defaults={'rows': 64, 'width': 16, 'scale': 3.0, 'seed': 0}, max_cycles=8000000),
}
