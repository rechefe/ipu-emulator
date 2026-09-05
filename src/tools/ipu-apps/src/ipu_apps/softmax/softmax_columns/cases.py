"""Runtime cases for softmax_columns, independent of pytest."""
from ipu_apps.softmax.test_support import random_case


CASES = {
    "default": random_case(axis=0, defaults={'rows': 64, 'width': 128, 'scale': 5.0, 'seed': 0}, max_cycles=8000000),
}
