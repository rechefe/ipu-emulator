"""Runnable cases for sample_descriptors."""
from .app import App
from ipu_apps.kernels.sampling.cases import make_cases

CASES = make_cases(App)
