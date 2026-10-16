"""Runnable cases for sample_descriptors_separable."""

from ipu_apps.kernels.sampling.cases import make_cases

from .app import App

CASES = make_cases(App)
