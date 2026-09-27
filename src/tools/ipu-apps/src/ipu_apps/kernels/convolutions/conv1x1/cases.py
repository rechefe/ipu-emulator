"""Runnable cases for conv1x1."""
from .app import App
from ipu_apps.kernels.convolutions.cases import make_cases

CASES = make_cases(App)
