"""Runnable cases for conv3x3_relu_cin1."""
from .app import App
from ipu_apps.kernels.convolutions.cases import make_cases

CASES = make_cases(App)
