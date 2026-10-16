"""Runnable cases for conv3x3_relu."""

from ipu_apps.kernels.convolutions.cases import make_cases

from .app import App

CASES = make_cases(App)
