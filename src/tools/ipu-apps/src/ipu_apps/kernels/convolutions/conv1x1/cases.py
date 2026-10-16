"""Runnable cases for conv1x1."""

from ipu_apps.kernels.convolutions.cases import make_cases

from .app import App

CASES = make_cases(App)
