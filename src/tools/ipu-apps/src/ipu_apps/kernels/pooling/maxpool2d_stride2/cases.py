"""Runnable cases for maxpool2d_stride2."""

from ipu_apps.kernels.pooling.cases import make_cases

from .app import App

CASES = make_cases(App)
