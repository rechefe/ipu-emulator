"""Runnable cases for maxpool2d_nms9."""

from ipu_apps.kernels.pooling.cases import make_cases

from .app import App

CASES = make_cases(App)
