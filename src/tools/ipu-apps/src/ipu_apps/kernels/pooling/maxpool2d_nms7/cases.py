"""Runnable cases for maxpool2d_nms7."""
from .app import App
from ipu_apps.kernels.pooling.cases import make_cases

CASES = make_cases(App)
