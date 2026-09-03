"""Pytest wrapper for the first-layer conv suite.

Reuses the self-contained test co-located with the app so it runs as a BUILD
target.  The asm is assembled in-process by the underlying test.
"""

from __future__ import annotations

from ipu_apps.convolutions_universal.conv.conv_first_layer.test_conv_first_layer import (  # noqa: F401
    test_256x256x3_to_128x128x16,
)
