"""Softmax applications (row / column variants) -- diagnostic-only subset.

Only `softmax_rows_partial` and `softmax_columns_packed` are vendored here
(see each submodule's docstring for provenance), the two variants this repo's
full-layer chain tests need for N_TOK in {16, 64}. The lookup/catalog registry
from `origin/pr4-registry-docs` (`ipu_apps.softmax.registry`) is NOT vendored
-- it depends on the `ipu_apps.kernel_registry` package, which is later,
unmerged infrastructure this diagnostic task does not need. Import the App
classes directly from their submodules instead:

    from ipu_apps.softmax.softmax_rows_partial import SoftmaxRowsPartialApp
    from ipu_apps.softmax.softmax_columns_packed import SoftmaxColumnsPackedApp
"""
