"""Softmax applications (row / column variants).

Five kernels split across two axes -- which direction the reduction runs, and
how the reduced length relates to the 128-element datapath:

======================  ==========================================
``softmax_rows``        one row of exactly 128 elements
``softmax_rows_partial``  rows shorter than 128, several packed per chunk
``softmax_rows_long``   rows longer than 128, reduced across chunks
``softmax_columns``     softmax down each column, width >= 65
``softmax_columns_packed``  softmax down each column, width <= 64
======================  ==========================================

All five share the base-2 reformulation

    softmax(x_i) = 2^(c*(x_i - xmax)) / SUM_j 2^(c*(x_j - xmax)),  c = log2(e)

so the IPU's native ``exp2`` activation applies directly, and all five run the
same four passes: row max, numerators, sum, normalise.

Each app is used directly::

    from ipu_apps.softmax.softmax_rows_long import SoftmaxRowsLongApp

    app = SoftmaxRowsLongApp(
        inst_path="softmax_rows_long.bin",
        input_path="logits.bin",
        output_path="probs.bin",
        rows=8, n=300,
    )
    state, cycles = app.run()

Every app reads and writes row-major ``(rows, cols)`` float32: the output file
has the same layout as the input file, whatever packing the kernel uses
internally.
"""
