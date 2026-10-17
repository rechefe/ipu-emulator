"""The projections' output transform (E07-02), shared by the family's templates.

``projection_output`` lays ``y``, ``(streams, n_tg, n_out, n_tok)``, out as
the kernels' ``ST.AAQ`` rows: grouped channel-major, row ``(p, tg, j)`` at
``p * n_tg * n_out + tg * n_out + j``, the first ``n_tok`` lanes valid. On
unpack the lanes past ``n_tok`` (what the store leaves there) are dropped.
That is the row-major default over the logical shape; it is named here so
``kernel.toml`` says what the store writes, as ``ipu_apps``' ``teardown`` and
``read_output`` did.
"""

from ipu_harness.packing import RowMajor
from ipu_isa.geometry import LANES


class ProjectionOutput(RowMajor):
    """``(..., n_out, n_tok)`` with ``n_tok <= LANES``: one row per output
    channel of each (stream, token group), the first ``n_tok`` lanes valid."""

    def rows(self, shape: tuple[int, ...]) -> int:
        """One row per index of every dimension but the tokens.

        Raises:
            ValueError: The tokens do not fit one row.
        """
        if not 0 < shape[-1] <= LANES:
            raise ValueError(f"projection_output: n_tok ({shape[-1]}) must fit one row, 1..{LANES}")
        return super().rows(shape)


projection_output = ProjectionOutput()
