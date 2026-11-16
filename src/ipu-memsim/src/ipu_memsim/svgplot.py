"""Two small charts written as SVG text: no plotting library needed.

:func:`line_chart` draws a few lines over a shared x axis; :func:`heatmap`
draws a grid of numbers as coloured cells.

One chart is a few series of ``(x, y)`` points over a shared x axis (the
swept parameter) and one y axis (cycles or slowdown). Each series has its own
colour *and* its own marker shape, a legend entry and a label at its last
point, so no series is told apart by colour alone. Hovering a point shows its
values (an SVG ``<title>``).

The colours are the first slots of a colour-blind-checked categorical palette,
always used in the same order.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from html import escape

__all__ = ["Series", "heatmap", "line_chart"]

# blue, orange, aqua, yellow, magenta, green, violet, red: a fixed order.
_COLORS = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
# One hue from light to dark, for the heatmap: more stall = darker.
_RAMP = ("#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
_SURFACE, _INK, _INK2, _MUTED, _GRID, _AXIS = (
    "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7",
)  # fmt: skip
_WIDTH, _HEIGHT = 760, 440
_LEFT, _RIGHT, _TOP, _BOTTOM = 80, 70, 70, 60  # room for labels around the plot
_LEGEND_ROW = 20  # the legend is a list under the plot


@dataclass(frozen=True)
class Series:
    """One line: a name and its ``(x, y)`` points, in x order."""

    name: str
    points: Sequence[tuple[float, float]]


def _nice_ticks(low: float, high: float, count: int = 5) -> list[float]:
    """About ``count`` round numbers covering ``low..high``."""
    if high <= low:
        high = low + 1.0
    raw = (high - low) / count
    power = 10 ** math.floor(math.log10(raw))
    step = next(m * power for m in (1, 2, 5, 10) if m * power >= raw)
    first = math.floor(low / step) * step
    ticks = [first + i * step for i in range(int((high - first) / step) + 2)]
    return [t for t in ticks if t <= high + step]


def _number(value: object) -> str:
    """A tick or tooltip number, short: 316553 -> 317k, 1.25 -> 1.25. A value
    that is not a number (a switch's setting) is shown as it is."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return str(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if abs(value) >= 10_000:
        return f"{value / 1000:.3g}k"
    return f"{value:.4g}"


def _marker(shape: int, x: float, y: float, color: str, tip: str) -> str:
    """An 8-pixel marker with a hover text; the shape depends on the series."""
    common = f'fill="{color}" stroke="{_SURFACE}" stroke-width="2"'
    title = f"<title>{escape(tip)}</title>"
    shape %= 4  # four shapes; colour and the end label tell lines 5 to 8 apart
    if shape == 0:
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5" {common}>{title}</circle>'
    if shape == 1:
        return f'<rect x="{x - 4.5:.1f}" y="{y - 4.5:.1f}" width="9" height="9" {common}>{title}</rect>'
    if shape == 2:
        points = f"{x:.1f},{y - 6:.1f} {x + 6:.1f},{y + 5:.1f} {x - 6:.1f},{y + 5:.1f}"
    else:
        points = f"{x:.1f},{y - 6:.1f} {x + 6:.1f},{y:.1f} {x:.1f},{y + 6:.1f} {x - 6:.1f},{y:.1f}"
    return f'<polygon points="{points}" {common}>{title}</polygon>'


def line_chart(  # noqa: C901, PLR0913 (one chart, drawn top to bottom)
    series: Sequence[Series],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
    log_x: bool = False,
    marks: Sequence[tuple[float, str]] = (),
) -> str:
    """The chart as an SVG document.

    Args:
        series: Up to eight lines (the palette's eight colours, in fixed order).
        title, subtitle: Text above the plot; the subtitle is where a toy
            model is labelled as one.
        x_label, y_label: Axis names.
        log_x: Space the x values by their logarithm (for 1, 10, 100, ...).
        marks: Labelled x positions drawn as dashed vertical lines (a
            reference value such as a measured anchor point).

    Raises:
        ValueError: More than eight series, or none.
    """
    if not 1 <= len(series) <= len(_COLORS):
        raise ValueError(f"a chart takes 1 to {len(_COLORS)} series, got {len(series)}")
    xs = sorted({x for s in series for x, _ in s.points})
    ys = [y for s in series for _, y in s.points]
    y_ticks = _nice_ticks(min(0.0, min(ys)) if min(ys) > 0 and min(ys) < 2 else min(ys), max(ys))
    y_low, y_high = y_ticks[0], y_ticks[-1]
    height = _HEIGHT + _LEGEND_ROW * len(series)
    plot_w, plot_h = _WIDTH - _LEFT - _RIGHT, _HEIGHT - _TOP - _BOTTOM

    def along(x: float) -> float:
        return math.log10(x) if log_x else x

    x_low, x_high = along(xs[0]), along(xs[-1])

    def px(x: float) -> float:
        span = x_high - x_low
        return _LEFT + (plot_w / 2 if span == 0 else (along(x) - x_low) / span * plot_w)

    def py(y: float) -> float:
        return _TOP + plot_h - (y - y_low) / (y_high - y_low) * plot_h

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {_WIDTH} {height}" '
        f'width="{_WIDTH}" height="{height}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{_WIDTH}" height="{height}" fill="{_SURFACE}"/>',
        f'<text x="{_LEFT}" y="28" font-size="16" font-weight="bold" fill="{_INK}">{escape(title)}</text>',
        f'<text x="{_LEFT}" y="48" font-size="12" fill="{_INK2}">{escape(subtitle)}</text>',
    ]
    for tick in y_ticks:  # horizontal grid lines and y tick labels
        y = py(tick)
        out.append(
            f'<line x1="{_LEFT}" y1="{y:.1f}" x2="{_LEFT + plot_w}" y2="{y:.1f}" stroke="{_GRID}"/>'
        )
        out.append(
            f'<text x="{_LEFT - 8}" y="{y + 4:.1f}" font-size="11" text-anchor="end" '
            f'fill="{_MUTED}">{_number(tick)}</text>'
        )
    base = _TOP + plot_h
    out.append(
        f'<line x1="{_LEFT}" y1="{base}" x2="{_LEFT + plot_w}" y2="{base}" stroke="{_AXIS}"/>'
    )
    for x in xs:  # one x tick per swept value
        out.append(
            f'<text x="{px(x):.1f}" y="{base + 18}" font-size="11" text-anchor="middle" '
            f'fill="{_MUTED}">{_number(x)}</text>'
        )
    out.append(
        f'<text x="{_LEFT + plot_w / 2:.1f}" y="{_HEIGHT - 14}" font-size="12" '
        f'text-anchor="middle" fill="{_INK2}">{escape(x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(18 {_TOP + plot_h / 2:.1f}) rotate(-90)" font-size="12" '
        f'text-anchor="middle" fill="{_INK2}">{escape(y_label)}</text>'
    )
    for x, text in marks:  # reference lines, behind the data
        if xs[0] <= x <= xs[-1]:
            out.append(
                f'<line x1="{px(x):.1f}" y1="{_TOP}" x2="{px(x):.1f}" y2="{base}" stroke="{_MUTED}" '
                f'stroke-dasharray="4 3"/>'
            )
            out.append(
                f'<text x="{px(x) + 4:.1f}" y="{_TOP + 10}" font-size="10" fill="{_INK2}">{escape(text)}</text>'
            )
    for index, one in enumerate(series):
        color = _COLORS[index]
        path = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in one.points)
        out.append(f'<polyline points="{path}" fill="none" stroke="{color}" stroke-width="2"/>')
        for x, y in one.points:
            tip = f"{one.name}: {x_label} = {_number(x)}, {y_label} = {y:,.4g}"
            out.append(_marker(index, px(x), py(y), color, tip))
        # The legend, under the plot: marker + name, in ink (not in the series colour).
        legend_y = _HEIGHT + 4 + index * _LEGEND_ROW
        out.append(_marker(index, _LEFT + 6, legend_y, color, one.name))
        out.append(
            f'<text x="{_LEFT + 20}" y="{legend_y + 4}" font-size="12" '
            f'fill="{_INK}">{escape(one.name)}</text>'
        )
    # Each line's last value, written beside its end. Placed top to bottom,
    # each pushed down just enough not to overlap the one above it.
    # With more than four lines the labels would pile up: the legend and the
    # hover text carry the values instead.
    ends = sorted((py(one.points[-1][1]), one.points[-1]) for one in series if len(series) <= 4)
    previous = -100.0
    for y, (last_x, last_y) in ends:
        y = max(y, previous + 13)
        previous = y
        out.append(
            f'<text x="{px(last_x) + 10:.1f}" y="{y + 4:.1f}" font-size="11" '
            f'fill="{_INK2}">{_number(last_y)}</text>'
        )
    out.append("</svg>")
    return "\n".join(out) + "\n"


def heatmap(  # noqa: PLR0913 (the grid and its four texts)
    xs: Sequence[float],
    ys: Sequence[float],
    cells: Sequence[Sequence[float | None]],
    *,
    title: str,
    subtitle: str,
    x_label: str,
    y_label: str,
) -> str:
    """A grid of values as an SVG document: one cell per ``(x, y)`` pair.

    Every cell shows its number, so the colour only helps the eye: a darker
    blue is a larger value. The colour steps are spaced by the logarithm of
    the value, because stall percentages spread over several powers of ten.
    A cell with no value (a configuration that cannot run) is grey, "n/a".

    Args:
        xs: The column values, left to right.
        ys: The row values; the first is drawn at the bottom.
        cells: ``cells[row][column]``, in the order of ``ys`` and ``xs``.
        title, subtitle, x_label, y_label: Text around the grid.
    """
    cell_w, cell_h = 86, 40
    left, top = 110, 70
    width = left + cell_w * len(xs) + 30
    height = top + cell_h * len(ys) + 60
    known = [v for row in cells for v in row if v is not None]
    low = math.log10(max(min(known, default=1.0), 0.01))
    high = math.log10(max(max(known, default=1.0), 0.01))

    def color(value: float) -> int:
        """The ramp step of a value: 0 (lightest) to 6 (darkest)."""
        if high <= low:
            return 0
        share = (math.log10(max(value, 0.01)) - low) / (high - low)
        return min(len(_RAMP) - 1, max(0, int(share * len(_RAMP))))

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Helvetica, Arial, sans-serif">',
        f'<rect width="{width}" height="{height}" fill="{_SURFACE}"/>',
        f'<text x="20" y="28" font-size="16" font-weight="bold" fill="{_INK}">{escape(title)}</text>',
        f'<text x="20" y="48" font-size="12" fill="{_INK2}">{escape(subtitle)}</text>',
    ]
    for r, y_value in enumerate(ys):
        y = top + cell_h * (len(ys) - 1 - r)  # the first row at the bottom
        out.append(
            f'<text x="{left - 8}" y="{y + cell_h / 2 + 4:.0f}" font-size="11" text-anchor="end" '
            f'fill="{_MUTED}">{_number(y_value)}</text>'
        )
        for c, x_value in enumerate(xs):
            x = left + cell_w * c
            value = cells[r][c]
            step = None if value is None else color(value)
            fill = "#e1e0d9" if step is None else _RAMP[step]
            text = (
                "n/a" if value is None else (f"{value:.3g}%" if value < 1000 else f"{value:,.0f}%")
            )
            ink = _INK if step is None or step < 3 else "#ffffff"
            tip = f"{x_label} = {_number(x_value)}, {y_label} = {_number(y_value)}: {text}"
            # A 2-pixel gap between cells: each rectangle is drawn slightly smaller.
            out.append(
                f'<rect x="{x + 1}" y="{y + 1}" width="{cell_w - 2}" height="{cell_h - 2}" rx="3" '
                f'fill="{fill}"><title>{escape(tip)}</title></rect>'
            )
            out.append(
                f'<text x="{x + cell_w / 2:.0f}" y="{y + cell_h / 2 + 4:.0f}" font-size="11" '
                f'text-anchor="middle" fill="{ink}" pointer-events="none">{text}</text>'
            )
    for c, x_value in enumerate(xs):
        out.append(
            f'<text x="{left + cell_w * c + cell_w / 2:.0f}" y="{top + cell_h * len(ys) + 16}" '
            f'font-size="11" text-anchor="middle" fill="{_MUTED}">{_number(x_value)}</text>'
        )
    out.append(
        f'<text x="{left + cell_w * len(xs) / 2:.0f}" y="{height - 14}" font-size="12" '
        f'text-anchor="middle" fill="{_INK2}">{escape(x_label)}</text>'
    )
    out.append(
        f'<text transform="translate(18 {top + cell_h * len(ys) / 2:.0f}) rotate(-90)" font-size="12" '
        f'text-anchor="middle" fill="{_INK2}">{escape(y_label)}</text>'
    )
    out.append("</svg>")
    return "\n".join(out) + "\n"
