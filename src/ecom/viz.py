"""One visual system for every figure (matplotlib reports + Plotly dashboard).

Palette: the validated reference palette (categorical slots in fixed order, a one-hue
blue sequential ramp, blue<->red diverging with a gray midpoint, reserved status colours).
Rules: no dual axes, recessive grid, 2px lines, colour follows the entity, never its rank.
"""

from __future__ import annotations

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots - assign in this order, never cycle past the end.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = SERIES
SEQ_BLUE = [
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
]
NEUTRAL_MID = "#f0efec"
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}

SEQ_CMAP = LinearSegmentedColormap.from_list("ecom_seq_blue", SEQ_BLUE)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "ecom_div", ["#104281", "#5598e7", NEUTRAL_MID, "#e66767", "#a32424"]
)

FONT = ["Segoe UI", "DejaVu Sans", "Arial", "sans-serif"]


def apply_matplotlib_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelcolor": INK_2,
            "ytick.labelcolor": INK_2,
            "lines.linewidth": 2,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.labelcolor": INK_2,
            "font.family": FONT,
            "font.size": 10,
            "axes.prop_cycle": mpl.cycler(color=SERIES),
            "image.cmap": "ecom_seq_blue",
        }
    )
    if "ecom_seq_blue" not in mpl.colormaps:
        mpl.colormaps.register(SEQ_CMAP)
        mpl.colormaps.register(DIV_CMAP)


def plotly_template() -> dict:
    """Plotly layout template matching the matplotlib style."""
    return {
        "layout": {
            "paper_bgcolor": SURFACE,
            "plot_bgcolor": SURFACE,
            "font": {"family": "Segoe UI, system-ui, sans-serif", "color": INK_2, "size": 13},
            "title": {"font": {"color": INK, "size": 15}, "x": 0, "xanchor": "left"},
            "colorway": SERIES,
            "xaxis": {
                "gridcolor": GRID,
                "linecolor": AXIS,
                "zeroline": False,
                "tickfont": {"color": INK_2},
            },
            "yaxis": {
                "gridcolor": GRID,
                "linecolor": AXIS,
                "zeroline": False,
                "tickfont": {"color": INK_2},
            },
            # legend below the x-axis title, never on top of it
            "legend": {
                "orientation": "h",
                "yanchor": "top",
                "y": -0.25,
                "x": 0,
                "font": {"color": INK_2},
            },
            "margin": {"l": 50, "r": 20, "t": 50, "b": 40},
            "hoverlabel": {"bgcolor": "#ffffff", "font": {"color": INK}},
        }
    }


apply_matplotlib_style()
