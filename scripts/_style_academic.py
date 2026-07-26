"""Matplotlib style for the journal-style figures.

Serif text with matching STIX maths, hairline spines, inward ticks, no fill and
no chartjunk. Restrained palette that survives greyscale printing: the four
series colours differ in lightness as well as hue.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (re-exported)
from matplotlib.ticker import (  # noqa: F401
    FuncFormatter, LogLocator, NullFormatter,
)

# Ink, and four series colours ordered by lightness so the ranking is legible
# in black and white as well as in colour.
K = "#000000"
K70 = "#4d4d4d"
K45 = "#8c8c8c"
K25 = "#bfbfbf"
NAVY = "#1f3b73"
RED = "#a4243b"
TEAL = "#1b6b6b"
OCHRE = "#a97c1f"

SERIES = (NAVY, RED, TEAL, OCHRE)

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 300,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,

    "font.family": "serif",
    "font.serif": ["STIXGeneral", "DejaVu Serif", "Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 8.5,

    "text.color": K,
    "axes.labelcolor": K,
    "axes.edgecolor": K,
    "axes.linewidth": 0.6,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "axes.titleweight": "normal",
    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.axisbelow": True,

    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.color": K,
    "ytick.color": K,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 3.0,
    "ytick.major.size": 3.0,
    "xtick.minor.size": 1.6,
    "ytick.minor.size": 1.6,

    "grid.color": "#d9d9d9",
    "grid.linewidth": 0.4,
    "grid.linestyle": "-",
    "axes.grid": False,

    "legend.frameon": False,
    "legend.fontsize": 7.5,
    "legend.handlelength": 1.9,
    "legend.labelspacing": 0.35,
    "legend.borderpad": 0.2,

    "lines.linewidth": 1.1,
    "lines.markersize": 3.2,
    "patch.linewidth": 0.6,
})


def panel_label(ax, letter: str, dx: float = -0.14, dy: float = 1.06) -> None:
    """A bold (a)/(b) marker in the axes' upper left, outside the frame."""
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="top", ha="left")


def caption(fig, text: str, y: float = -0.02, size: float = 7.5) -> None:
    """A figure-level note, set below the axes the way a journal would."""
    fig.text(0.5, y, text, ha="center", va="top", fontsize=size, color=K70,
             wrap=True)


def light_grid(ax, axis: str = "both") -> None:
    ax.grid(True, axis=axis, which="major")
    ax.set_axisbelow(True)


def percent_log_ticks(ax, which: str = "y", subs=(1.0,)) -> None:
    """Label a log axis as plain percentages rather than powers of ten.

    A log axis whose label reads "(%)" renders its ticks as 10^0, 10^1 and so
    on, which forces the reader to combine the exponent with the unit in the
    axis label. On a correlation or IC axis that is a real hazard: a tick
    reading 10^1 is easily taken for a correlation of ten rather than of ten
    per cent. This replaces the exponent labels with 1%, 10% and the like.

    `subs` selects which mantissas inside each decade get a major tick: pass
    (1,) for a wide range and (1, 2, 5) for a narrow one that would otherwise
    show only two labels.
    """
    axis = ax.yaxis if which == "y" else ax.xaxis
    axis.set_major_locator(LogLocator(base=10.0, subs=subs, numticks=20))
    axis.set_minor_locator(LogLocator(base=10.0, subs=(1, 2, 3, 4, 5, 6, 7, 8, 9),
                                      numticks=100))
    axis.set_minor_formatter(NullFormatter())

    def _fmt(v, _pos):
        if v <= 0:
            return ""
        if v >= 1:
            return f"{v:,.0f}%"
        text = f"{v:.4f}".rstrip("0").rstrip(".")
        return f"{text}%"

    axis.set_major_formatter(FuncFormatter(_fmt))
