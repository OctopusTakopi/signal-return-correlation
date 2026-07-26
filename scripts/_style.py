"""Shared matplotlib style and palette for report figures."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: F401  (re-exported)

INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
C8 = "#e34948"

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": "sans-serif", "font.size": 9,
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "axes.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.grid": True,
    "axes.axisbelow": True, "legend.frameon": False, "legend.fontsize": 8,
})


def percent_log_ticks(ax, which: str = "y", subs=(1.0,)) -> None:
    """Label a log axis as plain percentages rather than powers of ten.

    Mirrors the helper of the same name in _style_academic. A log axis whose
    label reads "(%)" renders ticks as 10^0, 10^1 and so on, so a tick reading
    10^1 is easily taken for a correlation of ten rather than of ten per cent.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

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
