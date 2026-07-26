"""Report figures. Reads only the JSON written by the verification scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import paths  # noqa: E402
from _style import (  # noqa: E402
    C1, C2, C3, C4, C8, INK, INK2, MUTED, percent_log_ticks, plt,
)

BPS = 1e-4
KAPPA = 3.0
EQUITY = engine.Market(0.03, 1.0, 5 * BPS, "US equity, 1-day horizon")


def load(name: str) -> dict:
    return json.loads((paths.RESULTS_DIR / name).read_text())


def fig_floor_surface() -> None:
    """The floor across horizon and cost -- why the same alpha is judged
    differently in equities and FX."""
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    taus = np.logspace(np.log10(1 / 1440), np.log10(20), 400)
    for cost_bps, colour, label in ((0.2, C3, "0.2 bps"), (2, C1, "2 bps"),
                                    (5, C2, "5 bps"), (20, C8, "20 bps")):
        floor = engine.rho_floor(cost_bps * BPS, 0.03, taus, KAPPA)
        ax.plot(taus, 100 * floor, color=colour, lw=1.4, label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("forecast horizon $\\tau$ (days)")
    ax.set_ylabel("minimum correlation")
    percent_log_ticks(ax, "y")
    ax.set_title("floor $= c/(3\\sigma\\sqrt{\\tau})$, $\\sigma=3\\%$/day", fontsize=9)
    ax.legend(title="round-trip cost", ncols=2)
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.axvline(1 / 1440, color=MUTED, lw=0.7, ls=":")
    ax.annotate("1 min", (1 / 1440, ax.get_ylim()[1]), fontsize=7, color=MUTED,
                ha="left", va="top", xytext=(2, -2), textcoords="offset points")
    ax.annotate("1 day", (1.0, ax.get_ylim()[1]), fontsize=7, color=MUTED,
                ha="left", va="top", xytext=(2, -2), textcoords="offset points")

    ax = axes[1]
    cases = load("examples.json")["cases"]
    names = [c["label"].replace(", ", "\n") for c in cases]
    stated = [c["stated_pct"] for c in cases]
    exact = [c["floor_pct"] for c in cases]
    xs = np.arange(len(cases))
    ax.bar(xs - 0.19, stated, 0.36, color=MUTED, label="thread's figure")
    ax.bar(xs + 0.19, exact, 0.36, color=C1, label="exact")
    for i, (s, e) in enumerate(zip(stated, exact)):
        ax.annotate(f"{e:.3f}%", (i + 0.19, e), ha="center", va="bottom",
                    fontsize=7, color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel("minimum correlation (%)")
    ax.set_title("the three worked examples", fontsize=9)
    ax.legend()

    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig1_floor.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_cost_cliff() -> None:
    """Trade frequency and surviving P&L as a function of the multiple."""
    rot = load("rules_of_thumb.json")
    mults = np.linspace(0.6, 5.0, 500)
    ks = KAPPA / mults
    freq = 100 * engine.trade_frequency(ks)
    capture = 100 * np.array([engine.gross_capture(m, KAPPA) for m in mults])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.9))

    ax = axes[0]
    ax.plot(mults, freq, color=C1, lw=1.6, label="periods with a trade")
    ax.plot(mults, capture, color=C2, lw=1.6, label="net P&L retained")
    for m, lab, col in ((1.0, "floor", MUTED), (1.5, '"ok"', C3), (2.0, '"very good"', C4)):
        ax.axvline(m, color=col, lw=0.8, ls="--")
        ax.annotate(lab, (m, 60.5), rotation=0, fontsize=7, color=col,
                    ha="center", va="top")
    # The thread's stated frequencies.
    ax.plot([1.5], [4.55], "o", ms=4, color=C3, zorder=5)
    ax.plot([2.0], [13.36], "o", ms=4, color=C4, zorder=5)
    ax.annotate("stated ~5%\nmodel 4.6%", (1.5, 4.55), fontsize=7, color=C3,
                xytext=(-46, 10), textcoords="offset points")
    ax.annotate("stated 20-30%\nmodel 13.4%", (2.0, 13.36), fontsize=7, color=C8,
                xytext=(8, -2), textcoords="offset points")
    ax.axhspan(20, 30, color=C8, alpha=0.07, lw=0)
    ax.set_xlabel("correlation as a multiple of the floor")
    ax.set_ylabel("percent")
    ax.set_title("what accrues above the floor", fontsize=9)
    ax.legend(loc="upper left")
    ax.set_ylim(0, 66)

    ax = axes[1]
    table = rot["table"]
    m = [r["multiple"] for r in table]
    net = [r["net_ir_annual_500_assets"] for r in table]
    gross = [r["gross_ir_annual_500_assets"] for r in table]
    ax.plot(m, gross, color=MUTED, lw=1.5, marker="o", ms=3,
            label="gross (costless)")
    ax.plot(m, net, color=C2, lw=1.6, marker="o", ms=3, label="net of cost")
    ax.fill_between(m, net, gross, color=C8, alpha=0.09, lw=0)
    ax.annotate("cost", (3.0, 3.8), fontsize=8, color=C8)
    ax.axhline(1.0, color=MUTED, lw=0.7, ls=":")
    ax.axvline(2.0, color=C4, lw=0.8, ls="--")
    ax.set_xlabel("correlation as a multiple of the floor")
    ax.set_ylabel("annualised IR, 500 independent names")
    ax.set_title("cost takes 84% of the Sharpe at 2x the floor", fontsize=9)
    ax.legend(loc="upper left")

    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig2_cliff.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_scatter_intuition() -> None:
    """What a 0.556% correlation actually looks like, next to 8.4%."""
    rng = np.random.default_rng(7)
    n = 4000
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.7))
    specs = [
        (EQUITY, 1.0, "equity, at the floor\n$\\rho=0.56\\%$"),
        (EQUITY, 3.0, "equity, 3x the floor\n$\\rho=1.67\\%$"),
        (engine.Market(0.003, 1 / 1440, 0.2 * BPS), 1.0,
         "1-min FX, at the floor\n$\\rho=8.4\\%$"),
    ]
    for ax, (market, mult, title) in zip(axes, specs):
        rho = mult * market.floor(KAPPA)
        x, y, beta = engine.simulate_returns_and_alpha(market, rho, n, rng)
        ax.scatter(x, 100 * y, s=2.2, color=C1, alpha=0.28, lw=0)
        grid = np.linspace(-3.6, 3.6, 50)
        ax.plot(grid, 100 * beta * grid, color=C2, lw=1.8)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.axvline(0, color=MUTED, lw=0.6)
        ax.set_xlabel("signal $x$ (standard deviations)")
        ax.set_title(title, fontsize=8.5)
        ax.set_xlim(-4, 4)
        ax.text(0.03, 0.03, f"measured $\\hat\\rho$ = {np.corrcoef(x, y)[0,1]*100:.2f}%",
                transform=ax.transAxes, fontsize=7.5, color=INK2)
    axes[0].set_ylabel("next-period return (%)")
    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig3_scatter.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_counterexamples() -> None:
    ce = load("counterexamples.json")
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))

    # A. sparse signal
    ax = axes[0]
    rows = ce["shape"]["rows"]
    p = np.array([r["sparsity"] for r in rows])
    ratio = np.array([r["vs_gaussian"] for r in rows])
    grid = np.logspace(-3.4, -0.3, 300)
    # closed form: net(p) = beta(sqrt(p) - kappa p) for sqrt(p) < 1/kappa
    gauss = engine.trunc_mean(KAPPA)
    curve = np.where(1 / np.sqrt(grid) > KAPPA, np.sqrt(grid) - KAPPA * grid, 0) / gauss
    ax.plot(grid, curve, color=C1, lw=1.5)
    ax.scatter(p, ratio, s=18, color=C2, zorder=5, label="simulated")
    ax.axhline(1.0, color=MUTED, lw=0.8, ls="--")
    ax.annotate("Gaussian alpha,\nsame $\\rho$", (2e-3, 1.0), fontsize=7,
                color=MUTED, va="bottom")
    ax.set_xscale("log")
    ax.set_xlabel("probability the signal fires")
    ax.set_ylabel("net P&L / Gaussian net P&L")
    ax.set_title("A. shape: same $\\rho$, up to 109x", fontsize=8.5)
    ax.legend(loc="upper left")

    # C. state-dependent costs
    ax = axes[1]
    rows = ce["conditioning"]["rows"]
    labels = ["edge in\ncalm", "edge in\nstress", "edge while\nhalted"]
    net = [r["net_sim_bps"] for r in rows]
    err = [r["net_sim_se_bps"] for r in rows]
    ax.bar(labels, net, 0.55, yerr=err, color=[C3, C4, C8], capsize=3,
           error_kw={"lw": 0.8, "ecolor": INK2})
    for i, v in enumerate(net):
        ax.annotate(f"{v:.2f}", (i, v), ha="center", va="bottom", fontsize=8,
                    color=INK)
    ax.set_ylabel("net P&L (bps/day)")
    ax.set_title(f"C. conditioning: one $\\rho$ = "
                 f"{ce['conditioning']['rho_pct']:.1f}%", fontsize=8.5)

    # D. estimation error
    ax = axes[2]
    rows = ce["estimation"]["head_to_head"]
    n_obs = [r["n_obs"] for r in rows]
    p_wrong = [100 * r["p_junk_beats_true"] for r in rows]
    ax.plot(n_obs, p_wrong, color=C2, lw=1.6, marker="o", ms=4)
    ax.axhline(50, color=MUTED, lw=0.8, ls="--")
    ax.annotate("coin flip", (n_obs[0], 50), fontsize=7, color=MUTED,
                va="bottom")
    short = {504: "1 name, 2y", 2520: "1 name, 10y",
             50_400: "100 names", 252_000: "500 names"}
    for r in rows:
        inside = r["n_obs"] >= 50_000
        ax.annotate(short.get(r["n_obs"], r["sample"]),
                    (r["n_obs"], 100 * r["p_junk_beats_true"]),
                    fontsize=6.5, color=INK2,
                    ha="right" if inside else "left",
                    xytext=(-5, 7) if inside else (5, 5),
                    textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("observations")
    ax.set_ylabel("P(pure noise scores higher)")
    ax.set_title("D. estimation: alpha at 2x floor", fontsize=8.5)
    # A probability axis must not extend below zero. The smallest value is
    # 0.01%, which sits on the spine with its label offset in points.
    ax.set_ylim(0, 60)

    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig4_counterexamples.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_breadth() -> None:
    ce = load("counterexamples.json")
    rows = ce["breadth"]["rows"]
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    corrs = sorted({r["alpha_correlation"] for r in rows})
    colours = [MUTED, C3, C1, C4, C8]
    for corr, colour in zip(corrs, colours):
        sub = sorted([r for r in rows if r["alpha_correlation"] == corr],
                     key=lambda r: r["names"])
        ax.plot([r["names"] for r in sub], [r["ir_actual"] for r in sub],
                color=colour, lw=1.5, marker="o", ms=3,
                label=f"{corr:.0%}" if corr else "0 (independent)")
    ax.set_xscale("log")
    ax.set_xlabel("names in the portfolio")
    ax.set_ylabel("annualised IR")
    ax.set_title(f"E. breadth at $\\rho$ = {ce['breadth']['rho_pct']:.2f}%\n"
                 "(2x the floor)", fontsize=9)
    ax.legend(title="correlation between\nthe alphas", fontsize=7,
              title_fontsize=7)
    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig5_breadth.png"
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def main() -> None:
    paths.ensure_directories()
    fig_floor_surface()
    fig_cost_cliff()
    fig_scatter_intuition()
    fig_counterexamples()
    fig_breadth()


if __name__ == "__main__":
    main()
