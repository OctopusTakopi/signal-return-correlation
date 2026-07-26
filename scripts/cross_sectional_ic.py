"""Cross-sectional IC: a different metric on the same panel, with a different floor.

Everything in sections 1-13 of README.md is a *time-series* statement -- one
asset, does the signal predict its own next return. A cross-sectional book asks
whether the signal ranks assets against each other at a point in time. The two
are routinely confused, usually to a strategy's cost.

Four findings, all simulated or derived, none asserted:

1. The COMMON-COMPONENT IC and the cross-sectional IC are orthogonal. Pooled
   covariance splits exactly into a between-date term Cov_t(xbar, ybar) and a
   within-date term E_t[Cov_i(x, y)], so a signal can carry a large statistic on
   one and exactly zero on the other. The "pooled" correlation over a flattened
   panel is neither and should never be quoted.

   Two cautions the report states and this script must not overstate. The
   date-mean statistic is NOT the general time-series IC: a conventional
   time-series IC is Corr_t(x_it, y_i,t+tau) per asset, and it can be positive
   while every date mean is zero. And the reported cross-sectional IC is the mean
   of date-wise CORRELATIONS, which is not the within-date covariance term
   rescaled; the two agree only when cross-sectional dispersion is constant
   through time.

2. The cross-sectional floor is per-NAME different, and its direction is an
   empirical question rather than a theorem. A dollar-neutral book earns the
   dispersion of returns, not their level. Under a HOMOGENEOUS equicorrelation
   model dispersion is smaller than single-name volatility by
   sqrt(1 - rho_r), which would RAISE the per-name bar by about 1.2x. Measured
   2026 dispersion (6.60%/day) instead EXCEEDS median single-name volatility
   (5.19%/day), because cross-sectional spread loads on the right tail of the
   volatility distribution, so the measured per-name floor is 0.79x the
   directional one: about 21% LOWER, not higher.

3. Common-position breadth saturates at 1/rho_r. A book holding the SAME
   position in every name has per-name P&L h*y_i, so its P&L streams inherit the
   RETURN correlation and N/(1 + (N-1) rho_r) counts correctly: about 3.2 bets on
   a 200-name crypto book at the 2026 mean rho_r = 0.3103, however many names are
   added.

   That restriction is load-bearing and this script does NOT generalise past it.
   Grinold-Kahn breadth counts independent P&L streams, which are governed by
   FORECAST correlation (README section 7E's rho_alpha), not by return
   correlation. The two come apart completely: with y_i = beta x_i + gamma f +
   eps_i and x_i, f, eps_i independent, returns correlate arbitrarily close to 1
   through gamma while the proportional-rule P&Ls stay uncorrelated and breadth
   stays at N. The IR-ratio column below is therefore the ratio of two MODELLED
   quantities, a degree-of-freedom ceiling over a common-position risk statistic,
   and is not a strategy information-ratio gain. Nothing here measures a forecast
   correlation, so no strategy breadth conclusion is drawn.

4. A portfolio has to clear a stricter bar than a name. The kappa-sigma floor
   asks whether the largest signal pays. A cross-sectional book holds the whole
   ranking, so what matters is whether the average name pays: a factor
   kappa/sqrt(pi/2) = 2.39 stricter at one full round trip per period, or about
   1.7 at the 1.41 turnover a signal-weighted book on an iid signal actually
   produces. Truncating to the extreme deciles raises gross and turnover
   together, so it helps a strong signal and hurts a weak one.

Output: results/cross_sectional_ic.json, figures/fig9-fig11.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import paths  # noqa: E402
from _style_academic import (  # noqa: E402
    K, K25, K45, K70, NAVY, OCHRE, RED, TEAL, caption, light_grid, panel_label,
    percent_log_ticks, plt,
)

BPS = 1e-4
KAPPA = 3.0
SEED = 8_1_1912  # Fisher's birthday; not chosen after seeing a result

# A top-200 USDT-perp universe in a normal regime. Inputs, not facts.
# Calibrated on 2026 Binance data by scripts/calibrate_from_data.py.
# 2026 only: the 2020-21 volatility and correlation regime no longer
# describes this market. See results/calibration.json, README section 15.
N_ASSETS = 200
TOTAL_VOL = 0.0519       # 2026 median single-name daily vol
# ONE correlation parameter, not two. A one-factor market has an asset-to-factor
# correlation q and a pairwise correlation rho_r = q^2, and quoting a single
# number for "the correlation" gets one of dispersion or breadth wrong. Everything
# below is derived from the pairwise figure, which is what a correlation matrix
# reports and what breadth needs.
# The MEAN off-diagonal pairwise correlation, not the median. Expanding
# BR_eff = N^2/(1'C1) = N/(1 + (N-1) rho_bar) shows the mean is the exact
# sufficient statistic for equal-weighted breadth, so a median is simply a
# different number here (0.291 on the same window) carrying no such property.
PAIRWISE_CORR = 0.3103   # 2026 mean off-diagonal pairwise correlation
MARKET_CORR = engine.factor_corr_from_pairwise(PAIRWISE_CORR)
# The homogeneous equicorrelation shortcut predicts dispersion
# = sigma sqrt(1 - rho_r). Against the 2026 measurement it underpredicts by
# about 20% once both sides are matched second moments; the earlier 51% figure
# compared a median of dispersions with a nonlinear function of two medians and
# is not a model test. What the 20% rejects is the HOMOGENEOUS shortcut, which
# assumes one volatility and one correlation for every name, not one-factor
# structure generally: a one-factor model with heterogeneous betas and
# heterogeneous residual variances is untested by this comparison. Dispersion is
# therefore measured rather than derived. One consequence is that the
# cross-sectional per-name floor is BELOW the directional one. See section 15.1.
DISPERSION_ONE_FACTOR = engine.dispersion_from_pairwise_corr(TOTAL_VOL,
                                                             PAIRWISE_CORR)
DISPERSION = 0.0660
SPREAD_BPS = 3.0         # not measurable from klines; still an assumption
VIP0_TAKER, VIP9_TAKER = 5.0, 1.7
COST_VIP0 = (2 * VIP0_TAKER + SPREAD_BPS) * BPS
COST_VIP9 = (2 * VIP9_TAKER + SPREAD_BPS) * BPS

HORIZONS = [("1 h", 1 / 24), ("4 h", 4 / 24), ("1 day", 1.0),
            ("3 days", 3.0), ("1 week", 7.0), ("1 month", 30.0)]


# ---------------------------------------------------------------------------
# 1. the two ICs are orthogonal
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PanelSpec:
    """A panel whose two ICs are set exactly and independently."""
    name: str
    ic_ts: float     # target time-series IC, on the date means
    ic_cs: float     # target cross-sectional IC, on the deviations


def simulate_panel(spec: PanelSpec, n_dates: int, rng, tau: float = 1.0):
    """Build a (T, N) panel with both ICs pinned to their targets.

    The signal always carries both a common component g_t and a cross-sectional
    component c_it, each of unit variance, so that both ICs are *defined*. What
    varies between specs is which component the returns respond to.

        x_it = (g_t + c_it) / sqrt(2)
        y_it = [b_ts g_t + sigma_m eta_t] + [b_cs c_it + sigma_d xi_it]

    Averaging across names kills c_it, so the time-series IC is
    b_ts/sqrt(b_ts^2 + sigma_m^2); demeaning across names kills g_t and the
    market leg, so the cross-sectional IC is b_cs/sqrt(b_cs^2 + sigma_d^2).
    Inverting each gives the two betas below, which is why the realised ICs come
    out on target rather than merely nearby.

    Giving the signal cross-sectional variation even when it has no
    cross-sectional *power* matters: a signal that is literally constant across
    names has an undefined, not a zero, cross-sectional IC, because the
    correlation is 0/0.
    """
    t, n = n_dates, N_ASSETS
    sigma_m = TOTAL_VOL * MARKET_CORR * np.sqrt(tau)
    sigma_d = DISPERSION * np.sqrt(tau)

    g = rng.standard_normal((t, 1))
    c = rng.standard_normal((t, n))
    c -= c.mean(axis=1, keepdims=True)
    c /= c.std(axis=1, keepdims=True)
    x = (g + c) / np.sqrt(2.0)

    b_ts = engine.beta_exact(spec.ic_ts, sigma_m, 1.0) if spec.ic_ts else 0.0
    b_cs = engine.beta_exact(spec.ic_cs, sigma_d, 1.0) if spec.ic_cs else 0.0

    market = b_ts * g + sigma_m * rng.standard_normal((t, 1))
    idio = b_cs * c + sigma_d * rng.standard_normal((t, n))
    return x, market + idio


def study_orthogonality(rng) -> list[dict]:
    specs = [
        PanelSpec("market timing only", 0.04, 0.0),
        PanelSpec("mostly timing", 0.04, 0.02),
        PanelSpec("balanced", 0.03, 0.03),
        PanelSpec("mostly ranking", 0.02, 0.04),
        PanelSpec("ranking only", 0.0, 0.04),
    ]
    rows = []
    for spec in specs:
        x, y = simulate_panel(spec, 20_000, rng)
        d = engine.panel_ic_decomposition(x, y)
        rows.append({"signal": spec.name, "ic_ts_target": spec.ic_ts,
                     "ic_cs_target": spec.ic_cs, **d})
    return rows


# ---------------------------------------------------------------------------
# 2-4. floors, breadth, and an end-to-end portfolio simulation
# ---------------------------------------------------------------------------

def neutral_backtest(cs_ic: float, tau: float, cost: float, n_dates: int, rng,
                     top_fraction: float = 1.0):
    """A dollar-neutral, signal-weighted book on a simulated panel.

    `top_fraction` < 1 trades only the most extreme names on each side, which is
    the cross-sectional version of the no-trade band. Weights are scaled to unit
    gross exposure so that costs and returns are both per unit of capital.
    """
    n = N_ASSETS
    idio_vol = DISPERSION * np.sqrt(tau)
    beta = engine.beta_exact(cs_ic, DISPERSION, tau)
    market_vol = TOTAL_VOL * MARKET_CORR * np.sqrt(tau)

    x = rng.standard_normal((n_dates, n))
    x -= x.mean(axis=1, keepdims=True)
    x /= x.std(axis=1, keepdims=True)
    y = (beta * x
         + idio_vol * rng.standard_normal((n_dates, n))
         + market_vol * rng.standard_normal((n_dates, 1)))

    w = x.copy()
    if top_fraction < 1.0:
        k = max(int(round(top_fraction * n / 2)), 1)
        keep = np.zeros_like(w, dtype=bool)
        order = np.argsort(x, axis=1)
        rows = np.arange(n_dates)[:, None]
        keep[rows, order[:, :k]] = True
        keep[rows, order[:, -k:]] = True
        w = np.where(keep, w, 0.0)
    w -= w.mean(axis=1, keepdims=True)          # enforce dollar neutrality
    gross = np.abs(w).sum(axis=1, keepdims=True)
    w = np.divide(w, gross, out=np.zeros_like(w), where=gross > 0)

    ret = (w * y).sum(axis=1)
    turn = np.abs(np.diff(w, axis=0, prepend=np.zeros((1, n)))).sum(axis=1)
    fees = 0.5 * cost * turn                    # cost per unit of position change
    net = ret - fees

    realised = engine.panel_ic_decomposition(x, y)
    return {
        "cs_ic_target": cs_ic,
        "cs_ic_realised": realised["cross_sectional_ic"],
        "top_fraction": top_fraction,
        "turnover_per_period": float(turn.mean()),
        "gross_bps": float(ret.mean() / BPS),
        "fees_bps": float(fees.mean() / BPS),
        "net_bps": float(net.mean() / BPS),
        "net_se_bps": float(net.std(ddof=1) / np.sqrt(n_dates) / BPS),
        "sharpe_gross_per_period": float(ret.mean() / ret.std(ddof=1)),
        "sharpe_net_per_period": float(net.mean() / net.std(ddof=1)),
        "ir_net_annual": float(net.mean() / net.std(ddof=1)
                               * np.sqrt(365.0 / tau)),
        "ic_times_sqrt_n": cs_ic * np.sqrt(N_ASSETS),
    }


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def fig_orthogonality(rows: list[dict], out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    idx = np.arange(len(rows))
    w = 0.26
    ts = [100 * r["time_series_ic"] for r in rows]
    cs = [100 * r["cross_sectional_ic"] for r in rows]
    pooled = [100 * r["pooled_ic"] for r in rows]
    ts_se = [100 * r["time_series_ic_se"] for r in rows]
    cs_se = [100 * r["cross_sectional_ic_se"] for r in rows]
    ekw = dict(lw=0.7, ecolor=K, capsize=1.8, capthick=0.7)
    ax.bar(idx - w, ts, w, color=NAVY, yerr=ts_se, error_kw=ekw,
           label="time-series IC")
    ax.bar(idx, cs, w, color=RED, yerr=cs_se, error_kw=ekw,
           label="cross-sectional IC")
    ax.bar(idx + w, pooled, w, color=K25, label="pooled (neither)")
    ax.axhline(0, color=K, lw=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels([r["signal"].replace(" only", "\nonly")
                        .replace("mostly ", "mostly\n")
                        .replace("balanced", "balanced\n") for r in rows],
                       fontsize=6.5)
    ax.set_ylabel("IC (%)")
    ax.set_title("one panel, three different numbers")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.06), ncols=3,
              fontsize=6.8, columnspacing=1.2)
    ax.set_ylim(-0.6, 5.9)
    light_grid(ax, axis="y")
    panel_label(ax, "a")

    ax = axes[1]
    first, last = rows[0], rows[-1]
    ax.scatter([100 * first["time_series_ic"]], [100 * first["cross_sectional_ic"]],
               s=26, color=NAVY, zorder=5, label="market timing only")
    ax.scatter([100 * last["time_series_ic"]], [100 * last["cross_sectional_ic"]],
               s=26, color=RED, marker="s", zorder=5, label="ranking only")
    mid = rows[1:-1]
    ax.scatter([100 * r["time_series_ic"] for r in mid],
               [100 * r["cross_sectional_ic"] for r in mid],
               s=20, facecolor="none", edgecolor=K70, zorder=4, label="mixtures")
    ax.axhline(0, color=K, lw=0.6)
    ax.axvline(0, color=K, lw=0.6)
    ax.set_xlabel("time-series IC (%)")
    ax.set_ylabel("cross-sectional IC (%)")
    ax.set_title("the two axes are independent")
    ax.set_xlim(-0.7, 5.6)
    ax.set_ylim(-0.7, 5.6)
    ax.legend(loc="upper left", fontsize=6.8)
    light_grid(ax)
    ax.annotate("times the market,\nranks nothing",
                xy=(100 * first["time_series_ic"], 100 * first["cross_sectional_ic"]),
                xytext=(3.5, 1.35), fontsize=6.8, color=K70, ha="center",
                arrowprops=dict(arrowstyle="->", color=K70, lw=0.5,
                                shrinkB=4))
    ax.annotate("ranks, times\nnothing",
                xy=(100 * last["time_series_ic"], 100 * last["cross_sectional_ic"]),
                xytext=(1.9, 3.1), fontsize=6.8, color=K70, ha="center",
                arrowprops=dict(arrowstyle="->", color=K70, lw=0.5,
                                shrinkB=4))
    panel_label(ax, "b")

    caption(fig,
            "Figure 9. The common-component (date-mean) IC and the "
            "cross-sectional IC measure orthogonal pieces of the same panel, so "
            "neither bounds the other and the pooled correlation over a "
            "flattened panel equals neither. A signal that only times the market "
            "scores zero cross-sectionally; a signal that only ranks scores zero "
            "in the date means. The date-mean statistic is not the general "
            "per-asset time-series IC, which can be positive while every date "
            "mean is zero. Quoting one metric for a strategy that trades on "
            "another is the most common way to mis-state an edge.", y=-0.07)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_floor_and_breadth(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    taus = np.logspace(np.log10(1 / 48), np.log10(45), 400)
    ax.plot(taus, 100 * engine.rho_floor(COST_VIP0, TOTAL_VOL, taus, KAPPA),
            color=NAVY, ls="--",
            label=f"directional, total vol {100*TOTAL_VOL:.0f}%/day")
    ax.plot(taus, 100 * engine.cross_sectional_floor(COST_VIP0, DISPERSION,
                                                     taus, KAPPA),
            color=RED, label=f"cross-sectional, dispersion "
                             f"{100*DISPERSION:.1f}%/day")
    ax.plot(taus, 100 * engine.portfolio_breakeven_ic(COST_VIP0, DISPERSION,
                                                      taus, turnover=1.0),
            color=K, ls=":", label="whole section, one round trip")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(1 / 48, 45)
    ax.set_ylim(0.08, 40)
    ax.set_xlabel(r"holding period $\tau$ (days)")
    ax.set_ylabel("minimum IC")
    percent_log_ticks(ax, "y", subs=(1.0, 2.0, 5.0))
    ax.set_title("VIP 0, alt perp, 13 bps round trip")
    ax.legend(loc="upper right")
    light_grid(ax)
    for tau, lab in ((1.0, "1 d"), (7.0, "1 w"), (30.0, "1 m")):
        ax.axvline(tau, color=K45, lw=0.4, ls=":")
        ax.text(tau, 0.093, lab, fontsize=6.5, color=K70, ha="center",
                bbox=dict(fc="white", ec="none", pad=0.6))
    panel_label(ax, "a")

    ax = axes[1]
    corrs = np.linspace(0.0, 0.95, 300)
    for n, colour, ls in ((20, K45, "-."), (50, OCHRE, "--"), (200, NAVY, "-")):
        ax.plot(corrs, [engine.breadth_common_position(n, c) for c in corrs],
                color=colour, ls=ls, label=f"directional, $N={n}$")
    ax.axhline(engine.breadth_neutral(N_ASSETS), color=RED, lw=1.1,
               label=f"dollar-neutral, $N={N_ASSETS}$")
    ax.axvspan(0.40, 0.75, color=K25, alpha=0.4, lw=0, zorder=0)
    ax.text(0.575, 270, "crypto", fontsize=7, color=K70, ha="center",
            style="italic")
    ax.set_yscale("log")
    ax.set_xlim(0, 0.95)
    ax.set_ylim(0.8, 260)   # neutral breadth is flat at N-1 = 199
    ax.set_xlabel("pairwise correlation of asset returns")
    ax.set_ylabel("effective breadth (bets per period)")
    ax.set_title("neutrality is what buys breadth")
    ax.legend(loc="center right", fontsize=6.8)
    light_grid(ax)
    bd = engine.breadth_common_position(N_ASSETS, PAIRWISE_CORR)
    bn = engine.breadth_neutral(N_ASSETS)
    ax.annotate(f"{bn/bd:.0f}$\\times$ the bets,\n"
                f"{np.sqrt(bn/bd):.1f}$\\times$ the IR",
                xy=(PAIRWISE_CORR, bd), xytext=(0.24, 9),
                fontsize=7, color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
    panel_label(ax, "b")

    caption(fig,
            "Figure 10. The two effects of going cross-sectional pull in opposite "
            "directions and the second wins comfortably. (a) A dollar-neutral book "
            f"earns dispersion ({100*DISPERSION:.2f}%/day) rather than total "
            f"volatility ({100*TOTAL_VOL:.0f}%/day), so its per-name floor is "
            f"{TOTAL_VOL/DISPERSION:.2f}$\\times$ the directional one, i.e. "
            f"{100*(1-TOTAL_VOL/DISPERSION):.0f}% LOWER; holding the whole ranking "
            "rather than only the extremes raises the bar further. (b) Common-position "
            r"breadth saturates at $1/\rho_r$ regardless of $N$, so at "
            f"$\\rho_r={PAIRWISE_CORR}$ a 200-name directional book has {bd:.2f} "
            f"independent bets against the {bn:.0f} the equicorrelation model "
            "assigns the neutral one. That $N-1$ is an independence bound rather "
            "than a measurement, and uses only the return correlation: a strategy "
            "information ratio also needs forecasts and weights. Both panels are "
            "driven by the same pairwise correlation, so the dispersion penalty "
            "and the modelled breadth gain cannot be tuned apart.", y=-0.07)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_backtest(sim: list[dict], trunc: list[dict], est: list[dict],
                 out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    ics = np.array([r["cs_ic_realised"] for r in sim])
    ax.plot(100 * ics, [r["sharpe_gross_per_period"] for r in sim],
            color=K45, ls="--", marker="o", ms=2.6,
            label="gross, simulated")
    ax.plot(100 * ics, ics * np.sqrt(N_ASSETS), color=K, ls=":",
            label=r"theory, $\mathrm{IC}\sqrt{N}$")
    ax.plot(100 * ics, [r["sharpe_net_per_period"] for r in sim],
            color=RED, marker="s", ms=2.6, label="net of VIP 0 fees")
    ax.plot(100 * np.array([r["cs_ic_realised"] for r in trunc]),
            [r["sharpe_net_per_period"] for r in trunc],
            color=NAVY, marker="^", ms=2.6,
            label="net, extreme deciles only")
    ax.axhline(0, color=K, lw=0.6)
    ax.set_xlabel("realised cross-sectional IC (%)")
    ax.set_ylabel("Sharpe ratio per rebalance")
    ax.set_title(f"daily rebalance, $N={N_ASSETS}$")
    ax.legend(loc="upper left")
    light_grid(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ics2 = np.array([r["ic"] for r in est])
    ax.plot(100 * ics2, [r["ts_obs"] for r in est], color=NAVY, ls="--",
            marker="o", ms=2.6, label="time-series, one name")
    ax.plot(100 * ics2, [r["cs_dates"] for r in est], color=RED, marker="s",
            ms=2.6, label=f"cross-sectional, $N={N_ASSETS}$")
    for days, lab in ((365, "1 year"), (365 * 10, "10 years")):
        ax.axhline(days, color=K45, lw=0.4, ls=":")
        ax.text(9.6, days * 1.3, lab, fontsize=6.5, color=K70, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("IC")
    percent_log_ticks(ax, "x", subs=(1.0, 2.0, 5.0))
    ax.set_ylabel(r"daily observations for $|t|=3$")
    ax.set_title("cross-sectional is $N$ times cheaper to measure")
    ax.legend(loc="upper right")
    light_grid(ax)
    panel_label(ax, "b")

    be = engine.portfolio_breakeven_ic(COST_VIP0, DISPERSION, 1.0,
                                       turnover=sim[0]["turnover_per_period"])
    caption(fig,
            "Figure 11. (a) A simulated dollar-neutral book reproduces "
            r"$\mathrm{IC}\sqrt{N}$ gross. VIP 0 fees on a daily rebalance put "
            f"the break-even at {100*be:.2f}% (theory) against a measured "
            f"{100*sim[2]['cs_ic_realised']:.2f}% -- above the "
            f"{100*engine.cross_sectional_floor(COST_VIP0, DISPERSION, 1.0):.2f}"
            r"% per-name floor, because the book holds the whole ranking rather "
            "than only its best names. Trading the extreme deciles raises gross "
            "and turnover together, so it helps a strong signal and hurts a weak "
            r"one. (b) The same $\sqrt{N}$ that buys breadth buys statistical "
            f"power: a 2% IC needs {est[2]['ts_obs']/365:.0f} years of one "
            f"name's history in the time series against {est[2]['cs_dates']:.0f} "
            "days across 200 of them.", y=-0.07)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


# ---------------------------------------------------------------------------

def main() -> None:
    paths.ensure_directories()
    rng = np.random.default_rng(SEED)
    out: dict = {"seed": SEED, "n_assets": N_ASSETS,
                 "total_vol_daily": TOTAL_VOL, "market_corr": MARKET_CORR,
                 "dispersion_daily": DISPERSION,
                 "cost_vip0_bps": COST_VIP0 / BPS,
                 "cost_vip9_bps": COST_VIP9 / BPS}

    print("=" * 78)
    print("1. Common-component and cross-sectional IC are orthogonal pieces")
    print("=" * 78)
    rows = study_orthogonality(rng)
    print(f"{'signal':21s} {'target':>13s}   {'measured TS IC':>18s} "
          f"{'measured CS IC':>18s} {'pooled':>8s}")
    print(f"{'':21s} {'TS':>6s} {'CS':>6s}   {'':>18s} {'':>18s} {'':>8s}")
    for r in rows:
        print(f"{r['signal']:21s} {100*r['ic_ts_target']:5.1f}% "
              f"{100*r['ic_cs_target']:5.1f}%   "
              f"{100*r['time_series_ic']:6.2f}% +/-"
              f"{100*r['time_series_ic_se']:5.2f} (t{r['time_series_ic_tstat']:5.1f}) "
              f"{100*r['cross_sectional_ic']:6.2f}% +/-"
              f"{100*r['cross_sectional_ic_se']:5.3f} "
              f"(t{r['cross_sectional_ic_tstat']:5.1f}) "
              f"{100*r['pooled_ic']:7.2f}%")
    out["orthogonality"] = rows
    print("\n  A signal can score on either axis independently: the timing-only")
    print("  row has a cross-sectional IC indistinguishable from zero, and the")
    print("  ranking-only row a time-series IC indistinguishable from zero. The")
    print("  pooled correlation is neither of them and should not be quoted for")
    print("  either kind of book.")
    print("\n  Note the error bars. Both ICs come from the same panel, but the")
    print("  time-series estimate uses T dates and the cross-sectional one uses")
    print("  N*T cells, so the second is ~sqrt(N) = "
          f"{np.sqrt(N_ASSETS):.0f}x tighter. That is finding 5, visible here.")

    print("\n" + "=" * 78)
    print("2. Three different floors")
    print("=" * 78)
    print(f"  single-name total vol   {100*TOTAL_VOL:5.2f}%/day")
    print(f"  market correlation      {MARKET_CORR:5.2f}")
    print(f"  cross-sec. dispersion   {100*DISPERSION:5.2f}%/day  MEASURED")
    print(f"  homog. equicorr shortcut {100*DISPERSION_ONE_FACTOR:5.2f}%/day  "
          f"(= median vol x sqrt(1 - median rho_r))")
    print("    NOT a model test: this mixes a median vol and a median")
    print("    correlation through a nonlinear formula and compares against a")
    print("    median dispersion. The matched second-moment comparison in")
    print("    calibrate_from_data.py puts the 2026 shortfall at ~20%, and what")
    print("    that rejects is the HOMOGENEOUS shortcut, not one-factor")
    print("    structure with heterogeneous betas.")
    print(f"  round trip, VIP 0       {COST_VIP0/BPS:5.2f} bps\n")
    print(f"{'horizon':>8s} {'directional':>13s} {'per name, CS':>13s} "
          f"{'section, t=2':>14s} {'section, t=1.41':>13s} {'ratio':>7s}")
    floors = []
    for hname, tau in HORIZONS:
        d = engine.rho_floor(COST_VIP0, TOTAL_VOL, tau, KAPPA)
        c = engine.cross_sectional_floor(COST_VIP0, DISPERSION, tau, KAPPA)
        p = engine.portfolio_breakeven_ic(COST_VIP0, DISPERSION, tau)
        pm = engine.portfolio_breakeven_ic(COST_VIP0, DISPERSION, tau,
                                           turnover=1.414)
        print(f"{hname:>8s} {100*d:12.2f}% {100*c:12.2f}% {100*p:13.2f}% "
              f"{100*pm:12.2f}% {pm/c:6.2f}x")
        floors.append({"horizon": hname, "tau_days": tau,
                       "directional_floor": d, "cs_name_floor": c,
                       "cs_portfolio_breakeven_full_roundtrip": p,
                       "cs_portfolio_breakeven_measured_turnover": pm})
    out["floors"] = floors
    ratio = TOTAL_VOL / DISPERSION
    verb = "lowers" if ratio < 1 else "raises"
    print(f"\n  Going cross-sectional {verb} the per-name bar, by "
          f"{ratio:.2f}x: measured dispersion")
    print(f"  ({100*DISPERSION:.2f}%) EXCEEDS median single-name vol "
          f"({100*TOTAL_VOL:.2f}%), because cross-sectional")
    print(f"  spread is driven by the right tail of the vol distribution. The "
          f"{TOTAL_VOL/DISPERSION_ONE_FACTOR:.2f}x penalty")
    print("  the homogeneous equicorrelation shortcut predicts is absent here.")
    print("  Holding")
    print(f"  the whole ranking raises it a further "
          f"{KAPPA*engine.SQRT_2_OVER_PI:.2f}x at one round trip per period,")
    print(f"  or {KAPPA*engine.SQRT_2_OVER_PI*1.414/2:.2f}x at the 1.41 turnover "
          "a signal-weighted book actually runs.")

    print("\n" + "=" * 78)
    print("3. Breadth: why crypto forces cross-sectional construction")
    print("=" * 78)
    print("  Both columns come from the SAME pairwise correlation, so the")
    print("  dispersion penalty and the breadth gain move together.\n")
    print(f"{'pairwise':>9s} {'factor':>7s} {'disp(1-fac)':>11s} "
          f"{'directional':>12s} {'neutral':>8s} {'IR ratio':>9s} "
          f"{'break-even':>11s}")
    print("  (the dispersion column is the HOMOGENEOUS equicorrelation")
    print("   prediction at that rho_r;")
    print("   the measured 2026 value is 6.60%, which the data prefers)")
    breadth = []
    for corr in (0.2, PAIRWISE_CORR, 0.4, 0.49, 0.55, 0.7, 0.8):
        bd = engine.breadth_common_position(N_ASSETS, corr)
        bn = engine.breadth_neutral(N_ASSETS)
        disp = engine.dispersion_from_pairwise_corr(TOTAL_VOL, corr)
        be = engine.portfolio_breakeven_ic(COST_VIP0, disp, 1.0, turnover=1.414)
        mark = " <-" if abs(corr - PAIRWISE_CORR) < 1e-9 else ""
        print(f"{corr:9.2f} {engine.factor_corr_from_pairwise(corr):7.2f} "
              f"{100*disp:10.2f}% {bd:12.2f} {bn:8.0f} "
              f"{np.sqrt(bn/bd):8.1f}x {100*be:10.2f}%{mark}")
        breadth.append({"pairwise_corr": corr,
                        "factor_corr": engine.factor_corr_from_pairwise(corr),
                        "dispersion": disp, "directional": bd, "neutral": bn,
                        "ir_ratio": float(np.sqrt(bn / bd)),
                        "portfolio_breakeven": be})
    out["breadth"] = breadth
    bd0 = engine.breadth_common_position(N_ASSETS, PAIRWISE_CORR)
    print(f"\n  At the central rho_r = {PAIRWISE_CORR} (factor correlation "
          f"{MARKET_CORR:.2f}) a 200-name")
    print(f"  directional book has {bd0:.2f} independent bets. Adding names does "
          f"not help:")
    print(f"  the ceiling is 1/rho_r = {1/PAIRWISE_CORR:.2f}. The equicorrelation")
    print(f"  MODEL sets neutral breadth to N-1 = "
          f"{engine.breadth_neutral(N_ASSETS):.0f}, which implies")
    los = min(r["ir_ratio"] for r in breadth)
    his = max(r["ir_ratio"] for r in breadth)
    print(f"  {np.sqrt(engine.breadth_neutral(N_ASSETS)/bd0):.1f}x the IR, and "
          f"{los:.0f}x to {his:.0f}x across the range above.")
    print("  N-1 is an INDEPENDENCE BOUND, not a measurement, and these ratios")
    print("  use only the RETURN correlation: a strategy IR ratio also needs")
    print("  forecasts and portfolio weights, which this script does not model.")
    print("  Read them as what the equicorrelation model implies. The 2026 fall")
    print("  in correlation cuts the modelled advantage: a less correlated")
    print("  market leaves a directional book more breadth of its own.")

    print("\n" + "=" * 78)
    print("4. End-to-end: a simulated dollar-neutral book at VIP 0")
    print("=" * 78)
    print(f"{'CS IC':>7s} {'realised':>9s} {'turnover':>9s} {'gross':>8s} "
          f"{'fees':>8s} {'net':>9s} {'+/-':>7s} {'SR/period':>10s} "
          f"{'net IR':>8s}")
    sim, trunc = [], []
    for ic in (0.005, 0.01, 0.02, 0.026, 0.03, 0.05, 0.08):
        r = neutral_backtest(ic, 1.0, COST_VIP0, 6000, rng)
        print(f"{100*ic:6.1f}% {100*r['cs_ic_realised']:8.2f}% "
              f"{r['turnover_per_period']:9.3f} {r['gross_bps']:7.2f}b "
              f"{r['fees_bps']:7.2f}b {r['net_bps']:8.2f}b "
              f"{r['net_se_bps']:6.2f}b {r['sharpe_net_per_period']:10.4f} "
              f"{r['ir_net_annual']:8.2f}")
        sim.append(r)
        trunc.append(neutral_backtest(ic, 1.0, COST_VIP0, 6000, rng,
                                      top_fraction=0.2))
    out["backtest_full"] = sim
    out["backtest_deciles"] = trunc

    print("\n  gross Sharpe against theory (IC sqrt(N)):")
    for r in sim:
        print(f"    IC {100*r['cs_ic_target']:5.1f}%  simulated "
              f"{r['sharpe_gross_per_period']:.4f}  theory "
              f"{r['ic_times_sqrt_n']:.4f}")

    print("\n  extreme deciles only (the cross-sectional no-trade band):")
    print(f"{'CS IC':>7s} {'turnover':>9s} {'net':>9s} {'+/-':>7s} "
          f"{'net IR':>8s}")
    for r in trunc:
        print(f"{100*r['cs_ic_target']:6.1f}% {r['turnover_per_period']:9.3f} "
              f"{r['net_bps']:8.2f}b {r['net_se_bps']:6.2f}b "
              f"{r['ir_net_annual']:8.2f}")

    print("\n" + "=" * 78)
    print("5. Cross-sectional ICs are far cheaper to establish")
    print("=" * 78)
    print(f"{'IC':>6s} {'TS obs (1 name)':>17s} {'as years':>10s} "
          f"{'CS dates (N=200)':>18s} {'as months':>11s}")
    est = []
    for ic in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10):
        ts = engine.obs_for_tstat(ic, 3.0)
        cs = engine.dates_for_cs_ic(ic, N_ASSETS, 3.0)
        print(f"{100*ic:5.1f}% {ts:17,.0f} {ts/365:9.1f}y "
              f"{cs:18,.0f} {cs/30:10.1f}m")
        est.append({"ic": ic, "ts_obs": ts, "cs_dates": cs,
                    "ratio": ts / cs})
    out["estimation"] = est
    print("\n  At t=3 the exact ratio used here is "
          f"N[(1 - IC^2)^2 + IC^2/9], i.e. N = {N_ASSETS} to within IC^2.")
    print("  The same sqrt(N) that buys breadth buys statistical power -- the")
    print("  symmetry in README section 7D.")
    print("\n  Caveat: this assumes a stable IC, so the per-date dispersion is")
    print("  1/sqrt(N). Measured IC series are far noisier than that because the")
    print("  true IC moves, and it is that instability, not sampling error, that")
    print("  usually binds. Use the measured sd(IC_t), not 1/sqrt(N).")

    fig_orthogonality(rows, paths.FIGURES_DIR / "fig9_ts_vs_cs.png")
    fig_floor_and_breadth(paths.FIGURES_DIR / "fig10_cs_floor_breadth.png")
    fig_backtest(sim, trunc, est, paths.FIGURES_DIR / "fig11_cs_backtest.png")

    path = paths.RESULTS_DIR / "cross_sectional_ic.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {paths.rel(path)}")


if __name__ == "__main__":
    main()
