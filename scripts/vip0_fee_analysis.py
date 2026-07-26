"""The correlation floor at Binance VIP 0: taker 5.0 bps, maker 2.0 bps.

VIP 0 is the tier almost every retail account actually trades on, and it changes
the conclusion qualitatively rather than quantitatively. Three findings, each
derived rather than asserted:

1. The retail wedge is a factor of ~2.9 in required correlation, which is a
   factor of ~8.6 in required holding period, because the floor goes as
   c/sqrt(tau) and therefore tau goes as c^2.

2. Pure spread capture does not self-finance at VIP 0, on majors. Quoting both
   sides pays 4 bps in maker fees to collect a spread of about 0.0151 bps on
   BTCUSDT perp, a net 3.98 bps. Note what this does and does not say: passive
   quoting is still the *cheapest* way to express a view at VIP 0 (3.98 bps
   against 10.02 bps for crossing both ways), so it remains the right execution.
   What it cannot do is pay for itself. Market making as a spread-capture
   business needs 2 x maker fee <= spread, which at VIP 0 means a quoted spread
   of at least 4 bps; at VIP 9 the same trade is already net negative, which is
   the entire reason the tier exists.

3. Fees and funding bracket the horizon from both ends, and for a book that pays
   funding this produces a horizon-free floor on the correlation at which a
   kappa=3 opportunity exists at ANY horizon:

       net(tau) = kappa rho sigma sqrt(tau) - c - f tau

   is positive for some tau if and only if

       rho > 2 sqrt(f c) / (kappa sigma),

   with the best horizon at tau* = (kappa rho sigma / (2 f))^2. Both closed forms
   are verified symbolically below.

Finding 3 is about the kappa=3 opportunity, not about profitability: under the
Gaussian model the band rule keeps a positive but rare tail edge at any
correlation, whatever the carry. It also assumes the book pays funding, which is
the long-biased case. A dollar-neutral book has zero EXPECTED funding only if the
rates are common across names or uncorrelated with the weights; otherwise a
neutral book can still carry a systematic funding bill. Both cases are reported.

Output: results/vip0_fee_analysis.json, figures/fig6-fig8.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import paths  # noqa: E402
from _style_academic import (  # noqa: E402
    K, K25, K45, K70, NAVY, OCHRE, RED, TEAL, caption, light_grid, panel_label,
    percent_log_ticks, plt,
)

BPS = 1e-4
KAPPA = 3.0
SEC = 1.0 / 86_400.0


@dataclass(frozen=True)
class Tier:
    name: str
    maker_bps: float
    taker_bps: float


VIP0 = Tier("VIP 0", 2.0, 5.0)
VIP0_BNB = Tier("VIP 0, BNB discount", 1.8, 4.5)
VIP9 = Tier("VIP 9", 0.0, 1.7)
TIERS = [VIP0, VIP0_BNB, VIP9]


@dataclass(frozen=True)
class Instrument:
    name: str
    short: str
    sigma_daily: float
    tick_bps: float
    spread_bps: float
    funding_bps_per_day: float


# Calibrated on 2026 Binance data by scripts/calibrate_from_data.py.
# 2026 only: the 2020-21 volatility and correlation regime no longer
# describes this market. See results/calibration.json, README section 15.
# Volatility, tick and funding are MEASURED; spread is not measurable from
# klines and stays an assumption. Funding is the MEDIAN absolute rate, which is
# 3.0 bps/day under either weighting; the symbol-day mean is 11.5 bps/day and the
# event-weighted mean 20.9, because the distribution is extremely fat-tailed. The
# sensitivity table in section 4 covers the range explicitly.
BTC = Instrument("BTCUSDT perp", "BTC", 0.0250, 0.0151, 0.0151, 3.0)
ETH = Instrument("ETHUSDT perp", "ETH", 0.0336, 0.0517, 0.0517, 3.0)
ALT = Instrument("mid-cap alt perp", "alt", 0.0519, 1.28, 3.0, 3.0)
INSTRUMENTS = [BTC, ETH, ALT]

HORIZONS = [
    ("1 s", SEC), ("10 s", 10 * SEC), ("1 min", 60 * SEC),
    ("5 min", 300 * SEC), ("30 min", 1800 * SEC), ("1 h", 3600 * SEC),
    ("4 h", 4 * 3600 * SEC), ("1 day", 1.0), ("3 days", 3.0), ("1 week", 7.0),
]


# ---------------------------------------------------------------------------
# costs
# ---------------------------------------------------------------------------

def cost_taker_taker(inst: Instrument, tier: Tier) -> float:
    """Cross in, cross out: two taker fees plus the full spread."""
    return 2 * tier.taker_bps * BPS + inst.spread_bps * BPS


def cost_maker_taker(inst: Instrument, tier: Tier) -> float:
    """Post to enter, cross to exit: one fee of each plus one half-spread."""
    return (tier.maker_bps + tier.taker_bps) * BPS + 0.5 * inst.spread_bps * BPS


def cost_maker_maker(inst: Instrument, tier: Tier) -> float:
    """Quote both sides: two maker fees, less the spread received.

    Negative is the point of a rebate tier. At VIP 0 it is firmly positive.
    """
    return 2 * tier.maker_bps * BPS - inst.spread_bps * BPS


STYLES = [
    ("cross in, cross out", cost_taker_taker),
    ("post in, cross out", cost_maker_taker),
    ("quote both sides", cost_maker_maker),
]


# ---------------------------------------------------------------------------
# the fee-and-funding lens
# ---------------------------------------------------------------------------

def verify_lens_symbolically() -> dict:
    """Prove the two closed forms this analysis leans on."""
    rho, sigma, u, c, f, kappa = sp.symbols(
        "rho sigma u c f kappa", positive=True
    )
    # Work in u = sqrt(tau) so the objective is a plain quadratic.
    net_u = kappa * rho * sigma * u - c - f * u**2

    u_star = sp.solve(sp.diff(net_u, u), u)[0]
    tau_star = sp.simplify(u_star**2)
    peak = sp.simplify(net_u.subs(u, u_star))

    rho_candidates = sp.solve(sp.Eq(peak, 0), rho)
    rho_star = [r for r in rho_candidates if r.is_positive is not False][0]
    target_rho = 2 * sp.sqrt(f * c) / (kappa * sigma)
    target_tau = (kappa * rho * sigma / (2 * f)) ** 2

    return {
        "u_star": str(sp.simplify(u_star)),
        "tau_star": str(tau_star),
        "tau_star_matches_closed_form":
            bool(sp.simplify(tau_star - target_tau) == 0),
        "peak_net": str(peak),
        "peak_net_matches_closed_form":
            bool(sp.simplify(peak - ((kappa * rho * sigma) ** 2 / (4 * f) - c))
                 == 0),
        "rho_floor": str(sp.simplify(rho_star)),
        "rho_floor_matches_closed_form":
            bool(sp.simplify(sp.expand(rho_star**2 - target_rho**2)) == 0),
    }


def _funding(inst: Instrument, funding_bps: float | None) -> float:
    return (inst.funding_bps_per_day if funding_bps is None
            else funding_bps) * BPS


def rho_floor_with_funding(inst: Instrument, cost: float,
                           funding_bps: float | None = None) -> float:
    return engine.rho_floor_with_carry(cost, inst.sigma_daily,
                                       _funding(inst, funding_bps), KAPPA)


def best_horizon(inst: Instrument, rho: float,
                 funding_bps: float | None = None) -> float:
    return engine.optimal_horizon_with_carry(rho, inst.sigma_daily,
                                             _funding(inst, funding_bps), KAPPA)


def net_edge(inst: Instrument, rho: float, tau, cost: float,
             with_funding: bool = True):
    """Net bps per round trip on a kappa-sigma signal."""
    carry = _funding(inst, None) if with_funding else 0.0
    return engine.net_edge_with_carry(rho, inst.sigma_daily, tau, cost,
                                      carry, KAPPA) / BPS


def min_horizon(inst: Instrument, cost: float, rho: float) -> float:
    return engine.min_horizon_fee_only(cost, inst.sigma_daily, rho, KAPPA)


def fmt_tau(tau_days: float) -> str:
    if tau_days <= 0:
        return "any"
    s = tau_days * 86_400
    if s < 1:
        return f"{s*1000:.0f} ms"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s/60:.0f} min"
    if tau_days < 2:
        return f"{s/3600:.1f} h"
    if tau_days < 400:
        return f"{tau_days:.1f} d"
    return f"{tau_days/365:.1f} yr"


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

IC_BAND = (0.01, 0.20)  # what a real crypto alpha plausibly attains


def fig_retail_wedge(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    taus = np.logspace(np.log10(SEC), np.log10(14.0), 500)
    for tier, colour, ls in ((VIP0, RED, "-"), (VIP0_BNB, OCHRE, "--"),
                             (VIP9, NAVY, "-")):
        c = cost_taker_taker(BTC, tier)
        ax.plot(taus, 100 * engine.rho_floor(c, BTC.sigma_daily, taus, KAPPA),
                color=colour, ls=ls,
                label=f"{tier.name} ({c/BPS:.1f} bps)")
    ax.axhspan(100 * IC_BAND[0], 100 * IC_BAND[1], color=K25, alpha=0.4, lw=0,
               zorder=0)
    ax.text(1.4e-5, 3.0, "plausible alpha", fontsize=7, color=K70,
            style="italic")
    # Above rho = 100% the required correlation is not attainable by any
    # signal whatsoever. Shade that band so the axis cannot be read as
    # displaying correlations above unity: where a curve enters it, the
    # kappa=3 criterion is unsatisfiable at that horizon at any quality.
    ax.axhspan(100, 500, color=RED, alpha=0.10, lw=0, zorder=0)
    ax.axhline(100, color=RED, lw=0.8, ls="--", zorder=1)
    ax.text(0.35, 205, "no signal can reach this band\n"
            r"($\rho_{\min}>100\%$: $\kappa=3$ unsatisfiable)",
            fontsize=7, color=RED, style="italic", ha="center", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(SEC, 14)
    ax.set_ylim(0.05, 500)
    ax.set_xlabel(r"holding period $\tau$ (days)")
    ax.set_ylabel(r"required correlation $\rho_{\min}$")
    percent_log_ticks(ax, "y")
    ax.set_title("BTC perp, cross in and cross out")
    ax.legend(loc="lower left")
    light_grid(ax)
    for label, tau in (("1 min", 60 * SEC), ("1 h", 3600 * SEC), ("1 d", 1.0)):
        ax.axvline(tau, color=K45, lw=0.4, ls=":")
        ax.text(tau, 0.062, label, fontsize=6.5, color=K70, ha="center",
                bbox=dict(fc="white", ec="none", pad=0.6))
    panel_label(ax, "a")

    ax = axes[1]
    ics = np.logspace(np.log10(0.005), np.log10(0.30), 400)
    for (label, fn), colour, ls in zip(STYLES, (RED, OCHRE, NAVY),
                                       ("-", "--", "-.")):
        c = fn(BTC, VIP0)
        if c <= 0:
            continue
        ax.plot(100 * ics, [min_horizon(BTC, c, r) for r in ics], color=colour,
                ls=ls, label=f"{label}, VIP 0 ({c/BPS:.2f} bps)")
    c9 = cost_taker_taker(BTC, VIP9)
    ax.plot(100 * ics, [min_horizon(BTC, c9, r) for r in ics], color=K45,
            lw=0.9, ls=(0, (1, 1.5)), label=f"cross both ways, VIP 9 "
                                            f"({c9/BPS:.2f} bps)")
    for tau, lab in ((60 * SEC, "1 min"), (3600 * SEC, "1 h"), (1.0, "1 day"),
                     (7.0, "1 week")):
        ax.axhline(tau, color=K45, lw=0.4, ls=":")
        ax.text(29, tau * 1.25, lab, fontsize=6.5, color=K70, ha="right")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.5, 30)
    ax.set_ylim(1e-5, 80)
    ax.set_xlabel("time-series IC")
    percent_log_ticks(ax, "x", subs=(1.0, 2.0, 5.0))
    ax.set_ylabel("shortest viable holding period (days)")
    ax.set_title("the fee expressed as required horizon")
    ax.legend(loc="lower left")
    light_grid(ax)
    panel_label(ax, "b")

    ratio = cost_taker_taker(BTC, VIP0) / cost_taker_taker(BTC, VIP9)
    caption(fig,
            "Figure 6. The retail fee wedge on BTCUSDT perpetuals. VIP 0 costs "
            f"{ratio:.2f}$\\times$ the round trip of the top tier, so it demands "
            f"{ratio:.2f}$\\times$ the correlation at a fixed horizon and "
            f"{ratio**2:.1f}$\\times$ the horizon at a fixed correlation, since "
            r"$\rho_{\min}\propto c/\sqrt{\tau}$. Shading in (a) is the range a "
            "real short-horizon crypto alpha plausibly attains, 1 to 20 per "
            "cent. Where a curve sits ABOVE the band the criterion demands "
            "more correlation than such an alpha delivers; inside the red band "
            "it demands more than 100 per cent, which no signal of any quality "
            "supplies. VIP 9 is inside the grey band from about 45 s to 5 h and "
            "VIP 0 from about 6 min to 43 h, the wedge restated as a horizon.",
            y=-0.07)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_making(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    x = np.arange(len(INSTRUMENTS))
    w = 0.34
    fees = [2 * VIP0.maker_bps for _ in INSTRUMENTS]
    spreads = [i.spread_bps for i in INSTRUMENTS]
    ax.bar(x - w / 2, fees, w, color=RED, label="maker fees paid, two sides")
    ax.bar(x + w / 2, spreads, w, color=NAVY, label="spread captured")
    for xi, v in zip(x - w / 2, fees):
        ax.text(xi, v * 1.3, f"{v:.1f}", ha="center", fontsize=6.5, color=K)
    for xi, v in zip(x + w / 2, spreads):
        ax.text(xi, v * 1.4, f"{v:g}", ha="center", fontsize=6.5, color=K)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([i.short for i in INSTRUMENTS])
    ax.set_ylim(5e-3, 80)
    ax.set_ylabel("bps per passive round trip")
    ax.set_title("cost of quoting both sides, VIP 0")
    ax.legend(loc="upper left")
    light_grid(ax, axis="y")
    ratio = 2 * VIP0.maker_bps / BTC.spread_bps
    ax.annotate(f"fee is {ratio:.0f}$\\times$\nthe spread",
                xy=(w / 2, BTC.spread_bps * 1.1), xytext=(0.30, 0.35),
                fontsize=7, color=RED, ha="center", va="center",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.6,
                                shrinkA=1, shrinkB=2))
    panel_label(ax, "a")

    ax = axes[1]
    # Stop the x-range where the curves would otherwise exit the frame: at a
    # 40 bps spread the VIP 0 line sits at -36, far below any sensible
    # y-limit, leaving most of the axis empty of data.
    grid = np.logspace(np.log10(0.005), np.log10(12), 400)
    for tier, colour, ls in ((VIP0, RED, "-"), (VIP0_BNB, OCHRE, "--"),
                             (VIP9, NAVY, "-.")):
        ax.plot(grid, 2 * tier.maker_bps - grid, color=colour, ls=ls,
                label=f"{tier.name}, maker {tier.maker_bps:.1f} bps")
    ax.axhline(0, color=K, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(0.005, 12)
    ax.set_ylim(-12.5, 5.5)
    ax.set_xlabel("quoted spread (bps)")
    ax.set_ylabel("net cost of a passive round trip (bps)")
    ax.set_title(r"break-even needs spread $\geq 2\times$ the maker fee")
    for inst, colour in zip(INSTRUMENTS, (K, K70, K45)):
        ax.axvline(inst.spread_bps, color=colour, lw=0.4, ls=":")
        ax.text(inst.spread_bps, 4.7, inst.short, fontsize=6.5, color=colour,
                ha="center", bbox=dict(fc="white", ec="none", pad=0.6))
    be = 2 * VIP0.maker_bps
    ax.plot([be], [0], marker="o", ms=3.5, color=RED, zorder=5)
    ax.annotate(f"VIP 0 break-even, {be:.0f} bps", xy=(be, 0),
                xytext=(0.42, 0.78), textcoords="axes fraction", fontsize=7,
                color=RED, ha="center",
                arrowprops=dict(arrowstyle="->", color=RED, lw=0.6))
    # The profitable half-plane is net cost < 0, i.e. BELOW the zero line.
    # Anchor the note in the empty lower-left region so it stays in frame.
    ax.text(0.0075, -6.0, "below the line:\nnet cost $<0$, profitable",
            fontsize=7, color=K70, style="italic", ha="left", va="center")
    ax.legend(loc="lower left")
    light_grid(ax)
    panel_label(ax, "b")

    caption(fig,
            "Figure 7. Spread capture does not self-finance at VIP 0. One tick "
            "on BTCUSDT perp is about 0.0151 bps while quoting both sides costs "
            f"4 bps in maker fees, so the fee exceeds the spread by {ratio:.0f}"
            r"$\times$ and the net is $+3.98$ bps. Passive quoting is still the "
            "cheapest way to express a view at this tier; what it cannot do is "
            "pay for itself. Break-even needs a quoted spread of at least twice "
            "the maker fee, which here means instruments carrying the worst "
            "adverse selection.", y=-0.07)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


def fig_lens(out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))
    cost = cost_taker_taker(BTC, VIP0)
    rho_abs = rho_floor_with_funding(BTC, cost)

    ax = axes[0]
    taus = np.logspace(-3, np.log10(300), 600)
    for rho, colour, ls in ((0.01, K45, "-."), (0.015, OCHRE, "--"),
                            (0.03, TEAL, "-"), (0.08, NAVY, "-")):
        ax.plot(taus, net_edge(BTC, rho, taus, cost, with_funding=True),
                color=colour, ls=ls, label=rf"IC $={100*rho:g}\%$")
        ax.plot(taus, net_edge(BTC, rho, taus, cost, with_funding=False),
                color=colour, ls=(0, (1, 2)), lw=0.6, alpha=0.7)
    ax.axhline(0, color=K, lw=0.7)
    ax.set_xscale("log")
    ax.set_xlim(1e-3, 300)
    ax.set_ylim(-28, 98)
    ax.set_xlabel(r"holding period $\tau$ (days)")
    ax.set_ylabel("net edge per round trip (bps)")
    ax.set_title("BTC perp, VIP 0, funding 3 bps/day")
    ax.legend(loc="upper left", ncols=2)
    light_grid(ax)
    ax.text(1.4e-3, -24, "dotted: fees only\nsolid: fees and funding",
            fontsize=6.5, color=K70)
    panel_label(ax, "a")

    ax = axes[1]
    ics = np.linspace(0.001, 0.10, 340)
    taus2 = np.logspace(-3, np.log10(2000), 340)
    G, T = np.meshgrid(ics, taus2)
    Z = (KAPPA * G * BTC.sigma_daily * np.sqrt(T)
         - cost - BTC.funding_bps_per_day * BPS * T) / BPS
    ax.contourf(100 * G, T, Z, levels=[0, 1e9], colors=[NAVY], alpha=0.16)
    ax.contour(100 * G, T, Z, levels=[0], colors=[NAVY], linewidths=1.0)
    Zf = (KAPPA * G * BTC.sigma_daily * np.sqrt(T) - cost) / BPS
    ax.contour(100 * G, T, Zf, levels=[0], colors=[RED], linewidths=0.9,
               linestyles="--")
    ax.axvline(100 * rho_abs, color=K, lw=0.6, ls=":")
    ax.annotate(rf"$\rho_{{\min}}=\dfrac{{2\sqrt{{fc}}}}{{3\sigma}}"
                rf"={100*rho_abs:.2f}\%$",
                xy=(100 * rho_abs, 2.6e-3), xytext=(100 * rho_abs + 1.0, 2.0e-3),
                fontsize=7.5, color=K, ha="left")
    rho_grid = np.linspace(rho_abs, 0.10, 200)
    ax.plot(100 * rho_grid, [best_horizon(BTC, r) for r in rho_grid],
            color=K70, lw=0.8)
    ax.text(6.4, 26, r"$\tau^{*}=\left(\dfrac{3\rho\sigma}{2f}\right)^{2}$",
            fontsize=7.5, color=K70, ha="center")
    ax.set_yscale("log")
    ax.set_xlim(0, 10)
    ax.set_ylim(1e-3, 2000)
    ax.set_xlabel("time-series IC (%)")
    ax.set_ylabel(r"holding period $\tau$ (days)")
    ax.set_title("feasible region")
    light_grid(ax)
    ax.text(4.6, 1.1, "profitable", fontsize=8, color=NAVY, ha="center")
    ax.text(0.75, 320, "fee floor", fontsize=6.5, color=RED, ha="center")
    panel_label(ax, "b")

    caption(fig,
            "Figure 8. Fees and funding bracket the holding period from both "
            "ends. Fees are fixed per round trip and so reward holding longer; "
            "funding accrues with time and so punishes it. For a book that pays "
            "funding the two close a lens, and no correlation below "
            rf"$2\sqrt{{fc}}/(3\sigma)={100*rho_abs:.2f}\%$ is profitable at any "
            "horizon whatsoever. A book symmetric in long and short has zero "
            "expected funding and faces only the dashed fee floor.", y=-0.07)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


# ---------------------------------------------------------------------------

def main() -> None:
    paths.ensure_directories()
    out: dict = {"kappa": KAPPA,
                 "tiers": [{"name": t.name, "maker_bps": t.maker_bps,
                            "taker_bps": t.taker_bps} for t in TIERS]}

    print("=" * 76)
    print("0. Symbolic verification of the fee-and-funding lens")
    print("=" * 76)
    lens = verify_lens_symbolically()
    for k, v in lens.items():
        print(f"  {k:34s} {v}")
    for key in ("rho_floor_matches_closed_form", "tau_star_matches_closed_form",
                "peak_net_matches_closed_form"):
        assert lens[key], key
    out["symbolic"] = lens

    print("\n" + "=" * 76)
    print("1. VIP 0 round-trip costs")
    print("=" * 76)
    print(f"{'instrument':6s} {'style':22s} {'VIP 0':>9s} {'+BNB':>9s} "
          f"{'VIP 9':>9s} {'VIP0/VIP9':>10s}")
    costs = []
    for inst in INSTRUMENTS:
        for label, fn in STYLES:
            c0, cb, c9 = fn(inst, VIP0), fn(inst, VIP0_BNB), fn(inst, VIP9)
            r = f"{c0/c9:.2f}x" if c9 > 0 else "n/a"
            print(f"{inst.short:6s} {label:22s} {c0/BPS:8.3f}b {cb/BPS:8.3f}b "
                  f"{c9/BPS:8.3f}b {r:>10s}")
            costs.append({"instrument": inst.name, "style": label,
                          "vip0_bps": c0 / BPS, "vip0_bnb_bps": cb / BPS,
                          "vip9_bps": c9 / BPS})
    out["costs"] = costs

    print("\n" + "=" * 76)
    print("2. Minimum IC at VIP 0, cross in and cross out (fees only)")
    print("=" * 76)
    print(f"{'horizon':>8s} " + " ".join(f"{i.short:>12s}" for i in INSTRUMENTS))
    floors: dict = {}
    for hname, tau in HORIZONS:
        cells = []
        for inst in INSTRUMENTS:
            f = engine.rho_floor(cost_taker_taker(inst, VIP0),
                                 inst.sigma_daily, tau, KAPPA)
            floors.setdefault(inst.short, {})[hname] = f
            cells.append("unreachable".rjust(12) if f > 1 else f"{100*f:11.2f}%")
        print(f"{hname:>8s} " + " ".join(cells))
    out["floors_vip0_taker"] = floors

    print("\n  where the floor is near or above 100%, both readings and the "
          "move-size test:")
    print(f"  {'instrument':10s} {'horizon':>8s} {'3-sd move':>11s} "
          f"{'cost':>8s} {'total-vol':>10s} {'residual':>9s} {'verdict':>13s}")
    conv = []
    for inst in INSTRUMENTS:
        for hname, tau in HORIZONS:
            c = cost_taker_taker(inst, VIP0)
            ft = engine.rho_floor(c, inst.sigma_daily, tau, KAPPA)
            if ft < 0.35:
                continue
            fr = engine.rho_floor_residual(c, inst.sigma_daily, tau, KAPPA)
            move = engine.kappa_sigma_move(inst.sigma_daily, tau, KAPPA)
            un = engine.is_unreachable(c, inst.sigma_daily, tau, KAPPA)
            print(f"  {inst.short:10s} {hname:>8s} {move/BPS:9.3f}b {c/BPS:6.2f}b "
                  f"{100*ft:9.2f}% {100*fr:8.2f}% "
                  f"{'unreachable' if un else 'reachable':>13s}")
            conv.append({"instrument": inst.name, "horizon": hname,
                         "kappa_sd_move_bps": move / BPS, "cost_bps": c / BPS,
                         "floor_total_vol": ft, "floor_residual_vol": fr,
                         "unreachable": bool(un)})
    out["conventions"] = conv

    print("\n" + "=" * 76)
    print("3. Shortest viable holding period at VIP 0, BTC (fees only)")
    print("=" * 76)
    print(f"{'IC':>6s} " + " ".join(f"{lab:>21s}" for lab, _ in STYLES))
    rows = []
    for ic in (0.005, 0.01, 0.02, 0.03, 0.05, 0.10, 0.20):
        cells, rec = [], {"ic": ic}
        for label, fn in STYLES:
            c = fn(BTC, VIP0)
            t = min_horizon(BTC, c, ic) if c > 0 else None
            cells.append(("never" if t is None else fmt_tau(t)).rjust(21))
            rec[label] = t
        print(f"{100*ic:5.1f}% " + " ".join(cells))
        rows.append(rec)
    out["min_horizon_btc_vip0"] = rows
    mm0, mm9 = cost_maker_maker(BTC, VIP0), cost_maker_maker(BTC, VIP9)
    print(f"  quoting both sides is the cheapest of the three at VIP 0 "
          f"({mm0/BPS:+.2f} bps), but it is")
    print(f"  positive, so spread capture does not self-finance and still needs "
          "alpha. At VIP 9")
    print(f"  the same trade is {mm9/BPS:+.3f} bps: paid to quote, which is what "
          "makes it a business.")

    print("\n" + "=" * 76)
    print("4. The horizon-free kappa=3 IC floor once funding is paid")
    print("=" * 76)
    print(f"{'instr':6s} {'tier':22s} {'fee':>8s} {'funding':>9s} "
          f"{'rho_min':>9s} {'best tau':>10s}")
    abs_rows = []
    for inst in INSTRUMENTS:
        for tier in TIERS:
            c = cost_taker_taker(inst, tier)
            r = rho_floor_with_funding(inst, c)
            print(f"{inst.short:6s} {tier.name:22s} {c/BPS:7.2f}b "
                  f"{inst.funding_bps_per_day:6.1f}b/d {100*r:8.2f}% "
                  f"{fmt_tau(best_horizon(inst, r)):>10s}")
            abs_rows.append({"instrument": inst.name, "tier": tier.name,
                             "cost_bps": c / BPS,
                             "funding_bps_per_day": inst.funding_bps_per_day,
                             "rho_min_abs": r,
                             "tau_star_days": best_horizon(inst, r)})
    out["absolute_floor"] = abs_rows

    print("\n  sensitivity to the funding assumption, BTC at VIP 0:")
    print(f"  {'funding':>11s} {'rho_min':>9s} {'best tau':>10s}")
    sens = []
    c0 = cost_taker_taker(BTC, VIP0)
    # 3.0 is the 2026 median absolute rate, 17.8 the 2026 mean.
    for f_bps in (0.0, 1.0, 3.0, 10.0, 17.8, 30.0):
        if f_bps == 0:
            print(f"  {f_bps:8.1f}b/d {'--':>9s} {'unbounded':>10s}")
            sens.append({"funding_bps_per_day": 0.0, "rho_min_abs": None,
                         "note": "symmetric book pays no expected funding"})
            continue
        r = rho_floor_with_funding(BTC, c0, funding_bps=f_bps)
        print(f"  {f_bps:8.1f}b/d {100*r:8.2f}% "
              f"{fmt_tau(best_horizon(BTC, r, funding_bps=f_bps)):>10s}")
        sens.append({"funding_bps_per_day": f_bps, "rho_min_abs": r,
                     "tau_star_days": best_horizon(BTC, r, funding_bps=f_bps)})
    out["funding_sensitivity_btc_vip0"] = sens

    print("\n" + "=" * 76)
    print("5. How much of the gross edge survives at VIP 0, BTC")
    print("=" * 76)
    print(f"{'horizon':>8s} {'IC':>7s} {'floor':>8s} {'multiple':>9s} "
          f"{'kept':>8s} {'net IR, 100 sym':>16s}")
    cap = []
    for hname, tau in (("1 h", 3600 * SEC), ("4 h", 4 * 3600 * SEC),
                       ("1 day", 1.0), ("3 days", 3.0)):
        floor = engine.rho_floor(c0, BTC.sigma_daily, tau, KAPPA)
        for ic in (0.03, 0.05, 0.10):
            m = ic / floor
            beta = engine.beta_source(ic, BTC.sigma_daily, tau)
            sr = engine.net_sharpe_per_period(beta, c0, BTC.sigma_daily, tau)
            ir = sr * np.sqrt(365.0 / tau * 100)
            print(f"{hname:>8s} {100*ic:6.1f}% {100*floor:7.2f}% {m:8.2f}x "
                  f"{100*engine.gross_capture(m, KAPPA):7.2f}% {ir:16.2f}")
            cap.append({"horizon": hname, "ic": ic, "floor": floor,
                        "multiple": m,
                        "gross_kept": engine.gross_capture(m, KAPPA),
                        "net_ir_100_symbols": ir})
        print()
    out["capture_vip0"] = cap
    print("  365 days a year, 100 symbols assumed independent. Read the breadth")
    print("  counterexample in README.md before believing that last column.")

    fig_retail_wedge(paths.FIGURES_DIR / "fig6_vip0_wedge.png")
    fig_making(paths.FIGURES_DIR / "fig7_vip0_making.png")
    fig_lens(paths.FIGURES_DIR / "fig8_vip0_lens.png")

    print("\n" + "=" * 76)
    print("6. Venue inputs are assumptions, not facts")
    print("=" * 76)
    print("  Fee tiers, tick sizes, spreads and funding above are order-of-")
    print("  magnitude defaults for Binance USDT-margined perps in a normal")
    print("  regime. Three of them move in ways that matter:")
    print("   - fee schedules are revised periodically, and the BNB discount and")
    print("     any market-maker programme terms are separate from the VIP table;")
    print("   - funding settles on a venue- and symbol-dependent interval,")
    print("     predominantly 4 or 8 hours on Binance with one-hour states also")
    print("     observed, and the interval itself can change, so the")
    print("     bps/day figure used for the carry term is a smoothed stand-in;")
    print("   - spreads and dispersion are regime-dependent and widen exactly")
    print("     when the floor falls, which is counterexample C.")
    print("  Replace all of them with the venue's current published schedule")
    print("  before acting on any number in this file.")

    path = paths.RESULTS_DIR / "vip0_fee_analysis.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {paths.rel(path)}")


if __name__ == "__main__":
    main()
