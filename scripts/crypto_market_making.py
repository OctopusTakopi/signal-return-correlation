"""The correlation floor applied to crypto perpetuals, taking vs making.

The thread's floor, rho >= c/(3 sigma sqrt(tau)), prices the cost of *crossing*
the spread. A market maker does not cross it -- they post and get paid it. The
two roles therefore face different thresholds, and the difference is large:

  TAKING   c = full round-trip crossing cost (taker fee x2 + spread paid)
           This is the thread's floor unchanged.

  MAKING   The decision an alpha feeds is not "trade or not" but where to place
           a quote. A maker is already quoting, so there is no crossing cost to
           clear. The binding constraint is *granularity*: a quote cannot be
           skewed by less than one tick, so the forecast must be able to move the
           fair value by at least a tick, which gives the same algebra with c
           replaced by the tick size.

           This formalisation is not the thread's, and it is a necessary
           condition only -- see the caveats printed at the end. Queue position
           and adverse selection, not this bound, decide whether a crypto making
           book actually makes money.

Exchange parameters are inputs, not facts: fee schedules and tick sizes change,
and the numbers below are defaults to be replaced with venue figures.

Output: results/crypto_market_making.json
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

BPS = 1e-4
KAPPA = 3.0
SEC = 1.0 / 86_400.0  # one second, in days

# Horizons a crypto book actually forecasts over.
HORIZONS = [
    ("100 ms", 0.1 * SEC),
    ("1 s", 1 * SEC),
    ("10 s", 10 * SEC),
    ("1 min", 60 * SEC),
    ("5 min", 300 * SEC),
    ("1 h", 3600 * SEC),
    ("8 h", 8 * 3600 * SEC),
    ("1 day", 1.0),
]


@dataclass(frozen=True)
class Instrument:
    name: str
    sigma_daily: float   # realised vol of the log price, per day
    tick_bps: float      # one tick, in bps of price
    spread_bps: float    # typical quoted spread, in bps
    note: str


# Calibrated on 2026 Binance data by scripts/calibrate_from_data.py.
# 2026 only: the 2020-21 volatility and correlation regime no longer
# describes this market. See results/calibration.json, README section 15.
# Volatility and tick size are MEASURED. Spread is not measurable from klines
# (see section 3 of the calibration script) and remains an assumption, floored
# at one tick for the majors where the book is known to sit one tick wide.
INSTRUMENTS = [
    Instrument("BTCUSDT perp", 0.0250, 0.0151, 0.0151,
               "2026: 2.50%/day, tick $0.10 at $66k = 0.0151 bps"),
    Instrument("ETHUSDT perp", 0.0336, 0.0517, 0.0517,
               "2026: 3.36%/day, tick $0.01 at $1.93k = 0.0517 bps"),
    Instrument("mid-cap alt perp", 0.0519, 1.28, 3.0,
               "2026 median vol and SOL-like tick; spread still an assumption"),
]

# Fee tiers, in bps per side. Negative maker = rebate.
FEE_TIERS = [
    ("VIP 0 (retail)", 2.0, 5.0),
    ("VIP 4", 1.2, 3.0),
    ("VIP 9", 0.0, 1.7),
    ("market-maker program", -0.5, 1.7),
]

# What a genuinely good short-horizon alpha looks like. Anything above the top of
# this range in a backtest is a bug or an overfit, not an alpha.
IC_REALISTIC = 0.03
IC_EXCELLENT = 0.10
IC_WORLD_CLASS = 0.20


def taker_floor(inst: Instrument, tau: float, taker_bps: float,
                cross_spread: bool = True) -> float:
    """Thread's floor for aggressing in and out.

    Round trip = two taker fees, plus the half-spread paid on each side when
    crossing a quoted spread rather than trading at the touch of a 1-tick book.
    """
    cost = 2 * taker_bps * BPS
    if cross_spread:
        cost += inst.spread_bps * BPS  # two half-spreads
    return engine.rho_floor(cost, inst.sigma_daily, tau, KAPPA)


def maker_floor(inst: Instrument, tau: float) -> float:
    """Granularity floor for a quote-skewing alpha: one tick."""
    return engine.rho_floor(inst.tick_bps * BPS, inst.sigma_daily, tau, KAPPA)


def maker_taker_floor(inst: Instrument, tau: float, maker_bps: float,
                      taker_bps: float) -> float:
    """Post to get in, cross to get out -- the common real-world compromise."""
    cost = (maker_bps + taker_bps) * BPS + 0.5 * inst.spread_bps * BPS
    return engine.rho_floor(max(cost, 0.0), inst.sigma_daily, tau, KAPPA)


def min_horizon_for_ic(inst: Instrument, cost_bps: float, ic: float) -> float:
    """Shortest horizon at which an alpha of correlation `ic` clears the floor.

    rho >= c/(kappa sigma sqrt(tau))  =>  sqrt(tau) >= c/(kappa sigma rho).
    """
    root = (cost_bps * BPS) / (KAPPA * inst.sigma_daily * ic)
    return root**2


def observations_for_ic(ic: float, tau: float, t_stat: float = 3.0) -> dict:
    n = engine.obs_for_tstat(ic, t_stat)
    per_day = 1.0 / tau
    return {"n": n, "days_one_symbol": n / per_day,
            "hours_one_symbol": 24 * n / per_day,
            "symbol_days": n / per_day}


def main() -> None:
    paths.ensure_directories()
    out: dict = {"kappa": KAPPA, "instruments": [], "fee_tiers": FEE_TIERS}

    print("=" * 78)
    print("1. TAKING: the thread's floor, crypto perps, aggressive both sides")
    print("=" * 78)
    print("minimum time-series IC needed to clear the round-trip cost.")
    print()
    print("`sigma` here is MEASURED TOTAL volatility, so Var(y) = sigma^2 tau and")
    print("rho = beta/(sigma sqrt(tau)) is exact -- the floor below is exact, not")
    print("approximate, and a value above 100% means no correlation satisfies the")
    print("kappa=3 relevance criterion. The reason is parameterisation-free: a 3-sd")
    print("move is smaller than the round trip. That is NOT the same as saying")
    print("trading loses; the band rule keeps a positive but very rare tail edge.")
    print("Section 1b shows both readings.\n")
    for tier_name, maker_bps, taker_bps in FEE_TIERS:
        print(f"--- {tier_name}: maker {maker_bps:+.1f} bps, "
              f"taker {taker_bps:.1f} bps ---")
        hdr = f"{'horizon':>8s} " + " ".join(
            f"{i.name.split()[0]:>16s}" for i in INSTRUMENTS)
        print(hdr)
        for hname, tau in HORIZONS:
            cells = []
            for inst in INSTRUMENTS:
                f = taker_floor(inst, tau, taker_bps)
                cells.append("unreachable".rjust(16) if f > 1.0
                             else f"{100*f:14.2f}%")
            print(f"{hname:>8s} " + " ".join(cells))
        print()

    print("=" * 78)
    print("1b. Why the unreachable cells are unreachable, and the other reading")
    print("=" * 78)
    print("A 3-sd move of size 3*sigma*sqrt(tau) is the reference move the")
    print("kappa=3 criterion is stated against, NOT a maximum: a Gaussian signal")
    print("exceeds any finite band eventually, so a rare tail trade survives at")
    print("any correlation. If the round trip exceeds the 3-sd move, what fails")
    print("is the criterion, not profitability. Where it does not, the")
    print("two conventions for sigma disagree: total vol gives the exact floor")
    print("c/(3 sigma sqrt(tau)); the thread's residual-vol model gives the exact")
    print("floor a/sqrt(1+a^2), always below 100%, but only by letting total vol")
    print("inflate to sigma/sqrt(1-rho^2).\n")
    print(f"{'instrument':14s} {'horizon':>8s} {'3-sd move':>11s} "
          f"{'round trip':>11s} {'total-vol':>10s} {'residual':>9s} {'verdict':>13s}")
    conv = []
    for inst in INSTRUMENTS:
        for hname, tau in (("1 s", 1 * SEC), ("10 s", 10 * SEC),
                           ("1 min", 60 * SEC), ("5 min", 300 * SEC)):
            c = 2 * 1.7 * BPS + inst.spread_bps * BPS   # top tier
            move = engine.kappa_sigma_move(inst.sigma_daily, tau, KAPPA)
            ft = engine.rho_floor(c, inst.sigma_daily, tau, KAPPA)
            fr = engine.rho_floor_residual(c, inst.sigma_daily, tau, KAPPA)
            un = engine.is_unreachable(c, inst.sigma_daily, tau, KAPPA)
            print(f"{inst.name.split()[0]:14s} {hname:>8s} {move/BPS:9.3f}b "
                  f"{c/BPS:9.3f}b {100*ft:9.2f}% {100*fr:8.2f}% "
                  f"{'unreachable' if un else 'reachable':>13s}")
            conv.append({"instrument": inst.name, "horizon": hname,
                         "kappa_sd_move_bps": move / BPS, "cost_bps": c / BPS,
                         "floor_total_vol": ft, "floor_residual_vol": fr,
                         "unreachable": bool(un)})
    out["conventions"] = conv
    print("\n  'unreachable' is a statement about the price and about the kappa=3")
    print("  criterion, not about the forecast: the whole 3-sd move is smaller than")
    print("  the fee, so no correlation satisfies the criterion. Expected net payoff")
    print("  under the band rule stays positive at any rho, just concentrated in a")
    print("  rare tail. Below that threshold the total-vol floor is the one to")
    print("  quote, since sigma was measured.\n")

    print("=" * 78)
    print("2. MAKING: quote-skew granularity floor (one tick)")
    print("=" * 78)
    print("minimum IC for the forecast to move a quote at all\n")
    hdr = f"{'horizon':>8s} " + " ".join(
        f"{i.name.split()[0]:>16s}" for i in INSTRUMENTS)
    print(hdr)
    for hname, tau in HORIZONS:
        cells = [f"{100*maker_floor(i, tau):14.3f}%" for i in INSTRUMENTS]
        print(f"{hname:>8s} " + " ".join(cells))

    print("\n--- and the middle case: post to enter, cross to exit (VIP 9) ---")
    print(hdr)
    for hname, tau in HORIZONS:
        cells = []
        for inst in INSTRUMENTS:
            f = maker_taker_floor(inst, tau, 0.0, 1.7)
            cells.append("unreachable".rjust(16) if f > 1.0 else f"{100*f:14.2f}%")
        print(f"{hname:>8s} " + " ".join(cells))

    print("\n" + "=" * 78)
    print("3. The binding constraint is horizon, not IC")
    print("=" * 78)
    print("shortest horizon at which a given IC can pay a round trip\n")
    rows = []
    for inst in INSTRUMENTS:
        print(f"--- {inst.name} (sigma = {100*inst.sigma_daily:.1f}%/day) ---")
        print(f"{'style':28s} {'cost':>8s} "
              f"{'IC 3%':>12s} {'IC 10%':>12s} {'IC 20%':>12s}")
        styles = [
            ("taker/taker, VIP 0", 2 * 5.0 + inst.spread_bps),
            ("taker/taker, VIP 9", 2 * 1.7 + inst.spread_bps),
            ("maker in, taker out, VIP 9", 1.7 + 0.5 * inst.spread_bps),
            ("maker/maker, MM programme", max(2 * -0.5, 0.0) + 0.0),
            ("quote skew (one tick)", inst.tick_bps),
        ]
        for label, cost_bps in styles:
            cells = []
            for ic in (IC_REALISTIC, IC_EXCELLENT, IC_WORLD_CLASS):
                if cost_bps <= 0:
                    cells.append("any".rjust(12))
                    continue
                tau = min_horizon_for_ic(inst, cost_bps, ic)
                cells.append(_fmt_tau(tau).rjust(12))
            print(f"{label:28s} {cost_bps:6.2f}bp " + " ".join(cells))
            rows.append({"instrument": inst.name, "style": label,
                         "cost_bps": cost_bps,
                         "min_horizon_days": {
                             f"ic_{int(100*ic)}pct":
                                 (min_horizon_for_ic(inst, cost_bps, ic)
                                  if cost_bps > 0 else 0.0)
                             for ic in (IC_REALISTIC, IC_EXCELLENT,
                                        IC_WORLD_CLASS)}})
        print()

    print("=" * 78)
    print("4. Measurability: one genuine advantage of this market")
    print("=" * 78)
    print("data needed to establish an IC at t = 3, one symbol\n")
    print(f"{'IC':>7s} {'horizon':>8s} {'observations':>14s} {'wall clock':>16s}")
    est = []
    for ic in (0.004, 0.01, 0.03, 0.10):
        for hname, tau in (("1 s", SEC), ("1 min", 60 * SEC), ("1 h", 3600 * SEC)):
            e = observations_for_ic(ic, tau)
            print(f"{100*ic:6.1f}% {hname:>8s} {e['n']:14,.0f} "
                  f"{_fmt_tau(e['days_one_symbol']):>16s}")
            est.append({"ic": ic, "horizon": hname, **e})
        print()

    for inst in INSTRUMENTS:
        out["instruments"].append({
            "name": inst.name, "sigma_daily": inst.sigma_daily,
            "tick_bps": inst.tick_bps, "spread_bps": inst.spread_bps,
            "note": inst.note,
            "taker_floor": {h: taker_floor(inst, t, 1.7)
                            for h, t in HORIZONS},
            "maker_skew_floor": {h: maker_floor(inst, t) for h, t in HORIZONS},
            "maker_taker_floor": {h: maker_taker_floor(inst, t, 0.0, 1.7)
                                  for h, t in HORIZONS},
        })
    out["min_horizons"] = rows
    out["estimation"] = est
    out["caveats"] = CAVEATS

    print("=" * 78)
    print("5. What these numbers do NOT establish")
    print("=" * 78)
    for i, c in enumerate(CAVEATS, 1):
        print(f"{i}. {c}")

    path = paths.RESULTS_DIR / "crypto_market_making.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {paths.rel(path)}")


CAVEATS = [
    "Adverse selection, not the tick, is a maker's real cost. The bound in "
    "section 2 asks only whether the forecast can move a quote; it does not "
    "ask whether the flow behind a fill is informed. E[move | filled] "
    "is the number that kills making books, and it is far larger than sigma "
    "times sqrt(tau) for toxic flow.",
    "Queue position is not in the model at all. A 0.5% IC that says 'lift the "
    "bid' is worth nothing 200 lots deep in the queue, where the fill never "
    "arrives on the intended side.",
    "On BTC and ETH perps the quoted spread is smaller than the fee. One tick "
    "on BTCUSDT is about 0.0151 bps while the VIP 0 maker fee is 2 bps per "
    "side -- about 132x the tick on one side and 264x on a two-sided round "
    "trip. Passive making on majors is "
    "not a strategy until the maker fee is at or below zero, which is a "
    "business-development problem, not a research problem.",
    "Latency is an IC multiplier. A signal with a 200 ms half-life measured "
    "offline has an effective IC of nearly zero when the round-trip reaction is "
    "500 ms. Measure IC against the return starting from the first moment "
    "action was possible.",
    "Perp funding is a real carry on inventory: predominantly 4-hourly on the "
    "2026 record, with 8-hour and occasional 1-hour states, and at extremes it "
    "annualises into three figures. A making book that is systematically long "
    "in a positive-funding regime is paying for the privilege.",
    "Two readings of sigma exist and they diverge as the floor approaches 1. "
    "Total (measured) volatility makes rho = beta/(sigma sqrt(tau)) exact and "
    "lets the floor exceed 100%, which means the kappa=3 criterion cannot be met "
    "at any correlation, though a rare tail edge survives. The thread's literal "
    "model treats sigma as residual, giving an exact floor a/sqrt(1+a^2) that is "
    "always under 100% -- but only because it lets total volatility inflate to "
    "sigma/sqrt(1-rho^2), contradicting the figure that was measured. Section 1b "
    "reports both.",
    "Perp funding intervals are venue- and symbol-dependent (Binance uses 4 and "
    "8 hour settlement depending on the contract, and can change the interval), "
    "and fee schedules are revised periodically. Every fee, tick, spread and "
    "funding figure in this file is an assumption to be replaced with the "
    "venue's current published schedule.",
    "Crypto volatility is regime-switching, so a single sigma is a fiction. The "
    "floor is proportional to 1/sigma, which means it *falls* when volatility "
    "spikes -- and that is exactly when spreads widen and adverse selection "
    "worsens. Counterexample C in README.md is this failure mode.",
    "These are necessary conditions on a single signal in isolation. A "
    "portfolio of weak, decorrelated signals clears the bar that none of them "
    "clears alone -- that is the breadth argument, and it is the whole reason "
    "anyone runs dozens of features.",
]


def _fmt_tau(tau_days: float) -> str:
    if tau_days <= 0:
        return "any"
    secs = tau_days * 86_400
    if secs < 1:
        return f"{secs*1000:.0f} ms"
    if secs < 90:
        return f"{secs:.1f} s"
    if secs < 5400:
        return f"{secs/60:.1f} min"
    if tau_days < 2:
        return f"{secs/3600:.1f} h"
    if tau_days < 400:
        return f"{tau_days:.1f} d"
    return f"{tau_days/365:.1f} yr"


if __name__ == "__main__":
    main()
