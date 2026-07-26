"""Reproduce every number the thread states, and audit the two rules of thumb.

Output: results/examples.json, results/rules_of_thumb.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import paths  # noqa: E402

BPS = 1e-4
KAPPA = 3.0

# The three cases the thread works through. `stated` is what the thread says.
CASES = [
    {
        "key": "equity",
        "label": "US equity, 1-day horizon",
        "sigma_daily": 0.03,
        "tau_days": 1.0,
        "cost": 5 * BPS,
        "stated": 0.005,
        "tweet": "5/9",
        "note": "3% daily vol, 5 bps to trade, 1-day forecast",
    },
    {
        "key": "equity_hedged",
        "label": "US equity, factor-hedged",
        "sigma_daily": 0.015,
        "tau_days": 1.0,
        "cost": 10 * BPS,
        "stated": 0.02,
        "tweet": "6/9",
        "note": "vol halved by the hedge, cost doubled by the hedge leg",
    },
    {
        "key": "fx_1min",
        "label": "FX, 1-minute horizon",
        "sigma_daily": 0.003,
        "tau_days": 1.0 / 1440.0,
        "cost": 0.2 * BPS,
        "stated": 0.085,
        "tweet": "7/9",
        "note": "0.3% daily vol, 0.2 bps to trade, 1 minute = 1/1440 day",
    },
]


def check_cases() -> list[dict]:
    rows = []
    for case in CASES:
        m = engine.Market(case["sigma_daily"], case["tau_days"], case["cost"], case["label"])
        floor = m.floor(KAPPA)
        beta_at_floor = engine.beta_source(floor, m.sigma_daily, m.tau_days)
        # Round-trip the exact correlation to size the approximation error.
        corr_if_beta_from_source = engine.corr_exact(
            beta_at_floor, m.sigma_daily, m.tau_days
        )
        rows.append(
            {
                **{k: case[k] for k in ("key", "label", "tweet", "note", "stated")},
                "sigma_daily": m.sigma_daily,
                "tau_days": m.tau_days,
                "cost": m.cost,
                "horizon_vol": m.horizon_vol,
                "floor_exact": floor,
                "floor_pct": 100 * floor,
                "stated_pct": 100 * case["stated"],
                "abs_error_pct": 100 * (floor - case["stated"]),
                "rel_error": floor / case["stated"] - 1.0,
                "beta_at_floor": beta_at_floor,
                "beta_at_floor_bps": beta_at_floor / BPS,
                "corr_approx_error": corr_if_beta_from_source - floor,
                "corr_approx_rel_error": corr_if_beta_from_source / floor - 1.0,
                "rounds_to_stated": bool(
                    abs(floor - case["stated"]) < 0.5 * 10 ** np.floor(
                        np.log10(case["stated"])
                    )
                ),
            }
        )
    return rows


def check_rules_of_thumb() -> dict:
    """The thread's two frequency claims, under the band rule |x| > c/beta.

    k = kappa/m, so the frequency is a function of the multiple only.
    """
    multiples = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 4.0, 5.0]
    rows = []
    for mult in multiples:
        k = engine.band_from_multiple(mult, KAPPA)
        # Net Sharpe is scale-free in the same way k is: fix the equity case
        # and the answer transfers, because beta and c both carry the units.
        eq = CASES[0]
        floor = engine.rho_floor(eq["cost"], eq["sigma_daily"], eq["tau_days"], KAPPA)
        beta = engine.beta_source(mult * floor, eq["sigma_daily"], eq["tau_days"])
        sr_period = engine.net_sharpe_per_period(
            beta, eq["cost"], eq["sigma_daily"], eq["tau_days"]
        )
        rows.append(
            {
                "multiple": mult,
                "band_k": k,
                "trade_frequency": engine.trade_frequency(k),
                "trade_frequency_pct": 100 * engine.trade_frequency(k),
                "gross_capture": engine.gross_capture(mult, KAPPA),
                "gross_capture_pct": 100 * engine.gross_capture(mult, KAPPA),
                "net_sharpe_per_period": sr_period,
                "net_ir_annual_1_asset": sr_period * np.sqrt(252.0),
                "net_ir_annual_500_assets": sr_period * np.sqrt(252.0 * 500),
                "gross_ir_annual_500_assets": engine.annualised_ir(
                    mult * floor, 1.0, 500
                ),
            }
        )

    # Invert, to find the multiple each stated frequency actually requires.
    stated = {
        "1.5x -> ~5%": {"stated_multiple": 1.5, "stated_freq": 0.05},
        "2x -> 20-30%": {"stated_multiple": 2.0, "stated_freq_low": 0.20,
                         "stated_freq_high": 0.30},
    }
    f_15 = engine.trade_frequency(engine.band_from_multiple(1.5, KAPPA))
    f_20 = engine.trade_frequency(engine.band_from_multiple(2.0, KAPPA))
    verdict = {
        "rule_1": {
            "claim": "1.5x the floor => a trade in about 5% of periods",
            "model_frequency": f_15,
            "model_frequency_pct": 100 * f_15,
            "stated_pct": 5.0,
            "verdict": "reproduces" if abs(f_15 - 0.05) < 0.01 else "does not reproduce",
        },
        "rule_2": {
            "claim": "2x the floor => a trade in 20-30% of periods",
            "model_frequency": f_20,
            "model_frequency_pct": 100 * f_20,
            "stated_range_pct": [20.0, 30.0],
            "multiple_needed_for_20pct": engine.multiple_from_frequency(0.20, KAPPA),
            "multiple_needed_for_30pct": engine.multiple_from_frequency(0.30, KAPPA),
            "verdict": "reproduces"
            if 0.20 <= f_20 <= 0.30
            else "does not reproduce",
        },
        "table": rows,
        "stated": stated,
    }
    return verdict


def _duration(years: float) -> str:
    if years >= 1.0:
        return f"{years:,.0f} asset-years"
    days = years * 252.0
    if days >= 1.0:
        return f"{days:,.1f} trading days"
    return f"{days * 6.5:,.1f} trading hours"


def check_estimability() -> list[dict]:
    """The quantity of data needed to *see* a correlation this small."""
    rows = []
    for case in CASES:
        m = engine.Market(case["sigma_daily"], case["tau_days"], case["cost"])
        floor = m.floor(KAPPA)
        for mult, name in ((1.0, "at the floor"), (2.0, "2x the floor")):
            rho = mult * floor
            n2 = engine.obs_for_tstat(rho, 2.0)
            n3 = engine.obs_for_tstat(rho, 3.0)
            periods_per_year = 252.0 / case["tau_days"]
            asset_years = n3 / periods_per_year
            rows.append(
                {
                    "key": case["key"],
                    "regime": name,
                    "rho": rho,
                    "rho_pct": 100 * rho,
                    "n_for_t2": n2,
                    "n_for_t3": n3,
                    # One asset-year = 252/tau observations. Reading this as
                    # "assets needed given one year of data" assumes the alphas
                    # are independent across assets -- the same assumption the
                    # fundamental law needs, and the one `effective_breadth`
                    # shows is usually false.
                    "asset_years_for_t3": asset_years,
                    "human_for_t3": _duration(asset_years),
                    "periods_per_year": periods_per_year,
                }
            )
    return rows


def main() -> None:
    paths.ensure_directories()

    cases = check_cases()
    print("=== Thread's worked examples ===")
    hdr = f"{'case':26s} {'stated':>8s} {'exact':>8s} {'rel err':>9s} {'beta@floor':>12s}"
    print(hdr)
    print("-" * len(hdr))
    for r in cases:
        print(
            f"{r['label']:26s} {r['stated_pct']:7.2f}% {r['floor_pct']:7.3f}% "
            f"{100*r['rel_error']:8.1f}% {r['beta_at_floor_bps']:9.3f} bps"
        )
    print(
        "\nlargest Corr approximation error across cases: "
        f"{max(abs(r['corr_approx_rel_error']) for r in cases)*100:.3f}% relative"
    )

    rot = check_rules_of_thumb()
    print("\n=== Rules of thumb (band rule |x| > kappa/m) ===")
    print(f"{'multiple':>9s} {'band k':>7s} {'trade freq':>11s} {'gross capture':>14s}"
          f" {'net IR, 500 names':>18s} {'gross IR, 500':>14s}")
    for r in rot["table"]:
        print(
            f"{r['multiple']:8.2f}x {r['band_k']:7.3f} "
            f"{r['trade_frequency_pct']:10.2f}% {r['gross_capture_pct']:13.2f}%"
            f" {r['net_ir_annual_500_assets']:18.2f} "
            f"{r['gross_ir_annual_500_assets']:14.2f}"
        )
    print(f"\nrule 1 ({rot['rule_1']['claim']}): "
          f"model gives {rot['rule_1']['model_frequency_pct']:.2f}% "
          f"-> {rot['rule_1']['verdict'].upper()}")
    print(f"rule 2 ({rot['rule_2']['claim']}): "
          f"model gives {rot['rule_2']['model_frequency_pct']:.2f}% "
          f"-> {rot['rule_2']['verdict'].upper()}")
    print(f"   20% needs {rot['rule_2']['multiple_needed_for_20pct']:.2f}x, "
          f"30% needs {rot['rule_2']['multiple_needed_for_30pct']:.2f}x the floor")

    est = check_estimability()
    print("\n=== How much data to see it (Pearson rho-hat) ===")
    print(f"{'case':16s} {'regime':13s} {'rho':>7s} {'N for t=3':>13s}   how long, one asset")
    for r in est:
        print(
            f"{r['key']:16s} {r['regime']:13s} {r['rho_pct']:6.3f}% "
            f"{r['n_for_t3']:12,.0f}   {r['human_for_t3']}"
        )

    # Bridge to the fundamental law.
    print("\n=== Fundamental-law bridge ===")
    law = []
    for case in cases:
        rho = case["floor_exact"]
        for breadth in (1, 100, 500):
            ir = engine.annualised_ir(rho, case["tau_days"], breadth)
            law.append(
                {"key": case["key"], "rho": rho, "breadth": breadth, "annual_ir": ir}
            )
            print(
                f"{case['key']:16s} rho={100*rho:6.3f}%  N={breadth:4d}  "
                f"annualised IR = {ir:6.2f}"
            )

    (paths.RESULTS_DIR / "examples.json").write_text(
        json.dumps({"cases": cases, "estimability": est, "fundamental_law": law},
                   indent=2, default=float) + "\n"
    )
    (paths.RESULTS_DIR / "rules_of_thumb.json").write_text(
        json.dumps(rot, indent=2, default=float) + "\n"
    )
    print("\nwrote results/examples.json, results/rules_of_thumb.json")


if __name__ == "__main__":
    main()
