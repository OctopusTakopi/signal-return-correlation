"""Monte Carlo: simulated price action, alphas built for it, and the floor.

Three jobs.

1. Confirm the closed forms in engine.py against brute-force simulation, so
   the report's numbers rest on something other than the algebra alone.
2. Walk the floor empirically: sweep the multiple of the floor and measure
   realised trade frequency, gross capture and net Sharpe.
3. Test the framing. The thread states that the floor must be exceeded "to be
   able to trade profitably". Under the thread's own Gaussian model that is
   too strong: with a no-trade band, *any* positive correlation has positive
   expected net P&L, since a larger signal always eventually arrives. Measure
   what actually dies below the floor.

Output: results/simulation.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import paths  # noqa: E402

BPS = 1e-4
KAPPA = 3.0
SEED = 20240627  # the thread's publication date
N_PATHS = 4_000_000

EQUITY = engine.Market(0.03, 1.0, 5 * BPS, "US equity, 1-day horizon")
FX = engine.Market(0.003, 1.0 / 1440.0, 0.2 * BPS, "FX, 1-minute horizon")


def validate_closed_forms(market: engine.Market, rng) -> list[dict]:
    """Simulation vs closed form, at several multiples of the floor."""
    floor = market.floor(KAPPA)
    out = []
    for mult in (1.0, 1.5, 2.0, 3.0):
        rho = mult * floor
        x, y, beta = engine.simulate_returns_and_alpha(market, rho, N_PATHS, rng)
        bt = engine.banded_backtest(x, y, beta, market.cost)

        realised_rho = float(np.corrcoef(x, y)[0, 1])
        se_rho = (1 - realised_rho**2) / np.sqrt(N_PATHS - 1)
        k = market.cost / beta
        theory_net = engine.net_pnl_per_period(beta, market.cost)
        theory_freq = engine.trade_frequency(k)
        theory_sr = engine.net_sharpe_per_period(
            beta, market.cost, market.sigma_daily, market.tau_days
        )
        se_net = bt["net_sd"] / np.sqrt(N_PATHS)

        out.append(
            {
                "market": market.label,
                "multiple": mult,
                "rho_target": rho,
                "rho_realised": realised_rho,
                "rho_se": se_rho,
                "rho_z": (realised_rho - rho) / se_rho,
                "beta": beta,
                "band_k": k,
                "freq_theory": theory_freq,
                "freq_sim": bt["trade_frequency"],
                "net_theory_bps": theory_net / BPS,
                "net_sim_bps": bt["net_mean"] / BPS,
                "net_sim_se_bps": se_net / BPS,
                "net_z": (bt["net_mean"] - theory_net) / se_net,
                "sharpe_theory": theory_sr,
                "sharpe_sim": bt["sharpe_per_period"],
                "gross_sim_bps": bt["gross_mean"] / BPS,
                "cost_sim_bps": bt["cost_mean"] / BPS,
            }
        )
    return out


def sweep_floor(market: engine.Market) -> list[dict]:
    """Closed-form sweep across and *below* the floor."""
    floor = market.floor(KAPPA)
    rows = []
    for mult in (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0):
        rho = mult * floor
        beta = engine.beta_source(rho, market.sigma_daily, market.tau_days)
        k = market.cost / beta
        net = engine.net_pnl_per_period(beta, market.cost)
        sr = engine.net_sharpe_per_period(
            beta, market.cost, market.sigma_daily, market.tau_days
        )
        periods_per_year = 252.0 / market.tau_days
        rows.append(
            {
                "multiple": mult,
                "rho": rho,
                "rho_pct": 100 * rho,
                "band_k": k,
                "trade_frequency": engine.trade_frequency(k),
                "net_bps_per_period": net / BPS,
                "gross_bps_per_period": engine.gross_pnl_per_period(beta) / BPS,
                "gross_capture": engine.gross_capture(mult, KAPPA),
                "net_sharpe_per_period": sr,
                "net_ir_annual_1_asset": sr * np.sqrt(periods_per_year),
                "net_ir_annual_500_assets": sr * np.sqrt(periods_per_year * 500),
                "trades_per_year_1_asset": engine.trade_frequency(k) * periods_per_year,
            }
        )
    return rows


def persistence_extension(market: engine.Market, rng) -> list[dict]:
    """What the iid-signal assumption costs the floor.

    The thread's model has no time index: x is a fresh draw each period, so
    every trade is a fresh round trip. Real alphas decay slowly. Give the
    signal an AR(1) autocorrelation phi, hold the position while the signal
    stays in the same direction, and charge c/2 per unit of position *change*.
    Gross is untouched (the marginal law of x is still N(0,1)); only turnover
    falls. So the true floor is lower than the thread's by a factor tied to
    holding time, and the faster the signal decays, the closer the thread's
    number gets to right.
    """
    floor = market.floor(KAPPA)
    n = 2_000_000
    rows = []
    for phi in (0.0, 0.5, 0.8, 0.9, 0.95, 0.99):
        for mult in (0.5, 1.0, 2.0):
            rho = mult * floor
            beta = engine.beta_exact(rho, market.sigma_daily, market.tau_days)
            # AR(1) with unit marginal variance.
            eta = rng.standard_normal(n)
            x = np.empty(n)
            x[0] = eta[0]
            root = np.sqrt(1.0 - phi**2)
            for i in range(1, n):
                x[i] = phi * x[i - 1] + root * eta[i]
            eps = rng.standard_normal(n)
            y = beta * x + market.horizon_vol * eps

            k = market.cost / beta
            h = np.where(np.abs(x) > k, np.sign(x), 0.0)
            # phi = 0 has a closed form under this cost convention; carry it so
            # the two conventions can be reconciled instead of just differing.
            f_theory = engine.trade_frequency(k)
            gross_theory = engine.gross_pnl_banded(beta, k)
            cost_theory_turnover = 0.5 * market.cost * engine.turnover_iid(f_theory)
            cost_theory_roundtrip = market.cost * f_theory
            turnover = np.abs(np.diff(h, prepend=0.0))
            gross = h * y
            cost = 0.5 * market.cost * turnover
            net = gross - cost
            rows.append(
                {
                    "market": market.label,
                    "phi": phi,
                    "multiple": mult,
                    "rho_pct": 100 * rho,
                    "realised_signal_autocorr": float(
                        np.corrcoef(x[:-1], x[1:])[0, 1]
                    ),
                    "trade_frequency": float((h != 0).mean()),
                    "turnover_per_period": float(turnover.mean()),
                    "mean_holding_periods": float(
                        (h != 0).mean() / max(turnover.mean() / 2.0, 1e-12)
                    ),
                    "gross_bps": float(gross.mean() / BPS),
                    "cost_bps": float(cost.mean() / BPS),
                    "net_bps": float(net.mean() / BPS),
                    "net_se_bps": float(net.std(ddof=1) / np.sqrt(n) / BPS),
                    "net_sharpe_per_period": float(net.mean() / net.std(ddof=1)),
                    "gross_theory_bps": gross_theory / BPS,
                    "turnover_theory_iid": engine.turnover_iid(f_theory),
                    "cost_theory_turnover_bps": cost_theory_turnover / BPS,
                    "cost_theory_roundtrip_bps": cost_theory_roundtrip / BPS,
                    "net_theory_turnover_bps": engine.net_pnl_turnover_iid(
                        beta, market.cost
                    )
                    / BPS,
                    "net_theory_roundtrip_bps": engine.net_pnl_per_period(
                        beta, market.cost
                    )
                    / BPS,
                }
            )
    return rows


def framing_test(market: engine.Market) -> dict:
    """Whether the floor is a profitability threshold or a relevance threshold."""
    floor = market.floor(KAPPA)
    probe = []
    for mult in (0.01, 0.1, 0.25, 0.5, 1.0):
        rho = mult * floor
        beta = engine.beta_source(rho, market.sigma_daily, market.tau_days)
        net = engine.net_pnl_per_period(beta, market.cost)
        k = market.cost / beta
        probe.append(
            {
                "multiple": mult,
                "rho_pct": 100 * rho,
                "band_k": k,
                "net_bps_per_period": net / BPS,
                "net_is_positive": bool(net > 0),
                "trade_frequency": engine.trade_frequency(k),
                "trades_per_year": engine.trade_frequency(k) * 252.0 / market.tau_days,
                "gross_capture": engine.gross_capture(mult, KAPPA),
            }
        )
    return {
        "claim": "you need to exceed the floor to be able to trade profitably",
        "verdict": "too strong under the thread's own model",
        "reason": (
            "net = beta E[(|x|-k)^+] with k = c/beta is strictly positive for "
            "every beta > 0, because a Gaussian signal always has some "
            "probability of exceeding any finite band. The floor is not where "
            "profit turns negative; it is where profit stops mattering."
        ),
        "what_dies_instead": (
            "at the floor the rule trades 0.27% of periods and keeps 0.10% of "
            "the costless gross P&L; with 500 independent names the annualised "
            "net IR is 0.03. That is economically dead, not loss-making."
        ),
        "probe": probe,
    }


def main() -> None:
    paths.ensure_directories()
    rng = np.random.default_rng(SEED)

    print(f"=== 1. Closed forms vs simulation (N = {N_PATHS:,} per cell) ===")
    validation = []
    for market in (EQUITY, FX):
        validation += validate_closed_forms(market, rng)
    hdr = (
        f"{'market':14s} {'mult':>5s} {'rho z':>7s} {'freq thy':>9s} {'freq sim':>9s} "
        f"{'net thy':>9s} {'net sim':>9s} {'net z':>7s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in validation:
        print(
            f"{r['market'][:14]:14s} {r['multiple']:4.2f}x {r['rho_z']:7.2f} "
            f"{100*r['freq_theory']:8.3f}% {100*r['freq_sim']:8.3f}% "
            f"{r['net_theory_bps']:8.4f} {r['net_sim_bps']:8.4f} {r['net_z']:7.2f}"
        )
    worst_z = max(abs(r["net_z"]) for r in validation)
    worst_rho_z = max(abs(r["rho_z"]) for r in validation)
    print(
        f"\nlargest |z| against closed form: net {worst_z:.2f}, rho {worst_rho_z:.2f} "
        "(|z| < 4 is noise at this sample size)"
    )

    print("\n=== 2. Sweeping the floor (US equity) ===")
    sweep = sweep_floor(EQUITY)
    hdr = (
        f"{'mult':>5s} {'rho':>7s} {'k':>6s} {'freq':>8s} {'trades/yr':>10s} "
        f"{'net bps':>9s} {'capture':>8s} {'net IR x500':>12s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sweep:
        print(
            f"{r['multiple']:4.2f}x {r['rho_pct']:6.3f}% {r['band_k']:6.2f} "
            f"{100*r['trade_frequency']:7.3f}% {r['trades_per_year_1_asset']:10.1f} "
            f"{r['net_bps_per_period']:9.4f} {100*r['gross_capture']:7.2f}% "
            f"{r['net_ir_annual_500_assets']:12.2f}"
        )

    print("\n=== 3. What the iid-signal assumption costs (US equity) ===")
    print("cost convention here: c/2 per unit of position change, not c per")
    print("period held. At phi=0 that is already c*f*(1-f/2), so it undercuts")
    print("section 2's c*f; the phi=0 rows below reconcile the two.")
    pers = persistence_extension(EQUITY, rng)
    hdr = (
        f"{'phi':>5s} {'mult':>5s} {'hold':>7s} {'turnover':>9s} {'gross':>8s} "
        f"{'cost':>8s} {'net bps':>9s} {'+/-':>7s} {'net thy':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in pers:
        thy = r["net_theory_turnover_bps"] if r["phi"] == 0.0 else float("nan")
        thy_s = f"{thy:9.4f}" if r["phi"] == 0.0 else " " * 9
        print(
            f"{r['phi']:5.2f} {r['multiple']:4.2f}x {r['mean_holding_periods']:7.2f} "
            f"{r['turnover_per_period']:9.4f} {r['gross_bps']:8.4f} "
            f"{r['cost_bps']:8.4f} {r['net_bps']:9.4f} {r['net_se_bps']:7.4f}{thy_s}"
        )
    # Headline: the gain persistence buys at 2x the floor.
    at2 = {r["phi"]: r for r in pers if r["multiple"] == 2.0}
    lift = at2[0.99]["net_bps"] / at2[0.0]["net_bps"]
    print(
        f"\nat 2x the floor, going from an iid signal to phi=0.99 multiplies net "
        f"P&L by {lift:.2f}x ({at2[0.0]['net_bps']:.3f} -> "
        f"{at2[0.99]['net_bps']:.3f} bps/period) purely by cutting turnover "
        f"{at2[0.0]['turnover_per_period']:.4f} -> "
        f"{at2[0.99]['turnover_per_period']:.4f}. The floor therefore depends on "
        "how fast the signal decays, which the thread's model has no room for."
    )

    print("\n=== 4. Profitability threshold, or relevance threshold ===")
    framing = framing_test(EQUITY)
    print(f"claim   : {framing['claim']}")
    print(f"verdict : {framing['verdict']}")
    hdr = f"{'mult':>7s} {'rho':>8s} {'k':>7s} {'net bps':>12s} {'trades/yr':>10s}"
    print(hdr)
    for r in framing["probe"]:
        print(
            f"{r['multiple']:6.2f}x {r['rho_pct']:7.4f}% {r['band_k']:7.2f} "
            f"{r['net_bps_per_period']:12.3e} {r['trades_per_year']:10.3f}"
        )
    print(f"\n{framing['reason']}")
    print(framing["what_dies_instead"])

    out = paths.RESULTS_DIR / "simulation.json"
    out.write_text(
        json.dumps(
            {
                "seed": SEED,
                "n_paths": N_PATHS,
                "kappa": KAPPA,
                "validation": validation,
                "sweep_equity": sweep,
                "persistence": pers,
                "framing": framing,
            },
            indent=2,
            default=float,
        )
        + "\n"
    )
    print(f"\nwrote {paths.rel(out)}")


if __name__ == "__main__":
    main()
