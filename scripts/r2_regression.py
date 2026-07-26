"""Verify the linear-regression interpretation of the IC cost floor.

Output: results/r2_regression.json.
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
SEED = 20_260_726

MARKETS = (
    ("equity", 0.03, 1.0, 5 * BPS),
    ("equity_hedged", 0.015, 1.0, 10 * BPS),
    ("fx_1min", 0.003, 1.0 / 1440.0, 0.2 * BPS),
)


def floor_rows() -> list[dict]:
    rows = []
    for name, sigma, tau, cost in MARKETS:
        ic_total = engine.rho_floor(cost, sigma, tau, KAPPA)
        ic_residual = engine.rho_floor_residual(
            cost, sigma, tau, KAPPA
        )
        row = {
            "market": name,
            "ic_floor_total": ic_total,
            "r2_floor_total": engine.r2_floor_total(
                cost, sigma, tau, KAPPA
            ),
            "r2_floor_total_pct": 100.0 * engine.r2_floor_total(
                cost, sigma, tau, KAPPA
            ),
            "ic_floor_residual": ic_residual,
            "r2_floor_residual": engine.r2_floor_residual(
                cost, sigma, tau, KAPPA
            ),
        }
        assert np.isclose(row["r2_floor_total"], ic_total**2)
        assert np.isclose(row["r2_floor_residual"], ic_residual**2)
        rows.append(row)
    return rows


def regression_rows() -> list[dict]:
    """Fit on one sample and score on a fresh sample."""
    rng = np.random.default_rng(SEED)
    sigma, tau = 0.03, 1.0
    rows = []
    for label, target_ic in (
        ("negative", -0.20),
        ("weak", 0.05),
        ("strong", 0.60),
    ):
        beta = engine.beta_exact(target_ic, sigma, tau)
        x_train = rng.standard_normal(200_000)
        y_train = beta * x_train + sigma * rng.standard_normal(x_train.size)
        intercept, slope = engine.fit_simple_regression(x_train, y_train)
        fitted_train = intercept + slope * x_train
        train_r2 = engine.r2_score(y_train, fitted_train)
        sample_ic = float(np.corrcoef(x_train, y_train)[0, 1])

        x_test = rng.standard_normal(300_000)
        y_test = beta * x_test + sigma * rng.standard_normal(x_test.size)
        test_r2 = engine.r2_score(y_test, intercept + slope * x_test)
        target_r2 = engine.r2_from_ic(target_ic)

        # Exact finite-sample OLS identity in training; convergence out of sample.
        assert np.isclose(train_r2, sample_ic**2, rtol=1e-11, atol=1e-12)
        assert abs(test_r2 - target_r2) < 0.003
        assert np.sign(slope) == np.sign(target_ic)
        rows.append(
            {
                "case": label,
                "target_ic": target_ic,
                "target_r2": target_r2,
                "fitted_intercept": intercept,
                "fitted_slope": slope,
                "train_sample_ic": sample_ic,
                "train_r2": train_r2,
                "test_r2": test_r2,
            }
        )
    return rows


def multiple_rows() -> list[dict]:
    """Translate the source's IC multiples into R² multiples."""
    return [
        {
            "ic_multiple": multiple,
            "r2_multiple": multiple**2,
            "no_trade_band_sigma": KAPPA / multiple,
            "trade_frequency": engine.trade_frequency(KAPPA / multiple),
        }
        for multiple in (1.0, 1.5, 2.0, 3.0)
    ]


def main() -> None:
    paths.ensure_directories()
    output = {
        "identity": "population simple-regression R2 = IC^2",
        "caveat": (
            "R2 loses the slope sign; out-of-sample R2 may be negative"
        ),
        "floors": floor_rows(),
        "regressions": regression_rows(),
        "multiple_mapping": multiple_rows(),
    }
    destination = paths.RESULTS_DIR / "r2_regression.json"
    with destination.open("w") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")

    print("Linear-regression R² verification")
    print("=" * 42)
    for row in output["floors"]:
        print(
            f"{row['market']:15s} IC {100 * row['ic_floor_total']:8.4f}%"
            f"  R² {row['r2_floor_total_pct']:9.6f}%"
        )
    print("\nSeeded train/test OLS")
    for row in output["regressions"]:
        print(
            f"{row['case']:15s} target {row['target_r2']:.6f}"
            f"  train {row['train_r2']:.6f}"
            f"  test {row['test_r2']:.6f}"
        )
    print(f"\nWrote {destination}")


if __name__ == "__main__":
    main()
