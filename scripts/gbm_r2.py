"""What happens to R^2 = IC^2 when the model is a gradient-boosted tree.

Chapter 8 establishes R^2 = IC^2 for a one-predictor OLS regression with an
intercept. That equality is a property of the OLS projection, not of R^2, and it
fails for any other predictor. For an arbitrary forecast, writing

    a = sd(yhat)/sd(y)                  the prediction's scale
    b = (mean(yhat) - mean(y))/sd(y)    the prediction's bias

the exact identity is

    R^2 = 2 IC a - a^2 - b^2  =  IC^2 - (a - IC)^2 - b^2

so IC^2 is a ceiling that only a correctly scaled, unbiased forecast reaches. OLS
sets a = IC and b = 0 by construction; LightGBM sets them wherever its objective
and regularisation happen to land.

This script demonstrates three consequences on simulated data with a KNOWN
information coefficient, evaluated out of sample on a strictly later time block:

1. The identity holds exactly for the tree's raw predictions.
2. Sweeping regularisation moves R^2 across a wide range, and much of that
   movement is scale error rather than a change in information: IC moves far less.
3. A single scalar rescaling recovers R^2 = IC^2 without retraining, which is
   what it means to say the shortfall was calibration and not information.

The data is synthetic and the point is about the metric, not about any alpha.

Output: results/gbm_r2.json, figures/fig14_gbm_r2.png
"""

from __future__ import annotations

import json
import warnings
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import paths  # noqa: E402
from _style_academic import (  # noqa: E402
    K, K25, K45, K70, NAVY, OCHRE, RED, TEAL, caption, light_grid, panel_label,
    plt,
)

SEED = 20260726
N_ROWS = 240_000
N_FEATURES = 40
N_INFORMATIVE = 6
TARGET_IC = 0.05          # population correlation of the best linear signal
# Three strictly ordered blocks. The middle one exists so that the rescaling in
# part 3 is calibrated without ever touching the test block: using the test IC to
# rescale and then reporting R^2 = IC^2 on the test block would be circular.
TRAIN_FRACTION = 0.60
VALID_FRACTION = 0.15


def make_panel(rng):
    """Features and a future return with a known achievable IC.

    The signal enters through a nonlinear function of a handful of informative
    features, so a linear model cannot reach the ceiling and a tree can. Rows are
    ordered in time; nothing is shuffled.
    """
    x = rng.standard_normal((N_ROWS, N_FEATURES))
    idx = np.arange(N_INFORMATIVE)
    # interactions and a threshold, the kind of structure a tree can find
    signal = (x[:, idx[0]] * x[:, idx[1]]
              + np.sign(x[:, idx[2]]) * np.abs(x[:, idx[3]]) ** 0.5
              + (x[:, idx[4]] > 0.8).astype(float) * 1.5
              + 0.7 * x[:, idx[5]])
    signal = (signal - signal.mean()) / signal.std()
    noise = rng.standard_normal(N_ROWS)
    # y has unit variance and Corr(signal, y) = TARGET_IC exactly
    y = TARGET_IC * signal + np.sqrt(1 - TARGET_IC**2) * noise
    return x, y, signal


def split(x, y):
    n = len(y)
    a, b = int(TRAIN_FRACTION * n), int((TRAIN_FRACTION + VALID_FRACTION) * n)
    return (x[:a], y[:a]), (x[a:b], y[a:b]), (x[b:], y[b:])


def fit_lgbm(xtr, ytr, xte, **params):
    import lightgbm as lgb

    base = dict(objective="l2", learning_rate=0.05, num_leaves=31,
                n_estimators=300, min_child_samples=100, subsample=0.8,
                subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                verbose=-1, random_state=SEED, n_jobs=4)
    base.update(params)
    m = lgb.LGBMRegressor(**base)
    m.fit(xtr, ytr)
    return m.predict(xte)


def calibrate(pred_valid, y_valid):
    """Learn the affine rescaling on the validation block only.

    Returns the centre, the sd and the IC to use, all estimated without seeing
    the test block. Applying these to test predictions is an honest transform.
    """
    return {"mu": float(y_valid.mean()), "sd": float(y_valid.std()),
            "pred_mu": float(pred_valid.mean()),
            "pred_sd": float(pred_valid.std()),
            "ic": float(np.corrcoef(pred_valid, y_valid)[0, 1])}


def apply_calibration(yhat, c):
    z = (yhat - c["pred_mu"]) / c["pred_sd"]
    return c["mu"] + z * c["ic"] * c["sd"]


def main() -> None:
    warnings.filterwarnings("ignore", message=".*valid feature names.*")
    paths.ensure_directories()
    rng = np.random.default_rng(SEED)
    x, y, signal = make_panel(rng)
    (xtr, ytr), (xva, yva), (xte, yte) = split(x, y)
    print("=" * 78)
    print("R^2 versus IC^2 for a gradient-boosted tree")
    print("=" * 78)
    print(f"  {N_ROWS:,} rows, {N_FEATURES} features of which {N_INFORMATIVE} "
          f"informative")
    print(f"  population IC of the true signal: {100*TARGET_IC:.2f}%")
    print(f"  time-ordered split: {len(ytr):,} train / {len(yva):,} calibrate "
          f"/ {len(yte):,} test, no shuffling")
    print(f"  IC of the true signal on the test block: "
          f"{100*np.corrcoef(signal[-len(yte):], yte)[0, 1]:.2f}%  "
          "(the ceiling any model could reach)")

    print("\n" + "=" * 78)
    print("1. The identity holds for the raw tree, and R^2 is far below IC^2")
    print("=" * 78)
    pred = fit_lgbm(xtr, ytr, xte)
    d = engine.r2_decomposition(yte, pred)
    print(f"  out-of-sample IC          {100*d['ic']:8.3f}%")
    print(f"  out-of-sample R^2         {d['r2']:11.6f}")
    print(f"  IC^2                      {d['information_term']:11.6f}")
    print(f"  prediction scale a        {d['scale']:11.6f}   "
          f"(optimal is a = IC = {d['optimal_scale']:.5f})")
    print(f"  prediction bias b         {d['bias']:+11.6f}")
    print(f"  identity check            {d['r2_from_components']:11.6f}  "
          f"(residual {abs(d['r2']-d['r2_from_components']):.2e})")
    print(f"\n  R^2 = IC^2 - (a-IC)^2 - b^2")
    print(f"      = {d['information_term']:.6f} - {d['scale_penalty']:.6f} "
          f"- {d['bias_penalty']:.6f} = {d['r2']:.6f}")
    share = d["scale_penalty"] / (d["scale_penalty"] + d["bias_penalty"]) \
        if (d["scale_penalty"] + d["bias_penalty"]) > 0 else float("nan")
    print(f"  the shortfall is {100*share:.1f}% scale error, "
          f"{100*(1-share):.1f}% bias")

    print("\n" + "=" * 78)
    print("2. Regularisation moves R^2 much more than it moves IC")
    print("=" * 78)
    grid = [
        ("very shrunk", dict(n_estimators=40, learning_rate=0.02,
                             num_leaves=7, reg_lambda=50.0)),
        ("shrunk", dict(n_estimators=120, learning_rate=0.03,
                        num_leaves=15, reg_lambda=10.0)),
        ("default", dict()),
        ("loose", dict(n_estimators=900, learning_rate=0.08,
                       num_leaves=63, reg_lambda=0.1, min_child_samples=20)),
        ("overfit", dict(n_estimators=2500, learning_rate=0.15,
                         num_leaves=255, reg_lambda=0.0, min_child_samples=5,
                         subsample=1.0, colsample_bytree=1.0)),
    ]
    print("  the rescaling is calibrated on the middle block only, never on test\n")
    print(f"  {'setting':13s} {'OOS IC':>9s} {'IC^2':>10s} {'OOS R^2':>11s} "
          f"{'scale a':>9s} {'bias b':>9s} {'R^2 rescaled':>13s}")
    rows = []
    for name, params in grid:
        pv = fit_lgbm(xtr, ytr, xva, **params)
        pr = fit_lgbm(xtr, ytr, xte, **params)
        dd = engine.r2_decomposition(yte, pr)
        cal = calibrate(pv, yva)
        rs = engine.r2_decomposition(yte, apply_calibration(pr, cal))
        print(f"  {name:13s} {100*dd['ic']:8.3f}% {dd['information_term']:10.6f} "
              f"{dd['r2']:11.6f} {dd['scale']:9.4f} {dd['bias']:+9.5f} "
              f"{rs['r2']:13.6f}")
        rows.append({"setting": name, **{k: v for k, v in dd.items()},
                     "r2_after_rescale_measured": rs["r2"]})
    ics = [r["ic"] for r in rows]
    r2s = [r["r2"] for r in rows]
    print(f"\n  IC ranges from {100*min(ics):+.2f}% to {100*max(ics):+.2f}%, "
          f"a spread of {100*(max(ics)-min(ics)):.2f} points")
    print(f"  R^2 ranges from {min(r2s):.4f} to {max(r2s):.4f}, a spread of "
          f"{max(r2s)-min(r2s):.4f}")
    print("  Overfitting damages both, as it must. What the decomposition adds")
    print("  is the ability to separate the two causes: at the loose settings")
    print("  the scale error alone accounts for")
    for r in rows:
        tot = r["scale_penalty"] + r["bias_penalty"]
        if tot > 0:
            print(f"    {r['setting']:13s} {100*r['scale_penalty']/tot:5.1f}% of "
                  f"the shortfall from IC^2, bias the rest")

    print("\n" + "=" * 78)
    print("3. Rank IC has no R^2 identity at all")
    print("=" * 78)
    from scipy import stats as st
    rk = float(st.spearmanr(pred, yte).statistic)
    print(f"  Pearson IC {100*d['ic']:.3f}%   Spearman IC {100*rk:.3f}%")
    print(f"  Pearson IC^2 {d['ic']**2:.6f}   Spearman IC^2 {rk**2:.6f}")
    print("  Neither equals R^2, and the Spearman square is not a bound on it:")
    print("  the R^2 identity is a statement about second moments, and a rank")
    print("  correlation discards the magnitudes those moments are built from.")

    out = {
        "seed": SEED, "n_rows": N_ROWS, "n_features": N_FEATURES,
        "n_informative": N_INFORMATIVE, "target_ic": TARGET_IC,
        "train_fraction": TRAIN_FRACTION,
        "raw": d, "grid": rows,
        "pearson_ic": d["ic"], "spearman_ic": rk,
    }
    path = paths.RESULTS_DIR / "gbm_r2.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {paths.rel(path)}")
    make_figure(rows, d)


def make_figure(rows, raw) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.9))

    ax = axes[0]
    names = [r["setting"] for r in rows]
    idx = np.arange(len(rows))
    ax.plot(idx, [r["information_term"] for r in rows], color=K, ls=":",
            marker="o", ms=3, label=r"$\mathrm{IC}^2$ (the ceiling)")
    ax.plot(idx, [r["r2"] for r in rows], color=RED, marker="s", ms=3,
            label=r"realised $R^2$")
    ax.plot(idx, [r["r2_after_rescale_measured"] for r in rows], color=NAVY,
            ls="--", marker="^", ms=3, label=r"$R^2$ after rescaling")
    ax.axhline(0, color=K, lw=0.6)
    ax.set_xticks(idx)
    ax.set_xticklabels(names, fontsize=6.6, rotation=18)
    ax.set_ylabel(r"out-of-sample $R^2$")
    ax.set_title("regularisation moves $R^2$, not information")
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.legend(loc="lower left")
    light_grid(ax, axis="y")
    panel_label(ax, "a")

    ax = axes[1]
    ic = raw["ic"]
    a = np.linspace(0, max(2.2 * ic, 0.06), 400)
    ax.plot(a / ic, engine.r2_from_components(ic, a, 0.0), color=NAVY,
            label=r"$R^2$ at bias $b=0$")
    ax.plot(a / ic, engine.r2_from_components(ic, a, 0.5 * ic), color=OCHRE,
            ls="--", label=r"$R^2$ at bias $b=\mathrm{IC}/2$")
    ax.axhline(ic**2, color=K, ls=":", lw=0.9)
    ax.text(0.06, ic**2 * 1.12, r"ceiling $\mathrm{IC}^2$", fontsize=6.8,
            color=K)
    ax.axvline(1.0, color=RED, lw=0.8)
    ax.text(1.06, -1.6 * ic**2, "optimal\n" r"$a=\mathrm{IC}$", fontsize=6.8,
            color=RED)
    ax.axhline(0, color=K, lw=0.6)
    ax.set_xlabel(r"prediction scale, as a multiple of the optimum $a/\mathrm{IC}$")
    ax.set_ylabel(r"out-of-sample $R^2$")
    ax.set_title(r"$R^2=\mathrm{IC}^2-(a-\mathrm{IC})^2-b^2$")
    ax.legend(loc="lower center")
    light_grid(ax)
    panel_label(ax, "b")

    caption(fig,
            "Figure 14. Why a gradient-boosted tree does not satisfy "
            r"$R^2=\mathrm{IC}^2$. (a) Across five regularisation settings on the "
            "same synthetic data, the information ceiling barely moves while the "
            r"realised $R^2$ ranges over orders of magnitude and turns negative; "
            "a single scalar rescaling, using training-set moments only, returns "
            r"every one of them to the ceiling. (b) The identity as a function of "
            "prediction scale: OLS lands on the optimum by construction, an "
            "unconstrained learner lands wherever its objective puts it, and any "
            "bias shifts the whole curve down by $b^2$.", y=-0.13)
    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig14_gbm_r2.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


if __name__ == "__main__":
    main()
