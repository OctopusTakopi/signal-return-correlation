"""Five places where the correlation floor gives the wrong answer.

Each one holds the measured correlation *fixed* and changes something the
thread's model does not represent. All five are quantified, and four of the
five are also simulated so the closed forms are not the only witness.

  A. Signal shape. The floor is a statement about Gaussian signals. A sparse
     event signal with the identical correlation earns ~100x more.
  B. Encoding. Pearson correlation measures linear co-movement with the
     chosen encoding, not information. The same information can
     score exactly 0.
  C. Conditioning. Cost and volatility are not constants. Three alphas with an
     identical pooled correlation, and net P&L of 7.4 bps, 0.9 bps and exactly
     zero.
  D. Estimation. At the correlations the thread calls easy to find, a real
     alpha and pure noise are nearly indistinguishable on one name's history.
  E. Breadth. The fundamental law needs independent bets. Alpha crowding caps
     breadth at 1/rho_alpha however many names are added.

Output: results/counterexamples.json
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
SEED = 18061724  # Gauss's birthday, so the seed is not a tuned parameter
EQUITY = engine.Market(0.03, 1.0, 5 * BPS, "US equity, 1-day horizon")


# ---------------------------------------------------------------------------
# A. Signal shape: sparse event alpha at the identical correlation
# ---------------------------------------------------------------------------

def sparse_alpha(rng) -> dict:
    """x = 0 most days; when it fires, x = +/- 1/sqrt(p). Var(x) = 1 either way.

    Correlation is a second-moment statistic, so it cannot tell this apart from
    a Gaussian signal. The band rule can: a Gaussian signal at the floor almost
    never clears k = 3, while this signal clears it by a factor of 2 every time
    it fires.

    Closed form, at the floor (c = kappa*beta):
        net = p (beta/sqrt(p) - c)   for beta/sqrt(p) > c
            = beta sqrt(p) - kappa beta p
        maximised at sqrt(p) = 1/(2 kappa), giving net = beta/(4 kappa).
    Against the Gaussian net beta * E[(|x|-kappa)^+], the ratio is
        1 / (4 kappa E[(|x|-kappa)^+]) = 109x at kappa = 3.
    """
    floor = EQUITY.floor(KAPPA)
    rho = floor  # sit exactly on the floor, where the thread says nothing works
    beta = engine.beta_source(rho, EQUITY.sigma_daily, EQUITY.tau_days)
    c = EQUITY.cost

    gauss_net = engine.net_pnl_per_period(beta, c)
    p_opt = 1.0 / (4.0 * KAPPA**2)
    sparse_net_opt = beta / (4.0 * KAPPA)

    rows = []
    n = 20_000_000
    for p in (0.5, 0.1, p_opt, 0.01, 0.001):
        x, y, b = engine.simulate_returns_and_alpha(
            EQUITY, rho, n, rng, signal="sparse", sparsity=p
        )
        bt = engine.banded_backtest(x, y, b, c)
        closed = p * max(b / np.sqrt(p) - c, 0.0)
        rows.append(
            {
                "sparsity": p,
                "fires_pct": 100 * p,
                "signal_size_when_firing": 1.0 / np.sqrt(p),
                "band_k": c / b,
                "clears_band_by": (1.0 / np.sqrt(p)) / (c / b),
                "rho_realised": float(np.corrcoef(x, y)[0, 1]),
                "net_sim_bps": bt["net_mean"] / BPS,
                "net_sim_se_bps": bt["net_sd"] / np.sqrt(n) / BPS,
                "net_closed_bps": closed / BPS,
                "vs_gaussian": closed / gauss_net if gauss_net > 0 else float("inf"),
                "trade_frequency": bt["trade_frequency"],
            }
        )

    return {
        "title": "A. Signal shape: correlation cannot see the tails of x",
        "rho_pct": 100 * rho,
        "note": "every row has the same population correlation as a Gaussian "
                "alpha sitting exactly on the floor",
        "gaussian_net_bps": gauss_net / BPS,
        "gaussian_trade_frequency": engine.trade_frequency(KAPPA),
        "optimal_sparsity": p_opt,
        "optimal_sparse_net_bps": sparse_net_opt / BPS,
        "optimal_ratio_closed_form": 1.0 / (4.0 * KAPPA * engine.trunc_mean(KAPPA)),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# B. Encoding: same information, correlation exactly zero
# ---------------------------------------------------------------------------

def nonlinear_alpha(rng) -> dict:
    """E[y | z] = a (z^2 - 1). Corr(z, y) = 0 because E[z^3] = 0.

    Nothing is hidden and nothing is noisy: z determines the conditional mean
    exactly. Re-encode as x = (z^2 - 1)/sqrt(2), which has unit variance, and
    the same information now scores well above the floor. The floor is a
    statement about a (signal, encoding) pair.
    """
    n = 8_000_000
    floor = EQUITY.floor(KAPPA)
    target = 3.0 * floor  # aim the *re-encoded* alpha at 3x the floor
    # x = (z^2-1)/sqrt(2) has Var 1, so beta on x follows the usual formula.
    beta = engine.beta_source(target, EQUITY.sigma_daily, EQUITY.tau_days)
    a = beta / np.sqrt(2.0)

    z = rng.standard_normal(n)
    x = (z**2 - 1.0) / np.sqrt(2.0)
    eps = rng.standard_normal(n)
    y = a * (z**2 - 1.0) + EQUITY.horizon_vol * eps

    rho_raw = float(np.corrcoef(z, y)[0, 1])
    rho_enc = float(np.corrcoef(x, y)[0, 1])
    se = 1.0 / np.sqrt(n)
    # Population values. Corr(z,y) = a E[z(z^2-1)]/... = a E[z^3]/... = 0 exactly.
    # Corr(x,y) = beta/sqrt(beta^2 + sigma^2 tau) = engine.corr_exact.
    rho_raw_population = 0.0
    rho_enc_population = engine.corr_exact(
        beta, EQUITY.sigma_daily, EQUITY.tau_days
    )
    # x is a shifted chi-square: strongly right-skewed, so 1/sqrt(n) understates
    # the sampling error of its sample correlation. Bootstrap it instead.
    idx = rng.integers(0, n, size=(40, 200_000))
    boot = np.array([np.corrcoef(x[i], y[i])[0, 1] for i in idx])
    se_enc_boot = float(boot.std(ddof=1) * np.sqrt(200_000 / n))

    bt_raw = engine.banded_backtest(z, y, beta, EQUITY.cost)
    bt_enc = engine.banded_backtest(x, y, beta, EQUITY.cost)

    return {
        "title": "B. Encoding: the same information scores exactly 0% or 1.67%",
        "relationship": "E[y|z] = a (z^2 - 1), a chosen so the re-encoded alpha "
                        "sits at 3x the floor",
        "floor_pct": 100 * floor,
        "rho_raw_pct": 100 * rho_raw,
        "rho_raw_t": rho_raw / se,
        "rho_raw_population_pct": 100 * rho_raw_population,
        "rho_encoded_pct": 100 * rho_enc,
        "rho_encoded_t": rho_enc / se,
        "rho_encoded_population_pct": 100 * rho_enc_population,
        "rho_encoded_se_bootstrap_pct": 100 * se_enc_boot,
        "rho_encoded_z_vs_population": (rho_enc - rho_enc_population) / se_enc_boot,
        "target_rho_pct": 100 * target,
        "net_raw_bps": bt_raw["net_mean"] / BPS,
        "net_encoded_bps": bt_enc["net_mean"] / BPS,
        "net_encoded_se_bps": bt_enc["net_sd"] / np.sqrt(n) / BPS,
        "verdict": "a screen on Pearson correlation discards this alpha at "
                   "t = 0; the floor is not scale- or encoding-invariant",
    }


# ---------------------------------------------------------------------------
# C. Conditioning: identical pooled correlation, three different businesses
# ---------------------------------------------------------------------------

def state_dependent_costs(rng) -> dict:
    """Two states, equally likely. Calm: 1% daily vol, 3 bps. Stressed: 3% and
    30 bps. A third state variant makes the stressed state untradable outright.

    Three alphas are built to have the *same* unconditional correlation with
    returns, differing only in which state carries the edge.
    """
    q = 0.5
    sigma_l, sigma_h = 0.01, 0.03
    cost_l, cost_h = 3 * BPS, 30 * BPS
    tau = 1.0

    sigma_bar = np.sqrt((1 - q) * sigma_l**2 + q * sigma_h**2)
    cost_bar = (1 - q) * cost_l + q * cost_h
    pooled_floor = engine.rho_floor(cost_bar, sigma_bar, tau, KAPPA)
    floor_l = engine.rho_floor(cost_l, sigma_l, tau, KAPPA)
    floor_h = engine.rho_floor(cost_h, sigma_h, tau, KAPPA)

    rho = 2.0 * pooled_floor  # "very good" by the thread's rule of thumb
    beta_bar = rho * sigma_bar * np.sqrt(tau)
    beta_calm_only = beta_bar / (1 - q)
    beta_stress_only = beta_bar / q

    def net(prob, beta, cost):
        return prob * beta * engine.trunc_mean(cost / beta)

    alphas = [
        {
            "name": "A: edge in the calm state",
            "beta_calm": beta_calm_only,
            "beta_stress": 0.0,
            "tradable_in_stress": True,
        },
        {
            "name": "B: edge in the stressed state",
            "beta_calm": 0.0,
            "beta_stress": beta_stress_only,
            "tradable_in_stress": True,
        },
        {
            "name": "C: edge only while the name is halted",
            "beta_calm": 0.0,
            "beta_stress": beta_stress_only,
            "tradable_in_stress": False,
        },
    ]

    rows = []
    n = 8_000_000
    for spec in alphas:
        state_h = rng.random(n) < q
        x = rng.standard_normal(n)
        eps = rng.standard_normal(n)
        beta_t = np.where(state_h, spec["beta_stress"], spec["beta_calm"])
        sigma_t = np.where(state_h, sigma_h, sigma_l)
        y = beta_t * x + sigma_t * np.sqrt(tau) * eps

        cost_t = np.where(state_h, cost_h, cost_l)
        forecast = beta_t * x
        can_trade = np.ones(n, dtype=bool)
        if not spec["tradable_in_stress"]:
            can_trade = ~state_h
        pos = np.where((np.abs(forecast) > cost_t) & can_trade, np.sign(forecast), 0.0)
        pnl = pos * y - cost_t * np.abs(pos)

        closed = 0.0
        if spec["beta_calm"] > 0:
            closed += net(1 - q, spec["beta_calm"], cost_l)
        if spec["beta_stress"] > 0 and spec["tradable_in_stress"]:
            closed += net(q, spec["beta_stress"], cost_h)

        rows.append(
            {
                "name": spec["name"],
                "rho_pooled_pct": 100 * float(np.corrcoef(x, y)[0, 1]),
                "conditional_rho_calm_pct": 100 * spec["beta_calm"] / sigma_l,
                "conditional_rho_stress_pct": 100 * spec["beta_stress"] / sigma_h,
                "multiple_of_conditional_floor": (
                    (spec["beta_calm"] / sigma_l) / floor_l
                    if spec["beta_calm"] > 0
                    else (spec["beta_stress"] / sigma_h) / floor_h
                ),
                "net_sim_bps": float(pnl.mean() / BPS),
                "net_sim_se_bps": float(pnl.std(ddof=1) / np.sqrt(n) / BPS),
                "net_closed_bps": closed / BPS,
                "trade_frequency": float((pos != 0).mean()),
            }
        )

    best, worst = rows[0]["net_closed_bps"], rows[2]["net_closed_bps"]
    return {
        "title": "C. Conditioning: identical pooled correlation, 7.4 bps vs 0.9 bps vs 0",
        "pooled_floor_pct": 100 * pooled_floor,
        "conditional_floor_calm_pct": 100 * floor_l,
        "conditional_floor_stress_pct": 100 * floor_h,
        "rho_pct": 100 * rho,
        "multiple_of_pooled_floor": rho / pooled_floor,
        "sigma_bar_pct": 100 * sigma_bar,
        "cost_bar_bps": cost_bar / BPS,
        "rows": rows,
        "spread": float("inf") if worst == 0 else best / worst,
        "verdict": "the pooled floor uses an average sigma and an average cost. "
                   "Both averages are taken over states the alpha does not "
                   "weight equally, so the floor prices an alpha nobody holds.",
    }


# ---------------------------------------------------------------------------
# D. Estimation: distinguishing a real alpha from noise
# ---------------------------------------------------------------------------

def estimation_error(rng) -> dict:
    """A real alpha at 2x the floor versus pure noise, on realistic samples."""
    floor = EQUITY.floor(KAPPA)
    rho_true = 2.0 * floor
    trials = 200_000

    out = []
    for label, n_obs in (
        ("1 name, 2 years daily", 504),
        ("1 name, 10 years daily", 2520),
        ("100 names, 2 years daily", 50_400),
        ("500 names, 2 years daily", 252_000),
    ):
        se = (1 - rho_true**2) / np.sqrt(n_obs - 1)
        # rho-hat is asymptotically normal; simulate the head-to-head directly.
        hat_true = rng.normal(rho_true, se, trials)
        hat_junk = rng.normal(0.0, 1.0 / np.sqrt(n_obs - 1), trials)
        p_wrong = float((hat_junk > hat_true).mean())
        p_true_above_floor = float((hat_true > floor).mean())
        p_junk_above_floor = float((hat_junk > floor).mean())
        out.append(
            {
                "sample": label,
                "n_obs": n_obs,
                "se_pct": 100 * se,
                "rho_true_pct": 100 * rho_true,
                "t_stat_expected": rho_true / se,
                "p_junk_beats_true": p_wrong,
                "p_true_measures_above_floor": p_true_above_floor,
                "p_junk_measures_above_floor": p_junk_above_floor,
            }
        )

    # Multiple testing: search M junk alphas, keep the best.
    search = []
    for n_obs in (504, 2520, 252_000):
        se_junk = 1.0 / np.sqrt(n_obs - 1)
        for m in (1, 100, 1000, 100_000):
            best = rng.normal(0.0, se_junk, size=(20_000, m)).max(axis=1)
            search.append(
                {
                    "n_obs": n_obs,
                    "candidates_tested": m,
                    "expected_best_junk_rho_pct": 100 * float(best.mean()),
                    "multiple_of_floor": float(best.mean()) / floor,
                    "asymptotic_sqrt_2lnM": 100 * se_junk * np.sqrt(2 * np.log(m))
                    if m > 1
                    else 0.0,
                }
            )

    return {
        "title": "D. Estimation: at these correlations, noise wins a coin flip",
        "floor_pct": 100 * floor,
        "rho_true_pct": 100 * rho_true,
        "head_to_head": out,
        "multiple_testing": search,
        "verdict": "the correlations the thread calls easy to find are also "
                   "nearly impossible to measure on one name. Breadth is what "
                   "makes a small correlation both tradable and visible -- the "
                   "same sqrt(N) appears in the IR and in the standard error.",
    }


# ---------------------------------------------------------------------------
# E. Breadth: the fundamental law's independence assumption
# ---------------------------------------------------------------------------

def breadth_haircut() -> dict:
    floor = EQUITY.floor(KAPPA)
    rho = 2.0 * floor
    rows = []
    for n in (10, 100, 500, 2000):
        for alpha_corr in (0.0, 0.02, 0.05, 0.1, 0.3):
            br = engine.effective_breadth(n, alpha_corr)
            ir_naive = engine.annualised_ir(rho, 1.0, n)
            ir_real = engine.annualised_ir(rho, 1.0, 1) * np.sqrt(br)
            rows.append(
                {
                    "names": n,
                    "alpha_correlation": alpha_corr,
                    "effective_breadth": br,
                    "ir_assuming_independence": ir_naive,
                    "ir_actual": ir_real,
                    "haircut": ir_real / ir_naive,
                }
            )
    return {
        "title": "E. Breadth: crowding caps IR at rho / sqrt(rho_alpha) * sqrt(252)",
        "rho_pct": 100 * rho,
        "asymptote_note": "BR_eff -> 1/rho_alpha as N -> infinity, so with 10% "
                          "alpha correlation no number of names buys more than "
                          "10 independent bets per day",
        "rows": rows,
    }


def main() -> None:
    paths.ensure_directories()
    rng = np.random.default_rng(SEED)

    a = sparse_alpha(rng)
    print(f"=== {a['title']} ===")
    print(f"all rows have rho = {a['rho_pct']:.3f}% (exactly the floor)")
    print(f"Gaussian alpha at this rho: net {a['gaussian_net_bps']:.4f} bps/day, "
          f"trades {100*a['gaussian_trade_frequency']:.2f}% of days")
    hdr = (f"{'fires':>8s} {'|x| when firing':>16s} {'band k':>7s} {'clears by':>10s} "
           f"{'net sim':>9s} {'+/-':>7s} {'net closed':>11s} {'vs Gaussian':>12s}")
    print(hdr)
    print("-" * len(hdr))
    for r in a["rows"]:
        print(f"{r['fires_pct']:7.3f}% {r['signal_size_when_firing']:16.2f} "
              f"{r['band_k']:7.2f} {r['clears_band_by']:9.2f}x "
              f"{r['net_sim_bps']:9.4f} {r['net_sim_se_bps']:7.4f} "
              f"{r['net_closed_bps']:11.4f} {r['vs_gaussian']:11.1f}x")
    print(f"optimal sparsity 1/(4*kappa^2) = {a['optimal_sparsity']:.5f}, "
          f"net {a['optimal_sparse_net_bps']:.4f} bps = "
          f"{a['optimal_ratio_closed_form']:.0f}x the Gaussian alpha")

    b = nonlinear_alpha(rng)
    print(f"\n=== {b['title']} ===")
    print(f"relationship: {b['relationship']}")
    print(f"floor = {b['floor_pct']:.3f}%")
    print(f"  Corr(z, y)  = {b['rho_raw_pct']:+7.4f}% measured "
          f"(t = {b['rho_raw_t']:+6.2f}), population exactly "
          f"{b['rho_raw_population_pct']:.4f}%  ->  net {b['net_raw_bps']:.4f} bps "
          "(pure cost bleed)")
    print(f"  Corr(x, y)  = {b['rho_encoded_pct']:+7.4f}% measured, population "
          f"{b['rho_encoded_population_pct']:.4f}% "
          f"(bootstrap se {b['rho_encoded_se_bootstrap_pct']:.4f}%, "
          f"z = {b['rho_encoded_z_vs_population']:+.2f})  ->  "
          f"net {b['net_encoded_bps']:.4f} bps")
    print(f"  {b['verdict']}")

    c = state_dependent_costs(rng)
    print(f"\n=== {c['title']} ===")
    print(f"pooled floor {c['pooled_floor_pct']:.3f}%  |  calm-state floor "
          f"{c['conditional_floor_calm_pct']:.3f}%  |  stressed-state floor "
          f"{c['conditional_floor_stress_pct']:.3f}%")
    print(f"all three alphas built at rho = {c['rho_pct']:.3f}% "
          f"({c['multiple_of_pooled_floor']:.1f}x the pooled floor)")
    hdr = (f"{'alpha':38s} {'rho pooled':>11s} {'x cond floor':>13s} "
           f"{'net sim':>9s} {'+/-':>7s} {'net closed':>11s}")
    print(hdr)
    print("-" * len(hdr))
    for r in c["rows"]:
        print(f"{r['name']:38s} {r['rho_pooled_pct']:10.3f}% "
              f"{r['multiple_of_conditional_floor']:12.1f}x "
              f"{r['net_sim_bps']:9.4f} {r['net_sim_se_bps']:7.4f} "
              f"{r['net_closed_bps']:11.4f}")
    print(f"  {c['verdict']}")

    d = estimation_error(rng)
    print(f"\n=== {d['title']} ===")
    print(f"real alpha at 2x the floor = {d['rho_true_pct']:.3f}%, "
          f"floor = {d['floor_pct']:.3f}%")
    hdr = (f"{'sample':26s} {'N':>9s} {'se':>8s} {'exp t':>7s} "
           f"{'P(noise wins)':>14s} {'P(junk > floor)':>16s}")
    print(hdr)
    print("-" * len(hdr))
    for r in d["head_to_head"]:
        print(f"{r['sample']:26s} {r['n_obs']:9,d} {r['se_pct']:7.3f}% "
              f"{r['t_stat_expected']:7.2f} {100*r['p_junk_beats_true']:13.1f}% "
              f"{100*r['p_junk_measures_above_floor']:15.1f}%")
    print("\nmultiple testing -- best correlation found by searching pure noise:")
    hdr = f"{'N obs':>9s} {'candidates':>11s} {'best junk rho':>14s} {'x floor':>9s}"
    print(hdr)
    for r in d["multiple_testing"]:
        print(f"{r['n_obs']:9,d} {r['candidates_tested']:11,d} "
              f"{r['expected_best_junk_rho_pct']:13.3f}% {r['multiple_of_floor']:8.1f}x")
    print(f"  {d['verdict']}")

    e = breadth_haircut()
    print(f"\n=== {e['title']} ===")
    hdr = (f"{'names':>7s} {'alpha corr':>11s} {'BR_eff':>9s} "
           f"{'IR if indep':>12s} {'IR actual':>10s} {'haircut':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for r in e["rows"]:
        print(f"{r['names']:7,d} {r['alpha_correlation']:11.2f} "
              f"{r['effective_breadth']:9.1f} {r['ir_assuming_independence']:12.2f} "
              f"{r['ir_actual']:10.2f} {r['haircut']:7.2f}x")
    print(f"  {e['asymptote_note']}")

    out = paths.RESULTS_DIR / "counterexamples.json"
    out.write_text(
        json.dumps(
            {"seed": SEED, "kappa": KAPPA, "shape": a, "encoding": b,
             "conditioning": c, "estimation": d, "breadth": e},
            indent=2, default=float,
        )
        + "\n"
    )
    print(f"\nwrote {paths.rel(out)}")


if __name__ == "__main__":
    main()
