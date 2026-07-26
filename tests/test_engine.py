"""Unit tests for engine.py.

Run with `python -m pytest tests -q` or `python tests/test_engine.py`.

These check the closed forms against independent numerical integration and
against the source's own stated numbers, so a refactor that quietly changes a
formula fails here rather than in the video.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import integrate, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402

BPS = 1e-4
KAPPA = 3.0
EQUITY = engine.Market(0.03, 1.0, 5 * BPS)
FX = engine.Market(0.003, 1.0 / 1440.0, 0.2 * BPS)


def close(a, b, rel=1e-9, abs_=0.0):
    assert np.isclose(a, b, rtol=rel, atol=abs_), f"{a!r} != {b!r}"


# --- the source's algebra ---------------------------------------------------

def test_corr_roundtrip_exact():
    for rho in (0.001, 0.005, 0.05, 0.5, 0.9):
        beta = engine.beta_exact(rho, 0.03, 1.0)
        close(engine.corr_exact(beta, 0.03, 1.0), rho)


def test_source_approximation_is_second_order():
    """corr_source overstates by exactly rho^2/2 to leading order."""
    for rho in (0.005, 0.02, 0.085):
        beta = engine.beta_source(rho, 0.03, 1.0)
        exact = engine.corr_exact(beta, 0.03, 1.0)
        rel = exact / rho - 1.0
        close(rel, -rho**2 / 2, rel=2e-2)


def test_worked_examples_match_the_thread():
    close(EQUITY.floor(KAPPA), 0.005555555, rel=1e-6)
    close(engine.Market(0.015, 1.0, 10 * BPS).floor(KAPPA), 0.022222222, rel=1e-6)
    close(FX.floor(KAPPA), 0.084327404, rel=1e-6)
    # The thread's rounded figures, to one significant figure.
    assert abs(EQUITY.floor(KAPPA) - 0.005) < 0.0006
    assert abs(FX.floor(KAPPA) - 0.085) < 0.001


def test_hedging_quadruples_the_floor():
    plain = engine.rho_floor(5 * BPS, 0.03, 1.0, KAPPA)
    hedged = engine.rho_floor(10 * BPS, 0.015, 1.0, KAPPA)
    close(hedged / plain, 4.0)


def test_floor_scales_as_one_over_root_tau():
    a = engine.rho_floor(5 * BPS, 0.03, 1.0, KAPPA)
    b = engine.rho_floor(5 * BPS, 0.03, 0.5, KAPPA)
    close(b / a, np.sqrt(2.0))


# --- linear-regression R² -------------------------------------------------

def test_population_r2_is_ic_squared():
    for rho in (-0.8, -0.2, 0.0, 0.005, 0.5):
        beta = engine.beta_exact(rho, 0.03, 1.0)
        close(engine.r2_exact(beta, 0.03, 1.0),
              engine.r2_from_ic(rho))
        close(engine.ic_magnitude_from_r2(rho**2), abs(rho))


def test_r2_inversion_keeps_sign_separate():
    for rho in (-0.8, -0.2, 0.2, 0.8):
        beta = engine.beta_from_r2(rho**2, 0.03, 1.0, sign=rho)
        close(beta, engine.beta_exact(rho, 0.03, 1.0))
    assert engine.r2_from_ic(-0.2) == engine.r2_from_ic(0.2)


def test_total_vol_r2_floor_is_squared_ic_floor():
    for market in (EQUITY, FX):
        close(engine.r2_floor_total(
            market.cost, market.sigma_daily, market.tau_days, KAPPA
        ), market.floor(KAPPA)**2)
    # A requirement above one is an impossibility result, not a score to clip.
    assert engine.r2_floor_total(0.01, 0.001, 1.0, KAPPA) > 1.0
    assert engine.is_unreachable(0.01, 0.001, 1.0, KAPPA)


def test_residual_vol_r2_floor_is_exact_at_cost_boundary():
    cost, sigma, tau = 5 * BPS, 0.03, 1.0
    floor = engine.r2_floor_residual(cost, sigma, tau, KAPPA)
    beta_at_boundary = cost / KAPPA
    close(engine.r2_exact(beta_at_boundary, sigma, tau), floor)
    a = engine.rho_floor(cost, sigma, tau, KAPPA)
    close(floor, a**2 / (1.0 + a**2))


def test_r2_odds_measure_the_loading_multiple():
    floor = engine.r2_floor_residual(5 * BPS, 0.03, 1.0, KAPPA)
    floor_odds = engine.r2_odds(floor)
    for multiple in (0.5, 1.0, 1.5, 2.0, 5.0):
        odds = multiple**2 * floor_odds
        score = odds / (1.0 + odds)
        close(engine.loading_multiple_from_r2(score, floor), multiple)
    # In the total-vol reading, 2x IC is exactly 4x R².
    rho = EQUITY.floor(KAPPA)
    close(engine.r2_from_ic(2 * rho) / engine.r2_from_ic(rho), 4.0)


def test_fitted_simple_regression_r2_equals_sample_correlation_squared():
    rng = np.random.default_rng(713)
    x = rng.normal(size=20_000)
    y = -0.4 * x + rng.normal(size=x.size)
    intercept, slope = engine.fit_simple_regression(x, y)
    score = engine.r2_score(y, intercept + slope * x)
    close(score, np.corrcoef(x, y)[0, 1]**2, rel=1e-12, abs_=1e-12)
    assert slope < 0.0


def test_out_of_sample_r2_can_be_negative():
    y = np.array([-2.0, -1.0, 1.0, 2.0])
    assert engine.r2_score(y, -y) < 0.0


# --- the band rule ---------------------------------------------------------

def test_band_depends_only_on_the_multiple():
    """k = kappa/m, whatever the market."""
    for market in (EQUITY, FX):
        for m in (0.5, 1.0, 2.0, 5.0):
            beta = engine.beta_source(m * market.floor(KAPPA),
                                      market.sigma_daily, market.tau_days)
            close(market.cost / beta, KAPPA / m, rel=1e-9)


def test_trunc_mean_against_quadrature():
    for k in (0.0, 0.5, 1.5, 3.0, 6.0):
        # quad's default absolute tolerance swamps the integral once k is large,
        # so ask for more accuracy rather than loosening the comparison.
        want, _ = integrate.quad(
            lambda u: 2 * (u - k) * stats.norm.pdf(u), k, k + 40,
            epsabs=1e-24, epsrel=1e-12,
        )
        close(engine.trunc_mean(k), want, rel=1e-7)


def test_trunc_second_moment_against_quadrature():
    for k in (0.0, 0.5, 1.5, 3.0, 6.0):
        want, _ = integrate.quad(
            lambda u: 2 * (u - k) ** 2 * stats.norm.pdf(u), k, k + 40,
            epsabs=1e-24, epsrel=1e-12,
        )
        close(engine.trunc_second_moment(k), want, rel=1e-7)


def test_trunc_moment_limits():
    close(engine.trunc_mean(0.0), np.sqrt(2 / np.pi))
    close(engine.trunc_second_moment(0.0), 1.0)


def test_rule_of_thumb_one_reproduces():
    """1.5x the floor gives a trade in about 5% of periods."""
    f = engine.trade_frequency(engine.band_from_multiple(1.5, KAPPA))
    close(f, 0.04550026, rel=1e-5)
    assert abs(f - 0.05) < 0.01, "the thread's 'about 5%' should hold"


def test_rule_of_thumb_two_fails():
    """2x the floor gives 13.4%, not the stated 20-30%."""
    f = engine.trade_frequency(engine.band_from_multiple(2.0, KAPPA))
    close(f, 0.13361440, rel=1e-5)
    assert not (0.20 <= f <= 0.30)
    close(engine.multiple_from_frequency(0.20, KAPPA), 2.34078, rel=1e-4)
    close(engine.multiple_from_frequency(0.30, KAPPA), 2.89459, rel=1e-4)


def test_net_pnl_is_positive_below_the_floor():
    """The floor is a relevance threshold, not a profitability threshold.

    Mathematically net P&L is strictly positive for every beta > 0. In float64
    it underflows to exactly zero once the band exceeds about k = 39, which for
    the equity case is m < 0.077. Test the claim where it is representable, and
    separately check the underflow is to zero and never to a negative number.
    """
    for m in (0.1, 0.25, 0.5, 0.99):
        beta = engine.beta_source(m * EQUITY.floor(KAPPA), 0.03, 1.0)
        net = engine.net_pnl_per_period(beta, EQUITY.cost)
        assert net > 0.0, f"m={m} gave {net}"
    # Deep below the floor the closed form underflows rather than turning
    # negative, which is the safe direction for a claim of positivity.
    for m in (0.01, 0.05):
        beta = engine.beta_source(m * EQUITY.floor(KAPPA), 0.03, 1.0)
        assert engine.net_pnl_per_period(beta, EQUITY.cost) == 0.0


def test_trunc_mean_underflow_boundary():
    """Document where float64 gives up, so a zero is never read as a result."""
    assert engine.trunc_mean(38.0) > 0.0
    assert engine.trunc_mean(40.0) == 0.0
    # monotone decreasing and never negative, everywhere either side
    ks = np.linspace(0, 45, 400)
    vals = np.array([engine.trunc_mean(k) for k in ks])
    assert np.all(vals >= 0.0)
    assert np.all(np.diff(vals) <= 1e-18)


def test_gross_capture_is_scale_free():
    """Capture depends on the multiple alone, so equities and FX agree."""
    for m in (1.0, 2.0, 4.0):
        want = engine.gross_capture(m, KAPPA)
        for market in (EQUITY, FX):
            beta = engine.beta_source(m * market.floor(KAPPA),
                                      market.sigma_daily, market.tau_days)
            got = (engine.net_pnl_per_period(beta, market.cost)
                   / engine.gross_pnl_per_period(beta))
            close(got, want, rel=1e-9)


def test_cost_conventions_reconcile_at_zero_autocorrelation():
    """Round-trip-per-period costs c*f; charging position changes costs
    c*f*(1-f/2). The second must be the cheaper of the two."""
    beta = engine.beta_source(2 * EQUITY.floor(KAPPA), 0.03, 1.0)
    k = EQUITY.cost / beta
    f = engine.trade_frequency(k)
    roundtrip = EQUITY.cost * f
    turnover = 0.5 * EQUITY.cost * engine.turnover_iid(f)
    close(turnover, roundtrip * (1 - f / 2))
    assert turnover < roundtrip
    close(engine.net_pnl_turnover_iid(beta, EQUITY.cost),
          engine.gross_pnl_banded(beta, k) - turnover)


# --- R-squared as a disguised Sharpe ratio --------------------------------

def test_r2_decomposition_is_exact_for_arbitrary_predictions():
    """R^2 = IC^2 - (a - IC)^2 - b^2 holds for any predictor, not just OLS."""
    rng = np.random.default_rng(11)
    for _ in range(6):
        n = 50_000
        y = rng.standard_normal(n) * rng.uniform(0.5, 2)
        yhat = (rng.uniform(-0.1, 0.1) * y
                + rng.standard_normal(n) * rng.uniform(0.1, 3)
                + rng.uniform(-2, 2))
        d = engine.r2_decomposition(y, yhat)
        close(d["r2"], d["r2_from_components"], rel=1e-9, abs_=1e-12)
        close(d["r2"], d["information_term"] - d["scale_penalty"]
              - d["bias_penalty"], rel=1e-9, abs_=1e-12)
        # IC^2 is an upper bound, attained only at a = IC and b = 0
        assert d["r2"] <= d["information_term"] + 1e-12


def test_ic_squared_is_the_attainable_ceiling():
    """Rescaling to a = IC and b = 0 turns any prediction's R^2 into IC^2."""
    rng = np.random.default_rng(12)
    n = 200_000
    y = rng.standard_normal(n)
    yhat = 0.04 * y + rng.standard_normal(n) * 2.0 + 3.0
    d = engine.r2_decomposition(y, yhat)
    assert d["r2"] < -1.0                      # badly scaled and badly biased
    ic = d["ic"]
    rescaled = (yhat - yhat.mean()) / yhat.std() * ic * y.std() + y.mean()
    close(engine.r2_decomposition(y, rescaled)["r2"], ic**2, rel=1e-6)
    close(engine.optimal_prediction_scale(ic), ic)
    close(engine.r2_upper_bound(ic), ic**2)


def test_ols_is_the_case_where_r2_equals_ic_squared():
    """OLS with an intercept sets a = IC and b = 0 by construction."""
    rng = np.random.default_rng(13)
    n = 100_000
    x = rng.standard_normal(n)
    y = 0.05 * x + rng.standard_normal(n)
    beta = np.cov(x, y, ddof=0)[0, 1] / x.var()
    fitted = y.mean() + beta * (x - x.mean())
    d = engine.r2_decomposition(y, fitted)
    close(d["scale"], d["ic"], rel=1e-9)
    close(d["bias"], 0.0, abs_=1e-12)
    close(d["r2"], d["ic"] ** 2, rel=1e-9)


def test_r2_from_ir_round_trips_within_its_own_convention():
    """The approximate bridge inverts itself.

    This only establishes internal consistency of R^2 ~ IR^2/BR, because it
    defines IR = IC sqrt(BR) and then verifies the same substitution. It is
    kept for that narrow purpose and must not be read as evidence that the
    bridge is exact: see the exact-model test below.
    """
    for ic in (0.005, 0.0111, 0.05):
        for br in (252.0, 252 * 100, 252 * 500):
            ir = ic * np.sqrt(br)
            close(engine.r2_from_ir(ir, br), engine.r2_from_ic(ic))
            close(engine.ir_from_r2(engine.r2_from_ic(ic), br), ir)
            close(engine.breadth_for_r2_and_ir(engine.r2_from_ic(ic), ir), br)


def test_the_exact_bridge_is_r2_over_one_plus_r2():
    """Under the model, IR^2/BR = R^2/(1+R^2), and the gap is O(rho^4)."""
    for rho in (0.005, 0.05, 0.3, 0.7):
        br = 252.0
        ir = engine.annualised_ir(rho, 1.0, 1, periods_per_year=br)
        close(ir**2 / br, rho**2 / (1 + rho**2))
        # the exact inverse recovers rho^2 where the approximation does not
        close(engine.r2_from_ir_exact(ir, br), rho**2)
        close(engine.ir_from_r2_exact(rho**2, br), ir)
    # the approximation is excellent at small IC and visibly wrong at large IC
    br = 252.0
    small = engine.annualised_ir(0.005, 1.0, 1, periods_per_year=br)
    assert abs(engine.r2_from_ir(small, br) / 0.005**2 - 1.0) < 1e-4
    big = engine.annualised_ir(0.7, 1.0, 1, periods_per_year=br)
    assert abs(engine.r2_from_ir(big, br) / 0.7**2 - 1.0) > 0.3


def test_equity_floor_is_roughly_a_sharpe_two_book():
    """The thread's 0.556% floor is an IR near 2 across 500 names daily.

    This is why the corresponding R^2 of 3.1e-5 looks like a failed model and is
    not one: R^2 carries no information about breadth, and breadth is the whole
    difference between hopeless and excellent here.
    """
    ic = 0.0055556
    br = 252 * 500
    close(engine.ir_from_r2(engine.r2_from_ic(ic), br), 1.9720, rel=1e-3)
    close(engine.r2_from_ic(ic), 3.0864e-5, rel=1e-3)
    # the same R^2 on a single name is worthless
    close(engine.ir_from_r2(3.0864e-5, 252.0), 0.0882, rel=1e-3)


def test_r2_is_quadratic_so_doubling_ic_quadruples_it():
    for ic in (0.005, 0.02):
        close(engine.r2_from_ic(2 * ic) / engine.r2_from_ic(ic), 4.0)


# --- which sigma: total or residual ---------------------------------------

def test_residual_floor_is_always_below_one():
    """a/sqrt(1+a^2) < 1 for every finite a, however large the cost."""
    for a_target in (0.1, 1.0, 1.34, 5.0, 100.0):
        cost = a_target * 3.0 * 0.025 * 1.0
        f = engine.rho_floor_residual(cost, 0.025, 1.0)
        assert 0.0 < f < 1.0
        close(f, a_target / np.sqrt(1 + a_target**2))


def test_residual_and_total_readings_agree_for_small_floors():
    """They differ by a^3/2, so the thread's own examples are unaffected."""
    for cost, sigma, tau in ((5e-4, 0.03, 1.0), (10e-4, 0.015, 1.0),
                             (0.2e-4, 0.003, 1 / 1440)):
        a = engine.rho_floor(cost, sigma, tau)
        r = engine.rho_floor_residual(cost, sigma, tau)
        assert abs(r / a - 1) < 0.6 * a**2, f"a={a}"


def test_unreachable_matches_the_move_size_test():
    """Unreachable iff the kappa-sd move is smaller than the cost, iff a > 1."""
    sigma, tau = 0.025, 1 / 86400
    for cost_bps in (1.0, 2.5, 2.552, 3.411, 10.0):
        cost = cost_bps * 1e-4
        move = engine.kappa_sigma_move(sigma, tau)
        a = engine.rho_floor(cost, sigma, tau)
        assert engine.is_unreachable(cost, sigma, tau) == (move < cost)
        assert engine.is_unreachable(cost, sigma, tau) == (a > 1.0)
    # the BTC 1-second cell quoted in README section 12
    close(engine.kappa_sigma_move(0.025, 1 / 86400) / 1e-4, 2.552, rel=1e-3)
    assert engine.is_unreachable(3.411e-4, 0.025, 1 / 86400)
    close(engine.rho_floor_residual(3.411e-4, 0.025, 1 / 86400), 0.8008,
          rel=1e-3)


# --- one-factor consistency: two correlations, not one --------------------

def test_pairwise_and_factor_correlations_are_not_interchangeable():
    for q in (0.3, 0.7, 0.9):
        close(engine.pairwise_from_factor_corr(q), q**2)
        close(engine.factor_corr_from_pairwise(q**2), q)
    # a factor correlation of 0.70 implies a pairwise correlation of 0.49
    close(engine.pairwise_from_factor_corr(0.70), 0.49)


def test_dispersion_is_consistent_between_the_two_parameterisations():
    total = 0.06
    for q in (0.3, 0.7, 0.9):
        close(engine.dispersion_from_factor_corr(total, q),
              engine.dispersion_from_pairwise_corr(total, q**2))
    close(engine.dispersion_from_factor_corr(0.06, 0.70), 0.0428486, rel=1e-5)
    close(engine.dispersion_from_pairwise_corr(0.06, 0.70), 0.0328634, rel=1e-5)


def test_breadth_and_dispersion_use_the_same_correlation():
    """The bug this pins: 0.70 used as factor corr for dispersion and as
    pairwise corr for breadth at the same time."""
    total, n = 0.06, 200
    pw = 0.49  # consistent with a 0.70 factor correlation
    disp = engine.dispersion_from_pairwise_corr(total, pw)
    close(disp, engine.dispersion_from_factor_corr(total, 0.70))
    bd = engine.breadth_common_position(n, pw)
    close(bd, 2.030, rel=1e-3)
    close(np.sqrt(engine.breadth_neutral(n) / bd), 9.90, rel=1e-3)
    # the inconsistent pairing that was there before
    bd_wrong = engine.breadth_common_position(n, 0.70)
    close(np.sqrt(engine.breadth_neutral(n) / bd_wrong), 11.82, rel=1e-3)
    # the conclusion survives the correction: still order 10x
    for pairwise in (0.3, 0.4, 0.49, 0.55, 0.7, 0.8):
        ratio = np.sqrt(engine.breadth_neutral(n)
                        / engine.breadth_common_position(n, pairwise))
        assert 7.0 < ratio < 13.0


# --- cross-sectional IC ----------------------------------------------------

def test_dispersion_from_factor_corr():
    for corr in (0.0, 0.5, 0.7, 0.9):
        d = engine.dispersion_from_factor_corr(0.06, corr)
        close(d, 0.06 * np.sqrt(1 - corr**2))
    close(engine.dispersion_from_factor_corr(0.06, 0.0), 0.06)
    close(engine.dispersion_from_factor_corr(0.06, 1.0), 0.0)


def test_cross_sectional_floor_is_higher_than_directional():
    """Dispersion < total vol, so the per-name bar rises by exactly vol/disp."""
    total, corr, cost, tau = 0.06, 0.7, 13e-4, 1.0
    disp = engine.dispersion_from_factor_corr(total, corr)
    directional = engine.rho_floor(cost, total, tau)
    cs = engine.cross_sectional_floor(cost, disp, tau)
    assert cs > directional
    close(cs / directional, total / disp)
    close(total / disp, 1 / np.sqrt(1 - corr**2))


def test_breadth_neutral_beats_directional_in_a_correlated_market():
    n, corr = 200, 0.7
    bd = engine.breadth_common_position(n, corr)
    bn = engine.breadth_neutral(n)
    close(bd, n / (1 + (n - 1) * corr))
    close(bn, n - 1)
    close(np.sqrt(bn / bd), 11.815, rel=1e-3)
    # a directional book cannot buy breadth by adding names
    assert engine.breadth_common_position(10_000, corr) < 1 / corr + 1e-6
    # and with uncorrelated returns the two agree
    close(engine.breadth_common_position(n, 0.0), float(n))


def test_portfolio_breakeven_is_where_gross_equals_cost():
    cost, disp, tau, turn = 13e-4, 0.0428, 1.0, 1.414
    ic = engine.portfolio_breakeven_ic(cost, disp, tau, turnover=turn)
    gross = ic * disp * np.sqrt(tau) / engine.SQRT_2_OVER_PI  # beta*sqrt(pi/2)
    close(gross, 0.5 * cost * turn)
    # one full round trip is kappa/sqrt(pi/2) times the per-name floor
    full = engine.portfolio_breakeven_ic(cost, disp, tau, turnover=2.0)
    close(full / engine.cross_sectional_floor(cost, disp, tau),
          3.0 * engine.SQRT_2_OVER_PI)
    # and reproduces the figure quoted in README section 14
    close(ic, 0.01714, rel=1e-3)


def _panel(ic_ts, ic_cs, n_dates, n_assets, rng, sigma_m=0.042, sigma_d=0.0428):
    g = rng.standard_normal((n_dates, 1))
    c = rng.standard_normal((n_dates, n_assets))
    c -= c.mean(axis=1, keepdims=True)
    c /= c.std(axis=1, keepdims=True)
    x = (g + c) / np.sqrt(2.0)
    b_ts = engine.beta_exact(ic_ts, sigma_m, 1.0) if ic_ts else 0.0
    b_cs = engine.beta_exact(ic_cs, sigma_d, 1.0) if ic_cs else 0.0
    y = (b_ts * g + sigma_m * rng.standard_normal((n_dates, 1))
         + b_cs * c + sigma_d * rng.standard_normal((n_dates, n_assets)))
    return x, y


def test_ic_decomposition_recovers_both_components():
    rng = np.random.default_rng(3)
    x, y = _panel(0.04, 0.03, 40_000, 200, rng)
    d = engine.panel_ic_decomposition(x, y)
    assert abs(d["time_series_ic"] - 0.04) < 4 * d["time_series_ic_se"]
    assert abs(d["cross_sectional_ic"] - 0.03) < 4 * d["cross_sectional_ic_se"]
    # the pooled correlation is neither of them
    assert not np.isclose(d["pooled_ic"], 0.04, atol=3e-3)
    assert not np.isclose(d["pooled_ic"], 0.03, atol=3e-3)


def test_ic_decomposition_is_orthogonal():
    """A timing-only panel scores zero cross-sectionally, and vice versa.

    Checked over several seeds because a single panel's estimate is noisy: an
    earlier one-seed reading looked significant and was not.
    """
    for ic_ts, ic_cs, key in ((0.04, 0.0, "cross_sectional_ic"),
                              (0.0, 0.04, "time_series_ic")):
        vals = []
        for seed in range(8):
            rng = np.random.default_rng(400 + seed)
            x, y = _panel(ic_ts, ic_cs, 8_000, 200, rng)
            vals.append(engine.panel_ic_decomposition(x, y)[key])
        vals = np.array(vals)
        t = vals.mean() / (vals.std(ddof=1) / np.sqrt(vals.size))
        assert abs(t) < 3.0, f"{key} biased away from zero, t = {t:.2f}"
        # and the component that should be there is unmistakably there
        assert abs(vals).max() < 0.02


def test_cs_ic_is_n_times_cheaper_to_establish():
    """The ratio is N(1 - IC^2)^2, i.e. N to within IC^2 -- not exactly N.

    obs_for_tstat carries Fisher's (1 - rho^2) factor and dates_for_cs_ic does
    not, so the two differ in the fourth significant figure at IC = 1% and the
    third at IC = 5%. Worth pinning rather than rounding away.
    """
    for ic in (0.01, 0.02, 0.05):
        for n in (50, 200):
            ts = engine.obs_for_tstat(ic, 3.0)
            cs = engine.dates_for_cs_ic(ic, n, 3.0)
            close(ts / cs, n * (1 - ic**2) ** 2 + n * ic**2 / 9.0, rel=1e-6)
            close(ts / cs, float(n), rel=1.1 * ic**2 * 2)
    # a measured date-to-date dispersion overrides the 1/sqrt(N) assumption
    loose = engine.dates_for_cs_ic(0.02, 200, 3.0, ic_sd=0.15)
    tight = engine.dates_for_cs_ic(0.02, 200, 3.0)
    assert loose > tight * 4


# --- fees and carry together ----------------------------------------------

def test_optimal_horizon_maximises_net_edge():
    """tau* = (kappa rho sigma / 2f)^2 really is the argmax."""
    sigma, cost, carry = 0.025, 10.011e-4, 3e-4
    for rho in (0.02, 0.05, 0.10):
        star = engine.optimal_horizon_with_carry(rho, sigma, carry)
        peak = engine.net_edge_with_carry(rho, sigma, star, cost, carry)
        grid = np.linspace(0.05 * star, 4 * star, 4001)
        best = engine.net_edge_with_carry(rho, sigma, grid, cost, carry).max()
        close(peak, best, rel=1e-6)
        # and the peak value matches its closed form
        close(peak, (3 * rho * sigma) ** 2 / (4 * carry) - cost)


def test_carry_floor_is_exactly_where_the_peak_touches_zero():
    sigma, cost, carry = 0.025, 10.011e-4, 3e-4
    rho = engine.rho_floor_with_carry(cost, sigma, carry)
    star = engine.optimal_horizon_with_carry(rho, sigma, carry)
    close(engine.net_edge_with_carry(rho, sigma, star, cost, carry), 0.0,
          abs_=1e-18)
    # just below the floor nothing works at any horizon
    grid = np.logspace(-4, 4, 20_000)
    worse = engine.net_edge_with_carry(0.98 * rho, sigma, grid, cost, carry)
    assert worse.max() < 0
    # just above it, something does
    better = engine.net_edge_with_carry(1.02 * rho, sigma, grid, cost, carry)
    assert better.max() > 0


def test_carry_floor_matches_the_vip0_figure():
    """The 1.46% quoted in README section 13."""
    close(engine.rho_floor_with_carry(10.011e-4, 0.025, 3e-4), 0.014614,
          rel=1e-4)
    close(engine.optimal_horizon_with_carry(0.014614, 0.025, 3e-4), 3.337,
          rel=1e-3)


def test_no_carry_means_no_absolute_floor():
    """With f = 0 every positive correlation clears the fee by holding longer."""
    sigma, cost = 0.025, 10.011e-4
    assert engine.rho_floor_with_carry(cost, sigma, 0.0) == 0.0
    assert engine.optimal_horizon_with_carry(0.05, sigma, 0.0) == float("inf")
    for rho in (0.0005, 0.005, 0.05):
        tau = engine.min_horizon_fee_only(cost, sigma, rho)
        close(engine.net_edge_with_carry(rho, sigma, tau, cost, 0.0), 0.0,
              abs_=1e-18)
        assert engine.net_edge_with_carry(rho, sigma, 1.01 * tau, cost, 0.0) > 0


def test_min_horizon_fee_only_inverts_the_source_floor():
    sigma, cost = 0.025, 10.011e-4
    for rho in (0.01, 0.05, 0.2):
        tau = engine.min_horizon_fee_only(cost, sigma, rho)
        close(engine.rho_floor(cost, sigma, tau, 3.0), rho)


def test_fee_wedge_is_quadratic_in_horizon():
    """A k-fold higher fee costs k^2 in holding period at fixed correlation."""
    sigma, rho = 0.025, 0.05
    a = engine.min_horizon_fee_only(10.011e-4, sigma, rho)
    b = engine.min_horizon_fee_only(3.411e-4, sigma, rho)
    close(a / b, (10.011 / 3.411) ** 2, rel=1e-9)


# --- the fundamental law ---------------------------------------------------

def test_correlation_is_the_per_bet_sharpe():
    for rho in (0.001, 0.01, 0.1):
        close(engine.sharpe_proportional_per_period(rho),
              rho / np.sqrt(1 + rho**2))
        close(engine.sharpe_proportional_per_period(rho), rho, rel=rho**2 + 1e-9)


def test_per_bet_sharpe_matches_simulated_payoff():
    """The Sharpe formula is checked against the payoff distribution itself.

    The previous version of this test compared the engine with the closed form
    it was derived from, so it could not detect a wrong closed form. This one
    simulates h y = beta x^2 + v x eps and measures the Sharpe directly. It is
    the test that would have caught rho/sqrt(1 + 2 rho^2).
    """
    rng = np.random.default_rng(20260726)
    n = 4_000_000
    for beta, v in ((0.30, 1.0), (0.60, 1.0), (1.00, 1.0)):
        x = rng.standard_normal(n)
        eps = rng.standard_normal(n)
        payoff = x * (beta * x + v * eps)
        measured = payoff.mean() / payoff.std()
        rho = beta / np.sqrt(beta**2 + v**2)
        close(engine.sharpe_proportional_per_period(rho), measured, rel=3e-3)
        # and the same number through the loading-to-noise argument
        close(engine.sharpe_from_loading_to_noise(beta / v), measured, rel=3e-3)


def test_the_two_sharpe_parameterisations_agree_under_the_exact_map():
    """SR(rho) = SR_r(r) exactly when r = rho/sqrt(1-rho^2), not when r = rho."""
    for rho in (0.05, 0.3, 0.7, 0.9):
        r = rho / np.sqrt(1 - rho**2)
        close(engine.sharpe_proportional_per_period(rho),
              engine.sharpe_from_loading_to_noise(r))
    # substituting rho for r is the error the audit found: 15% at rho = 0.71
    rho = 0.7071067811865476
    wrong = engine.sharpe_from_loading_to_noise(rho)
    right = engine.sharpe_proportional_per_period(rho)
    assert abs(wrong / right - 1.0) > 0.13


def test_ir_scales_as_root_breadth():
    a = engine.annualised_ir(0.01, 1.0, 1)
    b = engine.annualised_ir(0.01, 1.0, 100)
    close(b / a, 10.0)


def test_effective_breadth_asymptote():
    for corr in (0.02, 0.1, 0.3):
        assert engine.effective_breadth(10_000_000, corr) < 1 / corr
        close(engine.effective_breadth(10_000_000, corr), 1 / corr, rel=1e-4)
    close(engine.effective_breadth(500, 0.0), 500.0)


def test_obs_for_tstat():
    """A 0.556% correlation needs ~292k observations for t = 3."""
    n = engine.obs_for_tstat(0.00555556, 3.0)
    close(n, 291583.0, rel=1e-3)
    # and the standard error implied by that n reproduces the t stat
    rho = 0.00555556
    se = (1 - rho**2) / np.sqrt(n - 1)
    close(rho / se, 3.0, rel=1e-6)


# --- simulators ------------------------------------------------------------

def test_simulated_alpha_hits_its_target_correlation():
    rng = np.random.default_rng(0)
    rho = 2 * EQUITY.floor(KAPPA)
    x, y, beta = engine.simulate_returns_and_alpha(EQUITY, rho, 2_000_000, rng)
    hat = np.corrcoef(x, y)[0, 1]
    se = 1 / np.sqrt(2_000_000)
    assert abs(hat - rho) < 5 * se, f"{hat} vs {rho} (se {se})"
    close(np.var(x), 1.0, rel=0.01)


def test_sparse_signal_has_unit_variance_and_the_same_correlation():
    rng = np.random.default_rng(1)
    rho = EQUITY.floor(KAPPA)
    for p in (0.5, 0.05, 0.01):
        x, y, beta = engine.simulate_returns_and_alpha(
            EQUITY, rho, 2_000_000, rng, signal="sparse", sparsity=p
        )
        close(np.var(x), 1.0, rel=0.03)
        hat = np.corrcoef(x, y)[0, 1]
        assert abs(hat - rho) < 6 / np.sqrt(2_000_000 * p), f"p={p}: {hat}"


def test_banded_backtest_matches_the_closed_form():
    rng = np.random.default_rng(2)
    rho = 2 * EQUITY.floor(KAPPA)
    x, y, beta = engine.simulate_returns_and_alpha(EQUITY, rho, 2_000_000, rng)
    bt = engine.banded_backtest(x, y, beta, EQUITY.cost)
    k = EQUITY.cost / beta
    close(bt["trade_frequency"], engine.trade_frequency(k), rel=0.02)
    want = engine.net_pnl_per_period(beta, EQUITY.cost)
    se = bt["net_sd"] / np.sqrt(2_000_000)
    assert abs(bt["net_mean"] - want) < 5 * se


def test_sparse_beats_gaussian_by_the_predicted_factor():
    """The 109x claim, from the closed form and from simulation."""
    rho = EQUITY.floor(KAPPA)
    beta = engine.beta_source(rho, 0.03, 1.0)
    gauss = engine.net_pnl_per_period(beta, EQUITY.cost)
    p_opt = 1 / (4 * KAPPA**2)
    sparse = beta / (4 * KAPPA)
    close(sparse, p_opt * (beta / np.sqrt(p_opt) - EQUITY.cost), rel=1e-9)
    close(sparse / gauss, 1 / (4 * KAPPA * engine.trunc_mean(KAPPA)), rel=1e-9)
    assert 105 < sparse / gauss < 113

# --- covariance-based dispersion and breadth (audit items 7 and 8) ---------

def test_general_forms_reduce_to_the_equicorrelation_closed_forms():
    """tr(M S M)/(N-1) and N^2/(1'C1) must agree with the scalar formulas."""
    for n, rho, sig in ((50, 0.30, 0.02), (200, 0.55, 0.04), (10, 0.05, 0.01)):
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.0)
        cov = corr * sig**2
        close(engine.effective_breadth_from_corr(corr),
              engine.effective_breadth(n, rho))
        close(engine.expected_cs_dispersion(cov),
              engine.dispersion_from_pairwise_corr(sig, rho))
        close(engine.mean_offdiagonal(corr), rho)
        # equal variances and equicorrelation attain the independence bound
        close(engine.residual_participation_ratio(cov), engine.breadth_neutral(n))


def test_mean_not_median_is_the_breadth_sufficient_statistic():
    """A bimodal pairwise distribution: the median gives the wrong breadth.

    Half the pairs at 0.8 and half at 0.0 puts the median in the trough. Only
    the mean reproduces the breadth computed from the matrix itself.
    """
    n = 40
    half = n // 2
    corr = np.eye(n)
    # two blocks, tightly correlated inside and uncorrelated across
    corr[:half, :half] = 0.8
    corr[half:, half:] = 0.8
    np.fill_diagonal(corr, 1.0)
    from_matrix = engine.effective_breadth_from_corr(corr)
    mean_rho = engine.mean_offdiagonal(corr)
    off = corr[~np.eye(n, dtype=bool)]
    median_rho = float(np.median(off))
    close(engine.effective_breadth(n, mean_rho), from_matrix)
    # the median sits in the trough and misprices breadth badly
    assert abs(engine.effective_breadth(n, median_rho) / from_matrix - 1.0) > 0.3


def test_expected_cs_variance_matches_simulation():
    """The tr(M S M)/(N-1) formula against sampled cross-sectional variance."""
    rng = np.random.default_rng(11)
    n = 25
    a = rng.standard_normal((n, n)) * 0.3
    cov = a @ a.T / n + np.diag(rng.uniform(0.01, 0.05, n))
    draws = rng.multivariate_normal(np.zeros(n), cov, size=400_000)
    measured = draws.var(axis=1, ddof=1).mean()
    close(engine.expected_cs_variance(cov), measured, rel=2e-2)


def test_residual_participation_ratio_falls_below_n_minus_1_when_heterogeneous():
    """The spectral diagnostic, which is NOT breadth, drops when risk concentrates.

    Named for what it measures. The participation ratio of M Sigma M says how
    evenly residual risk is spread, and says nothing about how many profitable
    independent forecasts exist: see the docstring of the engine function.
    """
    rng = np.random.default_rng(5)
    n = 30
    load = rng.uniform(0.5, 1.5, n)
    cov = np.outer(load, load) * 0.04 + np.diag(rng.uniform(0.002, 0.05, n))
    assert engine.residual_participation_ratio(cov) < engine.breadth_neutral(n)


def test_clarke_transfer_coefficient_is_a_correlation_not_a_pnl_ratio():
    """TC = Corr(x, h) for h = sign(x)1{|x|>k}, checked against simulation.

    The point of the test is that TC is far larger than either the net-P&L
    retention or the net/gross IR ratio at the same band, so the three cannot be
    used interchangeably.
    """
    rng = np.random.default_rng(4242)
    x = rng.standard_normal(4_000_000)
    for m in (1.0, 1.5, 2.0, 3.0):
        k = engine.band_from_multiple(m, KAPPA)
        h = np.sign(x) * (np.abs(x) > k)
        # At m = 1 the band is k = 3, so only 0.27% of draws are nonzero and the
        # sampled correlation rests on ~10k points. A 2% tolerance is what that
        # supports, and it still separates 17% from 71% by a factor of four,
        # which is the confusion this test exists to catch.
        close(engine.transfer_coefficient_band(m, KAPPA),
              float(np.corrcoef(x, h)[0, 1]), rel=2e-2)
        # closed form
        close(engine.transfer_coefficient_band(m, KAPPA),
              2 * stats.norm.pdf(k) / np.sqrt(2 * stats.norm.sf(k)))
    # TC exceeds P&L retention everywhere: alignment survives, value does not
    for m in (1.0, 1.5, 2.0, 2.5, 3.0, 5.0):
        assert engine.transfer_coefficient_band(m, KAPPA) > engine.gross_capture(m, KAPPA)


def test_return_correlation_is_not_forecast_breadth():
    """Return correlation and strategy breadth come apart completely.

    y_i = beta x_i + gamma f + eps_i with x_i, f, eps_i independent. The common
    factor f makes returns arbitrarily correlated, while the proportional-rule
    P&Ls x_i*y_i stay uncorrelated, so breadth remains n. The
    return-correlation formula collapses to about 1 and is wrong by a factor of
    n. This is why `breadth_common_position` is restricted to books holding the
    same position in every name.
    """
    rng = np.random.default_rng(11)
    n, t, beta, gamma = 40, 200_000, 0.05, 10.0
    x = rng.standard_normal((t, n))
    f = rng.standard_normal((t, 1))
    eps = rng.standard_normal((t, n))
    pnl = x * (beta * x + gamma * f + eps)
    y = beta * x + gamma * f + eps

    iu = np.triu_indices(n, 1)
    rho_ret = np.corrcoef(y, rowvar=False)[iu].mean()
    rho_pnl = np.corrcoef(pnl, rowvar=False)[iu].mean()
    assert rho_ret > 0.95, rho_ret
    assert abs(rho_pnl) < 0.01, rho_pnl

    total = pnl.sum(axis=1)
    sr_one = np.mean([pnl[:, i].mean() / pnl[:, i].std() for i in range(n)])
    breadth_true = (total.mean() / total.std() / sr_one) ** 2
    assert breadth_true > 0.8 * n, breadth_true

    # the return-correlation formula is off by roughly a factor of n here
    from_returns = engine.breadth_common_position(n, float(rho_ret))
    assert from_returns < 1.3, from_returns
    assert breadth_true / from_returns > 20


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    raise SystemExit(1 if failed else 0)
