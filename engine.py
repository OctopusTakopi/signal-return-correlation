"""Closed-form analytics and simulators for the signal-correlation cost floor.

Model (source, tweet 2/9):

    y = beta * x + sigma * sqrt(tau) * eps

with Var(x) = Var(eps) = 1 and Cov(x, eps) = 0. `sigma` is a *daily*
volatility and `tau` is a horizon *in days*, so `sigma * sqrt(tau)` is the
residual volatility over the forecast horizon.

Everything named `*_source` reproduces an expression the thread states.
Everything else is this project's formalisation and is documented as such in
README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

SQRT_2_OVER_PI = float(np.sqrt(2.0 / np.pi))
_PHI = stats.norm.pdf
_Q = stats.norm.sf  # upper tail 1 - Phi
_PPF = stats.norm.ppf


# --------------------------------------------------------------------------
# 1. The source's algebra
# --------------------------------------------------------------------------

def corr_exact(beta: float, sigma: float, tau: float) -> float:
    """Corr(x, y) with no approximation: beta / sqrt(beta^2 + sigma^2 tau)."""
    return beta / np.sqrt(beta**2 + sigma**2 * tau)


def corr_source(beta: float, sigma: float, tau: float) -> float:
    """The thread's expression: beta / (sigma sqrt(tau)).

    Drops the beta^2 term from Var(y). Overstates |rho| by O(rho^2).
    """
    return beta / (sigma * np.sqrt(tau))


def beta_source(rho: float, sigma: float, tau: float) -> float:
    """Invert `corr_source`: beta = rho sigma sqrt(tau) (tweet 3/9)."""
    return rho * sigma * np.sqrt(tau)


def beta_exact(rho: float, sigma: float, tau: float) -> float:
    """Invert `corr_exact`: beta = rho sigma sqrt(tau) / sqrt(1 - rho^2)."""
    return rho * sigma * np.sqrt(tau) / np.sqrt(1.0 - rho**2)


def r2_from_ic(ic: float) -> float:
    """Population R² of a one-predictor OLS regression with an intercept.

    For any finite-variance `x` and `y`, simple-regression R² is Corr(x, y)².
    R² therefore preserves signal strength but loses the sign of the loading.
    """
    return ic**2


def ic_magnitude_from_r2(r2: float) -> float:
    """Recover |IC| from R²; the loading sign must be stored separately."""
    if not 0.0 <= r2 <= 1.0:
        raise ValueError("r2 must lie in [0, 1]")
    return float(np.sqrt(r2))


def r2_exact(beta: float, sigma_residual: float, tau: float) -> float:
    """Population R² for y = intercept + beta*x + residual.

    Here Var(x)=1 and the horizon residual variance is
    `sigma_residual**2 * tau`.
    """
    return beta**2 / (beta**2 + sigma_residual**2 * tau)


def beta_from_r2(r2: float, sigma_residual: float, tau: float,
                 sign: float = 1.0) -> float:
    """Loading implied by R² under the residual-volatility model.

    R² does not identify the sign, so `sign` supplies it. A perfect score would
    require an infinite loading relative to a positive residual variance.
    """
    if not 0.0 <= r2 < 1.0:
        raise ValueError("r2 must lie in [0, 1)")
    magnitude = sigma_residual * np.sqrt(tau) * np.sqrt(r2 / (1.0 - r2))
    return float(np.copysign(magnitude, sign))


def rho_floor(cost: float, sigma: float, tau: float, kappa: float = 3.0) -> float:
    """The source's correlation floor: c / (kappa sigma sqrt(tau)).

    `kappa` is the "how many standard deviations of signal must fail to pay
    for the trade" convention. The thread fixes kappa = 3 (tweet 4/9).

    WHICH SIGMA. The value returned depends on what `sigma` is, and the two
    readings diverge exactly where the floor approaches 1:

      total volatility   Var(y) = sigma^2 tau, so rho = beta/(sigma sqrt(tau))
                         holds EXACTLY and this function is the exact floor. A
                         return above 1 then means no correlation satisfies the
                         kappa-sigma criterion, because a kappa-sd move is
                         smaller than the cost. It does NOT mean trading loses:
                         see `fails_kappa_criterion`. This is the reading to use
                         with a measured volatility, and it is what the
                         application sections do.

      residual volatility  the thread's literal model y = beta x + sigma sqrt(tau)
                         eps. Then rho = beta/sqrt(beta^2 + sigma^2 tau) and this
                         function is an approximation; the exact floor is
                         `rho_floor_residual`, which is always below 1.

    The two agree to O(rho^2), which is why the distinction is invisible for the
    thread's own examples and decisive for sub-minute crypto.
    """
    return cost / (kappa * sigma * np.sqrt(tau))


def rho_floor_residual(cost: float, sigma_residual: float, tau: float,
                       kappa: float = 3.0) -> float:
    """Exact floor when `sigma_residual` is the *residual* volatility.

    Actionability is kappa*beta >= c with the exact loading
    beta = rho sigma sqrt(tau)/sqrt(1 - rho^2), so

        rho/sqrt(1 - rho^2) >= a,   a = c/(kappa sigma sqrt(tau))
        rho >= a/sqrt(1 + a^2),

    which lies strictly below 1 for every finite a. Verified symbolically in
    scripts/check_algebra.py.
    """
    a = rho_floor(cost, sigma_residual, tau, kappa)
    return a / np.sqrt(1.0 + a**2)


def kappa_sigma_move(sigma_total: float, tau: float, kappa: float = 3.0) -> float:
    """Size of a kappa-standard-deviation price move over the horizon.

    The parameterisation-free half of the floor: if this is smaller than the
    round-trip cost then no correlation, however high, lets a kappa-sigma
    reading of the signal pay for the trade it implies. No correlation appears
    in the test.
    """
    return kappa * sigma_total * np.sqrt(tau)


def fails_kappa_criterion(cost: float, sigma_total: float, tau: float,
                          kappa: float = 3.0) -> bool:
    """True when no correlation satisfies the kappa-sigma relevance criterion.

    This is a statement about the *criterion*, not about profitability. Under
    the Gaussian model the band rule still has strictly positive expected net
    payoff at any positive correlation, because E[(|x|-k)^+] > 0 for every
    finite k: see `trunc_mean` and Section 5 of the report. What this predicate
    detects is that the kappa-sigma reading the source treats as the relevant
    case cannot cover the cost, so the surviving opportunity lives further out
    in the tail and is correspondingly rare.

    `is_unreachable` remains available as a deprecated alias.
    """
    return kappa_sigma_move(sigma_total, tau, kappa) < cost


def is_unreachable(cost: float, sigma_total: float, tau: float,
                   kappa: float = 3.0) -> bool:
    """Deprecated alias for `fails_kappa_criterion`.

    The old name invited the reading that perfect foresight loses money here,
    which does not follow from the model. Retained so existing callers and
    result artefacts keep working.
    """
    return fails_kappa_criterion(cost, sigma_total, tau, kappa)


def r2_floor_total(cost: float, sigma_total: float, tau: float,
                   kappa: float = 3.0) -> float:
    """Required R² when `sigma_total` is measured total return volatility.

    Values above one are deliberately not clipped: they mean no regression
    score clears the kappa-sigma relevance criterion at that horizon. As with
    `fails_kappa_criterion`, that is a statement about the criterion and not a
    proof that no profitable trade exists.
    """
    return r2_from_ic(rho_floor(cost, sigma_total, tau, kappa))


def r2_floor_residual(cost: float, sigma_residual: float, tau: float,
                      kappa: float = 3.0) -> float:
    """Exact required R² when `sigma_residual` is residual volatility."""
    return r2_from_ic(
        rho_floor_residual(cost, sigma_residual, tau, kappa)
    )


def r2_odds(r2: float) -> float:
    """Explained-to-unexplained variance, R²/(1-R²).

    Under the residual-volatility model these odds equal
    beta²/(sigma_residual²*tau), so their ratio is the squared loading multiple.
    """
    if not 0.0 <= r2 < 1.0:
        raise ValueError("r2 must lie in [0, 1)")
    return r2 / (1.0 - r2)


def loading_multiple_from_r2(r2: float, floor_r2: float) -> float:
    """Loading multiple above a residual-model R² floor."""
    if not 0.0 < floor_r2 < 1.0:
        raise ValueError("floor_r2 must lie in (0, 1)")
    return float(np.sqrt(r2_odds(r2) / r2_odds(floor_r2)))


def fit_simple_regression(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Fit `y = intercept + slope*x` by ordinary least squares."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or x.size < 2:
        raise ValueError("x and y must be same-length one-dimensional arrays")
    x_centered = x - x.mean()
    denominator = float(x_centered @ x_centered)
    if denominator == 0.0:
        raise ValueError("x must have positive variance")
    slope = float((x_centered @ (y - y.mean())) / denominator)
    intercept = float(y.mean() - slope * x.mean())
    return intercept, slope


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; valid out of sample and possibly negative."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if (y_true.ndim != 1 or y_pred.ndim != 1
            or y_true.shape != y_pred.shape or y_true.size < 2):
        raise ValueError(
            "y_true and y_pred must be same-length one-dimensional arrays"
        )
    centered = y_true - y_true.mean()
    total = float(centered @ centered)
    if total == 0.0:
        raise ValueError("y_true must have positive variance")
    residual = y_true - y_pred
    return 1.0 - float(residual @ residual) / total


# --------------------------------------------------------------------------
# 2. This project's trading rule and its closed form
# --------------------------------------------------------------------------
# Each period, observe x, forecast E[y|x] = beta*x, and take a unit position
# in the direction of the forecast iff the forecast covers the round-trip
# cost c: |beta*x| > c. Otherwise stay flat. With k = c/beta this is the
# no-trade band |x| <= k.

def trunc_mean(k: float) -> float:
    """E[(|x| - k)^+] for x ~ N(0,1) = 2 (phi(k) - k Q(k)).

    Strictly positive for every finite k, but underflows to exactly 0.0 in
    float64 above k ~= 39 (where the true value is below 1e-320). A returned
    zero therefore means "smaller than float64 can hold", not "no edge"; see
    tests/test_engine.py::test_trunc_mean_underflow_boundary.
    """
    return 2.0 * (_PHI(k) - k * _Q(k))


def trunc_second_moment(k: float) -> float:
    """E[((|x| - k)^+)^2] for x ~ N(0,1) = 2 ((1 + k^2) Q(k) - k phi(k))."""
    return 2.0 * ((1.0 + k**2) * _Q(k) - k * _PHI(k))


def trade_frequency(k: float) -> float:
    """P(|x| > k) = 2 Q(k): fraction of periods with a trade to do."""
    return 2.0 * _Q(k)


def band_from_multiple(multiple: float, kappa: float = 3.0) -> float:
    """No-trade band k when rho = multiple * rho_floor.

    k = c/beta = c/(rho sigma sqrt(tau)) and c = kappa rho_floor sigma
    sqrt(tau), so k = kappa / multiple. Independent of sigma, tau and c.
    """
    return kappa / multiple


def multiple_from_frequency(freq: float, kappa: float = 3.0) -> float:
    """Inverse of `trade_frequency o band_from_multiple`.

    The multiple of the floor that a given trade frequency requires.
    """
    return kappa / _PPF(1.0 - freq / 2.0)


def gross_pnl_per_period(beta: float) -> float:
    """E[beta |x|] with no cost and no band: beta sqrt(2/pi)."""
    return beta * SQRT_2_OVER_PI


def gross_pnl_banded(beta: float, k: float) -> float:
    """E[beta |x| 1{|x| > k}] = beta * 2 phi(k). Gross P&L inside the band rule."""
    return beta * 2.0 * _PHI(k)


def turnover_iid(freq: float) -> float:
    """E|h_t - h_{t-1}| for the band rule on an iid signal: 2f - f^2.

    Positions are {-1, 0, +1} with P(0) = 1-f. Consecutive draws are
    independent, so a change of size 1 happens with probability 2f(1-f) and a
    flip of size 2 with probability f^2/2.
    """
    return 2.0 * freq - freq**2


# Two cost conventions appear in this project and they do not agree. Both are
# defensible; keep them distinct.
#
#   "round trip per period held"  cost = c * f
#       You liquidate at the end of every period. This is the convention
#       implied by the source's single-period comparison beta*x vs c, and it is
#       what `net_pnl_per_period` uses.
#
#   "charge only position changes"  cost = (c/2) * E|h_t - h_{t-1}|
#       You keep the position when the next period wants the same one. For an
#       iid signal this is c * f * (1 - f/2): cheaper than the first convention
#       by f/2 even with no signal persistence at all. This is the convention
#       the persistence experiment uses, because it is the only one in which
#       autocorrelation can show up.
#
# `net_pnl_turnover` implements the second for an iid signal so the two can be
# reconciled analytically at phi = 0.

def net_pnl_turnover_iid(beta: float, cost: float) -> float:
    """Band-rule net P&L under the position-change cost convention, iid signal."""
    if beta <= 0:
        return 0.0
    k = cost / beta
    f = trade_frequency(k)
    return gross_pnl_banded(beta, k) - 0.5 * cost * turnover_iid(f)


def net_pnl_per_period(beta: float, cost: float) -> float:
    """E[(beta|x| - c) 1{beta|x| > c}] = beta E[(|x| - k)^+]."""
    if beta <= 0:
        return 0.0
    return beta * trunc_mean(cost / beta)


def net_pnl_variance_per_period(
    beta: float, cost: float, sigma: float, tau: float
) -> float:
    """Var of the banded strategy's per-period P&L.

    P&L = 1{|x|>k} (beta|x| - c + sigma sqrt(tau) sgn(x) eps), so
    E[P&L^2] = beta^2 E[((|x|-k)^+)^2] + sigma^2 tau P(|x|>k).
    """
    if beta <= 0:
        return 0.0
    k = cost / beta
    second = beta**2 * trunc_second_moment(k) + sigma**2 * tau * trade_frequency(k)
    mean = net_pnl_per_period(beta, cost)
    return second - mean**2


def net_sharpe_per_period(
    beta: float, cost: float, sigma: float, tau: float
) -> float:
    var = net_pnl_variance_per_period(beta, cost, sigma, tau)
    if var <= 0:
        return 0.0
    return net_pnl_per_period(beta, cost) / np.sqrt(var)


def gross_capture(multiple: float, kappa: float = 3.0) -> float:
    """Net P&L as a fraction of the costless, bandless gross P&L.

    Depends only on the multiple of the floor -- beta, c, sigma, tau all
    cancel. This is the "cost cliff": how little of the raw signal value
    survives just above the floor.
    """
    k = band_from_multiple(multiple, kappa)
    return trunc_mean(k) / SQRT_2_OVER_PI


def transfer_coefficient_band(multiple: float, kappa: float = 3.0) -> float:
    """Clarke-de Silva-Thorley transfer coefficient of the no-trade-band rule.

    TC is defined as the correlation between risk-adjusted forecasts and the
    active weights actually held (Clarke, de Silva & Thorley 2002), so it
    measures loss of *alignment* under a constraint and carries no cost term.

    With forecast x and the band rule's weight h = sign(x) 1{|x| > k},

        Cov(x, h) = E[|x| 1{|x| > k}] = 2 phi(k),
        Var(h)    = P(|x| > k) = 2 Q(k),      Var(x) = 1,

    so

        TC = 2 phi(k) / sqrt(2 Q(k)).

    This is NOT the ratio of net to gross expected P&L, and not the ratio of net
    to gross information ratio: those additionally absorb the cost that the band
    exists to avoid. The band keeps most of the alignment even at the floor
    because it never takes a position in the wrong direction; what it loses is
    the proportional sizing.
    """
    k = band_from_multiple(multiple, kappa)
    return float(2.0 * _PHI(k) / np.sqrt(2.0 * _Q(k)))


# --------------------------------------------------------------------------
# 3. Fees and carry together
# --------------------------------------------------------------------------
# The source's floor prices a fixed cost per round trip, which rewards holding
# longer, because the edge available on a kappa-sigma signal grows as sqrt(tau)
# while the fee does not grow at all. A carry -- perpetual funding, borrow, a
# financing spread -- does the opposite: it accrues with time. The two together
# bracket the holding period from both ends:
#
#     net(tau) = kappa rho sigma sqrt(tau) - c - f tau
#
# which is a downward parabola in sqrt(tau). Its peak is at
# tau* = (kappa rho sigma / 2f)^2 and equals (kappa rho sigma)^2/(4f) - c, so it
# is positive for some tau if and only if
#
#     rho > 2 sqrt(f c) / (kappa sigma).
#
# That is a floor on correlation which no choice of horizon can escape, unlike
# the source's floor, which any correlation clears eventually by holding longer.
# Both closed forms are verified symbolically in scripts/vip0_fee_analysis.py.

def net_edge_with_carry(rho: float, sigma: float, tau, cost: float,
                        carry_per_day: float = 0.0, kappa: float = 3.0):
    """Net return per round trip on a kappa-sigma signal, after fee and carry."""
    tau = np.asarray(tau, dtype=float)
    return kappa * rho * sigma * np.sqrt(tau) - cost - carry_per_day * tau


def optimal_horizon_with_carry(rho: float, sigma: float, carry_per_day: float,
                               kappa: float = 3.0) -> float:
    """The horizon that maximises `net_edge_with_carry`: (kappa rho sigma/2f)^2.

    Undefined without a carry: with f = 0 the net edge rises without bound in
    tau, so there is no interior optimum.
    """
    if carry_per_day <= 0:
        return float("inf")
    return (kappa * rho * sigma / (2.0 * carry_per_day)) ** 2


def rho_floor_with_carry(cost: float, sigma: float, carry_per_day: float,
                         kappa: float = 3.0) -> float:
    """2 sqrt(f c)/(kappa sigma): the floor that no horizon escapes.

    Returns 0 when there is no carry, since then the source's floor applies at
    each horizon separately and every positive correlation clears it eventually.
    """
    if cost <= 0 or carry_per_day <= 0:
        return 0.0
    return 2.0 * np.sqrt(carry_per_day * cost) / (kappa * sigma)


def min_horizon_fee_only(cost: float, sigma: float, rho: float,
                         kappa: float = 3.0) -> float:
    """Invert the source's floor for tau: tau >= (c/(kappa sigma rho))^2."""
    if cost <= 0:
        return 0.0
    return (cost / (kappa * sigma * rho)) ** 2


# --------------------------------------------------------------------------
# 4. Cross-sectional information coefficient
# --------------------------------------------------------------------------
# Everything above is a *time-series* statement: one asset, does the signal
# predict that asset's own next return. A cross-sectional book asks a different
# question -- at each date, across assets, does the signal rank them -- and the
# two are measured on orthogonal pieces of the same panel.
#
# Decompose any panel into a date effect and a deviation from it:
#
#     x[i,t] = xbar[t] + xtilde[i,t],     sum_i xtilde[i,t] = 0
#     y[i,t] = ybar[t] + ytilde[i,t],     sum_i ytilde[i,t] = 0
#
# The common-component (date-mean) IC lives in (xbar, ybar) and the
# cross-sectional IC lives in
# (xtilde, ytilde). Because the two components are orthogonal by construction, a
# signal can carry any combination of the two, including a large IC on one and
# exactly zero on the other. `panel_ic_decomposition` measures all three.
#
# Two consequences for the floor:
#
#   * The denominator changes. A dollar-neutral book earns the *dispersion* of
#     returns, not their level, because the common move cancels. Under a
#     homogeneous equicorrelation model dispersion is smaller than total
#     volatility, which would make the per-name floor HIGHER for a
#     cross-sectional book than for a directional one on the same instrument.
#     Measured 2026 dispersion instead EXCEEDS median single-name volatility, so
#     the realised per-name floor is about 21% LOWER. The direction is an
#     empirical question, not a theorem.
#
#   * Breadth changes, and in the opposite direction. Directional positions all
#     load on the common factor, so their effective breadth saturates at
#     1/rho_return however many names are held. Dollar-neutral positions cancel
#     that factor, so the equicorrelation MODEL assigns breadth N-1. That N-1 is
#     an independence bound on a return covariance, not a count of profitable
#     independent forecasts: see `residual_participation_ratio`. Read the
#     comparison as what the model implies, not as a measured advantage.

# R^2 is IC^2 for a one-predictor regression, which makes it quadratic in the
# quantity that actually matters and therefore very hard to read. Substituting
# the fundamental law IR = IC sqrt(BR) gives the interpretive key:
#
#     R^2 = IC^2 = IR^2 / BR
#
# so an R^2 is only meaningful alongside a breadth. An R^2 of 3e-5 is a
# Sharpe-2 book across 500 names traded daily, and a catastrophe on one name.

# For an arbitrary predictor -- a gradient-boosted tree, a neural net, anything
# that is not the OLS projection of y on x -- R^2 = IC^2 no longer holds. Writing
# a = sd(yhat)/sd(y) for the scale of the prediction and b = (mean(yhat) -
# mean(y))/sd(y) for its bias, the exact identity is
#
#     R^2 = 2 rho a - a^2 - b^2
#         = IC^2 - (a - IC)^2 - b^2
#
# The second form is the useful one. It says R^2 <= IC^2 always, with equality
# only when the prediction is scaled to a = IC and unbiased at b = 0, and it
# splits the shortfall into a scale error and a bias error. OLS with an intercept
# hits both conditions by construction, which is why R^2 = IC^2 there and only
# there. Verified symbolically and against simulation in scripts/gbm_r2.py.

def r2_from_components(ic: float, scale: float, bias: float = 0.0) -> float:
    """Out-of-sample R^2 from correlation, prediction scale and prediction bias."""
    return 2.0 * ic * scale - scale**2 - bias**2


def r2_upper_bound(ic: float) -> float:
    """IC^2: the largest R^2 any rescaling of a prediction can reach."""
    return ic**2


def optimal_prediction_scale(ic: float) -> float:
    """The sd ratio that maximises R^2, which is IC itself.

    A raw forecast with sd(yhat) = sd(y) is over-scaled by a factor 1/IC. The
    shrinkage that maximises R^2 is therefore severe: at IC = 3% the prediction
    should carry 3% of the target's standard deviation, not 100% of it.
    """
    return ic


def r2_decomposition(y, yhat) -> dict:
    """Split a realised out-of-sample R^2 into information, scale and bias.

    Returns the three additive terms of R^2 = IC^2 - (a - IC)^2 - b^2 along with
    the R^2 that the same predictions would reach after an optimal affine
    rescaling, which is exactly IC^2.
    """
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    sy = y.std()
    if sy <= 0:
        raise ValueError("y has no variance")
    ic = float(np.corrcoef(yhat, y)[0, 1])
    a = float(yhat.std() / sy)
    b = float((yhat.mean() - y.mean()) / sy)
    sse = float(((y - yhat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    return {
        "r2": 1.0 - sse / sst,
        "r2_from_components": r2_from_components(ic, a, b),
        "ic": ic,
        "scale": a,
        "bias": b,
        "information_term": ic**2,
        "scale_penalty": (a - ic) ** 2,
        "bias_penalty": b**2,
        "r2_after_rescaling": ic**2,
        "optimal_scale": ic,
    }


def r2_from_ir(ir: float, breadth: float) -> float:
    """R^2 implied by an information ratio and a breadth, approximately.

    R^2 ~ IR^2 / BR. This is *not* an identity. It chains R^2 = rho^2, which
    needs a calibrated projection, with the idealised fundamental law
    IR = IC sqrt(BR), which needs a transfer coefficient of one, independent
    bets, and small IC. Under the proportional-payoff model of this report the
    exact relation is r2_from_ir_exact below. The approximation is excellent
    wherever IR^2/BR is small, which covers every equity example here, and
    degrades as IR^2/BR approaches one.
    """
    if breadth <= 0:
        raise ValueError("breadth must be positive")
    return ir**2 / breadth


def r2_from_ir_exact(ir: float, breadth: float) -> float:
    """R^2 implied by IR and BR under the exact proportional-payoff model.

    SR^2 = IR^2/BR and SR = rho/sqrt(1 + rho^2), so with R^2 = rho^2

        R^2 = (IR^2/BR) / (1 - IR^2/BR).

    Requires IR^2 < BR: a per-bet Sharpe of one is the ceiling of the
    proportional rule as rho -> 1.
    """
    if breadth <= 0:
        raise ValueError("breadth must be positive")
    s2 = ir**2 / breadth
    if s2 >= 1.0:
        raise ValueError("ir^2 must be below breadth: per-bet Sharpe caps at 1")
    return s2 / (1.0 - s2)


def ir_from_r2(r2: float, breadth: float) -> float:
    """Information ratio implied by an R^2 and a breadth, approximately.

    IR ~ sqrt(R^2 * BR), the inverse of `r2_from_ir` and subject to the same
    assumptions. See `ir_from_r2_exact` for the model-exact version.
    """
    if r2 < 0:
        raise ValueError("r2 must be non-negative")
    return float(np.sqrt(r2 * breadth))


def ir_from_r2_exact(r2: float, breadth: float) -> float:
    """IR implied by R^2 and BR under the exact proportional-payoff model.

    IR = sqrt(BR R^2 / (1 + R^2)), from SR = rho/sqrt(1 + rho^2).
    """
    if r2 < 0:
        raise ValueError("r2 must be non-negative")
    if breadth <= 0:
        raise ValueError("breadth must be positive")
    return float(np.sqrt(breadth * r2 / (1.0 + r2)))


def breadth_for_r2_and_ir(r2: float, ir: float) -> float:
    """Bets per year needed to turn a given R^2 into a given IR."""
    if r2 <= 0:
        raise ValueError("r2 must be positive")
    return ir**2 / r2


def cross_sectional_floor(cost: float, dispersion: float, tau: float,
                          kappa: float = 3.0) -> float:
    """Per-name floor for a cross-sectional book.

    Identical algebra to `rho_floor`, but `dispersion` is the cross-sectional
    standard deviation of horizon returns rather than a single asset's total
    volatility. Naming them apart is the point: substituting total volatility
    here understates the bar by exactly the factor by which the common move
    inflates single-name vol.
    """
    return rho_floor(cost, dispersion, tau, kappa)


def portfolio_breakeven_ic(cost: float, dispersion: float, tau: float,
                           turnover: float = 2.0) -> float:
    """IC at which a *whole* cross-section pays for its own turnover.

    The kappa-sigma floor asks whether the largest signal can pay. A
    cross-sectional book does not get to trade only its best names -- it holds
    the whole ranking -- so what matters is whether the average name pays. With
    unit gross exposure and weights proportional to the signal, expected gross
    return per period is beta / E|x| = beta sqrt(pi/2), so

        ic > (cost/2) * turnover / (sqrt(pi/2) * dispersion * sqrt(tau)).

    `turnover` is sum_i |w[i,t] - w[i,t-1]| at unit gross exposure, and cost is
    charged as (c/2) * turnover -- the same "pay for position changes" convention
    as the persistence experiment in section 2, under which one complete round
    trip (turnover = 2, the default) costs exactly c.

    At the default this is kappa/sqrt(pi/2) = 2.39 times `cross_sectional_floor`.
    A signal-weighted book on an iid signal turns over about 1.41 rather than 2,
    which brings the multiple down to about 1.7. Verified against simulation in
    scripts/cross_sectional_ic.py.
    """
    gross_per_beta = 1.0 / SQRT_2_OVER_PI  # = sqrt(pi/2) = 1.2533
    return (0.5 * cost * turnover) / (gross_per_beta * dispersion * np.sqrt(tau))


def breadth_common_position(n: int, return_corr: float) -> float:
    """Effective bets for a book holding the SAME position in every name.

    Valid only under that restriction, and the restriction is load-bearing.
    When every name carries a common position h, the per-name P&L is h*y_i, so
    the P&L streams inherit the RETURN correlation exactly and

        BR = n / (1 + (n-1) rho_return)

    is the right count, saturating at 1/rho_return. A market-timing book that
    goes long or short the whole universe on one signal is this case.

    It is NOT general breadth. Substituting a return correlation into
    `effective_breadth` for a book with per-name forecasts is a mistake, and an
    easy one to make. Grinold-Kahn breadth counts independent
    P&L streams, and those are governed by the correlation of the FORECASTS,
    not of the returns. The two come apart completely: with

        y_i = beta x_i + gamma f + eps_i,   x_i, f, eps_i all independent,

    the returns share the factor f and can be correlated arbitrarily close to 1
    through gamma, while the proportional-rule P&Ls x_i*y_i are uncorrelated
    across names and breadth stays at n. At gamma = 10 and n = 40 the measured
    return correlation is 0.99 and the true breadth 39.9, while this formula
    returns 1.01. See `tests/test_engine.py`.

    Use `effective_breadth(n, alpha_corr)` with a FORECAST correlation for
    anything other than the common-position case.

    `breadth_directional` remains as a deprecated alias.
    """
    return effective_breadth(n, return_corr)


def breadth_directional(n: int, return_corr: float) -> float:
    """Deprecated alias for `breadth_common_position`.

    The old name suggested this was the breadth of any directional strategy.
    It is the breadth of a common-position book specifically.
    """
    return breadth_common_position(n, return_corr)


def breadth_neutral(n: int) -> float:
    """Independence UPPER BOUND on the breadth of a dollar-neutral book.

    Removing the cross-sectional mean removes the common factor, leaving n
    residuals that carry n-1 degrees of freedom. That degree-of-freedom count
    is a dimensional ceiling, not a count of independent bets: attaining it
    requires a forecast for each residual direction whose errors are themselves
    uncorrelated, which no return covariance can establish. Every information
    ratio computed from this figure is therefore an upper bound.
    """
    return max(float(n - 1), 0.0)


# There are two correlations in a one-factor market and they are not the same
# number. With equal loadings, r_i = b f + e_i:
#
#     q     = Corr(r_i, f)      asset-to-factor
#     rho_r = Corr(r_i, r_j)    pairwise, between two assets
#
# and rho_r = q^2. Dispersion needs one, breadth needs the other, so a single
# figure quoted for "the correlation" will be wrong in one of the two places.
# These helpers each name which they take.

def pairwise_from_factor_corr(factor_corr: float) -> float:
    """rho_r = q^2 in a one-factor model with equal loadings."""
    return factor_corr**2


def factor_corr_from_pairwise(pairwise_corr: float) -> float:
    """q = sqrt(rho_r), the inverse of `pairwise_from_factor_corr`."""
    return float(np.sqrt(max(pairwise_corr, 0.0)))


def dispersion_from_pairwise_corr(total_vol: float,
                                  pairwise_corr: float) -> float:
    """Cross-sectional dispersion of returns from the *pairwise* correlation.

    For equicorrelated names, Var(r_i - r_bar) -> Var(r_i)(1 - rho_r) as N
    grows, so the dispersion a dollar-neutral book earns is
    total_vol * sqrt(1 - rho_r). Parameterising by rho_r rather than by q keeps
    dispersion and breadth consistent, since breadth needs rho_r too.
    """
    return total_vol * np.sqrt(max(1.0 - pairwise_corr, 0.0))


def dispersion_from_factor_corr(total_vol: float, factor_corr: float) -> float:
    """Same quantity, from the *asset-to-factor* correlation q.

    Equal to `dispersion_from_pairwise_corr(total_vol, q**2)`.
    """
    return dispersion_from_pairwise_corr(total_vol,
                                        pairwise_from_factor_corr(factor_corr))


def expected_cs_variance(cov: np.ndarray) -> float:
    """Expected cross-sectional variance of an equally weighted universe.

    For a single date with return covariance Sigma across N names, the
    cross-sectional sample variance around the equal-weighted mean has
    expectation

        E[s_CS^2] = tr(M Sigma M) / (N - 1),    M = Id - 11'/N.

    This is the general form of `dispersion_from_pairwise_corr`, which is its
    equicorrelation special case. Use it whenever variances are heterogeneous:
    unlike the equicorrelation shortcut it needs no assumption that names share
    a volatility, and it is the correct comparison for a *measured* dispersion,
    because both sides are then second moments of the same weighted average.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov must be square")
    if n < 2:
        raise ValueError("need at least two names")
    m = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(m @ cov @ m) / (n - 1))


def expected_cs_dispersion(cov: np.ndarray) -> float:
    """Square root of `expected_cs_variance`, in return units."""
    return float(np.sqrt(max(expected_cs_variance(cov), 0.0)))


def mean_offdiagonal(corr: np.ndarray) -> float:
    """Mean of the off-diagonal entries of a correlation matrix.

    This, and not the median, is the input the equicorrelation breadth formula
    requires: see `effective_breadth_from_corr` for why the two cannot be
    interchanged.
    """
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    if n < 2:
        raise ValueError("need at least two names")
    off = corr.sum() - np.trace(corr)
    return float(off / (n * (n - 1)))


def effective_breadth_from_corr(corr: np.ndarray) -> float:
    """Effective breadth of an equally weighted book from its correlation matrix.

    For N unit-variance bets with correlation matrix C, the equally weighted
    average signal has variance 1'C1/N^2, so

        BR_eff = N^2 / (1' C 1).

    Expanding 1'C1 = N + N(N-1) rho_bar shows this equals
    N / (1 + (N-1) rho_bar) with rho_bar the **mean** off-diagonal correlation.
    The mean is therefore not a convenient summary but the exact sufficient
    statistic; substituting a median gives a different and generally wrong
    answer, most badly when the pairwise distribution is multi-modal, because a
    median then sits in a trough that no part of the book occupies.
    """
    corr = np.asarray(corr, dtype=float)
    n = corr.shape[0]
    total = float(np.ones(n) @ corr @ np.ones(n))
    if total <= 0:
        raise ValueError("correlation matrix is not positive on the ones vector")
    return n**2 / total


def residual_participation_ratio(cov: np.ndarray) -> float:
    """Participation ratio of the cross-sectionally demeaned return covariance.

    Demeaning leaves residuals whose covariance is R = M Sigma M with
    M = Id - 11'/N. This returns

        (tr R)^2 / tr(R R),

    which attains its maximum of N-1 exactly when R = lambda * M, i.e. when the
    covariance is isotropic on the neutral subspace. It is NOT a statement about
    the demeaned residuals being mutually uncorrelated: they cannot be, since
    they sum to zero by construction.

    NOT strategy breadth, and deliberately not named as though it were. The
    Grinold-Kahn breadth of a dollar-neutral book counts independent *bets*,
    which depends on the covariance of the forecasts, on the portfolio weights,
    and on how both relate to returns. None of those appear here.

    It does not bound breadth from above either, so it is silent in BOTH
    directions. Where Sigma is positive definite on the neutral subspace R has
    rank N-1, and its directions then whiten into N-1 uncorrelated unit-RISK
    directions. That says nothing about bets: breadth counts profitable
    independent FORECASTS along those directions, and a return covariance cannot
    establish that any exist. Positive definiteness is not automatic either, so a
    caller wanting the rank should compute it rather than assume N-1; the
    calibration script does. Converting this ratio into an information ratio does
    not follow and is not done in this report.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    m = np.eye(n) - np.ones((n, n)) / n
    r = m @ cov @ m
    denom = float(np.trace(r @ r))
    if denom <= 0:
        return 0.0
    return float(np.trace(r) ** 2 / denom)


def panel_ic_decomposition(x: np.ndarray, y: np.ndarray) -> dict:
    """Split a (T, N) panel's correlation into its time-series and
    cross-sectional parts.

    Returns the pooled correlation (which is neither), the common-component IC
    (the date-mean statistic, NOT a general per-asset time-series IC) on the
    date means, the mean per-date cross-sectional IC, and its t statistic.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape or x.ndim != 2:
        raise ValueError("x and y must be the same (T, N) shape")

    x_bar, y_bar = x.mean(axis=1), y.mean(axis=1)
    x_tilde, y_tilde = x - x_bar[:, None], y - y_bar[:, None]

    per_date = np.full(x.shape[0], np.nan)
    for t in range(x.shape[0]):
        sx, sy = x_tilde[t].std(), y_tilde[t].std()
        if sx > 0 and sy > 0:
            per_date[t] = (x_tilde[t] @ y_tilde[t]) / (x.shape[1] * sx * sy)
    good = per_date[np.isfinite(per_date)]

    ts_ic = float(np.corrcoef(x_bar, y_bar)[0, 1])
    ts_se = (1.0 - ts_ic**2) / np.sqrt(max(x.shape[0] - 1, 1))
    return {
        "pooled_ic": float(np.corrcoef(x.ravel(), y.ravel())[0, 1]),
        # Kept under the original key for callers; this is the date-mean
        # statistic, documented as the common-component IC.
        "time_series_ic": ts_ic,
        "common_component_ic": ts_ic,
        # One estimate per date, so the common-component IC is measured on T points
        # while the cross-sectional IC is measured on N*T. That asymmetry is the
        # whole of `dates_for_cs_ic` and shows up as a much wider error bar here.
        "time_series_ic_se": float(ts_se),
        "time_series_ic_tstat": float(ts_ic / ts_se) if ts_se > 0 else 0.0,
        "cross_sectional_ic": float(good.mean()),
        "cross_sectional_ic_se": float(good.std(ddof=1) / np.sqrt(good.size)),
        "cross_sectional_ic_sd": float(good.std(ddof=1)),
        "cross_sectional_ic_tstat": float(
            good.mean() / (good.std(ddof=1) / np.sqrt(good.size))
        ),
        "n_dates": int(good.size),
        "n_assets": int(x.shape[1]),
    }


def dates_for_cs_ic(ic: float, n_assets: int, t_stat: float = 3.0,
                    ic_sd: float | None = None) -> float:
    """Dates needed to establish a cross-sectional IC at |t| = t_stat.

    Each date contributes one IC estimate whose sampling standard deviation is
    about 1/sqrt(n_assets) when the true IC is stable, so T ~ (t/ic)^2/n_assets.
    Pass `ic_sd` to use a measured date-to-date dispersion instead, which in
    practice is larger than 1/sqrt(N) because the true IC itself moves.
    """
    sd = (1.0 / np.sqrt(n_assets)) if ic_sd is None else ic_sd
    return (t_stat * sd / ic) ** 2


# --------------------------------------------------------------------------
# 5. Bridge to the law of active management (tweet 9/9)
# --------------------------------------------------------------------------

def sharpe_proportional_per_period(rho: float) -> float:
    """Per-period Sharpe of the costless proportional rule h = x.

    With y = beta x + v eps, where x and eps are independent standard normals
    and v = sigma sqrt(tau),

        E[h y] = beta,
        Var(h y) = beta^2 Var(x^2) + v^2 E[x^2] = 2 beta^2 + v^2,

    so SR = beta / sqrt(2 beta^2 + v^2). Substituting the *exact* inversion
    beta = rho v / sqrt(1 - rho^2) gives

        SR = rho / sqrt(1 + rho^2).

    The exact inversion is the one that matters. Substituting the leading-order
    inversion beta = rho v instead returns rho / sqrt(1 + 2 rho^2), which
    repeats inside this derivation the very error the report documents in the
    correlation formula: an approximation treated as exact. The two agree only
    to O(rho^3) and differ by 15% at rho = 0.71.

    To first order SR = rho, so the correlation is the per-period Sharpe ratio
    for the small correlations this report is mostly concerned with. Verified
    symbolically in scripts/check_algebra.py (check `proportional_sharpe`) and
    against simulation in tests/test_engine.py.
    """
    return rho / np.sqrt(1.0 + rho**2)


def sharpe_from_loading_to_noise(loading_to_noise: float) -> float:
    """The same Sharpe expressed through r = beta / v rather than through rho.

    SR = r / sqrt(1 + 2 r^2). Identical to `sharpe_proportional_per_period`
    under r = rho / sqrt(1 - rho^2), and stated separately because the two
    arguments are easy to confuse: r is the loading-to-noise ratio, rho is the
    Pearson correlation, and they coincide only to leading order.
    """
    r = loading_to_noise
    return r / np.sqrt(1.0 + 2.0 * r**2)


def annualised_ir(
    rho: float, tau: float, breadth: int = 1, periods_per_year: float = 252.0
) -> float:
    """Annualised IR of the proportional rule, BR = breadth * periods / tau.

    This is the exact model result, SR_per_bet sqrt(BR) with
    SR_per_bet = rho / sqrt(1 + rho^2), and not the idealised fundamental law
    IR = IC sqrt(BR). The two agree to O(rho^3); for the small correlations in
    the equity examples the difference is invisible, and at rho = 0.5 it is
    about 11%. Independence across bets is assumed and is a strong assumption:
    see the note on effective sample size in the report.
    """
    bets = breadth * periods_per_year / tau
    return sharpe_proportional_per_period(rho) * np.sqrt(bets)


def effective_breadth(n: int, alpha_corr: float) -> float:
    """Breadth after alpha correlation, N / (1 + (N-1) rho_alpha).

    Equicorrelated bets: the variance of the average signal is
    (1 + (N-1) rho_alpha)/N, so the count of *independent* bets is this.
    """
    return n / (1.0 + (n - 1) * alpha_corr)


def obs_for_tstat(rho: float, t: float = 2.0) -> float:
    """Observations needed for a Pearson rho-hat to reach |t| = t.

    se(rho-hat) ~ (1 - rho^2)/sqrt(N-1), so N ~ (t (1-rho^2)/rho)^2 + 1.
    """
    return (t * (1.0 - rho**2) / rho) ** 2 + 1.0


# --------------------------------------------------------------------------
# 6. Simulators -- simulated price action with alphas built for it
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class Market:
    """A simulated instrument."""

    sigma_daily: float
    tau_days: float
    cost: float
    label: str = ""

    @property
    def horizon_vol(self) -> float:
        return self.sigma_daily * np.sqrt(self.tau_days)

    def floor(self, kappa: float = 3.0) -> float:
        return rho_floor(self.cost, self.sigma_daily, self.tau_days, kappa)


def simulate_returns_and_alpha(
    market: Market,
    rho: float,
    n: int,
    rng: np.random.Generator,
    signal: str = "gaussian",
    sparsity: float | None = None,
):
    """Simulate horizon returns and a unit-variance alpha with correlation rho.

    The alpha is constructed *for* the price path, not fitted to it: the signal is
    drawn first and the return built as beta*x plus independent noise. That
    direction guarantees the population correlation
    is exactly `rho` (up to the O(rho^2) exactness choice below) with no
    in-sample fitting anywhere.

    `signal`:
      "gaussian" -- x ~ N(0,1)
      "sparse"   -- x = 0 with prob 1-p, else +/- 1/sqrt(p); Var(x) = 1
    """
    if signal == "gaussian":
        x = rng.standard_normal(n)
    elif signal == "sparse":
        if sparsity is None:
            raise ValueError("sparse signal needs `sparsity`")
        p = sparsity
        fires = rng.random(n) < p
        sign = rng.choice((-1.0, 1.0), size=n)
        x = np.where(fires, sign / np.sqrt(p), 0.0)
    else:
        raise ValueError(f"unknown signal {signal!r}")

    beta = beta_exact(rho, market.sigma_daily, market.tau_days)
    eps = rng.standard_normal(n)
    y = beta * x + market.horizon_vol * eps
    return x, y, beta


def banded_backtest(x: np.ndarray, y: np.ndarray, beta: float, cost: float):
    """Run the no-trade-band rule on a simulated sample.

    Returns per-period net P&L, the position series, and summary stats.
    """
    forecast = beta * x
    position = np.where(np.abs(forecast) > cost, np.sign(forecast), 0.0)
    pnl = position * y - cost * np.abs(position)
    traded = position != 0.0
    return {
        "pnl": pnl,
        "position": position,
        "net_mean": float(pnl.mean()),
        "net_sd": float(pnl.std(ddof=1)),
        "gross_mean": float((position * y).mean()),
        "cost_mean": float((cost * np.abs(position)).mean()),
        "trade_frequency": float(traded.mean()),
        "sharpe_per_period": float(pnl.mean() / pnl.std(ddof=1))
        if pnl.std(ddof=1) > 0
        else 0.0,
    }
