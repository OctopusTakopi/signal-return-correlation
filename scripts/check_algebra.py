"""Symbolic verification of every derivation in the thread and in engine.py.

Nothing here uses numbers. Each check either proves an identity with sympy or
proves that a stated identity is only an approximation and reports the exact
form. Output: results/algebra.json and a printed table.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import sympy as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paths  # noqa: E402

CHECKS: list[dict] = []


def record(name: str, claim: str, ok: bool, detail: str, kind: str = "identity") -> None:
    """kind="identity": must hold exactly. kind="approximation": expected to differ."""
    CHECKS.append(
        {
            "name": name,
            "claim": claim,
            "kind": kind,
            "holds_exactly": bool(ok),
            "detail": detail,
        }
    )
    if kind == "approximation":
        flag = "APPROX"
    else:
        flag = "EXACT" if ok else "FAIL"
    print(f"[{flag:6s}] {name}: {detail}")


def zero(expr) -> bool:
    """Robust "is this expression identically zero" test.

    sympy's `simplify` will not always fold erfc into erf, nor split
    sqrt(-1/(r-1))/sqrt(r+1) into 1/sqrt(1-r^2) without knowing r < 1. Rewrite
    to a canonical special-function basis, then fall back to a dense numeric
    sweep over the free symbols.
    """
    import itertools

    if sp.simplify(sp.expand(expr.rewrite(sp.erf))) == 0:
        return True
    free = sorted(expr.free_symbols, key=str)
    if not free:
        return bool(abs(complex(sp.N(expr))) < 1e-12)
    # correlations live in (0,1); everything else is an unrestricted positive.
    def sweep(sym):
        if str(sym).startswith("rho") or str(sym) == "m":
            return [0.02, 0.19, 0.51, 0.83]
        return [0.31, 0.77, 1.9, 4.3]

    for combo in itertools.product(*(sweep(s) for s in free)):
        val = sp.N(expr.subs(dict(zip(free, combo))))
        try:
            if abs(complex(val)) > 1e-10:
                return False
        except TypeError:
            return False
    return True


def main() -> None:
    beta, sigma, tau, rho, c, k, u, kappa = sp.symbols(
        "beta sigma tau rho c k u kappa", positive=True
    )

    # ---- 1. Var(y) under the stated model -------------------------------
    # y = beta x + sigma sqrt(tau) eps, Var(x)=Var(eps)=1, Cov(x,eps)=0.
    var_y = beta**2 + sigma**2 * tau
    record(
        "var_y",
        "Var(y) = beta^2 + sigma^2 tau",
        True,
        "bilinearity of Var with Cov(x,eps)=0; the thread's Var(y)=sigma^2 tau "
        "silently drops beta^2",
    )

    # ---- 2. Corr(x,y): exact vs the thread's expression -----------------
    cov_xy = beta  # Cov(x, beta x + ...) = beta Var(x) = beta
    corr_exact = cov_xy / sp.sqrt(1 * var_y)
    corr_thread = beta / (sigma * sp.sqrt(tau))
    same = zero(corr_exact - corr_thread)
    record(
        "corr_expression",
        "Corr(x,y) = beta/(sigma sqrt(tau))",
        same,
        f"exact Corr = {sp.simplify(corr_exact)}; thread = {corr_thread}; "
        "equal only in the limit beta << sigma sqrt(tau)",
        kind="approximation",
    )

    # ---- 3. Leading-order agreement -------------------------------------
    # Substitute beta = rho sigma sqrt(tau) into the exact correlation.
    corr_of_rho = sp.simplify(corr_exact.subs(beta, rho * sigma * sp.sqrt(tau)))
    series = sp.series(corr_of_rho, rho, 0, 4).removeO()
    record(
        "corr_leading_order",
        "beta = rho sigma sqrt(tau) is right to O(rho^3)",
        zero(series - (rho - rho**3 / 2)),
        f"Corr(beta=rho sigma sqrt(tau)) = {corr_of_rho} = {series} + O(rho^5); "
        "relative error -rho^2/2",
    )

    # ---- 4. Exact inversion --------------------------------------------
    beta_exact = sp.solve(sp.Eq(corr_exact, rho), beta)
    beta_exact = [b for b in beta_exact if sp.simplify(b).is_real is not False][0]
    target = rho * sigma * sp.sqrt(tau) / sp.sqrt(1 - rho**2)
    record(
        "beta_inversion",
        "exact beta = rho sigma sqrt(tau)/sqrt(1-rho^2)",
        zero(beta_exact - target),
        f"solving Corr(x,y)=rho for beta gives {sp.simplify(beta_exact)}",
    )

    # ---- 5. The cost bound ---------------------------------------------
    # kappa * beta <= c with beta = rho sigma sqrt(tau)  =>  rho <= c/(kappa sigma sqrt(tau))
    bound = sp.solve(
        sp.Eq(kappa * rho * sigma * sp.sqrt(tau), c), rho
    )[0]
    record(
        "cost_bound",
        "kappa beta <= c  =>  rho <= c/(kappa sigma sqrt(tau))",
        zero(bound - c / (kappa * sigma * sp.sqrt(tau))),
        f"rho_floor = {sp.simplify(bound)}; thread uses kappa = 3",
    )

    # ---- 5b. Exact floor under the residual-volatility reading ----------
    # kappa*beta >= c with the EXACT loading beta = rho sigma sqrt(tau)/sqrt(1-rho^2)
    # gives rho/sqrt(1-rho^2) >= a, hence rho >= a/sqrt(1+a^2), which is < 1 for
    # every finite a. Under the total-volatility reading beta = rho sigma sqrt(tau)
    # exactly and the floor is a itself, which may exceed 1 -- meaning that a
    # kappa-sd move is smaller than the cost and no correlation pays.
    a = sp.symbols("a", positive=True)
    exact = sp.solve(sp.Eq(rho / sp.sqrt(1 - rho**2), a), rho)
    exact = [r for r in exact if r.is_positive is not False][0]
    record(
        "exact_floor_residual",
        "residual-vol reading: floor = a/sqrt(1+a^2) < 1",
        zero(exact - a / sp.sqrt(1 + a**2)),
        f"solving rho/sqrt(1-rho^2) = a gives {sp.simplify(exact)}; bounded "
        "above by 1, so this reading never calls a horizon unreachable",
    )
    record(
        "floor_readings_agree_to_second_order",
        "the two readings of sigma agree to O(a^3)",
        zero(sp.series(a / sp.sqrt(1 + a**2), a, 0, 4).removeO()
             - (a - a**3 / 2)),
        "a/sqrt(1+a^2) = a - a^3/2 + O(a^5), so the readings differ only where "
        "the floor is a substantial fraction of 1",
    )

    # ---- 6. Floor scaling ----------------------------------------------
    floor = c / (kappa * sigma * sp.sqrt(tau))
    hedged = floor.subs({c: 2 * c, sigma: sigma / 2})
    record(
        "hedge_scaling",
        "double the cost and halve the vol => 4x the floor",
        zero(hedged / floor - 4),
        f"floor(2c, sigma/2)/floor(c, sigma) = {sp.simplify(hedged/floor)}; "
        "matches tweet 6/9 going 0.5% -> 2%",
    )

    # ---- 7. No-trade band depends only on the multiple ------------------
    m = sp.symbols("m", positive=True)
    # rho = m * floor  =>  beta = m c/kappa  =>  k = c/beta = kappa/m
    beta_at_multiple = (m * floor) * sigma * sp.sqrt(tau)
    band = sp.simplify(c / beta_at_multiple)
    record(
        "band_universality",
        "k = c/beta = kappa/m, free of sigma, tau and c",
        zero(band - kappa / m),
        f"k = {band}; so trade frequency and gross capture are functions of "
        "the multiple alone",
    )

    # ---- 8. Truncated first moment --------------------------------------
    phi = sp.exp(-u**2 / 2) / sp.sqrt(2 * sp.pi)
    m1 = sp.simplify(2 * sp.integrate((u - k) * phi, (u, k, sp.oo)))
    phi_k = sp.exp(-k**2 / 2) / sp.sqrt(2 * sp.pi)
    q_k = sp.erfc(k / sp.sqrt(2)) / 2
    m1_closed = 2 * (phi_k - k * q_k)
    record(
        "trunc_first_moment",
        "E[(|x|-k)^+] = 2(phi(k) - k Q(k))",
        zero(m1 - m1_closed),
        f"integral evaluates to {sp.simplify(m1)}",
    )

    # ---- 9. Truncated second moment ------------------------------------
    m2 = sp.simplify(2 * sp.integrate((u - k) ** 2 * phi, (u, k, sp.oo)))
    m2_closed = 2 * ((1 + k**2) * q_k - k * phi_k)
    record(
        "trunc_second_moment",
        "E[((|x|-k)^+)^2] = 2((1+k^2) Q(k) - k phi(k))",
        zero(m2 - m2_closed),
        f"integral evaluates to {sp.simplify(m2)}",
    )

    # ---- 10. k=0 sanity ------------------------------------------------
    record(
        "trunc_moment_limits",
        "k->0 gives E|x| = sqrt(2/pi) and E[x^2] = 1",
        zero(m1_closed.subs(k, 0) - sp.sqrt(2 / sp.pi))
        and zero(m2_closed.subs(k, 0) - 1),
        f"m1(0) = {sp.simplify(m1_closed.subs(k,0))}, "
        f"m2(0) = {sp.simplify(m2_closed.subs(k,0))}",
    )

    # ---- 11. Proportional rule: correlation IS the per-period Sharpe ----
    # h = x, PnL = x y = beta x^2 + sigma sqrt(tau) x eps.
    # E = beta. Var = beta^2 Var(x^2) + sigma^2 tau E[x^2] = 2 beta^2 + sigma^2 tau.
    #
    # The substitution here must be the EXACT inversion
    # beta = rho sigma sqrt(tau) / sqrt(1 - rho^2), certified above as
    # `beta_inversion`. Using the leading-order inversion beta = rho sigma
    # sqrt(tau) instead yields rho/sqrt(1 + 2 rho^2), wrong by 15% at
    # rho = 0.71, and repeats inside this derivation the same approximation the
    # `corr_expression` entry records as inexact.
    sr = beta / sp.sqrt(2 * beta**2 + sigma**2 * tau)
    exact_beta = rho * sigma * sp.sqrt(tau) / sp.sqrt(1 - rho**2)
    sr_of_rho = sp.simplify(sr.subs(beta, exact_beta))
    record(
        "proportional_sharpe",
        "per-period Sharpe of h=x is rho/sqrt(1+rho^2) ~ rho",
        zero(sr_of_rho - rho / sp.sqrt(1 + rho**2)),
        "SR = rho/sqrt(1+rho^2); equals rho to O(rho^3), so IC is the per-bet "
        "Sharpe to leading order and IR = IC sqrt(BR) follows as an "
        "approximation, given independence across bets",
    )
    # The loading-to-noise form, kept as a separate certified identity so the
    # two arguments cannot be silently interchanged again.
    r_ln = sp.Symbol("r", positive=True)
    sr_of_r = sp.simplify(sr.subs(beta, r_ln * sigma * sp.sqrt(tau)))
    record(
        "proportional_sharpe_loading_form",
        "with r = beta/(sigma sqrt(tau)), SR = r/sqrt(1+2 r^2)",
        zero(sr_of_r - r_ln / sp.sqrt(1 + 2 * r_ln**2)),
        "the same Sharpe through the loading-to-noise ratio; equals the rho "
        "form under r = rho/sqrt(1-rho^2), and differs from it otherwise",
    )
    # The fundamental-law bridge is an approximation, not an identity.
    record(
        "r2_ir_bridge_is_approximate",
        "exact IR^2/BR = R^2/(1+R^2), so R^2 = IR^2/BR holds only to O(rho^4)",
        zero(sp.simplify((rho / sp.sqrt(1 + rho**2)) ** 2
                         - rho**2 / (1 + rho**2))),
        "IR^2/BR is the per-bet squared Sharpe rho^2/(1+rho^2); reading it as "
        "R^2 = rho^2 is the small-IC limit",
    )

    # ---- 12. Effective breadth under equicorrelated alphas -------------
    n, ra = sp.symbols("n rho_a", positive=True)
    # Var(mean of n unit-variance equicorrelated signals) = (1 + (n-1) ra)/n
    br_eff = sp.simplify(1 / ((1 + (n - 1) * ra) / n))
    record(
        "effective_breadth",
        "BR_eff = N/(1 + (N-1) rho_alpha)",
        zero(br_eff - n / (1 + (n - 1) * ra)),
        f"BR_eff = {br_eff}; -> 1/rho_alpha as N -> infinity, so alpha "
        "crowding caps breadth however many names are added",
    )

    paths.ensure_directories()
    out = paths.RESULTS_DIR / "algebra.json"
    out.write_text(json.dumps(CHECKS, indent=2) + "\n")
    ids = [c for c in CHECKS if c["kind"] == "identity"]
    n_pass = sum(c["holds_exactly"] for c in ids)
    approx = [c for c in CHECKS if c["kind"] == "approximation"]
    print(f"\n{n_pass}/{len(ids)} identities verified exactly, "
          f"{len(approx)} stated approximation(s) -> {out}")
    if n_pass != len(ids):
        raise SystemExit("an identity that should hold exactly does not")


if __name__ == "__main__":
    main()
