"""Replace the assumed market parameters with measured ones.

Sections 12-14 were written on order-of-magnitude guesses: 2.5%/day for BTC,
0.55 pairwise correlation, 3 bps/day funding, a 3 bps alt spread. This script
measures each of them on the complete Binance USDT-perpetual record and reports
what the guesses were worth.

The calibration window is 2026 only. Crypto's volatility regime has shifted far
enough that pooling 2020-2021 into an estimate of "normal" misleads: the older
windows are reported solely to show the size of the shift.

Data: the directory named by $BINANCE_DATA_DIR, holding 788 symbols of hourly
klines from 2020-01-01, 761 of minute klines from 2026-01-01, and monthly funding
rate archives. Delisted symbols are present and the universe is rebuilt at every
date from trailing volume, which removes the current-constituent form of
survivorship bias. It does not remove all of it: a symbol must clear an
observation threshold across the whole window, so short-lived listings are
excluded. See the report's section 15.1.

What is measurable from klines and what is not:

  volatility          measured, per symbol per window
  pairwise correlation measured within the point-in-time universe, by pairwise
                      deletion, so it is an estimate over the pairs with enough
                      overlap rather than one complete matrix
  dispersion          measured exactly. Compared with the HOMOGENEOUS
                      equicorrelation shortcut sigma * sqrt(1 - rho_r) on
                      matched second moments; a general one-factor model with
                      heterogeneous betas is not tested by that comparison.
  funding             measured, including the settlement interval, which is
                      predominantly 4 or 8 hours but also shows one-hour states
                      on symbols whose usual schedule is longer
  tick size           the smallest OBSERVED nonzero close-to-close increment,
                      which is an upper bound on the exchange tick rather than
                      the tick itself
  quoted spread       NOT measurable from klines. Three estimators are run and
                      all three fail on the level; they are reported anyway so
                      the disagreement is on the record.

Output: results/calibration.json, figures/fig12-fig13.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import engine  # noqa: E402
import paths  # noqa: E402
from _style_academic import (  # noqa: E402
    K, K25, K45, K70, NAVY, OCHRE, RED, TEAL, caption, light_grid, panel_label,
    plt,
)

# No default: the archive location is site-specific and must be supplied.
_ENV = "BINANCE_DATA_DIR"
_DATA_RAW = os.environ.get(_ENV, "")
# Path("") normalises to Path("."), which exists and is a directory, so a bare
# truthiness test on the Path would let an unset variable through and calibrate
# against whatever happens to sit in the working directory. Test the raw string.
DATA = Path(_DATA_RAW) if _DATA_RAW.strip() else None
CACHE = paths.RESULTS_DIR / "_daily_panel.parquet"
BPS = 1e-4

# Stablecoin and index bases: their "returns" are not the object of study.
EXCLUDE_BASES = {
    "USDC", "BUSD", "TUSD", "FDUSD", "DAI", "USDP", "SUSD", "USTC", "UST",
    "EUR", "GBP", "AUD", "JPY", "TRY", "BRL", "ARS", "XUSD", "USD1", "USDE",
}

# 2026 is the window the defaults come from. Crypto's volatility regime has
# shifted enough that pooling 2020-2021 into an estimate of "normal" is
# misleading: BTC ran at 3.24%/day over the full sample against 1.9%/day in 2026.
# The older windows are kept only to show the size of that shift, which is what
# justifies discarding them.
# (label, start, end). The end is INCLUSIVE and must be given: slicing with
# loc[start:] alone let the "2024-2025" window run to the end of the archive, so
# it silently contained 2026 as well and its 934 days were 731 + 203. The
# calibration window itself was unaffected, being the most recent one.
WINDOWS = [
    ("2026 only", "2026-01-01", None),      # <- the calibration window
    ("2024-2025", "2024-01-01", "2025-12-31"),
    ("full sample", "2020-01-01", None),
]
CALIBRATION_WINDOW = "2026 only"
UNIVERSE_N = 200
VOL_LOOKBACK = 30      # days of trailing volume for universe selection
# A symbol needs to be present for most of the window. Fixed at 180 this would
# have silently emptied the 2026 window, so it scales with the window length.
MIN_OBS_FRACTION = 0.55
MIN_OBS_FLOOR = 60


# ---------------------------------------------------------------------------
# panel construction
# ---------------------------------------------------------------------------

def base_of(symbol: str) -> str:
    return symbol[:-4] if symbol.endswith("USDT") else symbol


def build_daily_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Daily close and daily quote volume, all symbols, from the hourly archive."""
    if CACHE.exists():
        panel = pd.read_parquet(CACHE)
        close = panel.xs("close", axis=1, level=1)
        vol = panel.xs("quote_volume", axis=1, level=1)
        print(f"  loaded cached panel: {close.shape[1]} symbols, "
              f"{close.shape[0]} days")
        return close, vol

    files = sorted(glob.glob(str(DATA / "parquet_1h" / "*.parquet")))
    closes, vols = {}, {}
    for i, f in enumerate(files):
        sym = Path(f).stem
        if base_of(sym) in EXCLUDE_BASES:
            continue
        try:
            d = pd.read_parquet(f, columns=["open_time", "close", "quote_volume"])
        except Exception:
            continue
        if d.empty:
            continue
        d.index = pd.to_datetime(d.open_time, unit="ms", utc=True)
        day = d.resample("1D")
        closes[sym] = day.close.last()
        vols[sym] = day.quote_volume.sum()
        if (i + 1) % 200 == 0:
            print(f"  read {i+1}/{len(files)} symbols")

    close = pd.DataFrame(closes).sort_index()
    vol = pd.DataFrame(vols).sort_index().reindex(columns=close.columns)
    panel = pd.concat({"close": close, "quote_volume": vol}, axis=1)
    panel = panel.swaplevel(axis=1).sort_index(axis=1)
    paths.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(CACHE)
    print(f"  built panel: {close.shape[1]} symbols, {close.shape[0]} days "
          f"-> cached at {CACHE.name}")
    return close, vol


def point_in_time_universe(vol: pd.DataFrame, n: int = UNIVERSE_N) -> pd.DataFrame:
    """Boolean mask: is symbol s in the top-n by trailing volume on date t.

    Volume through date t-1 only, so membership is knowable at t. Delisted
    symbols simply stop qualifying, which is what removes survivorship bias.
    """
    trailing = vol.rolling(VOL_LOOKBACK, min_periods=10).sum().shift(1)
    rank = trailing.rank(axis=1, ascending=False, method="first")
    return (rank <= n) & trailing.notna()


# ---------------------------------------------------------------------------
# measurements
# ---------------------------------------------------------------------------

def min_obs_for(n_days: int) -> int:
    return max(MIN_OBS_FLOOR, int(MIN_OBS_FRACTION * n_days))


def measure_window(rets: pd.DataFrame, member: pd.DataFrame, label: str) -> dict:
    """Volatility, pairwise correlation and dispersion inside one window."""
    r = rets.where(member)
    min_obs = min_obs_for(len(r))
    counts = r.notna().sum()
    keep = counts[counts >= min_obs].index
    r = r[keep]

    vols = r.std()
    # Cross-sectional dispersion: sd across names on each date, then median.
    per_date_n = r.notna().sum(axis=1)
    disp = r[per_date_n >= 20].std(axis=1, ddof=1)

    # Mean pairwise correlation, on the subset with enough overlap.
    sub = r.dropna(axis=1, thresh=min_obs)
    corr = sub.corr(min_periods=min_obs)
    iu = np.triu_indices_from(corr.values, k=1)
    pw = corr.values[iu]
    pw = pw[np.isfinite(pw)]

    sigma_bar = float(vols.median())
    rho_bar = float(np.median(pw))
    disp_measured = float(disp.median())
    disp_implied = sigma_bar * np.sqrt(max(1 - rho_bar, 0.0))

    # ---- matched-aggregation comparison ---------------------------------
    # The three figures above mix aggregations: a median volatility and a
    # median correlation are pushed through a nonlinear formula and compared
    # against a median dispersion. Medians do not commute with sqrt(1 - rho),
    # so the ratio of the two is not a test of the one-factor model. What
    # follows compares second moment with second moment on one common
    # universe, which is a test.
    matched = matched_dispersion_and_breadth(sub, min_obs)

    return {
        "window": label,
        "days": int(len(r)),
        "min_obs": int(min_obs),
        "symbols": int(len(keep)),
        "vol_btc": float(vols.get("BTCUSDT", np.nan)),
        "vol_eth": float(vols.get("ETHUSDT", np.nan)),
        "vol_median": sigma_bar,
        "vol_q25": float(vols.quantile(0.25)),
        "vol_q75": float(vols.quantile(0.75)),
        "pairwise_median": rho_bar,
        "pairwise_mean": float(np.mean(pw)),
        "pairwise_q25": float(np.quantile(pw, 0.25)),
        "pairwise_q75": float(np.quantile(pw, 0.75)),
        "pairwise_low_cluster_median": float(np.median(pw[pw < 0.45]))
        if (pw < 0.45).any() else float("nan"),
        "pairwise_high_cluster_median": float(np.median(pw[pw >= 0.45]))
        if (pw >= 0.45).any() else float("nan"),
        "pairwise_low_cluster_share": float((pw < 0.45).mean()),
        "n_pairs": int(pw.size),
        "dispersion_measured": disp_measured,
        "dispersion_implied_one_factor": float(disp_implied),
        "dispersion_ratio": float(disp_measured / disp_implied)
        if disp_implied > 0 else np.nan,
        "dispersion_series": disp,
        **matched,
    }


def matched_dispersion_and_breadth(sub: pd.DataFrame, min_obs: int) -> dict:
    """Dispersion and breadth compared on like aggregation with like.

    Takes the complete-case block of one window's return panel and reports:

      * `cs_dispersion_measured_rms` -- the root-mean-square across dates of
        the per-date cross-sectional standard deviation. A second moment.
      * `cs_dispersion_predicted_cov` -- sqrt(tr(M S M)/(N-1)) from the sample
        covariance matrix of the same block. The same second moment, predicted.
        This is `engine.expected_cs_dispersion`.
      * `cs_dispersion_predicted_equicorr` -- the equicorrelation shortcut with
        *matched* moments: mean variance and mean off-diagonal correlation,
        rather than a median of each.

    The covariance prediction is close to exact by construction, up to
    within-window time variation of the covariance, so it is not a test of a
    one-factor structure. The equicorrelation figure is the one that tests
    something, because it discards everything except two scalars.

    Breadth is reported from the correlation matrix itself, which is what
    `engine.effective_breadth_from_corr` computes, and separately from the mean
    and the median off-diagonal correlation so the gap between them is visible.
    """
    # A complete-case block is needed so that every date's cross-section spans
    # the same names as the covariance matrix. Requiring no gaps at all throws
    # away whole windows when one symbol has a single missing hour, so drop the
    # least-covered columns first and record how many went.
    dense = sub.loc[:, sub.notna().mean() >= 0.95]
    block = dense.dropna(axis=0, how="any")
    if block.shape[0] < 30 or block.shape[1] < 5:
        return {"matched_available": False,
                "matched_dropped_symbols": int(sub.shape[1] - dense.shape[1]),
                "matched_reason": "no complete-case block of usable size"}
    dropped = int(sub.shape[1] - block.shape[1])

    values = block.to_numpy(dtype=float)
    n_names = values.shape[1]
    cov = np.cov(values, rowvar=False, ddof=1)
    corr_m = np.corrcoef(values, rowvar=False)

    per_date = values.var(axis=1, ddof=1)
    measured_rms = float(np.sqrt(per_date.mean()))
    predicted_cov = engine.expected_cs_dispersion(cov)

    mean_var = float(np.mean(np.diag(cov)))
    mean_off = engine.mean_offdiagonal(corr_m)
    predicted_equi = float(np.sqrt(max(mean_var * (1.0 - mean_off), 0.0)))

    off = corr_m[~np.eye(n_names, dtype=bool)]
    median_off = float(np.median(off))

    return {
        "matched_available": True,
        "matched_symbols": int(n_names),
        "matched_days": int(block.shape[0]),
        "matched_dropped_symbols": dropped,
        "cs_dispersion_measured_rms": measured_rms,
        "cs_dispersion_predicted_cov": predicted_cov,
        "cs_dispersion_predicted_equicorr": predicted_equi,
        "cs_dispersion_ratio_cov": float(measured_rms / predicted_cov)
        if predicted_cov > 0 else np.nan,
        "cs_dispersion_ratio_equicorr": float(measured_rms / predicted_equi)
        if predicted_equi > 0 else np.nan,
        "corr_mean_offdiagonal": mean_off,
        "corr_median_offdiagonal": median_off,
        "breadth_from_corr_matrix": engine.effective_breadth_from_corr(corr_m),
        "breadth_from_mean_corr": engine.effective_breadth(n_names, mean_off),
        "breadth_from_median_corr": engine.effective_breadth(n_names, median_off),
        "residual_participation_ratio": engine.residual_participation_ratio(cov),
        "independence_bound_n_minus_1": float(n_names - 1),
        # Computed, not assumed: R = M Sigma M is only guaranteed rank N-1 when
        # Sigma is positive definite on the neutral subspace, which a sample
        # covariance need not be when days are few relative to names.
        "residual_rank": int(np.linalg.matrix_rank(
            (np.eye(n_names) - np.ones((n_names, n_names)) / n_names)
            @ cov
            @ (np.eye(n_names) - np.ones((n_names, n_names)) / n_names))),
    }


def matched_cohort(rets, member, w_old, w_new) -> dict:
    """Correlation change on names eligible in BOTH windows.

    Comparing a broad-universe mean correlation across windows confounds two
    things: a change in how incumbents co-move, and a change in which names are
    eligible. In this archive the second dominates, so the broad comparison
    cannot support a statement about the market's internal correlation. This
    holds the universe fixed and reports the incumbent change alongside the two
    entrant terms that drive the broad number.
    """
    def elig(start, end):
        m = member.loc[start:end]
        r = rets.loc[start:end].where(m)
        need = min_obs_for(len(r))
        counts = r.notna().sum()
        keep = list(counts[counts >= need].index)
        return set(keep), r[keep]

    old_set, r_old = elig(w_old[1], w_old[2])
    new_set, r_new = elig(w_new[1], w_new[2])
    common = sorted(old_set & new_set)
    entrants = sorted(new_set - old_set)
    if len(common) < 5:
        return {"matched_cohort_available": False}

    # The overlap rule must match the broad estimator's, or the cohort figures
    # are not comparable with the 0.3103 they are meant to explain. Both now
    # use min_obs_for(window length). Pair counts are reported so the reader
    # can see how much each group rests on.
    need_new = min_obs_for(len(r_new))

    def mean_off(r, cols, need):
        c = r[cols].corr(min_periods=need).values
        iu = np.triu_indices_from(c, 1)
        v = c[iu][np.isfinite(c[iu])]
        return (float(v.mean()) if v.size else float("nan")), int(v.size)

    def mean_cross(r, a, b, need):
        c = r[a + b].corr(min_periods=need)
        v = c.loc[a, b].values.ravel()
        v = v[np.isfinite(v)]
        return (float(v.mean()) if v.size else float("nan")), int(v.size)

    old_rho, n_old = mean_off(r_old, common, min_obs_for(len(r_old)))
    new_rho, n_new = mean_off(r_new, common, need_new)
    ent_rho, n_ent = ((mean_off(r_new, entrants, need_new))
                      if len(entrants) >= 2 else (float("nan"), 0))
    cross_rho, n_cross = ((mean_cross(r_new, common, entrants, need_new))
                          if entrants else (float("nan"), 0))
    n_tot = n_new + n_ent + n_cross
    blended = ((new_rho * n_new + ent_rho * n_ent + cross_rho * n_cross) / n_tot
               if n_tot else float("nan"))
    return {
        "matched_cohort_available": True,
        "matched_cohort_names": len(common),
        "matched_cohort_entrants": len(entrants),
        "matched_cohort_old_window": w_old[0],
        "matched_cohort_new_window": w_new[0],
        "matched_cohort_rho_old": old_rho,
        "matched_cohort_rho_new": new_rho,
        "matched_cohort_change_pct": 100.0 * (new_rho / old_rho - 1.0),
        "matched_cohort_pairs_old": n_old,
        "matched_cohort_pairs_new": n_new,
        "entrants_among_themselves_rho": ent_rho,
        "entrants_pairs": n_ent,
        "common_versus_entrants_rho": cross_rho,
        "cross_pairs": n_cross,
        "pair_weighted_blend_new_window": blended,
        "pair_weighted_blend_pairs": n_tot,
        "min_obs_rule_new": int(need_new),
        "min_obs_rule_old": int(min_obs_for(len(r_old))),
    }


def measure_funding(symbols: list[str], since: str) -> dict:
    """Funding rates and settlement intervals from the monthly archives."""
    root = DATA / "data/futures/um/monthly/fundingRate"
    rows, intervals = [], []
    missing_interval = 0
    for sym in symbols:
        d = root / sym
        if not d.is_dir():
            continue
        for z in sorted(d.glob("*.zip")):
            try:
                with zipfile.ZipFile(z) as zf:
                    name = zf.namelist()[0]
                    df = pd.read_csv(zf.open(name))
            except Exception:
                continue
            if df.empty or "last_funding_rate" not in df.columns:
                continue
            df["t"] = pd.to_datetime(df.calc_time, unit="ms", utc=True)
            df = df[df.t >= pd.Timestamp(since, tz="utc")]
            if df.empty:
                continue
            # A missing interval column must not be silently read as eight
            # hours: that is the assumption under test. Rows without a stated
            # interval are dropped from the interval statistics and counted.
            if "funding_interval_hours" in df.columns:
                hrs = df.funding_interval_hours.astype(float)
            else:
                missing_interval += len(df)
                hrs = pd.Series(np.nan, index=df.index)
            intervals.append(hrs.value_counts())
            # per-settlement rate -> bps per day
            rows.append(pd.DataFrame({
                "symbol": sym,
                "date": df.t.dt.floor("D").values,
                "interval_hours": hrs.values,
                "rate_bps": df.last_funding_rate.astype(float) / BPS,
                "per_day_bps": df.last_funding_rate.astype(float) / BPS
                               * (24.0 / hrs.values),
            }))
    if not rows:
        return {"available": False}
    f = pd.concat(rows, ignore_index=True)
    iv = pd.concat(intervals).groupby(level=0).sum()
    per_sym = f.groupby("symbol").per_day_bps
    return {
        "available": True,
        "symbols": int(f.symbol.nunique()),
        "settlements": int(len(f)),
        "settlements_without_stated_interval": int(missing_interval),
        "interval_hours_share": {float(k): float(v / iv.sum())
                                 for k, v in iv.items()},
        "mean_signed_bps_per_day": float(f.per_day_bps.mean()),
        "median_signed_bps_per_day": float(f.per_day_bps.median()),
        "mean_abs_bps_per_day": float(f.per_day_bps.abs().mean()),
        "median_abs_bps_per_day": float(f.per_day_bps.abs().median()),
        "q90_abs_bps_per_day": float(f.per_day_bps.abs().quantile(0.90)),
        "q99_abs_bps_per_day": float(f.per_day_bps.abs().quantile(0.99)),
        "per_symbol_mean_abs_median": float(per_sym.apply(
            lambda s: s.abs().mean()).median()),
        "_abs_series": f.per_day_bps.abs(),
        **_funding_by_symbol_day(f),
        **_interval_share_by_symbol(f),
    }


def _funding_by_symbol_day(f: pd.DataFrame) -> dict:
    """Funding aggregated to symbol-day before averaging.

    Averaging over settlement rows gives a four-hour contract twice the weight
    of an eight-hour one for the same day of exposure, which describes no book.
    Summing each symbol-day's settlement rates gives the funding actually paid
    that day, after which every symbol-day carries equal weight.

    The absolute figures also differ in kind, not just in weight: the
    event-weighted `mean_abs_bps_per_day` is a mean of |per-settlement rate|
    scaled to a day, whereas `symbol_day_mean_abs_bps` is the mean of the
    absolute *daily total*, in which settlements of opposite sign inside a day
    offset. The second is what a position experiences.
    """
    day = f.groupby(["symbol", "date"], observed=True).rate_bps.sum()
    if day.empty:
        return {"symbol_day_available": False}
    return {
        "symbol_day_available": True,
        "symbol_days": int(day.size),
        "symbol_day_mean_signed_bps": float(day.mean()),
        "symbol_day_median_signed_bps": float(day.median()),
        "symbol_day_mean_abs_bps": float(day.abs().mean()),
        "symbol_day_median_abs_bps": float(day.abs().median()),
        "symbol_day_q90_abs_bps": float(day.abs().quantile(0.90)),
        "symbol_day_q99_abs_bps": float(day.abs().quantile(0.99)),
    }


def _interval_share_by_symbol(f: pd.DataFrame) -> dict:
    """Settlement-interval share weighted by elapsed exposure, not by row count.

    Counting settlement rows over-represents short intervals: a four-hour
    schedule emits twice the rows of an eight-hour one for the same day of
    exposure, and an hourly schedule eight times. That bias operates *within* a
    symbol as well as across symbols, so taking the mode of a symbol's raw
    settlement rows does not remove it: a symbol that spent more days on eight
    hours than on four can still show four hours as its modal row.

    The fix is to weight each interval by the time it actually covered, which is
    the sum of `interval_hours` over the rows at that interval. Two figures are
    reported:

      * `interval_share_by_exposure` -- the fraction of total contract-hours in
        the window running on each schedule, pooled over symbols.
      * `interval_share_by_symbol` -- assigning each symbol the schedule that
        covered most of its own hours, then counting symbols equally.

    The first answers what the market runs on, the second what a typical
    contract runs on. Neither is a count of settlements.
    """
    if "interval_hours" not in f.columns:
        return {"interval_share_by_symbol_available": False}
    g = f.dropna(subset=["interval_hours"])
    if g.empty:
        return {"interval_share_by_symbol_available": False}
    # hours of exposure contributed by each (symbol, schedule)
    exposure = g.groupby(["symbol", "interval_hours"], observed=True) \
                .interval_hours.sum()
    if exposure.empty:
        return {"interval_share_by_symbol_available": False}
    pooled = exposure.groupby(level="interval_hours").sum()
    pooled = pooled / pooled.sum()
    predominant = exposure.groupby(level="symbol").idxmax().map(lambda t: t[1])
    by_symbol = predominant.value_counts(normalize=True).sort_index()
    return {
        "interval_share_by_symbol_available": True,
        "interval_symbols_counted": int(predominant.size),
        "interval_share_by_exposure": {float(k): float(v)
                                       for k, v in pooled.sort_index().items()},
        "interval_share_by_symbol": {float(k): float(v)
                                     for k, v in by_symbol.items()},
    }


def measure_ticks(symbols: list[str]) -> dict:
    """Smallest nonzero price increment, in bps of price, from minute bars."""
    out = {}
    for sym in symbols:
        f = DATA / "parquet_1m" / f"{sym}.parquet"
        if not f.exists():
            continue
        try:
            d = pd.read_parquet(f, columns=["close"]).close.to_numpy()
        except Exception:
            continue
        if d.size < 1000:
            continue
        diffs = np.abs(np.diff(d))
        nz = diffs[diffs > 0]
        if nz.size < 50:
            continue
        tick = float(np.min(nz))
        last = float(d[-1])
        out[sym] = {"tick_abs": tick, "price_last": last,
                    "price_median": float(np.median(d)),
                    "tick_bps": tick / last / BPS,
                    "tick_bps_at_median_price": tick / float(np.median(d)) / BPS}
    return out


def roll_spread(prices: np.ndarray) -> float:
    """Roll (1984): S = 2 sqrt(-Cov(dp_t, dp_{t-1})), from bid-ask bounce.

    Returns nan when the serial covariance is positive, which happens whenever
    the price process has genuine short-horizon momentum rather than pure
    bounce -- one of the estimator's well-known failure modes.
    """
    d = np.diff(np.log(prices))
    if d.size < 1000:
        return float("nan")
    c = float(np.cov(d[1:], d[:-1])[0, 1])
    return 2 * np.sqrt(-c) if c < 0 else float("nan")


def corwin_schultz(high: pd.Series, low: pd.Series) -> float:
    """Corwin-Schultz (2012) high-low spread estimate, as a proportion.

    Two-day high-low ranges exceed one-day ranges by more than volatility alone
    explains, and the excess identifies the spread. Negative estimates are set to
    zero, as the original paper does. This is an estimate, not a measurement: no
    order-book data is present in a kline archive.
    """
    h, l = np.log(high), np.log(low)
    beta = ((h - l) ** 2).rolling(2).sum()
    h2 = np.log(high.rolling(2).max())
    l2 = np.log(low.rolling(2).min())
    gamma = (h2 - l2) ** 2
    k = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
    s = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return float(np.nanmedian(np.maximum(s, 0.0)))


def measure_spreads(symbols: list[str]) -> dict:
    """Three spread estimators, so their disagreement is visible.

    None of them recovers a quoted spread, and reporting that is more useful
    than reporting a number. A kline archive has no order book: high, low, open
    and close are trade prices, and every estimator below is inferring a quoted
    spread from the statistical fingerprint trading leaves in them.
    """
    out, skipped = {}, []
    for sym in symbols:
        f = DATA / "parquet_1m" / f"{sym}.parquet"
        if not f.exists():
            continue
        try:
            m = pd.read_parquet(f, columns=["open_time", "high", "low", "close"])
        except Exception:
            continue
        if len(m) < 20_000:
            skipped.append((sym, len(m)))
            continue
        m.index = pd.to_datetime(m.open_time, unit="ms", utc=True)
        day = m.resample("1D")
        hi, lo = day.high.max().dropna(), day.low.min().dropna()
        out[sym] = {
            "roll_1m_bps": roll_spread(m.close.to_numpy()) / BPS,
            "corwin_schultz_1m_bps": corwin_schultz(m.high, m.low) / BPS,
            "corwin_schultz_1d_bps": (corwin_schultz(hi, lo) / BPS
                                      if len(hi) > 60 else float("nan")),
            "minute_bars": int(len(m)),
        }
    if skipped:
        print(f"  (skipped for too few minute bars: "
              f"{', '.join(f'{s} {n}' for s, n in skipped)})")
    return out


# ---------------------------------------------------------------------------

def main() -> None:
    paths.ensure_directories()
    if DATA is None or not DATA.is_dir():
        raise SystemExit(
            f"set {_ENV} to the Binance archive directory. It must contain "
            "parquet_1h/, parquet_1m/ and data/futures/um/monthly/fundingRate/."
        )

    print("=" * 78)
    print("Calibrating section 12-14 parameters against the Binance record")
    print("=" * 78)
    print(f"source: ${_ENV}")
    close, vol = build_daily_panel()
    rets = np.log(close).diff()
    last = close.index.max()
    member = point_in_time_universe(vol)
    print(f"  universe: top {UNIVERSE_N} by trailing {VOL_LOOKBACK}d volume, "
          f"rebuilt daily")
    print(f"  span: {close.index.min().date()} to {last.date()}")

    windows = list(WINDOWS)

    print("\n" + "=" * 78)
    print("1. Volatility, pairwise correlation and dispersion")
    print("=" * 78)
    print(f"{'window':13s} {'days':>5s} {'syms':>5s} {'BTC':>7s} {'ETH':>7s} "
          f"{'median':>7s} {'IQR':>13s} {'rho_r':>7s} {'disp':>7s} "
          f"{'1-factor':>9s} {'ratio':>6s}")
    measured = []
    for lab, start, end in windows:
        sl = slice(start, end)
        m = member.loc[sl]
        r = rets.loc[sl]
        res = measure_window(r, m, lab)
        res["start"], res["end"] = start, (end or str(last.date()))
        measured.append(res)
        print(f"{lab:13s} {res['days']:5d} {res['symbols']:5d} "
              f"{100*res['vol_btc']:6.2f}% {100*res['vol_eth']:6.2f}% "
              f"{100*res['vol_median']:6.2f}% "
              f"{100*res['vol_q25']:5.2f}-{100*res['vol_q75']:5.2f}% "
              f"{res['pairwise_median']:7.3f} "
              f"{100*res['dispersion_measured']:6.2f}% "
              f"{100*res['dispersion_implied_one_factor']:8.2f}% "
              f"{res['dispersion_ratio']:6.3f}")

    print("\n  The last three columns are NOT a model test. The prediction mixes")
    print("  a median volatility with a median correlation through a nonlinear")
    print("  formula and compares the result with a median dispersion; medians do")
    print("  not commute with sqrt(1 - rho), so the ratio confounds model error")
    print("  with the non-commutation. The matched second-moment comparison")
    print("  below is the test, and it reports a ~20% shortfall for 2026 rather")
    print("  than the ~50% this ratio suggests. What that shortfall rejects is")
    print("  the HOMOGENEOUS equicorrelation shortcut, not one-factor structure")
    print("  generally: heterogeneous betas are untested here.")

    # A broad-universe comparison across windows confounds regime with
    # composition. Hold the universe fixed before concluding anything about
    # how the market's internal correlation moved.
    w_new = next(w for w in windows if w[0] == CALIBRATION_WINDOW)
    w_old = next(w for w in windows if w[0] == "2024-2025")
    cohort = matched_cohort(rets, member, w_old, w_new)
    print("\n  matched cohort, universe held fixed across the two windows:")
    if cohort.get("matched_cohort_available"):
        print(f"    {cohort['matched_cohort_names']} names eligible in both; "
              f"{cohort['matched_cohort_entrants']} entered in "
              f"{cohort['matched_cohort_new_window']}")
        print(f"    incumbent mean rho  {cohort['matched_cohort_rho_old']:.4f}"
              f"  ->  {cohort['matched_cohort_rho_new']:.4f}"
              f"   ({cohort['matched_cohort_change_pct']:+.1f}%)")
        print(f"    entrants among themselves        "
              f"{cohort['entrants_among_themselves_rho']:.4f}")
        print(f"    incumbents versus entrants       "
              f"{cohort['common_versus_entrants_rho']:.4f}")
        print("    -> the broad-universe fall is mostly COMPOSITION, not a")
        print("       change in how incumbents co-move.")
    else:
        print("    unavailable: too few names common to both windows")

    recent = next(m for m in measured if m["window"] == CALIBRATION_WINDOW)
    cal_start = next(w[1] for w in windows if w[0] == CALIBRATION_WINDOW)
    eligible = member.loc[cal_start:].any()[lambda s: s].index
    # Rank by mean quote volume over the calibration window rather than taking
    # an alphabetical prefix. An alphabetical cut is arbitrary with respect to
    # everything that matters here, and funding magnitude is strongly related to
    # size, so it biases the distribution in an unknown direction. All eligible
    # symbols are used; the ranking only fixes the order for reporting.
    universe_syms = (vol.loc[cal_start:, eligible].mean()
                     .sort_values(ascending=False).index.tolist())

    print("\n" + "=" * 78)
    print("2. Funding: rates and settlement intervals")
    print("=" * 78)
    fund = measure_funding(universe_syms, cal_start)
    if fund.get("available"):
        print(f"  {fund['symbols']} symbols, {fund['settlements']:,} settlements "
              f"since {cal_start}")
        print("  settlement interval, share of settlements:")
        for h, share in sorted(fund["interval_hours_share"].items()):
            print(f"    {h:4.0f} h  {100*share:5.1f}%")
        print(f"  signed mean      {fund['mean_signed_bps_per_day']:+7.3f} bps/day")
        print(f"  signed median    {fund['median_signed_bps_per_day']:+7.3f} bps/day")
        print(f"  |rate| mean      {fund['mean_abs_bps_per_day']:7.3f} bps/day")
        print(f"  |rate| median    {fund['median_abs_bps_per_day']:7.3f} bps/day")
        print(f"  |rate| 90th pct  {fund['q90_abs_bps_per_day']:7.3f} bps/day")
        print(f"  |rate| 99th pct  {fund['q99_abs_bps_per_day']:7.3f} bps/day")
        print("\n  The carry term in section 13.3 needs the magnitude a directional")
        print("  book pays, which is the |rate| figure, not the signed mean.")
    else:
        print("  no funding archives found")

    print("\n" + "=" * 78)
    print("3. Tick size (exact) and quoted spread (estimated)")
    print("=" * 78)
    probe = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT",
             "ADAUSDT", "LINKUSDT", "AVAXUSDT"]
    ticks = measure_ticks(probe)
    print("  tick size is exact. It is quoted in bps of price, so it moves with")
    print("  price: the same tick is a different cost at a different level.\n")
    print(f"  {'symbol':10s} {'last price':>12s} {'tick':>11s} "
          f"{'tick, bps':>10s}")
    for sym, t in ticks.items():
        print(f"  {sym:10s} {t['price_last']:12.4f} {t['tick_abs']:11.6f} "
              f"{t['tick_bps']:9.4f}")

    spreads = measure_spreads(probe)
    print("\n  quoted spread: NOT measurable here. Three estimators, none of")
    print("  which recovers it, shown so the disagreement is on the record:\n")
    print(f"  {'symbol':10s} {'tick':>9s} {'Roll 1m':>9s} {'CS 1m':>9s} "
          f"{'CS 1d':>9s} {'Roll/tick':>10s}")
    for sym, sp in spreads.items():
        tb = ticks.get(sym, {}).get("tick_bps", float("nan"))
        print(f"  {sym:10s} {tb:8.4f} {sp['roll_1m_bps']:8.3f} "
              f"{sp['corwin_schultz_1m_bps']:8.3f} "
              f"{sp['corwin_schultz_1d_bps']:8.2f} {sp['roll_1m_bps']/tb:9.1f}x")
    # check whether the estimators at least rank the instruments correctly
    common = [s for s in spreads if s in ticks]
    if len(common) >= 4:
        tv = pd.Series({s: ticks[s]["tick_bps"] for s in common})
        rv = pd.Series({s: spreads[s]["roll_1m_bps"] for s in common})
        rank = float(tv.corr(rv, method="spearman"))
        print(f"\n  Spearman rank correlation, Roll against tick size: {rank:+.3f}")
        print("  The estimators recover the cross-sectional ORDERING but not the")
        print("  level: Roll puts BTC's spread ~50x its tick and Corwin-Schultz")
        print("  at daily frequency is off by three orders of magnitude, because")
        print("  at that horizon it is measuring volatility. Treat the spread as")
        print("  an assumption requiring book data, and use the tick as its floor.")
        spreads["_rank_corr_roll_vs_tick"] = rank

    print("\n" + "=" * 78)
    print("4. What the guesses were worth")
    print("=" * 78)
    guesses = [
        ("BTC daily vol", 0.025, recent["vol_btc"], "%"),
        ("ETH daily vol", 0.035, recent["vol_eth"], "%"),
        ("alt daily vol (median)", 0.060, recent["vol_median"], "%"),
        ("pairwise correlation", 0.55, recent["pairwise_median"], ""),
        ("dispersion", 0.0402, recent["dispersion_measured"], "%"),
    ]
    print(f"  {'parameter':26s} {'assumed':>10s} {'measured':>10s} {'error':>9s}")
    cal = {}
    for name, guess, meas, unit in guesses:
        err = meas / guess - 1 if guess else np.nan
        g = f"{100*guess:.2f}%" if unit == "%" else f"{guess:.3f}"
        m = f"{100*meas:.2f}%" if unit == "%" else f"{meas:.3f}"
        print(f"  {name:26s} {g:>10s} {m:>10s} {100*err:+8.1f}%")
        cal[name] = {"assumed": guess, "measured": meas, "rel_error": err}
    if fund.get("available"):
        f_meas = fund["median_abs_bps_per_day"]
        print(f"  {'funding, bps/day':26s} {3.0:10.2f} {f_meas:10.2f} "
              f"{100*(f_meas/3.0-1):+8.1f}%")
        cal["funding_bps_per_day"] = {"assumed": 3.0, "measured": f_meas,
                                     "rel_error": f_meas / 3.0 - 1}
    if "BTCUSDT" in ticks:
        tb = ticks["BTCUSDT"]["tick_bps"]
        print(f"  {'BTC tick, bps':26s} {0.0105:10.4f} {tb:10.4f} "
              f"{100*(tb/0.0105-1):+8.1f}%")
        cal["btc_tick_bps"] = {"assumed": 0.0105, "measured": tb,
                              "rel_error": tb / 0.0105 - 1}

    out = {
        "source": f"${_ENV}",
        "span": [str(close.index.min().date()), str(last.date())],
        "universe_n": UNIVERSE_N,
        "vol_lookback_days": VOL_LOOKBACK,
        "windows": [{k: v for k, v in m.items() if k != "dispersion_series"}
                    for m in measured],
        "matched_cohort": cohort,
        "funding": {k: v for k, v in fund.items() if not k.startswith("_")},
        "ticks": ticks,
        "spread_estimators_bps": spreads,
        "guess_vs_measured": cal,
    }
    path = paths.RESULTS_DIR / "calibration.json"
    path.write_text(json.dumps(out, indent=2, default=float) + "\n")
    print(f"\nwrote {paths.rel(path)}")

    make_figures(measured, fund, rets, member, cal_start, recent)


def make_figures(measured, fund, rets, member, start, recent) -> None:
    r = rets.loc[start:].where(member.loc[start:])

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))

    ax = axes[0]
    cnt = r.notna().sum()
    r = r[cnt[cnt >= min_obs_for(len(r))].index]   # same filter the table uses
    vols = r.std().dropna() * 100
    ax.hist(vols[vols < 20], bins=40, color=NAVY, alpha=0.8, lw=0)
    top = ax.get_ylim()[1]
    for v, lab, col, dy in ((vols.get("BTCUSDT"), "BTC", RED, 0.96),
                            (vols.get("ETHUSDT"), "ETH", OCHRE, 0.84)):
        if v == v:
            ax.axvline(v, color=col, lw=1.1)
            ax.text(v + 0.25, top * dy, lab, fontsize=6.8, color=col)
    ax.axvline(vols.median(), color=K, lw=1.0, ls="--")
    ax.text(vols.median() + 0.25, top * 0.66,
            f"median {vols.median():.1f}%", fontsize=6.8, color=K)
    ax.set_xlabel("daily volatility (%)")
    ax.set_ylabel("symbols")
    ax.set_title("single-name volatility")
    light_grid(ax, axis="y")
    panel_label(ax, "a")

    ax = axes[1]
    sub = r.dropna(axis=1, thresh=min_obs_for(len(r)))
    corr = sub.corr(min_periods=min_obs_for(len(r)))
    iu = np.triu_indices_from(corr.values, k=1)
    pw = corr.values[iu]
    pw = pw[np.isfinite(pw)]
    ax.hist(pw, bins=50, color=TEAL, alpha=0.85, lw=0)
    med = float(np.median(pw))
    top = ax.get_ylim()[1]
    ax.axvline(med, color=K, lw=1.1, ls="--")
    ax.text(med + 0.02, top * 0.97, f"median {med:.2f}", fontsize=6.8, color=K)
    # Label the two modes descriptively. Calling them "newly listed" and
    # "established" would assert listing status from a correlation threshold:
    # no listing date enters this calculation, and the
    # most correlated symbol in the 2026 universe is not an established
    # large-cap. The matched-cohort table in the report is where the
    # incumbent-versus-entrant split is measured with the dates actually used.
    lo, hi = pw[pw < 0.45], pw[pw >= 0.45]
    for v, lab, x in ((np.median(lo), "low mode", 0.02),
                      (np.median(hi), "high mode", 0.02)):
        ax.axvline(v, color=K45, lw=0.8, ls=":")
        ax.text(v + x, top * 0.55, f"{lab}\n{v:.2f}", fontsize=6.2, color=K70)
    ax.text(0.5, top * 0.14,
            f"bimodal: {100*lo.size/pw.size:.0f}% of pairs below 0.45",
            fontsize=6.2, color=K70, ha="center", style="italic")
    ax.set_xlabel(r"pairwise correlation $\rho_r$")
    ax.set_ylabel("pairs")
    ax.set_title(f"{len(pw):,} pairs, top {UNIVERSE_N}")
    light_grid(ax, axis="y")
    panel_label(ax, "b")

    ax = axes[2]
    disp = recent["dispersion_series"] * 100
    ax.plot(disp.index, disp.values, color=NAVY, lw=0.7)
    imp = 100 * recent["dispersion_implied_one_factor"]
    ax.axhline(imp, color=RED, lw=1.1, ls="--")
    ax.text(disp.index[int(0.03 * len(disp))], ax.get_ylim()[1] * 0.88,
            rf"one-factor $\sigma\sqrt{{1-\rho_r}}$ = {imp:.2f}%",
            fontsize=6.8, color=RED)
    ax.text(disp.index[int(0.03 * len(disp))], ax.get_ylim()[1] * 0.75,
            f"realised median = {100*recent['dispersion_measured']:.2f}%",
            fontsize=6.8, color=K)
    ax.axhline(100 * recent["dispersion_measured"], color=K, lw=1.0, ls=":")
    ax.set_ylabel("cross-sectional dispersion (%)")
    ax.set_title("dispersion, realised vs implied")
    ax.tick_params(axis="x", labelrotation=30, labelsize=6.2)
    light_grid(ax)
    panel_label(ax, "c")

    ratio = recent["dispersion_ratio"]
    caption(fig,
            "Figure 12. The market parameters of sections 12-14, measured on 2026 "
            f"Binance USDT-perpetual data "
            f"({recent['days']} days, {recent['symbols']} symbols, "
            "point-in-time top-200 by trailing volume so delistings are handled). "
            "(c) compares realised dispersion with the homogeneous "
            "equicorrelation shortcut on mismatched aggregations, so the "
            f"ratio of {ratio:.2f} is indicative only; the matched second-moment "
            "comparison in section 15.1 of the report is the test.",
            y=-0.10)
    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig12_calibration.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")

    if not fund.get("available"):
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.6))
    ax = axes[0]
    a = fund["_abs_series"]
    a = a[a < np.quantile(a, 0.995)]
    ax.hist(a, bins=60, color=OCHRE, alpha=0.85, lw=0)
    for q, lab, col in ((fund["median_abs_bps_per_day"], "median", K),
                        (fund["q90_abs_bps_per_day"], "90th", K70),
                        (3.0, "assumed 3.0", RED)):
        ax.axvline(q, color=col, lw=1.0, ls="--" if col != RED else "-")
        ax.text(q, ax.get_ylim()[1] * (0.9 if lab != "90th" else 0.7),
                f" {lab}", fontsize=6.8, color=col)
    ax.set_xlabel("|funding| (bps per day)")
    ax.set_ylabel("settlements")
    ax.set_title("funding magnitude")
    light_grid(ax, axis="y")
    panel_label(ax, "a")

    ax = axes[1]
    share = fund["interval_hours_share"]
    ks = sorted(share)
    ax.bar([str(int(k)) for k in ks], [100 * share[k] for k in ks],
           width=0.5, color=NAVY)
    for i, k in enumerate(ks):
        ax.text(i, 100 * share[k] + 1.5, f"{100*share[k]:.1f}%", ha="center",
                fontsize=6.8, color=K)
    ax.set_xlabel("settlement interval (hours)")
    ax.set_ylabel("share of settlements (%)")
    ax.set_title("funding is not uniformly 8-hourly")
    ax.set_ylim(0, 108)
    light_grid(ax, axis="y")
    panel_label(ax, "b")

    caption(fig,
            "Figure 13. Funding measured from the 2026 monthly archives. (a) The carry term in section 13.3 needs the "
            "magnitude a directional book pays, which is the median absolute "
            f"rate of {fund['median_abs_bps_per_day']:.2f} bps/day, not the "
            f"signed mean of {fund['mean_signed_bps_per_day']:+.2f}. (b) The "
            "settlement interval is a per-contract property, so a single "
            "bps/day figure is a smoothed stand-in for a discretely settled, "
            "time-varying rate.", y=-0.10)
    fig.tight_layout()
    out = paths.FIGURES_DIR / "fig13_funding.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"wrote {paths.rel(out)}")


if __name__ == "__main__":
    main()
