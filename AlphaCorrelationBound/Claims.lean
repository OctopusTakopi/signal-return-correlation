import Mathlib

/-!
# Formal checks for the alpha-correlation bound

This file verifies the deterministic algebra used by `README.md`. It does not
attempt to certify empirical inputs, Monte Carlo output, or decimal
approximations to Gaussian tail probabilities.
-/

namespace AlphaCorrelationBound

noncomputable section

/-- Exact correlation when the signal loading is `β` and horizon residual
volatility is `v = σ * sqrt τ`. -/
def corrExact (β v : ℝ) : ℝ :=
  β / Real.sqrt (β ^ 2 + v ^ 2)

/-- Exact inversion of `corrExact` for a nonnegative correlation below one. -/
def betaExact (ρ v : ℝ) : ℝ :=
  ρ * v / Real.sqrt (1 - ρ ^ 2)

/-- The source's small-correlation approximation to `betaExact`. -/
def betaApprox (ρ v : ℝ) : ℝ :=
  ρ * v

/-- Exact correlation produced when the loading-to-noise ratio is `r`. -/
def corrFromLoadingRatio (r : ℝ) : ℝ :=
  r / Real.sqrt (1 + r ^ 2)

/-- The source's correlation floor, with horizon volatility `v = σ * sqrt τ`. -/
def rhoFloor (c κ v : ℝ) : ℝ :=
  c / (κ * v)

/-- Population R² of the simple regression of returns on one signal. -/
def regressionR2 (β v : ℝ) : ℝ :=
  β ^ 2 / (β ^ 2 + v ^ 2)

/-- Simple-regression R² written as a function of IC. -/
def r2FromIC (ρ : ℝ) : ℝ :=
  ρ ^ 2

/-- Exact R² floor when `v` is residual rather than total volatility. -/
def r2FloorResidual (c κ v : ℝ) : ℝ :=
  rhoFloor c κ v ^ 2 / (1 + rhoFloor c κ v ^ 2)

/-- Explained-to-unexplained variance ratio. -/
def r2Odds (q : ℝ) : ℝ :=
  q / (1 - q)

/-- No-trade band implied by cost `c` and loading `β`. -/
def noTradeBand (c β : ℝ) : ℝ :=
  c / β

/-- Effective breadth for `n` equicorrelated bets. -/
def effectiveBreadth (n ρa : ℝ) : ℝ :=
  n / (1 + (n - 1) * ρa)

/-- Fee-and-carry edge as a quadratic in `s = sqrt τ`. -/
def netEdgeSqrtHorizon (a c f s : ℝ) : ℝ :=
  a * s - c - f * s ^ 2

/-- Residual volatility in the one-factor construction used in README §13. -/
def residualVol (σ factorCorr : ℝ) : ℝ :=
  σ * Real.sqrt (1 - factorCorr ^ 2)

/-- Pairwise asset correlation implied when two assets have the same
correlation to one common factor and independent residuals. -/
def pairwiseCorrFromFactorCorr (factorCorr : ℝ) : ℝ :=
  factorCorr ^ 2

/-- Approximate observations required for a time-series correlation t-stat. -/
def tsObservations (ρ t : ℝ) : ℝ :=
  (t * (1 - ρ ^ 2) / ρ) ^ 2 + 1

/-- Approximate dates required for a stable cross-sectional IC t-stat. -/
def csDates (ρ n t : ℝ) : ℝ :=
  t ^ 2 / (n * ρ ^ 2)

theorem corr_exact_sq (β v : ℝ) :
    corrExact β v ^ 2 = β ^ 2 / (β ^ 2 + v ^ 2) := by
  rw [corrExact, div_pow]
  congr 1
  rw [Real.sq_sqrt]
  positivity

theorem regression_r2_eq_corr_sq (β v : ℝ) :
    regressionR2 β v = corrExact β v ^ 2 := by
  rw [corr_exact_sq]
  rfl

theorem regression_r2_eq_ic_sq (β v ρ : ℝ)
    (hρ : corrExact β v = ρ) :
    regressionR2 β v = r2FromIC ρ := by
  rw [regression_r2_eq_corr_sq, hρ]
  rfl

theorem r2_ic_multiple (m ρ : ℝ) :
    r2FromIC (m * ρ) = m ^ 2 * r2FromIC ρ := by
  simp only [r2FromIC, mul_pow]

theorem beta_exact_sq (ρ v : ℝ) (hρ : ρ ^ 2 < 1) :
    betaExact ρ v ^ 2 = ρ ^ 2 * v ^ 2 / (1 - ρ ^ 2) := by
  rw [betaExact, div_pow, mul_pow, Real.sq_sqrt (by positivity)]

theorem corr_beta_exact_sq (ρ v : ℝ) (hρ : ρ ^ 2 < 1) (hv : v ≠ 0) :
    corrExact (betaExact ρ v) v ^ 2 = ρ ^ 2 := by
  have hden : 1 - ρ ^ 2 ≠ 0 := ne_of_gt (sub_pos.mpr hρ)
  rw [corr_exact_sq, beta_exact_sq ρ v hρ]
  field_simp [hden, hv]
  ring

theorem corr_beta_exact
    (ρ v : ℝ) (hρ0 : 0 ≤ ρ) (hρ : ρ ^ 2 < 1) (hv : 0 < v) :
    corrExact (betaExact ρ v) v = ρ := by
  have hsq := corr_beta_exact_sq ρ v hρ (ne_of_gt hv)
  have hcorr : 0 ≤ corrExact (betaExact ρ v) v := by
    simp only [corrExact, betaExact]
    positivity
  nlinarith

theorem corr_of_loading_ratio_sq (r v : ℝ) (hv : v ≠ 0) :
    corrExact (betaApprox r v) v ^ 2 = r ^ 2 / (1 + r ^ 2) := by
  rw [corr_exact_sq]
  simp only [betaApprox, mul_pow]
  field_simp
  ring

theorem residual_r2_floor_eq_corr_sq (c κ v : ℝ) :
    r2FloorResidual c κ v =
      corrFromLoadingRatio (rhoFloor c κ v) ^ 2 := by
  simp only [r2FloorResidual, corrFromLoadingRatio, div_pow]
  rw [Real.sq_sqrt]
  positivity

theorem r2_odds_eq_loading_ratio_sq
    (β v : ℝ) (hv : v ≠ 0) :
    r2Odds (regressionR2 β v) = β ^ 2 / v ^ 2 := by
  simp only [r2Odds, regressionR2]
  field_simp
  ring

theorem regression_r2_at_cost_boundary
    (c κ v : ℝ) (hκ : κ ≠ 0) (hv : v ≠ 0) :
    regressionR2 (c / κ) v = r2FloorResidual c κ v := by
  simp only [regressionR2, r2FloorResidual, rhoFloor]
  field_simp
  ring

theorem floor_from_cost_boundary
    (ρ c κ v : ℝ) (hκ : κ ≠ 0) (hv : v ≠ 0)
    (hboundary : κ * (ρ * v) = c) :
    ρ = rhoFloor c κ v := by
  rw [rhoFloor]
  apply (eq_div_iff (mul_ne_zero hκ hv)).2
  nlinarith

theorem hedge_quadruples_floor
    (c κ v : ℝ) (hκ : κ ≠ 0) (hv : v ≠ 0) :
    rhoFloor (2 * c) κ (v / 2) = 4 * rhoFloor c κ v := by
  simp only [rhoFloor]
  field_simp
  ring

theorem band_depends_only_on_multiple
    (c κ v m : ℝ) (hc : c ≠ 0) (hκ : κ ≠ 0) (hv : v ≠ 0) (hm : m ≠ 0) :
    noTradeBand c (betaApprox (m * rhoFloor c κ v) v) = κ / m := by
  simp only [noTradeBand, betaApprox, rhoFloor]
  field_simp

theorem equity_floor :
    rhoFloor (5 / 10000 : ℝ) 3 (3 / 100) = 1 / 180 := by
  norm_num [rhoFloor]

theorem factor_hedged_floor :
    rhoFloor (10 / 10000 : ℝ) 3 (15 / 1000) = 1 / 45 := by
  norm_num [rhoFloor]

theorem factor_hedged_is_four_times_equity :
    rhoFloor (10 / 10000 : ℝ) 3 (15 / 1000) =
      4 * rhoFloor (5 / 10000 : ℝ) 3 (3 / 100) := by
  norm_num [rhoFloor]

theorem floor_multiple_one_point_five_band :
    noTradeBand (5 / 10000 : ℝ)
        (betaApprox ((3 / 2) * rhoFloor (5 / 10000 : ℝ) 3 (3 / 100))
          (3 / 100)) =
      2 := by
  norm_num [noTradeBand, betaApprox, rhoFloor]

theorem floor_multiple_two_band :
    noTradeBand (5 / 10000 : ℝ)
        (betaApprox (2 * rhoFloor (5 / 10000 : ℝ) 3 (3 / 100))
          (3 / 100)) =
      3 / 2 := by
  norm_num [noTradeBand, betaApprox, rhoFloor]

theorem effective_breadth_identity
    (n ρa : ℝ) (hn : n ≠ 0) :
    1 / ((1 + (n - 1) * ρa) / n) = effectiveBreadth n ρa := by
  simp only [effectiveBreadth]
  field_simp

theorem residual_vol_sq
    (σ factorCorr : ℝ) (hCorr : factorCorr ^ 2 ≤ 1) :
    residualVol σ factorCorr ^ 2 = σ ^ 2 * (1 - factorCorr ^ 2) := by
  simp only [residualVol, mul_pow]
  rw [Real.sq_sqrt (sub_nonneg.mpr hCorr)]

theorem factor_corr_seventy_implies_pairwise_forty_nine :
    pairwiseCorrFromFactorCorr (7 / 10 : ℝ) = 49 / 100 := by
  norm_num [pairwiseCorrFromFactorCorr]

theorem directional_breadth_for_factor_corr_seventy :
    effectiveBreadth 200 (pairwiseCorrFromFactorCorr (7 / 10 : ℝ)) =
      20000 / 9851 := by
  norm_num [effectiveBreadth, pairwiseCorrFromFactorCorr]

theorem estimation_sample_ratio
    (ρ n t : ℝ) (hρ : ρ ≠ 0) (hn : n ≠ 0) (ht : t ≠ 0) :
    tsObservations ρ t / csDates ρ n t =
      n * (1 - ρ ^ 2) ^ 2 + n * ρ ^ 2 / t ^ 2 := by
  simp only [tsObservations, csDates]
  field_simp

theorem carry_complete_square
    (a c f s : ℝ) (hf : f ≠ 0) :
    a ^ 2 / (4 * f) - c - netEdgeSqrtHorizon a c f s =
      (2 * f * s - a) ^ 2 / (4 * f) := by
  simp only [netEdgeSqrtHorizon]
  field_simp
  ring

theorem carry_peak_attained
    (a c f : ℝ) (hf : f ≠ 0) :
    netEdgeSqrtHorizon a c f (a / (2 * f)) = a ^ 2 / (4 * f) - c := by
  simp only [netEdgeSqrtHorizon]
  field_simp
  ring

theorem carry_peak_positive_iff
    (a c f : ℝ) (hf : 0 < f) :
    a ^ 2 / (4 * f) - c > 0 ↔ a ^ 2 > 4 * f * c := by
  constructor <;> intro h
  · have h4f : 0 < 4 * f := by positivity
    have hdiv : c < a ^ 2 / (4 * f) := by linarith
    have hmul := (lt_div_iff₀ h4f).mp hdiv
    nlinarith
  · have h4f : 0 < 4 * f := by positivity
    have hmul : c * (4 * f) < a ^ 2 := by nlinarith
    have hdiv := (lt_div_iff₀ h4f).mpr hmul
    linarith

end

end AlphaCorrelationBound
