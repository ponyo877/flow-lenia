//! Overlap area `I(x', x)` for reintegration tracking (paper Eq. 6).
//!
//! The reintegration scheme moves mass from a source cell `x'` toward a
//! distribution `D(μ, σ)` — a uniform square of half-width `σ` centred at
//! `μ = x' + dt·F`. The fraction of mass that lands in a *target* cell `x`
//! is the integral of `D` over the unit square at `x`:
//!
//! ```text
//! I(x', x) = ∫_{Ω(x)} D(μ, σ) dA
//! ```
//!
//! For axis-aligned squares this collapses to the product of two 1D
//! overlap widths divided by the area `4σ²` of the source distribution.
//!
//! **JAX reference** (`reintegration_tracking.py:57-58`):
//! ```python
//! sz   = .5 - dpmu + self.sigma
//! area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)), axis=2) / (4*self.sigma**2)
//! ```
//!
//! The `min(1, 2σ)` upper clamp is critical for mass conservation in the
//! high-temperature regime (`σ > 0.5`) — without it the 1D overlap width
//! could exceed both the cell width (`1`) and the distribution width
//! (`2σ`), and the resulting area would *not* integrate to 1 over the
//! whole grid.
//!
//! Callers handle border conditions (torus minimum-modulo wrap, wall
//! clipping) on the signed `dpmu` *before* passing it in; this function
//! itself takes signed distances and applies `abs` internally — see the
//! design rationale in the doc comment of [`overlap_area`].

/// Overlap area `I` for a single source-target cell pair.
///
/// Arguments:
/// - `dpmu_y`, `dpmu_x`: **signed** distance from the distribution centre
///   to the target-cell centre, in cells. `f32::abs` is applied internally
///   so the sign is irrelevant — callers can pass `(target_centre − μ)`
///   directly without taking an absolute value first. This is a
///   deliberate API choice (M1.10 design decision): asking callers to
///   pre-abs would be an unenforceable precondition, and the torus
///   minimum-modulo wrap upstream operates on the *signed* distance
///   anyway.
/// - `sigma`: distribution half-width (paper Eq. 6 `s`, JAX
///   `Config.sigma`). Must be positive in practice; `σ = 0` is
///   mathematically undefined (the formula divides by `4σ²`) and the
///   function returns `NaN`, matching JAX. UI parameter range is
///   `[0.1, 2.0]` per DESIGN.md §6 / §7 so this case is not exercised
///   at runtime.
///
/// Returns: `I ∈ [0, 1]` for valid `σ > 0`. Zero when the cells do not
/// overlap; one in the degenerate limit `σ → 0+` at `dpmu = 0` (the
/// distribution collapses to a delta on the target cell centre).
#[inline]
#[must_use]
pub fn overlap_area(dpmu_y: f32, dpmu_x: f32, sigma: f32) -> f32 {
    // Internal abs — see the API doc above for the rationale.
    let abs_y = dpmu_y.abs();
    let abs_x = dpmu_x.abs();

    // `min(1, 2σ)` upper clamp (JAX utils.py:58).
    let upper = (2.0 * sigma).min(1.0);

    // 1D overlap widths.
    let sz_y = (0.5 - abs_y + sigma).clamp(0.0, upper);
    let sz_x = (0.5 - abs_x + sigma).clamp(0.0, upper);

    // Normalisation by the source distribution area (4σ²).
    (sz_x * sz_y) / (4.0 * sigma * sigma)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// At the distribution centre (`dpmu = (0, 0)`) with the default
    /// `σ = 0.65`, the formula degenerates to:
    ///   `sz_y = sz_x = clamp(0.5 + 0.65, 0, min(1, 1.3)) = clamp(1.15, 0, 1) = 1`
    ///   `I = (1 · 1) / (4 · 0.65²) = 1 / 1.69 ≈ 0.591716`
    ///
    /// This is the *peak* value of `I` for `σ = 0.65` — every other
    /// `(dpmu_y, dpmu_x)` pair gives a smaller or equal value at this σ.
    #[test]
    fn overlap_area_centred_at_distribution_center() {
        let i = overlap_area(0.0, 0.0, 0.65);
        assert_relative_eq!(i, 1.0 / 1.69, epsilon = 1e-5);
    }

    /// Cells far enough away from the distribution centre (`|dpmu| > 0.5 + σ`)
    /// receive zero mass: the `clamp(..., 0, ...)` lower bound zeroes both
    /// overlap widths.
    #[test]
    fn overlap_area_far_distance_returns_zero() {
        // 10 cells away, σ = 0.65 — well outside the half-width-1 reach.
        let i = overlap_area(10.0, 10.0, 0.65);
        assert_eq!(i, 0.0);
        // Only one axis far away — still zero, since `sz_y = 0` zeroes the product.
        let i = overlap_area(10.0, 0.1, 0.65);
        assert_eq!(i, 0.0);
        let i = overlap_area(0.1, 10.0, 0.65);
        assert_eq!(i, 0.0);
    }

    /// Hand-computed partial overlap: `dpmu = (0.3, 0.0)`, `σ = 0.5`.
    ///   `upper  = min(1, 2·0.5) = 1`
    ///   `sz_y   = clamp(0.5 - 0.3 + 0.5, 0, 1) = clamp(0.7, 0, 1) = 0.7`
    ///   `sz_x   = clamp(0.5 - 0   + 0.5, 0, 1) = clamp(1.0, 0, 1) = 1.0`
    ///   `I      = (0.7 · 1.0) / (4 · 0.25)    = 0.7 / 1.0 = 0.7`
    #[test]
    fn overlap_area_partial_overlap_handcoded() {
        let i = overlap_area(0.3, 0.0, 0.5);
        assert_relative_eq!(i, 0.7, epsilon = 1e-6);
    }

    /// `σ < 0.5`: the `min(1, 2σ)` upper clamp picks `2σ`.
    ///
    ///   σ = 0.3 → upper = 0.6
    ///   dpmu = 0: sz_y = sz_x = clamp(0.5 + 0.3, 0, 0.6) = 0.6
    ///   I = (0.6²) / (4 · 0.09) = 0.36 / 0.36 = **1.0**
    ///
    /// This shows that with `σ < 0.5` the distribution is narrower than
    /// the cell and concentrates 100 % of its mass at the centre cell.
    /// Removing the `min(1, 2σ)` clamp would let `sz` reach `0.8` here
    /// and produce `0.64 / 0.36 ≈ 1.78` — far above 1, breaking mass
    /// conservation.
    #[test]
    fn overlap_area_sigma_below_half_uses_2sigma_clip() {
        let i = overlap_area(0.0, 0.0, 0.3);
        assert_relative_eq!(i, 1.0, epsilon = 1e-6);
    }

    /// `σ > 0.5`: the `min(1, 2σ)` upper clamp picks `1`.
    ///
    ///   σ = 0.65 → upper = 1.0
    ///   dpmu = 0: sz_y = sz_x = clamp(0.5 + 0.65, 0, 1.0) = 1.0
    ///   I = (1²) / (4 · 0.4225) = 1 / 1.69 ≈ 0.591716
    ///
    /// Distribution overflows the unit cell, so only a fraction of its
    /// mass lands here even at the centre.
    #[test]
    fn overlap_area_sigma_above_half_uses_one_clip() {
        let i = overlap_area(0.0, 0.0, 0.65);
        assert_relative_eq!(i, 1.0 / 1.69, epsilon = 1e-5);
    }

    /// Function takes `|dpmu_y|` and `|dpmu_x|` internally — sign of the
    /// inputs is irrelevant. Asserted to **bit-exact equality** because
    /// `f32::abs` is deterministic and the rest of the formula consumes
    /// only the absolute value (no path-dependent rounding).
    #[test]
    fn overlap_area_is_symmetric_in_sign() {
        let sigma = 0.5_f32;
        for (dy, dx) in [(0.3_f32, 0.2_f32), (0.0, 0.4), (0.7, 0.0), (0.5, 0.5)] {
            let pp = overlap_area(dy, dx, sigma);
            let pn = overlap_area(dy, -dx, sigma);
            let np = overlap_area(-dy, dx, sigma);
            let nn = overlap_area(-dy, -dx, sigma);
            assert_eq!(pp.to_bits(), pn.to_bits(), "(+, -) at ({dy}, {dx})");
            assert_eq!(pp.to_bits(), np.to_bits(), "(-, +) at ({dy}, {dx})");
            assert_eq!(pp.to_bits(), nn.to_bits(), "(-, -) at ({dy}, {dx})");
        }
    }

    /// **Partition of unity** — the uniform square distribution `D(μ, σ)`
    /// integrates to 1, so summing `I(x', x)` over a target grid that
    /// fully contains the distribution must recover 1.
    ///
    /// This verifies the `1 / (4σ²)` normalisation in [`overlap_area`]
    /// independently of the full Eq. 6 mass conservation (M1.11 covers
    /// that, including the source-mass weighting).
    ///
    /// Choice of `μ = (3.7, 4.2)`: explicitly non-aligned with cell
    /// centres so the test exercises the `clamp(...)` lower-bound zero
    /// branch on the cells whose target-centre is more than `0.5 + σ`
    /// away from `μ`. A 20×20 grid easily contains the `|dpmu| < 0.5+σ`
    /// support for `σ = 0.65`.
    #[test]
    fn overlap_area_partition_of_unity_across_target_grid() {
        let mu_y = 3.7_f32;
        let mu_x = 4.2_f32;
        let sigma = 0.65_f32;

        let mut total = 0.0_f32;
        for ty in 0..20 {
            for tx in 0..20 {
                let dpmu_y = (ty as f32 + 0.5) - mu_y;
                let dpmu_x = (tx as f32 + 0.5) - mu_x;
                total += overlap_area(dpmu_y, dpmu_x, sigma);
            }
        }
        assert_relative_eq!(total, 1.0, epsilon = 1e-5);
    }

    /// Partition-of-unity should hold across the *entire* valid `σ` range
    /// (Table 1: `s = 0.65` default, UI slider `[0.1, 2.0]`). This
    /// regression-locks the `min(1, 2σ)` clamp: removing it would break
    /// the `σ = 0.3` case (the distribution narrower than the cell would
    /// over-count) and the `σ = 1.5` case (the distribution wider than
    /// the cell would over-count too).
    #[test]
    fn overlap_area_partition_of_unity_across_sigma_range() {
        for &sigma in &[0.1_f32, 0.3, 0.5, 0.65, 1.0, 1.5, 2.0] {
            let mu_y = 7.3_f32;
            let mu_x = 7.7_f32;
            // Make the grid wide enough for the `σ = 2.0` reach
            // (centre ± (0.5 + σ) = ± 2.5 → up to 5 cells of support;
            // 15×15 with μ near the middle is more than enough).
            let mut total = 0.0_f32;
            for ty in 0..15 {
                for tx in 0..15 {
                    let dpmu_y = (ty as f32 + 0.5) - mu_y;
                    let dpmu_x = (tx as f32 + 0.5) - mu_x;
                    total += overlap_area(dpmu_y, dpmu_x, sigma);
                }
            }
            assert_relative_eq!(total, 1.0, epsilon = 1e-5,);
            // Add context to the assert above on failure.
            assert!((total - 1.0).abs() < 1e-5, "σ = {sigma}: total = {total}");
        }
    }

    /// `σ = 0` is mathematically undefined (formula divides by `4σ² = 0`).
    /// We return `NaN`, matching JAX (`jnp.clip(sz, 0, 0) = 0` and then
    /// `0 / 0 = NaN`). Production parameter ranges never reach 0
    /// (`σ ∈ [0.001, 2.0]` per `KernelEntry::sample_random` and the UI
    /// slider lower bound `0.1`), so this is purely a documentation test.
    #[test]
    fn overlap_area_zero_sigma_is_nan() {
        let i = overlap_area(0.0, 0.0, 0.0);
        assert!(i.is_nan(), "σ = 0 should be NaN, got {i}");
    }
}
