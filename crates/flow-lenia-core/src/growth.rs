//! Growth function G_i for Flow-Lenia (paper Eq. 2).
//!
//! Implements the Gaussian growth function:
//!
//! ```text
//! G_i(x; μ_i, σ_i) = 2 · exp(-(μ_i - x)² / (2 σ_i²)) - 1
//! ```
//!
//! Range `[-1, 1]`: peak `G = 1` at `x = μ_i`, asymptotes to `-1` as
//! `|x - μ_i| / σ_i → ∞`.
//!
//! **JAX cross-reference** (`utils.py:11-14`):
//!
//! ```python
//! bell = lambda x, m, s: jnp.exp(-((x-m)/s)**2 / 2)
//! def growth(U, m, s):
//!     return bell(U, m, s) * 2 - 1
//! ```
//!
//! The two formulations are mathematically identical: paper writes
//! `-(μ - x)² / (2 σ²)`, JAX writes `-((x - m) / s)² / 2`. They agree under
//! the trivial mapping `m = μ`, `s = σ`. Note that `σ` here is the *growth*
//! width on a single kernel ([`crate::params::KernelEntry::sigma`]), **not**
//! the *reintegration* width [`crate::config::FlowLeniaConfig::sigma`] — the
//! two are unrelated despite sharing the symbol in the literature.

/// Gaussian "bell" function `bell(x; m, s) = exp(-((x - m) / s)² / 2)`.
///
/// Mirrors JAX `utils.py:11`. Exposed as a named helper so that:
///   - the `2·bell − 1` rescaling in [`growth`] is unambiguous, and
///   - unit tests can pin the bell-shape (peak, symmetry) without coupling
///     to the `[-1, 1]` rescaling.
#[inline]
#[must_use]
pub fn bell(x: f32, m: f32, s: f32) -> f32 {
    let z = (x - m) / s;
    (-(z * z) / 2.0).exp()
}

/// Growth function `G(x; μ, σ) = 2 · bell(x; μ, σ) - 1`.
///
/// Direct port of JAX `utils.py:13-14` and an exact restatement of paper
/// Eq. 2. Range `[-1, 1]`.
///
/// # Examples
///
/// ```
/// use flow_lenia_core::growth::growth;
///
/// // Peak at x = μ.
/// assert!((growth(0.15, 0.15, 0.02) - 1.0).abs() < 1e-6);
/// // Far tail → -1.
/// assert!((growth(1.0, 0.15, 0.02) - (-1.0)).abs() < 1e-6);
/// ```
#[inline]
#[must_use]
pub fn growth(x: f32, mu: f32, sigma: f32) -> f32 {
    2.0 * bell(x, mu, sigma) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `bell(μ; μ, σ) = exp(0) = 1` — the peak of the Gaussian.
    #[test]
    fn bell_peak_is_one() {
        for (mu, sigma) in [(0.0_f32, 0.001_f32), (0.15, 0.02), (0.5, 0.1), (0.99, 0.18)] {
            assert_relative_eq!(bell(mu, mu, sigma), 1.0, epsilon = 1e-6);
        }
    }

    /// `bell(μ + d; μ, σ) = bell(μ - d; μ, σ)` — radial symmetry.
    /// This is the defining symmetry of the Gaussian, so this test guards
    /// against accidentally writing `(x - m)` as `(m - x)` and forgetting
    /// the sign-cancellation in the `z²` (which would still be symmetric,
    /// but at any future refactor the property must hold).
    #[test]
    fn bell_is_symmetric_about_mu() {
        let mu = 0.15;
        let sigma = 0.02;
        for d in [0.005_f32, 0.01, 0.04, 0.1] {
            assert_relative_eq!(
                bell(mu + d, mu, sigma),
                bell(mu - d, mu, sigma),
                epsilon = 1e-6
            );
        }
    }

    /// `bell(μ + σ; μ, σ) = exp(-0.5)` — known value at one-σ offset.
    ///
    /// Hand-computed: `z = (μ+σ - μ)/σ = 1`, `bell = exp(-1²/2) = exp(-0.5) ≈ 0.60653...`.
    #[test]
    fn bell_known_value_at_one_sigma_offset() {
        let mu = 0.15;
        let sigma = 0.02;
        let expected = (-0.5_f32).exp();
        assert_relative_eq!(bell(mu + sigma, mu, sigma), expected, epsilon = 1e-6);
        assert_relative_eq!(bell(mu - sigma, mu, sigma), expected, epsilon = 1e-6);
    }

    /// 2025 paper Eq. 2: `G(x; μ, σ) = 2·exp(-(μ-x)²/(2σ²)) - 1`.
    /// At `x = μ`: `G = 2·exp(0) - 1 = 2·1 - 1 = 1`.
    /// JAX reference: `utils.py:13` (`growth(U, m, s) = bell(U, m, s)*2 - 1`).
    #[test]
    fn growth_at_mu_is_one() {
        for (mu, sigma) in [(0.0_f32, 0.001_f32), (0.15, 0.02), (0.5, 0.1), (0.99, 0.18)] {
            assert_relative_eq!(growth(mu, mu, sigma), 1.0, epsilon = 1e-6);
        }
    }

    /// `|x - μ| / σ → ∞` ⇒ `bell → 0` ⇒ `G → -1`.
    ///
    /// Practical test: at 50σ, `exp(-1250)` underflows to f32 0 (smallest
    /// positive subnormal is ≈ `1.4e-45`, `exp(-1250) ≈ 1e-543`). So G
    /// reaches exactly `-1`, not just asymptotically.
    #[test]
    fn growth_at_infinity_approaches_minus_one() {
        let mu = 0.15_f32;
        let sigma = 0.02_f32;
        assert_relative_eq!(growth(mu + 50.0 * sigma, mu, sigma), -1.0, epsilon = 1e-6);
        assert_relative_eq!(growth(mu - 50.0 * sigma, mu, sigma), -1.0, epsilon = 1e-6);
        // Edge of the typical activation range (x = 1.0 with μ = 0.15, σ = 0.02):
        // |x - μ|/σ = 42.5, still far enough for total underflow.
        assert_relative_eq!(growth(1.0, mu, sigma), -1.0, epsilon = 1e-6);
    }

    /// Output range stays in `[-1, 1]` for the full activation domain and
    /// the full JAX growth-parameter range
    /// (`μ ∈ [0.05, 0.5]`, `σ ∈ [0.001, 0.18]` — see
    /// `flow_lenia_core::params::KernelEntry::sample_random`).
    ///
    /// 1001 sample points × several (μ, σ) pairs gives ~5k checks — fast
    /// enough to run on every `cargo test` and dense enough to catch a
    /// sign flip or a missing `2·`.
    #[test]
    fn growth_range_is_minus_one_to_one() {
        let params = [
            (0.05_f32, 0.001_f32),
            (0.15, 0.02),
            (0.5, 0.1),
            (0.5, 0.18),
            (0.05, 0.18),
        ];
        for x_i in 0..=1000_i32 {
            let x = x_i as f32 / 1000.0; // x ∈ [0, 1] step 0.001
            for &(mu, sigma) in &params {
                let g = growth(x, mu, sigma);
                // Strict bounds: G is *exactly* in [-1, 1] mathematically.
                // Allow one ulp of f32 slack to cover the `2·bell − 1`
                // rounding when bell is extremely close to 0 or 1.
                assert!(g >= -1.0 - f32::EPSILON, "G({x}; {mu}, {sigma}) = {g} < -1");
                assert!(g <= 1.0 + f32::EPSILON, "G({x}; {mu}, {sigma}) = {g} > 1");
            }
        }
    }

    /// Known value at one-σ offset for the *growth* function (not just bell):
    /// `G(μ+σ; μ, σ) = 2·exp(-0.5) - 1 ≈ 0.213_061_32`.
    #[test]
    fn growth_known_value_at_one_sigma_offset() {
        let mu = 0.15;
        let sigma = 0.02;
        let expected = 2.0 * (-0.5_f32).exp() - 1.0;
        assert_relative_eq!(growth(mu + sigma, mu, sigma), expected, epsilon = 1e-6);
        // Symmetric on the other side.
        assert_relative_eq!(growth(mu - sigma, mu, sigma), expected, epsilon = 1e-6);
    }

    /// `growth` and `bell` are consistent: `growth = 2·bell − 1` everywhere.
    ///
    /// Guards against future refactors that inline `growth` and accidentally
    /// drop the `−1` (which would still pass `growth_at_mu_is_one` because
    /// `2·1 = 2 ≠ 1` would fail loudly, but not all tests cover every linear
    /// combination).
    #[test]
    fn growth_equals_two_bell_minus_one() {
        let mu = 0.15_f32;
        let sigma = 0.02_f32;
        for x in [0.0_f32, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 1.0] {
            assert_relative_eq!(
                growth(x, mu, sigma),
                2.0 * bell(x, mu, sigma) - 1.0,
                epsilon = 1e-7
            );
        }
    }
}
