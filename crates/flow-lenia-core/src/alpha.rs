//! Diffusion-weight α field for the Flow-Lenia flow equation (paper Eq. 5).
//!
//! Mode-selection table (DESIGN.md §4.1.5):
//!
//! | `paper_strict` | α formula                                | per-channel? |
//! |----------------|------------------------------------------|--------------|
//! | `false` (default, JAX-compat) | `α(x, c) = clip((A_c(x) / β_A)², 0, 1)` | yes |
//! | `true` (paper Eq. 5)          | `α(x) = clip((A_Σ(x) / β_A)^n, 0, 1)`   | no (shared)  |
//!
//! In paper-strict mode `n` is the paper Eq. 5 exponent (UI-tunable).
//! In JAX-compat mode `n` is hard-coded to 2 (`flowlenia.py:98`); the
//! function signature reflects that — only `β_A` is taken.
//!
//! The public entry point [`alpha`] returns shape `(H, W, C)` in both
//! modes — paper-strict broadcasts the shared scalar field across all
//! channels so callers do not need to special-case the mode at the
//! point of use (and cannot accidentally read the wrong axis).

use crate::config::FlowLeniaConfig;
use crate::state::{ActivationField, AlphaField, AXIS_C};
use ndarray::{Array2, Array3};

/// Compute the α field for paper Eq. 5.
///
/// Dispatches on [`FlowLeniaConfig::paper_strict`]. Returns shape
/// `(H, W, C)` regardless of mode.
#[must_use]
pub fn alpha(a: &ActivationField, cfg: &FlowLeniaConfig) -> AlphaField {
    if cfg.paper_strict {
        let shared = alpha_paper(a, cfg.beta_a, cfg.n);
        let (h, w) = shared.dim();
        let c = a.shape()[2];
        // Broadcast (H, W) → (H, W, C).
        Array3::from_shape_fn((h, w, c), |(y, x, _)| shared[[y, x]])
    } else {
        alpha_jax_compat(a, cfg.beta_a)
    }
}

/// JAX-compat α: per-channel, exponent **hard-coded to 2**.
///
/// Mirrors JAX `flowlenia.py:98`
/// `alpha = jnp.clip((A[:, :, None, :] / self.cfg.C) ** 2, .0, 1.)`
/// (with `β_A` parameterised instead of the JAX `cfg.C` literal — see
/// DESIGN.md §4.1.5 for the rationale).
///
/// Note the signature *deliberately* does not take `n`: in JAX-compat
/// mode the exponent is fixed at 2 and any UI slider for `n` is
/// effectively ignored. Forcing the signature to reflect that
/// structurally prevents the "I changed n in the UI but α didn't move"
/// failure mode.
#[must_use]
pub(crate) fn alpha_jax_compat(a: &ActivationField, beta_a: f32) -> AlphaField {
    a.mapv(|v| {
        let z = v / beta_a;
        (z * z).clamp(0.0, 1.0)
    })
}

/// Paper-strict α: shared across channels, computed from `A_Σ = Σ_c A_c`.
///
/// Implements paper Eq. 5 verbatim:
/// `α(x) = clip((A_Σ(x) / β_A)^n, 0, 1)`. Returns `Array2<f32>` (no C
/// axis); the dispatcher [`alpha`] broadcasts it to `(H, W, C)` for
/// callers.
#[must_use]
pub(crate) fn alpha_paper(a: &ActivationField, beta_a: f32, n: f32) -> Array2<f32> {
    // `a_sum` has shape (H, W) — JAX-equivalent `A.sum(axis=-1)`.
    let a_sum = a.sum_axis(AXIS_C);
    a_sum.mapv(|v| (v / beta_a).powf(n).clamp(0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::FlowLeniaConfig;
    use approx::assert_relative_eq;
    use ndarray::{s, Array3};

    fn cfg_jax(beta_a: f32) -> FlowLeniaConfig {
        FlowLeniaConfig {
            paper_strict: false,
            beta_a,
            ..FlowLeniaConfig::default()
        }
    }

    fn cfg_paper(beta_a: f32, n: f32) -> FlowLeniaConfig {
        FlowLeniaConfig {
            paper_strict: true,
            beta_a,
            n,
            ..FlowLeniaConfig::default()
        }
    }

    /// JAX-compat mode produces a *different* α value for each channel
    /// (channel-1 saturated to 1.0 while channel-0 is at 0.25).
    #[test]
    fn alpha_jax_compat_per_channel_differs() {
        let mut a: ActivationField = Array3::zeros((4, 4, 2));
        a.slice_mut(s![.., .., 0]).fill(1.0);
        a.slice_mut(s![.., .., 1]).fill(2.0);

        let out = alpha(&a, &cfg_jax(2.0));

        // ch 0: clip((1/2)², 0, 1) = 0.25
        // ch 1: clip((2/2)², 0, 1) = 1.0
        for y in 0..4 {
            for x in 0..4 {
                assert_relative_eq!(out[[y, x, 0]], 0.25, epsilon = 1e-6);
                assert_relative_eq!(out[[y, x, 1]], 1.0, epsilon = 1e-6);
            }
        }
    }

    /// Paper-strict mode broadcasts a *single* α scalar field across
    /// all C channels: `out[y, x, 0] ≡ out[y, x, 1]` for every cell.
    #[test]
    fn alpha_paper_strict_shared_across_channels() {
        let mut a: ActivationField = Array3::zeros((4, 4, 2));
        a.slice_mut(s![.., .., 0]).fill(1.0);
        a.slice_mut(s![.., .., 1]).fill(2.0);

        let out = alpha(&a, &cfg_paper(2.0, 2.0));

        // A_Σ = 1 + 2 = 3; clip((3/2)², 0, 1) = clip(2.25, 0, 1) = 1.0
        for y in 0..4 {
            for x in 0..4 {
                assert_relative_eq!(out[[y, x, 0]], 1.0, epsilon = 1e-6);
                // Crucial: same value across the channel axis.
                assert_relative_eq!(out[[y, x, 1]], out[[y, x, 0]], epsilon = 1e-12);
            }
        }
    }

    /// paper_strict ON vs OFF must produce *measurably* different α
    /// fields on the same input. This is the "modes differ" guarantee
    /// from DESIGN.md §5.1.
    #[test]
    fn alpha_modes_differ() {
        // Use values that stay below the clip cap in both modes so the
        // formulas are exercised, not the upper saturator.
        let mut a: ActivationField = Array3::zeros((3, 3, 2));
        a.slice_mut(s![.., .., 0]).fill(0.5);
        a.slice_mut(s![.., .., 1]).fill(0.3);

        let jax_out = alpha(&a, &cfg_jax(2.0));
        let paper_out = alpha(&a, &cfg_paper(2.0, 2.0));

        // JAX: ch0 = (0.5/2)² = 0.0625, ch1 = (0.3/2)² = 0.0225
        // Paper: A_Σ = 0.8 → (0.8/2)² = 0.16 (shared)
        assert_relative_eq!(jax_out[[0, 0, 0]], 0.0625, epsilon = 1e-6);
        assert_relative_eq!(jax_out[[0, 0, 1]], 0.0225, epsilon = 1e-6);
        assert_relative_eq!(paper_out[[0, 0, 0]], 0.16, epsilon = 1e-6);
        assert_relative_eq!(paper_out[[0, 0, 1]], 0.16, epsilon = 1e-6);

        // Sanity: at least one cell genuinely differs.
        let any_differs = jax_out
            .indexed_iter()
            .any(|((y, x, c), &v)| (v - paper_out[[y, x, c]]).abs() > 1e-6);
        assert!(any_differs, "JAX and paper modes should give different α");
    }

    /// α clips to 1 when the squared/`n`-th-power term would exceed it.
    #[test]
    fn alpha_clip_upper_bound() {
        let a: ActivationField = Array3::from_elem((2, 2, 1), 100.0);
        for v in alpha(&a, &cfg_jax(2.0)).iter() {
            assert_relative_eq!(*v, 1.0, epsilon = 1e-6);
        }
        // Paper mode with `n=2`: (100/2)² = 2500, clipped to 1.
        for v in alpha(&a, &cfg_paper(2.0, 2.0)).iter() {
            assert_relative_eq!(*v, 1.0, epsilon = 1e-6);
        }
    }

    /// α = 0 when A is identically zero (both modes).
    #[test]
    fn alpha_clip_lower_bound() {
        let a: ActivationField = Array3::zeros((2, 2, 1));
        for v in alpha(&a, &cfg_jax(2.0)).iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
        for v in alpha(&a, &cfg_paper(2.0, 2.0)).iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    /// Hand-computed JAX-compat reference values.
    ///
    /// β_A = 2 throughout:
    ///   A_c = β_A   → α = (2/2)² = 1
    ///   A_c = β_A/2 → α = (1/2)² = 0.25
    ///   A_c = 0     → α = 0
    #[test]
    fn alpha_handcoded_jax_value() {
        let cfg = cfg_jax(2.0);

        let a: ActivationField = Array3::from_elem((1, 1, 1), 2.0);
        assert_relative_eq!(alpha(&a, &cfg)[[0, 0, 0]], 1.0, epsilon = 1e-6);

        let a: ActivationField = Array3::from_elem((1, 1, 1), 1.0);
        assert_relative_eq!(alpha(&a, &cfg)[[0, 0, 0]], 0.25, epsilon = 1e-6);

        let a: ActivationField = Array3::zeros((1, 1, 1));
        assert_relative_eq!(alpha(&a, &cfg)[[0, 0, 0]], 0.0, epsilon = 1e-6);
    }

    /// Hand-computed paper-strict reference values.
    ///
    /// β_A = 2, two channels (A_Σ = A_0 + A_1):
    ///   A_Σ = β_A,   n=2 → α = (2/2)²   = 1
    ///   A_Σ = β_A/2, n=2 → α = (1/2)²   = 0.25
    ///   A_Σ = β_A/2, n=4 → α = (1/2)⁴   = 0.0625
    #[test]
    fn alpha_handcoded_paper_value() {
        let mut a: ActivationField = Array3::zeros((1, 1, 2));

        a[[0, 0, 0]] = 1.0;
        a[[0, 0, 1]] = 1.0; // A_Σ = 2
        let out = alpha(&a, &cfg_paper(2.0, 2.0));
        assert_relative_eq!(out[[0, 0, 0]], 1.0, epsilon = 1e-6);
        // Shared across channels:
        assert_relative_eq!(out[[0, 0, 1]], 1.0, epsilon = 1e-6);

        a[[0, 0, 0]] = 0.5;
        a[[0, 0, 1]] = 0.5; // A_Σ = 1
        assert_relative_eq!(
            alpha(&a, &cfg_paper(2.0, 2.0))[[0, 0, 0]],
            0.25,
            epsilon = 1e-6
        );
        assert_relative_eq!(
            alpha(&a, &cfg_paper(2.0, 4.0))[[0, 0, 0]],
            0.0625,
            epsilon = 1e-6
        );
    }
}
