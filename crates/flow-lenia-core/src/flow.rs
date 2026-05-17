//! Flow field `F` for the Flow-Lenia step (paper Eq. 5).
//!
//! ```text
//! F_i = (1 - α) · ∇U_i - α · ∇A_Σ
//! ```
//!
//! where:
//! - `α` is per-channel in JAX-compat (default) mode and shared-across-C
//!   in paper-strict mode — [`crate::alpha::alpha`] hides the distinction;
//! - `∇U_i` is the per-channel gradient of the affinity (M1.12 will compute
//!   `U`; for M1.9 it is just an input);
//! - `∇A_Σ` is the gradient of the total mass (shared across channels by
//!   construction).
//!
//! JAX cross-reference (`flowlenia.py:94-101`):
//! ```python
//! nabla_U = sobel(U)
//! nabla_A = sobel(A.sum(axis=-1, keepdims=True))
//! alpha   = jnp.clip((A[:, :, None, :]/self.cfg.C)**2, .0, 1.)
//! F = nabla_U * (1 - alpha) - nabla_A * alpha
//! ```
//!
//! Our signature takes the three inputs explicitly rather than computing
//! them inside `flow` itself — this keeps the unit test for Eq. 5 free of
//! Sobel-precision noise, and lets the integration step in M1.13 share
//! intermediate buffers if it wants to.

use crate::state::{AlphaField, FlowField, FLOW_DX, FLOW_DY};
use ndarray::{Array3, Array4};

/// Compute the flow field `F` per paper Eq. 5.
///
/// Inputs (all using the (H, W, …, C) JAX-axis convention):
/// - `grad_u`: per-channel gradient of `U`. Shape `(H, W, 2, C)`.
/// - `grad_a_sum`: gradient of total mass `A_Σ`. Shape `(H, W, 2)`.
/// - `alpha`: per-channel α (shared values in paper-strict mode).
///   Shape `(H, W, C)`.
///
/// Output: [`FlowField`] (`Array4<f32>`) of shape `(H, W, 2, C)`.
///
/// # Panics
///
/// Panics if the input shapes are inconsistent (`grad_u` defines `(H, W, C)`
/// and `grad_a_sum`/`alpha` must match it). Shape mismatches are upstream
/// programming errors, not data-dependent failures, so a panic is the
/// right escalation here.
#[must_use]
pub fn flow(grad_u: &Array4<f32>, grad_a_sum: &Array3<f32>, alpha: &AlphaField) -> FlowField {
    let (h, w, two, c) = grad_u.dim();
    assert_eq!(two, 2, "grad_u AXIS_FLOW dimension must be 2 (got {two})");
    assert_eq!(
        grad_a_sum.dim(),
        (h, w, 2),
        "grad_a_sum shape mismatch (expected ({h}, {w}, 2), got {:?})",
        grad_a_sum.dim()
    );
    assert_eq!(
        alpha.dim(),
        (h, w, c),
        "alpha shape mismatch (expected ({h}, {w}, {c}), got {:?})",
        alpha.dim()
    );

    let mut out = Array4::<f32>::zeros((h, w, 2, c));
    // Loop ordering mirrors the (H, W, 2, C) layout so the inner-most
    // dimension is contiguous. For each (y, x, ci) we read α once and
    // ∇A_Σ once per (dy, dx); the per-channel grad_u read picks up the
    // remaining variance.
    for y in 0..h {
        for x in 0..w {
            // ∇A_Σ is shared across channels — hoist it out of the C loop.
            let gas_dy = grad_a_sum[[y, x, FLOW_DY]];
            let gas_dx = grad_a_sum[[y, x, FLOW_DX]];
            for ci in 0..c {
                let a = alpha[[y, x, ci]];
                let one_minus_a = 1.0 - a;
                out[[y, x, FLOW_DY, ci]] = one_minus_a * grad_u[[y, x, FLOW_DY, ci]] - a * gas_dy;
                out[[y, x, FLOW_DX, ci]] = one_minus_a * grad_u[[y, x, FLOW_DX, ci]] - a * gas_dx;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::{ActivationField, FlowFieldExt};
    use approx::assert_relative_eq;
    use ndarray::Array3;

    /// Build a non-trivial `(H, W, 2, C)` `∇U` so the tests catch any
    /// axis-swap inside `flow`.
    fn build_grad_u(h: usize, w: usize, c: usize) -> Array4<f32> {
        Array4::from_shape_fn((h, w, 2, c), |(y, x, fi, ci)| {
            (y as i32 - x as i32 + (fi as i32) * 13 + (ci as i32) * 31) as f32 / 7.0
        })
    }

    fn build_grad_a_sum(h: usize, w: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, 2), |(y, x, fi)| {
            (y as i32 + 2 * x as i32 + (fi as i32) * 5) as f32 / 11.0
        })
    }

    /// `α ≡ 0` → `F = ∇U` everywhere (the diffusion term drops out).
    #[test]
    fn flow_zero_alpha_is_pure_affinity() {
        let (h, w, c) = (4, 4, 2);
        let grad_u = build_grad_u(h, w, c);
        let grad_a_sum = build_grad_a_sum(h, w);
        let alpha: AlphaField = Array3::zeros((h, w, c));

        let f = flow(&grad_u, &grad_a_sum, &alpha);

        for ((y, x, fi, ci), &v) in f.indexed_iter() {
            assert_relative_eq!(v, grad_u[[y, x, fi, ci]], epsilon = 1e-6);
        }
    }

    /// `α ≡ 1` → `F = -∇A_Σ`, *shared* across the C axis.
    #[test]
    fn flow_unit_alpha_is_pure_diffusion() {
        let (h, w, c) = (4, 4, 3);
        let grad_u = build_grad_u(h, w, c);
        let grad_a_sum = build_grad_a_sum(h, w);
        let alpha: AlphaField = Array3::from_elem((h, w, c), 1.0);

        let f = flow(&grad_u, &grad_a_sum, &alpha);

        for ((y, x, fi, ci), &v) in f.indexed_iter() {
            // -∇A_Σ has no channel index — same value for every ci.
            assert_relative_eq!(v, -grad_a_sum[[y, x, fi]], epsilon = 1e-6);
            let _ = ci;
        }
    }

    /// Hand-computed Eq. 5: every cell of a `3×3×(2 channels)` grid with
    /// known constant α, `∇U`, and `∇A_Σ`. Catches sign errors, axis swaps,
    /// per-channel/shared confusion.
    ///
    /// Setup:
    ///   α  : ch0 = 0.25, ch1 = 0.5  (constant across the grid)
    ///   ∇U : ch0 (dy, dx) = (1, 2), ch1 (dy, dx) = (3, 4)
    ///   ∇A_Σ : (dy, dx) = (0.5, 1.0)
    ///
    /// Expected (per the formula `F = (1-α)∇U − α∇A_Σ`):
    ///   ch0 dy = 0.75 · 1 − 0.25 · 0.5 = 0.625
    ///   ch0 dx = 0.75 · 2 − 0.25 · 1.0 = 1.25
    ///   ch1 dy = 0.50 · 3 − 0.50 · 0.5 = 1.25
    ///   ch1 dx = 0.50 · 4 − 0.50 · 1.0 = 1.5
    #[test]
    fn flow_eq_5_handcoded() {
        let (h, w, c) = (3, 3, 2);

        let alpha: AlphaField =
            Array3::from_shape_fn((h, w, c), |(_, _, ci)| if ci == 0 { 0.25 } else { 0.5 });
        let grad_u = Array4::from_shape_fn((h, w, 2, c), |(_, _, fi, ci)| match (fi, ci) {
            (0, 0) => 1.0_f32, // ch0 dy
            (1, 0) => 2.0,     // ch0 dx
            (0, 1) => 3.0,     // ch1 dy
            (1, 1) => 4.0,     // ch1 dx
            _ => unreachable!(),
        });
        let grad_a_sum =
            Array3::from_shape_fn((h, w, 2), |(_, _, fi)| if fi == 0 { 0.5_f32 } else { 1.0 });

        let f = flow(&grad_u, &grad_a_sum, &alpha);

        for y in 0..h {
            for x in 0..w {
                assert_relative_eq!(f[[y, x, FLOW_DY, 0]], 0.625, epsilon = 1e-6);
                assert_relative_eq!(f[[y, x, FLOW_DX, 0]], 1.25, epsilon = 1e-6);
                assert_relative_eq!(f[[y, x, FLOW_DY, 1]], 1.25, epsilon = 1e-6);
                assert_relative_eq!(f[[y, x, FLOW_DX, 1]], 1.5, epsilon = 1e-6);
            }
        }
    }

    /// `FlowFieldExt::dy()` / `dx()` views align with the explicit
    /// `[y, x, FLOW_DY, ci]` indexing used inside `flow`. Smoke test the
    /// integration of M1.9's struct trait with `flow`'s output layout.
    #[test]
    fn flow_output_dy_dx_accessors_align() {
        // Make ∇U_dy ≠ ∇U_dx so a transpose bug would surface.
        let (h, w, c) = (3, 3, 2);
        let alpha: AlphaField = Array3::zeros((h, w, c)); // α = 0 → F = ∇U
        let grad_u = Array4::from_shape_fn((h, w, 2, c), |(_, _, fi, ci)| (fi * 10 + ci) as f32);
        let grad_a_sum: Array3<f32> = Array3::zeros((h, w, 2));
        let f = flow(&grad_u, &grad_a_sum, &alpha);

        let dy_view = f.dy();
        let dx_view = f.dx();
        for ((y, x, ci), &dy) in dy_view.indexed_iter() {
            assert_relative_eq!(dy, ci as f32, epsilon = 1e-6); // fi = 0 path
            let _ = (y, x);
        }
        for ((y, x, ci), &dx) in dx_view.indexed_iter() {
            assert_relative_eq!(dx, (10 + ci) as f32, epsilon = 1e-6); // fi = 1 path
            let _ = (y, x);
        }
    }

    /// `ActivationField` and `AlphaField` are both `Array3` — sanity check
    /// that the `(H, W, C)` shape they share is the same one `flow`
    /// consumes via `alpha`. Compile-time-ish check exercised at runtime.
    #[test]
    fn flow_alpha_shape_matches_activation_shape() {
        let a: ActivationField = Array3::zeros((4, 5, 2));
        let alpha: AlphaField = a.clone(); // re-use the same shape
        assert_eq!(alpha.shape(), &[4, 5, 2]);
    }
}
