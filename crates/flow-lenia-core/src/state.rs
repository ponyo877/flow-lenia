//! Type aliases and axis conventions for Flow-Lenia 3D / 4D fields.
//!
//! All fields use the **(H, W, …)** axis order with the *channel* (C) axis
//! at the end — matching JAX `flowlenia.py:80` `Float[Array, "X Y C"]` and
//! `flowlenia.py:94` `nabla_U: (x, y, 2, c)`. Sticking to one convention
//! everywhere keeps `KernelEntry::{c0, c1}` indexing, `sum_axis(AXIS_C)`,
//! and JAX cross-references all unambiguous.

use ndarray::{Array2, Array3, Array4, ArrayView3, Axis};

// ─────────────────────────────────────────────────────────────────────
// 3D fields: ActivationField (A), AlphaField (α), U (affinity), …
// Shape (H, W, C). Channel axis = AXIS_C.
// ─────────────────────────────────────────────────────────────────────

/// 3D activation field. Shape `(H, W, C)`. Paper symbol: `A`.
pub type ActivationField = Array3<f32>;

/// 3D α (diffusion weight) field. Shape `(H, W, C)`.
///
/// In `paper_strict` mode every cell `(y, x)` carries the same value
/// across all C channels (broadcast from a single shared scalar).
/// In JAX-compat (default) mode each channel has its own α value.
pub type AlphaField = Array3<f32>;

/// Height (row) axis of any `(H, W, …)` field.
pub const AXIS_H: Axis = Axis(0);
/// Width (column) axis of any `(H, W, …)` field.
pub const AXIS_W: Axis = Axis(1);
/// Channel axis of a 3D `(H, W, C)` field. **Do not confuse with
/// [`AXIS_FLOW`]**: they have the same numeric value `Axis(2)` but apply
/// to fields of different rank.
pub const AXIS_C: Axis = Axis(2);

// ─────────────────────────────────────────────────────────────────────
// 4D field: FlowField (F), per-channel ∇U, …
// Shape (H, W, 2, C). Flow axis = AXIS_FLOW (Axis(2)).
// ─────────────────────────────────────────────────────────────────────

/// 4D flow field. Shape `(H, W, 2, C)`. Paper symbol: `F`.
///
/// The size-2 third axis is the (dy, dx) flow vector component — index
/// [`FLOW_DY`] for `∂/∂y`, index [`FLOW_DX`] for `∂/∂x`. This matches
/// JAX `flowlenia.py:94` `nabla_U: (x, y, 2, c)`.
///
/// We deliberately encode `F` as an `Array4` instead of a struct with
/// two `Array3` fields because:
/// - paper Eq. 5 `F = (1-α)∇U − α∇A_Σ` is one broadcast operation in
///   `ndarray`, not two;
/// - downstream reintegration tracking (M1.11) reads the (dy, dx) pair
///   together via a single slice;
/// - the `Array4` layout matches the JAX reference, easing porting.
///
/// To keep the `dy / dx` order unambiguous at call sites we expose the
/// [`FlowFieldExt`] accessor trait — callers write `f.dy()` / `f.dx()`
/// rather than `f.index_axis(Axis(2), 0)` etc.
pub type FlowField = Array4<f32>;

/// Flow (vector-component) axis of a 4D `(H, W, 2, C)` field.
pub const AXIS_FLOW: Axis = Axis(2);

/// Index of the `dy` (∂/∂y) component on [`AXIS_FLOW`].
pub const FLOW_DY: usize = 0;
/// Index of the `dx` (∂/∂x) component on [`AXIS_FLOW`].
pub const FLOW_DX: usize = 1;

/// Convenience accessors for [`FlowField`] (and any shape-compatible
/// `Array4<f32>` representing a `(H, W, 2, C)` vector-component field).
///
/// Prefer `f.dy()` / `f.dx()` over `f.index_axis(AXIS_FLOW, 0)` to make
/// component selection self-documenting and to centralise the
/// `FLOW_DY = 0` / `FLOW_DX = 1` choice.
pub trait FlowFieldExt {
    /// `∂/∂y` component slice. Shape `(H, W, C)`.
    fn dy(&self) -> ArrayView3<'_, f32>;
    /// `∂/∂x` component slice. Shape `(H, W, C)`.
    fn dx(&self) -> ArrayView3<'_, f32>;
}

impl FlowFieldExt for FlowField {
    fn dy(&self) -> ArrayView3<'_, f32> {
        self.index_axis(AXIS_FLOW, FLOW_DY)
    }
    fn dx(&self) -> ArrayView3<'_, f32> {
        self.index_axis(AXIS_FLOW, FLOW_DX)
    }
}

// ─────────────────────────────────────────────────────────────────────
// Shared helpers
// ─────────────────────────────────────────────────────────────────────

/// Sum activations over the channel axis. Paper notation: `A_Σ(x, y) = Σ_c A_c(x, y)`.
///
/// Returns shape `(H, W)`. Reused by `grad_a_sum`, the reintegration
/// stochastic sampling (M1.11), and the `paper_strict` branch of
/// [`crate::alpha::alpha`].
#[must_use]
pub fn sum_channels(a: &ActivationField) -> Array2<f32> {
    a.sum_axis(AXIS_C)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// `sum_channels` correctness on a hand-built `(2, 2, 3)` input.
    #[test]
    fn sum_channels_correctness() {
        let mut a: ActivationField = Array3::zeros((2, 2, 3));
        a[[0, 0, 0]] = 1.0;
        a[[0, 0, 1]] = 2.0;
        a[[0, 0, 2]] = 3.0;
        a[[1, 1, 0]] = 10.0;
        a[[1, 1, 1]] = 20.0;
        // (1, 0) / (0, 1) untouched → 0.

        let s = sum_channels(&a);
        assert_eq!(s.shape(), &[2, 2]);
        assert_relative_eq!(s[[0, 0]], 6.0, epsilon = 1e-6);
        assert_relative_eq!(s[[1, 1]], 30.0, epsilon = 1e-6);
        assert_relative_eq!(s[[0, 1]], 0.0, epsilon = 1e-6);
        assert_relative_eq!(s[[1, 0]], 0.0, epsilon = 1e-6);
    }

    /// `FlowFieldExt::dy()` and `dx()` return slices at the correct
    /// `AXIS_FLOW` index. Guards against an off-by-one swap of
    /// `FLOW_DY` / `FLOW_DX`.
    #[test]
    fn flow_field_ext_accessors_match_index_axis() {
        let f: FlowField = Array4::from_shape_fn((2, 3, 2, 4), |(y, x, fi, ci)| {
            (y + x * 10 + fi * 100 + ci * 1000) as f32
        });
        let dy = f.dy();
        let dx = f.dx();
        assert_eq!(dy.shape(), &[2, 3, 4]);
        assert_eq!(dx.shape(), &[2, 3, 4]);
        for ((y, x, ci), &v) in dy.indexed_iter() {
            assert_eq!(v, f[[y, x, FLOW_DY, ci]]);
        }
        for ((y, x, ci), &v) in dx.indexed_iter() {
            assert_eq!(v, f[[y, x, FLOW_DX, ci]]);
        }
    }
}
