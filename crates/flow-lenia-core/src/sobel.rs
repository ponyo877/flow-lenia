//! Sobel filter for Flow-Lenia (paper Eq. 5 gradient term, JAX `utils.py:16-37`).
//!
//! **Q3b confirmed** (`references/JAX_NOTES.md` §3): JAX uses the *unnormalised*
//! Sobel coefficients
//!
//! ```text
//! kx = [[ 1,  0, -1],         ky = kx.T = [[ 1,  2,  1],
//!       [ 2,  0, -2],                      [ 0,  0,  0],
//!       [ 1,  0, -1]]                      [-1, -2, -1]]
//! ```
//!
//! and applies them via `jsp.signal.convolve2d(A, k, mode='same', boundary=…)`,
//! which performs *mathematical* convolution (kernel flipped 180° then
//! correlated). Our [`crate::convolve::convolve2d`] is correlation, so we
//! pre-flip the kernels here — the 180° rotation lands us at
//! `[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]` for x and the transpose-then-flip
//! for y, which is the "textbook" Sobel correlation form a reader would
//! expect to see in a graphics paper.
//!
//! Boundary semantics follow the [`BorderMode`] argument (DESIGN.md §4.4
//! Rev.4 extension): `Torus` → circular wrap, `Wall` → zero padding.

use crate::config::BorderMode;
use crate::convolve::convolve2d;
use crate::state::{sum_channels, ActivationField, FLOW_DX, FLOW_DY};
use ndarray::{array, s, Array2, Array3, Array4};

/// 180°-rotated `kx` — our `convolve2d` is correlation, JAX `convolve2d`
/// is convolution, so we flip the JAX kernel once here. Kernel coefficients
/// after the flip: `[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]`.
fn kx_correlation() -> Array2<f32> {
    array![[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
}

/// 180°-rotated `ky` (= `kx.T`). After the flip:
/// `[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]`.
fn ky_correlation() -> Array2<f32> {
    array![[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
}

/// Sobel gradients of a 2D field, returned in JAX's `(dy, dx)` order
/// (`utils.py:34-37` returns `(sobel_y, sobel_x)`).
///
/// We use a named struct rather than a tuple so callers cannot accidentally
/// swap the axes via destructuring. Field naming matches NumPy's
/// `[row, col] = [y, x]` indexing convention.
#[derive(Debug, Clone)]
pub struct SobelGradients {
    /// `∂A/∂y` — first derivative along the row axis.
    /// Computed with the (flipped) `ky` kernel.
    pub dy: Array2<f32>,
    /// `∂A/∂x` — first derivative along the column axis.
    /// Computed with the (flipped) `kx` kernel.
    pub dx: Array2<f32>,
}

/// Compute both Sobel gradients in one call. Equivalent to calling
/// [`sobel_y`] and [`sobel_x`] separately (see
/// `sobel_consistent_with_separate_functions` test for the contract).
#[must_use]
pub fn sobel(a: &Array2<f32>, border: BorderMode) -> SobelGradients {
    SobelGradients {
        dy: sobel_y(a, border),
        dx: sobel_x(a, border),
    }
}

/// `∂A/∂x` via the (180°-flipped) JAX `kx` correlation kernel.
#[must_use]
pub fn sobel_x(a: &Array2<f32>, border: BorderMode) -> Array2<f32> {
    convolve2d(a, &kx_correlation(), border)
}

/// `∂A/∂y` via the (180°-flipped) JAX `ky = kx.T` correlation kernel.
#[must_use]
pub fn sobel_y(a: &Array2<f32>, border: BorderMode) -> Array2<f32> {
    convolve2d(a, &ky_correlation(), border)
}

/// Compute per-channel Sobel gradients of a 3D field.
///
/// Input shape `(H, W, C)`, output shape `(H, W, 2, C)` — the size-2 third
/// axis follows the `FlowField` convention with [`FLOW_DY`] at index 0
/// and [`FLOW_DX`] at index 1 (see [`crate::state`]).
///
/// Each channel is processed independently by calling [`sobel`] on the
/// corresponding `Array2` slice. Output is bit-for-bit identical to
/// looping `sobel(a.slice(s![.., .., ci]).to_owned(), border)` per
/// channel — this is the contract pinned by the
/// `sobel_per_channel_matches_independent_calls` test.
#[must_use]
pub fn sobel_per_channel(field: &ActivationField, border: BorderMode) -> Array4<f32> {
    let (h, w, c) = field.dim();
    let mut out = Array4::<f32>::zeros((h, w, 2, c));
    for ci in 0..c {
        // `to_owned()` is required because `sobel` takes `&Array2<f32>`,
        // not `&ArrayView2<f32>`. Per-channel cost is one allocation of
        // (H, W) f32 — negligible against the convolution itself.
        let channel = field.slice(s![.., .., ci]).to_owned();
        let g = sobel(&channel, border);
        out.slice_mut(s![.., .., FLOW_DY, ci]).assign(&g.dy);
        out.slice_mut(s![.., .., FLOW_DX, ci]).assign(&g.dx);
    }
    out
}

/// Compute the gradient of the total mass `A_Σ = Σ_c A_c`.
///
/// Returns shape `(H, W, 2)` with the same flow-axis indexing as
/// [`sobel_per_channel`] / [`crate::state::FlowField`] ([`FLOW_DY`] at 0,
/// [`FLOW_DX`] at 1). This is the diffusion-term direction in paper
/// Eq. 5 (`-α · ∇A_Σ`).
///
/// Equivalent to `sum_channels(a)` followed by [`sobel`] — pinned by
/// the `grad_a_sum_matches_explicit_computation` test. We expose it as
/// a single entry point so callers (M1.13 step-update integration) do
/// not need to remember the (sum → sobel) chain or its axis order.
#[must_use]
pub fn grad_a_sum(a: &ActivationField, border: BorderMode) -> Array3<f32> {
    let a_sum = sum_channels(a);
    let g = sobel(&a_sum, border);
    let (h, w) = g.dy.dim();
    let mut out = Array3::<f32>::zeros((h, w, 2));
    out.slice_mut(s![.., .., FLOW_DY]).assign(&g.dy);
    out.slice_mut(s![.., .., FLOW_DX]).assign(&g.dx);
    out
}

#[cfg(test)]
mod tests {
    //! Expected values for the 8×8 grid tests were measured against JAX
    //! `jax.scipy.signal.convolve2d` on **2026-05-17** using the venv
    //! described in README "JAX fixture re-generation". Specifically:
    //!
    //!  - `Wall` boundary: `boundary='fill', fillvalue=0`
    //!  - `Torus` boundary: manual `jnp.pad(..., mode='wrap')` then
    //!    `mode='valid'` (JAX 0.10.0 does not implement
    //!    `boundary='wrap'` for `convolve2d`).
    //!
    //! Measurements are reproduced inline in the test bodies — no fixture
    //! file is needed because the kernels are fixed 3×3 and the inputs are
    //! trivial ramps / constants.
    //!
    //! See also the M1.7 commit message for the measurement script and the
    //! reasoning that confirmed the **+8** interior value (vs the +4 I had
    //! originally — that was a row-summation off-by-2 caught at the
    //! "deep-dive" follow-up step).

    use super::*;
    use approx::assert_relative_eq;

    // ─── Field builders ──────────────────────────────────────────────

    fn x_ramp(h: usize, w: usize) -> Array2<f32> {
        Array2::from_shape_fn((h, w), |(_y, x)| x as f32)
    }

    fn y_ramp(h: usize, w: usize) -> Array2<f32> {
        Array2::from_shape_fn((h, w), |(y, _x)| y as f32)
    }

    /// Assert two 8×8 `Array2<f32>` fields are element-wise equal, surfacing
    /// the (y, x) coordinate on the first mismatch.
    fn assert_grid_eq(got: &Array2<f32>, expected: &Array2<f32>) {
        assert_eq!(got.shape(), expected.shape(), "shape mismatch");
        for ((y, x), &e) in expected.indexed_iter() {
            let g = got[[y, x]];
            assert_relative_eq!(g, e, epsilon = 1e-6);
            // assert_relative_eq! already aborts on mismatch; the loop
            // index is recoverable from the panic stack trace if needed.
            let _ = (y, x);
        }
    }

    // ─── Constant-field tests ────────────────────────────────────────

    /// **Torus**: Sobel of a constant field is exactly 0 everywhere
    /// because the wrap preserves the constant, and `Σ K = 0` for both
    /// kx and ky kernels.
    #[test]
    fn sobel_constant_field_is_zero_under_torus() {
        let a = Array2::<f32>::from_elem((8, 8), 5.0);
        let g = sobel(&a, BorderMode::Torus);
        for v in g.dx.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
        for v in g.dy.iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    /// **Wall**: the orthogonal-axis invariance only holds in the
    /// interior. At the boundary the zero-pad introduces a mismatch
    /// with the constant field — e.g. corner cell `(0, 0)` of a
    /// constant-5 field under kx gives
    /// `0·(−1) + 0·0 + 0·1 + 0·(−2) + 5·0 + 5·2 + 0·(−1) + 5·0 + 5·1
    ///  = 15`, which JAX (`boundary='fill'`) also reports. We assert 0
    /// only on the inner 6×6 region.
    #[test]
    fn sobel_constant_field_is_zero_in_interior_under_wall() {
        let a = Array2::<f32>::from_elem((8, 8), 5.0);
        let g = sobel(&a, BorderMode::Wall);
        for y in 1..7 {
            for x in 1..7 {
                assert_relative_eq!(g.dx[[y, x]], 0.0, epsilon = 1e-6);
                assert_relative_eq!(g.dy[[y, x]], 0.0, epsilon = 1e-6);
            }
        }
    }

    // ─── Wall boundary, full 8×8 expected grids ──────────────────────

    /// `A(x, y) = x`, Sobel-x, `BorderMode::Wall`.
    ///
    /// Expected (full 8×8) from JAX `jsp.signal.convolve2d(A, kx,
    /// mode='same', boundary='fill', fillvalue=0)` on 2026-05-17:
    #[test]
    fn sobel_x_with_x_ramp_matches_jax_wall() {
        let a = x_ramp(8, 8);
        #[rustfmt::skip]
        let expected = array![
            [  3.,   6.,   6.,   6.,   6.,   6.,   6., -18.],
            [  4.,   8.,   8.,   8.,   8.,   8.,   8., -24.],
            [  4.,   8.,   8.,   8.,   8.,   8.,   8., -24.],
            [  4.,   8.,   8.,   8.,   8.,   8.,   8., -24.],
            [  4.,   8.,   8.,   8.,   8.,   8.,   8., -24.],
            [  4.,   8.,   8.,   8.,   8.,   8.,   8., -24.],
            [  4.,   8.,   8.,   8.,   8.,   8.,   8., -24.],
            [  3.,   6.,   6.,   6.,   6.,   6.,   6., -18.],
        ];
        assert_grid_eq(&sobel_x(&a, BorderMode::Wall), &expected);
    }

    /// `A(x, y) = y`, Sobel-y, `BorderMode::Wall`. Expected is the
    /// transpose of the x case — verified by JAX on 2026-05-17.
    #[test]
    fn sobel_y_with_y_ramp_matches_jax_wall() {
        let a = y_ramp(8, 8);
        #[rustfmt::skip]
        let expected = array![
            [  3.,   4.,   4.,   4.,   4.,   4.,   4.,   3.],
            [  6.,   8.,   8.,   8.,   8.,   8.,   8.,   6.],
            [  6.,   8.,   8.,   8.,   8.,   8.,   8.,   6.],
            [  6.,   8.,   8.,   8.,   8.,   8.,   8.,   6.],
            [  6.,   8.,   8.,   8.,   8.,   8.,   8.,   6.],
            [  6.,   8.,   8.,   8.,   8.,   8.,   8.,   6.],
            [  6.,   8.,   8.,   8.,   8.,   8.,   8.,   6.],
            [-18., -24., -24., -24., -24., -24., -24., -18.],
        ];
        assert_grid_eq(&sobel_y(&a, BorderMode::Wall), &expected);
    }

    // ─── Torus boundary, full 8×8 expected grids ─────────────────────

    /// `A(x, y) = x`, Sobel-x, `BorderMode::Torus`.
    ///
    /// Every row is identical (`A(x, y) = x` is `y`-invariant). Left and
    /// right columns wrap into each other, giving the symmetric ±24 at
    /// the edges and +8 in the interior — verified by JAX on 2026-05-17
    /// (manual circular pad + valid convolve workaround).
    #[test]
    fn sobel_x_with_x_ramp_matches_jax_torus() {
        let a = x_ramp(8, 8);
        let row: [f32; 8] = [-24., 8., 8., 8., 8., 8., 8., -24.];
        let expected = Array2::from_shape_fn((8, 8), |(_y, x)| row[x]);
        assert_grid_eq(&sobel_x(&a, BorderMode::Torus), &expected);
    }

    /// `A(x, y) = y`, Sobel-y, `BorderMode::Torus`. Transpose of the x case.
    #[test]
    fn sobel_y_with_y_ramp_matches_jax_torus() {
        let a = y_ramp(8, 8);
        let col: [f32; 8] = [-24., 8., 8., 8., 8., 8., 8., -24.];
        let expected = Array2::from_shape_fn((8, 8), |(y, _x)| col[y]);
        assert_grid_eq(&sobel_y(&a, BorderMode::Torus), &expected);
    }

    // ─── Orthogonal-axis (zero-gradient) cross check ─────────────────

    /// **Torus**: `∂x/∂y = 0` and `∂y/∂x = 0` exactly, on every cell.
    /// The ramps are translation-invariant under torus wrap so the
    /// orthogonal Sobel axis cancels everywhere.
    #[test]
    fn sobel_orthogonal_axis_returns_zero_under_torus() {
        let a_x = x_ramp(8, 8);
        for v in sobel_y(&a_x, BorderMode::Torus).iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
        let a_y = y_ramp(8, 8);
        for v in sobel_x(&a_y, BorderMode::Torus).iter() {
            assert_relative_eq!(*v, 0.0, epsilon = 1e-6);
        }
    }

    /// **Wall**: orthogonal-axis invariance only in the interior 6×6,
    /// for the same zero-pad mismatch reason described on
    /// `sobel_constant_field_is_zero_in_interior_under_wall`. JAX
    /// `boundary='fill'` exhibits the same non-zero boundary band.
    #[test]
    fn sobel_orthogonal_axis_returns_zero_in_interior_under_wall() {
        let a_x = x_ramp(8, 8);
        let dy = sobel_y(&a_x, BorderMode::Wall);
        for y in 1..7 {
            for x in 1..7 {
                assert_relative_eq!(dy[[y, x]], 0.0, epsilon = 1e-6);
            }
        }
        let a_y = y_ramp(8, 8);
        let dx = sobel_x(&a_y, BorderMode::Wall);
        for y in 1..7 {
            for x in 1..7 {
                assert_relative_eq!(dx[[y, x]], 0.0, epsilon = 1e-6);
            }
        }
    }

    // ─── Combined `sobel()` API consistency ──────────────────────────

    /// `sobel()` returns exactly the same `dx` as `sobel_x()` and the same
    /// `dy` as `sobel_y()`. Bit-for-bit equality (same code path).
    #[test]
    fn sobel_consistent_with_separate_functions() {
        // Use a non-trivial input so the equality is actually exercised.
        let a = Array2::from_shape_fn((8, 8), |(y, x)| ((y * 13 + x * 7) % 11) as f32 / 11.0);

        for border in [BorderMode::Wall, BorderMode::Torus] {
            let g = sobel(&a, border);
            let dx_alone = sobel_x(&a, border);
            let dy_alone = sobel_y(&a, border);

            // Bit-for-bit identical (both go through `convolve2d` with the
            // same kernel — there is no other code path).
            for ((y, x), &v) in g.dx.indexed_iter() {
                assert_eq!(v.to_bits(), dx_alone[[y, x]].to_bits(), "dx @ ({y}, {x})");
            }
            for ((y, x), &v) in g.dy.indexed_iter() {
                assert_eq!(v.to_bits(), dy_alone[[y, x]].to_bits(), "dy @ ({y}, {x})");
            }
        }
    }

    // ─── sobel_per_channel / grad_a_sum (M1.9 helpers) ───────────────

    /// `sobel_per_channel(a)` per-channel slice ≡ `sobel(a.slice(c))`
    /// bit-for-bit, for every channel and both border modes. This is
    /// the contract the (H, W, 2, C) layout promises: no reordering or
    /// extra accumulation happens inside the helper.
    #[test]
    fn sobel_per_channel_matches_independent_calls() {
        let a: ActivationField = ndarray::Array3::from_shape_fn((8, 8, 3), |(y, x, ci)| {
            ((y * 13 + x * 7 + ci * 5) % 11) as f32 / 11.0
        });

        for border in [BorderMode::Wall, BorderMode::Torus] {
            let pc = sobel_per_channel(&a, border);
            assert_eq!(pc.shape(), &[8, 8, 2, 3]);

            for ci in 0..3 {
                let slice = a.slice(s![.., .., ci]).to_owned();
                let g = sobel(&slice, border);
                for y in 0..8 {
                    for x in 0..8 {
                        assert_eq!(
                            pc[[y, x, FLOW_DY, ci]].to_bits(),
                            g.dy[[y, x]].to_bits(),
                            "dy mismatch @ ({y}, {x}, ci={ci}, border={border:?})"
                        );
                        assert_eq!(
                            pc[[y, x, FLOW_DX, ci]].to_bits(),
                            g.dx[[y, x]].to_bits(),
                            "dx mismatch @ ({y}, {x}, ci={ci}, border={border:?})"
                        );
                    }
                }
            }
        }
    }

    /// `grad_a_sum(a) ≡ sobel(sum_channels(a))` bit-for-bit. Pins the
    /// (sum_channels → sobel) wiring so future refactors of either
    /// helper cannot silently change the gradient direction or order.
    #[test]
    fn grad_a_sum_matches_explicit_computation() {
        let a: ActivationField = ndarray::Array3::from_shape_fn((8, 8, 3), |(y, x, ci)| {
            ((y * 13 + x * 7 + ci * 5) % 11) as f32 / 11.0
        });

        for border in [BorderMode::Wall, BorderMode::Torus] {
            let direct = grad_a_sum(&a, border);
            assert_eq!(direct.shape(), &[8, 8, 2]);

            let summed = sum_channels(&a);
            let g = sobel(&summed, border);
            for y in 0..8 {
                for x in 0..8 {
                    assert_eq!(
                        direct[[y, x, FLOW_DY]].to_bits(),
                        g.dy[[y, x]].to_bits(),
                        "dy mismatch @ ({y}, {x}, border={border:?})"
                    );
                    assert_eq!(
                        direct[[y, x, FLOW_DX]].to_bits(),
                        g.dx[[y, x]].to_bits(),
                        "dx mismatch @ ({y}, {x}, border={border:?})"
                    );
                }
            }
        }
    }
}
