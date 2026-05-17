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
use ndarray::{array, Array2};

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
}
