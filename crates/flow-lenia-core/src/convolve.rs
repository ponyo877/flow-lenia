//! 2D direct convolution for Flow-Lenia (paper Eq. 3 numerator, JAX
//! `flowlenia.py:82-86`).
//!
//! M1 uses direct (spatial-domain) convolution per `DESIGN.md` §1.7. FFT-based
//! convolution is a stretch goal in M6. The kernel `K` produced by
//! [`crate::kernel::compute_kernel`] is radially symmetric, so *correlation*
//! and *mathematical convolution* yield identical results — this
//! implementation uses correlation (`output[y,x] = Σ A[y+dy, x+dx]·K[dy,dx]`)
//! because it maps directly onto the natural WGSL formulation
//! (`DESIGN.md` §4.4 / §4.1.2).
//!
//! ──────────────────────────────────────────────────────────────────────
//! Testing strategy (M1.6 — L1 + L2 hybrid)
//! ──────────────────────────────────────────────────────────────────────
//!
//! **L1 — naive_reference comparison**
//!   Guarantees internal consistency: the optimised `convolve2d` produces
//!   the same output as a textbook O(N²·M²) implementation
//!   (`convolve2d_naive`). Catches: efficiency bugs, branch errors,
//!   boundary-condition bugs.
//!   Does **not** catch: misinterpretation of convolution vs correlation
//!   semantics, paper/JAX semantic mismatches.
//!
//! **L2 — jax_fixture comparison** (see `tests/jax_fixture_smoke.rs`)
//!   Guarantees external compatibility: the Rust output matches JAX
//!   `jax.scipy.signal.convolve2d(A, K, mode='same', boundary='wrap'|'fill')`
//!   within `1e-3` relative error.
//!   Catches: convolution vs correlation flip, kernel orientation bugs,
//!   border-condition interpretation mismatches.
//!   Tolerance `1e-3` allows for addition-order differences between
//!   Rust (`ndarray` fold) and JAX (`numpy` reduce / FFT path).
//!
//! **L3 — bit-identical**
//!   Not implemented in M1. Deferred to M6 if the stretch-goal FFT path
//!   needs cross-validation with JAX.
//! ──────────────────────────────────────────────────────────────────────

use crate::config::BorderMode;
use ndarray::Array2;

/// Sample `a[y, x]` with the given boundary policy.
///
/// `(y, x)` may be negative or out-of-bounds; the policy decides what to
/// return.
#[inline]
fn sample_at(a: &Array2<f32>, y: i32, x: i32, border: BorderMode) -> f32 {
    let (h, w) = a.dim();
    match border {
        BorderMode::Torus => {
            // `rem_euclid` is the mathematical modulo (always non-negative
            // for positive divisors), so negative indices wrap correctly.
            let yy = y.rem_euclid(h as i32) as usize;
            let xx = x.rem_euclid(w as i32) as usize;
            a[[yy, xx]]
        }
        BorderMode::Wall => {
            if y < 0 || y >= h as i32 || x < 0 || x >= w as i32 {
                0.0
            } else {
                a[[y as usize, x as usize]]
            }
        }
    }
}

/// **Naive** O(H·W·kh·kw) 2D correlation. Reference implementation.
///
/// `output[y, x] = Σ_{dy, dx} activation[y + dy, x + dx] · kernel[er + dy, er + dx]`
///
/// where `(dy, dx)` ranges over `(-er_y..=er_y, -er_x..=er_x)` with
/// `er_y = kh / 2`, `er_x = kw / 2`. Kernel side lengths must be odd.
///
/// Border behaviour follows [`BorderMode`].
///
/// # Panics
///
/// Panics if `kernel`'s side lengths are even (a centred kernel must have
/// odd dimensions so that the offset `er = (k-1)/2` is well-defined).
#[must_use]
pub(crate) fn convolve2d_naive(
    activation: &Array2<f32>,
    kernel: &Array2<f32>,
    border: BorderMode,
) -> Array2<f32> {
    let (h, w) = activation.dim();
    let (kh, kw) = kernel.dim();
    assert!(kh % 2 == 1, "kernel height must be odd (got {kh})");
    assert!(kw % 2 == 1, "kernel width must be odd (got {kw})");
    let er_y = (kh / 2) as i32;
    let er_x = (kw / 2) as i32;

    let mut out = Array2::<f32>::zeros((h, w));

    for y in 0..h as i32 {
        for x in 0..w as i32 {
            let mut sum = 0.0_f32;
            // Ordering: dy outer, dx inner (matches DESIGN.md §4.5 for
            // numerical-order alignment with the reintegration loop).
            for dy in -er_y..=er_y {
                for dx in -er_x..=er_x {
                    let a_val = sample_at(activation, y + dy, x + dx, border);
                    let k_val = kernel[[(dy + er_y) as usize, (dx + er_x) as usize]];
                    sum += a_val * k_val;
                }
            }
            out[[y as usize, x as usize]] = sum;
        }
    }
    out
}

/// 2D correlation with the same semantics as [`convolve2d_naive`].
///
/// Currently a thin wrapper around the naive implementation. The separate
/// name preserves the L1 testing contract — future optimisations (e.g.,
/// SIMD inner loop, circular-radius early exit, FFT in M6) must continue
/// to satisfy `convolve2d ≡ convolve2d_naive` (within numeric tolerance for
/// FFT path).
#[must_use]
pub fn convolve2d(
    activation: &Array2<f32>,
    kernel: &Array2<f32>,
    border: BorderMode,
) -> Array2<f32> {
    convolve2d_naive(activation, kernel, border)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::compute_kernel;
    use crate::params::KernelEntry;
    use approx::assert_relative_eq;

    fn delta(h: usize, w: usize, y: usize, x: usize) -> Array2<f32> {
        let mut a = Array2::<f32>::zeros((h, w));
        a[[y, x]] = 1.0;
        a
    }

    fn ones(h: usize, w: usize) -> Array2<f32> {
        Array2::<f32>::from_elem((h, w), 1.0)
    }

    /// A simple identifiable 3×3 kernel for boundary tests:
    /// ```text
    /// 1 2 3
    /// 4 5 6
    /// 7 8 9
    /// ```
    /// (Asymmetric so border bugs show up as wrong sums rather than
    /// silently cancelling.)
    fn k3x3() -> Array2<f32> {
        Array2::<f32>::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
            .unwrap()
    }

    // ─── L1: internal consistency ────────────────────────────────────

    /// `convolve2d(δ, K) = K_flipped` for correlation; here K is symmetric
    /// in our radially-symmetric use-case, but we still test with an
    /// asymmetric 3×3 kernel to make sure the kernel indexing isn't
    /// transposed.
    ///
    /// With an *interior* delta (i.e. far from the border so torus/wall
    /// don't matter) at `(2, 2)` on a 7×7 grid and a 3×3 kernel centred
    /// at `(2, 2)`:
    ///
    ///   output[y, x] = Σ_{dy, dx} A[y+dy, x+dx] · K[er+dy, er+dx]
    ///                = K[er+(2-y), er+(2-x)]   (only A[2,2] = 1 contributes)
    ///                = K[2 - (y - 2) + 0, ...]
    ///
    /// So output is K *centred at (2, 2)*, indexed as K[er - dy, er - dx]
    /// where (dy, dx) is the offset of the output cell from the delta.
    /// For the asymmetric K above, this confirms correlation semantics
    /// (not convolution — which would be K[er + dy, er + dx]).
    #[test]
    fn convolve_with_delta_yields_kernel() {
        let a = delta(7, 7, 2, 2);
        let k = k3x3();
        let out = convolve2d(&a, &k, BorderMode::Wall);

        // The cell at (y, x) = (2 + dy, 2 + dx) for (dy, dx) ∈ {-1, 0, +1}²
        // receives K[1 - dy, 1 - dx] (correlation form).
        // E.g. output[1, 1] = K[1-(-1), 1-(-1)] = K[2, 2] = 9.
        //      output[3, 3] = K[1-1, 1-1]      = K[0, 0] = 1.
        approx::assert_relative_eq!(out[[1, 1]], 9.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[1, 2]], 8.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[1, 3]], 7.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[2, 1]], 6.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[2, 2]], 5.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[2, 3]], 4.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[3, 1]], 3.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[3, 2]], 2.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[3, 3]], 1.0, epsilon = 1e-6);
        // Distant cells are 0.
        approx::assert_relative_eq!(out[[0, 0]], 0.0, epsilon = 1e-6);
        approx::assert_relative_eq!(out[[6, 6]], 0.0, epsilon = 1e-6);
    }

    /// For a constant input `A ≡ c`, the convolution output is `c · Σ K`
    /// at every interior cell (boundary-independent: ones · K = Σ K for
    /// torus, and ones-with-zero-pad ≠ Σ K only near the boundary).
    /// Specifically, using a *normalised* kernel (Σ K = 1) with constant
    /// input under **torus** boundary, output ≡ c everywhere.
    #[test]
    fn convolve_normalised_kernel_constant_input_is_identity_torus() {
        let a = ones(10, 10) * 0.7_f32;
        let entry = KernelEntry {
            c0: 0,
            c1: 0,
            r: 0.4,
            a: [0.25, 0.5, 0.75],
            b: [1.0, 0.7, 0.4],
            w: [0.05, 0.05, 0.05],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        };
        let k = compute_kernel(5.0, &entry);
        let out = convolve2d(&a, &k, BorderMode::Torus);

        // Every cell ≈ 0.7 · 1.0 = 0.7. f32 sum has the same floor we
        // documented in `kernel_f32_sum_residual_stays_below_floor`
        // (≈ few 10^-6); 0.7 amplifies that to ≈ few 10^-6 absolute.
        // We use relative_eq with a generous epsilon to absorb it.
        for v in out.iter() {
            assert_relative_eq!(*v, 0.7, epsilon = 1e-5);
        }
    }

    /// Torus boundary: an asymmetric kernel applied at a *corner* must wrap
    /// to the opposite corner. Concrete check: 4×4 grid, A is zero except
    /// at the bottom-right cell (3, 3); convolution with k3x3 at the
    /// top-left output cell (0, 0) must see A[3, 3] wrapping in via the
    /// kernel's *bottom-right* entry (after the correlation flip,
    /// K[er+dy, er+dx] with dy=-1, dx=-1 → K[0, 0] = 1).
    ///
    /// Manual computation for output[0, 0] with torus wrap:
    ///   neighbours (dy, dx) ∈ {-1, 0, +1}² of (0, 0) wrap as:
    ///     (-1, -1) → (3, 3) = 1.0  weighted by K[0, 0] = 1
    ///     (-1,  0) → (3, 0) = 0.0
    ///     (-1, +1) → (3, 1) = 0.0
    ///     ( 0, -1) → (0, 3) = 0.0
    ///     ( 0,  0) → (0, 0) = 0.0
    ///     ( 0, +1) → (0, 1) = 0.0
    ///     (+1, -1) → (1, 3) = 0.0
    ///     (+1,  0) → (1, 0) = 0.0
    ///     (+1, +1) → (1, 1) = 0.0
    ///   → output[0, 0] = 1.0 · 1.0 = 1.0
    #[test]
    fn convolve_torus_boundary_wraps_correctly() {
        let mut a = Array2::<f32>::zeros((4, 4));
        a[[3, 3]] = 1.0;
        let k = k3x3();
        let out = convolve2d(&a, &k, BorderMode::Torus);

        // Top-left corner sees (3, 3) wrap in via K[0, 0] = 1.
        assert_relative_eq!(out[[0, 0]], 1.0, epsilon = 1e-6);
    }

    /// Wall boundary: same grid as above but with zero-padding, the
    /// out-of-bounds neighbour does NOT wrap, so output[0, 0] = 0.
    ///
    /// Conversely, output[3, 3] sees the value 1.0 at its own location
    /// with K[er, er] = K[1, 1] = 5 → output[3, 3] = 5.0.
    #[test]
    fn convolve_wall_boundary_zero_pads_correctly() {
        let mut a = Array2::<f32>::zeros((4, 4));
        a[[3, 3]] = 1.0;
        let k = k3x3();
        let out = convolve2d(&a, &k, BorderMode::Wall);

        assert_relative_eq!(out[[0, 0]], 0.0, epsilon = 1e-6);
        // (3, 3) only has 4 valid neighbours; A[3, 3] · K[1, 1] = 1.0 · 5.0.
        assert_relative_eq!(out[[3, 3]], 5.0, epsilon = 1e-6);
    }

    /// `convolve2d` (public API) and `convolve2d_naive` must produce the
    /// same output bit-for-bit at present (the public API is a thin
    /// wrapper). Future optimisations must preserve this — or, if the
    /// optimisation introduces approximation, the wrapper switches to
    /// relative_eq with a documented tolerance.
    #[test]
    fn convolve_public_api_matches_naive() {
        let entry = KernelEntry {
            c0: 0,
            c1: 0,
            r: 0.5,
            a: [0.25, 0.5, 0.75],
            b: [1.0, 0.7, 0.4],
            w: [0.05, 0.05, 0.05],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        };
        let k = compute_kernel(5.0, &entry);

        // Use a deterministic non-trivial activation so the test catches
        // any reordering-induced divergence.
        let mut a = Array2::<f32>::zeros((16, 16));
        for y in 0..16 {
            for x in 0..16 {
                a[[y, x]] = ((y * 13 + x * 7) % 11) as f32 / 11.0;
            }
        }

        for border in [BorderMode::Torus, BorderMode::Wall] {
            let pub_out = convolve2d(&a, &k, border);
            let naive_out = convolve2d_naive(&a, &k, border);
            // Currently identical (wrapper is thin) — assert bit-equality.
            for (p, n) in pub_out.iter().zip(naive_out.iter()) {
                assert_eq!(p.to_bits(), n.to_bits(), "border = {border:?}");
            }
        }
    }

    /// Sanity for `sample_at` itself: torus wrap is correct on negative
    /// indices and on indices ≥ size.
    #[test]
    fn sample_at_torus_wrap() {
        let mut a = Array2::<f32>::zeros((4, 4));
        a[[0, 0]] = 1.0;
        a[[3, 3]] = 9.0;

        assert_relative_eq!(sample_at(&a, -1, -1, BorderMode::Torus), 9.0, epsilon = 0.0);
        assert_relative_eq!(sample_at(&a, 4, 4, BorderMode::Torus), 1.0, epsilon = 0.0);
        assert_relative_eq!(sample_at(&a, 7, 7, BorderMode::Torus), 9.0, epsilon = 0.0);
    }

    /// Sanity for `sample_at` itself: wall yields 0 outside the grid.
    #[test]
    fn sample_at_wall_zero_pad() {
        let mut a = Array2::<f32>::zeros((4, 4));
        a[[0, 0]] = 1.0;

        assert_relative_eq!(sample_at(&a, -1, 0, BorderMode::Wall), 0.0, epsilon = 0.0);
        assert_relative_eq!(sample_at(&a, 0, -1, BorderMode::Wall), 0.0, epsilon = 0.0);
        assert_relative_eq!(sample_at(&a, 4, 0, BorderMode::Wall), 0.0, epsilon = 0.0);
        assert_relative_eq!(sample_at(&a, 0, 0, BorderMode::Wall), 1.0, epsilon = 0.0);
    }
}
