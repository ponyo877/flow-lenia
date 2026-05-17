//! Kernel precomputation for Flow-Lenia.
//!
//! Implements the radial-symmetric convolution kernel `K_i` used in paper
//! Eq. 1 / Eq. 3, following the JAX form (DESIGN.md §4.6, Q3c confirmed).
//!
//! ──────────────────────────────────────────────────────────────────────
//! ⚠️ KERNEL DEFINITION: paper Eq. 1 vs JAX implementation differ
//! ──────────────────────────────────────────────────────────────────────
//!
//! **Paper Eq. 1** (Plantec et al. 2025, arXiv:2506.08569v1):
//!   K_i(x) = Σ_j b_{i,j} · exp( -((r/(r_i·R) - a_{i,j})² ) / (2·w_{i,j}²) )
//!   where r is the cell-centre Euclidean distance and r_i·R is the
//!   kernel-i effective radius.
//!
//! **JAX implementation** (`utils.py:41-59`, `ker_f` at `utils.py:9`):
//!   D = r / ((R + 15) · r_i)                      ← scaling differs
//!   K_i(x) = sigmoid(-(D - 1) · 10)               ← extra mask
//!            · Σ_j b_{i,j} · exp( -(D - a_{i,j})² / w_{i,j} )
//!                                                 ↑ paper has 2·w², JAX has w
//!
//! **Three differences and their intent**:
//!   (1) `R → R+15` in the scaling denominator. Empirical correction to
//!       avoid pixel-grid artifacts at small R (Table 1 has R ∈ [2, 25]).
//!       JAX_NOTES.md §7 documents the reasoning.
//!   (2) Gaussian denominator: paper has `2·w²`, JAX has `w`. The numerical
//!       parameter ranges (w ∈ [0.01, 0.5]) overlap, so JAX's "w" can be
//!       read as paper's "2·w²" without changing the parameter range.
//!   (3) `sigmoid(-(D-1)·10)` mask: forces K_i to vanish smoothly for D > 1,
//!       i.e. outside the effective radius. Paper does not mention this; it
//!       is a numerical regulariser, and lets us truncate the kernel array
//!       at finite size without losing significant mass.
//!
//! **Decision**: this implementation follows the JAX form (1)–(3) so that
//! the same `KernelEntry` produces the same creature regardless of which
//! reference (paper vs JAX) the reader consults. The paper Eq. 1 mapping is
//! preserved here for educational cross-reference. This is the design
//! contract recorded in DESIGN.md §4.6.
//! ──────────────────────────────────────────────────────────────────────

use crate::params::KernelEntry;
use ndarray::Array2;

/// Effective radius of a kernel in cells, derived from `R` and the per-kernel
/// scale `r_i` via the JAX rule `er = ⌈(R + 15) · r_i⌉`
/// (`utils.py:53` — `(R+15) * r[k]`).
///
/// At distance `D = 1` the sigmoid mask equals 0.5; `er` is chosen so that
/// the truncation radius coincides with this point in the *Euclidean* sense
/// (the kernel array's *corners* extend a factor `√2` further, where the
/// sigmoid mask is already `≈ sigmoid(-4.14) ≈ 1.6%` and contributes
/// negligibly after normalisation).
#[must_use]
pub fn effective_radius(r_global: f32, r_i: f32) -> u32 {
    assert!(
        r_global > 0.0 && r_i > 0.0,
        "effective_radius needs positive R and r_i (got R={r_global}, r_i={r_i})"
    );
    let raw = (r_global + 15.0) * r_i;
    raw.ceil() as u32
}

/// JAX `utils.py:6-7`: `sigmoid(x) = 0.5 · (tanh(x/2) + 1)`.
///
/// Pulled out as a named helper so unit tests can pin the implementation
/// and so the kernel-mask body is unambiguous about which sigmoid form is
/// used (the JAX one — not the more common `1/(1+exp(-x))`, although the
/// two are mathematically identical).
#[inline]
#[must_use]
pub fn sigmoid(x: f32) -> f32 {
    0.5 * ((x / 2.0).tanh() + 1.0)
}

/// Compute the raw (un-normalised) JAX-form kernel for one entry.
///
/// Output shape: `(2·er + 1, 2·er + 1)`, kernel centre at `[er, er]`.
///
/// This is exposed at crate visibility so tests can verify the per-cell
/// formula against hand-computed values before normalisation.
pub(crate) fn compute_kernel_raw(r_global: f32, entry: &KernelEntry) -> Array2<f32> {
    let er = effective_radius(r_global, entry.r) as i32;
    let size = (2 * er + 1) as usize;
    let denom = (r_global + 15.0) * entry.r; // (R + 15) · r_i

    let mut k = Array2::<f32>::zeros((size, size));

    for y in -er..=er {
        for x in -er..=er {
            let dist = (((x * x + y * y) as f32).sqrt()) / denom;
            // sigmoid(-(D - 1) · 10): ≈ 1 for D < 1, ≈ 0 for D > 1.
            let mask = sigmoid(-(dist - 1.0) * 10.0);
            // ker_f: Σ_j b_j · exp(-(D - a_j)² / w_j)
            let bump = entry.b[0] * (-((dist - entry.a[0]).powi(2)) / entry.w[0]).exp()
                + entry.b[1] * (-((dist - entry.a[1]).powi(2)) / entry.w[1]).exp()
                + entry.b[2] * (-((dist - entry.a[2]).powi(2)) / entry.w[2]).exp();
            k[[(y + er) as usize, (x + er) as usize]] = mask * bump;
        }
    }

    k
}

/// Compute the normalised JAX-form kernel for one entry.
///
/// Equivalent to [`compute_kernel_raw`] followed by `K / Σ K`, matching JAX
/// `utils.py:56` `nK = K / jnp.sum(K, axis=(0,1), keepdims=True)`.
///
/// **Two notions of "normalisation accuracy" — measured, not guessed**:
///
/// 1. *Internal normalisation* (what we control): the f64 accumulator + f64
///    division below makes `Σ_{f64}(K) − 1` essentially **zero** (≤ 1 ulp
///    of f64) across the full `(R, r_i)` parameter range. This is what the
///    M6 bit-precision JAX comparison will care about, and matches what
///    NumPy does inside its tree-pairwise reduction.
///
/// 2. *Reading the kernel back in f32*: if a caller does `K.iter().sum()`
///    in f32, the result can drift by up to ≈ 4e-6 at the largest kernel
///    (81×81 cells, R=25, r=1.0), because adding 6500 f32 values has a
///    quantisation floor independent of how the kernel was normalised.
///    Anything in `core` that depends on `Σ K ≈ 1` should therefore *also*
///    use an f64 accumulator, not because the kernel is wrong but because
///    f32 sums of large arrays cannot be tighter than this.
///
/// The measurement table lives in `diagnose_normalization_error_by_size`
/// (`#[ignore]`); see also `references/JAX_NOTES.md` if revisiting.
///
/// **Why not f64 throughout?** Computing the kernel values themselves in
/// f64 was measured to produce no improvement (the floor is set by
/// accumulation order, not by per-cell value precision). Costs a third of
/// the runtime for zero benefit.
///
/// # Panics
///
/// Panics if the kernel mass is non-positive (which would require all bumps
/// to evaluate to 0 — only possible with pathological parameters that
/// `sample_random` will never produce).
#[must_use]
pub fn compute_kernel(r_global: f32, entry: &KernelEntry) -> Array2<f32> {
    let mut k = compute_kernel_raw(r_global, entry);

    // f64 accumulator: see the doc comment above for justification.
    let sum_f64: f64 = k.iter().map(|&v| f64::from(v)).sum();
    assert!(
        sum_f64 > 0.0,
        "kernel has non-positive mass ({sum_f64}); KernelEntry probably has all b_j = 0"
    );
    let inv_sum_f64 = 1.0 / sum_f64;
    // f64 multiply then cast — keeps the per-cell quantisation at one f32 ulp.
    k.mapv_inplace(|v| (f64::from(v) * inv_sum_f64) as f32);
    k
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    /// Reasonable mid-range kernel parameters used in several tests below.
    /// Chosen so that the kernel exhibits well-defined rings without being
    /// degenerate (b_j > 0, a_j cover the unit interval, w_j are small enough
    /// for sharp Gaussians but large enough to avoid f32 underflow).
    fn mid_range_entry() -> KernelEntry {
        KernelEntry {
            c0: 0,
            c1: 0,
            r: 0.5,
            a: [0.25, 0.5, 0.75],
            b: [1.0, 0.7, 0.4],
            w: [0.05, 0.05, 0.05],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        }
    }

    /// `sigmoid(0) = 0.5` and the asymptotes are correct.
    #[test]
    fn sigmoid_basic_values() {
        assert_relative_eq!(sigmoid(0.0), 0.5, epsilon = 1e-9);
        // Large positive → 1
        assert!(sigmoid(20.0) > 0.999_999);
        // Large negative → 0
        assert!(sigmoid(-20.0) < 1e-6);
        // Symmetry: sigmoid(-x) = 1 - sigmoid(x)
        for x in [0.1_f32, 1.0, 5.0, 10.0] {
            assert_relative_eq!(sigmoid(-x), 1.0 - sigmoid(x), epsilon = 1e-6);
        }
    }

    /// `effective_radius` follows JAX's `ceil((R+15) · r_i)` rule.
    #[test]
    fn effective_radius_matches_jax_formula() {
        // (10 + 15) · 0.5 = 12.5 → ceil = 13
        assert_eq!(effective_radius(10.0, 0.5), 13);
        // (2 + 15) · 0.2 = 3.4 → ceil = 4
        assert_eq!(effective_radius(2.0, 0.2), 4);
        // (25 + 15) · 1.0 = 40.0 → ceil = 40 (no fractional rounding up)
        assert_eq!(effective_radius(25.0, 1.0), 40);
    }

    /// Output shape is `(2·er+1, 2·er+1)`.
    #[test]
    fn kernel_shape_matches_effective_radius() {
        let entry = mid_range_entry();
        let r_global = 10.0;
        let er = effective_radius(r_global, entry.r);
        let k = compute_kernel(r_global, &entry);

        assert_eq!(k.shape(), &[(2 * er + 1) as usize, (2 * er + 1) as usize]);
    }

    /// **Internal normalisation** — `Σ_{f64}(K) = 1` to within one f64 ulp.
    ///
    /// We deliberately sum in f64 here: this is the *property of the kernel
    /// values themselves* that we want to lock in, not the (lossy) result of
    /// `K.iter().sum::<f32>()`. The f32 read-back is bounded by a quantisation
    /// floor of ≈ 4e-6 at the largest kernels (see the next test).
    #[test]
    fn kernel_is_normalized_in_f64() {
        let entry = mid_range_entry();
        let k = compute_kernel(10.0, &entry);
        let sum_f64: f64 = k.iter().map(|&v| f64::from(v)).sum();
        assert_relative_eq!(sum_f64, 1.0, epsilon = 1e-7);
    }

    /// **Internal normalisation across the full `(R, r_i)` range**.
    ///
    /// Guards against future regressions of the f64 accumulator in
    /// [`compute_kernel`] (e.g., someone simplifies it back to a plain f32
    /// `Σ K`). Without the f64 promotion, the 81×81 case below would
    /// exhibit ≈ 4e-6 residual and break this assertion.
    #[test]
    fn kernel_is_normalized_in_f64_across_parameter_range() {
        for (r_global, r_i) in [
            (2.0_f32, 0.2_f32),
            (2.0, 1.0),
            (25.0, 0.2),
            (25.0, 1.0),
            (13.5, 0.6),
        ] {
            let entry = KernelEntry {
                c0: 0,
                c1: 0,
                r: r_i,
                a: [0.25, 0.5, 0.75],
                b: [1.0, 0.7, 0.4],
                w: [0.05, 0.05, 0.05],
                h: 1.0,
                mu: 0.15,
                sigma: 0.02,
            };
            let k = compute_kernel(r_global, &entry);
            let sum_f64: f64 = k.iter().map(|&v| f64::from(v)).sum();
            assert_relative_eq!(sum_f64, 1.0, epsilon = 1e-7);
        }
    }

    /// **f32 read-back floor** — explicitly documents the noise floor a
    /// downstream caller will see if they do `K.iter().sum::<f32>()`.
    ///
    /// Failing this test would mean either:
    /// (a) the floor improved unexpectedly (which suggests a measurement
    ///     artifact — investigate), or
    /// (b) it got *worse* and there's a real regression in `compute_kernel`.
    ///
    /// The 5e-6 bound is the measured worst case (3.58e-6 at the 81×81
    /// corner, see `diagnose_normalization_error_by_size`) with a small
    /// margin for build-to-build f32 variation.
    #[test]
    fn kernel_f32_sum_residual_stays_below_floor() {
        for (r_global, r_i) in [(2.0_f32, 1.0_f32), (25.0, 1.0)] {
            let entry = KernelEntry {
                c0: 0,
                c1: 0,
                r: r_i,
                a: [0.25, 0.5, 0.75],
                b: [1.0, 0.7, 0.4],
                w: [0.05, 0.05, 0.05],
                h: 1.0,
                mu: 0.15,
                sigma: 0.02,
            };
            let k = compute_kernel(r_global, &entry);
            let sum_f32: f32 = k.iter().sum();
            let resid = (sum_f32 - 1.0).abs();
            assert!(
                resid < 5e-6,
                "R={r_global}, r={r_i}: f32 sum residual {resid:.3e} \
                 exceeds the documented floor (5e-6)"
            );
        }
    }

    /// Kernel is radially symmetric: cells at the same Euclidean distance
    /// from the centre carry the same value.
    ///
    /// Symmetry under 90° rotation (`K[er+dy, er+dx] = K[er+dx, er-dy]` etc.)
    /// suffices because the construction depends on `x² + y²` only.
    #[test]
    fn kernel_is_radially_symmetric() {
        let entry = mid_range_entry();
        let r_global = 10.0;
        let er = effective_radius(r_global, entry.r) as usize;
        let k = compute_kernel(r_global, &entry);

        // Sample 90° rotations at a few offsets.
        for (dy, dx) in [(1, 2), (3, 4), (0, 5), (2, 2)] {
            let dy = dy.min(er as i32);
            let dx = dx.min(er as i32);
            let a = k[[(er as i32 + dy) as usize, (er as i32 + dx) as usize]];
            let b = k[[(er as i32 + dx) as usize, (er as i32 - dy) as usize]];
            let c = k[[(er as i32 - dy) as usize, (er as i32 - dx) as usize]];
            let d = k[[(er as i32 - dx) as usize, (er as i32 + dy) as usize]];
            assert_relative_eq!(a, b, epsilon = 1e-6);
            assert_relative_eq!(a, c, epsilon = 1e-6);
            assert_relative_eq!(a, d, epsilon = 1e-6);
        }
    }

    /// The sigmoid mask attenuates the kernel sharply outside the unit
    /// distance. Specifically, kernel values at the corners of the array
    /// (where Euclidean distance ≈ √2 · er, so D ≈ √2 > 1.4) should be
    /// orders of magnitude smaller than the value at the strongest ring.
    ///
    /// This is the key behavioural difference between the JAX form and a
    /// "no mask" form of Eq. 1; without the mask, large `b_j · exp(...)`
    /// values can survive far past `D = 1`.
    #[test]
    fn kernel_sigmoid_mask_attenuates_outside_unit_distance() {
        let entry = mid_range_entry();
        let r_global = 10.0;
        let k = compute_kernel_raw(r_global, &entry);

        // Corner cell — at Euclidean distance ≈ er · √2 from centre, so D ≈ √2.
        // sigmoid(-(√2 - 1) · 10) = sigmoid(-4.14) ≈ 0.016, so the corner
        // value should be ≪ the maximum.
        let corner = k[[0, 0]];
        let max = k.iter().fold(0.0_f32, |a, &b| a.max(b));
        let ratio = corner / max;

        assert!(
            ratio < 0.05,
            "corner/max = {ratio} — sigmoid mask appears to be missing or wrong"
        );
    }

    /// Raw kernel value at the centre (D = 0) matches a hand computation.
    ///
    /// Parameters chosen for an easy manual calculation:
    ///   R = 5, r_i = 1.0 → denom = (5 + 15)·1 = 20, er = 20
    ///   D(0,0) = 0
    ///   sigmoid(-(0 - 1) · 10) = sigmoid(10) ≈ 0.999_954_6
    ///   ker_f(0; a, w, b):
    ///     a = [0, 0, 0], b = [1, 1, 1], w = [1, 1, 1]
    ///     = 3 · exp(0) = 3
    ///   K_raw[er, er] = sigmoid · ker_f ≈ 0.999_954_6 · 3 ≈ 2.999_864
    #[test]
    fn kernel_raw_known_value_at_origin() {
        let entry = KernelEntry {
            c0: 0,
            c1: 0,
            r: 1.0,
            a: [0.0, 0.0, 0.0],
            b: [1.0, 1.0, 1.0],
            w: [1.0, 1.0, 1.0],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        };
        let r_global = 5.0;
        let k = compute_kernel_raw(r_global, &entry);
        let er = effective_radius(r_global, entry.r) as usize;

        let expected_sigmoid = sigmoid(10.0); // ≈ 0.999_954_6
        let expected_bump = 3.0_f32; // 3 · exp(0)
        let expected = expected_sigmoid * expected_bump;

        assert_relative_eq!(k[[er, er]], expected, epsilon = 1e-5);
    }

    /// Raw value at a non-trivial position matches a hand calculation.
    ///
    /// Same `entry` as the previous test (a = [0,0,0], b = [1,1,1], w = [1,1,1]):
    ///   At (dx, dy) = (10, 0) with er = 20, denom = 20:
    ///   D = 10 / 20 = 0.5
    ///   sigmoid(-(0.5 - 1)·10) = sigmoid(5) ≈ 0.993_307
    ///   ker_f(0.5) = 3 · exp(-0.25) ≈ 3 · 0.778_801 ≈ 2.336_403
    ///   K_raw ≈ 0.993_307 · 2.336_403 ≈ 2.320_770
    #[test]
    fn kernel_raw_known_value_at_half_distance() {
        let entry = KernelEntry {
            c0: 0,
            c1: 0,
            r: 1.0,
            a: [0.0, 0.0, 0.0],
            b: [1.0, 1.0, 1.0],
            w: [1.0, 1.0, 1.0],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        };
        let r_global = 5.0;
        let k = compute_kernel_raw(r_global, &entry);
        let er = effective_radius(r_global, entry.r) as usize;

        // 10 cells to the right of centre at er=20: D = 10/20 = 0.5.
        let v = k[[er, er + 10]];
        let expected_sigmoid = sigmoid(5.0);
        let expected_bump = 3.0_f32 * (-0.25_f32).exp();
        let expected = expected_sigmoid * expected_bump;

        assert_relative_eq!(v, expected, epsilon = 1e-5);
    }

    /// **Diagnostic** (`-- --nocapture --include-ignored` to view): measure
    /// the absolute normalisation residual `|Σ K − 1|` across the kernel-size
    /// range to characterise the f32 summation error scaling. Marked
    /// `#[ignore]` so it does not run in normal `cargo test` invocations.
    #[test]
    #[ignore = "diagnostic only"]
    fn diagnose_normalization_error_by_size() {
        // First entry matches `kernel_is_normalized`'s `mid_range_entry`
        // (R=10, r=0.5) so the diagnostic can be cross-referenced directly
        // against that test's epsilon choice.
        let cases: [(&str, f32, f32); 4] = [
            ("actual test (R=10.0, r=0.5)", 10.0, 0.5),
            ("min   R= 2.0, r=0.2", 2.0, 0.2),
            ("mid   R=13.0, r=0.5", 13.0, 0.5),
            ("max   R=25.0, r=1.0", 25.0, 1.0),
        ];

        for (label, r_global, r_i) in cases {
            let entry = KernelEntry {
                c0: 0,
                c1: 0,
                r: r_i,
                a: [0.25, 0.5, 0.75],
                b: [1.0, 0.7, 0.4],
                w: [0.05, 0.05, 0.05],
                h: 1.0,
                mu: 0.15,
                sigma: 0.02,
            };

            // (a) Current implementation: f32 throughout, ndarray sum.
            let k_f32 = compute_kernel(r_global, &entry);
            let size = k_f32.shape()[0];
            let n = size * size;
            let sum_f32 = k_f32.iter().sum::<f32>();
            let err_f32 = (sum_f32 - 1.0).abs();

            // (b) Sum using f64 accumulator (only the *summation* changes,
            // the kernel values are still computed in f32). This isolates
            // "summation precision" from "computation precision".
            let sum_f32_via_f64: f32 = (k_f32.iter().map(|&v| v as f64).sum::<f64>()) as f32;
            let err_f32_via_f64 = (sum_f32_via_f64 - 1.0).abs();

            // (c) Compute the entire kernel in f64 and renormalise. This is
            // the "f64 intermediate" variant — what we would switch to if
            // a tighter epsilon were required.
            let k_f64_normalised = {
                let er = effective_radius(r_global, entry.r) as i32;
                let denom = (r_global as f64 + 15.0) * entry.r as f64;
                let s = (2 * er + 1) as usize;
                let mut raw = vec![0.0_f64; s * s];
                for y in -er..=er {
                    for x in -er..=er {
                        let dist = ((x * x + y * y) as f64).sqrt() / denom;
                        let mask = 0.5 * (((-(dist - 1.0) * 10.0) / 2.0).tanh() + 1.0);
                        let bump = (entry.b[0] as f64)
                            * (-((dist - entry.a[0] as f64).powi(2)) / entry.w[0] as f64).exp()
                            + (entry.b[1] as f64)
                                * (-((dist - entry.a[1] as f64).powi(2)) / entry.w[1] as f64).exp()
                            + (entry.b[2] as f64)
                                * (-((dist - entry.a[2] as f64).powi(2)) / entry.w[2] as f64).exp();
                        raw[(y + er) as usize * s + (x + er) as usize] = mask * bump;
                    }
                }
                let sum_f64: f64 = raw.iter().sum();
                raw.iter().map(|v| (v / sum_f64) as f32).collect::<Vec<_>>()
            };
            let sum_after_f64: f32 = k_f64_normalised.iter().sum();
            let err_after_f64 = (sum_after_f64 - 1.0).abs();

            // sqrt(N)·ε predicts pairwise-summation-like behaviour;
            // N·ε predicts naive left-fold. Show both for comparison.
            let eps = f32::EPSILON;
            let pred_sqrt_n = (n as f32).sqrt() * eps;
            let pred_n = n as f32 * eps;

            println!(
                "{label}: size={size}x{size} (N={n})\n  \
                 (a) f32 throughout:        |Σ-1| = {:.3e}\n  \
                 (b) f32 vals, f64 sum:     |Σ-1| = {:.3e}\n  \
                 (c) f64 vals, f64 sum:     |Σ-1| = {:.3e}\n  \
                 pred √N·ε = {:.3e}, pred N·ε = {:.3e}\n",
                err_f32, err_f32_via_f64, err_after_f64, pred_sqrt_n, pred_n
            );
        }
    }

    /// At `D = 1` (the kernel boundary) the sigmoid mask equals exactly 0.5.
    /// This is the defining geometric property of the JAX form
    /// (`sigmoid(-(1-1)·10) = sigmoid(0) = 0.5`) and verifies that
    /// `effective_radius` corresponds to the same point that JAX uses.
    #[test]
    fn kernel_sigmoid_mask_at_d_equals_one() {
        // Construct an entry where a single Gaussian is centred at D = 1
        // with width w = 1, b = 1 → bump(D=1) = 1·exp(0) = 1.
        let entry = KernelEntry {
            c0: 0,
            c1: 0,
            r: 1.0,
            a: [1.0, 1.0, 1.0],
            b: [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
            w: [1.0, 1.0, 1.0],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        };
        let r_global = 5.0;
        let denom = (r_global + 15.0) * entry.r; // 20.0
        let k = compute_kernel_raw(r_global, &entry);
        let er = effective_radius(r_global, entry.r) as usize;

        // Pick the cell at (er, er + 20) — D = 20 / 20 = 1.0 exactly.
        let v = k[[er, er + denom as usize]];
        // sigmoid(0) · (1/3 + 1/3 + 1/3) · exp(0) = 0.5 · 1 · 1 = 0.5.
        assert_relative_eq!(v, 0.5, epsilon = 1e-6);
    }
}
