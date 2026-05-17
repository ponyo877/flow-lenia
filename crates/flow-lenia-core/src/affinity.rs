//! Affinity field `U` for Flow-Lenia (paper Eq. 3 and Eq. 7).
//!
//! The affinity is the per-channel scalar field that drives the flow:
//!
//! **Eq. 3 — constant per-kernel weights `h_i`** (default Flow-Lenia):
//!
//! ```text
//! U_j(x) = Σ_{i : c_i^1 = j}  h_i · G_i( (K_i ∗ A_{c_i^0})(x) )
//! ```
//!
//! **Eq. 7 — cell-localised weights `P_i(x)`** (parameter-embedding
//! variant; JAX `flowlenia_params.py:91`):
//!
//! ```text
//! U_j(x) = Σ_{i : c_i^1 = j}  P_i(x) · G_i( (K_i ∗ A_{c_i^0})(x) )
//! ```
//!
//! Eq. 7 strictly generalises Eq. 3: setting `P_i(x) ≡ h_i` (constant
//! across cells) recovers Eq. 3. We expose the two as separate functions
//! rather than an enum so that:
//! - the common Eq. 3 path doesn't pay the cost of broadcasting a
//!   `(H, W, K)` weight tensor that is identically `h_i` everywhere;
//! - the call sites in M1.13 can pick the variant statically.
//!
//! The `affinity_localized_with_uniform_p_equals_constant_h` unit test
//! locks the "uniform-P specialisation" property structurally.
//!
//! ──────────────────────────────────────────────────────────────────────
//! JAX cross-reference (`flowlenia.py:80-90`):
//!
//! ```python
//! fA = jnp.fft.fft2(A, axes=(0,1))                                # (X,Y,C)
//! fAk = fA[:, :, self.cfg.c0]                                     # (X,Y,K)
//! U   = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))       # (X,Y,K)
//! U   = growth(U, self.m, self.s) * self.h                        # Eq. 3 (h is (K,))
//! U   = jnp.dstack([U[:, :, self.cfg.c1[c]].sum(axis=-1)
//!                   for c in range(self.cfg.C)])                  # (X,Y,C)
//! ```
//!
//! JAX uses FFT-based convolution; our M1 reference uses direct
//! correlation via [`crate::convolve::convolve2d`] (DESIGN.md §1.7).
//! Mathematically equivalent for radially-symmetric kernels.
//! ──────────────────────────────────────────────────────────────────────

use crate::config::BorderMode;
use crate::convolve::convolve2d;
use crate::growth::growth;
use crate::kernel::KernelMeta;
use crate::state::{ActivationField, UField, AXIS_C};
use ndarray::{s, Array2, Array3};

/// Compute the affinity field `U` with **constant per-kernel weights**
/// `h_i` (paper Eq. 3).
///
/// Arguments:
/// - `a`: activation field. Shape `(H, W, C)`.
/// - `kernels`: precomputed normalised kernel arrays, one per entry in
///   `meta` and `h`. Each kernel is `Array2<f32>` of odd side length
///   matching `2·meta[i].effective_radius + 1`.
/// - `meta`: per-kernel `(source_channel, target_channel, μ, σ, er)` —
///   see [`KernelMeta`]. Length must equal `kernels.len()`.
/// - `h`: per-kernel weight `h_i` (paper Eq. 3). Length must equal
///   `kernels.len()`.
/// - `border`: boundary condition for the convolution.
///
/// Returns: [`UField`] of shape `(H, W, C)` where `C` is inferred from
/// the maximum `target_channel + 1` across `meta` (or `a.dim().2`,
/// whichever is larger). Channels with no incoming kernel receive `0`.
///
/// # Panics
///
/// Panics if:
/// - `kernels.len() != meta.len()` or `kernels.len() != h.len()`;
/// - any `meta[i].source_channel >= a.dim().2`;
/// - any `meta[i].target_channel >= a.dim().2` (we size `U`'s channel
///   axis to match `A`'s — see DESIGN.md §3 where `A` and `U` share `C`).
///
/// All three are upstream programming errors, not data-dependent
/// failures.
#[must_use]
pub fn affinity_with_constant_weights(
    a: &ActivationField,
    kernels: &[Array2<f32>],
    meta: &[KernelMeta],
    h: &[f32],
    border: BorderMode,
) -> UField {
    assert_eq!(
        kernels.len(),
        meta.len(),
        "kernels.len() ({}) != meta.len() ({})",
        kernels.len(),
        meta.len()
    );
    assert_eq!(
        kernels.len(),
        h.len(),
        "kernels.len() ({}) != h.len() ({})",
        kernels.len(),
        h.len()
    );

    let (height, width, channels) = a.dim();
    let mut u = Array3::<f32>::zeros((height, width, channels));

    for (i, (k_arr, m)) in kernels.iter().zip(meta.iter()).enumerate() {
        let src = m.source_channel as usize;
        let tgt = m.target_channel as usize;
        assert!(
            src < channels,
            "kernel {i}: source_channel {src} >= C={channels}"
        );
        assert!(
            tgt < channels,
            "kernel {i}: target_channel {tgt} >= C={channels}"
        );

        // K_i ∗ A_{c_i^0}
        let a_src = a.index_axis(AXIS_C, src).to_owned();
        let conv = convolve2d(&a_src, k_arr, border);

        // h_i · G_i(K_i ∗ A_{c_i^0}), accumulated into U[:, :, c_i^1].
        let h_i = h[i];
        let mu = m.mu;
        let sigma = m.sigma;
        let mut u_tgt = u.slice_mut(s![.., .., tgt]);
        for ((y, x), &v) in conv.indexed_iter() {
            u_tgt[[y, x]] += h_i * growth(v, mu, sigma);
        }
    }
    u
}

/// Compute the affinity field `U` with **cell-localised per-kernel
/// weights** `P_i(x)` (paper Eq. 7, parameter embedding).
///
/// Arguments are the same as [`affinity_with_constant_weights`] except
/// `h: &[f32]` is replaced by `p_map: &Array3<f32>` of shape
/// `(H, W, K)`, where `K = kernels.len()`. `p_map[[y, x, i]]` is the
/// effective weight of kernel `i` at cell `(y, x)`.
///
/// Setting `p_map[[y, x, i]] = h_i` for every `(y, x)` recovers Eq. 3 —
/// see [`affinity_localized_with_uniform_p_equals_constant_h`] in this
/// module's tests, which locks this property structurally.
///
/// # Panics
///
/// Panics on the same shape mismatches as
/// [`affinity_with_constant_weights`], plus:
/// - `p_map.dim() != (H, W, K)` where `(H, W, _) = a.dim()` and
///   `K = kernels.len()`.
#[must_use]
pub fn affinity_with_localized_weights(
    a: &ActivationField,
    kernels: &[Array2<f32>],
    meta: &[KernelMeta],
    p_map: &Array3<f32>,
    border: BorderMode,
) -> UField {
    assert_eq!(
        kernels.len(),
        meta.len(),
        "kernels.len() ({}) != meta.len() ({})",
        kernels.len(),
        meta.len()
    );

    let (height, width, channels) = a.dim();
    let num_kernels = kernels.len();
    assert_eq!(
        p_map.dim(),
        (height, width, num_kernels),
        "p_map shape {:?} != expected ({}, {}, {})",
        p_map.dim(),
        height,
        width,
        num_kernels
    );

    let mut u = Array3::<f32>::zeros((height, width, channels));

    for (i, (k_arr, m)) in kernels.iter().zip(meta.iter()).enumerate() {
        let src = m.source_channel as usize;
        let tgt = m.target_channel as usize;
        assert!(
            src < channels,
            "kernel {i}: source_channel {src} >= C={channels}"
        );
        assert!(
            tgt < channels,
            "kernel {i}: target_channel {tgt} >= C={channels}"
        );

        let a_src = a.index_axis(AXIS_C, src).to_owned();
        let conv = convolve2d(&a_src, k_arr, border);

        let mu = m.mu;
        let sigma = m.sigma;
        let p_i = p_map.index_axis(AXIS_C, i);
        let mut u_tgt = u.slice_mut(s![.., .., tgt]);
        for ((y, x), &v) in conv.indexed_iter() {
            u_tgt[[y, x]] += p_i[[y, x]] * growth(v, mu, sigma);
        }
    }
    u
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernel::compute_kernel;
    use crate::params::KernelEntry;
    use approx::assert_relative_eq;
    use ndarray::Array3;

    /// Three kernels touching each pair of channels (0, 1, 2) so the
    /// per-target-channel grouping in Eq. 3 has non-trivial structure
    /// across multiple tests.
    fn three_kernel_setup() -> (Vec<Array2<f32>>, Vec<KernelMeta>, Vec<f32>) {
        let r_global = 5.0_f32;
        let entries = [
            KernelEntry {
                c0: 0,
                c1: 0,
                r: 0.4,
                a: [0.25, 0.5, 0.75],
                b: [1.0, 0.7, 0.4],
                w: [0.05, 0.05, 0.05],
                h: 0.5,
                mu: 0.15,
                sigma: 0.02,
            },
            KernelEntry {
                c0: 1,
                c1: 0,
                r: 0.5,
                a: [0.2, 0.5, 0.8],
                b: [0.8, 1.0, 0.5],
                w: [0.05, 0.05, 0.05],
                h: 0.3,
                mu: 0.20,
                sigma: 0.03,
            },
            KernelEntry {
                c0: 2,
                c1: 1,
                r: 0.6,
                a: [0.3, 0.5, 0.7],
                b: [0.6, 1.0, 0.6],
                w: [0.06, 0.06, 0.06],
                h: 0.9,
                mu: 0.25,
                sigma: 0.04,
            },
        ];
        let kernels: Vec<Array2<f32>> =
            entries.iter().map(|e| compute_kernel(r_global, e)).collect();
        let meta: Vec<KernelMeta> = entries
            .iter()
            .map(|e| KernelMeta {
                source_channel: e.c0,
                target_channel: e.c1,
                mu: e.mu,
                sigma: e.sigma,
                effective_radius: crate::kernel::effective_radius(r_global, e.r),
            })
            .collect();
        let h: Vec<f32> = entries.iter().map(|e| e.h).collect();
        (kernels, meta, h)
    }

    /// **Constant input → analytically predictable output**.
    ///
    /// For `A_c ≡ c_val` (constant per channel) under a normalised kernel
    /// `K_i` with torus boundary:
    ///   `(K_i ∗ A_{c_i^0})(x) ≈ c_val`
    /// (residual ≈ 5e-6 — documented in M1.4
    /// `kernel_f32_sum_residual_stays_below_floor`).
    ///
    /// Hence each kernel contributes `h_i · G_i(c_val)` to
    /// `U_{c_i^1}(x)`, independent of `x`. We can hand-compute the
    /// expected per-channel constant U:
    ///   `U_j ≈ Σ_{i : c_i^1 = j}  h_i · G_i(c_val_{c_i^0})`
    ///
    /// Tolerance: 1e-4. Per-kernel residual is `≈ |h_i| · |G'_i| · 5e-6`;
    /// across 3 kernels with `|h_i| ≤ 1` and `|G'_i| · σ_i⁻¹ ≤ ~50` near
    /// the growth peak, 1e-4 gives a safe margin while still catching a
    /// real sign-flip or coefficient drop.
    #[test]
    fn affinity_constant_input_yields_g_of_constant() {
        let (kernels, meta, h) = three_kernel_setup();

        // Constant per-channel activation. Each channel's constant is
        // chosen so the growth at (μ_i, σ_i) is well within the bell.
        let (height, width, channels) = (16, 16, 3);
        let c_vals = [0.15_f32, 0.20, 0.25];
        let mut a: ActivationField = Array3::zeros((height, width, channels));
        for ci in 0..channels {
            for y in 0..height {
                for x in 0..width {
                    a[[y, x, ci]] = c_vals[ci];
                }
            }
        }

        let u = affinity_with_constant_weights(&a, &kernels, &meta, &h, BorderMode::Torus);

        // Expected per-channel constant:
        //   target 0: contributions from kernels 0 (src=0) and 1 (src=1).
        //   target 1: contribution from kernel 2 (src=2).
        //   target 2: no contribution → 0.
        let expected_0 = h[0] * growth(c_vals[0], meta[0].mu, meta[0].sigma)
            + h[1] * growth(c_vals[1], meta[1].mu, meta[1].sigma);
        let expected_1 = h[2] * growth(c_vals[2], meta[2].mu, meta[2].sigma);
        let expected_2 = 0.0_f32;

        for y in 0..height {
            for x in 0..width {
                assert_relative_eq!(u[[y, x, 0]], expected_0, epsilon = 1e-4);
                assert_relative_eq!(u[[y, x, 1]], expected_1, epsilon = 1e-4);
                assert_relative_eq!(u[[y, x, 2]], expected_2, epsilon = 1e-4);
            }
        }
    }

    /// **Hand-coded Eq. 3 on a delta input**.
    ///
    /// Setup: single-channel grid (`C = 1`), single kernel (`K = 1`), with
    /// the *raw* kernel array supplied directly so the convolution
    /// arithmetic is fully predictable.
    ///
    /// `A` is a δ at the interior cell `(3, 3)`; `K` is a 3×3 identifiable
    /// asymmetric kernel (same one as `convolve.rs::k3x3`). Convolution
    /// yields a 3×3 patch at `(2..=4, 2..=4)` equal to the kernel rotated
    /// by the correlation flip — see
    /// [`crate::convolve::tests::convolve_with_delta_yields_kernel`]. The
    /// rest of the grid stays zero, so:
    ///   `U[y, x] = h · G(K_corr[y - 2, x - 2]; μ, σ)` for the 3×3 patch,
    ///   `U[y, x] = h · G(0; μ, σ)`                  elsewhere.
    ///
    /// Concrete choice: `h = 1.0`, `μ = 1.0`, `σ = 1.0`.
    /// Hand evaluation at the centre cell `(3, 3)`:
    ///   `K_corr[1, 1] = K[1, 1] = 5.0`
    ///   `G(5.0; 1.0, 1.0) = 2·exp(-((5-1)/1)²/2) − 1 = 2·exp(-8) − 1
    ///                     ≈ 2·3.3546e-4 − 1
    ///                     ≈ -0.999_329_07`
    /// At a "far" cell `(7, 7)`: `G(0; 1.0, 1.0) = 2·exp(-0.5) − 1
    ///                                            ≈ 0.213_061_32`.
    #[test]
    fn affinity_eq3_handcoded_on_delta_input() {
        // C = 1, K = 1.
        let height = 9;
        let width = 9;
        let channels = 1;
        let mut a: ActivationField = Array3::zeros((height, width, channels));
        a[[3, 3, 0]] = 1.0;

        // Supply the kernel array directly. K_meta points to
        // `effective_radius = 1` so the 3×3 array is consistent with the
        // odd-side convention.
        let k: Array2<f32> =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();
        let meta = vec![KernelMeta {
            source_channel: 0,
            target_channel: 0,
            mu: 1.0,
            sigma: 1.0,
            effective_radius: 1,
        }];
        let h = vec![1.0_f32];

        let u = affinity_with_constant_weights(&a, &[k], &meta, &h, BorderMode::Wall);

        // Centre cell — convolution there is K[1, 1] = 5.0.
        let expected_centre = 1.0 * growth(5.0, 1.0, 1.0);
        assert_relative_eq!(u[[3, 3, 0]], expected_centre, epsilon = 1e-6);

        // A far cell — convolution is 0, so U = G(0; 1, 1).
        let expected_far = 1.0 * growth(0.0, 1.0, 1.0);
        assert_relative_eq!(u[[7, 7, 0]], expected_far, epsilon = 1e-6);

        // A diagonal-adjacent cell to the delta — convolution there is
        // K[0, 0] = 1.0 (see convolve_with_delta_yields_kernel). At (4, 4),
        // the correlation form gives K[er - (4-3), er - (4-3)] = K[0, 0] = 1.
        let expected_adj = 1.0 * growth(1.0, 1.0, 1.0);
        assert_relative_eq!(u[[4, 4, 0]], expected_adj, epsilon = 1e-6);
    }

    /// **Zero input → uniform `U = h · G(0)`**.
    ///
    /// With `A ≡ 0`, the convolution is identically 0 (zero · anything =
    /// 0, regardless of border condition), so `G_i(0) = 2·bell(0; μ, σ) −
    /// 1` is the only contributor, and `U_j ≡ Σ_{i : c_i^1 = j} h_i ·
    /// G_i(0)` uniformly across the grid.
    ///
    /// The "approaches −1 if μ ≫ 0" regression is implicit: with the
    /// `three_kernel_setup` parameters (μ ≈ 0.15..0.25, σ ≈ 0.02..0.04),
    /// `G_i(0)` is close to −1 because `|0 − μ| / σ ≫ 1`. So `U_0` is
    /// roughly `-(h_0 + h_1) = -0.8` and `U_1` is roughly `-h_2 = -0.9`.
    #[test]
    fn affinity_zero_input_is_constant_h_times_g_of_zero() {
        let (kernels, meta, h) = three_kernel_setup();
        let (height, width, channels) = (10, 10, 3);
        let a: ActivationField = Array3::zeros((height, width, channels));

        let u = affinity_with_constant_weights(&a, &kernels, &meta, &h, BorderMode::Torus);

        let g0_0 = growth(0.0, meta[0].mu, meta[0].sigma);
        let g0_1 = growth(0.0, meta[1].mu, meta[1].sigma);
        let g0_2 = growth(0.0, meta[2].mu, meta[2].sigma);
        let expected_0 = h[0] * g0_0 + h[1] * g0_1;
        let expected_1 = h[2] * g0_2;

        for y in 0..height {
            for x in 0..width {
                // Convolution of zero is exactly 0 — no floating-point
                // residual at all. Bit-equal tolerance.
                assert_relative_eq!(u[[y, x, 0]], expected_0, epsilon = 1e-6);
                assert_relative_eq!(u[[y, x, 1]], expected_1, epsilon = 1e-6);
                assert_relative_eq!(u[[y, x, 2]], 0.0, epsilon = 1e-6);
                // Sanity: with these growth params, ch0 and ch1 sit near
                // −(h_0 + h_1) = −0.8 and −h_2 = −0.9 respectively.
                assert!(u[[y, x, 0]] < -0.7, "U_0 = {} (expected near -0.8)", u[[y, x, 0]]);
                assert!(u[[y, x, 1]] < -0.85, "U_1 = {} (expected near -0.9)", u[[y, x, 1]]);
            }
        }
    }

    /// **Eq. 7 with `P_i(x) ≡ h_i` reduces to Eq. 3**.
    ///
    /// This is the structural lock: any divergence between the two
    /// implementations under the "uniform P" specialisation is a bug in
    /// either the per-cell broadcast (Eq. 7 side) or the per-kernel scalar
    /// multiply (Eq. 3 side). Asserted bit-equal because both branches
    /// follow the same `(growth(conv) · weight) += U` accumulation order
    /// — no path-dependent f32 rounding should differ.
    ///
    /// (If a future refactor makes the two paths take different
    /// accumulation orders, switch this assertion to `assert_relative_eq!`
    /// with `epsilon = 1e-6` and document why.)
    #[test]
    fn affinity_localized_with_uniform_p_equals_constant_h() {
        let (kernels, meta, h) = three_kernel_setup();

        // Non-trivial input so accidental "everything is zero" doesn't
        // mask a divergence.
        let (height, width, channels) = (12, 12, 3);
        let mut a: ActivationField = Array3::zeros((height, width, channels));
        for y in 0..height {
            for x in 0..width {
                for ci in 0..channels {
                    a[[y, x, ci]] =
                        ((y * 13 + x * 7 + ci * 31) % 17) as f32 / 17.0;
                }
            }
        }

        // P_i(x) = h_i broadcast across the grid.
        let mut p_map: Array3<f32> = Array3::zeros((height, width, kernels.len()));
        for i in 0..kernels.len() {
            for y in 0..height {
                for x in 0..width {
                    p_map[[y, x, i]] = h[i];
                }
            }
        }

        let u_eq3 =
            affinity_with_constant_weights(&a, &kernels, &meta, &h, BorderMode::Torus);
        let u_eq7 =
            affinity_with_localized_weights(&a, &kernels, &meta, &p_map, BorderMode::Torus);

        assert_eq!(u_eq3.dim(), u_eq7.dim());
        for ((y, x, ci), &v3) in u_eq3.indexed_iter() {
            let v7 = u_eq7[[y, x, ci]];
            assert_eq!(
                v3.to_bits(),
                v7.to_bits(),
                "Eq.3 vs Eq.7 (uniform P) divergence at ({y}, {x}, {ci}): \
                 {v3} vs {v7}"
            );
        }
    }

    /// Defensive shape-mismatch panics: `meta.len() != kernels.len()`.
    #[test]
    #[should_panic(expected = "kernels.len()")]
    fn affinity_panics_on_meta_length_mismatch() {
        let (kernels, meta, h) = three_kernel_setup();
        let bad_meta = meta[..2].to_vec();
        let a: ActivationField = Array3::zeros((4, 4, 3));
        let _ = affinity_with_constant_weights(&a, &kernels, &bad_meta, &h, BorderMode::Torus);
    }

    /// Defensive shape-mismatch panic: `p_map.dim().2 != kernels.len()`.
    #[test]
    #[should_panic(expected = "p_map shape")]
    fn affinity_localized_panics_on_p_map_shape_mismatch() {
        let (kernels, meta, _h) = three_kernel_setup();
        let a: ActivationField = Array3::zeros((4, 4, 3));
        let bad_p: Array3<f32> = Array3::zeros((4, 4, kernels.len() + 1));
        let _ =
            affinity_with_localized_weights(&a, &kernels, &meta, &bad_p, BorderMode::Torus);
    }
}
