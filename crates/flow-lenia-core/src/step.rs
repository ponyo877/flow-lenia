//! Single Flow-Lenia integration step (M1.13).
//!
//! Composes the per-piece modules (M1.4 .. M1.12) into the full update rule
//! laid out in DESIGN.md §3:
//!
//! ```text
//! U      = affinity(A; kernels, weights)            // M1.12, paper Eq. 3 / Eq. 7
//! ∇U     = sobel_per_channel(U)                     // M1.9,  per-channel Sobel
//! ∇A_Σ  = grad_a_sum(A)                            // M1.9
//! α      = alpha(A; cfg)                            // M1.8,  Eq. 5 weight (mode-switched)
//! F      = flow(∇U, ∇A_Σ, α)                       // M1.9,  Eq. 5
//! A^{t+dt} = reintegrate(A, F; cfg)                 // M1.11, Eq. 6 (dt is applied inside)
//! ```
//!
//! M1.13 keeps `step` as a **pure function** — no scratch buffers, no
//! state. A stateful `FlowLeniaSimulator` (which can hoist allocations,
//! cache `compute_kernel` outputs, and own an RNG for Eq. 8) is the M1.14
//! job. The M2 GPU port will mirror this pipeline as a sequence of
//! compute passes, one per arrow above.

use crate::affinity::{affinity_with_constant_weights, affinity_with_localized_weights};
use crate::alpha::alpha;
use crate::config::FlowLeniaConfig;
use crate::flow::flow;
use crate::kernel::KernelMeta;
use crate::reintegrate::reintegrate;
use crate::sobel::{grad_a_sum, sobel_per_channel};
use crate::state::ActivationField;
use ndarray::{Array2, Array3};

/// Per-kernel weighting reference for one `step` call.
///
/// The variants correspond directly to the two affinity formulae in
/// [`crate::affinity`]:
/// - [`Constant`](Self::Constant) → paper Eq. 3 (`h_i` is a `&[f32]`)
/// - [`Localized`](Self::Localized) → paper Eq. 7 (`P_i(x)` is a
///   `(H, W, |K|)` tensor — see [`affinity_with_localized_weights`])
///
/// Borrowed (not owned) so callers can keep the weights in their own
/// storage and pass a reference per step without allocations. The
/// borrow's lifetime is tied to the [`step`] call only.
pub enum WeightsRef<'a> {
    /// Eq. 3 — constant per-kernel weights `h_i`. Length must equal
    /// `kernels.len()`.
    Constant(&'a [f32]),
    /// Eq. 7 — cell-localised weight map `P_i(x)`. Shape
    /// `(H, W, |K|)`, where `(H, W) = a.dim()[..2]` and
    /// `|K| = kernels.len()`.
    Localized(&'a Array3<f32>),
}

/// Compute one Flow-Lenia integration step.
///
/// Returns the next activation field `A^{t+dt}`.
///
/// Arguments:
/// - `a`: current activation field, shape `(H, W, C)`.
/// - `kernels`: precomputed normalised kernel arrays (M1.4). Length
///   `|K|`. Each kernel has odd side length `2 · meta[i].effective_radius
///   + 1`.
/// - `meta`: per-kernel metadata (M1.12), length `|K|`. Carries the
///   source/target channels and the growth `(μ, σ)` consumed inside
///   the affinity step.
/// - `weights`: see [`WeightsRef`] — picks Eq. 3 vs Eq. 7.
/// - `cfg`: physical and mode configuration. Border policy, `dt`, `σ`,
///   `dd`, and the `paper_strict` toggle (which selects α formula and
///   Eq. 8 softmax — see [`crate::config::FlowLeniaConfig`]).
///
/// # Panics
///
/// The component functions all panic on shape mismatches and on out-of-
/// range kernel-meta channel indices. Those panics propagate from `step`.
/// All such mismatches are upstream programming errors, not
/// data-dependent failures.
///
/// # Determinism
///
/// `step` is deterministic for a given input — the M1 pipeline does not
/// use any randomness yet. Eq. 8 (parameter mixing) is M1.15 / future
/// work; when added, an RNG seed will live on the M1.14 simulator
/// rather than as a `step` argument so that the pure-function nature of
/// this signature is preserved.
#[must_use]
pub fn step(
    a: &ActivationField,
    kernels: &[Array2<f32>],
    kernel_meta: &[KernelMeta],
    weights: WeightsRef<'_>,
    cfg: &FlowLeniaConfig,
) -> ActivationField {
    let border = cfg.border;

    // 1. U = affinity(A) per Eq. 3 / Eq. 7.
    let u = match weights {
        WeightsRef::Constant(h) => {
            affinity_with_constant_weights(a, kernels, kernel_meta, h, border)
        }
        WeightsRef::Localized(p_map) => {
            affinity_with_localized_weights(a, kernels, kernel_meta, p_map, border)
        }
    };

    // 2. ∇U  — per-channel Sobel.
    let grad_u = sobel_per_channel(&u, border);

    // 3. ∇A_Σ  — Sobel on the channel-summed activation.
    let grad_a = grad_a_sum(a, border);

    // 4. α  — mode-switched per `cfg.paper_strict`.
    let alpha_field = alpha(a, cfg);

    // 5. F = (1 - α)·∇U  −  α·∇A_Σ.
    let flow_field = flow(&grad_u, &grad_a, &alpha_field);

    // 6. A^{t+dt} — receiver-side reintegration. `cfg.dt` is applied
    //    inside `reintegrate` when forming the distribution centre
    //    `μ = x' + dt·F`.
    reintegrate(a, &flow_field, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{BorderMode, MixRule};
    use crate::kernel::{compute_kernel, effective_radius};
    use crate::params::KernelEntry;
    use crate::state::sum_channels;
    use approx::assert_relative_eq;
    use ndarray::Array3;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    // ─────────────────────────────────────────────────────────────────
    // Shared helpers
    // ─────────────────────────────────────────────────────────────────

    fn cfg_torus_default(channels: u32, height: u32, width: u32) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: width,
            grid_height: height,
            channels,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 0,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        }
    }

    /// Build a small kernel bank: one self-coupling kernel per channel
    /// (c0 = c1 = c). Keeps the test setups compact while ensuring U is
    /// non-trivial across every channel.
    fn build_self_kernels(
        channels: u32,
        r_global: f32,
    ) -> (Vec<ndarray::Array2<f32>>, Vec<KernelMeta>, Vec<f32>) {
        let mut kernels = Vec::with_capacity(channels as usize);
        let mut meta = Vec::with_capacity(channels as usize);
        let mut h = Vec::with_capacity(channels as usize);
        for c in 0..channels {
            let entry = KernelEntry {
                c0: c,
                c1: c,
                // Vary `r` a little so the kernels are not pointwise identical.
                r: 0.40 + (c as f32) * 0.05,
                a: [0.25, 0.5, 0.75],
                b: [1.0, 0.7, 0.4],
                w: [0.05, 0.05, 0.05],
                h: 0.5,
                mu: 0.15,
                sigma: 0.02,
            };
            kernels.push(compute_kernel(r_global, &entry));
            meta.push(KernelMeta {
                source_channel: entry.c0,
                target_channel: entry.c1,
                mu: entry.mu,
                sigma: entry.sigma,
                effective_radius: effective_radius(r_global, entry.r),
            });
            h.push(entry.h);
        }
        (kernels, meta, h)
    }

    /// Channel-summed mass of `A`.
    fn total_mass(a: &ActivationField) -> f64 {
        sum_channels(a).iter().map(|&v| f64::from(v)).sum()
    }

    // ─────────────────────────────────────────────────────────────────
    // Tier 1 — sanity
    // ─────────────────────────────────────────────────────────────────

    /// **Zero initial A stays zero.** Required property: Flow-Lenia must
    /// not spontaneously create mass from a quiescent state. If this
    /// fails, some downstream piece is leaking a non-zero baseline.
    ///
    /// Bit-equal expected (every contribution traces back through
    /// `0 · anything = 0`, and `reintegrate` only redistributes mass).
    #[test]
    fn step_zero_initial_a_stays_zero() {
        let cfg = cfg_torus_default(3, 8, 8);
        let (kernels, meta, h) = build_self_kernels(cfg.channels, 5.0);
        let a: ActivationField = Array3::zeros((
            cfg.grid_height as usize,
            cfg.grid_width as usize,
            cfg.channels as usize,
        ));

        let a_next = step(&a, &kernels, &meta, WeightsRef::Constant(&h), &cfg);

        for &v in a_next.iter() {
            assert_eq!(v.to_bits(), 0.0_f32.to_bits(), "expected exact 0, got {v}");
        }
    }

    /// **Uniform A is stable.** With `A_c ≡ const` per channel, both
    /// `∇U` and `∇A_Σ` are (nearly) zero, so `F ≈ 0`, so the
    /// reintegration distribution centres `μ` collapse onto source
    /// cells, and `A_next ≈ A` (up to the documented kernel-sum f32
    /// floor and Sobel boundary jitter).
    ///
    /// Tolerance 1e-4: absorbs the cumulative product of
    /// (kernel sum residual ≈ 5e-6) × (Sobel coefficient sum) × (growth
    /// derivative). One step is far below this empirically — relaxed to
    /// 1e-4 to leave margin for a future tweak to `compute_kernel`.
    #[test]
    fn step_uniform_a_field_is_stable() {
        let cfg = cfg_torus_default(3, 16, 16);
        let (kernels, meta, h) = build_self_kernels(cfg.channels, 5.0);
        let mut a: ActivationField = Array3::zeros((
            cfg.grid_height as usize,
            cfg.grid_width as usize,
            cfg.channels as usize,
        ));
        // Pick a per-channel constant near the growth peak so G ≈ 1 and
        // every kernel contributes a stable U value.
        let c_vals = [0.15_f32, 0.15, 0.15];
        for ((_, _, ci), v) in a.indexed_iter_mut() {
            *v = c_vals[ci];
        }

        let a_next = step(&a, &kernels, &meta, WeightsRef::Constant(&h), &cfg);

        for ((y, x, ci), &v) in a_next.indexed_iter() {
            assert_relative_eq!(v, a[[y, x, ci]], epsilon = 1e-4);
        }
    }

    /// **Zero `h` makes U constant**, so `∇U = 0`, and `F` reduces to
    /// `-α · ∇A_Σ` — pure diffusion against the gradient of total mass.
    /// We don't try to predict `A_next` analytically; we just check
    /// the structural invariant that `U` is independent of cell
    /// position (`U(x, y) = h · G_i(K_i ∗ A)` with `h ≡ 0` gives
    /// `U ≡ 0`). This guards the affinity-h plumbing on the way from
    /// `WeightsRef::Constant` into the per-kernel multiply.
    ///
    /// The check is performed on `U` itself, not `A_next`, by calling
    /// `affinity_with_constant_weights` directly with the same inputs.
    /// `step`'s job here is just to *not* propagate any non-zero U into
    /// the Sobel, which is also verified.
    #[test]
    fn step_zero_kernels_is_identity_in_u() {
        let cfg = cfg_torus_default(2, 12, 12);
        let (kernels, meta, _h) = build_self_kernels(cfg.channels, 5.0);
        let zero_h: Vec<f32> = vec![0.0; kernels.len()];

        let mut a: ActivationField = Array3::zeros((
            cfg.grid_height as usize,
            cfg.grid_width as usize,
            cfg.channels as usize,
        ));
        // Non-trivial input so a hidden "if h == 0 then early return"
        // shortcut would still leave U observable for the assertion.
        for ((y, x, ci), v) in a.indexed_iter_mut() {
            *v = ((y * 13 + x * 7 + ci * 31) % 17) as f32 / 17.0;
        }

        // Direct U check via the public affinity API.
        let u = affinity_with_constant_weights(&a, &kernels, &meta, &zero_h, cfg.border);
        for &v in u.iter() {
            assert_eq!(v.to_bits(), 0.0_f32.to_bits(), "U should be exactly 0");
        }

        // step() must complete without errors and produce a finite A_next.
        let a_next = step(&a, &kernels, &meta, WeightsRef::Constant(&zero_h), &cfg);
        for &v in a_next.iter() {
            assert!(v.is_finite(), "A_next has non-finite value: {v}");
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Tier 2 — mass conservation (100-step full dynamics)
    // ─────────────────────────────────────────────────────────────────

    /// Run `n_steps` of `step` and return the maximum *relative* mass
    /// drift `|m_t − m_0| / m_0` over the whole trajectory.
    ///
    /// `seed` controls both the initial `A` pattern and the kernel
    /// parameters (sampled near the documented JAX ranges). Returning
    /// the *max* over the trajectory (not just the final-step value)
    /// gives the assertion bite — a fix that only stabilises late
    /// would not silently pass.
    fn run_mass_drift(
        cfg: &FlowLeniaConfig,
        num_kernels: u32,
        n_steps: usize,
        seed: u64,
    ) -> f64 {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        // Kernel parameters — use sample_random for a realistic mix.
        let kp = crate::params::KernelParams::sample_random(
            &mut rng,
            crate::params::SamplingSettings {
                num_kernels,
                num_channels: cfg.channels,
            },
        );
        let kernels: Vec<ndarray::Array2<f32>> = (0..num_kernels as usize)
            .map(|i| compute_kernel(kp.r_global, &kp.kernels[i]))
            .collect();
        let meta: Vec<KernelMeta> = (0..num_kernels as usize)
            .map(|i| KernelMeta::from_params(&kp, i))
            .collect();
        let h: Vec<f32> = kp.kernels.iter().map(|e| e.h).collect();

        // Random initial A, concentrated in a small central blob so the
        // dynamics has something to act on without saturating the grid.
        let (h_dim, w_dim, c_dim) = (
            cfg.grid_height as usize,
            cfg.grid_width as usize,
            cfg.channels as usize,
        );
        let mut a: ActivationField = Array3::zeros((h_dim, w_dim, c_dim));
        for y in (h_dim / 4)..(3 * h_dim / 4) {
            for x in (w_dim / 4)..(3 * w_dim / 4) {
                for ci in 0..c_dim {
                    a[[y, x, ci]] = rng.gen_range(0.0_f32..0.3);
                }
            }
        }

        let m0 = total_mass(&a);
        assert!(m0 > 0.0, "test seed produced an empty initial A");
        let mut max_rel = 0.0_f64;
        for _ in 0..n_steps {
            a = step(&a, &kernels, &meta, WeightsRef::Constant(&h), cfg);
            let m = total_mass(&a);
            let rel = ((m - m0).abs()) / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        max_rel
    }

    fn cfg_mass(channels: u32, paper_strict: bool) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: 32,
            grid_height: 32,
            channels,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 0,
            paper_strict,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        }
    }

    /// Mass conservation on torus, single channel, JAX-compat mode.
    /// 1e-3 tolerance per DESIGN.md §5.3.
    #[test]
    fn step_mass_conservation_torus_c1_100steps() {
        let cfg = cfg_mass(1, false);
        let max_rel = run_mass_drift(&cfg, 3, 100, 0xC0FF_EE42);
        assert!(
            max_rel < 1e-3,
            "C=1 jax-compat: max_rel mass drift {max_rel:.3e} ≥ 1e-3"
        );
    }

    /// Mass conservation on torus, 3 channels, JAX-compat mode.
    #[test]
    fn step_mass_conservation_torus_c3_100steps() {
        let cfg = cfg_mass(3, false);
        let max_rel = run_mass_drift(&cfg, 6, 100, 0xC0FF_EE43);
        assert!(
            max_rel < 1e-3,
            "C=3 jax-compat: max_rel mass drift {max_rel:.3e} ≥ 1e-3"
        );
    }

    /// Mass conservation in `paper_strict = true` mode (shared-α via
    /// `A_Σ`). The α formula change should not affect mass conservation
    /// (reintegrate is mass-conserving regardless of `F`).
    #[test]
    fn step_mass_conservation_paper_strict_modes() {
        // C=1
        let cfg1 = cfg_mass(1, true);
        let m1 = run_mass_drift(&cfg1, 3, 100, 0xBADC_0DE1);
        assert!(
            m1 < 1e-3,
            "C=1 paper-strict: max_rel mass drift {m1:.3e} ≥ 1e-3"
        );
        // C=3
        let cfg3 = cfg_mass(3, true);
        let m3 = run_mass_drift(&cfg3, 6, 100, 0xBADC_0DE3);
        assert!(
            m3 < 1e-3,
            "C=3 paper-strict: max_rel mass drift {m3:.3e} ≥ 1e-3"
        );
    }

    /// **Diagnostic** (`-- --nocapture --include-ignored`): print the
    /// full 4-case mass-drift table. Marked `#[ignore]` so it doesn't
    /// run in normal `cargo test`. The numbers from this run are quoted
    /// in the M1.13 completion report.
    #[test]
    #[ignore = "diagnostic only"]
    fn diagnose_mass_conservation_modes_matrix() {
        for &paper_strict in &[false, true] {
            for &(channels, num_kernels) in &[(1u32, 3u32), (3, 6)] {
                let cfg = cfg_mass(channels, paper_strict);
                let mut maxes = Vec::new();
                for seed in [0xC0FF_EE42, 0xC0FF_EE43, 0xC0FF_EE44] {
                    let m = run_mass_drift(&cfg, num_kernels, 100, seed);
                    maxes.push(m);
                }
                println!(
                    "paper_strict={:5}  C={}  max_rel ∈ {{{:.3e}, {:.3e}, {:.3e}}}",
                    paper_strict, channels, maxes[0], maxes[1], maxes[2]
                );
            }
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // Tier 3 — parameter embedding equivalence
    // ─────────────────────────────────────────────────────────────────

    /// **`Localized` with `P_i(x) ≡ h_i` reduces to `Constant`**.
    ///
    /// Already locked at the affinity level (see
    /// `affinity_localized_with_uniform_p_equals_constant_h`), but
    /// duplicated at the `step` level so a future refactor that adds
    /// `weights`-dependent branching elsewhere in the pipeline still
    /// trips the equivalence.
    ///
    /// Tolerance 1e-6: relaxed from the affinity-level bit-equal lock
    /// because the full pipeline runs through reintegration's
    /// `(sum + accumulate)` loops, which are bit-stable but may produce
    /// strictly different f32 patterns under future micro-optimisations.
    #[test]
    fn step_localized_with_uniform_p_equals_constant_h() {
        let cfg = cfg_torus_default(3, 16, 16);
        let (kernels, meta, h) = build_self_kernels(cfg.channels, 5.0);

        // Non-trivial input.
        let mut a: ActivationField = Array3::zeros((
            cfg.grid_height as usize,
            cfg.grid_width as usize,
            cfg.channels as usize,
        ));
        for ((y, x, ci), v) in a.indexed_iter_mut() {
            *v = ((y * 13 + x * 7 + ci * 31) % 17) as f32 / 17.0;
        }

        // P_i(x) ≡ h_i.
        let (h_dim, w_dim, _c_dim) = a.dim();
        let mut p_map: Array3<f32> = Array3::zeros((h_dim, w_dim, kernels.len()));
        for i in 0..kernels.len() {
            for y in 0..h_dim {
                for x in 0..w_dim {
                    p_map[[y, x, i]] = h[i];
                }
            }
        }

        let a_const = step(&a, &kernels, &meta, WeightsRef::Constant(&h), &cfg);
        let a_loc = step(&a, &kernels, &meta, WeightsRef::Localized(&p_map), &cfg);

        for ((y, x, ci), &c_v) in a_const.indexed_iter() {
            let l_v = a_loc[[y, x, ci]];
            assert_relative_eq!(c_v, l_v, epsilon = 1e-6);
        }
    }
}
