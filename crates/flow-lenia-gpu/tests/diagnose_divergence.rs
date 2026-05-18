//! M2.8 follow-up diagnostics — investigate whether the observed
//! `C = 3 / K = 10 / 100 step` CPU vs GPU field divergence is true
//! chaotic dynamics or hides a latent implementation bug.
//!
//! All three tests are `#[ignore]`'d so normal `cargo test` skips them;
//! run with:
//!
//! ```text
//! cargo test --release -p flow-lenia-gpu --test diagnose_divergence \
//!     -- --ignored --nocapture
//! ```
//!
//! Each test prints a table of measured numbers — no `assert!` on the
//! growth rates themselves, the human reads the numbers and decides
//! the interpretation.

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    state::ActivationField,
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{GpuContext, GpuStepPipeline};
use ndarray::Array3;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;

const SEED: u64 = 42;
const GRID: u32 = 32;
const NUM_KERNELS: u32 = 10;

fn fixture_cfg(channels: u32, paper_strict: bool, border: BorderMode) -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: GRID,
        grid_height: GRID,
        channels,
        dt: 0.2,
        sigma: 0.65,
        n: 2.0,
        beta_a: 2.0,
        dd: 5,
        num_kernels: NUM_KERNELS,
        paper_strict,
        border,
        mix_rule: MixRule::Stochastic,
    }
}

/// (max_abs, max_rel) of two same-shape `ActivationField`s, using the
/// `abs / max(abs, 1e-6)` relative-error guard.
fn field_diff(a: &ActivationField, b: &ActivationField) -> (f32, f32) {
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for ((y, x, c), &av) in a.indexed_iter() {
        let bv = b[[y, x, c]];
        let abs_err = (av - bv).abs();
        let rel_err = abs_err / av.abs().max(1e-6);
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
    }
    (max_abs, max_rel)
}

fn headless_ctx() -> GpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    GpuContext::new_blocking(instance, None)
}

// ─────────────────────────────────────────────────────────────────────
// Experiment 1: step-by-step CPU vs GPU divergence growth.
//
// Same `(cfg, seed=42)` as the M1.15 fixtures. C=3, paper_strict=false,
// torus. Walk both implementations forward one step at a time and
// measure `(max_abs, max_rel)` after each step.
//
// Expected signatures of the three hypotheses:
//   (A) chaotic: `log(max_rel)` ≈ linear in n  →  exponential growth
//   (B) sudden bug: huge jump at some step
//   (C) accumulating bug: `max_rel` grows polynomially (n, n², …)
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "diagnostic only (M2.8 verify)"]
fn diagnose_c3_divergence_growth() {
    let cfg = fixture_cfg(3, false, BorderMode::Torus);
    let cpu_sim_init = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_sim_init.activation().clone();
    let kernel_params = cpu_sim_init.kernel_params().clone();

    let ctx = headless_ctx();
    let mut cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
    let mut gpu_pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

    eprintln!("\n[Experiment 1] step-by-step CPU vs GPU divergence (C=3, K=10, torus, seed=42)");
    eprintln!(" step    max_abs    max_rel     log10(max_rel)  log_ratio_to_prev");

    let mut prev_log = f32::NEG_INFINITY;
    // Sample at step counts 1..=20 plus every 10 thereafter up to 100.
    let checkpoints: Vec<u32> = (1..=20).chain((30..=100).step_by(10)).collect();
    let mut prev_step: u32 = 0;
    for &target_step in &checkpoints {
        let dn = target_step - prev_step;
        cpu_sim.step_many(dn);
        gpu_pipeline.run_steps(&ctx, dn);
        prev_step = target_step;

        let cpu_a = cpu_sim.activation();
        let gpu_a = gpu_pipeline.readback_activation(&ctx);
        let (max_abs, max_rel) = field_diff(cpu_a, &gpu_a);
        let log_rel = max_rel.log10();
        let log_ratio = if prev_log.is_finite() {
            (log_rel - prev_log) / dn as f32
        } else {
            0.0
        };
        eprintln!(
            "  {:3}   {:.3e}   {:.3e}   {:+.3}        {:+.3} /step",
            target_step, max_abs, max_rel, log_rel, log_ratio
        );
        prev_log = log_rel;
    }
}

// ─────────────────────────────────────────────────────────────────────
// Experiment 2: CPU-only Lyapunov exponent estimate.
//
// Perturb the initial state by `epsilon * unit_random_perturbation` and
// run **both copies on CPU** for 100 steps. The growth of the
// difference between the two trajectories is the empirical Lyapunov
// estimate for this dynamics.
//
// Done for C=1 and C=3 separately. If C=3's λ ≫ C=1's λ, the
// chaotic-dynamics explanation for the M2.8 C=3 divergence holds. If
// the two λ's are comparable, the chaotic-dynamics explanation is
// unlikely.
// ─────────────────────────────────────────────────────────────────────

fn add_perturbation(a: &ActivationField, epsilon: f32, seed: u64) -> ActivationField {
    let (h, w, c) = a.dim();
    let mut rng = ChaCha8Rng::seed_from_u64(seed);
    let mut perturbed = a.clone();
    for y in 0..h {
        for x in 0..w {
            for ci in 0..c {
                // Uniform perturbation in [-epsilon, +epsilon).
                perturbed[[y, x, ci]] += rng.gen_range(-epsilon..epsilon);
            }
        }
    }
    perturbed
}

fn lyapunov_run(channels: u32, label: &str) {
    let cfg = fixture_cfg(channels, false, BorderMode::Torus);

    let base = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = base.activation().clone();
    let kernel_params = base.kernel_params().clone();
    let epsilon = 1e-6_f32;
    let perturbed_a = add_perturbation(&initial_a, epsilon, SEED ^ 0xDEAD_BEEF);

    // Two CPU simulators with identical kernels but different initial A.
    let mut sim_baseline =
        FlowLeniaSimulator::from_components(initial_a, kernel_params.clone(), cfg);
    let mut sim_perturbed = FlowLeniaSimulator::from_components(perturbed_a, kernel_params, cfg);

    eprintln!("\n[Experiment 2 — {label}] CPU Lyapunov estimate (eps = {epsilon:.0e})");
    eprintln!(" step    max_abs    max_rel     log10(max_rel)  ");

    let checkpoints: Vec<u32> = (1..=20).chain((30..=100).step_by(10)).collect();
    let mut prev_step: u32 = 0;
    for &target_step in &checkpoints {
        let dn = target_step - prev_step;
        sim_baseline.step_many(dn);
        sim_perturbed.step_many(dn);
        prev_step = target_step;

        let (max_abs, max_rel) = field_diff(sim_baseline.activation(), sim_perturbed.activation());
        eprintln!(
            "  {:3}   {:.3e}   {:.3e}   {:+.3}",
            target_step,
            max_abs,
            max_rel,
            max_rel.log10()
        );
    }
}

#[test]
#[ignore = "diagnostic only (M2.8 verify)"]
fn estimate_lyapunov_exponent_c1_vs_c3() {
    lyapunov_run(1, "C=1");
    lyapunov_run(3, "C=3");
}

// ─────────────────────────────────────────────────────────────────────
// Experiment 3: precise 10-step C=3 measurement.
//
// Confirms the numbers Experiment 1 reports at step=10 against a clean
// fresh-simulator setup. If the exponential-growth interpretation is
// right, `max_abs` at step 10 should be ≪ 1 (very small) and the
// `step → 100` extrapolation should overshoot the observed 100-step
// number (because Lyapunov saturation kicks in once the perturbation
// reaches O(1) of the field magnitude).
// ─────────────────────────────────────────────────────────────────────

#[test]
fn c3_divergence_at_10_steps() {
    let cfg = fixture_cfg(3, false, BorderMode::Torus);
    let mut cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();

    let ctx = headless_ctx();
    let mut gpu_pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

    let n: u32 = 10;
    cpu_sim.step_many(n);
    gpu_pipeline.run_steps(&ctx, n);

    let (max_abs, max_rel) = field_diff(
        cpu_sim.activation(),
        &gpu_pipeline.readback_activation(&ctx),
    );
    eprintln!(
        "\n[Experiment 3] C=3 K=10 torus, 10 steps : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
    );

    // Sanity: 10-step max_abs must be < 1 — otherwise the dynamics has
    // already saturated and the "chaotic" interpretation is moot.
    assert!(max_abs < 1.0, "10-step max_abs {max_abs} already saturated");
}

// `Array3` is referenced via the type alias `ActivationField`; this
// keeps the import warning-free under cfg(test).
#[allow(dead_code)]
fn _silence_unused_import(_: Array3<f32>) {}
