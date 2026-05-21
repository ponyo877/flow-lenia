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

// ─────────────────────────────────────────────────────────────────────
// M6.A.4.5 Experiment 4: per-step rel growth by grid (C=1).
//
// The M6.A.4 finding was that GPU vs CPU `field_rel` at 10 steps grows
// super-linearly with grid: 3.6e-5 (g32) → 1.1e-4 (g64) → 4.5e-4 (g128)
// → 1.1e-3 (g256). This experiment captures the rel growth at every
// step (1..10) for each grid so we can tell:
//
//   (i)   does the grid scaling appear at step 1 (a constant per-grid
//         offset, suggesting a deterministic grid-dependent bias) or
//         only after a few steps (suggesting chaos amplification);
//   (ii)  is the per-step growth linear (systematic bias accumulating)
//         or exponential (Lyapunov-type chaos);
//   (iii) is there a single "step where things blow up" or a smooth
//         monotone growth.
//
// All measurements are at C=1 / paper_strict=false / Torus / seed=42.
// The other modes are not expected to change the shape of the growth
// curve materially — if they do, that itself is a finding.
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "M6.A.4.5 diagnostic — per-step rel growth across grids"]
fn m6a45_per_step_rel_growth_by_grid_c1() {
    let ctx = headless_ctx();
    let grids: &[u32] = &[32, 64, 128, 256];
    let n_steps: u32 = 10;

    eprintln!(
        "\n[M6.A.4.5 Experiment 4] per-step rel growth by grid \
         (C=1, paper_strict=false, Torus, seed=42)"
    );
    eprintln!(" grid   step    max_abs    max_rel     log10(max_rel)  rel/step_1");

    for &grid in grids {
        let cfg = FlowLeniaConfig {
            grid_width: grid,
            grid_height: grid,
            channels: 1,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: NUM_KERNELS,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        };
        let cpu_init = FlowLeniaSimulator::new(cfg, SEED);
        let initial_a = cpu_init.activation().clone();
        let kernel_params = cpu_init.kernel_params().clone();

        let mut cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
        let mut gpu_pipeline =
            GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

        let mut step1_rel = f32::NAN;
        for step in 1..=n_steps {
            cpu_sim.step();
            gpu_pipeline.run_steps(&ctx, 1);
            let cpu_a = cpu_sim.activation();
            let gpu_a = gpu_pipeline.readback_activation(&ctx);
            let (max_abs, max_rel) = field_diff(cpu_a, &gpu_a);
            if step == 1 {
                step1_rel = max_rel;
            }
            let ratio = if step1_rel > 0.0 {
                max_rel / step1_rel
            } else {
                f32::NAN
            };
            eprintln!(
                "  {:3}   {:3}    {:.3e}   {:.3e}   {:+.3}        {:.2}",
                grid,
                step,
                max_abs,
                max_rel,
                max_rel.log10(),
                ratio
            );
        }
        eprintln!();
    }
}

// ─────────────────────────────────────────────────────────────────────
// M6.A.4.5 Experiment 5: CPU-only Lyapunov by grid (C=1).
//
// Purpose: confirm or refute H1 — "Lyapunov exponent of the
// Flow-Lenia dynamics is grid-dependent at C=1 / K=10". If yes,
// Experiment 4's GPU-vs-CPU rel growth at large grids is a faithful
// reflection of intrinsic dynamics, not a GPU-side numerical artefact;
// and the M6.A.4 tolerance has to scale with grid accordingly.
//
// Method: perturb the initial state by `epsilon = 1e-6` and run TWO
// CPU simulators forward. The two trajectories diverge at the
// dynamics' Lyapunov rate; if that rate is grid-dependent, we see
// it here without any GPU involvement.
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "M6.A.4.5 diagnostic — CPU Lyapunov by grid (C=1)"]
fn m6a45_cpu_lyapunov_by_grid_c1() {
    let grids: &[u32] = &[32, 64, 128, 256];
    let n_steps: u32 = 20;
    let epsilon: f32 = 1e-6;

    eprintln!(
        "\n[M6.A.4.5 Experiment 5] CPU-only Lyapunov by grid \
         (C=1, paper_strict=false, Torus, seed=42, eps=1e-6)"
    );
    eprintln!(" grid   step    max_abs    max_rel     log10(max_rel)");

    for &grid in grids {
        let cfg = FlowLeniaConfig {
            grid_width: grid,
            grid_height: grid,
            channels: 1,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: NUM_KERNELS,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        };

        let base = FlowLeniaSimulator::new(cfg, SEED);
        let initial_a = base.activation().clone();
        let kernel_params = base.kernel_params().clone();
        let perturbed_a = add_perturbation(&initial_a, epsilon, SEED ^ 0xDEAD_BEEF);

        let mut sim_baseline = FlowLeniaSimulator::from_components(
            initial_a,
            kernel_params.clone(),
            cfg,
        );
        let mut sim_perturbed =
            FlowLeniaSimulator::from_components(perturbed_a, kernel_params, cfg);

        for step in 1..=n_steps {
            sim_baseline.step();
            sim_perturbed.step();
            let (max_abs, max_rel) =
                field_diff(sim_baseline.activation(), sim_perturbed.activation());
            eprintln!(
                "  {:3}   {:3}    {:.3e}   {:.3e}   {:+.3}",
                grid,
                step,
                max_abs,
                max_rel,
                max_rel.log10()
            );
        }
        eprintln!();
    }
}

// ─────────────────────────────────────────────────────────────────────
// M6.A.4.5 Experiment 6: GPU rel non-determinism check.
//
// Run the g256 C=1 10-step GPU vs CPU comparison N times with the
// same seed in the same process. If the GPU pipeline is fully
// deterministic given a fixed driver / shader-compile state, all N
// runs should produce *bit-identical* GPU outputs and therefore the
// same `max_rel`. Any variance comes from sources outside our
// control (driver retries, async kernel reordering, …) and informs
// how wide the regression tolerance margin needs to be.
//
// Used to pick between the M6.A.4.5 tiered-tolerance variants:
//   - variance < 2× → 5× safety margin is enough (case A)
//   - variance ≥ 5× → 10× safety margin (case B)
// ─────────────────────────────────────────────────────────────────────

#[test]
#[ignore = "M6.A.4.5 diagnostic — chaos non-determinism at g256 C=1"]
fn m6a45_chaos_nondeterminism_g256_c1() {
    let ctx = headless_ctx();
    let n_runs: usize = 5;

    let cfg = FlowLeniaConfig {
        grid_width: 256,
        grid_height: 256,
        channels: 1,
        dt: 0.2,
        sigma: 0.65,
        n: 2.0,
        beta_a: 2.0,
        dd: 5,
        num_kernels: NUM_KERNELS,
        paper_strict: true, // matches the case_g256_psT_btT_c1 (the worst-case M6.A.4 measurement)
        border: BorderMode::Torus,
        mix_rule: MixRule::Stochastic,
    };
    let cpu_init = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_init.activation().clone();
    let kernel_params = cpu_init.kernel_params().clone();

    let mut cpu_ref = FlowLeniaSimulator::new(cfg, SEED);
    cpu_ref.step_many(10);
    let cpu_a = cpu_ref.activation().clone();

    let mut rels: Vec<f32> = Vec::with_capacity(n_runs);
    for run in 0..n_runs {
        let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);
        pipeline.run_steps(&ctx, 10);
        let gpu_a = pipeline.readback_activation(&ctx);
        let (_max_abs, max_rel) = field_diff(&cpu_a, &gpu_a);
        eprintln!(
            "  run {} / {}: max_rel = {:.6e}",
            run + 1,
            n_runs,
            max_rel
        );
        rels.push(max_rel);
    }

    let n = rels.len() as f32;
    let mean: f32 = rels.iter().sum::<f32>() / n;
    let min = rels.iter().copied().fold(f32::INFINITY, f32::min);
    let max = rels.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let var: f32 = rels.iter().map(|r| (r - mean) * (r - mean)).sum::<f32>() / n;
    let std = var.sqrt();
    let ratio = if min > 0.0 { max / min } else { f32::NAN };

    eprintln!("\n[M6.A.4.5 Experiment 6] g256 C=1 10-step GPU rel non-determinism");
    eprintln!("  n_runs = {n_runs}");
    eprintln!("  min    = {min:.6e}");
    eprintln!("  max    = {max:.6e}");
    eprintln!("  mean   = {mean:.6e}");
    eprintln!("  std    = {std:.6e}");
    eprintln!("  max/min ratio = {ratio:.4}x");
    eprintln!();
}

// `Array3` is referenced via the type alias `ActivationField`; this
// keeps the import warning-free under cfg(test).
#[allow(dead_code)]
fn _silence_unused_import(_: Array3<f32>) {}
