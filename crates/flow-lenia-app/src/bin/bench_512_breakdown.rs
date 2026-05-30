#![deny(warnings)]
//! M6.C-3-3 per-pass GPU timing breakdown at N=512.
//!
//! CLAUDE.md 原則 1 (原因究明先行): before applying subgroup /
//! mixed-precision / workgroup-tuning optimizations to the 512 hi-end
//! pipeline, measure **which pass dominates** so the optimization
//! targets the real bottleneck (C-2 taught us a dispatch-count
//! micro-opt is ~0× when the path is compute-bound elsewhere).
//!
//! Uses `GpuStepPipeline::profile_passes_fft` (CPU clock + per-pass
//! submit/poll; the GPU `TIMESTAMP_QUERY` variant hung on wgpu 29 +
//! Metal, see `profile_passes_fft` rustdoc). Breakdown is **relative**
//! — absolute per-pass µs include a uniform `submit + poll(Wait)`
//! floor that is bounded above by ~1.55 ms (M6.C-3-3 empirical) but
//! not measured in isolation; do NOT compare absolute values with
//! `bench_c2_configs` ms/step.
//!
//! **Sanity check (M6.C-3-3 adversarial-reviewer MUST item)**: before
//! the per-pass breakdown runs, the bench verifies that
//! `profile_passes_fft` and the production `step()` path produce the
//! same activation state after the same step count, at N=64 K=10 C=3.
//! Bit-equal would prove the per-pass encoder split (vs single
//! `record_step_fft` encoder) is GPU-side identical; if instead a
//! small `rel` is reported, it pins down the cost of the split for
//! later interpretation.
//!
//! ```text
//! cargo run --release --bin bench_512_breakdown
//! ```

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{pipeline::ConvolveMode, GpuContext, GpuStepPipeline};

const SEED: u64 = 1729;
const ITERS: u32 = 30;
const SANITY_STEPS: u32 = 10;
const SANITY_GRID: u32 = 64;
const SANITY_CHANNELS: u32 = 3;

fn cfg(grid: u32, channels: u32) -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: grid,
        grid_height: grid,
        channels,
        dt: 0.2,
        sigma: 0.65,
        n: 2.0,
        beta_a: 2.0,
        dd: 5,
        num_kernels: 10,
        paper_strict: false,
        border: BorderMode::Torus,
        mix_rule: MixRule::Stochastic,
    }
}

fn breakdown(ctx: &GpuContext, grid: u32, channels: u32) {
    let c = cfg(grid, channels);
    let cpu_init = FlowLeniaSimulator::new(c, SEED);
    let initial_a = cpu_init.activation().clone();
    let kernel_params = cpu_init.kernel_params().clone();
    let mut pipeline =
        GpuStepPipeline::new_with_mode(ctx, &c, &kernel_params, &initial_a, ConvolveMode::Fft);
    assert_eq!(pipeline.convolve_mode(), ConvolveMode::Fft);

    let per_pass = pipeline.profile_passes_fft(ctx, ITERS);
    let total_ns: f64 = per_pass.iter().map(|(_, ns)| ns).sum();

    eprintln!("\n=== N={grid} C={channels} K=10 per-pass breakdown (mean over {ITERS} steps) ===");
    for (label, ns) in &per_pass {
        eprintln!(
            "  {label:<16} {us:>9.3} µs  ({pct:>5.1}%)",
            us = ns / 1000.0,
            pct = 100.0 * ns / total_ns,
        );
    }
    eprintln!(
        "  {:<16} {us:>9.3} µs  (sum-of-passes; wall ms/step differs by submit overhead)",
        "TOTAL",
        us = total_ns / 1000.0,
    );
}

/// M6.C-3-3 adversarial-reviewer MUST item: prove that
/// `profile_passes_fft` (per-pass individual encoders + per-pass
/// drains) and the production `step()` (single encoder per step)
/// reach the same activation state after the same number of steps.
/// If they don't match bit-equal, the relative breakdown still
/// describes a real GPU workload but it's a *different* one from
/// production, which would invalidate judgement A's per-pass
/// percentages.
///
/// Runs at N=64 K=10 C=3 (small + fast) for `SANITY_STEPS` steps.
/// Same seed → identical CPU init → identical initial activation
/// on both pipelines.
fn sanity_check_profile_matches_step(ctx: &GpuContext) -> bool {
    let c = cfg(SANITY_GRID, SANITY_CHANNELS);
    let cpu_init = FlowLeniaSimulator::new(c, SEED);
    let initial_a = cpu_init.activation().clone();
    let kernel_params = cpu_init.kernel_params().clone();

    // Pipeline A: production step() path.
    let mut pa =
        GpuStepPipeline::new_with_mode(ctx, &c, &kernel_params, &initial_a, ConvolveMode::Fft);
    // profile_passes_fft = 5 warmup + ITERS_arg timed = (5 + n) total
    // step()s. Match that for parity.
    let total_steps = 5 + SANITY_STEPS;
    for _ in 0..total_steps {
        pa.step(ctx);
    }
    let a_final = pa.readback_activation(ctx);

    // Pipeline B: profile_passes_fft path (same initial state).
    let mut pb =
        GpuStepPipeline::new_with_mode(ctx, &c, &kernel_params, &initial_a, ConvolveMode::Fft);
    let _ = pb.profile_passes_fft(ctx, SANITY_STEPS);
    let b_final = pb.readback_activation(ctx);

    assert_eq!(a_final.dim(), b_final.dim(), "activation shape mismatch");
    let mut max_abs: f32 = 0.0;
    let mut max_rel: f32 = 0.0;
    let mut a_norm: f32 = 0.0;
    let mut b_norm: f32 = 0.0;
    for (av, bv) in a_final.iter().zip(b_final.iter()) {
        let d = (av - bv).abs();
        if d > max_abs {
            max_abs = d;
        }
        let denom = av.abs().max(bv.abs()).max(1e-9);
        let r = d / denom;
        if r > max_rel {
            max_rel = r;
        }
        a_norm += av * av;
        b_norm += bv * bv;
    }
    let a_norm = a_norm.sqrt();
    let b_norm = b_norm.sqrt();
    let global_rel = (a_norm - b_norm).abs() / a_norm.max(1e-9);

    eprintln!(
        "\n--- sanity: step() vs profile_passes_fft (N={SANITY_GRID} C={SANITY_CHANNELS} K=10, {total_steps} steps each)"
    );
    eprintln!("    max |Δ|       = {max_abs:.3e}");
    eprintln!("    max rel       = {max_rel:.3e}");
    eprintln!("    ‖A‖₂          = {a_norm:.6e}");
    eprintln!("    ‖B‖₂          = {b_norm:.6e}");
    eprintln!("    ‖A‖−‖B‖/‖A‖   = {global_rel:.3e}");

    // Bit-equal is the ideal; we expect either exact equality or
    // floating-point noise from encoder-boundary scheduling
    // differences. 1e-5 rel is the same ceiling the M2.6+ GPU-CPU
    // tests adopt for "same physics" (CLAUDE.md 5-layer test).
    let ok = max_rel < 1e-5;
    if !ok {
        eprintln!(
            "    !! relative {max_rel:.3e} > 1e-5 — profile_passes_fft samples a DIFFERENT pipeline"
        );
    } else {
        eprintln!("    OK: relative within 1e-5 → same physics confirmed");
    }
    ok
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking_with_timestamps(instance);
    let info = ctx.adapter.get_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);
    eprintln!("timestamp period: {} ns/tick", ctx.queue.get_timestamp_period());

    let ok = sanity_check_profile_matches_step(&ctx);
    assert!(
        ok,
        "profile_passes_fft does not reproduce step() output — breakdown numbers are not \
         describing the production pipeline. Investigate before trusting per-pass percentages."
    );

    // 256 for reference (Stage 1 target), then 512 (Stage 2 / final).
    breakdown(&ctx, 256, 3);
    breakdown(&ctx, 512, 3);
}
