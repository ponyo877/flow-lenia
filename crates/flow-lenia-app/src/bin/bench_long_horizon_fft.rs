#![deny(warnings)]
//! M6.C-1-6 Phase A — long-horizon FFT-vs-Direct stability measurement.
//!
//! C-1-4-b's `gpu_pipeline_fft_mode_matches_direct_n64_c1_short`
//! measured a 5-step direct-vs-fft `max_rel ≈ 1.7e-4` at C=1 and
//! C-1-5-a's C=3 sibling measured `≈ 2.1e-4`. C-1-4-b adversarial-
//! reviewer S2 flagged the implied per-step amplification factor
//! (~1.6×/step at 5-step horizon) and demanded a long-horizon
//! projection before the FFT path can be considered production-ready
//! for chaotic-dynamics simulations.
//!
//! This binary measures **horizon ∈ {10, 50, 100}** step direct-vs-fft
//! `max_rel` at N=64, C ∈ {1, 3}, K=10 Torus. It also runs the
//! **identical-kernels** controlled variant (C-1-4-b S2: K=10 copies of
//! one kernel vs K=10 scaling kernels) to separate "FFT injects a
//! per-step rounding error that downstream chaos amplifies" from
//! "the kernel-parameter scaling in the random test biases later
//! kernels higher".
//!
//! ## Scope (M6.C-1-6 scope-guardian)
//!
//! - **N=64 only**. N=256 long-horizon measurement requires
//!   cold-boot-quiesced sessions to be honest about thermal drift
//!   (CLAUDE.md §5); deferred to Stage 1 / C-2 if Ponyo877-san wants
//!   the actual number rather than the Amdahl extrapolation.
//! - **Diagnostic-only**: this binary's output feeds BENCH §14 (M6.C-1
//!   retro + Stage 1 input) and is not a regression test. Each
//!   trial is single-shot; CLAUDE.md "Multiple runs: N=3 median" is
//!   intentionally relaxed for the long-horizon path because we are
//!   characterising error growth across steps, not the per-step time
//!   distribution.
//! - **Honest framing**: the `max_rel` numbers are direct-vs-fft on
//!   the SAME GPU initial state, isolating FFT-injected error
//!   amplified by chaotic dynamics. They are NOT comparable to
//!   M6.A.4.5 BENCH §8 numbers (which are GPU-vs-CPU, a different
//!   per-step injection mechanism).
//!
//! ## Output
//!
//! Prints to stderr in a form ready to transcribe into BENCH §14:
//!
//! ```text
//! C=1 random-kernels:    horizon 10:max_rel=...  50:...  100:...
//! C=1 identical-kernels: horizon 10:max_rel=...  50:...  100:...
//! C=3 random-kernels:    ...
//! C=3 identical-kernels: ...
//! ```
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin bench_long_horizon_fft
//! ```

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    params::{KernelEntry, KernelParams},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{pipeline::ConvolveMode, GpuContext, GpuStepPipeline};

const SEED: u64 = 1729;
const GRID: u32 = 64;
const NUM_KERNELS: u32 = 10;
const HORIZONS: &[u32] = &[10, 50, 100];

fn cfg_for(channels: u32) -> FlowLeniaConfig {
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
        paper_strict: false,
        border: BorderMode::Torus,
        mix_rule: MixRule::Stochastic,
    }
}

/// Run `n_steps` of direct and fft pipelines from the same initial
/// state, return per-cell max_rel and max_abs.
fn run_pair(
    ctx: &GpuContext,
    cfg: &FlowLeniaConfig,
    kernel_params: &KernelParams,
    initial_a: &flow_lenia_core::state::ActivationField,
    n_steps: u32,
) -> (f32, f32) {
    let mut direct = GpuStepPipeline::new_with_mode(
        ctx,
        cfg,
        kernel_params,
        initial_a,
        ConvolveMode::Direct,
    );
    let mut fft = GpuStepPipeline::new_with_mode(
        ctx,
        cfg,
        kernel_params,
        initial_a,
        ConvolveMode::Fft,
    );
    direct.run_steps(ctx, n_steps);
    fft.run_steps(ctx, n_steps);
    let direct_a = direct.readback_activation(ctx);
    let fft_a = fft.readback_activation(ctx);

    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for ((y, x, c), &d) in direct_a.indexed_iter() {
        let f = fft_a[[y, x, c]];
        let abs_err = (d - f).abs();
        let rel_err = abs_err / d.abs().max(1e-6);
        max_abs = max_abs.max(abs_err);
        max_rel = max_rel.max(rel_err);
    }
    (max_abs, max_rel)
}

/// Replace `kernel_params.kernels` with K identical copies of the first
/// kernel — the C-1-4-b S2 controlled experiment. If the per-kernel
/// random-test monotonic `max_rel` growth (k=0: 1.5e-6 → k=9: 2.6e-6)
/// is due to **kernel-parameter scaling** (later kernels have bigger
/// support → bigger spectrum → bigger error in absolute terms), the
/// identical-kernels run should show a **flat** profile. If it shows
/// the same growth, the cause is **FFT injects per-step error that
/// downstream chaos amplifies** independent of kernel choice.
fn make_identical_kernel_params(params: &KernelParams) -> KernelParams {
    let first = params.kernels[0].clone();
    let identical: Vec<KernelEntry> = (0..params.kernels.len()).map(|_| first.clone()).collect();
    KernelParams {
        r_global: params.r_global,
        kernels: identical,
    }
}

fn measure_set(
    ctx: &GpuContext,
    channels: u32,
    use_identical: bool,
) -> Vec<(u32, f32, f32)> {
    let cfg = cfg_for(channels);
    let cpu_init = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_init.activation().clone();
    let base_params = cpu_init.kernel_params().clone();
    let kernel_params = if use_identical {
        make_identical_kernel_params(&base_params)
    } else {
        base_params
    };

    let mut results: Vec<(u32, f32, f32)> = Vec::with_capacity(HORIZONS.len());
    for &h in HORIZONS {
        let (max_abs, max_rel) = run_pair(ctx, &cfg, &kernel_params, &initial_a, h);
        results.push((h, max_abs, max_rel));
    }
    results
}

fn print_set(label: &str, results: &[(u32, f32, f32)]) {
    eprintln!("\n{label}");
    eprintln!("  horizon  max_abs        max_rel");
    for (h, max_abs, max_rel) in results {
        eprintln!("  {h:>7}  {max_abs:.4e}  {max_rel:.4e}");
    }
    // Per-step amplification factor estimate (geometric):
    // (max_rel[100] / max_rel[10]) ^ (1 / 90) — only meaningful when
    // the dynamics has not saturated. If max_rel[100] ≥ O(0.1), the
    // dynamics is in the saturated chaotic regime and the geometric
    // mean here is just the saturation level, not a true growth rate.
    let r10 = results[0].2;
    let r100 = results[2].2;
    if r10 > 0.0 && r100 > 0.0 {
        let factor = (r100 / r10).powf(1.0 / 90.0);
        let saturation_warning = if r100 > 0.1 { "  ⚠ may be saturated" } else { "" };
        eprintln!(
            "  per-step amplification (geom over horizon 10→100): {factor:.4}× {saturation_warning}"
        );
    }
}

fn main() {
    eprintln!(
        "M6.C-1-6 bench_long_horizon_fft\n\
         N={GRID}, K={NUM_KERNELS}, Torus, horizons {HORIZONS:?}\n\
         direct vs fft on same initial state (single-trial diagnostic)"
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    let c1_random = measure_set(&ctx, 1, false);
    let c1_identical = measure_set(&ctx, 1, true);
    let c3_random = measure_set(&ctx, 3, false);
    let c3_identical = measure_set(&ctx, 3, true);

    print_set("=== C=1 random-kernels ===", &c1_random);
    print_set("=== C=1 identical-kernels (C-1-4-b S2 controlled) ===", &c1_identical);
    print_set("=== C=3 random-kernels ===", &c3_random);
    print_set("=== C=3 identical-kernels (C-1-4-b S2 controlled) ===", &c3_identical);

    eprintln!(
        "\nNote: max_rel here is direct-vs-fft on same GPU initial state\n\
         (FFT-injected per-step error amplified by chaotic dynamics),\n\
         not GPU-vs-CPU as in BENCH §8."
    );
}
