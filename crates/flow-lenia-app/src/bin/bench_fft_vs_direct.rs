#![deny(warnings)]
//! M6.C-1-4-b paired-run measurement: FFT mode vs Direct mode
//! end-to-end step throughput, **early-exit gate** for the M6.C-1
//! milestone (post/pre ratio < 2× ⇒ Phase 3 改訂条件 2 で Claude Web
//! 送信、≥ 2× ⇒ 自走 commit + push、C-1-5 へ).
//!
//! ## Measurement protocol (CLAUDE.md §測定プロトコル準拠)
//!
//! - **Paired run**: D F D F D F (direct then fft, repeated 3×).
//!   Thermal interleave keeps both modes within the same machine
//!   state envelope; the ratio absorbs slow thermal drift that
//!   would confound a sequential "all D, then all F" measurement.
//! - **Multiple runs**: N=3 trials per mode, **median** reported.
//!   Mean would be skewed by a single thermal spike; median is the
//!   M6.A.6 / perf_regression convention.
//! - **Quiesced state**: the caller is responsible for stopping
//!   trunk serve / cargo build / browser windows before running.
//!   We do NOT enforce this — running on a noisy host just inflates
//!   the standard deviation, which the median absorbs as best it
//!   can but not perfectly.
//! - **Honest framing**: the printed ratio is observed-at-this-run,
//!   not a long-term performance guarantee. If the gate result sits
//!   inside the ±10 % thermal noise band CLAUDE.md §9 documents,
//!   that fact is printed too.
//!
//! ## Configuration (C-1-4-b scope)
//!
//! N=64, C=1, K=10, Torus border, 100 steps per trial. This is the
//! only configuration the FFT path supports today (C=1 + grid ∈
//! {64, 256} per `ConvolveMode::Fft` assertions). N=256 measurement
//! is C-1-5 follow-up if this gate passes.
//!
//! ## Gate condition
//!
//! ```text
//! ratio = direct_ms_per_step / fft_ms_per_step
//!       = fft_sps / direct_sps
//! ```
//! "FFT is X× faster than Direct" convention. `ratio ≥ 2×` means
//! FFT is at least twice as fast end-to-end.
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin bench_fft_vs_direct
//! ```
//!
//! Exit code:
//! - `0`: ratio ≥ 2× (gate pass, auto-commit allowed)
//! - `1`: ratio < 2× (early-exit gate, Claude Web notification)
//! - `2`: build/setup error

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{pipeline::ConvolveMode, GpuContext, GpuStepPipeline};
use std::process::ExitCode;
use std::time::Instant;

const SEED: u64 = 1729;
const GRID: u32 = 64;
const NUM_KERNELS: u32 = 10;
const N_STEPS: u32 = 100;
const N_TRIALS: usize = 3;
const GATE_RATIO: f64 = 2.0;

fn cfg_for_channels(channels: u32) -> FlowLeniaConfig {
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

fn measure_trial(ctx: &GpuContext, channels: u32, mode: ConvolveMode) -> f64 {
    let cfg = cfg_for_channels(channels);
    let cpu_init = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_init.activation().clone();
    let kernel_params = cpu_init.kernel_params().clone();
    let mut pipeline =
        GpuStepPipeline::new_with_mode(ctx, &cfg, &kernel_params, &initial_a, mode);

    // Warmup: 20 steps to absorb shader-cache cold start + initial
    // dispatch ladder. Matches bench_step.rs warmup discipline.
    pipeline.run_steps(ctx, 20);

    let started = Instant::now();
    pipeline.run_steps(ctx, N_STEPS);
    let elapsed = started.elapsed();
    elapsed.as_secs_f64() / f64::from(N_STEPS) * 1000.0 // ms / step
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Run a paired D F D F D F measurement for one channel count and
/// return (ratio, gate_pass).
fn measure_channels(ctx: &GpuContext, channels: u32) -> (f64, bool) {
    let mut direct_samples: Vec<f64> = Vec::with_capacity(N_TRIALS);
    let mut fft_samples: Vec<f64> = Vec::with_capacity(N_TRIALS);

    eprintln!("\n=== C={channels} paired-run D F D F D F ===");
    for trial in 0..N_TRIALS {
        let d_ms = measure_trial(ctx, channels, ConvolveMode::Direct);
        direct_samples.push(d_ms);
        let f_ms = measure_trial(ctx, channels, ConvolveMode::Fft);
        fft_samples.push(f_ms);
        eprintln!(
            "  trial {trial}: direct={d_ms:.3} ms/step  fft={f_ms:.3} ms/step  \
             ratio={ratio:.3}×",
            ratio = d_ms / f_ms
        );
    }

    let mut direct_sorted = direct_samples.clone();
    direct_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mut fft_sorted = fft_samples.clone();
    fft_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let direct_min = direct_sorted[0];
    let direct_max = *direct_sorted.last().unwrap();
    let fft_min = fft_sorted[0];
    let fft_max = *fft_sorted.last().unwrap();
    let direct_median = median(&mut direct_samples.clone());
    let fft_median = median(&mut fft_samples.clone());
    let ratio = direct_median / fft_median;

    eprintln!(
        "  direct: median {direct_median:.3} ms/step  (min {direct_min:.3}, max {direct_max:.3}, \
         range {range:.3} ms = {pct:.2}% of median)",
        range = direct_max - direct_min,
        pct = 100.0 * (direct_max - direct_min) / direct_median
    );
    eprintln!(
        "  fft   : median {fft_median:.3} ms/step  (min {fft_min:.3}, max {fft_max:.3}, \
         range {range:.3} ms = {pct:.2}% of median)",
        range = fft_max - fft_min,
        pct = 100.0 * (fft_max - fft_min) / fft_median
    );
    eprintln!(
        "  sps  : direct {direct_sps:.1}  fft {fft_sps:.1}",
        direct_sps = 1000.0 / direct_median,
        fft_sps = 1000.0 / fft_median
    );
    eprintln!("  ratio (FFT vs Direct speedup) = {ratio:.3}×");
    if (ratio - 1.0).abs() < 0.10 {
        eprintln!("  ⚠ ratio within ±10% of 1× — thermal noise band");
    }
    (ratio, ratio >= GATE_RATIO)
}

fn main() -> ExitCode {
    eprintln!(
        "M6.C-1-5-b bench_fft_vs_direct (C=1 + C=3 multi-channel)\n\
         config: N={GRID}, K={NUM_KERNELS}, Torus, {N_STEPS} steps × {N_TRIALS} trials\n\
         gate: FFT must be ≥ {GATE_RATIO:.1}× faster than Direct (per channel count)"
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    // M6.C-1-5-b: measure both C=1 (existing C-1-4-b path) and C=3
    // (Flow-Lenia default + new multi-channel FFT support from
    // C-1-5-a). The early-exit gate checks each independently —
    // both must pass ≥ 2× for auto-commit.
    let (ratio_c1, pass_c1) = measure_channels(&ctx, 1);
    let (ratio_c3, pass_c3) = measure_channels(&ctx, 3);

    eprintln!("\n=== Summary ===");
    eprintln!("C=1 ratio = {ratio_c1:.3}×  gate {}", if pass_c1 { "PASS" } else { "FAIL" });
    eprintln!("C=3 ratio = {ratio_c3:.3}×  gate {}", if pass_c3 { "PASS" } else { "FAIL" });

    if pass_c1 && pass_c3 {
        eprintln!(
            "✓ GATE PASS (both channel counts ≥ {GATE_RATIO:.1}×) — auto-commit allowed"
        );
        ExitCode::from(0)
    } else {
        eprintln!(
            "✗ EARLY EXIT GATE (at least one channel count < {GATE_RATIO:.1}×) — \
             Phase 3 改訂条件 2 trigger, Claude Web notification required \
             (do NOT auto-commit)"
        );
        ExitCode::from(1)
    }
}
