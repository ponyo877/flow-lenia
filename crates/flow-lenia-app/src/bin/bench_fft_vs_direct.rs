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
const CHANNELS: u32 = 1;
const NUM_KERNELS: u32 = 10;
const N_STEPS: u32 = 100;
const N_TRIALS: usize = 3;
const GATE_RATIO: f64 = 2.0;

fn cfg() -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: GRID,
        grid_height: GRID,
        channels: CHANNELS,
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

fn measure_trial(ctx: &GpuContext, mode: ConvolveMode) -> f64 {
    let cfg = cfg();
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

fn main() -> ExitCode {
    eprintln!(
        "M6.C-1-4-b bench_fft_vs_direct\n\
         config: N={GRID}, C={CHANNELS}, K={NUM_KERNELS}, Torus, \
         {N_STEPS} steps × {N_TRIALS} trials, paired D F D F D F\n\
         gate: FFT must be ≥ {GATE_RATIO:.1}× faster than Direct"
    );
    eprintln!("---");

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    let mut direct_samples: Vec<f64> = Vec::with_capacity(N_TRIALS);
    let mut fft_samples: Vec<f64> = Vec::with_capacity(N_TRIALS);

    // Paired interleave: D F D F D F.
    for trial in 0..N_TRIALS {
        let d_ms = measure_trial(&ctx, ConvolveMode::Direct);
        direct_samples.push(d_ms);
        let f_ms = measure_trial(&ctx, ConvolveMode::Fft);
        fft_samples.push(f_ms);
        eprintln!(
            "trial {trial}: direct={d_ms:.3} ms/step  fft={f_ms:.3} ms/step  \
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

    eprintln!("---");
    // Round 1 review S-4: print (min, median, max) so future re-runs
    // can spot envelope drift across trials.
    eprintln!(
        "direct: median {direct_median:.3} ms/step  (min {direct_min:.3}, max {direct_max:.3}, \
         range {range:.3} ms = {pct:.2}% of median)",
        range = direct_max - direct_min,
        pct = 100.0 * (direct_max - direct_min) / direct_median
    );
    eprintln!(
        "fft   : median {fft_median:.3} ms/step  (min {fft_min:.3}, max {fft_max:.3}, \
         range {range:.3} ms = {pct:.2}% of median)",
        range = fft_max - fft_min,
        pct = 100.0 * (fft_max - fft_min) / fft_median
    );
    eprintln!(
        "sps  : direct {direct_sps:.1}  fft {fft_sps:.1}",
        direct_sps = 1000.0 / direct_median,
        fft_sps = 1000.0 / fft_median
    );
    eprintln!("ratio (FFT vs Direct speedup) = {ratio:.3}×");

    // Honest framing: thermal noise band per BENCH §9 / CLAUDE.md §5
    // can be 7-27 %. If the result is within ±10 % of 1×, that's
    // "no measurable speedup at this measurement budget".
    if (ratio - 1.0).abs() < 0.10 {
        eprintln!("⚠ ratio is within ±10% of 1× — within thermal noise band, treat as 'no measurable difference'");
    }

    eprintln!("---");
    if ratio >= GATE_RATIO {
        eprintln!(
            "✓ GATE PASS: ratio {ratio:.3}× ≥ {GATE_RATIO:.1}× — auto-commit allowed"
        );
        ExitCode::from(0)
    } else {
        eprintln!(
            "✗ EARLY EXIT GATE: ratio {ratio:.3}× < {GATE_RATIO:.1}× — \
             Phase 3 改訂条件 2 trigger, Claude Web notification required \
             (do NOT auto-commit)"
        );
        ExitCode::from(1)
    }
}
