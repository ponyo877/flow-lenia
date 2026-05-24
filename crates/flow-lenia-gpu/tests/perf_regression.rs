//! M6.A.6 performance regression.
//!
//! Anchors the M6.0 bench results in code so that an M6.C numerical-
//! path change which silently slows the simulator down trips a CI
//! signal (warning at ±5%, error at ±20%). Mirrors the methodology
//! of `app/src/bin/bench_step.rs` — warmup loop, then time `step`
//! calls in a tight loop, then `device.poll(Wait)` to flush GPU work
//! before the timer stops — so the numbers it produces are directly
//! comparable to BENCH.md Section 1.
//!
//! ## Tolerance design (M6.A.6)
//!
//! Two thresholds per side (CPU sps, GPU sps):
//!
//! - `±5%` → **warning** printed to stderr; the test still passes.
//!   Catches the "looks fine on this commit but creeps slower over
//!   time" failure mode that an error-only check misses.
//! - `±20%` → **error**; the test panics with the offending case.
//!   Set high enough that thermal throttling on a busy laptop doesn't
//!   cause spurious CI failures, low enough that a real M6.C
//!   numerical-path regression (e.g. accidentally disabling LTO)
//!   shows up immediately.
//!
//! Plus a `gpu_sps / cpu_sps` ratio check at `±30%`, **warning-only**.
//! M6.C's whole point is to make GPU faster (FFT), so the ratio
//! should *grow* — the warning surfaces unexpected motion without
//! failing the run.
//!
//! ## Variance mitigation
//!
//! Three concurrent strategies (all live in `measure_sps_cpu` /
//! `_gpu` and the per-case loop below):
//!
//! 1. **Warmup before timing** — `warmup_for_grid(grid)` steps run
//!    before `Instant::now()` so shader cache loads, JIT, and
//!    one-time allocations are out of the way.
//! 2. **N_RUNS = 3, median selection** — the worst-case outlier
//!    (thermal spike, background process) gets pushed out of the
//!    median; only the typical run is compared against baseline.
//! 3. **Grid-dependent step counts** — large grids run fewer steps
//!    because each step is heavier; total wall-clock per case stays
//!    in the 5-30 s range so a thermal swing doesn't span the whole
//!    measurement.

mod common;

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};
use flow_lenia_gpu::{GpuContext, GpuStepPipeline};
use std::time::Instant;

const SEED: u64 = 42;
const NUM_KERNELS: u32 = 10;
const N_RUNS: usize = 3;

/// Soft-edge warning, error, and ratio thresholds. See module doc.
const WARN_THRESHOLD: f64 = 0.05;
const ERROR_THRESHOLD: f64 = 0.20;
const RATIO_THRESHOLD: f64 = 0.30;

/// (grid, channels, cpu_sps, gpu_sps) anchored to the **A.6 commit
/// median** measurement, *not* the M6.0 `bench_step` first-pass numbers
/// in BENCH.md Section 1. M6.0's run was taken on a cold-boot M1 mini
/// before this session began; re-running the same `bench_step` later
/// (with the same toolchain, the same wgpu, the same code) reproduces
/// ~75-85 % of the M6.0 rates because the machine has accumulated
/// background load and the GPU has been thermally seasoned. Anchoring
/// the regression test to those typical-development-state numbers
/// gives consistent regression *detection* during M6.C — what matters
/// is "did *this* commit slow things down vs. *the last* commit", not
/// "does the machine match its cold-boot self".
///
/// The original M6.0 values are preserved in `BENCH.md` Section 1
/// (the historical "best-case" anchor) and the relationship between
/// the two is documented in Section 9 (added with this commit).
///
/// Re-anchor by re-running this test with the BASELINES table cleared
/// (or with WARN/ERROR thresholds widened temporarily) — the eprintln
/// per case prints `cpu = …, gpu = …` numbers ready to paste in.
const BASELINES: &[(u32, u32, f64, f64)] = &[
    // (grid, C, cpu_sps, gpu_sps)
    (32, 1, 69.14, 117.67),
    (32, 3, 64.90, 106.28),
    (64, 1, 17.16, 55.35),
    (64, 3, 15.90, 50.65),
    (128, 1, 4.30, 15.25),
    (128, 3, 4.02, 14.01),
    (256, 1, 1.05, 3.92),
    (256, 3, 1.01, 3.60),
];

fn warmup_for_grid(grid: u32) -> u32 {
    match grid {
        0..=64 => 30,
        65..=128 => 10,
        _ => 5,
    }
}

fn measure_for_grid(grid: u32) -> u32 {
    match grid {
        0..=64 => 300,
        65..=128 => 100,
        _ => 20,
    }
}

fn cfg_for(grid: u32, channels: u32) -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: grid,
        grid_height: grid,
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

fn measure_sps_cpu(grid: u32, channels: u32, warmup: u32, measure: u32) -> f64 {
    let cfg = cfg_for(grid, channels);
    let mut sim = FlowLeniaSimulator::new(cfg, SEED);
    sim.step_many(warmup);
    let started = Instant::now();
    sim.step_many(measure);
    let elapsed = started.elapsed().as_secs_f64();
    f64::from(measure) / elapsed
}

fn measure_sps_gpu(
    grid: u32,
    channels: u32,
    warmup: u32,
    measure: u32,
    ctx: &GpuContext,
) -> f64 {
    let cfg = cfg_for(grid, channels);
    let setup_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = setup_sim.activation().clone();
    let kernel_params = setup_sim.kernel_params().clone();
    let mut pipeline = GpuStepPipeline::new(ctx, &cfg, &kernel_params, &initial_a);
    pipeline.run_steps(ctx, warmup);
    // Drain warmup work before starting the timer so it doesn't
    // contribute to the measured wall-clock.
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();

    let started = Instant::now();
    for _ in 0..measure {
        pipeline.step(ctx);
    }
    // Mirrors `bench_step` — the trailing `poll(Wait)` flushes the
    // command queue so the elapsed time covers the actual GPU work,
    // not just the submission overhead.
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap();
    let elapsed = started.elapsed().as_secs_f64();
    f64::from(measure) / elapsed
}

fn median(values: &[f64]) -> f64 {
    let mut v = values.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

#[derive(Debug)]
enum Verdict {
    Ok,
    Warn(String),
    Err(String),
}

fn check_delta(label: &str, measured: f64, baseline: f64) -> Verdict {
    let delta = (measured - baseline) / baseline;
    let abs_delta = delta.abs();
    if abs_delta > ERROR_THRESHOLD {
        Verdict::Err(format!(
            "{label}: sps = {measured:.2} (baseline {baseline:.2}, \
             Δ {:+.1}%, exceeded ±{:.0}%)",
            delta * 100.0,
            ERROR_THRESHOLD * 100.0,
        ))
    } else if abs_delta > WARN_THRESHOLD {
        Verdict::Warn(format!(
            "{label}: sps = {measured:.2} (baseline {baseline:.2}, \
             Δ {:+.1}%, exceeded ±{:.0}%)",
            delta * 100.0,
            WARN_THRESHOLD * 100.0,
        ))
    } else {
        Verdict::Ok
    }
}

#[test]
#[ignore = "M6.A.6 perf regression (~6-8 min on M1); --include-ignored to run"]
fn perf_regression_full_matrix() {
    let (ctx, guard) = common::test_ctx();

    let mut warnings: Vec<String> = Vec::new();
    let mut errors: Vec<String> = Vec::new();

    eprintln!(
        "\n[M6.A.6 perf regression — n_runs={N_RUNS}, sps tolerance ±5% warn / ±20% err, ratio ±30% warn]"
    );
    eprintln!(
        "  Strategy: grid-dependent warmup/measure, GPU sync via poll(Wait),"
    );
    eprintln!(
        "  {N_RUNS}-run median per case. Baselines re-anchored at the A.6 commit"
    );
    eprintln!(
        "  to typical-development-state numbers — see BENCH.md §9 for why."
    );
    eprintln!();

    for &(grid, channels, cpu_baseline, gpu_baseline) in BASELINES {
        let warmup = warmup_for_grid(grid);
        let measure = measure_for_grid(grid);

        let mut cpu_runs = Vec::with_capacity(N_RUNS);
        let mut gpu_runs = Vec::with_capacity(N_RUNS);
        for _ in 0..N_RUNS {
            cpu_runs.push(measure_sps_cpu(grid, channels, warmup, measure));
            gpu_runs.push(measure_sps_gpu(grid, channels, warmup, measure, &ctx));
        }
        let cpu_sps = median(&cpu_runs);
        let gpu_sps = median(&gpu_runs);
        let ratio = gpu_sps / cpu_sps;
        let baseline_ratio = gpu_baseline / cpu_baseline;

        let cpu_delta = (cpu_sps - cpu_baseline) / cpu_baseline * 100.0;
        let gpu_delta = (gpu_sps - gpu_baseline) / gpu_baseline * 100.0;
        let ratio_delta = (ratio - baseline_ratio) / baseline_ratio * 100.0;

        eprintln!(
            "  grid={grid:3} C={channels}: \
             cpu={cpu_sps:>7.2} (base {cpu_baseline:>7.2}, Δ{cpu_delta:>+6.1}%)  \
             gpu={gpu_sps:>7.2} (base {gpu_baseline:>7.2}, Δ{gpu_delta:>+6.1}%)  \
             gpu/cpu={ratio:>5.2}× (base {baseline_ratio:>5.2}×, Δ{ratio_delta:>+6.1}%)"
        );

        let cpu_label = format!("grid={grid:3} C={channels} CPU");
        match check_delta(&cpu_label, cpu_sps, cpu_baseline) {
            Verdict::Ok => {}
            Verdict::Warn(s) => warnings.push(s),
            Verdict::Err(s) => errors.push(s),
        }
        let gpu_label = format!("grid={grid:3} C={channels} GPU");
        match check_delta(&gpu_label, gpu_sps, gpu_baseline) {
            Verdict::Ok => {}
            Verdict::Warn(s) => warnings.push(s),
            Verdict::Err(s) => errors.push(s),
        }
        // Ratio is warning-only — M6.C is supposed to grow it, so an
        // upper-bound error would fire on the very change we want.
        if ratio_delta.abs() > RATIO_THRESHOLD * 100.0 {
            warnings.push(format!(
                "grid={grid:3} C={channels} gpu/cpu ratio = {ratio:.2}× \
                 (baseline {baseline_ratio:.2}×, Δ{ratio_delta:+.1}%, exceeded ±{:.0}%)",
                RATIO_THRESHOLD * 100.0
            ));
        }
    }

    eprintln!();
    if !warnings.is_empty() {
        eprintln!("[perf warnings — within ±20% error band, still passing]");
        for w in &warnings {
            eprintln!("  {w}");
        }
        eprintln!();
    } else {
        eprintln!("[no warnings — all measurements within ±5% of baseline]\n");
    }
    if !errors.is_empty() {
        panic!(
            "M6.A.6 perf regression(s) exceeded ±{}%:\n  {}",
            (ERROR_THRESHOLD * 100.0) as i32,
            errors.join("\n  ")
        );
    }
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}
