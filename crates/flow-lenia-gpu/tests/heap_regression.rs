//! M6.A.8 — long-horizon heap leak regression.
//!
//! Builds a `GpuStepPipeline` and runs 10 000 steps, comparing the
//! `peak_alloc::PeakAlloc` current-allocation reading before and
//! after the loop, plus a mid-loop peak sample to distinguish a
//! one-time transient (peak settles by mid-loop) from a slow leak
//! that pushes peak up across the loop. Catches the failure mode
//! where an M6.C refactor introduces a per-step `Vec`, `String`,
//! or `Box` that accumulates across the simulator's steady-state
//! loop.
//!
//! ## Scope: CPU heap only (Rust allocator)
//!
//! `peak_alloc` wraps `std::alloc::System` and so reports only
//! allocations Rust knows about — `Vec`, `Box`, `String`, owned
//! `Array3`, etc. **GPU memory** (wgpu buffers, textures, command
//! encoders) is managed by the OS-level Metal driver outside the
//! Rust allocator and does not show up here. M6.A.8 explicitly
//! leaves GPU-side leak detection out of scope (CLAUDE.md "Scope
//! 制約"); manual macOS Activity Monitor inspection is the M6.A.9
//! recipe for the GPU side.
//!
//! ## Grid choice: single grid is sufficient *by construction*
//!
//! Steady-state per-step Rust allocation in `GpuStepPipeline::step`
//! is the same shape regardless of grid: one command-encoder build,
//! six `record(...)` calls (no per-step bind-group reallocation —
//! all bind groups are pre-built in `GpuStepPipeline::new`, see
//! M6.0 §3 audit), one queue submit, one ping-index swap. None of
//! these per-step allocations are grid-dependent. The wgpu buffers
//! that *do* scale with grid² are constructed once in
//! `GpuStepPipeline::new` (outside the timed loop) and freed when
//! the pipeline drops (outside the post-loop reading). So a leak
//! that survives the per-step loop will surface at 64×64 just as
//! it would at 256×256. Running multiple grids would only catch a
//! pathology where the leak is gated on grid-dependent control
//! flow — not currently a credible concern given how uniform the
//! step path is.
//!
//! ## Methodology
//!
//! 1. Construct the pipeline (one-time GPU buffer + bind-group
//!    allocations, ~ 600 KB of Rust handles at 64×64 / C=3).
//! 2. Warmup 100 steps so first-dispatch shader-compile arenas
//!    settle and the kernel-buffers upload path (which fires in
//!    step 1, well before any baseline reading) is fully drained.
//! 3. Drain the wgpu queue with `device.poll(Wait)` so any
//!    deferred drop-on-fence Rust work completes.
//! 4. Read `current_usage_as_kb()` once — this is the *baseline*.
//! 5. Run 5 000 steps, drain queue, take the *mid* peak reading.
//! 6. Run another 5 000 steps, drain queue, take the *post*
//!    readings (current + peak).
//! 7. Assert:
//!    - `post_current - baseline_current < CURRENT_DELTA_LIMIT_KB`
//!      (signed: only positive growth is a leak).
//!    - `(post_peak - baseline_peak) - (mid_peak - baseline_peak)
//!      < PEAK_DRIFT_LIMIT_KB` — i.e. peak should not climb
//!      meaningfully *between* mid-loop and post-loop. A single
//!      one-time transient settles by the mid sample.
//!
//! ## Wall-clock note
//!
//! 10 000 steps × ~ 7 ms/step at 64×64 / C=3 on a cold-boot M1
//! gives a naive lower bound of 70 s plus warmup + two drains. The
//! M6.A.8 commit measured **197 s** because the host was thermally
//! degraded after the M6.A.6 / A.7 perf sweeps (same dynamic as
//! Section 9: warm-state run rates land at 0.73-0.93× of cold).
//! No independent cold-boot measurement was taken; treat 80 s as
//! an extrapolation from Section 1's `16.29 ms/step` cold figure
//! × 10 000 steps, not as a measurement.

mod common;

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};
use flow_lenia_gpu::GpuStepPipeline;
use peak_alloc::PeakAlloc;

// `#[global_allocator]` is process-wide. For an integration-test
// binary this means every test in `heap_regression.rs` runs through
// `PeakAlloc`; other integration-test binaries are unaffected
// because each compiles to its own binary. Production crates
// (`flow-lenia-app`, `flow-lenia-web`) never transitively pull
// `peak_alloc` — it's a dev-dependency.
#[global_allocator]
static GLOBAL: PeakAlloc = PeakAlloc;

const SEED: u64 = 42;
const NUM_KERNELS: u32 = 10;
const N_STEPS_TOTAL: u32 = 10_000;
const N_STEPS_HALF: u32 = N_STEPS_TOTAL / 2;
/// 500 KB. Sized from the M6.A.8 measured transient (post − baseline
/// = +270.70 KB on the commit-time run): ~ 2 × observed gives a
/// signal-to-noise band that catches anything sustained while
/// absorbing the run-to-run jitter on the wgpu drop/poll path.
/// Original M6.A.8 used 1 MB; review tightened it to 500 KB to
/// halve the detection floor (a 50 B/step leak now visibly breaches
/// the band at 10 K steps instead of being absorbed).
const CURRENT_DELTA_LIMIT_KB: f32 = 500.0;
/// 256 KB. The peak should not *climb* between the mid and post
/// readings — a one-time transient (~ +1.1 MB observed at the M6.A.8
/// commit) settles inside the first half. If peak keeps growing,
/// that's a slow leak that only shows up in peak (something is
/// repeatedly allocating a slightly larger working set).
const PEAK_DRIFT_LIMIT_KB: f32 = 256.0;

fn cfg() -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: 64,
        grid_height: 64,
        channels: 3,
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

fn drain(ctx: &flow_lenia_gpu::GpuContext, label: &str) {
    ctx.device
        .poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })
        .unwrap_or_else(|e| panic!("device.poll(Wait) failed at {label}: {e:?}"));
}

#[test]
#[ignore = "M6.A.8 heap leak regression (~80-200 s, 10 000 step); --include-ignored to run"]
fn heap_no_leak_10k_step_g64_c3() {
    let (ctx, guard) = common::test_ctx();
    let cfg = cfg();

    let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

    pipeline.run_steps(&ctx, 100);
    drain(&ctx, "warmup");

    let baseline_current_kb = GLOBAL.current_usage_as_kb();
    let baseline_peak_kb = GLOBAL.peak_usage_as_kb();

    let started = std::time::Instant::now();

    for _ in 0..N_STEPS_HALF {
        pipeline.step(&ctx);
    }
    drain(&ctx, "mid");
    let mid_current_kb = GLOBAL.current_usage_as_kb();
    let mid_peak_kb = GLOBAL.peak_usage_as_kb();

    for _ in 0..N_STEPS_HALF {
        pipeline.step(&ctx);
    }
    drain(&ctx, "post");

    let elapsed_s = started.elapsed().as_secs_f64();
    let post_current_kb = GLOBAL.current_usage_as_kb();
    let post_peak_kb = GLOBAL.peak_usage_as_kb();

    let delta_current_kb = post_current_kb - baseline_current_kb;
    let mid_delta_peak_kb = mid_peak_kb - baseline_peak_kb;
    let post_delta_peak_kb = post_peak_kb - baseline_peak_kb;
    let peak_drift_kb = post_delta_peak_kb - mid_delta_peak_kb;

    eprintln!(
        "\n[M6.A.8 heap regression — 64×64 C=3, {N_STEPS_TOTAL} step, {elapsed_s:.1}s]"
    );
    eprintln!("  baseline current  : {baseline_current_kb:>9.2} KB");
    eprintln!("  mid      current  : {mid_current_kb:>9.2} KB");
    eprintln!("  post     current  : {post_current_kb:>9.2} KB");
    eprintln!(
        "  Δ current         : {delta_current_kb:>+9.2} KB \
         (limit +{CURRENT_DELTA_LIMIT_KB:.0} KB, signed leak detector)"
    );
    eprintln!("  baseline peak     : {baseline_peak_kb:>9.2} KB");
    eprintln!("  mid      peak     : {mid_peak_kb:>9.2} KB (Δ {mid_delta_peak_kb:>+8.2} KB)");
    eprintln!("  post     peak     : {post_peak_kb:>9.2} KB (Δ {post_delta_peak_kb:>+8.2} KB)");
    eprintln!(
        "  peak drift mid→post: {peak_drift_kb:>+8.2} KB \
         (limit ±{PEAK_DRIFT_LIMIT_KB:.0} KB)"
    );
    eprintln!();

    // Signed comparison — leaks are positive growth. A current that
    // *drops* by 1 MB+ doesn't happen in practice and isn't a leak;
    // do not use `.abs()` here.
    assert!(
        delta_current_kb < CURRENT_DELTA_LIMIT_KB,
        "heap current grew by {delta_current_kb:+.2} KB across {N_STEPS_TOTAL} steps, \
         exceeded +{CURRENT_DELTA_LIMIT_KB:.0} KB tolerance. \
         An M6.C refactor likely introduced a per-step Rust allocation \
         (Vec / Box / String) that accumulates across the steady-state loop. \
         Run with --nocapture to see the mid + post breakdown above; the \
         mid sample bisects which half of the loop is leaking."
    );

    // The transient-vs-leak discriminator: peak should not *climb*
    // between mid and post. If it does, the leak is in something
    // whose working set grows monotonically.
    //
    // Peak is high-water-mark and therefore monotone non-decreasing,
    // so `peak_drift_kb` is always ≥ 0 in practice; the `.abs()` is
    // defensive against future PeakAlloc API guarantees we don't
    // want to depend on.
    assert!(
        peak_drift_kb.abs() < PEAK_DRIFT_LIMIT_KB,
        "peak drifted by {peak_drift_kb:+.2} KB between mid and post halves, \
         exceeded ±{PEAK_DRIFT_LIMIT_KB:.0} KB tolerance. The mid-loop \
         peak should have already captured any one-time transient — a \
         post-loop peak that climbed further means the working set is \
         growing across the loop, not a one-shot allocation."
    );

    // A.7 validation pattern: if FLOW_LENIA_VALIDATE=1 was set, any
    // wgpu validation error during the 10 K-step loop is also a
    // failure, surfaced through the test_ctx guard.
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}
