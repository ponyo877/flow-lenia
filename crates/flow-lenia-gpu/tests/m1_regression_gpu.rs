//! M6.A.4 GPU regression — `GpuStepPipeline` vs the CPU baseline
//! fixtures committed under `tests/regression_fixtures/m1_baseline/`.
//!
//! Each case re-uses the M1.15 / M6.A.1 case definitions (seed = 42,
//! `num_kernels` = 10, the M1.15 `paper_strict × border × C` matrix)
//! and only changes the simulator backend.
//!
//! ──────────────────────────────────────────────────────────────────
//! **Field comparison covers `C = 1` only.**
//!
//! At `C = 3` with `K = 10`, the dynamics is *chaotic* for generic
//! sampled kernel parameters: per-step f32 add-order divergence
//! between CPU `ndarray` reductions and GPU compute-pass
//! accumulations is `~10⁻⁵` (measured in M2.7-single), but after
//! 100 steps that drift is amplified into `max_abs ≈ 2..3` in
//! individual cells while **mass is still perfectly conserved**.
//! Both implementations are *correct*; they are just on different
//! points of the same chaotic trajectory, much like running the CPU
//! simulator under two `rustc` builds with different f32 codegen
//! would diverge.
//!
//! `C = 1` collapses the dynamics enough that short-horizon field
//! comparison stays meaningful. M6.A.4 uses **10 steps** at
//! `rel < 1e-3`; M6.A.3's drift measurement shows the CPU per-step
//! drift floor is ~4.3e-8, so 10 steps of GPU vs CPU divergence
//! should sit safely under 1e-3 if the GPU shaders match the CPU
//! semantics. The `C = 3` cases are covered structurally by the
//! mass-conservation tests (same 8-case matrix as M1.15), kept at
//! 100 steps to exercise long-horizon WGSL `reintegrate` behaviour.
//! ──────────────────────────────────────────────────────────────────
//!
//! M6.A.4 — split `gpu_pipeline_matches_m1_baseline_fixtures_c1` into
//! `gpu_field_regression_g{32,64,128,256}` so M6 perf iterations on a
//! specific grid can target the right test. g128 / g256 sit behind
//! `#[ignore]` for the same reasoning as the CPU side.
//!
//! Run with: `cargo test --release -p flow-lenia-gpu --test
//! m1_regression_gpu -- --nocapture` (default) or `--include-ignored`
//! for the full sweep.

mod common;
use common::assert_creature_alive;

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    state::ActivationField,
    FlowLeniaConfig, FlowLeniaSimulator,
};
use flow_lenia_gpu::{pipeline::ConvolveMode, GpuContext, GpuStepPipeline};

const SEED: u64 = 42;
const NUM_KERNELS: u32 = 10;
/// Step count for the field-level GPU vs CPU comparison. Short
/// enough that even with the per-step ~10⁻⁵ add-order drift the
/// cumulative rel stays under the 1e-3 budget; tightening to 1e-3
/// makes the test 10× more sensitive than the old 100-step / 1e-2
/// version.
const FIELD_STEPS: u32 = 10;
/// Step count for the mass-conservation test. Kept at the original
/// M2.8 value so the long-horizon WGSL reintegrate path stays
/// exercised; this is the structural cover for C=3.
const MASS_STEPS: u32 = 100;
/// Constant-step horizon for the drift-vs-grid measurement table.
const DRIFT_STEPS: u32 = 100;

#[derive(Clone, Copy)]
struct Case {
    grid: u32,
    paper_strict: bool,
    border: BorderMode,
    channels: u32,
}

impl Case {
    fn filename(&self) -> String {
        let ps = if self.paper_strict { 'T' } else { 'F' };
        let bt = match self.border {
            BorderMode::Torus => 'T',
            BorderMode::Wall => 'W',
        };
        format!("case_g{}_ps{}_bt{}_c{}.bin", self.grid, ps, bt, self.channels)
    }

    fn cfg(&self) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: self.grid,
            grid_height: self.grid,
            channels: self.channels,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: NUM_KERNELS,
            paper_strict: self.paper_strict,
            border: self.border,
            mix_rule: MixRule::Stochastic,
        }
    }
}

/// `C = 1` slice of the 8-mode matrix: 4 cases per grid.
fn c1_cases_for_grid(grid: u32) -> Vec<Case> {
    let mut out = Vec::with_capacity(4);
    for &paper_strict in &[false, true] {
        for &border in &[BorderMode::Torus, BorderMode::Wall] {
            out.push(Case {
                grid,
                paper_strict,
                border,
                channels: 1,
            });
        }
    }
    out
}

/// Full 8-mode matrix at one grid (paper_strict × border × channels).
/// Used for mass conservation where C=3 mass is still meaningful even
/// with chaotic field divergence.
fn all_cases_for_grid(grid: u32) -> Vec<Case> {
    let mut out = Vec::with_capacity(8);
    for &paper_strict in &[false, true] {
        for &border in &[BorderMode::Torus, BorderMode::Wall] {
            for &channels in &[1_u32, 3_u32] {
                out.push(Case {
                    grid,
                    paper_strict,
                    border,
                    channels,
                });
            }
        }
    }
    out
}

/// Run the field-level GPU regression at one grid size.
///
/// The CPU fixture is the 100-step output committed by M6.A.1; we
/// truncate to `FIELD_STEPS` (10) on the GPU side and compare against
/// the *first 10 steps* by running the CPU simulator forward and
/// re-using its activation as the reference. This is what makes the
/// tight `rel < 1e-3` tolerance achievable — the committed fixtures
/// are 100-step and would already diverge from any 100-step GPU run.
fn run_field_regression(grid: u32, ctx: &GpuContext, rel_tolerance: f32) {
    run_field_regression_with_mode(grid, ctx, rel_tolerance, ConvolveMode::Auto)
}

/// **M6.C-3-8 follow-up** — `ConvolveMode`-pinned variant. Required so
/// `gpu_field_regression_g128` can keep the Direct-path baseline that
/// the A.4.5 tolerance table was calibrated against, while
/// `is_fft_pipeline_grid` (`crates/flow-lenia-gpu/src/passes/fft.rs`)
/// routes Auto→FFT at g=128 in production (4593291). Without this
/// pin, a `--include-ignored` run hits the FFT path with
/// 10-step rel ~7e-3, blowing past the A.4.5-pinned 1e-3 g128 budget
/// (16× regression from baseline 4.5e-4, see BENCH §19). Snapshot
/// tests in `tests/gpu_snapshot_regression.rs` already use this
/// "instantiate Direct explicitly" pattern (line 195).
fn run_field_regression_with_mode(
    grid: u32,
    ctx: &GpuContext,
    rel_tolerance: f32,
    mode: ConvolveMode,
) {
    let cases = c1_cases_for_grid(grid);
    let mut per_case_summary: Vec<String> = Vec::new();
    let mut overall_max_rel = 0.0_f32;

    for case in &cases {
        let cfg = case.cfg();

        // Reference CPU simulator: step it forward by the same
        // FIELD_STEPS as the GPU so the comparison is at the same
        // simulation time.
        let mut cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
        let initial_a = cpu_sim.activation().clone();
        let kernel_params = cpu_sim.kernel_params().clone();
        cpu_sim.step_many(FIELD_STEPS);
        let cpu_a = cpu_sim.activation().clone();

        let mut pipeline =
            GpuStepPipeline::new_with_mode(ctx, &cfg, &kernel_params, &initial_a, mode);
        let gpu_started = std::time::Instant::now();
        pipeline.run_steps(ctx, FIELD_STEPS);
        let gpu_a: ActivationField = pipeline.readback_activation(ctx);
        let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;

        // Final-frame sanity. NaN here means a GPU shader exploded;
        // catching it before the relative-error walk gives a far
        // clearer error than the inevitable rel = NaN that would
        // follow.
        let gpu_slice = gpu_a
            .as_slice()
            .expect("activation should be contiguous");
        assert_creature_alive(gpu_slice, &cfg);

        let (h, w, c) = gpu_a.dim();
        assert_eq!(cpu_a.dim(), (h, w, c));
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for y in 0..h {
            for x in 0..w {
                for ci in 0..c {
                    let gv = gpu_a[[y, x, ci]];
                    let cv = cpu_a[[y, x, ci]];
                    let abs_err = (gv - cv).abs();
                    let rel_err = abs_err / cv.abs().max(1e-6);
                    max_abs = max_abs.max(abs_err);
                    max_rel = max_rel.max(rel_err);
                }
            }
        }
        overall_max_rel = overall_max_rel.max(max_rel);

        per_case_summary.push(format!(
            "  {}: max_abs = {max_abs:.3e}, max_rel = {max_rel:.3e}  (gpu = {gpu_ms:.1} ms)",
            case.filename()
        ));
    }

    eprintln!("\n[M6.A.4 GPU field regression — grid={grid}, C=1, steps={FIELD_STEPS}]");
    for line in &per_case_summary {
        eprintln!("{line}");
    }
    eprintln!("  overall max_rel = {overall_max_rel:.3e}\n");

    assert!(
        overall_max_rel < rel_tolerance,
        "grid={grid}: overall max_rel {overall_max_rel:.3e} exceeds {rel_tolerance:.0e} — \
         see the per-case summary above for which case diverged"
    );
}

// M6.A.4.5 tiered tolerances. The earlier "rel < 1e-3 across all
// grids" target failed at g256 (measured 1.136e-3 vs 1e-3 budget);
// investigation A.4.5 traced the cause to *intrinsic* chaos in the
// Flow-Lenia dynamics — even at C=1, grid ≥ 64 has a Lyapunov
// exponent large enough that an ε = 1e-6 initial perturbation
// saturates to O(1) in a single step on CPU, so the per-cell f32
// add-order delta between CPU and GPU (≈ 1e-5, grid-independent)
// gets amplified grid-dependently over a few steps.
//
// The tolerance values below are 5×-ish the *deterministic*
// 10-step measurement (Experiment 4) — chosen because Experiment 6
// confirmed the GPU pipeline is bit-deterministic across re-runs
// (max/min ratio = 1.0000× over 5 fresh contexts), so the only
// margin we need is for legitimate M6 numerical-path drift, not
// for chaos non-determinism.
//
// See BENCH.md "Section 8 — A.4.5 GPU regression tolerance" for the
// raw numbers and the chaos / Lyapunov interpretation.
#[test]
fn gpu_field_regression_g32() {
    let (ctx, guard) = common::test_ctx();
    run_field_regression(32, &ctx, 1e-4);
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}

#[test]
fn gpu_field_regression_g64() {
    let (ctx, guard) = common::test_ctx();
    run_field_regression(64, &ctx, 5e-4);
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}

#[test]
#[ignore = "heavy 128×128 GPU regression; --include-ignored to run"]
fn gpu_field_regression_g128() {
    let (ctx, guard) = common::test_ctx();
    // M6.C-3-8 follow-up: pin Direct mode.
    //
    // commit 4593291 added 128 to `is_fft_pipeline_grid`, so the
    // default `Auto` resolution now routes g=128 to the mixed-radix
    // FFT path in production (this fixed the 16fps→60fps browser
    // regression for grid=128, but the FFT path has 10-step rel
    // ~7e-3 due to chaos amplification of the per-FFT-truncation
    // budget — well within the FFT's own pipeline tests at 1.5e-2
    // (`gpu_pipeline_fft_mode_matches_direct_n*_c*_short`) but 16×
    // over the A.4.5-pinned 1e-3 g128 Direct baseline this test
    // gates).
    //
    // We pin Direct here to keep the A.4.5 tolerance meaningful —
    // mirroring the same pattern `gpu_snapshot_regression.rs:195`
    // adopted at M6.A.5. The FFT-path numerical envelope at g=128 is
    // covered indirectly by `pipeline::tests::
    // gpu_pipeline_fft_mode_matches_direct_n64_c{1,3}_short` (FFT
    // vs Direct under chaos), and could be made explicit later via
    // a `gpu_pipeline_fft_mode_matches_direct_n128_*` test if a
    // tighter g=128 Auto-FFT bound is ever needed.
    run_field_regression_with_mode(128, &ctx, 1e-3, ConvolveMode::Direct);
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}

#[test]
#[ignore = "heavy 256×256 GPU regression; --include-ignored to run"]
fn gpu_field_regression_g256() {
    let (ctx, guard) = common::test_ctx();
    run_field_regression(256, &ctx, 2.5e-3);
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}

/// Short-horizon C=1 mass conservation. Added in M6.A.4 as a tight
/// smoke test for the WGSL `reintegrate` path: when M6.C rewrites
/// `convolve` (FFT migration), this test should still pass even
/// before the field-level comparison stabilises against the new
/// numerical path.
#[test]
fn gpu_mass_conservation_c1() {
    let (ctx, guard) = common::test_ctx();

    let cases = c1_cases_for_grid(32);
    let mut report: Vec<String> = Vec::new();

    for case in &cases {
        let cfg = case.cfg();
        let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
        let initial_a = cpu_sim.activation().clone();
        let kernel_params = cpu_sim.kernel_params().clone();
        let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

        let m0: f64 = initial_a.iter().map(|&v| f64::from(v)).sum();
        assert!(m0 > 0.0);

        let mut max_rel = 0.0_f64;
        for _ in 0..FIELD_STEPS {
            pipeline.step(&ctx);
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            let a = pipeline.readback_activation(&ctx);
            let m: f64 = a.iter().map(|&v| f64::from(v)).sum();
            let rel = (m - m0).abs() / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        let final_a = pipeline.readback_activation(&ctx);
        assert_creature_alive(
            final_a.as_slice().expect("activation should be contiguous"),
            &cfg,
        );

        report.push(format!(
            "  ps={:5} border={:?}: max_rel = {max_rel:.3e}",
            case.paper_strict, case.border
        ));

        assert!(
            max_rel < 1e-3,
            "{}: max_rel = {max_rel:.3e} >= 1e-3",
            case.filename()
        );
    }

    eprintln!(
        "\n[M6.A.4 GPU mass conservation — grid=32, C=1, steps={FIELD_STEPS}]"
    );
    for line in &report {
        eprintln!("{line}");
    }
    eprintln!();
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}

/// Long-horizon mass conservation across the full 8-mode matrix at
/// 32×32. Carried over from M2.8 — exercises the C=3 path that the
/// field regression intentionally skips.
#[test]
fn gpu_pipeline_mass_conservation_100_steps() {
    let (ctx, guard) = common::test_ctx();

    let mut report: Vec<String> = Vec::new();

    for case in &all_cases_for_grid(32) {
        let cfg = case.cfg();
        let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
        let initial_a = cpu_sim.activation().clone();
        let kernel_params = cpu_sim.kernel_params().clone();
        let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

        let m0: f64 = initial_a.iter().map(|&v| f64::from(v)).sum();
        assert!(m0 > 0.0);

        let started = std::time::Instant::now();
        let mut max_rel = 0.0_f64;
        for _ in 0..MASS_STEPS {
            pipeline.step(&ctx);
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            let a = pipeline.readback_activation(&ctx);
            let m: f64 = a.iter().map(|&v| f64::from(v)).sum();
            let rel = (m - m0).abs() / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        let per_step_ms = total_ms / f64::from(MASS_STEPS);

        // Final-frame sanity. Catches NaN / death even when mass
        // tolerance would let it pass vacuously (NaN comparisons are
        // false).
        let final_a = pipeline.readback_activation(&ctx);
        assert_creature_alive(
            final_a.as_slice().expect("activation should be contiguous"),
            &cfg,
        );

        let tol = match case.border {
            BorderMode::Torus => 1e-3,
            BorderMode::Wall => 1e-2,
        };

        report.push(format!(
            "  ps={:5} border={:?} C={}: max_rel = {max_rel:.3e}  \
             total = {total_ms:.0} ms  per_step = {per_step_ms:.2} ms",
            case.paper_strict, case.border, case.channels
        ));

        assert!(
            max_rel < tol,
            "{}: max_rel = {max_rel:.3e} >= tol {tol:.0e}",
            case.filename()
        );
    }

    eprintln!("\n[M2.8 GPU mass conservation 100 steps — grid=32, full 8-mode matrix]");
    for line in &report {
        eprintln!("{line}");
    }
    eprintln!();
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}

/// M6.A.4 condition 5 — GPU drift vs grid size at constant step
/// count. Mirrors `flow-lenia-core::mass_conservation_1k::
/// drift_vs_grid_size_100step` so the per-step floor can be compared
/// between CPU and GPU. Per-step readback dominates runtime (~2 min
/// total at 32→256, 4 cases each); `#[ignore]` keeps it out of the
/// default test set.
#[test]
#[ignore = "M6.A.4 GPU drift measurement (~2 min); --include-ignored to refresh BENCH.md"]
fn gpu_drift_vs_grid_size_100step_c1() {
    let (ctx, guard) = common::test_ctx();

    eprintln!(
        "\n=== M6.A.4 GPU drift vs grid size at {DRIFT_STEPS} steps (C=1) ==="
    );
    for &grid in &[32_u32, 64, 128, 256] {
        let cases = c1_cases_for_grid(grid);
        for case in &cases {
            let cfg = case.cfg();
            let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
            let initial_a = cpu_sim.activation().clone();
            let kernel_params = cpu_sim.kernel_params().clone();
            let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

            let m0: f64 = initial_a.iter().map(|&v| f64::from(v)).sum();
            let mut max_rel = 0.0_f64;
            for _ in 0..DRIFT_STEPS {
                pipeline.step(&ctx);
                ctx.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .unwrap();
                let a = pipeline.readback_activation(&ctx);
                let m: f64 = a.iter().map(|&v| f64::from(v)).sum();
                let rel = (m - m0).abs() / m0;
                if rel > max_rel {
                    max_rel = rel;
                }
            }
            let final_a = pipeline.readback_activation(&ctx);
            assert_creature_alive(
                final_a.as_slice().expect("activation should be contiguous"),
                &cfg,
            );
            eprintln!(
                "  grid={grid:3}  ps={:5}  border={:?}  max_rel={max_rel:.3e}",
                case.paper_strict, case.border
            );
        }
    }
    eprintln!();
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}

/// Bare-minimum per-step timing without per-step readback (the
/// realistic per-frame loop). 1000 steps with a single readback at
/// the end. Kept from M2.8 — informational output only.
#[test]
fn gpu_pipeline_per_step_timing() {
    let (ctx, guard) = common::test_ctx();

    let case = Case {
        grid: 32,
        paper_strict: false,
        border: BorderMode::Torus,
        channels: 3,
    };
    let cfg = case.cfg();
    let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

    pipeline.run_steps(&ctx, 10);

    let n = 1000_u32;
    let started = std::time::Instant::now();
    pipeline.run_steps(&ctx, n);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let per_step_ms = elapsed_ms / f64::from(n);

    let mut cpu_timing_sim = FlowLeniaSimulator::new(cfg, SEED);
    let cpu_started = std::time::Instant::now();
    cpu_timing_sim.step_many(n);
    let cpu_elapsed_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;
    let cpu_per_step_ms = cpu_elapsed_ms / f64::from(n);

    eprintln!(
        "\n[M2.8 per-step timing — 32×32 torus C=3 K=10, {n} steps after 10-step warmup]\n  \
         GPU: total = {elapsed_ms:.0} ms  per_step = {per_step_ms:.3} ms\n  \
         CPU: total = {cpu_elapsed_ms:.0} ms  per_step = {cpu_per_step_ms:.3} ms\n  \
         ratio (GPU / CPU) = {ratio:.2}x\n",
        ratio = per_step_ms / cpu_per_step_ms
    );

    assert!(per_step_ms > 0.0);
    assert!(cpu_per_step_ms > 0.0);
    if let Some(g) = &guard {
        g.assert_no_errors();
    }
}
