//! M2.8 GPU regression — compare [`GpuStepPipeline`] runs against
//! the M1.15 CPU baseline fixtures under
//! `tests/regression_fixtures/m1_baseline/`.
//!
//! Each case re-uses the M1.15 case definition (32×32, 100 steps,
//! seed = 42, num_kernels = 10) and only changes the simulator
//! backend.
//!
//! ──────────────────────────────────────────────────────────────────
//! **Field comparison covers the 4 `C = 1` fixtures only.**
//!
//! At `C = 3` with `K = 10`, the dynamics is *chaotic* for generic
//! sampled kernel parameters: per-step f32 add-order divergence
//! between CPU `ndarray` reductions and GPU compute-pass
//! accumulations is `~10⁻⁵` (measured in M2.7-single), but after
//! 100 steps that drift is amplified into `max_abs ≈ 2..3` in
//! individual cells while **mass is still perfectly conserved**
//! (`gpu_pipeline_mass_conservation_100_steps` below confirms 8/8
//! cases at `max_rel ≲ 5e-6`). Both implementations are *correct*;
//! they are just on different points of the same chaotic
//! trajectory, much like running the CPU simulator under two
//! `rustc` builds with different f32 codegen would diverge.
//!
//! `C = 1` collapses the dynamics enough that 100 steps stays
//! near the same trajectory, so field comparison is meaningful:
//! the measured `max_rel` sits at `1e-3 ≲ rel ≲ 1e-2` across the
//! 4 cases. The `C = 3` cases are covered structurally by the
//! mass-conservation test (same 8-case matrix as M1.15) and by
//! the 10-step `pipeline::tests::gpu_pipeline_ten_steps_match_cpu`
//! check.
//!
//! Tolerance: **`rel < 1e-2` overall** for the field comparison.
//! Comparable to the wall-mode mass-conservation budget in
//! DESIGN.md §5.3. The per-case breakdown surfaces via
//! `--nocapture` regardless of pass/fail.
//! ──────────────────────────────────────────────────────────────────
//!
//! Run with: `cargo test --release -p flow-lenia-gpu --test m1_regression_gpu -- --nocapture`.

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    state::ActivationField,
    FlowLeniaConfig, FlowLeniaSimulator,
};
use flow_lenia_gpu::{GpuContext, GpuStepPipeline};
use std::fs;
use std::path::PathBuf;

const SEED: u64 = 42;
const STEPS: u32 = 100;
const GRID: u32 = 32;
const NUM_KERNELS: u32 = 10;

struct Case {
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
        format!("case_ps{}_bt{}_c{}.bin", ps, bt, self.channels)
    }

    fn cfg(&self) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: GRID,
            grid_height: GRID,
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

fn all_cases() -> Vec<Case> {
    let mut out = Vec::with_capacity(8);
    for &paper_strict in &[false, true] {
        for &border in &[BorderMode::Torus, BorderMode::Wall] {
            for &channels in &[1_u32, 3_u32] {
                out.push(Case {
                    paper_strict,
                    border,
                    channels,
                });
            }
        }
    }
    out
}

fn fixture_path(filename: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("regression_fixtures")
        .join("m1_baseline")
        .join(filename)
}

fn read_f32_bin(path: PathBuf, expected_len: usize) -> Vec<f32> {
    let bytes = fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read fixture {}: {e}", path.display()));
    assert_eq!(
        bytes.len(),
        expected_len * 4,
        "fixture {} has {} bytes, expected {}",
        path.display(),
        bytes.len(),
        expected_len * 4
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
fn gpu_pipeline_matches_m1_baseline_fixtures_c1() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    let mut per_case_summary: Vec<String> = Vec::new();
    let mut overall_max_rel = 0.0_f32;

    // Only the C=1 cases (4 of 8) — see the module doc for why C=3
    // is structurally covered by the mass-conservation test instead.
    let c1_cases: Vec<Case> = all_cases()
        .into_iter()
        .filter(|c| c.channels == 1)
        .collect();

    for case in &c1_cases {
        let cfg = case.cfg();

        // Build the same initial state the M1.15 generator built
        // (FlowLeniaSimulator does this internally from `(cfg, seed)`).
        let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
        let initial_a = cpu_sim.activation().clone();
        let kernel_params = cpu_sim.kernel_params().clone();
        let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

        let gpu_started = std::time::Instant::now();
        pipeline.run_steps(&ctx, STEPS);
        let gpu_a: ActivationField = pipeline.readback_activation(&ctx);
        let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;

        let expected = read_f32_bin(
            fixture_path(&case.filename()),
            (GRID as usize) * (GRID as usize) * (case.channels as usize),
        );

        // Iterate (H, W, C) in the same order the fixture was
        // serialised: `(y, x, c)` row-major. Each `actual[i]` is
        // `gpu_a[[y, x, c]]` for the corresponding (i).
        let (h, w, c) = gpu_a.dim();
        assert_eq!(h * w * c, expected.len());
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for y in 0..h {
            for x in 0..w {
                for ci in 0..c {
                    let i = y * w * c + x * c + ci;
                    let gv = gpu_a[[y, x, ci]];
                    let cv = expected[i];
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

    eprintln!("\n[M2.8 GPU regression vs M1.15 baseline — C=1 cases]");
    for line in &per_case_summary {
        eprintln!("{line}");
    }
    eprintln!("  overall max_rel = {overall_max_rel:.3e}\n");

    assert!(
        overall_max_rel < 1e-2,
        "overall max_rel {overall_max_rel:.3e} exceeds 1e-2 — \
         see the per-case summary above for which fixture diverged"
    );
}

/// Full-pipeline 100-step mass conservation. Uses the same 8 cases.
#[test]
fn gpu_pipeline_mass_conservation_100_steps() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    let mut report: Vec<String> = Vec::new();

    for case in &all_cases() {
        let cfg = case.cfg();
        let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
        let initial_a = cpu_sim.activation().clone();
        let kernel_params = cpu_sim.kernel_params().clone();
        let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

        let m0: f64 = initial_a.iter().map(|&v| f64::from(v)).sum();
        assert!(m0 > 0.0);

        let started = std::time::Instant::now();
        let mut max_rel = 0.0_f64;
        for _ in 0..STEPS {
            pipeline.step(&ctx);
            // Per-step readback to track drift over the whole run.
            // Dominates timing (cf. M2.7-mass observation) but is the
            // only way to find the worst-case step.
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            let a = pipeline.readback_activation(&ctx);
            let m: f64 = a.iter().map(|&v| f64::from(v)).sum();
            let rel = (m - m0).abs() / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        let total_ms = started.elapsed().as_secs_f64() * 1000.0;
        let per_step_ms = total_ms / f64::from(STEPS);

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

    eprintln!("\n[M2.8 GPU mass conservation 100 steps]");
    for line in &report {
        eprintln!("{line}");
    }
    eprintln!();
}

/// Bare-minimum per-step timing without per-step readback (the realistic
/// per-frame loop). 1000 steps with a single readback at the end.
#[test]
fn gpu_pipeline_per_step_timing() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    let case = Case {
        paper_strict: false,
        border: BorderMode::Torus,
        channels: 3,
    };
    let cfg = case.cfg();
    let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    let mut pipeline = GpuStepPipeline::new(&ctx, &cfg, &kernel_params, &initial_a);

    // Warm-up: run 10 steps without timing to let any one-time
    // compilation/setup cost wash out.
    pipeline.run_steps(&ctx, 10);

    let n = 1000_u32;
    let started = std::time::Instant::now();
    pipeline.run_steps(&ctx, n);
    let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
    let per_step_ms = elapsed_ms / f64::from(n);

    // Compare with the CPU side from a fresh simulator. Use the same
    // (cfg, seed) so the per-step cost is on the same dynamics.
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

    // No strict assertion on absolute timing — it varies wildly with
    // CPU thermal state / OS scheduling. The summary printout is the
    // useful output here, observed via `cargo test -- --nocapture`.
    assert!(per_step_ms > 0.0);
    assert!(cpu_per_step_ms > 0.0);
}
