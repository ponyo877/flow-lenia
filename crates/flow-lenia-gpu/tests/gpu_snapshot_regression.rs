//! M6.A.5 GPU snapshot regression.
//!
//! Compares the current `GpuStepPipeline` output against a fixed
//! snapshot committed under `tests/regression_fixtures/gpu_baseline/`.
//! The snapshot was generated at the M6.A.5 commit using the same
//! Rust toolchain (1.95.0) and wgpu (29.0.3) shipped in M6.A.1 — see
//! `tests/regression_fixtures/gpu_baseline/manifest.json` for the
//! provenance.
//!
//! ## What this tests vs. what `m1_regression_gpu` tests
//!
//! `m1_regression_gpu::gpu_field_regression_g{N}` measures
//! **GPU vs CPU** field rel at the matching grid, C=1, 10 step.
//! It catches "the GPU has drifted away from the CPU baseline" — the
//! traditional cross-implementation regression.
//!
//! `gpu_snapshot_regression::gpu_snapshot_g{N}_*step` measures
//! **GPU(current) vs GPU(committed snapshot)** field rel. It catches
//! "the GPU has drifted from its own pre-M6 self" — useful when an
//! M6.C refactor introduces a numerical-path change that's still
//! correct vs. CPU but visibly different from the previous WGSL.
//!
//! Failure interpretation table for M6.C debugging:
//!
//! | A.4 fails | A.5 fails | Likely cause                                  |
//! |-----------|-----------|-----------------------------------------------|
//! | yes       | yes       | GPU shaders genuinely regressed numerically.  |
//! | yes       | no        | GPU drifted from CPU but matches old GPU — CPU side may have changed; investigate symmetric. |
//! | no        | yes       | GPU output changed but CPU agreement preserved within A.4 tolerance — could be acceptable M6.C drift, check BENCH §9 budget. |
//! | no        | no        | All clear.                                    |
//!
//! ## Grids and step counts
//!
//! M6.A.4.5 found Flow-Lenia is strongly chaotic at grid ≥ 64
//! (Lyapunov saturation in 1 step under ε = 1e-6 perturbation), so
//! the snapshot horizons are kept short enough that GPU vs CPU rel
//! stays in the chaos-noise-floor regime:
//!
//! | grid | step | observed rel (M6.A.4 Exp 4) | M6.C drift tolerance |
//! |-----:|-----:|----------------------------:|---------------------:|
//! |   32 |   10 |                     1.93e-5 |              1e-4    |
//! |   64 |    5 |                     4.10e-4 |              3e-4 *  |
//! |  128 |    3 |                     4.87e-4 |              2e-4 *  |
//!
//! \* The M6.C drift tolerance is *not* the M6.A.4 GPU-vs-CPU
//! tolerance; this is "how much the GPU's own output is allowed to
//! shift between M6.A and a future M6.C step before we flag it for
//! review". For an initial regression run the observed rel against
//! the snapshot is zero by construction (the snapshot *is* the
//! current GPU output). The non-zero tolerances exist so a future
//! M6.C numerical-path change can introduce a documented bit of
//! drift without immediately failing — review the diff before any
//! upper-budget approval.
//!
//! 256×256 is intentionally excluded: at the chaos-saturation regime,
//! the per-cell GPU output diverges grid-dependently even between
//! genuinely-equivalent runs, so a fixed snapshot would be too
//! brittle to serve as a regression anchor.
//!
//! ## Regenerating the snapshot
//!
//! The `generate_gpu_snapshots` test below is `#[ignore]`-gated; run
//! it explicitly to write fresh snapshot binaries:
//!
//! ```text
//! cargo test --release -p flow-lenia-gpu --test gpu_snapshot_regression \
//!     generate_gpu_snapshots -- --include-ignored --nocapture
//! ```
//!
//! Only do this when an M6.C numerical-path change has been reviewed
//! and approved — replacing the snapshot blesses whatever new GPU
//! output the simulator produces.

mod common;
use common::assert_creature_alive;

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    state::ActivationField,
    FlowLeniaConfig, FlowLeniaSimulator,
};
use flow_lenia_gpu::{GpuContext, GpuStepPipeline};
use std::fs;
use std::io::Write;
use std::path::PathBuf;

const SEED: u64 = 42;
const NUM_KERNELS: u32 = 10;

#[derive(Clone, Copy)]
struct Case {
    grid: u32,
    paper_strict: bool,
    border: BorderMode,
}

impl Case {
    fn filename(&self) -> String {
        let ps = if self.paper_strict { 'T' } else { 'F' };
        let bt = match self.border {
            BorderMode::Torus => 'T',
            BorderMode::Wall => 'W',
        };
        format!("case_g{}_ps{}_bt{}_c1.bin", self.grid, ps, bt)
    }

    fn cfg(&self) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: self.grid,
            grid_height: self.grid,
            channels: 1,
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

fn cases_for_grid(grid: u32) -> Vec<Case> {
    let mut out = Vec::with_capacity(4);
    for &paper_strict in &[false, true] {
        for &border in &[BorderMode::Torus, BorderMode::Wall] {
            out.push(Case {
                grid,
                paper_strict,
                border,
            });
        }
    }
    out
}

fn snapshot_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("regression_fixtures")
        .join("gpu_baseline")
}

fn snapshot_path(case: &Case) -> PathBuf {
    snapshot_dir().join(case.filename())
}

fn read_f32_bin(path: PathBuf, expected_len: usize) -> Vec<f32> {
    let bytes = fs::read(&path)
        .unwrap_or_else(|e| panic!("failed to read snapshot {}: {e}", path.display()));
    assert_eq!(
        bytes.len(),
        expected_len * 4,
        "snapshot {} has {} bytes, expected {}",
        path.display(),
        bytes.len(),
        expected_len * 4
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn write_f32_bin(path: PathBuf, data: &[f32]) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).unwrap_or_else(|e| {
            panic!("mkdir {}: {e}", parent.display())
        });
    }
    let mut f = fs::File::create(&path)
        .unwrap_or_else(|e| panic!("create {}: {e}", path.display()));
    for &v in data {
        f.write_all(&v.to_le_bytes())
            .unwrap_or_else(|e| panic!("write {}: {e}", path.display()));
    }
}

fn run_gpu_at_case(case: &Case, n_steps: u32, ctx: &GpuContext) -> ActivationField {
    let cfg = case.cfg();
    let cpu_init = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_init.activation().clone();
    let kernel_params = cpu_init.kernel_params().clone();

    let mut pipeline = GpuStepPipeline::new(ctx, &cfg, &kernel_params, &initial_a);
    pipeline.run_steps(ctx, n_steps);
    let a = pipeline.readback_activation(ctx);

    assert_creature_alive(
        a.as_slice().expect("activation should be contiguous"),
        &cfg,
    );
    a
}

fn run_snapshot_regression(grid: u32, n_steps: u32, rel_tolerance: f32) {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    let cases = cases_for_grid(grid);
    let mut per_case_summary: Vec<String> = Vec::new();
    let mut overall_max_rel = 0.0_f32;

    for case in &cases {
        let gpu_a = run_gpu_at_case(case, n_steps, &ctx);
        let (h, w, c) = gpu_a.dim();

        let snapshot = read_f32_bin(snapshot_path(case), h * w * c);
        assert_eq!(
            snapshot.len(),
            h * w * c,
            "{}: snapshot length mismatch",
            case.filename()
        );

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for y in 0..h {
            for x in 0..w {
                for ci in 0..c {
                    let gv = gpu_a[[y, x, ci]];
                    let i = y * w * c + x * c + ci;
                    let sv = snapshot[i];
                    let abs_err = (gv - sv).abs();
                    let rel_err = abs_err / sv.abs().max(1e-6);
                    max_abs = max_abs.max(abs_err);
                    max_rel = max_rel.max(rel_err);
                }
            }
        }
        overall_max_rel = overall_max_rel.max(max_rel);

        per_case_summary.push(format!(
            "  {}: max_abs = {max_abs:.3e}, max_rel = {max_rel:.3e}",
            case.filename()
        ));
    }

    eprintln!(
        "\n[M6.A.5 GPU snapshot regression — grid={grid}, C=1, steps={n_steps}]"
    );
    for line in &per_case_summary {
        eprintln!("{line}");
    }
    eprintln!("  overall max_rel = {overall_max_rel:.3e}  (tolerance {rel_tolerance:.0e})\n");

    assert!(
        overall_max_rel < rel_tolerance,
        "grid={grid}: GPU output drifted from committed snapshot \
         (max_rel {overall_max_rel:.3e} >= tolerance {rel_tolerance:.0e}). \
         Either an M6.C numerical-path change introduced more drift than \
         budgeted, or the snapshot itself was generated under a different \
         driver / wgpu version — check `tests/regression_fixtures/\
         gpu_baseline/manifest.json` for the recorded provenance."
    );
}

#[test]
fn gpu_snapshot_g32_10step() {
    run_snapshot_regression(32, 10, 1e-4);
}

#[test]
fn gpu_snapshot_g64_5step() {
    run_snapshot_regression(64, 5, 3e-4);
}

#[test]
fn gpu_snapshot_g128_3step() {
    run_snapshot_regression(128, 3, 2e-4);
}

/// Regenerate the GPU snapshot binaries plus a fresh `manifest.json`.
/// `#[ignore]` so a normal `cargo test` run never overwrites the
/// committed baseline — invoke explicitly with `--include-ignored`
/// after an approved M6.C numerical-path change.
#[test]
#[ignore = "M6.A.5 snapshot regenerator; --include-ignored to overwrite the committed baseline"]
fn generate_gpu_snapshots() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);

    let dir = snapshot_dir();
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));

    let plan: &[(u32, u32)] = &[(32, 10), (64, 5), (128, 3)];
    let mut written: Vec<(Case, u32, usize)> = Vec::new();

    for &(grid, n_steps) in plan {
        for case in cases_for_grid(grid) {
            let gpu_a = run_gpu_at_case(&case, n_steps, &ctx);
            let slice = gpu_a
                .as_slice()
                .expect("activation should be contiguous");
            let path = snapshot_path(&case);
            write_f32_bin(path.clone(), slice);
            eprintln!(
                "  wrote {} ({} bytes, {} step)",
                path.display(),
                slice.len() * 4,
                n_steps
            );
            written.push((case, n_steps, slice.len() * 4));
        }
    }

    write_manifest(&dir, &written);
}

fn write_manifest(dir: &PathBuf, written: &[(Case, u32, usize)]) {
    let today = ymd_today();
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"generated_at\": \"{today}\",\n"));
    out.push_str("  \"rust_toolchain\": \"1.95.0\",\n");
    out.push_str("  \"wgpu_version\": \"29.0.3\",\n");
    out.push_str("  \"adapter\": \"Apple M1 (IntegratedGpu, Metal)\",\n");
    out.push_str(
        "  \"fixture_purpose\": \"M6.A.5 GPU output snapshot — pre-M6.C numerical-path baseline\",\n",
    );
    out.push_str(&format!(
        "  \"commit_sha\": \"{}\",\n",
        git_head_sha()
    ));
    out.push_str(&format!("  \"num_kernels\": {NUM_KERNELS},\n"));
    out.push_str(&format!("  \"seed\": {SEED},\n"));
    out.push_str("  \"cases\": [\n");
    for (i, (case, n_steps, n_bytes)) in written.iter().enumerate() {
        let border_name = match case.border {
            BorderMode::Torus => "Torus",
            BorderMode::Wall => "Wall",
        };
        out.push_str("    {\n");
        out.push_str(&format!("      \"filename\": \"{}\",\n", case.filename()));
        out.push_str(&format!("      \"grid\": {},\n", case.grid));
        out.push_str(&format!("      \"paper_strict\": {},\n", case.paper_strict));
        out.push_str(&format!("      \"border\": \"{border_name}\",\n"));
        out.push_str("      \"channels\": 1,\n");
        out.push_str(&format!("      \"steps\": {n_steps},\n"));
        out.push_str(&format!(
            "      \"shape\": [{0}, {0}, 1],\n",
            case.grid
        ));
        out.push_str("      \"dtype\": \"f32_le\",\n");
        out.push_str(&format!("      \"bytes\": {n_bytes}\n"));
        out.push_str("    }");
        if i + 1 < written.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n");
    out.push_str("}\n");

    let path = dir.join("manifest.json");
    fs::write(&path, out).unwrap_or_else(|e| panic!("write manifest: {e}"));
    eprintln!("  wrote {}", path.display());
}

fn git_head_sha() -> String {
    use std::process::Command;
    let out = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output();
    if let Ok(out) = out {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    "unknown".to_string()
}

fn ymd_today() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86_400) as i64;
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i64 + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = (doy - (153 * mp + 2) / 5 + 1) as u32;
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as u32;
    let y = if m <= 2 { y + 1 } else { y };
    format!("{:04}-{:02}-{:02}", y, m, d)
}
