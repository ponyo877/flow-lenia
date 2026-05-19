#![deny(warnings)]
//! Generate M1 baseline regression fixtures.
//!
//! Writes one fixture file per (grid, paper_strict, border, channels)
//! combination plus a `manifest.json` to
//! `tests/regression_fixtures/m1_baseline/` (relative to the repo
//! root). Each fixture is the activation field after `SEED`-seeded
//! simulation of `STEPS` steps, serialised as raw little-endian `f32`
//! in `(H, W, C)` row-major order.
//!
//! Run with:
//!
//! ```text
//! cargo run --release --bin generate_m1_fixtures
//! ```
//!
//! Bit-equality with these fixtures is enforced by
//! `crates/flow-lenia-core/tests/m1_regression.rs` so callers should
//! regenerate (and commit) the fixtures *only* when the dynamics or
//! supporting infra (Rust toolchain, `ndarray`) intentionally changes
//! — see README.md "Regression fixtures" section.
//!
//! M6.A.1 — extended from 32×32-only (8 cases) to {32, 64, 128, 256}×
//! {1, 3 channels} (32 cases). The filename now embeds the grid as
//! `case_g{N}_ps{T,F}_bt{T,W}_c{1,3}.bin`. 512×512 is intentionally
//! excluded: bit-equal regression at 512 would cost 24 MB of fixture
//! storage and ~10 minutes of CPU regeneration time for marginal
//! coverage value above what the GPU-mass-conservation layer provides
//! (see DESIGN.md M6.A scope notes).

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

const SEED: u64 = 42;
const STEPS: u32 = 100;
const GRIDS: &[u32] = &[32, 64, 128, 256];
const NUM_KERNELS: u32 = 10;

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

fn all_cases() -> Vec<Case> {
    let mut out = Vec::with_capacity(GRIDS.len() * 8);
    for &grid in GRIDS {
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
    }
    out
}

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("regression_fixtures")
        .join("m1_baseline")
}

fn write_bin(path: &Path, data: &[f32]) {
    let mut f = fs::File::create(path)
        .unwrap_or_else(|e| panic!("failed to create {}: {e}", path.display()));
    for &v in data {
        f.write_all(&v.to_le_bytes())
            .unwrap_or_else(|e| panic!("write failed for {}: {e}", path.display()));
    }
}

fn write_manifest(dir: &Path, cases: &[Case], total_elapsed_ms: u128) {
    // Hand-rolled JSON to avoid pulling in serde_json. The format is
    // small and stable, and the test harness intentionally does *not*
    // parse this file (hardcoded case list — see m1_regression.rs).
    let today = ymd_today();
    let mut out = String::new();
    out.push_str("{\n");
    out.push_str(&format!("  \"generated_at\": \"{today}\",\n"));
    out.push_str("  \"rust_toolchain\": \"1.95.0\",\n");
    out.push_str(&format!(
        "  \"ndarray_version\": \"{}\",\n",
        cargo_lock_version("ndarray")
    ));
    out.push_str(&format!(
        "  \"wgpu_version\": \"{}\",\n",
        cargo_lock_version("wgpu")
    ));
    out.push_str(
        "  \"fixture_purpose\": \"M6 baseline regression - pre-tuning snapshot\",\n",
    );
    out.push_str(&format!(
        "  \"commit_sha\": \"{}\",\n",
        git_head_sha()
    ));
    out.push_str(&format!("  \"num_kernels\": {NUM_KERNELS},\n"));
    out.push_str(&format!("  \"seed\": {SEED},\n"));
    out.push_str(&format!("  \"steps\": {STEPS},\n"));
    out.push_str(&format!("  \"generated_in_ms\": {total_elapsed_ms},\n"));
    out.push_str("  \"cases\": [\n");
    for (i, c) in cases.iter().enumerate() {
        let border_name = match c.border {
            BorderMode::Torus => "Torus",
            BorderMode::Wall => "Wall",
        };
        out.push_str("    {\n");
        out.push_str(&format!("      \"filename\": \"{}\",\n", c.filename()));
        out.push_str(&format!("      \"grid\": {},\n", c.grid));
        out.push_str(&format!("      \"paper_strict\": {},\n", c.paper_strict));
        out.push_str(&format!("      \"border\": \"{border_name}\",\n"));
        out.push_str(&format!("      \"channels\": {},\n", c.channels));
        out.push_str(&format!("      \"seed\": {SEED},\n"));
        out.push_str(&format!("      \"steps\": {STEPS},\n"));
        out.push_str(&format!(
            "      \"shape\": [{0}, {0}, {1}],\n",
            c.grid, c.channels
        ));
        out.push_str("      \"dtype\": \"f32_le\"\n");
        out.push_str("    }");
        if i + 1 < cases.len() {
            out.push(',');
        }
        out.push('\n');
    }
    out.push_str("  ]\n");
    out.push_str("}\n");

    let path = dir.join("manifest.json");
    fs::write(&path, out).unwrap_or_else(|e| panic!("failed to write manifest: {e}"));
    println!("  wrote {}", path.display());
}

/// Best-effort `git rev-parse HEAD` for manifest provenance. Run from
/// the package's manifest dir; if `git` is unavailable or the repo
/// state is unusable, we fall back to `"unknown"` — the manifest is
/// informational, not load-bearing.
fn git_head_sha() -> String {
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

/// Best-effort YYYY-MM-DD using only `std::time`. Avoids pulling in
/// `chrono` for one date stamp.
fn ymd_today() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    // Days since 1970-01-01 (UTC). Civil-from-days algorithm
    // (Hinnant, "chrono-Compatible Low-Level Date Algorithms"), the
    // canonical proleptic Gregorian conversion routine.
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

/// Read the *resolved* version of a workspace dependency from the
/// committed `Cargo.lock`. Used to capture per-crate version provenance
/// in the manifest. Falls back to `"unknown"` if the lockfile is
/// missing or the crate isn't present — the manifest is descriptive
/// metadata, not load-bearing.
fn cargo_lock_version(crate_name: &str) -> String {
    let cargo_lock = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("Cargo.lock");
    let Ok(content) = fs::read_to_string(&cargo_lock) else {
        return "unknown".to_string();
    };
    let target = format!("name = \"{crate_name}\"");
    let mut in_section = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == target {
            in_section = true;
            continue;
        }
        if in_section && trimmed.starts_with("version = \"") {
            return trimmed
                .trim_start_matches("version = \"")
                .trim_end_matches('"')
                .to_string();
        }
        if in_section && trimmed.starts_with("[[package]]") {
            // Moved past the section without finding a version.
            break;
        }
    }
    "unknown".to_string()
}

fn main() {
    let dir = fixture_dir();
    fs::create_dir_all(&dir).unwrap_or_else(|e| panic!("mkdir {}: {e}", dir.display()));
    println!("generating fixtures into {}", dir.display());

    let cases = all_cases();
    let started = Instant::now();
    for case in &cases {
        let case_started = Instant::now();
        let mut sim = FlowLeniaSimulator::new(case.cfg(), SEED);
        sim.step_many(STEPS);

        // `.as_slice()` is `Some(_)` because the simulator always owns
        // a contiguous Array3 (built by `Array3::zeros` or
        // `Array3::from_shape_fn`).
        let slice = sim
            .activation()
            .as_slice()
            .expect("activation array should be contiguous");
        let path = dir.join(case.filename());
        write_bin(&path, slice);
        println!(
            "  {} ({} bytes, {:.2}s)",
            case.filename(),
            slice.len() * 4,
            case_started.elapsed().as_secs_f64()
        );
    }
    let total_elapsed = started.elapsed();
    write_manifest(&dir, &cases, total_elapsed.as_millis());
    println!(
        "done in {:.2}s ({} cases)",
        total_elapsed.as_secs_f64(),
        cases.len()
    );
}
