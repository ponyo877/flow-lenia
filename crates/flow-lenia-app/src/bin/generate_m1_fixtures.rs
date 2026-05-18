#![deny(warnings)]
//! Generate M1 baseline regression fixtures.
//!
//! Writes 8 fixture files + a `manifest.json` to
//! `tests/regression_fixtures/m1_baseline/` (relative to the repo
//! root). Each fixture is the activation field after `SEED`-seeded
//! simulation of `STEPS` steps under one of the 8 mode combinations
//! (`paper_strict × border × C`), serialised as raw little-endian
//! `f32` in `(H, W, C)` row-major order.
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

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;

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
    out.push_str("  \"rust_toolchain\": \"1.87.0\",\n");
    out.push_str(&format!(
        "  \"ndarray_version\": \"{}\",\n",
        ndarray_version()
    ));
    out.push_str(&format!(
        "  \"grid\": {{\"width\": {GRID}, \"height\": {GRID}}},\n"
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
        out.push_str(&format!("      \"paper_strict\": {},\n", c.paper_strict));
        out.push_str(&format!("      \"border\": \"{border_name}\",\n"));
        out.push_str(&format!("      \"channels\": {},\n", c.channels));
        out.push_str(&format!(
            "      \"shape\": [{GRID}, {GRID}, {}],\n",
            c.channels
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

/// Read the `ndarray` dependency version from the (committed)
/// `Cargo.lock` so the manifest captures the *resolved* version, not
/// the requirement-range string from `Cargo.toml`. Falls back to
/// `"unknown"` if anything is off — the manifest is descriptive
/// metadata, not load-bearing.
fn ndarray_version() -> String {
    let cargo_lock = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("Cargo.lock");
    let Ok(content) = fs::read_to_string(&cargo_lock) else {
        return "unknown".to_string();
    };
    let mut in_ndarray = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "name = \"ndarray\"" {
            in_ndarray = true;
            continue;
        }
        if in_ndarray && trimmed.starts_with("version = \"") {
            return trimmed
                .trim_start_matches("version = \"")
                .trim_end_matches('"')
                .to_string();
        }
        if in_ndarray && trimmed.starts_with("[[package]]") {
            // moved past the ndarray section without finding a version
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
