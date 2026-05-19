//! M1 baseline regression — bit-equal comparison against the
//! committed fixtures (`tests/regression_fixtures/m1_baseline/`).
//!
//! See the M1.15 README "Regression fixtures" section for the
//! reproducibility contract. In short: identical Rust toolchain
//! (1.95.0 as of M6.A.1) and identical `ndarray` resolved version are
//! required.
//!
//! The case list is hardcoded here (rather than parsed from
//! `manifest.json`) so this test stays free of any JSON-parsing
//! dependency. The manifest exists as human-readable provenance and
//! can be re-derived from this code via
//! `cargo run --release --bin generate_m1_fixtures`.
//!
//! M6.A.2 — split the original single `m1_regression_matches_baseline_fixtures`
//! test into one `#[test]` per grid so `cargo test` reports the
//! grid-level pass/fail breakdown directly and `cargo test
//! m1_regression_g64` can run just that grid during M6 perf
//! iterations.

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};
use std::fs;
use std::path::PathBuf;

const SEED: u64 = 42;
const STEPS: u32 = 100;
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

/// Eight-case matrix (`paper_strict × border × channels`) at a single
/// grid size. Mirrors the iteration order used by
/// `generate_m1_fixtures`'s inner loop so fixture filenames and test
/// case ordering stay in lock-step.
fn cases_for_grid(grid: u32) -> Vec<Case> {
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

/// Run the 8-case regression for one grid size. Panics on the first
/// mismatch (across any case, any element) with a message that names
/// both the grid and the case filename so a future failure points
/// directly at the offending `(grid, paper_strict, border, channels)`
/// combination.
fn run_grid_regression(grid: u32) {
    for case in cases_for_grid(grid) {
        let cfg = case.cfg();
        let mut sim = FlowLeniaSimulator::new(cfg, SEED);
        sim.step_many(STEPS);

        let actual = sim
            .activation()
            .as_slice()
            .expect("activation should be contiguous");

        let expected_len =
            (case.grid as usize) * (case.grid as usize) * (case.channels as usize);
        let expected = read_f32_bin(fixture_path(&case.filename()), expected_len);

        assert_eq!(
            actual.len(),
            expected.len(),
            "grid={} {}: shape mismatch (got {}, expected {})",
            grid,
            case.filename(),
            actual.len(),
            expected.len()
        );

        let mut first_mismatch: Option<(usize, f32, f32)> = None;
        for (i, (&got, &want)) in actual.iter().zip(expected.iter()).enumerate() {
            if got.to_bits() != want.to_bits() {
                first_mismatch = Some((i, got, want));
                break;
            }
        }

        if let Some((i, got, want)) = first_mismatch {
            let abs_err = (got - want).abs();
            panic!(
                "grid={grid} {}: bit-equal mismatch at element {i} \
                 (got = {got:e} [bits 0x{:08x}], expected = {want:e} [bits 0x{:08x}], \
                 abs_err = {abs_err:e}). \
                 Toolchain or `ndarray` may have changed — see README \
                 \"Regression fixtures\" for the reproducibility contract.",
                case.filename(),
                got.to_bits(),
                want.to_bits()
            );
        }
    }
}

#[test]
fn m1_regression_g32() {
    run_grid_regression(32);
}

#[test]
fn m1_regression_g64() {
    run_grid_regression(64);
}

// M6.A.2.1 — g128 / g256 are gated behind `#[ignore]` because their
// 100-step regression at the 128×128 / 256×256 grid takes ~3 minutes
// (g128) and ~13 minutes (g256) on the M1 baseline, which is too slow
// for default `cargo test` runs during M6 perf iterations. They still
// run under `cargo test -- --include-ignored` (or by name with the
// flag set), and the convention is to run the full set before pushing
// any commit that could touch the simulator's numerical path.
#[test]
#[ignore = "heavy 128×128 regression (~3 min); --include-ignored to run"]
fn m1_regression_g128() {
    run_grid_regression(128);
}

#[test]
#[ignore = "heavy 256×256 regression (~13 min); --include-ignored to run"]
fn m1_regression_g256() {
    run_grid_regression(256);
}
