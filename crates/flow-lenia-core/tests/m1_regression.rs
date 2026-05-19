//! M1 baseline regression — bit-equal comparison against the
//! committed fixtures (`tests/regression_fixtures/m1_baseline/`).
//!
//! See the M1.15 README "Regression fixtures" section for the
//! reproducibility contract. In short: identical Rust toolchain
//! (1.87.0) and identical `ndarray` resolved version are required.
//!
//! The case list is hardcoded here (rather than parsed from
//! `manifest.json`) so this test stays free of any JSON-parsing
//! dependency. The manifest exists as human-readable provenance and
//! can be re-derived from this code via
//! `cargo run --release --bin generate_m1_fixtures`.

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};
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
        // M6.A.1 — the fixture filename now embeds the grid as
        // `case_g{N}_ps{T,F}_bt{T,W}_c{1,3}.bin`. A.1 still runs the
        // regression at the M1 baseline 32×32 only; A.2 will widen
        // the case-set to the other grid sizes already generated.
        let ps = if self.paper_strict { 'T' } else { 'F' };
        let bt = match self.border {
            BorderMode::Torus => 'T',
            BorderMode::Wall => 'W',
        };
        format!("case_g{}_ps{}_bt{}_c{}.bin", GRID, ps, bt, self.channels)
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
fn m1_regression_matches_baseline_fixtures() {
    for case in &all_cases() {
        let cfg = case.cfg();
        let mut sim = FlowLeniaSimulator::new(cfg, SEED);
        sim.step_many(STEPS);

        let actual = sim
            .activation()
            .as_slice()
            .expect("activation should be contiguous");

        let expected_len = (GRID as usize) * (GRID as usize) * (case.channels as usize);
        let expected = read_f32_bin(fixture_path(&case.filename()), expected_len);

        assert_eq!(
            actual.len(),
            expected.len(),
            "{}: shape mismatch (got {}, expected {})",
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
                "{}: bit-equal mismatch at element {i} \
                 (got = {got:e}, expected = {want:e}, abs_err = {abs_err:e}). \
                 Toolchain or `ndarray` may have changed — see README \
                 \"Regression fixtures\" for the reproducibility contract.",
                case.filename()
            );
        }
    }
}
