//! Mass conservation regression — one `#[test]` per grid size, plus
//! two one-off measurement tests used to populate BENCH.md.
//!
//! Each grid test runs the CPU simulator for the documented step
//! count over the M1.15 8-case matrix (`paper_strict × border ×
//! channels`) and asserts that the total mass stays within the
//! per-border tolerance budget (Torus 1e-3, Wall 1e-2). The
//! `assert_creature_alive` sanity check from `tests/common/mod.rs`
//! runs at the end of each case to catch NaN explosion or activation
//! collapse that a relative-error mass test would let slip.
//!
//! M6.A.3 tiered step counts (see BENCH.md "Section 5 — drift vs
//! grid size at 100 step" for the empirical justification):
//!
//! | Grid | Steps | Cases | Est. runtime |
//! |------|-------|-------|--------------|
//! | 32   | 1000  | 8 (M1.15 baseline, unchanged) | ~44 s |
//! | 64   | 500   | 8                              | ~4 min |
//! | 128  | 200   | 8                              | ~6 min |
//! | 256  | 200   | 8                              | ~23 min |
//! | 512  | 50    | 4 (paper_strict=F only)        | ~12 min |
//!
//! All five are `#[ignore]`d so default `cargo test` runs them as
//! "ignored". Use `cargo test --release -p flow-lenia-core --test
//! mass_conservation_1k -- --include-ignored --nocapture` to run them.
//!
//! Two one-off measurement tests live alongside:
//!
//! - `drift_vs_grid_size_100step` — empirical drift table feeding
//!   BENCH.md Section 5. Re-run after any simulator change that could
//!   shift the f32 accumulation path.
//! - `baseline_64x64_1000step` — the 7.7-minute reference run that
//!   anchors the choice of 500 step for the regular `g64` test.
//!
//! Both are `#[ignore]`d; they're not regression-detecting, just
//! reproducible measurement.

mod common;
use common::assert_creature_alive;

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};

const SEED: u64 = 42;
const NUM_KERNELS: u32 = 10;

#[derive(Clone, Copy)]
struct Case {
    grid: u32,
    paper_strict: bool,
    border: BorderMode,
    channels: u32,
}

impl Case {
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

/// 8-case matrix (`paper_strict × border × channels`) at a fixed grid.
/// `paper_strict_filter = Some(false)` halves the case set for 512
/// where the full 8 cases would push the test runtime above 50 min.
fn cases_for_grid(grid: u32, paper_strict_filter: Option<bool>) -> Vec<Case> {
    let mut out = Vec::new();
    let ps_values: &[bool] = match paper_strict_filter {
        Some(true) => &[true],
        Some(false) => &[false],
        None => &[false, true],
    };
    for &paper_strict in ps_values {
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

fn total_mass(sim: &FlowLeniaSimulator) -> f64 {
    sim.total_mass().iter().map(|&v| f64::from(v)).sum()
}

fn tolerance_for(border: BorderMode) -> f64 {
    match border {
        BorderMode::Torus => 1e-3,
        BorderMode::Wall => 1e-2,
    }
}

/// Run the mass-conservation matrix at a fixed `(grid, n_steps)`. Each
/// case panics on either a tolerance violation or an
/// `assert_creature_alive` failure; the eprintln-rendered matrix
/// surfaces every case (passed or failed) under `--nocapture`.
fn run_mass_conservation(grid: u32, n_steps: u32, cases: &[Case]) {
    let mut report: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for case in cases {
        let cfg = case.cfg();
        let mut sim = FlowLeniaSimulator::new(cfg, SEED);
        let m0 = total_mass(&sim);
        assert!(m0 > 0.0, "grid={grid}: initial mass should be positive");

        let mut max_rel = 0.0_f64;
        for _ in 0..n_steps {
            sim.step();
            let m = total_mass(&sim);
            let rel = (m - m0).abs() / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }

        // Final sanity. Runs before the tolerance check because a NaN
        // mass passes `rel < tol` vacuously (NaN compares produce
        // false), and the sanity panic message points at the failure
        // mode more usefully than "tolerance exceeded" would.
        assert_creature_alive(
            sim.activation()
                .as_slice()
                .expect("activation should be contiguous"),
            &cfg,
        );

        let tol = tolerance_for(case.border);
        let label = format!(
            "grid={grid:3}  paper_strict={:5}  border={:?}  C={}  steps={n_steps:4}  max_rel={:.3e}",
            case.paper_strict, case.border, case.channels, max_rel,
        );
        report.push(label.clone());
        if max_rel >= tol {
            failures.push(format!("{label}  (tol {tol:.0e})"));
        }
    }

    eprintln!("\n=== mass conservation matrix (grid={grid}, steps={n_steps}) ===");
    for line in &report {
        eprintln!("  {line}");
    }
    eprintln!("======================================================\n");

    assert!(
        failures.is_empty(),
        "mass-conservation budget exceeded for: {failures:#?}"
    );
}

/// Same iteration as `run_mass_conservation` but returns the per-case
/// `max_rel` without asserting tolerance. Used by
/// `drift_vs_grid_size_100step` so a single grid's outlier doesn't
/// abort the rest of the sweep before we've printed the full table.
fn measure_drift_for_grid(grid: u32, n_steps: u32, cases: &[Case]) -> Vec<(Case, f64)> {
    let mut out = Vec::with_capacity(cases.len());
    for case in cases {
        let cfg = case.cfg();
        let mut sim = FlowLeniaSimulator::new(cfg, SEED);
        let m0 = total_mass(&sim);
        assert!(m0 > 0.0, "grid={grid}: initial mass should be positive");

        let mut max_rel = 0.0_f64;
        for _ in 0..n_steps {
            sim.step();
            let m = total_mass(&sim);
            let rel = (m - m0).abs() / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        assert_creature_alive(
            sim.activation()
                .as_slice()
                .expect("activation should be contiguous"),
            &cfg,
        );
        out.push((*case, max_rel));
    }
    out
}

#[test]
#[ignore = "mass conservation g32 (1000 step, ~44 s); --include-ignored under --release"]
fn mass_conservation_g32() {
    let cases = cases_for_grid(32, None);
    run_mass_conservation(32, 1000, &cases);
}

#[test]
#[ignore = "mass conservation g64 (500 step, ~4 min); --include-ignored under --release"]
fn mass_conservation_g64() {
    let cases = cases_for_grid(64, None);
    run_mass_conservation(64, 500, &cases);
}

#[test]
#[ignore = "mass conservation g128 (200 step, ~6 min); --include-ignored under --release"]
fn mass_conservation_g128() {
    let cases = cases_for_grid(128, None);
    run_mass_conservation(128, 200, &cases);
}

#[test]
#[ignore = "mass conservation g256 (200 step, ~23 min); --include-ignored under --release"]
fn mass_conservation_g256() {
    let cases = cases_for_grid(256, None);
    run_mass_conservation(256, 200, &cases);
}

#[test]
#[ignore = "mass conservation g512 (50 step, 4 cases paper_strict=F, ~12 min); --include-ignored under --release"]
fn mass_conservation_g512() {
    let cases = cases_for_grid(512, Some(false));
    run_mass_conservation(512, 50, &cases);
}

/// M6.A.3 condition 3: drift vs grid size at constant step count.
/// The measured values feed BENCH.md Section 5 and inform whether the
/// tiered step counts above keep regression-detection power consistent
/// across grids. Re-run by hand after any simulator change that could
/// shift the f32 accumulation path.
///
/// 4 cases per grid (paper_strict=false, both borders, both channels)
/// keep the matrix readable; the table is printed via eprintln and
/// captured under `--nocapture`.
#[test]
#[ignore = "M6.A.3 drift measurement (~42 min); --include-ignored to refresh BENCH.md Section 5"]
fn drift_vs_grid_size_100step() {
    eprintln!("\n=== M6.A.3 drift vs grid size at 100 steps ===");
    for &grid in &[32_u32, 64, 128, 256, 512] {
        let cases = cases_for_grid(grid, Some(false));
        let results = measure_drift_for_grid(grid, 100, &cases);
        for (case, max_rel) in &results {
            eprintln!(
                "  grid={grid:3}  border={:?}  C={}  max_rel={:.3e}",
                case.border, case.channels, max_rel
            );
        }
    }
    eprintln!();
}

/// M6.A.3 condition 1: 64×64 1000-step baseline. The choice of 500
/// step for the regular `mass_conservation_g64` test cuts runtime by
/// half; this test records what the *full* 1000-step value would be,
/// so BENCH.md can report the gap between regression-budget (1e-3)
/// and observed-drift (this value).
#[test]
#[ignore = "M6.A.3 64×64 1000-step baseline (~7.7 min); --include-ignored to refresh BENCH.md"]
fn baseline_64x64_1000step() {
    let cases = cases_for_grid(64, None);
    let results = measure_drift_for_grid(64, 1000, &cases);
    eprintln!("\n=== M6.A.3 baseline 64×64 1000-step max_rel ===");
    for (case, max_rel) in &results {
        eprintln!(
            "  paper_strict={:5}  border={:?}  C={}  max_rel={:.3e}",
            case.paper_strict, case.border, case.channels, max_rel
        );
    }
    eprintln!();
}
