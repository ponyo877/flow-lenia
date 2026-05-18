//! 1000-step mass-conservation matrix (M1.15).
//!
//! Extends the M1.13 100-step `step::tests::step_mass_conservation_*`
//! suite to 1000 steps and the full 8-case `paper_strict × border × C`
//! matrix. Single `#[ignore]`d test so it does not slow normal
//! `cargo test`; run with:
//!
//! ```text
//! cargo test --release -p flow-lenia-core --test mass_conservation_1k \
//!     -- --ignored --nocapture
//! ```
//!
//! Tolerances (DESIGN.md §5.3 / §1.4):
//! - `BorderMode::Torus`: 1e-3 (mass is mathematically conserved; only
//!   f32 accumulation drift is observed — M1.13 measured ~4.5e-6 at
//!   100 steps and the drift grows sub-linearly in step count, so
//!   1000 steps should still sit at ~10^-5).
//! - `BorderMode::Wall`: 1e-2 (the μ-clip in `reintegrate` is *not*
//!   mass-conserving at the boundary — M1.11 documented this; the
//!   relaxed bound is the M1.15 acceptance criterion for the wall
//!   path).
//!
//! Initial state and kernel sampling come from
//! `FlowLeniaSimulator::new(cfg, 42)`, so the runs are deterministic
//! and can be reproduced via the M1.14 simulator.

use flow_lenia_core::{
    config::{BorderMode, MixRule},
    FlowLeniaConfig, FlowLeniaSimulator,
};

fn total_mass(sim: &FlowLeniaSimulator) -> f64 {
    sim.total_mass().iter().map(|&v| f64::from(v)).sum()
}

fn case_cfg(paper_strict: bool, border: BorderMode, channels: u32) -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: 32,
        grid_height: 32,
        channels,
        dt: 0.2,
        sigma: 0.65,
        n: 2.0,
        beta_a: 2.0,
        dd: 5,
        num_kernels: 10,
        paper_strict,
        border,
        mix_rule: MixRule::Stochastic,
    }
}

#[test]
#[ignore = "1000-step mass conservation; run with --include-ignored under --release"]
fn mass_conservation_1000_steps_all_modes() {
    let n_steps = 1000_u32;
    let seed = 42_u64;

    let mut report: Vec<String> = Vec::new();
    let mut failures: Vec<String> = Vec::new();

    for &paper_strict in &[false, true] {
        for &border in &[BorderMode::Torus, BorderMode::Wall] {
            for &channels in &[1_u32, 3_u32] {
                let cfg = case_cfg(paper_strict, border, channels);
                let mut sim = FlowLeniaSimulator::new(cfg, seed);
                let m0 = total_mass(&sim);
                assert!(m0 > 0.0, "initial mass should be positive");

                let mut max_rel = 0.0_f64;
                for _ in 0..n_steps {
                    sim.step();
                    let m = total_mass(&sim);
                    let rel = (m - m0).abs() / m0;
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }

                let tol = match border {
                    BorderMode::Torus => 1e-3,
                    BorderMode::Wall => 1e-2,
                };
                let label = format!(
                    "paper_strict={:5}  border={:?}  C={}  max_rel={:.3e}",
                    paper_strict, border, channels, max_rel
                );
                report.push(label.clone());
                if max_rel >= tol {
                    failures.push(format!("{label}  (tol {tol:.0e})"));
                }
            }
        }
    }

    eprintln!("\n=== 1000-step mass conservation matrix ===");
    for line in &report {
        eprintln!("  {line}");
    }
    eprintln!("==========================================\n");

    assert!(
        failures.is_empty(),
        "mass-conservation budget exceeded for: {failures:#?}"
    );
}
