#![deny(warnings)]
//! Native CPU binary for Flow-Lenia.
//!
//! Runs the CPU reference implementation (`flow-lenia-core`) with
//! terminal visualisation (`flow_lenia_app::render_terminal`). Implemented
//! in M1.14 per DESIGN.md §8.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --bin native_cpu -- [seed] [steps] [render_every]
//! ```
//!
//! All three arguments are optional positional integers with the
//! defaults: `seed = 42`, `steps = 1000`, `render_every = 10`. We
//! deliberately *do not* pull in `clap` for M1.14 — a 3-arg positional
//! interface keeps the binary's startup cost minimal and the dependency
//! tree small. Promote to `clap` once flag-parsing pain shows up.

use flow_lenia_app::render_terminal::{clear_terminal, render_to_terminal};
use flow_lenia_core::{FlowLeniaConfig, FlowLeniaSimulator};
use std::time::{Duration, Instant};

fn parse_arg<T: std::str::FromStr>(idx: usize, default: T) -> T {
    std::env::args()
        .nth(idx)
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn main() {
    let seed: u64 = parse_arg(1, 42);
    let steps: u32 = parse_arg(2, 1000);
    let render_every: u32 = parse_arg(3, 10).max(1);

    let cfg = FlowLeniaConfig::demo_default();
    let mut sim = FlowLeniaSimulator::new(cfg, seed);

    // Initial render so we show the seed state, not the t=1 state.
    clear_terminal();
    render_to_terminal(&sim);
    println!(
        "step={:5}  mass=[{}]  seed={}",
        sim.step_count(),
        format_mass(&sim.total_mass()),
        seed
    );

    let frame_period = Duration::from_millis(50);
    let mut last_render = Instant::now();
    for _ in 0..steps {
        sim.step();
        if sim.step_count() as u32 % render_every == 0 {
            let elapsed = last_render.elapsed();
            if elapsed < frame_period {
                std::thread::sleep(frame_period - elapsed);
            }
            clear_terminal();
            render_to_terminal(&sim);
            println!(
                "step={:5}  mass=[{}]  seed={}",
                sim.step_count(),
                format_mass(&sim.total_mass()),
                seed
            );
            last_render = Instant::now();
        }
    }
}

fn format_mass(m: &[f32]) -> String {
    m.iter()
        .map(|v| format!("{v:8.2}"))
        .collect::<Vec<_>>()
        .join(", ")
}
