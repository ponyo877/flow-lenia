// Shared WGSL prelude (M2.4): types, constants, and helper functions
// that every flow-lenia compute shader uses.
//
// Concatenated at the front of each shader source string at build
// time on the Rust side via `concat!(include_str!("types.wgsl"), …)`.
// WGSL has no `#include` of its own; this is the agreed-upon way to
// share definitions across files in the project.

struct Meta {
    source_channel: u32,
    target_channel: u32,
    mu: f32,
    sigma: f32,
};

struct Globals {
    h: u32,
    w: u32,
    c: u32,
    k: u32,
    max_side: u32,
    half_side: u32,
    border: u32,    // 0 = Torus, 1 = Wall
    _pad: u32,      // reserved (M2.6 may use for paper_strict flag)
};

const BORDER_TORUS: u32 = 0u;
const BORDER_WALL: u32 = 1u;

// Hard cap on |K| (paper Table 1 / DESIGN.md §7 UI slider). Used to
// size the constant-weights `h_buf` array on the host side. The shader
// never iterates past `globals.k`, so unused tail entries simply hold 0.
const MAX_KERNELS: u32 = 45u;

// Hard cap on C (DESIGN.md §7 UI slider 1..=3). Used by the
// per-target-channel accumulator in the affinity_growth shaders.
const MAX_CHANNELS: u32 = 3u;

// Lenia growth function `G(x; μ, σ) = 2·exp(-((x-μ)/σ)²/2) - 1` (paper
// Eq. 2 / JAX `utils.py:11-14`). Same formula as
// `flow_lenia_core::growth::growth` on the CPU side.
fn growth_fn(x: f32, mu: f32, sigma: f32) -> f32 {
    let z = (x - mu) / sigma;
    return 2.0 * exp(-z * z * 0.5) - 1.0;
}
