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
    border: u32,          // 0 = Torus, 1 = Wall
    paper_strict: u32,    // 0 = JAX-compat, 1 = paper Eq. 5 (M2.6)
    beta_a: f32,          // critical mass β_A (paper Eq. 5; M2.6)
    n: f32,               // α exponent (paper Eq. 5; M2.6)
    dd: u32,              // Chebyshev neighbourhood radius (M2.7)
    sigma: f32,           // reintegration σ (paper Eq. 6; M2.7)
    dt: f32,              // integration step (paper Eq. 6; M2.7)
    // 12 bytes of padding so the struct is 64 bytes total (a
    // 16-byte multiple, as the uniform layout rule demands).
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
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

// Border-resolved sample location.
//
// `x` and `y` are the in-bounds source coordinates the caller should
// read from; `valid` is 1 when the input was inside the grid (or
// wrapped onto it under `BORDER_TORUS`), and 0 when the input was
// outside the grid under `BORDER_WALL`. Callers should branch on
// `valid` before consuming `(x, y)`.
//
// WGSL `bool` cannot live in storage/uniform structs portably, so
// `valid` is encoded as `u32` (`0 | 1`).
struct BorderSample {
    x: u32,
    y: u32,
    valid: u32,
};

// Reintegration overlap area I(x', x) for a single source-target cell
// pair (paper Eq. 6 / JAX `reintegration_tracking.py:57-58`). Identical
// formula to `flow_lenia_core::overlap::overlap_area` (M1.10).
//
//     I = (sz_x · sz_y) / (4 σ²)
//
// with `sz_a = clamp(0.5 − |dpmu_a| + σ, 0, min(1, 2σ))`. The
// `min(1, 2σ)` upper clamp is correctness-critical for `σ > 0.5` (M1.10).
//
// `dpmu_*` is the **signed** distance from the distribution centre `μ`
// to the target cell centre, in cells. `abs` is applied internally so
// callers can pass the raw difference without an explicit `.abs()`.
fn overlap_area(dpmu_y: f32, dpmu_x: f32, sigma: f32) -> f32 {
    let abs_y = abs(dpmu_y);
    let abs_x = abs(dpmu_x);
    let upper = min(2.0 * sigma, 1.0);
    let sz_y = clamp(0.5 - abs_y + sigma, 0.0, upper);
    let sz_x = clamp(0.5 - abs_x + sigma, 0.0, upper);
    return (sz_x * sz_y) / (4.0 * sigma * sigma);
}

// Resolve `(x, y)` against the grid `(w, h)` per `border`:
//   - `BORDER_TORUS`: wrap modulo into `[0, w) × [0, h)`, `valid = 1`.
//   - `BORDER_WALL`: pass through if in-bounds (`valid = 1`); zero +
//     `valid = 0` otherwise — callers should treat the sample as 0.
//
// Inputs are signed so caller arithmetic on offsets (e.g.
// `centre_y + dy`) can take negative values without an explicit cast.
fn border_resolve(x: i32, y: i32, w: i32, h: i32, border: u32) -> BorderSample {
    var out: BorderSample;
    if (border == BORDER_TORUS) {
        let sx = ((x % w) + w) % w;
        let sy = ((y % h) + h) % h;
        out.x = u32(sx);
        out.y = u32(sy);
        out.valid = 1u;
    } else {
        if (x < 0 || x >= w || y < 0 || y >= h) {
            out.x = 0u;
            out.y = 0u;
            out.valid = 0u;
        } else {
            out.x = u32(x);
            out.y = u32(y);
            out.valid = 1u;
        }
    }
    return out;
}
