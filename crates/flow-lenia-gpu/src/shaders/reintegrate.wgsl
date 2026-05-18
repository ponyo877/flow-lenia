// Flow-Lenia reintegration tracking (M2.7) — paper Eq. 6.
//
//     A^{t+dt}_i(x)  =  Σ_{x' ∈ N(x)}  A^t_i(x') · I_i(x', x)
//     I_i(x', x)     =  ∫_{Ω(x)} D(x' + dt·F^t_i(x'), σ) dA
//
// Receiver-side loop: each invocation accumulates contributions from
// every source cell `x'` in its Chebyshev `dd` neighbourhood.
//
// CPU reference: `flow_lenia_core::reintegrate::reintegrate` (M1.11).
//
// ──────────────────────────────────────────────────────────────────
// Three correctness-critical points (M1.11 documented these in detail
// after the CPU implementation; the WGSL port hits the same traps):
//
// 1. **`ma = dd − σ` clip on `dt·F`** — without this, an oversized
//    `dt·F` can push `μ` outside the `dd` neighbourhood and the
//    contribution silently disappears, breaking mass conservation
//    (JAX `reintegration_tracking.py:48`).
//
// 2. **Wall μ-clip** — under `BORDER_WALL` the distribution centre
//    `μ = src + dt·F` is clamped to `[σ, side − σ]` so the entire
//    distribution stays inside the grid (JAX
//    `reintegration_tracking.py:64-65`). Slightly non-conservative
//    at the boundary; M1.11 measured ~1e-2 worst-case.
//
// 3. **`dpmu` in LOGICAL (unwrapped) coordinates** — both `μ` and
//    `target_centre` are computed in unwrapped source-cell space:
//    `μ = (src_x_unwrapped + 0.5) + dt·F`, then
//    `dpmu = target_centre − μ`. The `dd` loop handles wrap by
//    iterating signed offsets; there is no min-modulo trick needed
//    because the dy/dx outer offsets already span the wrap-around
//    contributors. The wrapped index (`src.x`, `src.y`) is used only
//    to read `A` and `F` from the buffer; it never enters the
//    distance arithmetic.
// ──────────────────────────────────────────────────────────────────
//
// Layout / binding contract (must agree with `passes/reintegrate.rs`):
//   @binding(0) a_in:    storage<read>,        channel-major (C, H, W)
//   @binding(1) flow:    storage<read>,        (C, H, W, 2) axis-flow inner
//   @binding(2) a_out:   storage<read_write>,  channel-major (C, H, W)
//   @binding(3) globals: uniform<Globals>
//
// Globals consumed: h, w, c, border, dd, sigma, dt.

@group(0) @binding(0) var<storage, read>       a_in:    array<f32>;
@group(0) @binding(1) var<storage, read>       flow:    array<f32>;
@group(0) @binding(2) var<storage, read_write> a_out:   array<f32>;
@group(0) @binding(3) var<uniform>             globals: Globals;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= globals.w || gid.y >= globals.h) {
        return;
    }
    let target_x = i32(gid.x);
    let target_y = i32(gid.y);
    let w_i = i32(globals.w);
    let h_i = i32(globals.h);
    let plane: u32 = globals.h * globals.w;

    let sigma = globals.sigma;
    let dt = globals.dt;
    let dd_i = i32(globals.dd);
    let ma = f32(dd_i) - sigma; // flow-amplitude clip (point 1)

    let target_cx = f32(target_x) + 0.5;
    let target_cy = f32(target_y) + 0.5;

    // Per-target-channel accumulator. Fixed cap at MAX_CHANNELS = 3
    // (DESIGN.md §7); inner loop stops at globals.c so unused tail
    // slots stay at 0 and are not written below.
    var accum: array<f32, MAX_CHANNELS>;
    for (var cc: u32 = 0u; cc < MAX_CHANNELS; cc = cc + 1u) {
        accum[cc] = 0.0;
    }

    for (var dy: i32 = -dd_i; dy <= dd_i; dy = dy + 1) {
        for (var dx: i32 = -dd_i; dx <= dd_i; dx = dx + 1) {
            // Logical (unwrapped) source coordinates — used for dpmu
            // arithmetic only.
            let src_x_unwrapped = target_x + dx;
            let src_y_unwrapped = target_y + dy;

            // Wrapped/clipped source index — used to read A and F.
            let src = border_resolve(src_x_unwrapped, src_y_unwrapped, w_i, h_i, globals.border);
            if (src.valid == 0u) {
                // Wall: source outside grid contributes 0. Skip rather
                // than write 0 reads (those reads would still touch
                // memory inefficiently).
                continue;
            }

            // Source-centre in logical coordinates.
            let src_cx = f32(src_x_unwrapped) + 0.5;
            let src_cy = f32(src_y_unwrapped) + 0.5;

            for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
                // F at the wrapped/clipped source.
                let f_base = cc * plane * 2u + src.y * globals.w * 2u + src.x * 2u;
                let f_y_raw = flow[f_base + 0u];
                let f_x_raw = flow[f_base + 1u];

                // Point (1): clamp dt·F to [-ma, +ma].
                let dtf_y = clamp(dt * f_y_raw, -ma, ma);
                let dtf_x = clamp(dt * f_x_raw, -ma, ma);

                // μ in LOGICAL source coordinates.
                var mu_y = src_cy + dtf_y;
                var mu_x = src_cx + dtf_x;

                // Point (2): wall μ-clip.
                if (globals.border == BORDER_WALL) {
                    mu_x = clamp(mu_x, sigma, f32(globals.w) - sigma);
                    mu_y = clamp(mu_y, sigma, f32(globals.h) - sigma);
                }

                // Point (3): single subtraction in logical coords.
                let dpmu_y = target_cy - mu_y;
                let dpmu_x = target_cx - mu_x;

                let area = overlap_area(dpmu_y, dpmu_x, sigma);

                // A at the wrapped/clipped source.
                let a_idx = cc * plane + src.y * globals.w + src.x;
                accum[cc] = accum[cc] + a_in[a_idx] * area;
            }
        }
    }

    for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
        let out_idx = cc * plane + gid.y * globals.w + gid.x;
        a_out[out_idx] = accum[cc];
    }
}
