// Flow-Lenia affinity-growth pass (M2.4) — constant per-kernel
// weights variant (paper Eq. 3, JAX `flowlenia.py:80-90`).
//
//     U_j(x) = Σ_{i : c_i^1 = j}  h_i · G_i(pre_g[x, i])
//
// `pre_g[x, i]` is `(K_i ∗ A_{c_i^0})(x)` computed by `convolve.wgsl`.
// `Meta`, `Globals`, `MAX_CHANNELS`, and `growth_fn` come from
// `types.wgsl` (prepended by the host on pipeline build).
//
// Layout / binding contract (must agree with `passes/affinity_growth.rs`):
//   @binding(0) pre_g:     storage<read>, cell-major (H, W, K)
//                           pre_g[y * W * K + x * K + ki]
//   @binding(1) meta_arr:  storage<read>, array<Meta>
//   @binding(2) h_weights: storage<read>, array<f32>           — h_i (Eq. 3)
//   @binding(3) u_out:     storage<read_write>, channel-major (C, H, W)
//                           u_out[c * H * W + y * W + x]
//   @binding(4) globals:   uniform<Globals>
//
// Workgroup: (8, 8, 1) — 1 invocation = 1 cell, inner K loop
// accumulating into a per-target-channel local array. Output written
// per (cell, target channel).

@group(0) @binding(0) var<storage, read>       pre_g:     array<f32>;
@group(0) @binding(1) var<storage, read>       meta_arr:  array<Meta>;
@group(0) @binding(2) var<storage, read>       h_weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> u_out:     array<f32>;
@group(0) @binding(4) var<uniform>             globals:   Globals;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= globals.w || y >= globals.h) {
        return;
    }

    // Per-(cell, target channel) accumulator. `array<f32, MAX_CHANNELS>`
    // is the only WGSL form for a local fixed-size array — `globals.c`
    // is not a compile-time constant, but the inner loops below clamp
    // to `globals.c < MAX_CHANNELS`, so unused tail slots stay zero
    // and are simply not written back below.
    var u_accum: array<f32, MAX_CHANNELS>;
    for (var cc: u32 = 0u; cc < MAX_CHANNELS; cc = cc + 1u) {
        u_accum[cc] = 0.0;
    }

    let base = y * globals.w * globals.k + x * globals.k;
    for (var ki: u32 = 0u; ki < globals.k; ki = ki + 1u) {
        let pg = pre_g[base + ki];
        let g = growth_fn(pg, meta_arr[ki].mu, meta_arr[ki].sigma);
        let tgt = meta_arr[ki].target_channel;
        u_accum[tgt] = u_accum[tgt] + h_weights[ki] * g;
    }

    let plane = globals.h * globals.w;
    for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
        u_out[cc * plane + y * globals.w + x] = u_accum[cc];
    }
}
