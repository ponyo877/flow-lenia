// Flow-Lenia affinity-growth pass (M2.4) — cell-localised weights
// variant (paper Eq. 7, JAX `flowlenia_params.py:91`).
//
//     U_j(x) = Σ_{i : c_i^1 = j}  P_i(x) · G_i(pre_g[x, i])
//
// Same structure as `affinity_growth_constant.wgsl`, only the
// per-kernel weight comes from a per-cell map instead of a global
// `h_i`. The two shaders share the `types.wgsl` prelude and their
// bind-group layouts are intentionally parallel so a future M5
// painter can switch between variants without re-allocating the
// surrounding buffers.
//
// Layout / binding contract:
//   @binding(0) pre_g:     storage<read>, cell-major (H, W, K)
//   @binding(1) meta_arr:  storage<read>, array<Meta>
//   @binding(2) p_map:     storage<read>, cell-major (H, W, K) — same layout as pre_g
//                           p_map[y * W * K + x * K + ki]
//   @binding(3) u_out:     storage<read_write>, channel-major (C, H, W)
//   @binding(4) globals:   uniform<Globals>

@group(0) @binding(0) var<storage, read>       pre_g:    array<f32>;
@group(0) @binding(1) var<storage, read>       meta_arr: array<Meta>;
@group(0) @binding(2) var<storage, read>       p_map:    array<f32>;
@group(0) @binding(3) var<storage, read_write> u_out:    array<f32>;
@group(0) @binding(4) var<uniform>             globals:  Globals;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= globals.w || y >= globals.h) {
        return;
    }

    var u_accum: array<f32, MAX_CHANNELS>;
    for (var cc: u32 = 0u; cc < MAX_CHANNELS; cc = cc + 1u) {
        u_accum[cc] = 0.0;
    }

    let base = y * globals.w * globals.k + x * globals.k;
    for (var ki: u32 = 0u; ki < globals.k; ki = ki + 1u) {
        let pg = pre_g[base + ki];
        let g = growth_fn(pg, meta_arr[ki].mu, meta_arr[ki].sigma);
        let tgt = meta_arr[ki].target_channel;
        // The only line that differs from the Eq. 3 shader.
        u_accum[tgt] = u_accum[tgt] + p_map[base + ki] * g;
    }

    let plane = globals.h * globals.w;
    for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
        u_out[cc * plane + y * globals.w + x] = u_accum[cc];
    }
}
