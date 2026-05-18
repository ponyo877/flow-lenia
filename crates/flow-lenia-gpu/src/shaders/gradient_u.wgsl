// Flow-Lenia per-channel Sobel for ∇U (M2.5).
//
// CPU reference: `flow_lenia_core::sobel::sobel_per_channel` (M1.9).
// JAX reference: `utils.py:16-37` (`sobel_x` / `sobel_y` / `sobel`).
//
// Sobel kernel — **correlation form** (rotated 180° from the standard
// `convolve` form, matching M1.7 design):
//
//     kx_corr = [[-1,  0,  1],
//                [-2,  0,  2],
//                [-1,  0,  1]]
//     ky_corr = kx_corr.T
//
// For each `(dy, dx) ∈ {-1, 0, 1}²`:
//     coef_x = dx * (2 if dy == 0 else 1)
//     coef_y = dy * (2 if dx == 0 else 1)
//
// `select(false_v, true_v, cond)` is WGSL's branchless ternary
// (note the argument order — false first, true second).
//
// Output layout (axis-flow innermost, see DESIGN.md §3 + M2.5
// design judgment):
//
//     grad_u_out[c * H * W * 2 + y * W * 2 + x * 2 + flow_axis]
//
// flow_axis ∈ {0 = dy, 1 = dx}, mirroring M1.9 `FLOW_DY` / `FLOW_DX`.

@group(0) @binding(0) var<storage, read>       a_in:        array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_u_out:  array<f32>;
@group(0) @binding(2) var<uniform>             globals:     Globals;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= globals.w || gid.y >= globals.h) {
        return;
    }
    let xi = i32(gid.x);
    let yi = i32(gid.y);
    let w_i = i32(globals.w);
    let h_i = i32(globals.h);
    let plane: u32 = globals.h * globals.w;

    for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
        var grad_y: f32 = 0.0;
        var grad_x: f32 = 0.0;
        let channel_base = cc * plane;

        for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
            for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
                let sample = border_resolve(xi + dx, yi + dy, w_i, h_i, globals.border);
                if (sample.valid == 0u) {
                    continue;
                }
                let a_val = a_in[channel_base + sample.y * globals.w + sample.x];
                let coef_x = f32(dx) * select(1.0, 2.0, dy == 0);
                let coef_y = f32(dy) * select(1.0, 2.0, dx == 0);
                grad_x = grad_x + a_val * coef_x;
                grad_y = grad_y + a_val * coef_y;
            }
        }

        let base = cc * plane * 2u + gid.y * globals.w * 2u + gid.x * 2u;
        grad_u_out[base + 0u] = grad_y; // FLOW_DY = 0
        grad_u_out[base + 1u] = grad_x; // FLOW_DX = 1
    }
}
