// Flow-Lenia Sobel on `A_Σ = Σ_c A_c` (M2.5).
//
// CPU reference: `flow_lenia_core::sobel::grad_a_sum` (M1.9), which
// itself is `sobel(A.sum(axis=2))`. Here `A_Σ` is computed on the fly
// inside the Sobel inner loop — with `C ≤ 3` and a 3×3 stencil the
// total per-cell read count is `≤ 27`, so a pre-pass for `A_Σ` would
// add more global traffic than it saves.
//
// Output layout (matches M2.5 design): axis-flow innermost.
//
//     grad_a_sum_out[y * W * 2 + x * 2 + flow_axis]
//
// flow_axis ∈ {0 = dy, 1 = dx}. Same convention as
// `gradient_u.wgsl`.

@group(0) @binding(0) var<storage, read>       a_in:            array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_a_sum_out:  array<f32>;
@group(0) @binding(2) var<uniform>             globals:         Globals;

// Sum across channels at one cell. Border-resolved already, so this
// just iterates over `globals.c` slabs.
fn sample_a_sum_at(x: u32, y: u32) -> f32 {
    var s: f32 = 0.0;
    let plane: u32 = globals.h * globals.w;
    for (var cc: u32 = 0u; cc < globals.c; cc = cc + 1u) {
        s = s + a_in[cc * plane + y * globals.w + x];
    }
    return s;
}

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= globals.w || gid.y >= globals.h) {
        return;
    }
    let xi = i32(gid.x);
    let yi = i32(gid.y);
    let w_i = i32(globals.w);
    let h_i = i32(globals.h);

    var grad_y: f32 = 0.0;
    var grad_x: f32 = 0.0;

    for (var dy: i32 = -1; dy <= 1; dy = dy + 1) {
        for (var dx: i32 = -1; dx <= 1; dx = dx + 1) {
            let sample = border_resolve(xi + dx, yi + dy, w_i, h_i, globals.border);
            if (sample.valid == 0u) {
                continue;
            }
            let a_sum_val = sample_a_sum_at(sample.x, sample.y);
            let coef_x = f32(dx) * select(1.0, 2.0, dy == 0);
            let coef_y = f32(dy) * select(1.0, 2.0, dx == 0);
            grad_x = grad_x + a_sum_val * coef_x;
            grad_y = grad_y + a_sum_val * coef_y;
        }
    }

    let base = gid.y * globals.w * 2u + gid.x * 2u;
    grad_a_sum_out[base + 0u] = grad_y; // FLOW_DY = 0
    grad_a_sum_out[base + 1u] = grad_x; // FLOW_DX = 1
}
