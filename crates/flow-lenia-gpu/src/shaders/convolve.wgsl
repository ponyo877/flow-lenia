// Flow-Lenia convolve compute shader (M2.3)
//
// Computes the per-(cell, kernel) convolution that feeds M2.4's
// growth-and-aggregation pass:
//
//     pre_g[y, x, ki] = (K_i ∗ A_{c_i^0})(x, y)
//
// References:
//   - Paper Eq. 7 / Eq. 3 convolution part.
//   - JAX `flowlenia.py:82-86` (FFT path, equivalent for radially-symmetric K).
//   - CPU reference: `flow_lenia_core::convolve::convolve2d` (M1.6).
//
// Layout / binding contract (must agree with `passes/convolve.rs`):
//   @binding(0) a_in:      storage<read>, channel-major flat:
//                           a_in[c * H * W + y * W + x]
//   @binding(1) kernels:   storage<read>, zero-padded packed (M2.2 Plan A):
//                           kernels[ki * max_side² + ky * max_side + kx]
//   @binding(2) meta_arr:  storage<read>, array<Meta> (runtime-sized) — see M2.3
//                           note in kernel_buffers.rs about UNIFORM|STORAGE.
//   @binding(3) pre_g_out: storage<read_write>, cell-major flat:
//                           pre_g_out[y * W * K + x * K + ki]
//   @binding(4) globals:   uniform<Globals>
//
// Workgroup design: (8, 8, 1) → 1 invocation = 1 cell × inner K loop.
// Rationale (M2.3 design judgment): K varies per `cfg.num_kernels`
// without pipeline rebuild; bind-group management is simpler; WGSL
// compiles the inner K loop tightly. M6 may revisit if K dominates.

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
    _pad: u32,
};

const BORDER_TORUS: u32 = 0u;
const BORDER_WALL: u32 = 1u;

@group(0) @binding(0) var<storage, read>       a_in:      array<f32>;
@group(0) @binding(1) var<storage, read>       kernels:   array<f32>;
@group(0) @binding(2) var<storage, read>       meta_arr:  array<Meta>;
@group(0) @binding(3) var<storage, read_write> pre_g_out: array<f32>;
@group(0) @binding(4) var<uniform>             globals:   Globals;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let x = gid.x;
    let y = gid.y;
    if (x >= globals.w || y >= globals.h) {
        return;
    }

    let w_i = i32(globals.w);
    let h_i = i32(globals.h);
    let half = i32(globals.half_side);
    let max_side = i32(globals.max_side);
    let kernel_stride: u32 = u32(max_side) * u32(max_side);
    let plane: u32 = globals.h * globals.w;

    for (var ki: u32 = 0u; ki < globals.k; ki = ki + 1u) {
        let src_c = meta_arr[ki].source_channel;
        var acc: f32 = 0.0;

        for (var dy: i32 = -half; dy <= half; dy = dy + 1) {
            for (var dx: i32 = -half; dx <= half; dx = dx + 1) {
                var sx: i32 = i32(x) + dx;
                var sy: i32 = i32(y) + dy;
                var contribute: bool = true;

                if (globals.border == BORDER_TORUS) {
                    // Mathematical modulo (always non-negative).
                    sx = ((sx % w_i) + w_i) % w_i;
                    sy = ((sy % h_i) + h_i) % h_i;
                } else {
                    // Wall: zero outside grid.
                    if (sx < 0 || sx >= w_i || sy < 0 || sy >= h_i) {
                        contribute = false;
                    }
                }

                if (contribute) {
                    let a_idx = src_c * plane + u32(sy) * globals.w + u32(sx);
                    let ky_idx = u32(dy + half);
                    let kx_idx = u32(dx + half);
                    let k_idx = ki * kernel_stride
                        + ky_idx * u32(max_side)
                        + kx_idx;
                    acc = acc + a_in[a_idx] * kernels[k_idx];
                }
            }
        }

        // Cell-major output: pre_g[y, x, ki].
        let out_idx = u32(y) * globals.w * globals.k + u32(x) * globals.k + ki;
        pre_g_out[out_idx] = acc;
    }
}
