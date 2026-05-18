// Flow-Lenia visualisation render pass (M2.9).
//
// Renders the activation field A directly from the channel-major
// storage buffer onto a colour target, without any intermediate
// texture. Channel → RGB mapping:
//
//   C = 1 → R only      (G = B = 0)
//   C = 2 → R + G       (B = 0)
//   C ≥ 3 → R + G + B
//
// Up-scaling is "nearest neighbour": each grid cell occupies an
// `upscale × upscale` block of fragments. Pixel-perfect; no smoothing.
// Gamma correction is left to a future UI step (M4).
//
// Vertex stage emits a single triangle that covers the entire viewport
// using the "oversize triangle" trick (3 vertices at (-1, -1),
// (3, -1), (-1, 3)) so we don't need a vertex / index buffer.

// Local globals (separate from the compute-side Globals struct so the
// visualization pass can be reused on a buffer that came from any
// pipeline configuration).
struct VisualizeGlobals {
    h: u32,
    w: u32,
    c: u32,
    upscale: u32,
};

@group(0) @binding(0) var<storage, read>  a_in:    array<f32>;
@group(0) @binding(1) var<uniform>        globals: VisualizeGlobals;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Oversize-triangle clip-space positions:
    //   i = 0 → (-1, -1)
    //   i = 1 → ( 3, -1)
    //   i = 2 → (-1,  3)
    let x = f32(i & 1u) * 4.0 - 1.0;
    let y = f32((i >> 1u) & 1u) * 4.0 - 1.0;
    return vec4<f32>(x, y, 0.0, 1.0);
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    // `frag_coord.xy` is in pixel units (`[0, target_w) × [0, target_h)`,
    // top-left origin for WebGPU).
    let x_px = u32(frag_coord.x);
    let y_px = u32(frag_coord.y);

    // Nearest-neighbour upscale: each `upscale × upscale` block of
    // pixels maps to one grid cell.
    let x_cell = x_px / globals.upscale;
    let y_cell = y_px / globals.upscale;

    // Guard against any fragment that lands outside the logical grid
    // (e.g. if the target size is slightly bigger than
    // grid × upscale). Wgpu rasterises whole pixels, so this is rare,
    // but cheap to defend.
    if (x_cell >= globals.w || y_cell >= globals.h) {
        return vec4<f32>(0.0, 0.0, 0.0, 1.0);
    }

    let plane: u32 = globals.h * globals.w;
    let base = y_cell * globals.w + x_cell;

    // Channels are read defensively — `c < 1` is impossible by
    // contract (the simulator requires `channels > 0`), but the C=2
    // and C=3 branches guard against out-of-range reads on smaller
    // configurations.
    let r = a_in[base];
    var g: f32 = 0.0;
    var b: f32 = 0.0;
    if (globals.c >= 2u) {
        g = a_in[plane + base];
    }
    if (globals.c >= 3u) {
        b = a_in[plane * 2u + base];
    }

    return vec4<f32>(
        clamp(r, 0.0, 1.0),
        clamp(g, 0.0, 1.0),
        clamp(b, 0.0, 1.0),
        1.0,
    );
}
