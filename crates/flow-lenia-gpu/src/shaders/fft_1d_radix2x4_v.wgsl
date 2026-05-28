// Flow-Lenia FFT primitive — M6.C-3-2 V-axis (column-stride) variant
// of `fft_1d_radix2x4.wgsl` (mixed-radix N = 2 × 4^k).
//
// Mirrors `fft_1d_radix4_v.wgsl` exactly except for the mixed-radix
// load reversal + the radix-2 combine stage that `fft_1d_radix2x4.wgsl`
// (the H-axis sibling) introduced. See that file's header for the
// full algorithm derivation. The only V-specific deltas vs the H
// mixed shader are (1) column-stride load/store, (2) the input is
// always complex (V-axis inputs come from an H-axis forward or a
// V-axis forward), so there is no real-load branch.
//
// Decomposition: N = 2M, M = 4^k. scratch[0..M) ← even-indexed rows
// (base-4 reversed), scratch[M..2M) ← odd-indexed rows; 4^k radix-4
// stages produce E (block 0) and O (block 1); a final radix-2 combine
// yields X[k] = E[k] + W_N^k O[k], X[k+M] = E[k] - W_N^k O[k].
//
// Binding contract: identical to `fft_1d_radix4_v.wgsl`
//   @binding(0) input    storage<read>  complex f32 interleaved,
//                          input[2*(row*N+col)] = re, +1 = im
//   @binding(1) twiddles storage<read>  vec2<f32> W_N^k (full N)
//   @binding(2) output   storage<read_write> vec2<f32> complex,
//                          output[row*N+col]
//   @binding(3) params   uniform FftParams { n, num_rows(=col count),
//                          direction, _pad }

override WORKGROUP_X: u32 = 512u;

struct FftParams {
    n: u32,
    num_rows: u32, // V axis: column count (workgroup grid extent)
    direction: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       input:    array<f32>;
@group(0) @binding(1) var<storage, read>       twiddles: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output:   array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params:   FftParams;

var<workgroup> scratch: array<vec2<f32>, 512>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_mul_i(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.y, a.x);
}

fn digit_reverse_4_dynamic(i: u32, m: u32) -> u32 {
    var x = i;
    var r: u32 = 0u;
    var size: u32 = m;
    while (size > 1u) {
        r = (r << 2u) | (x & 3u);
        x = x >> 2u;
        size = size >> 2u;
    }
    return r;
}

@compute @workgroup_size(WORKGROUP_X, 1, 1)
fn fft_1d_radix2x4_v(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = wid.x;
    if (col >= params.num_rows) {
        return;
    }
    let tid = lid.x;
    let n = params.n;
    if (tid >= n) {
        return;
    }
    let m = n / 2u;

    // Mixed-radix digit reversal, column-stride load. V-axis input is
    // always complex; inverse conjugates on load (Method B).
    let p = tid % m;
    let block = tid / m;
    let src_row = 2u * digit_reverse_4_dynamic(p, m) + block;
    let base = 2u * (src_row * n + col);
    if (params.direction == 0u) {
        scratch[tid] = vec2<f32>(input[base], input[base + 1u]);
    } else {
        scratch[tid] = vec2<f32>(input[base], -input[base + 1u]);
    }
    workgroupBarrier();

    // log₄(M) radix-4 DIT stages (identical butterfly to the H mixed
    // sibling).
    var stage_size: u32 = 4u;
    while (stage_size <= n) {
        let quarter = stage_size / 4u;
        if (tid < n / 4u) {
            let butterfly_idx = tid;
            let group_idx = butterfly_idx / quarter;
            let local_idx = butterfly_idx % quarter;
            let scratch_base = group_idx * stage_size + local_idx;

            let twiddle_stride = n / stage_size;
            let w_idx = local_idx * twiddle_stride;
            let w1 = twiddles[w_idx];
            let w2 = twiddles[2u * w_idx];
            let w3 = twiddles[3u * w_idx];

            let p0 = scratch[scratch_base + 0u * quarter];
            let p1 = scratch[scratch_base + 1u * quarter];
            let p2 = scratch[scratch_base + 2u * quarter];
            let p3 = scratch[scratch_base + 3u * quarter];

            let q1 = complex_mul(p1, w1);
            let q2 = complex_mul(p2, w2);
            let q3 = complex_mul(p3, w3);

            let t0 = p0 + q2;
            let t1 = p0 - q2;
            let t2 = q1 + q3;
            let t3 = complex_mul_i(q1 - q3);

            scratch[scratch_base + 0u * quarter] = t0 + t2;
            scratch[scratch_base + 1u * quarter] = t1 - t3;
            scratch[scratch_base + 2u * quarter] = t0 - t2;
            scratch[scratch_base + 3u * quarter] = t1 + t3;
        }
        workgroupBarrier();
        stage_size = stage_size * 4u;
    }

    // Radix-2 combine.
    if (tid < m) {
        let k = tid;
        let e = scratch[k];
        let o = scratch[k + m];
        let w = twiddles[k];
        let wo = complex_mul(o, w);
        scratch[k] = e + wo;
        scratch[k + m] = e - wo;
    }
    workgroupBarrier();

    // Store: column-stride; inverse closes Method B (conjugate + 1/N).
    let val = scratch[tid];
    let out_idx = tid * n + col;
    if (params.direction == 0u) {
        output[out_idx] = val;
    } else {
        let inv_n = 1.0 / f32(n);
        output[out_idx] = vec2<f32>(val.x, -val.y) * inv_n;
    }
}
