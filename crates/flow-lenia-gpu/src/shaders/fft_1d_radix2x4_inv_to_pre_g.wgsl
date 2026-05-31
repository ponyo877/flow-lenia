// Flow-Lenia FFT primitive — M6.C-3-2 mixed-radix (N = 2 × 4^k)
// version of `fft_1d_radix4_inv_to_pre_g.wgsl` (C-2-1-a kernel fusion
// case c).
//
// H-axis **inverse** mixed-radix 1D FFT with the layout transpose
// folded into the output store: takes the V-axis-inverse output
// (complex N×N for kernel ki) and writes the real-valued cell-major
// `pre_g[y * W * K + x * K + ki]` slot in a single dispatch. This is
// the 512-capable sibling of the radix-4 inv-to-pre_g shader; the
// only differences from the radix-4 version are the mixed-radix load
// reversal + the radix-2 combine stage (see `fft_1d_radix2x4.wgsl`
// header for the algorithm). Store path is identical to the radix-4
// inv-to-pre_g shader: Method B close (conjugate + 1/N), drop imag,
// cell-major write index `(row*N + tid)*k_total + ki`.

override WORKGROUP_X: u32 = 512u;

struct InvPreGParams {
    n: u32,
    num_rows: u32,
    ki: u32,
    k_total: u32,
};

@group(0) @binding(0) var<storage, read>       input:     array<f32>;
@group(0) @binding(1) var<storage, read>       twiddles:  array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> pre_g_out: array<f32>;
@group(0) @binding(3) var<uniform>             params:    InvPreGParams;

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
fn fft_1d_radix2x4_inv_to_pre_g(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.num_rows) {
        return;
    }
    let tid = lid.x;
    let n = params.n;
    // M6.C-3-8 follow-up: uniform-CF barrier (Chrome WGSL strict).
    // See `fft_1d_radix4.wgsl` for the full rationale.
    let in_range = tid < n;
    let m = n / 2u;
    let row_base = row * n;

    // Mixed-radix digit reversal + conjugate-load (Method B inverse).
    if (in_range) {
        let p = tid % m;
        let block = tid / m;
        let src = 2u * digit_reverse_4_dynamic(p, m) + block;
        let base = 2u * (row_base + src);
        scratch[tid] = vec2<f32>(input[base], -input[base + 1u]);
    }
    workgroupBarrier();

    // log₄(M) radix-4 DIT stages.
    var stage_size: u32 = 4u;
    while (stage_size <= n) {
        let quarter = stage_size / 4u;
        if (tid < n / 4u) {
            let butterfly_idx = tid;
            let group_idx = butterfly_idx / quarter;
            let local_idx = butterfly_idx % quarter;
            let bbase = group_idx * stage_size + local_idx;

            let twiddle_stride = n / stage_size;
            let w_idx = local_idx * twiddle_stride;
            let w1 = twiddles[w_idx];
            let w2 = twiddles[2u * w_idx];
            let w3 = twiddles[3u * w_idx];

            let p0 = scratch[bbase + 0u * quarter];
            let p1 = scratch[bbase + 1u * quarter];
            let p2 = scratch[bbase + 2u * quarter];
            let p3 = scratch[bbase + 3u * quarter];

            let q1 = complex_mul(p1, w1);
            let q2 = complex_mul(p2, w2);
            let q3 = complex_mul(p3, w3);

            let t0 = p0 + q2;
            let t1 = p0 - q2;
            let t2 = q1 + q3;
            let t3 = complex_mul_i(q1 - q3);

            scratch[bbase + 0u * quarter] = t0 + t2;
            scratch[bbase + 1u * quarter] = t1 - t3;
            scratch[bbase + 2u * quarter] = t0 - t2;
            scratch[bbase + 3u * quarter] = t1 + t3;
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

    // Store: Method B close (conjugate + 1/N) AND drop imag AND
    // transpose to cell-major pre_g layout.
    if (in_range) {
        let val = scratch[tid];
        let inv_n = 1.0 / f32(n);
        let real_result = val.x * inv_n;
        let out_idx = (row_base + tid) * params.k_total + params.ki;
        pre_g_out[out_idx] = real_result;
    }
}
