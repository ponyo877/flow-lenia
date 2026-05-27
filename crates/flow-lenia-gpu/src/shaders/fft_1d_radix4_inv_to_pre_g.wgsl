// Flow-Lenia FFT primitive — M6.C-2-1-a kernel fusion case c.
//
// H-axis inverse 1D FFT (radix-4, workgroup-tiled) **with the
// final layout transpose folded into the output store**: instead
// of writing complex `output[row * N + col] = vec2<f32>` and then
// running a separate `FftToPreGPass` to transpose K complex spectra
// into cell-major real `pre_g[y * W * K + x * K + ki]`, this entry
// function does the inverse FFT and the layout transpose in one
// dispatch.
//
// Per kernel inverse path (previously, see ConvolveFftPass C-1-5-a):
//
// ```
// for ki in 0..K {
//     copy k_spectra[ki slice] → inv_in
//     inverse_2d (V → scratch → H → spectrum_staging)
//     copy spectrum_staging → k_complex_out[ki slice]
// }
// FftToPreGPass: k_complex_out → pre_g_out  (1 extra dispatch)
// ```
//
// Per kernel inverse path (C-2-1-a, this file):
//
// ```
// for ki in 0..K {
//     copy k_spectra[ki slice] → inv_in
//     V-axis inverse → scratch_complex
//     H-axis inverse-to-pre_g (this shader, params.ki=ki) → pre_g_out  ← real, cell-major
// }
// // no per-kernel copy_buffer_to_buffer to k_complex_out
// // no separate transpose dispatch
// ```
//
// **Dispatch count saved at K=10**: 11
// (10 × copy_buffer_to_buffer to k_complex_out + 1 transpose dispatch).
// On Metal at 32–71 μs/dispatch (Maczan 2026 arXiv:2604.02344) this
// is **350–780 μs/step** savings — directly contributing to the
// C-2 target 1.5–2× end-to-end ratio. Measured in C-2-5.
//
// **Algorithm**: identical to `fft_1d_radix4.wgsl`'s H-axis inverse
// path. The only differences are:
// 1. `output` binding is `array<f32>` (real, cell-major) instead
//    of `array<vec2<f32>>` (complex).
// 2. `params` has additional `ki` and `k_total` u32 fields for the
//    pre_g layout `pre_g[y * W * K + x * K + ki]`.
// 3. The final store drops the imag part (which is ≈ 0 modulo the
//    chaos drift documented in BENCH §14) and writes a single f32
//    per cell.
//
// The radix-4 butterfly + workgroup-memory tiling + digit-reverse-
// base-4 load + Method B inverse (`IDFT(y) = (1/N) conj(DFT(conj(y)))`)
// are all identical to the C-1-2 implementation. Duplicated rather
// than refactored because the binding contract is different
// (output type + params shape) and a single shader with two output
// modes via a uniform flag would force every existing inverse caller
// to redundantly bind `pre_g_out` even when not used. C-2-5 perf
// phase will decide if the duplication is worth absorbing back into
// `fft_1d_radix4.wgsl` once the dispatch-fusion gain is verified.

override WORKGROUP_X: u32 = 256u;

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

var<workgroup> scratch: array<vec2<f32>, 256>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_mul_i(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.y, a.x);
}

fn digit_reverse_4_dynamic(i: u32, n: u32) -> u32 {
    var x = i;
    var r: u32 = 0u;
    var size: u32 = n;
    while (size > 1u) {
        r = (r << 2u) | (x & 3u);
        x = x >> 2u;
        size = size >> 2u;
    }
    return r;
}

@compute @workgroup_size(WORKGROUP_X, 1, 1)
fn fft_1d_radix4_inv_to_pre_g(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.num_rows) {
        return;
    }
    let tid = lid.x;
    let n = params.n;
    if (tid >= n) {
        return;
    }
    let row_base = row * n;

    // Load: digit-reversed conjugate (Method B inverse, see
    // fft_1d_radix4.wgsl header for the identity proof).
    let src = digit_reverse_4_dynamic(tid, n);
    let base = 2u * (row_base + src);
    scratch[tid] = vec2<f32>(input[base], -input[base + 1u]);
    workgroupBarrier();

    // log_4(n) radix-4 butterfly stages — identical to C-1-2 inverse.
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

    // Store: Method B close (conjugate + 1/N normalize) AND drop imag,
    // AND transpose to cell-major pre_g layout. The cell-major write
    // index is `(row * n + tid) * k_total + ki` — one f32 per cell
    // for the per-kernel slice ki.
    let val = scratch[tid];
    let inv_n = 1.0 / f32(n);
    let real_result = val.x * inv_n;
    let out_idx = (row_base + tid) * params.k_total + params.ki;
    pre_g_out[out_idx] = real_result;
}
