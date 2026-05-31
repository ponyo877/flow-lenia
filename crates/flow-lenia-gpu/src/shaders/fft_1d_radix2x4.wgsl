// Flow-Lenia FFT primitive — M6.C-3-1 mixed-radix FFT for
// N = 2 × 4^k (i.e. N ∈ {8, 32, 128, 512}). H-axis, forward + inverse.
//
// **Why this shader exists**: `fft_1d_radix4.wgsl` only handles pure
// powers of 4 (N ∈ {64, 256}). The final goal is 512×512 (512 = 2^9,
// NOT a power of 4: 4^4=256, 4^5=1024). 512 = 2 × 256 = 2 × 4^4, so a
// single radix-2 stage on top of the existing radix-4 machinery
// completes it. Generalised to N = 2 × 4^k so N=8 (= 2 × 4) is
// testable as a small cross-check of the radix-2 stage.
//
// **Decomposition (radix-2 DIT, even/odd outermost, radix-4 inner)**:
//
//   N = 2M,  M = 4^k.  DIT radix-2:  X[k] = E[k] + W_N^k O[k],
//                                    X[k+M] = E[k] - W_N^k O[k]
//   where E = M-point DFT of even-indexed inputs x[0],x[2],…
//         O = M-point DFT of odd-indexed inputs  x[1],x[3],…
//
//   - **Load** (digit reversal): scratch[0..M) ← evens in base-4
//     reversed order, scratch[M..2M) ← odds. For thread tid:
//       p = tid mod M,  block = tid / M (0=even, 1=odd)
//       src = 2 · digit_reverse_4(p, M) + block
//     so scratch[tid] = input[src].
//   - **4^k radix-4 stages** (stage_size 4→16→…→M): the EXISTING
//     radix-4 DIT butterfly, run with n=N. The loop `stage_size <= N`
//     naturally stops at M (next is 4M > N=2M), and at stage_size=M
//     the n/4 = M/2 butterflies split into group_idx ∈ {0,1}, keeping
//     the two M-blocks independent → block 0 becomes E, block 1 O.
//     Twiddle: w_idx = local_idx · (N/stage_size); at the top stage
//     stride = N/M = 2, so twiddles[2·local_idx] = W_N^{2 local_idx}
//     = W_M^{local_idx} — exactly the M-point DFT twiddles, taken for
//     free from the N-entry forward table.
//   - **1 radix-2 combine stage**: threads tid < M compute
//       X[k]   = E[k] + W_N^k · O[k]
//       X[k+M] = E[k] - W_N^k · O[k]
//     with W_N^k = twiddles[k] (forward table).
//
// **Direction** is handled exactly like `fft_1d_radix4.wgsl`: Method B
// inverse `IDFT(y) = (1/N) conj(DFT(conj(y)))` — conjugate-load,
// run the forward butterflies + radix-2 combine UNCHANGED (forward
// twiddles), conjugate-store + 1/N. No direction-dependent twiddle
// anywhere (the C-1-2 position-reversal bug from conjugate-twiddle is
// thereby avoided here too).
//
// Binding contract: identical to `fft_1d_radix4.wgsl`
//   @binding(0) input    storage<read>  real f32 (fwd) / complex vec2 (inv)
//   @binding(1) twiddles storage<read>  vec2<f32> W_N^k, full N entries
//   @binding(2) output   storage<read_write> vec2<f32> complex, natural order
//   @binding(3) params   uniform FftParams { n, num_rows, direction, _pad }

override WORKGROUP_X: u32 = 512u;

struct FftParams {
    n: u32,
    num_rows: u32,
    direction: u32, // 0 = forward, 1 = inverse
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       input:    array<f32>;
@group(0) @binding(1) var<storage, read>       twiddles: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output:   array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params:   FftParams;

// Worst-case scratch: N=512 complex × 8 byte = 4 KB (M1 32 KB
// threadgroup memory has ample headroom).
var<workgroup> scratch: array<vec2<f32>, 512>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_mul_i(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.y, a.x);
}

// Base-4 digit reverse over log₄(m) digits (m must be a power of 4).
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
fn fft_1d_radix2x4(
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
    let m = n / 2u; // M = 4^k, the radix-4 sub-DFT size
    let row_base = row * n;

    // Load with mixed-radix digit reversal:
    //   p = tid mod M, block = tid / M (0=even, 1=odd)
    //   src = 2·digit_reverse_4(p, M) + block
    if (in_range) {
        let p = tid % m;
        let block = tid / m;
        let src = 2u * digit_reverse_4_dynamic(p, m) + block;
        if (params.direction == 0u) {
            scratch[tid] = vec2<f32>(input[row_base + src], 0.0);
        } else {
            let base = 2u * (row_base + src);
            scratch[tid] = vec2<f32>(input[base], -input[base + 1u]);
        }
    }
    workgroupBarrier();

    // log₄(M) radix-4 DIT stages. Identical butterfly to
    // fft_1d_radix4.wgsl; the `stage_size <= n` loop runs 4→…→M and
    // at stage_size=M splits into 2 independent M-blocks (E, O).
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

    // Radix-2 combine (final, largest-stride stage). After the radix-4
    // stages: scratch[0..M) = E, scratch[M..2M) = O. Combine with
    // W_N^k = twiddles[k].
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

    // Store: natural frequency order. Inverse closes Method B
    // (conjugate + 1/N).
    if (in_range) {
        let val = scratch[tid];
        if (params.direction == 0u) {
            output[row_base + tid] = val;
        } else {
            let inv_n = 1.0 / f32(n);
            output[row_base + tid] = vec2<f32>(val.x, -val.y) * inv_n;
        }
    }
}
