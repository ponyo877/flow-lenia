// Flow-Lenia FFT primitive — M6.C-1-1.
//
// 1D complex Cooley-Tukey FFT, **radix-4 in-place DIT**, fixed
// N = 256, **workgroup-memory tiled**: one workgroup processes one
// row of 256 complex samples and runs all log₄(256) = 4 butterfly
// stages inside a single dispatch, with `workgroupBarrier()` between
// stages. This deliberately avoids the per-stage-dispatch pattern
// (see WebTide ocean implementation analysed in `docs/M6_literature_
// survey.md §2.3`); on Metal the per-dispatch API cost is 32-71 μs
// (Maczan 2026, arXiv:2604.02344), so 4 stages × `num_rows` dispatches
// per axis would eat the 60-FPS frame budget by itself.
//
// Real-input optimisation (RFFT packing N=256 real → N/2=128 complex)
// is **not** in C-1-1 scope: 128 = 2 × 64 is not a clean radix-4 size
// (would need a mixed-radix tail stage). C-1-1 keeps the input fully
// complex (imag = 0 on load); RFFT-packed variant is deferred to a
// later sub-step once the dispatch shape stabilises. The 2× memory /
// 2× compute cost is acceptable for a primitive that has zero callers
// inside the simulator step until C-1-4.
//
// References:
//   - fgiesen, "Notes on FFTs: for implementers" (2023): radix-4 over
//     radix-2 for fewer twiddle reads, Cooley-Tukey in-place over
//     Stockham ping-pong to keep working set in L1 (here: 32 KB
//     threadgroup memory).
//   - Lloyd 2008, "Fast computation of general Fourier Transforms on
//     GPUs", Microsoft TR-2008-62.
//   - WebTide WGSL FFT (BarthPaleologue) as a negative example of
//     per-stage dispatch — survey §2.3.
//
// Layout / binding contract (must agree with `passes/fft.rs`):
//   @binding(0) input:    storage<read>, real f32 flat:
//                            input[row * N + i],  i ∈ [0, 256)
//   @binding(1) twiddles: storage<read>, vec2<f32> = (cos θ, -sin θ),
//                            full N entries indexed in [0, N). The
//                            butterfly needs W_N^{k}, W_N^{2k},
//                            W_N^{3k}; `2k` and `3k` reach up to ~3N/4
//                            so the "first-quadrant only" economy
//                            doesn't fit a 3-twiddle butterfly without
//                            folding logic — see precompute_twiddles_1d
//                            rustdoc for the trade-off.
//   @binding(2) output:   storage<read_write>, vec2<f32> complex flat:
//                            output[row * N + i],  i ∈ [0, 256)
//                            **natural frequency order** (DC, f1, …,
//                            fN-1); for real input the bins
//                            (N/2+1 .. N-1) are the complex conjugate
//                            of (N/2-1 .. 1) — downstream consumers
//                            may exploit or ignore this symmetry.
//   @binding(3) params:   uniform<FftParams>{ n: u32 = 256, num_rows: u32 }
//                            `n` is asserted == 256 by
//                            `FftPass::upload_params` for C-1-1
//                            (callers constructing their own uniform
//                            buffer bypass this gate); C-1-2 will
//                            generalise the WGSL to runtime-N and
//                            the assertion lifts with it.
//
// Workgroup design: (256, 1, 1) = 1 invocation per complex sample,
// 256 threads per row, dispatched (num_rows, 1, 1) workgroups. Each
// stage `stage_size ∈ {4, 16, 64, 256}` has `N/4 = 64` active
// butterflies — only the first 64 threads do work each stage; the
// other 192 threads idle through the stage barrier. This wastes
// occupancy by design: keeping the load step "1 thread = 1 sample"
// matches the natural shape of input/output, and the stage idle is
// cheap compared to either a stride-quartering load or extra
// barrier-sync work.
//
// Bit / digit-reversal: input is digit-reversed in **base 4** on
// load (DIT convention), so the in-place butterflies produce
// natural-ordered output. log₄(256) = 4 digits, so the reversal is
// 4 lookup-free arithmetic shifts; no precomputed table.
//
// Threadgroup memory: 256 × vec2<f32> = 2 KB scratch, well inside
// the M1 G13 32 KB threadgroup-memory budget (survey §5.2). No
// register-pressure concerns at this size on M1's 208 KiB/threadgroup
// register file.

struct FftParams {
    n: u32,
    num_rows: u32,
};

@group(0) @binding(0) var<storage, read>       input:    array<f32>;
@group(0) @binding(1) var<storage, read>       twiddles: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output:   array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params:   FftParams;

var<workgroup> scratch: array<vec2<f32>, 256>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    // (a.x + i a.y)(b.x + i b.y) = (ax bx - ay by) + i (ax by + ay bx)
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// i * (x + iy) = -y + ix — radix-4 needs this 90° rotation on the
// (q1 - q3) intermediate; inlining it as one swap+negate beats a
// runtime twiddle multiply by (0, 1).
fn complex_mul_i(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.y, a.x);
}

// Base-4 digit reverse for the 4-digit (= log₄ 256) index. Cheaper
// than a precomputed table at this size; reads are uniform across
// threads of the workgroup so there is no divergence cost.
fn digit_reverse_4(i: u32) -> u32 {
    let d0 = i & 3u;
    let d1 = (i >> 2u) & 3u;
    let d2 = (i >> 4u) & 3u;
    let d3 = (i >> 6u) & 3u;
    return (d0 << 6u) | (d1 << 4u) | (d2 << 2u) | d3;
}

@compute @workgroup_size(256, 1, 1)
fn fft_1d_radix4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.num_rows) {
        return;
    }
    let tid = lid.x;
    let n = params.n; // == 256 by FftPass::upload_params precondition
    let row_base = row * n;

    // Load: real input → complex scratch (imag = 0), digit-reversed
    // base 4. After this load + barrier the in-place butterflies
    // produce natural-ordered output.
    let src = digit_reverse_4(tid);
    scratch[tid] = vec2<f32>(input[row_base + src], 0.0);
    workgroupBarrier();

    // log₄(256) = 4 stages: stage_size grows 4 → 16 → 64 → 256.
    // Each stage has N/4 = 64 butterflies, each combining 4 samples
    // spaced by `quarter = stage_size / 4` inside their group of
    // `stage_size` consecutive elements.
    var stage_size: u32 = 4u;
    while (stage_size <= n) {
        let quarter = stage_size / 4u;

        // Only the first 64 threads (= N/4) run a butterfly each
        // stage; threads ∈ [64, 256) idle through this stage's
        // barrier. See the workgroup-design rationale at the top of
        // the file: keeping load/store as "1 thread = 1 sample" is
        // worth the per-stage idle.
        if (tid < n / 4u) {
            let butterfly_idx = tid;
            let group_idx = butterfly_idx / quarter;
            let local_idx = butterfly_idx % quarter;
            let base = group_idx * stage_size + local_idx;

            // Twiddle stride: stage `s` uses W_N^{k * (N / stage_size)}
            // for k ∈ [0, stage_size/4). The twiddle buffer holds
            // W_N^k for the **full** k ∈ [0, N) (256 entries for N=256)
            // — see `precompute_twiddles_1d` rustdoc for why the
            // first-quadrant-only economy was rejected (2*w_idx and
            // 3*w_idx reach up to ~3N/4 and would OOB-read 0).
            let twiddle_stride = n / stage_size;
            let w_idx = local_idx * twiddle_stride;
            let w1 = twiddles[w_idx];
            let w2 = twiddles[2u * w_idx];
            let w3 = twiddles[3u * w_idx];

            let p0 = scratch[base + 0u * quarter];
            let p1 = scratch[base + 1u * quarter];
            let p2 = scratch[base + 2u * quarter];
            let p3 = scratch[base + 3u * quarter];

            let q1 = complex_mul(p1, w1);
            let q2 = complex_mul(p2, w2);
            let q3 = complex_mul(p3, w3);

            // Radix-4 DIT butterfly (matches Cooley-Tukey "decimation
            // in time" sign convention `W = exp(-2πi/N)`):
            //   t0 = p0 + q2,  t1 = p0 - q2,  t2 = q1 + q3,
            //   t3 = i * (q1 - q3)
            //   out[base + 0*q] = t0 + t2
            //   out[base + 1*q] = t1 - t3
            //   out[base + 2*q] = t0 - t2
            //   out[base + 3*q] = t1 + t3
            // Sanity check: at the final stage (stage_size = n) the
            // four outputs sit at base, base+n/4, base+n/2, base+3n/4
            // which is the natural-ordered frequency layout.
            let t0 = p0 + q2;
            let t1 = p0 - q2;
            let t2 = q1 + q3;
            let t3 = complex_mul_i(q1 - q3);

            scratch[base + 0u * quarter] = t0 + t2;
            scratch[base + 1u * quarter] = t1 - t3;
            scratch[base + 2u * quarter] = t0 - t2;
            scratch[base + 3u * quarter] = t1 + t3;
        }
        workgroupBarrier();
        stage_size = stage_size * 4u;
    }

    // Store: 1 thread = 1 complex output sample. Natural frequency
    // order (DC at index 0, increasing frequency thereafter, with the
    // real-input conjugate-symmetric upper half implicit).
    output[row_base + tid] = scratch[tid];
}
