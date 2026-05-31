// Flow-Lenia FFT primitive — M6.C-1-1 (forward, N=256) → M6.C-1-2
// (dynamic N ∈ {64, 256}, forward/inverse via direction flag, H axis).
//
// 1D complex Cooley-Tukey FFT, **radix-4 in-place DIT**,
// **workgroup-memory tiled**: one workgroup processes one row of N
// complex samples and runs all log₄(N) butterfly stages inside a
// single dispatch, with `workgroupBarrier()` between stages.
//
// **Dynamic N support (C-1-2)**: `WORKGROUP_X` is a WGSL pipeline-
// override constant; the host sets it to N at pipeline construction
// (see `FftPass::new(ctx, n)`). The same `digit_reverse_4_dynamic`
// works for any N = 4^k (i.e. log₄ N digits in base 4), and the
// `while stage_size <= n` loop walks the log₄(N) stages. N values
// **must be a pure power of 4**: {4, 16, 64, 256, 1024, …}.
// Flow-Lenia grids 32 / 128 / 512 are excluded here (mixed-radix
// needed) — Ponyo877 さん 承認の defer に M6.C-1-2 scope。
//
// **Direction flag (C-1-2)**: `params.direction ∈ {0=forward, 1=inverse}`.
// Inverse re-uses the **same forward butterfly** unchanged, applying
// the standard `IDFT(y) = (1/N) conj(DFT(conj(y)))` identity at the
// load (conjugate input) and store (conjugate + 1/N normalise) boundaries.
//
// A first C-1-2 attempt used "conjugate the twiddle table only" and
// hit a subtle bug: the radix-4 butterfly bakes a `+i*(q1-q3)` term
// (corresponding to W_S^{S/4} = -i in the forward direction); switching
// from W to W^* requires that intermediate to flip to `-i*(q1-q3)` as
// well, otherwise the algorithm computes a position-reversed output
// (`output[k] = N * x[(-k) mod N]` instead of `N * x[k]`). Caught by
// `fft_1d_gpu_inverse_round_trip` at idx=1 — manual N=4 derivation
// matched the wrong answer (`output[1] = 4*x[3]` instead of `4*x[1]`).
// Switching to the conjugate-load / conjugate-store identity avoids
// having to fork the butterfly for direction.
//
// Per-stage dispatch (WebTide pattern) was rejected in C-1-1 because
// Metal per-dispatch overhead is 32-71 μs (Maczan 2026); 1 dispatch
// per axis covers all log₄(N) stages and keeps the 2D path (this
// shader + `fft_1d_radix4_v.wgsl`) at 2 dispatches per direction.
//
// References:
//   - fgiesen 2023, "Notes on FFTs: for implementers"
//   - Lloyd 2008, Microsoft TR-2008-62
//   - docs/M6_literature_survey.md §2 for the design rationale
//     trail (Cooley-Tukey over Stockham, radix-4 over radix-2,
//     workgroup-tiled over per-stage dispatch).
//
// Layout / binding contract (must agree with `passes/fft.rs`):
//   @binding(0) input:    storage<read>, real f32 flat,
//                            input[row * N + i]  for direction=forward,
//                         OR  storage<read>, complex vec2<f32> flat,
//                            input[row * N + i]  for direction=inverse
//   @binding(1) twiddles: storage<read>, vec2<f32> = (cos θ, -sin θ)
//                            for θ = 2π k / N — the **forward** table;
//                            inverse uses conjugate in-shader.
//                            Full N entries (see precompute_twiddles_1d
//                            rustdoc for why N/4 economy fails).
//   @binding(2) output:   storage<read_write>, vec2<f32> complex flat,
//                            output[row * N + i],  natural frequency
//                            order (DC at idx 0).
//                            For direction=inverse, also real-valued
//                            modulo numerical drift; caller may take
//                            `.x` and discard `.y`.
//   @binding(3) params:   uniform<FftParams>{ n, num_rows, direction, _pad }

// Pipeline-override constant: host pins this to N at pipeline build
// (`FftPass::new(ctx, n)`). Must equal `params.n` set in the uniform.
override WORKGROUP_X: u32 = 256u;

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

// Worst-case scratch: N=256 complex × 8 byte = 2 KB. M1 32 KB
// threadgroup memory has ample headroom. WGSL requires this to be
// a fixed-size array; sizing for 256 covers all supported N ≤ 256.
var<workgroup> scratch: array<vec2<f32>, 256>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_mul_i(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.y, a.x);
}

// No direction-dependent twiddle conjugation — the inverse handles
// direction via load/store conjugation only (see file header).

// Dynamic base-4 digit reverse over log₄(n) digits. Walks the
// reversal one base-4 digit at a time until `size` drops to 1. For
// n=256 this runs 4 iterations; for n=64 it runs 3. Branch-free
// inside the loop body; `size` is uniform across the workgroup so
// no divergence cost.
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
fn fft_1d_radix4(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let row = wid.x;
    if (row >= params.num_rows) {
        return;
    }
    let tid = lid.x;
    let n = params.n;
    let row_base = row * n;

    // **Uniform control flow for `workgroupBarrier()`** (Chrome WGSL
    // strict-spec, M6.C-3-8 follow-up fix). An earlier version exited
    // with `if (tid >= n) return;` before the barrier — Metal/Naga
    // accepted that, but Chrome's Tint validator rejects it because
    // `tid = lid.x` is non-uniform across the workgroup and an early
    // return makes the subsequent barrier itself non-uniform. Threads
    // with `tid >= n` (only when WORKGROUP_X > params.n) now skip the
    // load/store but **still reach every barrier**, which is what WGSL
    // requires. In practice WORKGROUP_X == n for the {64, 256} sizes
    // this shader serves, so the guard is dead code; it stays in for
    // spec compliance.
    let in_range = tid < n;

    // Load: bit/digit-reversed input → complex scratch. For
    // direction=forward the input is real (imag = 0); for
    // direction=inverse the input is complex and we **conjugate on
    // load** (negate imag) so the unchanged forward butterfly
    // produces conj(N * IDFT) which the store-side conjugation
    // converts back to N * IDFT, then /N gives IDFT.
    if (in_range) {
        let src = digit_reverse_4_dynamic(tid, n);
        if (params.direction == 0u) {
            scratch[tid] = vec2<f32>(input[row_base + src], 0.0);
        } else {
            let base = 2u * (row_base + src);
            scratch[tid] = vec2<f32>(input[base], -input[base + 1u]);
        }
    }
    workgroupBarrier();

    // log₄(n) stages: stage_size grows 4 → 16 → … → n.
    // Each stage has n/4 butterflies; only the first n/4 threads
    // run a butterfly. Per-stage idle (75 % at the largest sizes
    // active here) is documented in C-1-1 rustdoc as a design choice.
    var stage_size: u32 = 4u;
    while (stage_size <= n) {
        let quarter = stage_size / 4u;

        if (tid < n / 4u) {
            let butterfly_idx = tid;
            let group_idx = butterfly_idx / quarter;
            let local_idx = butterfly_idx % quarter;
            let base = group_idx * stage_size + local_idx;

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

    // Store: 1 thread = 1 complex output sample. For direction=inverse
    // conjugate the butterfly output (close the identity `IDFT(y) =
    // (1/N) conj(DFT(conj(y)))`) and normalise by 1/N (rustfft
    // convention: forward unnormalised, inverse divides by N).
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
