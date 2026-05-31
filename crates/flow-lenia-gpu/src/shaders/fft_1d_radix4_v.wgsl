// Flow-Lenia FFT primitive — M6.C-1-2 V-axis (column-stride) variant
// of `fft_1d_radix4.wgsl`.
//
// 1D complex Cooley-Tukey FFT, **radix-4 in-place DIT**, executed
// per-column of an `H × N` row-major complex buffer. One workgroup
// processes one column of N complex samples; the column index is
// `wid.x`, the sample index within the column is `lid.x`. Both load
// and store therefore use the stride `N` (= row width) rather than 1.
//
// The 2D forward path is:
//   real H×N → `fft_1d_radix4` (per-row forward) → complex H×N
//            → `fft_1d_radix4_v` (per-column forward) → complex H×N spectrum
//
// The 2D inverse path is its reverse, with `direction = 1` flipping
// the twiddle conjugation + applying the 1/N normalisation per axis.
//
// V-axis access pattern is **column-stride**: thread `lid.x` reads
// `input[(lid.x * N) + wid.x]`. On M1 G13 each warp's 32 threads
// scatter across `32 × N` byte rows of the storage buffer; this is
// strictly worse than the row-stride pattern in `fft_1d_radix4.wgsl`,
// but transposing instead would require an extra pass + extra
// dispatch (Maczan 2026: 32-71 μs per dispatch on Metal) and a full
// scratch buffer the size of the input. Net dispatch budget for the
// 2D path stays at 2 (one per axis) with this layout, vs 4 with
// transpose+transform pairs. Whether memory bandwidth actually
// dominates is a C-1-3 perf question; correctness in C-1-2 doesn't
// depend on the answer.
//
// Otherwise this file mirrors `fft_1d_radix4.wgsl` exactly — same
// dynamic-N via `override WORKGROUP_X`, same direction flag for
// forward / inverse, same digit-reverse-base-4 input ordering, same
// workgroup-tiled single-dispatch all-stages design. The only deltas
// are (1) the load/store indexing, (2) the input format: V-axis
// inputs are **always complex** (the H-axis forward already produced
// complex output, and the V-axis inverse comes back from the V-axis
// forward), so the `direction == 0` real-load branch is removed.
//
// Layout / binding contract (must agree with `passes/fft.rs`):
//   @binding(0) input:    storage<read>, complex f32 flat (interleaved
//                            as 2 × f32 per complex sample),
//                            input[2 * (row * N + col)] = re,
//                            input[2 * (row * N + col) + 1] = im
//   @binding(1) twiddles: storage<read>, vec2<f32> = (cos θ, -sin θ)
//                            for forward; inverse conjugates in-shader.
//                            Same `precompute_twiddles_1d(n)` table the
//                            H axis uses — single CPU-built table per N.
//   @binding(2) output:   storage<read_write>, vec2<f32> complex flat,
//                            output[row * N + col]
//   @binding(3) params:   uniform<FftParams>{ n, num_rows, direction, _pad }
//                            For the V-axis pass, `num_rows` is the
//                            column count, i.e. the dispatch's
//                            workgroup-grid x-extent: dispatch
//                            (num_cols, 1, 1) workgroups, each
//                            doing one column of `n` samples.

override WORKGROUP_X: u32 = 256u;

struct FftParams {
    n: u32,
    num_rows: u32, // for V axis: column count (workgroup grid extent)
    direction: u32,
    _pad: u32,
};

@group(0) @binding(0) var<storage, read>       input:    array<f32>;
@group(0) @binding(1) var<storage, read>       twiddles: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output:   array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params:   FftParams;

var<workgroup> scratch: array<vec2<f32>, 256>;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

fn complex_mul_i(a: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(-a.y, a.x);
}

// No direction-dependent twiddle conjugation — see fft_1d_radix4.wgsl
// header for why conjugate-twiddle alone is insufficient for radix-4
// inverse. Inverse handles direction via conjugate-load / conjugate-
// store boundaries.

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
fn fft_1d_radix4_v(
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(workgroup_id) wid: vec3<u32>,
) {
    let col = wid.x;
    if (col >= params.num_rows) {
        return;
    }
    let tid = lid.x;
    let n = params.n;
    // M6.C-3-8 follow-up: Chrome strict WGSL needs every workgroupBarrier
    // in uniform control flow. See `fft_1d_radix4.wgsl` for the full
    // rationale; same fix applied here (wrap load/store in `if (in_range)`
    // instead of early-returning on `tid >= n`).
    let in_range = tid < n;

    // Load: digit-reversed-base-4 row index, column-stride access.
    // `input` is bound as `array<f32>` and each complex sample is two
    // consecutive f32, so the byte index for (row, col) complex sample
    // is `2 * (row * N + col)`. Inverse conjugates on load (negate
    // imag) — see fft_1d_radix4.wgsl header for the identity.
    // Same `if`-style as the H-axis sibling for cross-file readability
    // (Round 1 review N2).
    if (in_range) {
        let src_row = digit_reverse_4_dynamic(tid, n);
        let base = 2u * (src_row * n + col);
        if (params.direction == 0u) {
            scratch[tid] = vec2<f32>(input[base], input[base + 1u]);
        } else {
            scratch[tid] = vec2<f32>(input[base], -input[base + 1u]);
        }
    }
    workgroupBarrier();

    // Identical butterfly to the H-axis variant — see
    // `fft_1d_radix4.wgsl` for the radix-4 DIT derivation.
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

    // Store: column-stride; inverse closes the conjugate identity
    // (negate imag) and divides by N.
    if (in_range) {
        let val = scratch[tid];
        let out_idx = tid * n + col;
        if (params.direction == 0u) {
            output[out_idx] = val;
        } else {
            let inv_n = 1.0 / f32(n);
            output[out_idx] = vec2<f32>(val.x, -val.y) * inv_n;
        }
    }
}
