// Flow-Lenia spectral multiply — M6.C-1-3 (C=1) → M6.C-1-5-a (C>=1).
//
// Per-pixel complex multiply of `C` input channel spectra against
// `K` pre-FFT'd kernel spectra, producing `K` output spectra. For
// each kernel `k`, the source channel `src_c = kernel_routing[k]`
// selects which channel-spectrum to multiply against (mirroring
// `meta_arr[ki].source_channel` from the direct `convolve.wgsl`
// path).
//
// **Convolution Theorem** with multi-channel routing:
// ```
// pre_g[y, x, k] = (K_k ∗ A_{src_c})[y, x]
//                = ifft(fft(A_{src_c}) ⊙ fft(K_k))[y, x]
// ```
//
// **Layout** (must agree with `passes/spectral_multiply.rs` and
// `passes/convolve_fft.rs`):
//   @binding(0) input_spectra:    array<vec2<f32>> length C*N²,
//                                   channel-major:
//                                   `input_spectra[(c * N + row) * N + col]`
//                                   = `A_c`'s spectrum at (row, col).
//                                   For C=1 this collapses to a single
//                                   N² spectrum at offset 0.
//   @binding(1) kernel_fft:       array<vec2<f32>> length K*N²,
//                                   `kernel_fft[(k * N + row) * N + col]`
//   @binding(2) output_spectra:   array<vec2<f32>> length K*N²,
//                                   `output[k * N² + cell]` = K spectra
//   @binding(3) kernel_routing:   array<u32> length K,
//                                   `kernel_routing[k]` = source channel
//                                   index ∈ [0, C) for kernel k.
//                                   For C=1 every entry is 0.
//   @binding(4) params:           uniform<SmParams>{ n, k, c, _pad }
//
// **Dispatch shape (M6.C-2-2 vec4 packing)**: 1D `(workgroups, 1, 1)`
// with workgroup_size `(256, 1, 1)` and **2 (k, row, col) triples
// per thread** (= 4 f32 = 1 vec4 worth of SIMD work). Index decoding:
// `i_base = gid.x * 2`, then the two cells at `i_base` and `i_base+1`
// are processed in a `for (var di: u32 = 0u; di < 2u; ...)` loop.
// The unroll is small enough that Naga reliably emits paired f32
// FMAs on the Metal target — M1 G13's 32-wide SIMD then issues both
// complex multiplies in one warp cycle for the active half of the
// workgroup. Compared to the C-1-3 1-cell-per-thread layout this
// halves the dispatch grid size + halves the per-thread bind-group
// + workgroup-id arithmetic overhead.
//
// Pre-C-2-2 layout was "1 thread = 1 complex multiply"; the change
// is binding-layout-compatible (same `array<vec2<f32>>` storage
// view, no stride change) so existing tests pass without
// modification beyond the dispatch-count helper. Larger unroll
// (e.g. 4 cells / thread) was considered and rejected for C-2-2
// because the per-thread register pressure starts to hurt M1
// occupancy at 4× — measure first in C-2-5 before pushing further.
//
// **Reuse opportunity deferred** (C-1-3 Round 1 review #6): with the
// 1D layout, each `(src_c, cell)` of `input_spectra` is fetched
// once per kernel that routes to that channel. For C-1-5+ perf
// phase, a 2D dispatch `(N²/256, K, 1)` would let a workgroup
// fetch its row of input cells once per K-axis sweep.

struct SmParams {
    n: u32,
    k: u32,
    c: u32,
    _pad0: u32,
};

@group(0) @binding(0) var<storage, read>       input_spectra:  array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       kernel_fft:     array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output_spectra: array<vec2<f32>>;
@group(0) @binding(3) var<storage, read>       kernel_routing: array<u32>;
@group(0) @binding(4) var<uniform>             params:         SmParams;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256, 1, 1)
fn spectral_multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i_base = gid.x * 2u;
    let cells_per_kernel = params.n * params.n;
    let total = cells_per_kernel * params.k;
    // 2-cell-per-thread unroll (C-2-2). The two iterations share
    // `cells_per_kernel`/`total` loads and let the M1 G13 SIMD lane
    // pair the two complex multiplies into one warp issue.
    for (var di: u32 = 0u; di < 2u; di = di + 1u) {
        let i = i_base + di;
        if (i >= total) {
            return;
        }
        let k = i / cells_per_kernel;
        let cell = i % cells_per_kernel;
        let src_c = kernel_routing[k];
        let a_idx = src_c * cells_per_kernel + cell;
        let a = input_spectra[a_idx];
        let b = kernel_fft[i];
        output_spectra[i] = complex_mul(a, b);
    }
}
