// Flow-Lenia spectral multiply — M6.C-1-3.
//
// Per-pixel complex multiply of an `N × N` input spectrum against
// K pre-FFT'd kernel spectra, producing `K` output spectra. This
// replaces the direct-convolution inner loop in `ConvolvePass` for
// the FFT-based path (`Convolution Theorem`: convolution in the
// spatial domain = element-wise multiplication in the frequency
// domain).
//
// **Layout** (must agree with `passes/spectral_multiply.rs`):
//   @binding(0) input_spectrum:  array<vec2<f32>> length N²,
//                                  row-major `input[row * N + col]`
//   @binding(1) kernel_fft:      array<vec2<f32>> length K*N²,
//                                  `kernel_fft[(k * N + row) * N + col]`
//   @binding(2) output_spectra:  array<vec2<f32>> length K*N²,
//                                  same layout as kernel_fft
//   @binding(3) params:          uniform<SmParams>{ n, k, _pad2 }
//
// **Dispatch shape**: 1D `(workgroups, 1, 1)` with workgroup_size
// `(256, 1, 1)` and one thread per `(k, row, col)` triple. Index
// decoding: `i = k * N² + row * N + col`. Chosen over a 3D
// `(N/wgX, N/wgY, K)` grid for simplicity at this stage —
// performance comparison between layouts is a M6.C-1-5 question
// (per scope-guardian item C in the pre-impl review).
//
// **Reuse opportunity deferred** (Round 1 review #6): with the
// current 1D layout, each cell of `input_spectrum` is fetched K
// times across the full dispatch (once per (k, cell) thread), so
// input-spectrum bandwidth is K× higher than the algorithmic
// minimum. A 2D dispatch `(N²/256, K, 1)` would let a workgroup
// fetch `input_spectrum[cell]` once into a register / shared
// memory and reuse it across the K-axis. C-1-5 perf phase will
// measure whether that wins under the M1 memory subsystem; until
// then, the simpler 1D layout is preferred. Kernel-fft reads have
// no inter-thread reuse (each thread reads one unique cell), so
// no workgroup-memory cache helps that side regardless.

struct SmParams {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       input_spectrum: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read>       kernel_fft:     array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> output_spectra: array<vec2<f32>>;
@group(0) @binding(3) var<uniform>             params:         SmParams;

fn complex_mul(a: vec2<f32>, b: vec2<f32>) -> vec2<f32> {
    return vec2<f32>(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

@compute @workgroup_size(256, 1, 1)
fn spectral_multiply(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let cells_per_kernel = params.n * params.n;
    let total = cells_per_kernel * params.k;
    if (i >= total) {
        return;
    }
    let cell = i % cells_per_kernel;
    let a = input_spectrum[cell];
    let b = kernel_fft[i];
    output_spectra[i] = complex_mul(a, b);
}
