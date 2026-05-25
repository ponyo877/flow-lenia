// Flow-Lenia FFT → pre_g layout transposition — M6.C-1-4.
//
// After the per-kernel inverse 2D FFT, `ConvolveFftPass` has K
// complex spectra of shape `K × N × N` (k-major, vec2<f32>). The
// downstream `affinity_growth` pass expects `pre_g` in
// **cell-major** layout: `pre_g[y * W * K + x * K + ki] = f32`
// (real-valued, see `convolve.wgsl` line 21 binding contract).
//
// This shader takes the .x (real part) of each complex bin and
// writes it to the pre_g layout. Imaginary parts should be ≈ 0
// for a real-valued convolution result and are dropped.
//
// **C=1 only** in M6.C-1-4: this primary path assumes every
// kernel reads from channel 0 of the input activation (i.e.
// `meta_arr[ki].source_channel == 0` for all k). Multi-channel
// support — where different kernels read different source
// channels and `ConvolveFftPass` must run one forward FFT per
// channel and route per-kernel — is deferred to a separate
// sub-step (C-1-5 candidate). Round 1 review M3 correction:
// the standard `tests/m1_regression_gpu.rs` runs at C=3 and is
// NOT the early-exit gate host. The C=1 testbed is
// `tests/diagnose_divergence.rs`; C-1-4-b's scope will name the
// exact perf measurement target.
//
// **Layout** (must agree with `passes/convolve_fft.rs`):
//   @binding(0) input_complex: array<vec2<f32>> length K*N²,
//                                 k-major: input[(k * N + y) * N + x]
//   @binding(1) pre_g_out:     array<f32> length N²*K,
//                                 cell-major: pre_g_out[y * N * K + x * K + ki]
//                                 (N = W = H here)
//   @binding(2) params:        uniform<PreGParams>{ n, k, _pad0, _pad1 }
//
// **Dispatch shape**: 1D `(workgroups, 1, 1)` × workgroup_size 256,
// one thread per (k, y, x) output cell. The index decoding mirrors
// `spectral_multiply.wgsl`'s 1D layout for consistency.

struct PreGParams {
    n: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
};

@group(0) @binding(0) var<storage, read>       input_complex: array<vec2<f32>>;
@group(0) @binding(1) var<storage, read_write> pre_g_out:     array<f32>;
@group(0) @binding(2) var<uniform>             params:        PreGParams;

@compute @workgroup_size(256, 1, 1)
fn fft_complex_to_pre_g(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let cells_per_kernel = params.n * params.n;
    let total = cells_per_kernel * params.k;
    if (i >= total) {
        return;
    }
    // Input layout: input[(k * N + y) * N + x] = vec2 spectrum bin
    // for kernel k at (y, x). Thread index i covers this in (k, y, x)
    // row-major-within-k order.
    let k = i / cells_per_kernel;
    let rem = i % cells_per_kernel;
    let y = rem / params.n;
    let x = rem % params.n;
    // Output layout (matches existing convolve.wgsl pre_g_out):
    //   pre_g[y * (W * K) + x * K + k]
    let out_idx = y * params.n * params.k + x * params.k + k;
    pre_g_out[out_idx] = input_complex[i].x;
}
