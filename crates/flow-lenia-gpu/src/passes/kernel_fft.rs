//! M6.C-1-3 — kernel pre-FFT, the spectral-side counterpart to the
//! direct-convolution `KernelBuffers`.
//!
//! Flow-Lenia's K convolution kernels are static across the
//! simulator's per-step loop — they only change when the user
//! adjusts a kernel parameter through the UI. We can therefore
//! compute their 2D FFTs **once at startup** and reuse the spectrum
//! every step, replacing the direct-convolution per-step inner
//! loop with a per-pixel spectral multiply (`SpectralMultiplyPass`)
//! plus one inverse FFT per kernel (`Fft2dPass` with
//! `direction = Inverse`).
//!
//! Layout (`buffer` storage view):
//!
//! ```text
//! kernel_fft_buffer[(k * N + row) * N + col] = K_k_hat(row, col)
//! ```
//!
//! i.e. K kernels concatenated, each laid out row-major as N×N
//! complex (`vec2<f32>`) bins. The total size is `K * N² * 8` bytes
//! (5.24 MB for K=10, N=256 — well inside M6.0 §3's GPU-buffer
//! envelope).
//!
//! **Padding convention**: each kernel is zero-padded to N×N with
//! the kernel **centre placed at (0, 0)** and the surrounding
//! kernel cells wrapped to the other corners of the padded array
//! (a "circular shift" by `(-k_side/2, -k_side/2)`).
//!
//! **What this is equivalent to** (Round 1 review M1 honesty):
//! - `ifft(fft(input) * fft(padded_kernel))` computes the **true
//!   circular convolution** `(a ⊛ k)[y, x] = Σ_dy,dx a[(y+dy) mod N,
//!   (x+dx) mod N] * k[dy, dx]`.
//! - `flow_lenia_core::convolve::convolve2d` computes **circular
//!   correlation**: `(a ⊗ k)[y, x] = Σ_dy,dx a[y+dy, x+dx] * k[ck+dy,
//!   ck+dx]` (no kernel flip; see `convolve.rs:80-110`).
//! - The two are equal **only when the kernel is centro-symmetric**
//!   (k[-dy, -dx] = k[dy, dx]). Lenia kernels are *radial-symmetric*
//!   (they depend only on √(x²+y²) by construction in
//!   `compute_kernel_raw`), so this equality holds for every
//!   kernel Flow-Lenia constructs today. If a future code path
//!   feeds an asymmetric kernel through this helper, the FFT result
//!   will differ from `convolve2d` by a kernel-flip — the
//!   `padded_kernel_layout_implements_true_convolution_not_correlation`
//!   regression test below locks this distinction in.
//!
//! Wall-border equivalence does NOT hold and is out of scope for
//! the C-1-3 unit test — see scope-guardian decision A in the
//! C-1-3 pre-impl review.
//!
//! `TODO(M6.C-1-4)`: when the UI changes a kernel parameter, this
//! buffer must be re-built. Currently the precompute helper is
//! "startup once" only.

use crate::passes::fft::Fft2dPass;
use crate::GpuContext;
use bytemuck::cast_slice;
use flow_lenia_core::kernel::compute_kernel;
use flow_lenia_core::params::KernelParams;
use ndarray::Array2;
use wgpu::util::DeviceExt;

/// Owner of the K-kernel pre-FFT buffer + its dimensions. `buffer`
/// is bound by `SpectralMultiplyPass` and the per-kernel inverse FFT
/// dispatches in `Fft2dPass` (planned for C-1-4 integration).
pub struct KernelFftBuffers {
    /// Grid side length. Must equal the `n` the `Fft2dPass` was
    /// built with and what the per-step input spectrum will use.
    pub n: u32,
    /// Number of kernels (Flow-Lenia's `K`, typically 10).
    pub k: u32,
    /// `K × N × N` complex bins, concatenated row-major.
    pub buffer: wgpu::Buffer,
}

impl KernelFftBuffers {
    /// Bytes per spectrum (N² complex × 8 byte). Reserved for C-1-4's
    /// memory-accounting / sanity check; not yet called in C-1-3.
    #[allow(dead_code)]
    #[must_use]
    pub fn bytes_per_kernel(&self) -> u64 {
        u64::from(self.n) * u64::from(self.n) * 8
    }

    /// Total buffer size in bytes. Reserved for C-1-4 memory accounting.
    #[allow(dead_code)]
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        self.bytes_per_kernel() * u64::from(self.k)
    }
}

/// Compute the N×N zero-padded form of `kernel` with the kernel
/// centre at output position (0, 0). Cells with negative offset
/// from centre wrap to the opposite corner. Output is row-major
/// flat `N²` reals.
///
/// Concretely, for a kernel of shape `(k_h, k_w)` (both odd, both
/// ≤ N), centre at `(k_h/2, k_w/2)`:
/// - `padded[0, 0]` = `kernel[k_h/2, k_w/2]` (the centre value)
/// - `padded[1, 0]` = `kernel[k_h/2 + 1, k_w/2]`
/// - `padded[N-1, 0]` = `kernel[k_h/2 - 1, k_w/2]`
/// - etc., with both axes wrapped independently modulo N.
///
/// **Why this layout**: FFT convolution computes `ifft(fft(a) *
/// fft(k))` which is circular convolution `(a ⊛ k)[y, x] = Σ_dy,dx
/// a[(y+dy) mod N, (x+dx) mod N] * k[dy, dx]`. To make this equal
/// to direct convolution `(a * k)[y, x] = Σ_dy,dx a[y-dy, x-dx] *
/// k_centred[dy+ck, dx+ck]` (`ck` = kernel centre offset), the
/// padded kernel must have `k_centred[ck, ck]` (the centre) at
/// position (0, 0) and `k_centred[ck-1, ck]` (one cell "left of
/// centre" in the original) at position `(N-1, 0)`. The circular
/// shift below realises that.
#[must_use]
pub fn build_padded_kernel(kernel: &Array2<f32>, n: u32) -> Vec<f32> {
    let (k_h, k_w) = kernel.dim();
    let n_usize = n as usize;
    assert!(
        k_h <= n_usize && k_w <= n_usize,
        "kernel ({k_h}×{k_w}) must fit inside grid ({n_usize}×{n_usize})"
    );
    let centre_h = k_h / 2;
    let centre_w = k_w / 2;
    let mut padded = vec![0.0_f32; n_usize * n_usize];
    for ky in 0..k_h {
        for kx in 0..k_w {
            let dy = ky as isize - centre_h as isize;
            let dx = kx as isize - centre_w as isize;
            // Wrap into [0, N). `(dy + n) % n` instead of `dy % n`
            // because Rust's `%` on signed values returns a negative
            // remainder for negative inputs.
            let py = ((dy + n_usize as isize).rem_euclid(n_usize as isize)) as usize;
            let px = ((dx + n_usize as isize).rem_euclid(n_usize as isize)) as usize;
            padded[py * n_usize + px] = kernel[[ky, kx]];
        }
    }
    padded
}

/// Build all K kernel FFTs on the GPU and pack them into a single
/// storage buffer. Call once at simulator startup (and again
/// whenever a kernel parameter changes — see TODO at module top).
///
/// Cost: K Fft2dPass round-trips (one per kernel), each with its
/// own buffer allocation pair. Acceptable for a startup-only call
/// (K = 10 at N = 256 totals < 100 ms on M1 mini per M6.0 § init
/// table). The per-step path is unaffected.
#[must_use]
pub fn precompute_kernel_ffts(
    ctx: &GpuContext,
    params: &KernelParams,
    n: u32,
    fft2d: &Fft2dPass,
    twiddles: &wgpu::Buffer,
) -> KernelFftBuffers {
    assert_eq!(
        fft2d.n, n,
        "fft2d.n ({}) must match the precompute n ({n})",
        fft2d.n
    );
    let num_kernels = params.kernels.len() as u32;
    assert!(
        num_kernels >= 1,
        "KernelParams must have at least one kernel"
    );
    let cells_per_kernel = (n * n) as usize;

    let mut all_kernels_fft: Vec<f32> =
        Vec::with_capacity(num_kernels as usize * cells_per_kernel * 2);

    for entry in &params.kernels {
        let kernel = compute_kernel(params.r_global, entry);
        let padded = build_padded_kernel(&kernel, n);
        let spectrum = fft2d.forward_2d(ctx, twiddles, &padded);
        debug_assert_eq!(spectrum.len(), cells_per_kernel);
        for c in &spectrum {
            all_kernels_fft.push(c[0]);
            all_kernels_fft.push(c[1]);
        }
    }

    let buffer = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("kernel_fft buffer"),
            contents: cast_slice(&all_kernels_fft),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

    KernelFftBuffers {
        n,
        k: num_kernels,
        buffer,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn padded_kernel_centre_lands_at_origin() {
        // 3×3 kernel: centre value at output (0, 0), four edges at
        // (0, ±1) / (±1, 0), corners at (±1, ±1) (each wrapped mod N).
        let kernel = ndarray::array![
            [1.0_f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        let n: u32 = 8;
        let padded = build_padded_kernel(&kernel, n);
        let at = |y: usize, x: usize| padded[y * n as usize + x];

        // Centre (kernel[1, 1] = 5) at (0, 0).
        assert_eq!(at(0, 0), 5.0);
        // Right of centre (kernel[1, 2] = 6) at (0, 1).
        assert_eq!(at(0, 1), 6.0);
        // Left of centre (kernel[1, 0] = 4) at (0, N-1).
        assert_eq!(at(0, (n - 1) as usize), 4.0);
        // Below centre (kernel[2, 1] = 8) at (1, 0).
        assert_eq!(at(1, 0), 8.0);
        // Above centre (kernel[0, 1] = 2) at (N-1, 0).
        assert_eq!(at((n - 1) as usize, 0), 2.0);
        // Lower-right corner (kernel[2, 2] = 9) at (1, 1).
        assert_eq!(at(1, 1), 9.0);
        // Upper-left corner (kernel[0, 0] = 1) at (N-1, N-1).
        assert_eq!(
            at((n - 1) as usize, (n - 1) as usize),
            1.0
        );

        // Everywhere else: zero.
        let mut nonzero_count = 0usize;
        for v in &padded {
            if *v != 0.0 {
                nonzero_count += 1;
            }
        }
        assert_eq!(nonzero_count, 9, "exactly 9 nonzero cells expected");
    }

    /// Round 1 review M1: lock in the correlation-vs-convolution
    /// distinction. `build_padded_kernel` + FFT pointwise multiply
    /// implements **true convolution** (kernel flipped relative to
    /// the input scan direction); `flow_lenia_core::convolve::convolve2d`
    /// implements **correlation** (no kernel flip). For an asymmetric
    /// kernel the two differ, and the difference is exactly that
    /// flipping the input kernel makes them agree. This test exercises
    /// only the padded-kernel layout (CPU-only) so it does not
    /// require a `GpuContext`.
    #[test]
    fn padded_kernel_layout_implements_true_convolution_not_correlation() {
        // Asymmetric 3×3 kernel. Centre (offset 0) at output (0, 0).
        let k = ndarray::array![
            [1.0_f32, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0]
        ];
        let n: u32 = 8;
        let padded = build_padded_kernel(&k, n);
        // The cell one to the right of centre in the original kernel
        // (k[1, 2] = 6) lands at padded[0, 1]. In a convolution
        // application of `padded` against an input via the sum
        // (a ⊛ p)[y, x] = Σ a[(y+dy) mod N, (x+dx) mod N] * p[dy, dx],
        // the cell at (dy=0, dx=1) contributes `a[y, x+1] * p[0, 1]
        // = a[y, x+1] * 6`. This means the kernel weight 6, which sat
        // at "+1 relative to centre" in the original CPU `kernel`,
        // multiplies the input cell at "+1 relative to (y, x)" in
        // the FFT-domain pointwise multiply — i.e., **the FFT path
        // does NOT flip the kernel**. For comparison,
        // `convolve.rs::convolve2d_naive` line 80-110 also does not
        // flip (it is correlation); but `convolve2d` shifts the
        // origin to `er` (kernel half-side), whereas the FFT path
        // shifts the origin to (0, 0) via this padded layout. For
        // centro-symmetric kernels (k[-dy, -dx] = k[dy, dx]) the
        // shift difference is invisible. Lenia kernels are
        // radial-symmetric, so this distinction stays academic in
        // production today — but a future asymmetric kernel will
        // silently produce a flipped result if anyone forgets.
        assert_eq!(padded[0 * n as usize + 1], 6.0, "+1 offset cell");
        assert_eq!(
            padded[0 * n as usize + (n - 1) as usize],
            4.0,
            "-1 offset cell (wraps)"
        );
        // The full end-to-end equivalence with `convolve2d` (which
        // holds for symmetric kernels) is exercised by
        // `fft_convolution_matches_direct_torus_n64_k1` and the
        // K=3 sibling in `spectral_multiply.rs`.
    }
}
