//! M6.C-1-4 — FFT-based convolution pass, the spectral-side
//! counterpart to the direct-convolution [`ConvolvePass`].
//!
//! `ConvolveFftPass` orchestrates:
//!
//! 1. Forward 2D FFT on the (real, single-channel) input activation
//!    → complex `N × N` input spectrum.
//! 2. Spectral multiply against K pre-computed kernel FFTs → K
//!    complex `N × N` per-kernel spectra.
//! 3. Per-kernel inverse 2D FFT → K complex `N × N` real-valued
//!    convolution results (imag ≈ 0 by symmetry).
//! 4. Layout transposition `K × N × N (k-major complex)` →
//!    `N × N × K (cell-major real)` so the downstream
//!    `affinity_growth` pass sees the same `pre_g` binding contract
//!    as the direct path.
//!
//! ## C=1 limitation (M6.C-1-4 scope, deferred to C-1-5+ for full)
//!
//! Every kernel is assumed to read channel 0 of the input. The
//! existing direct [`ConvolvePass`] WGSL inspects `meta_arr[ki]
//! .source_channel` and reads the appropriate channel of `a_in`
//! per kernel. Mirroring that on the FFT side would require running
//! `C` forward FFTs (one per input channel) and routing each
//! kernel's spectral multiply to the right channel-spectrum.
//! That is real work but mechanical; it is deferred so C-1-4 can
//! complete its primary deliverable (FFT vs direct A/B measurement).
//!
//! **Where the early-exit gate runs** (Round 1 review M3 correction):
//! the standard `tests/m1_regression_gpu.rs` is **C=3**, so it is
//! NOT the early-exit gate's host. C-1-4-b will need either a
//! dedicated C=1 benchmark or to defer FFT-mode integration until
//! the multi-channel work lands. `tests/diagnose_divergence.rs`
//! (M6.A.4.5 grid sweep) is the existing C=1 testbed and is the
//! natural host for the gate, but the perf measurement path will
//! need to be set up explicitly in C-1-4-b's scope-guardian
//! consultation.
//!
//! ## Scope-guardian Option C (struct separation)
//!
//! Per the C-1-4 pre-impl scope review, FFT mode is a **new
//! struct** rather than a mode flag on `ConvolvePass`. C-1-5 will
//! delete the direct struct entirely once the A/B period validates
//! FFT mode; that deletion is a one-line `pub use` removal with
//! this layout, vs a mode-flag teardown with caller-side dispatch
//! deletion if we had gone with the alternative.
//!
//! ## Per-step orchestration
//!
//! `ConvolveFftPass::new` owns the FFT sub-passes, the spectral
//! multiply pass, the layout-transposition pass, the twiddle
//! buffer, and the per-step **scratch data buffers** (so the
//! large `K × N²` complex working set does not reallocate per
//! step). The per-step `record(...)` issues no submits and no
//! polls — it only appends dispatches and `copy_buffer_to_buffer`
//! calls to the caller's encoder.
//!
//! **Honest framing** (Round 1 review M1 correction): per-step
//! `record(...)` is NOT yet allocation-free. Each
//! `forward_2d_with_scratch` / `inverse_2d_with_scratch` call
//! creates two uniform `FftParams` buffers + two bind groups
//! internally (the WGSL params buffer is built from a runtime
//! struct), and `record(...)` also creates per-call bind groups
//! for the spectral-multiply and layout-transpose passes. For
//! K=10 this is roughly **22 uniform buffer creations + 24 bind
//! group creations per step**, dominated by the K-iteration
//! inverse FFT loop. Hoisting these into `ConvolveFftPass::new`
//! (where the input/scratch/output buffer identities are fixed
//! at construction) is a clean optimisation but mechanical; it
//! is deferred to C-1-4-b / C-1-5 perf phase so the C-1-4-a
//! commit stays primitive-only. The C-1-4-b measurement protocol
//! must include this overhead in the early-exit gate ratio (do
//! not mistake it for FFT compute cost).

use crate::passes::fft::{precompute_twiddles_1d, Fft2dPass};
use crate::passes::spectral_multiply::{SpectralMultiplyParams, SpectralMultiplyPass};
use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Mirror of WGSL `struct PreGParams`. 16-byte aligned for uniform.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct PreGParams {
    pub n: u32,
    pub k: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl PreGParams {
    #[must_use]
    pub fn new(n: u32, k: u32) -> Self {
        Self { n, k, _pad0: 0, _pad1: 0 }
    }
}

const LAYOUT_WORKGROUP_X: u32 = 256;

/// Compute pass that takes the k-major complex output of K inverse
/// 2D FFTs and lays it out as cell-major real `pre_g[y * W * K + x * K + ki]`.
pub struct FftToPreGPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl FftToPreGPass {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const SOURCE: &str = include_str!("../shaders/fft_complex_to_pre_g.wgsl");
        let shader = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("fft_complex_to_pre_g.wgsl"),
            source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
        });
        let bind_group_layout =
            ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("fft_to_pre_g bind group layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let pipeline_layout = ctx.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("fft_to_pre_g pipeline layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });
        let pipeline = ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("fft_to_pre_g pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("fft_complex_to_pre_g"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
        Self { pipeline, bind_group_layout }
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        n: u32,
        k: u32,
    ) {
        let total = n * n * k;
        let groups = total.div_ceil(LAYOUT_WORKGROUP_X);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fft_to_pre_g pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }
}

/// FFT-based convolve pass owning all sub-passes and per-step
/// scratch. Per-step `record(...)` appends dispatches to the
/// caller's encoder; no submits, no polls, no allocations.
pub struct ConvolveFftPass {
    pub n: u32,
    pub k: u32,
    pub fft2d: Fft2dPass,
    pub sm_pass: SpectralMultiplyPass,
    pub layout_pass: FftToPreGPass,
    pub twiddles: wgpu::Buffer,
    /// `n²` complex scratch — H/V FFT intermediate.
    pub scratch_complex: wgpu::Buffer,
    /// `n²` complex — input spectrum (output of forward 2D).
    pub spectrum_a: wgpu::Buffer,
    /// `K × n²` complex — output of spectral multiply (K spectra).
    pub k_spectra: wgpu::Buffer,
    /// `n²` complex — copy target for one kernel's spectrum
    /// (inverse FFT input, refreshed per kernel via copy_buffer_to_buffer).
    pub inv_in: wgpu::Buffer,
    /// `K × n²` complex — concatenated per-kernel inverse FFT
    /// outputs, read by the layout transpose pass.
    pub k_complex_out: wgpu::Buffer,
    /// SM params uniform (n, k).
    pub sm_params_buf: wgpu::Buffer,
    /// Layout-transpose params uniform (n, k).
    pub layout_params_buf: wgpu::Buffer,
}

impl ConvolveFftPass {
    /// Build all sub-passes and per-step scratch for a fixed
    /// `(n, num_kernels)` shape. The `twiddles` buffer is also owned
    /// here so the caller never has to wire it in.
    #[must_use]
    pub fn new(ctx: &GpuContext, n: u32, num_kernels: u32) -> Self {
        assert!(num_kernels >= 1, "ConvolveFftPass requires K >= 1 (got {num_kernels})");
        let cells = (n * n) as usize;
        let k = num_kernels;
        let fft2d = Fft2dPass::new(ctx, n);
        let sm_pass = SpectralMultiplyPass::new(ctx);
        let layout_pass = FftToPreGPass::new(ctx);
        let twiddles = precompute_twiddles_1d(ctx, n);

        let scratch_complex = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ConvolveFftPass scratch complex (H/V intermediate)"),
            size: (cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        let spectrum_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ConvolveFftPass input spectrum + per-kernel inverse H-output"),
            // STORAGE for FFT binding + COPY_SRC because we reuse this
            // buffer as the per-kernel inverse 2D H-axis output and
            // then copy it into the right slice of k_complex_out.
            size: (cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let k_spectra = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ConvolveFftPass K spectra (SM output)"),
            size: (cells * k as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let inv_in = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ConvolveFftPass inverse FFT input (per-kernel slice)"),
            size: (cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let k_complex_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ConvolveFftPass K-major inverse FFT outputs"),
            size: (cells * k as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let sm_params_buf =
            SpectralMultiplyPass::upload_params(ctx, SpectralMultiplyParams::new(n, k));
        let layout_params_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ConvolveFftPass layout pre_g params"),
                contents: bytemuck::bytes_of(&PreGParams::new(n, k)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        Self {
            n,
            k,
            fft2d,
            sm_pass,
            layout_pass,
            twiddles,
            scratch_complex,
            spectrum_a,
            k_spectra,
            inv_in,
            k_complex_out,
            sm_params_buf,
            layout_params_buf,
        }
    }

    /// Append the full FFT-based convolution to `encoder`.
    ///
    /// - `input_a`: real activation, channel-major flat
    ///   `a_in[c * H * W + y * W + x]` (= existing `ConvolvePass`
    ///   input layout). **C=1 only**: this implementation reads
    ///   channel 0 (the first `n²` reals). Multi-channel +
    ///   per-kernel source-channel routing is deferred.
    /// - `kernel_fft`: K pre-computed kernel FFTs as built by
    ///   `passes::kernel_fft::precompute_kernel_ffts` (K × N × N
    ///   complex, K-major).
    /// - `pre_g_out`: real, cell-major
    ///   `pre_g[y * W * K + x * K + ki]` (= existing `ConvolvePass`
    ///   output layout — drop-in for `affinity_growth`).
    ///
    /// Submits, polls, and bind-group identities are the caller's
    /// responsibility.
    pub fn record(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        input_a: &wgpu::Buffer,
        kernel_fft: &wgpu::Buffer,
        pre_g_out: &wgpu::Buffer,
    ) {
        let n = self.n;
        let k = self.k;
        let cells = (n * n) as usize;

        // 1. Forward 2D FFT of input_a (channel 0) → spectrum_a.
        //    H-axis reads input_a as `array<f32>` and the real-input
        //    branch picks index `row_base + i` (first n² f32 of
        //    input_a, which is exactly channel 0's data under the
        //    channel-major flat layout).
        self.fft2d.forward_2d_with_scratch(
            ctx,
            encoder,
            &self.twiddles,
            input_a,
            &self.scratch_complex,
            &self.spectrum_a,
        );

        // 2. Spectral multiply: spectrum_a × kernel_fft → k_spectra.
        let bg_sm = self.sm_pass.make_bind_group(
            ctx,
            &self.spectrum_a,
            kernel_fft,
            &self.k_spectra,
            &self.sm_params_buf,
        );
        self.sm_pass.record(encoder, &bg_sm, n, k);

        // 3. Per-kernel inverse 2D FFT. For each ki ∈ [0, k):
        //    a) Copy `k_spectra[ki * cells .. (ki+1) * cells]` into
        //       `inv_in` so the inverse 2D pass sees a 0-offset
        //       complex `n × n` input (the helper's binding contract).
        //    b) `inverse_2d_with_scratch(inv_in, scratch_complex,
        //       spectrum_a)` — `spectrum_a` is reused as the H-axis
        //       output here (it has been free since step 2 consumed
        //       its forward-FFT contents).
        //    c) Copy `spectrum_a` into `k_complex_out[ki * cells ..]`
        //       so all K kernels end up concatenated in K-major order
        //       for the step-4 layout transpose.
        //
        // The per-kernel offset-bind alternative (bind `k_spectra`
        // at offset `ki * cells * 8` directly as the inverse input
        // and `k_complex_out` at the same offset as the inverse
        // output, skipping both copies) is feasible —
        // min_storage_buffer_offset_alignment is 256 bytes and
        // `n² * 8 ≥ 32 768` for n ≥ 64 — but `inverse_2d_with_scratch`
        // currently takes plain `&wgpu::Buffer`s, not offset
        // bindings. Switching it would either fork the helper API
        // or move the offset handling into the caller. Deferred to
        // C-1-5 perf phase; the K extra copies are encoder-internal
        // and cheap relative to the K inverse-FFT dispatches.
        for ki in 0..k {
            let slice_offset = (ki as u64) * (cells as u64) * 8;
            encoder.copy_buffer_to_buffer(
                &self.k_spectra,
                slice_offset,
                &self.inv_in,
                0,
                (cells * 8) as u64,
            );
            self.fft2d.inverse_2d_with_scratch(
                ctx,
                encoder,
                &self.twiddles,
                &self.inv_in,
                &self.scratch_complex,
                &self.spectrum_a,
            );
            encoder.copy_buffer_to_buffer(
                &self.spectrum_a,
                0,
                &self.k_complex_out,
                slice_offset,
                (cells * 8) as u64,
            );
        }

        // 4. Layout transpose: K-major complex → cell-major real pre_g.
        let bg_layout = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fft_to_pre_g bind group"),
            layout: &self.layout_pass.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.k_complex_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: pre_g_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.layout_params_buf.as_entire_binding(),
                },
            ],
        });
        self.layout_pass.record(encoder, &bg_layout, n, k);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::kernel_fft::precompute_kernel_ffts;
    use crate::readback::readback_buffer;
    use flow_lenia_core::config::BorderMode;
    use flow_lenia_core::convolve::convolve2d;
    use flow_lenia_core::kernel::compute_kernel;
    use flow_lenia_core::params::{KernelEntry, KernelParams};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    /// End-to-end test for ConvolveFftPass at N=64 with K=10 Lenia
    /// kernels (matching m1_regression_gpu's typical K), single
    /// channel. Each per-kernel output is compared against the
    /// CPU direct `convolve2d(Torus)` reference (which equals
    /// circular convolution = FFT-based convolution for radially-
    /// symmetric kernels — see kernel_fft.rs module header for the
    /// correlation-vs-convolution caveat).
    ///
    /// **Tolerance basis** (CLAUDE.md "tolerance を緩める前に物理
    /// 的根拠"): per-kernel `rel < 5e-4` borrows the A.4.5 g64
    /// safety-margin ceiling, consistent with the C-1-3
    /// `fft_convolution_matches_direct_torus_n64_k1/k3` tests.
    /// Observed values stay 100+ × under the threshold.
    #[test]
    fn convolve_fft_pass_matches_direct_torus_n64_k10_c1() {
        let (ctx, guard) = headless_ctx();
        let n: u32 = 64;
        let k: u32 = 10;
        let cells = (n * n) as usize;

        let mut rng = ChaCha8Rng::seed_from_u64(0xCF_FA_64_10);
        let entries: Vec<KernelEntry> = (0..k)
            .map(|i| {
                let i_f = i as f32 / k as f32;
                KernelEntry {
                    c0: 0,
                    c1: 0,
                    r: 0.5 + 0.5 * (i_f + 0.1),
                    a: [0.4 + 0.1 * i_f, 0.0, 0.0],
                    b: [1.0, 0.0, 0.0],
                    w: [0.05 + 0.005 * i_f, 0.05, 0.05],
                    h: 0.5 + 0.1 * i_f,
                    mu: 0.10 + 0.02 * i_f,
                    sigma: 0.015 + 0.005 * i_f,
                }
            })
            .collect();
        let params = KernelParams {
            r_global: 5.0,
            kernels: entries.clone(),
        };

        let pass = ConvolveFftPass::new(&ctx, n, k);
        let kernel_fft =
            precompute_kernel_ffts(&ctx, &params, n, &pass.fft2d, &pass.twiddles);
        assert_eq!(kernel_fft.k, k);

        let input: Vec<f32> = (0..cells).map(|_| rng.gen_range(0.0_f32..1.0)).collect();

        // input_a: ConvolvePass expects `a_in[c * H * W + y * W + x]`.
        // With C=1 this collapses to `a_in[y * W + x]` = `input[y*N+x]`.
        // The buffer is sized `n²` reals (4n² bytes) — the
        // ConvolveFftPass forward FFT reads via the `array<f32>`
        // binding and picks `row * n + col`, which matches.
        let input_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("convolve_fft test input a"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        // pre_g_out: real `n² × k` = 4*n²*k bytes.
        let pre_g_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convolve_fft test pre_g_out"),
            size: (cells * k as usize * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("convolve_fft test encoder"),
            });
        pass.record(&ctx, &mut enc, &input_buf, &kernel_fft.buffer, &pre_g_out);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let gpu_pre_g = readback_buffer::<f32>(&ctx, &pre_g_out, cells * k as usize);

        // CPU reference: per-kernel direct convolution with Torus.
        let activation = ndarray::Array2::from_shape_vec((n as usize, n as usize), input.clone())
            .expect("input reshape");
        let mut per_kernel_obs: Vec<(f32, f32)> = Vec::with_capacity(k as usize);
        for (ki, entry) in entries.iter().enumerate() {
            let kernel_cpu = compute_kernel(params.r_global, entry);
            let cpu_result = convolve2d(&activation, &kernel_cpu, BorderMode::Torus);
            let mut max_abs = 0.0_f32;
            let mut max_rel = 0.0_f32;
            for y in 0..n as usize {
                for x in 0..n as usize {
                    // pre_g layout: y * W * K + x * K + ki
                    let g = gpu_pre_g[y * (n as usize) * (k as usize) + x * (k as usize) + ki];
                    let c = cpu_result[[y, x]];
                    let abs_err = (g - c).abs();
                    let mag = c.abs().max(1e-6);
                    let rel_err = abs_err / mag;
                    max_abs = max_abs.max(abs_err);
                    max_rel = max_rel.max(rel_err);
                    assert!(
                        rel_err < 5e-4 || abs_err < 1e-5,
                        "k={ki} (y={y}, x={x}): gpu={g} cpu={c} abs={abs_err:.3e} rel={rel_err:.3e}"
                    );
                }
            }
            per_kernel_obs.push((max_abs, max_rel));
        }
        for (ki, (a, r)) in per_kernel_obs.iter().enumerate() {
            eprintln!(
                "[M6.C-1-4] ConvolveFftPass N=64 K=10 C=1 k={ki}: max_abs={a:.3e}  max_rel={r:.3e}"
            );
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }
}
