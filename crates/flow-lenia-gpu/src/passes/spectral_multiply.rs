//! M6.C-1-3 — spectral multiply compute pass.
//!
//! Wraps `src/shaders/spectral_multiply.wgsl`. Per-step the
//! activation spectrum (one `N × N` complex array) is multiplied
//! against the K pre-FFT'd kernel spectra (one `K × N × N` complex
//! buffer owned by `KernelFftBuffers`) to produce K output spectra
//! (`K × N × N` complex). The inverse FFT of each output spectrum
//! is the convolution of the activation with kernel k.
//!
//! Per-step usage (C-1-4 will wire this in):
//!
//! ```text
//! pass2d.h.record(...);   // input → row-FFT'd
//! pass2d.v.record(...);   // row-FFT'd → 2D spectrum
//! sm_pass.record(...);    // 2D spectrum × kernel_fft → K spectra
//! // ... K × pass2d inverse to recover K convolutions
//! ```

use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Mirror of WGSL `struct SmParams`. The two `_pad` u32 fields make
/// the struct 16-byte aligned per WGSL uniform alignment rules.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct SpectralMultiplyParams {
    pub n: u32,
    pub k: u32,
    pub _pad0: u32,
    pub _pad1: u32,
}

impl SpectralMultiplyParams {
    #[must_use]
    pub fn new(n: u32, k: u32) -> Self {
        Self {
            n,
            k,
            _pad0: 0,
            _pad1: 0,
        }
    }
}

const WORKGROUP_X: u32 = 256;

/// Compiled spectral multiply pass.
pub struct SpectralMultiplyPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl SpectralMultiplyPass {
    /// Compile the shader and build the pipeline. Call once per
    /// `GpuContext`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const SOURCE: &str = include_str!("../shaders/spectral_multiply.wgsl");
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("spectral_multiply.wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("spectral_multiply bind group layout"),
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
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
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

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("spectral_multiply pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("spectral_multiply pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("spectral_multiply"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Assemble a bind group.
    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        input_spectrum: &wgpu::Buffer,
        kernel_fft: &wgpu::Buffer,
        output_spectra: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("spectral_multiply bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input_spectrum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernel_fft.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_spectra.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// Dispatch enough workgroups to cover `n * n * k` complex cells.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        n: u32,
        k: u32,
    ) {
        let total = n * n * k;
        let groups = total.div_ceil(WORKGROUP_X);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("spectral_multiply pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(groups, 1, 1);
    }

    /// Convenience: upload `SpectralMultiplyParams` as a uniform.
    #[must_use]
    pub fn upload_params(ctx: &GpuContext, params: SpectralMultiplyParams) -> wgpu::Buffer {
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("spectral_multiply params uniform"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::passes::fft::{
        precompute_twiddles_1d, Fft2dPass, FftDirection, FftParams,
    };
    use crate::passes::kernel_fft::precompute_kernel_ffts;
    use crate::readback::readback_buffer;
    use crate::GpuContext;
    use flow_lenia_core::config::BorderMode;
    use flow_lenia_core::convolve::convolve2d;
    use flow_lenia_core::kernel::compute_kernel;
    use flow_lenia_core::params::{KernelEntry, KernelParams};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use wgpu::util::DeviceExt;

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    /// Test 1 — spectral multiply alone: GPU result vs CPU
    /// per-cell complex multiply on hand-crafted spectrum buffers.
    /// Isolates the multiply kernel from any FFT dependencies.
    #[test]
    fn spectral_multiply_pointwise_matches_cpu() {
        let (ctx, guard) = headless_ctx();
        let pass = SpectralMultiplyPass::new(&ctx);

        let n: u32 = 64;
        let k: u32 = 3;
        let cells = (n * n) as usize;
        let mut rng = ChaCha8Rng::seed_from_u64(0x5E_C7_1A_42);

        let input: Vec<f32> = (0..(cells * 2)).map(|_| rng.gen_range(-1.0_f32..1.0)).collect();
        let kernels: Vec<f32> = (0..(cells * k as usize * 2))
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let input_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sm test input"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let kernel_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("sm test kernel"),
                contents: bytemuck::cast_slice(&kernels),
                usage: wgpu::BufferUsages::STORAGE,
            });
        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("sm test output"),
            size: (cells * k as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buf =
            SpectralMultiplyPass::upload_params(&ctx, SpectralMultiplyParams::new(n, k));
        let bg = pass.make_bind_group(&ctx, &input_buf, &kernel_buf, &output_buf, &params_buf);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("sm test encoder"),
            });
        pass.record(&mut enc, &bg, n, k);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let gpu = readback_buffer::<f32>(&ctx, &output_buf, cells * k as usize * 2);

        // CPU reference: per (k, cell), out = input_cell * kernel_cell
        // (complex multiply).
        let mut max_abs = 0.0_f32;
        for ki in 0..k as usize {
            for c in 0..cells {
                let ar = input[c * 2];
                let ai = input[c * 2 + 1];
                let br = kernels[(ki * cells + c) * 2];
                let bi = kernels[(ki * cells + c) * 2 + 1];
                let cpu_re = ar * br - ai * bi;
                let cpu_im = ar * bi + ai * br;
                let i = ki * cells + c;
                let gpu_re = gpu[i * 2];
                let gpu_im = gpu[i * 2 + 1];
                let abs = (gpu_re - cpu_re).abs().max((gpu_im - cpu_im).abs());
                max_abs = max_abs.max(abs);
                assert!(
                    abs < 1e-5,
                    "sm mismatch k={ki} cell={c}: gpu=({gpu_re:e},{gpu_im:e}) cpu=({cpu_re:e},{cpu_im:e})"
                );
            }
        }
        eprintln!("[M6.C-1-3] spectral_multiply vs CPU : max_abs={max_abs:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// End-to-end FFT convolution helper — common code for the K=1
    /// and K=3 distinct-kernel tests. GPU path: forward 2D FFT →
    /// spectral multiply → per-kernel inverse 2D FFT. Returns
    /// `Vec<Vec<f32>>` indexed by kernel, each inner vec row-major
    /// N×N reals.
    fn run_fft_convolution_torus(
        ctx: &GpuContext,
        fft2d: &Fft2dPass,
        twiddles: &wgpu::Buffer,
        sm_pass: &SpectralMultiplyPass,
        params: &KernelParams,
        input: &[f32],
        n: u32,
    ) -> Vec<Vec<f32>> {
        let cells = (n * n) as usize;
        assert_eq!(input.len(), cells);

        let kernel_fft = precompute_kernel_ffts(ctx, params, n, fft2d, twiddles);
        let k = kernel_fft.k;
        assert_eq!(k, params.kernels.len() as u32);

        // Two scratch buffers (8N² bytes each) reused for the
        // forward 2D + all per-kernel inverse 2D dispatches.
        let buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft conv test scratch A"),
            size: (cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft conv test scratch B"),
            size: (cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // SM output buffer: K × N² complex spectra.
        let buf_sm_out = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft conv test SM output K spectra"),
            size: (cells * k as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Per-kernel inverse output buffers: one N×N complex each.
        // Read back individually after dispatch.
        let inv_out_bufs: Vec<wgpu::Buffer> = (0..k)
            .map(|ki| {
                ctx.device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some(&format!("fft conv test inv out k{ki}")),
                    size: (cells * 2 * 4) as u64,
                    usage: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                })
            })
            .collect();

        let staging = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fft conv test input staging"),
                contents: bytemuck::cast_slice(input),
                usage: wgpu::BufferUsages::COPY_SRC,
            });

        let params_h_fwd = fft2d
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let params_v_fwd = fft2d
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let params_v_inv = fft2d
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Inverse));
        let params_h_inv = fft2d
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Inverse));
        let sm_params_buf =
            SpectralMultiplyPass::upload_params(ctx, SpectralMultiplyParams::new(n, k));

        let bg_h_fwd =
            fft2d
                .h
                .make_bind_group(ctx, &buf_a, twiddles, &buf_b, &params_h_fwd);
        let bg_v_fwd =
            fft2d
                .v
                .make_bind_group(ctx, &buf_b, twiddles, &buf_a, &params_v_fwd);
        let bg_sm =
            sm_pass.make_bind_group(ctx, &buf_a, &kernel_fft.buffer, &buf_sm_out, &sm_params_buf);

        // Forward path (single encoder, single submit covers staging
        // copy + H + V + SM).
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft conv test forward encoder"),
            });
        enc.copy_buffer_to_buffer(&staging, 0, &buf_a, 0, (cells * 4) as u64);
        fft2d.h.record(&mut enc, &bg_h_fwd, n);
        fft2d.v.record(&mut enc, &bg_v_fwd, n);
        sm_pass.record(&mut enc, &bg_sm, n, k);
        ctx.queue.submit([enc.finish()]);

        // Per-kernel inverse 2D FFT: read kernel k's slice of
        // buf_sm_out and write recovered convolution to inv_out_bufs[k].
        // The slice offset in buf_sm_out is `k * N² * 8` bytes; for a
        // separate-buffer-per-inverse design we copy that slice into
        // a fresh `buf_inv_in` scratch buffer. Simpler than offset
        // bind groups for this test path. Production C-1-4 will use
        // a different orchestration.
        let mut gpu_results: Vec<Vec<f32>> = Vec::with_capacity(k as usize);
        for ki in 0..k as usize {
            let buf_inv_in = ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("fft conv test inv in k{ki}")),
                size: (cells * 2 * 4) as u64,
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            });
            let bg_v_inv = fft2d.v.make_bind_group(
                ctx,
                &buf_inv_in,
                twiddles,
                &buf_b,
                &params_v_inv,
            );
            let bg_h_inv = fft2d.h.make_bind_group(
                ctx,
                &buf_b,
                twiddles,
                &inv_out_bufs[ki],
                &params_h_inv,
            );
            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some(&format!("fft conv test inverse k{ki}")),
                });
            // Copy kernel ki's spectrum slice into buf_inv_in.
            enc.copy_buffer_to_buffer(
                &buf_sm_out,
                (ki as u64) * (cells as u64) * 8,
                &buf_inv_in,
                0,
                (cells * 8) as u64,
            );
            fft2d.v.record(&mut enc, &bg_v_inv, n);
            fft2d.h.record(&mut enc, &bg_h_inv, n);
            ctx.queue.submit([enc.finish()]);
            ctx.device
                .poll(wgpu::PollType::Wait {
                    submission_index: None,
                    timeout: None,
                })
                .unwrap();
            let flat = readback_buffer::<f32>(ctx, &inv_out_bufs[ki], cells * 2);
            gpu_results.push(flat.chunks_exact(2).map(|c| c[0]).collect());
        }
        gpu_results
    }

    /// Compare GPU FFT-conv per-kernel result against CPU
    /// `convolve2d` (Torus). Returns (max_abs, max_rel).
    fn compare_gpu_vs_cpu_per_kernel(
        gpu: &[f32],
        cpu: &ndarray::Array2<f32>,
        n: u32,
        rel_tol: f32,
        abs_tol: f32,
    ) -> (f32, f32) {
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for y in 0..n as usize {
            for x in 0..n as usize {
                let g = gpu[y * n as usize + x];
                let c = cpu[[y, x]];
                let abs_err = (g - c).abs();
                let mag = c.abs().max(1e-6);
                let rel_err = abs_err / mag;
                max_abs = max_abs.max(abs_err);
                max_rel = max_rel.max(rel_err);
                assert!(
                    rel_err < rel_tol || abs_err < abs_tol,
                    "(y={y}, x={x}): gpu={g} cpu={c} abs={abs_err:.3e} rel={rel_err:.3e}"
                );
            }
        }
        (max_abs, max_rel)
    }

    /// End-to-end FFT convolution at N=64 with K=1 Lenia kernel,
    /// compared against `flow_lenia_core::convolve::convolve2d`
    /// under `BorderMode::Torus` (the FFT-equivalence border per
    /// circular convolution theorem; equality additionally requires
    /// the kernel to be centro-symmetric — Lenia kernels are
    /// radial-symmetric by construction, see kernel_fft.rs module
    /// header).
    ///
    /// **Tolerance basis** (Round 1 review #3, honest framing):
    /// the `rel < 5e-4` threshold is borrowed from the M6.A.4.5
    /// committed g64 budget — but A.4.5 was derived from the
    /// chaotic-dynamics CPU↔GPU field regression, not an FFT-
    /// vs-direct convolution error analysis. We adopt the number
    /// here as a **safety-margin** ceiling: observed max_rel on
    /// M1 mini 2026-05-25 is ~2.78e-6 (~180× headroom under 5e-4).
    /// If a future commit lowers the observed value sustainably,
    /// tighten the bound. The threshold is NOT a theoretical
    /// error bound, just a sustained-noise ceiling.
    #[test]
    fn fft_convolution_matches_direct_torus_n64_k1() {
        let (ctx, guard) = headless_ctx();
        let n: u32 = 64;
        let fft2d = Fft2dPass::new(&ctx, n);
        let twiddles = precompute_twiddles_1d(&ctx, n);
        let sm_pass = SpectralMultiplyPass::new(&ctx);

        let entry = KernelEntry {
            c0: 0,
            c1: 0,
            r: 1.0,
            a: [0.5, 0.0, 0.0],
            b: [1.0, 0.0, 0.0],
            w: [0.05, 0.05, 0.05],
            h: 1.0,
            mu: 0.15,
            sigma: 0.02,
        };
        let params = KernelParams {
            r_global: 5.0,
            kernels: vec![entry.clone()],
        };

        let mut rng = ChaCha8Rng::seed_from_u64(0x6F_C7_64_01);
        let cells = (n * n) as usize;
        let input: Vec<f32> = (0..cells).map(|_| rng.gen_range(0.0_f32..1.0)).collect();

        let gpu_results =
            run_fft_convolution_torus(&ctx, &fft2d, &twiddles, &sm_pass, &params, &input, n);
        assert_eq!(gpu_results.len(), 1);

        let activation = ndarray::Array2::from_shape_vec((n as usize, n as usize), input.clone())
            .expect("input reshape");
        let kernel_cpu = compute_kernel(params.r_global, &entry);
        let cpu_result = convolve2d(&activation, &kernel_cpu, BorderMode::Torus);
        let (max_abs, max_rel) =
            compare_gpu_vs_cpu_per_kernel(&gpu_results[0], &cpu_result, n, 5e-4, 1e-5);
        eprintln!(
            "[M6.C-1-3] fft_conv vs direct N=64 K=1 Torus : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// End-to-end FFT convolution at N=64 with K=3 **distinct**
    /// kernels, compared per-kernel against `convolve2d` under
    /// Torus. Round 1 review #2: locks the K-axis indexing
    /// `(k * N + row) * N + col` against a K=1 path that could
    /// accidentally hide a stride bug. Each kernel here has a
    /// different `mu`, `sigma`, and `r` so the per-kernel CPU
    /// references are distinct and the K-axis pointer arithmetic
    /// is exercised on three independent kernel spectra.
    #[test]
    fn fft_convolution_matches_direct_torus_n64_k3_distinct() {
        let (ctx, guard) = headless_ctx();
        let n: u32 = 64;
        let fft2d = Fft2dPass::new(&ctx, n);
        let twiddles = precompute_twiddles_1d(&ctx, n);
        let sm_pass = SpectralMultiplyPass::new(&ctx);

        let entries: Vec<KernelEntry> = vec![
            KernelEntry {
                c0: 0, c1: 0,
                r: 1.0,
                a: [0.5, 0.0, 0.0], b: [1.0, 0.0, 0.0], w: [0.05, 0.05, 0.05],
                h: 1.0, mu: 0.15, sigma: 0.02,
            },
            KernelEntry {
                c0: 0, c1: 0,
                r: 0.7,
                a: [0.4, 0.0, 0.0], b: [1.0, 0.0, 0.0], w: [0.05, 0.05, 0.05],
                h: 0.8, mu: 0.20, sigma: 0.025,
            },
            KernelEntry {
                c0: 0, c1: 0,
                r: 1.2,
                a: [0.6, 0.0, 0.0], b: [1.0, 0.0, 0.0], w: [0.05, 0.05, 0.05],
                h: 0.5, mu: 0.10, sigma: 0.015,
            },
        ];
        let params = KernelParams {
            r_global: 5.0,
            kernels: entries.clone(),
        };

        let mut rng = ChaCha8Rng::seed_from_u64(0x6F_C7_64_03);
        let cells = (n * n) as usize;
        let input: Vec<f32> = (0..cells).map(|_| rng.gen_range(0.0_f32..1.0)).collect();

        let gpu_results =
            run_fft_convolution_torus(&ctx, &fft2d, &twiddles, &sm_pass, &params, &input, n);
        assert_eq!(gpu_results.len(), 3);

        let activation = ndarray::Array2::from_shape_vec((n as usize, n as usize), input.clone())
            .expect("input reshape");
        let mut per_kernel_obs: Vec<(f32, f32)> = Vec::with_capacity(3);
        for (ki, entry) in entries.iter().enumerate() {
            let kernel_cpu = compute_kernel(params.r_global, entry);
            let cpu_result = convolve2d(&activation, &kernel_cpu, BorderMode::Torus);
            let (max_abs, max_rel) =
                compare_gpu_vs_cpu_per_kernel(&gpu_results[ki], &cpu_result, n, 5e-4, 1e-5);
            per_kernel_obs.push((max_abs, max_rel));
        }
        for (ki, (a, r)) in per_kernel_obs.iter().enumerate() {
            eprintln!(
                "[M6.C-1-3] fft_conv vs direct N=64 K=3 Torus k={ki}: \
                 max_abs={a:.3e}  max_rel={r:.3e}"
            );
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }
}
