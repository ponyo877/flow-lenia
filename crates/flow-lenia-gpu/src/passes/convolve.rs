//! M2.3 convolve compute pass.
//!
//! Owns the compiled `convolve.wgsl` pipeline and the bind-group
//! layout it expects. Per-step usage:
//!
//! ```text
//! let pass = ConvolvePass::new(&ctx);
//! // ... allocate a_in, kernels (M2.2), pre_g_out, globals_uniform ...
//! let mut enc = ctx.device.create_command_encoder(...);
//! pass.record(&mut enc, &bind_group, grid_h, grid_w);
//! ctx.queue.submit([enc.finish()]);
//! ```
//!
//! `record` only appends the compute dispatch; the caller chooses
//! when to submit and how to synchronise with downstream passes
//! (M2.4+).

use crate::{globals::GpuGlobals, kernel_buffers::GpuKernelBuffers, GpuContext};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// Workgroup tile side (x, y). Matches `@workgroup_size(8, 8, 1)` in
/// `convolve.wgsl`. Exposed for downstream tests that need to compute
/// the dispatch count.
pub const WORKGROUP_X: u32 = 8;
pub const WORKGROUP_Y: u32 = 8;

/// Compiled convolve pass: pipeline + bind-group layout.
pub struct ConvolvePass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl ConvolvePass {
    /// Compile the shader and build the pipeline. Call once per
    /// `GpuContext`; the resulting pipeline is reused for every
    /// step.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        // `types.wgsl` defines `Meta`, `Globals`, the border constants
        // and `growth_fn`. We prepend it to the per-pass shader source
        // at build time because WGSL has no native include directive.
        const SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/convolve.wgsl"),
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("convolve.wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("convolve bind group layout"),
                    entries: &[
                        // a_in: storage<read>
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
                        // kernels: storage<read>
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
                        // meta_arr: storage<read> (see M2.2 STORAGE|UNIFORM note)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // pre_g_out: storage<read_write>
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // globals: uniform
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
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
                label: Some("convolve pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("convolve compute pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Allocate a fresh `pre_g_out` storage buffer sized for the
    /// current grid + kernel count. `H · W · K` `f32` elements.
    ///
    /// Returned with `STORAGE | COPY_SRC` usage so callers can read
    /// it back for testing without a separate dedicated buffer.
    #[must_use]
    pub fn allocate_pre_g(
        ctx: &GpuContext,
        height: u32,
        width: u32,
        num_kernels: u32,
    ) -> wgpu::Buffer {
        let bytes = u64::from(height) * u64::from(width) * u64::from(num_kernels) * 4;
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("convolve pre_g_out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Upload a single [`GpuGlobals`] as a uniform buffer.
    #[must_use]
    pub fn upload_globals(ctx: &GpuContext, globals: &GpuGlobals) -> wgpu::Buffer {
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("convolve globals"),
                contents: cast_slice(std::slice::from_ref(globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Build the bind group that the convolve pass expects.
    ///
    /// Caller owns all five buffers; the bind group only borrows them.
    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        a_in: &wgpu::Buffer,
        kernels: &GpuKernelBuffers,
        pre_g_out: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("convolve bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernels.kernels.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: kernels.meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: pre_g_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: globals.as_entire_binding(),
                },
            ],
        })
    }

    /// Append a single `dispatch_workgroups` to `encoder` covering
    /// the full grid (rounded up to the workgroup tile size).
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("convolve compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let wg_x = width.div_ceil(WORKGROUP_X);
        let wg_y = height.div_ceil(WORKGROUP_Y);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation_buffer::upload_activation;
    use crate::kernel_buffers::upload_kernels;
    use crate::readback::readback_buffer;
    use flow_lenia_core::{
        config::BorderMode, convolve::convolve2d, kernel::compute_kernel, params::KernelParams,
        params::SamplingSettings,
    };
    use ndarray::{Array3, Axis};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    /// Helper: build a random activation field with deterministic values.
    fn make_activation(rng: &mut ChaCha8Rng, h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0_f32..1.0))
    }

    struct CaseParams {
        seed: u64,
        h: usize,
        w: usize,
        c: usize,
        num_kernels: u32,
        border: BorderMode,
    }

    /// Run the convolve pass and compare against the CPU reference
    /// `convolve2d` per channel/kernel. Returns `(max_abs, max_rel,
    /// gpu_ms, cpu_ms)`.
    fn run_case(ctx: &GpuContext, pass: &ConvolvePass, case: &CaseParams) -> (f32, f32, f64, f64) {
        let CaseParams {
            seed,
            h,
            w,
            c,
            num_kernels,
            border,
        } = *case;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let activation = make_activation(&mut rng, h, w, c);
        let params = KernelParams::sample_random(
            &mut rng,
            SamplingSettings {
                num_kernels,
                num_channels: c as u32,
            },
        );

        // GPU side ------------------------------------------------------------
        let gpu_started = Instant::now();
        let a_buf = upload_activation(ctx, &activation);
        let kernels = upload_kernels(ctx, &params);
        let pre_g_buf = ConvolvePass::allocate_pre_g(ctx, h as u32, w as u32, kernels.count);
        let globals = GpuGlobals::new(
            h as u32,
            w as u32,
            c as u32,
            kernels.count,
            kernels.max_side,
            border,
        );
        let globals_buf = ConvolvePass::upload_globals(ctx, &globals);
        let bind_group = pass.make_bind_group(ctx, &a_buf, &kernels, &pre_g_buf, &globals_buf);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("convolve test encoder"),
            });
        pass.record(&mut enc, &bind_group, h as u32, w as u32);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        let gpu_ms = gpu_started.elapsed().as_secs_f64() * 1000.0;

        // Readback ------------------------------------------------------------
        let pre_g_flat = readback_buffer::<f32>(ctx, &pre_g_buf, h * w * kernels.count as usize);

        // CPU reference -------------------------------------------------------
        let cpu_started = Instant::now();
        let mut cpu_results: Vec<f32> = Vec::with_capacity(h * w * kernels.count as usize);
        // Precompute per-kernel CPU convolution slices.
        let cpu_kernels: Vec<ndarray::Array2<f32>> = params
            .kernels
            .iter()
            .map(|e| compute_kernel(params.r_global, e))
            .collect();
        let mut cpu_per_kernel: Vec<ndarray::Array2<f32>> =
            Vec::with_capacity(params.kernels.len());
        for (entry, kernel) in params.kernels.iter().zip(cpu_kernels.iter()) {
            let src_c = entry.c0 as usize;
            let a_src = activation.index_axis(Axis(2), src_c).to_owned();
            let conv = convolve2d(&a_src, kernel, border);
            cpu_per_kernel.push(conv);
        }
        // Re-emit in cell-major order to match the GPU layout.
        for y in 0..h {
            for x in 0..w {
                for per_kernel in cpu_per_kernel.iter().take(kernels.count as usize) {
                    cpu_results.push(per_kernel[[y, x]]);
                }
            }
        }
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

        // Compare -------------------------------------------------------------
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (i, (&gpu_v, &cpu_v)) in pre_g_flat.iter().zip(cpu_results.iter()).enumerate() {
            let abs_err = (gpu_v - cpu_v).abs();
            let rel_err = abs_err / cpu_v.abs().max(1e-6);
            if abs_err > max_abs {
                max_abs = abs_err;
            }
            if rel_err > max_rel {
                max_rel = rel_err;
            }
            // Guard a single element so the first divergence shows up
            // immediately if we're at >1e-2 (clearly broken).
            assert!(
                rel_err < 1e-2 || abs_err < 1e-5,
                "case ({h}×{w}, C={c}, K={num_kernels}, {border:?}) \
                 @ index {i}: gpu={gpu_v} cpu={cpu_v} abs={abs_err:.3e} rel={rel_err:.3e}"
            );
        }
        (max_abs, max_rel, gpu_ms, cpu_ms)
    }

    /// Three configurations roughly mirroring the M1.6 L2 fixture cases.
    /// All use small (C, K) to keep the test fast; the convolve
    /// correctness does not depend on those numbers.
    #[test]
    fn convolve_pass_matches_cpu_reference() {
        let (ctx, guard) = headless_ctx();
        let pass = ConvolvePass::new(&ctx);

        let labelled_cases: &[(&str, CaseParams)] = &[
            (
                "32×32 torus C=2 K=4",
                CaseParams {
                    seed: 0xC0FF_EE42_u64,
                    h: 32,
                    w: 32,
                    c: 2,
                    num_kernels: 4,
                    border: BorderMode::Torus,
                },
            ),
            (
                "32×32 wall  C=2 K=4",
                CaseParams {
                    seed: 0xC0FF_EE43_u64,
                    h: 32,
                    w: 32,
                    c: 2,
                    num_kernels: 4,
                    border: BorderMode::Wall,
                },
            ),
            (
                "32×32 torus C=3 K=10",
                CaseParams {
                    seed: 0xC0FF_EE44_u64,
                    h: 32,
                    w: 32,
                    c: 3,
                    num_kernels: 10,
                    border: BorderMode::Torus,
                },
            ),
        ];

        for (name, case) in labelled_cases {
            let (max_abs, max_rel, gpu_ms, cpu_ms) = run_case(&ctx, &pass, case);
            eprintln!(
                "[M2.3] {name} : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
                 gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
            );
            assert!(max_rel < 1e-4, "{name} max_rel = {max_rel:.3e} (>= 1e-4)");
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }
}
