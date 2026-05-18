//! M2.5 gradient compute pass (∇U and ∇A_Σ).
//!
//! Two pipelines, two distinct bind-group layouts:
//! - `pipeline_u`     : ∇U per channel, output `(C, H, W, 2)`.
//! - `pipeline_a_sum` : ∇A_Σ (A_Σ summed on-the-fly), output `(H, W, 2)`.
//!
//! Both apply a 3×3 Sobel in correlation form (M1.7 / M1.9 CPU
//! convention) with border handling delegated to the shared
//! `border_resolve` helper added to `types.wgsl` in M2.5.
//!
//! Output layout puts the flow axis (`FLOW_DY = 0`, `FLOW_DX = 1`)
//! at the innermost stride so the M2.6 flow pass reads both
//! components of one cell as adjacent `f32`s.

use crate::{globals::GpuGlobals, GpuContext};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// Workgroup tile dimensions. Match `@workgroup_size(8, 8, 1)` in
/// both `gradient_*.wgsl` files.
pub const WORKGROUP_X: u32 = 8;
pub const WORKGROUP_Y: u32 = 8;

/// Compiled gradient passes — `∇U` and `∇A_Σ`.
pub struct GradientPass {
    pub pipeline_u: wgpu::ComputePipeline,
    pub pipeline_a_sum: wgpu::ComputePipeline,
    pub bind_group_layout_u: wgpu::BindGroupLayout,
    pub bind_group_layout_a_sum: wgpu::BindGroupLayout,
}

impl GradientPass {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const U_SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/gradient_u.wgsl"),
        );
        const A_SUM_SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/gradient_a_sum.wgsl"),
        );
        let module_u = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gradient_u.wgsl"),
                source: wgpu::ShaderSource::Wgsl(U_SOURCE.into()),
            });
        let module_a_sum = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("gradient_a_sum.wgsl"),
                source: wgpu::ShaderSource::Wgsl(A_SUM_SOURCE.into()),
            });

        // Both passes have the same three-binding shape:
        //   0 = a_in (storage<read>)
        //   1 = output (storage<read_write>)
        //   2 = globals (uniform)
        let make_layout = |label: &str| {
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(label),
                    entries: &[
                        bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                        bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                        bgl_entry(2, wgpu::BufferBindingType::Uniform),
                    ],
                })
        };
        let bind_group_layout_u = make_layout("gradient_u bind group layout");
        let bind_group_layout_a_sum = make_layout("gradient_a_sum bind group layout");

        let make_pipeline =
            |bgl: &wgpu::BindGroupLayout, module: &wgpu::ShaderModule, label: &str| {
                let pl = ctx
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some(label),
                        bind_group_layouts: &[Some(bgl)],
                        immediate_size: 0,
                    });
                ctx.device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: Some(label),
                        layout: Some(&pl),
                        module,
                        entry_point: Some("main"),
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                        cache: None,
                    })
            };

        let pipeline_u = make_pipeline(&bind_group_layout_u, &module_u, "gradient_u pipeline");
        let pipeline_a_sum = make_pipeline(
            &bind_group_layout_a_sum,
            &module_a_sum,
            "gradient_a_sum pipeline",
        );

        Self {
            pipeline_u,
            pipeline_a_sum,
            bind_group_layout_u,
            bind_group_layout_a_sum,
        }
    }

    /// Allocate `(C, H, W, 2)` flat output for ∇U. `STORAGE | COPY_SRC`.
    #[must_use]
    pub fn allocate_grad_u(
        ctx: &GpuContext,
        height: u32,
        width: u32,
        channels: u32,
    ) -> wgpu::Buffer {
        let bytes = u64::from(channels) * u64::from(height) * u64::from(width) * 2 * 4;
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gradient grad_u"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Allocate `(H, W, 2)` flat output for ∇A_Σ. `STORAGE | COPY_SRC`.
    #[must_use]
    pub fn allocate_grad_a_sum(ctx: &GpuContext, height: u32, width: u32) -> wgpu::Buffer {
        let bytes = u64::from(height) * u64::from(width) * 2 * 4;
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gradient grad_a_sum"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Upload a single `GpuGlobals` uniform buffer (32 bytes).
    #[must_use]
    pub fn upload_globals(ctx: &GpuContext, globals: &GpuGlobals) -> wgpu::Buffer {
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("gradient globals"),
                contents: cast_slice(std::slice::from_ref(globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    #[must_use]
    pub fn make_bind_group_u(
        &self,
        ctx: &GpuContext,
        a_in: &wgpu::Buffer,
        grad_u_out: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gradient_u bind group"),
            layout: &self.bind_group_layout_u,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_u_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: globals.as_entire_binding(),
                },
            ],
        })
    }

    #[must_use]
    pub fn make_bind_group_a_sum(
        &self,
        ctx: &GpuContext,
        a_in: &wgpu::Buffer,
        grad_a_sum_out: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("gradient_a_sum bind group"),
            layout: &self.bind_group_layout_a_sum,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_a_sum_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: globals.as_entire_binding(),
                },
            ],
        })
    }

    pub fn record_u(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        self.dispatch(&self.pipeline_u, encoder, bind_group, height, width);
    }

    pub fn record_a_sum(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        self.dispatch(&self.pipeline_a_sum, encoder, bind_group, height, width);
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gradient compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        let wg_x = width.div_ceil(WORKGROUP_X);
        let wg_y = height.div_ceil(WORKGROUP_Y);
        pass.dispatch_workgroups(wg_x, wg_y, 1);
    }
}

#[inline]
fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{activation_buffer::upload_activation, readback::readback_buffer};
    use flow_lenia_core::{
        config::BorderMode,
        sobel::{grad_a_sum, sobel_per_channel},
        state::FLOW_DY,
    };
    use ndarray::Array3;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn headless_ctx() -> GpuContext {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        GpuContext::new_blocking(instance, None)
    }

    fn make_random_activation(rng: &mut ChaCha8Rng, h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0_f32..1.0))
    }

    /// Run the ∇U pass and read back the `(C, H, W, 2)` flat buffer.
    fn run_grad_u(
        ctx: &GpuContext,
        pass: &GradientPass,
        activation: &Array3<f32>,
        border: BorderMode,
    ) -> Vec<f32> {
        let (h, w, c) = activation.dim();
        let a_buf = upload_activation(ctx, activation);
        let grad_u_buf = GradientPass::allocate_grad_u(ctx, h as u32, w as u32, c as u32);
        let globals = GpuGlobals::new(h as u32, w as u32, c as u32, 0, 1, border);
        let globals_buf = GradientPass::upload_globals(ctx, &globals);
        let bg = pass.make_bind_group_u(ctx, &a_buf, &grad_u_buf, &globals_buf);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("grad_u test encoder"),
            });
        pass.record_u(&mut enc, &bg, h as u32, w as u32);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        readback_buffer::<f32>(ctx, &grad_u_buf, c * h * w * 2)
    }

    /// Run the ∇A_Σ pass and read back the `(H, W, 2)` flat buffer.
    fn run_grad_a_sum(
        ctx: &GpuContext,
        pass: &GradientPass,
        activation: &Array3<f32>,
        border: BorderMode,
    ) -> Vec<f32> {
        let (h, w, c) = activation.dim();
        let a_buf = upload_activation(ctx, activation);
        let grad_buf = GradientPass::allocate_grad_a_sum(ctx, h as u32, w as u32);
        let globals = GpuGlobals::new(h as u32, w as u32, c as u32, 0, 1, border);
        let globals_buf = GradientPass::upload_globals(ctx, &globals);
        let bg = pass.make_bind_group_a_sum(ctx, &a_buf, &grad_buf, &globals_buf);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("grad_a_sum test encoder"),
            });
        pass.record_a_sum(&mut enc, &bg, h as u32, w as u32);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        readback_buffer::<f32>(ctx, &grad_buf, h * w * 2)
    }

    /// ∇U: GPU `(C, H, W, 2)` flat vs CPU `Array4<f32>` `(H, W, 2, C)`.
    /// CPU axis order has the flow axis at position 2 and channel at 3.
    /// We compare in (y, x, axis, c) order.
    #[test]
    fn gradient_u_matches_cpu_sobel_per_channel() {
        let ctx = headless_ctx();
        let pass = GradientPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0x6_AD00_F0F0);

        let (h, w, c) = (32_usize, 32_usize, 3_usize);
        let a = make_random_activation(&mut rng, h, w, c);

        let started = Instant::now();
        let gpu = run_grad_u(&ctx, &pass, &a, BorderMode::Torus);
        let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;

        let cpu_started = Instant::now();
        let cpu = sobel_per_channel(&a, BorderMode::Torus); // (H, W, 2, C)
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    for ax in 0..2usize {
                        let gpu_v = gpu[ci * h * w * 2 + y * w * 2 + x * 2 + ax];
                        let cpu_v = cpu[[y, x, ax, ci]];
                        let abs_err = (gpu_v - cpu_v).abs();
                        let rel_err = abs_err / cpu_v.abs().max(1e-6);
                        max_abs = max_abs.max(abs_err);
                        max_rel = max_rel.max(rel_err);
                        assert!(
                            rel_err < 1e-4 || abs_err < 1e-5,
                            "(y={y}, x={x}, ax={ax}, c={ci}): gpu={gpu_v:e} cpu={cpu_v:e} \
                             abs={abs_err:.3e} rel={rel_err:.3e}"
                        );
                    }
                }
            }
        }
        eprintln!(
            "[M2.5-∇U]   32×32 torus C=3 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
             gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
        );
    }

    /// ∇A_Σ: GPU `(H, W, 2)` flat vs CPU `Array3<f32>` `(H, W, 2)`.
    /// Same axis order; straightforward element-wise compare.
    #[test]
    fn gradient_a_sum_matches_cpu() {
        let ctx = headless_ctx();
        let pass = GradientPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0x6_AD11_F0F1);

        let (h, w, c) = (32_usize, 32_usize, 3_usize);
        let a = make_random_activation(&mut rng, h, w, c);

        let started = Instant::now();
        let gpu = run_grad_a_sum(&ctx, &pass, &a, BorderMode::Torus);
        let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;

        let cpu_started = Instant::now();
        let cpu = grad_a_sum(&a, BorderMode::Torus); // (H, W, 2)
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for y in 0..h {
            for x in 0..w {
                for ax in 0..2usize {
                    let gpu_v = gpu[y * w * 2 + x * 2 + ax];
                    let cpu_v = cpu[[y, x, ax]];
                    let abs_err = (gpu_v - cpu_v).abs();
                    let rel_err = abs_err / cpu_v.abs().max(1e-6);
                    max_abs = max_abs.max(abs_err);
                    max_rel = max_rel.max(rel_err);
                    assert!(
                        rel_err < 1e-4 || abs_err < 1e-5,
                        "(y={y}, x={x}, ax={ax}): gpu={gpu_v:e} cpu={cpu_v:e} \
                         abs={abs_err:.3e} rel={rel_err:.3e}"
                    );
                }
            }
        }
        eprintln!(
            "[M2.5-∇A_Σ] 32×32 torus C=3 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
             gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
        );
    }

    /// A(x, y) = x: interior ∂x = +8 (M1.7 sobel_x_with_x_ramp). C=1
    /// so A_Σ = A, and the same +8 value should appear in both
    /// ∇U[0] and ∇A_Σ[0] for the FLOW_DX component at interior cells.
    ///
    /// Torus is checked because Sobel on a linear ramp under wall has
    /// boundary artifacts (M1.7 design).
    #[test]
    fn gradient_x_ramp_interior_partial_dx_is_eight() {
        let ctx = headless_ctx();
        let pass = GradientPass::new(&ctx);

        let (h, w, c) = (8_usize, 8_usize, 1_usize);
        // A(y, x, 0) = x. Avoid the torus wrap discontinuity at x=0/W-1
        // by only checking the strict interior (1 <= x <= W-2).
        let mut a: Array3<f32> = Array3::zeros((h, w, c));
        for y in 0..h {
            for x in 0..w {
                a[[y, x, 0]] = x as f32;
            }
        }

        // For a linear x ramp on a torus, the wrap at x=0/x=W-1 makes
        // the boundary Sobel non-uniform; use Wall border so the
        // interior `1..=W-2` cells see the expected +8 without wrap
        // contamination. The wall band itself (x=0, x=W-1) is excluded
        // from the assertion.
        let gpu_u = run_grad_u(&ctx, &pass, &a, BorderMode::Wall);
        let gpu_as = run_grad_a_sum(&ctx, &pass, &a, BorderMode::Wall);

        // FLOW_DY = 0, FLOW_DX = 1. Use the constants to be explicit.
        let dx_axis = 1_usize;
        let _ = FLOW_DY; // touch the import

        for y in 1..(h - 1) {
            for x in 1..(w - 1) {
                // Single channel (C=1) → channel offset 0 is implicit;
                // dropping the `0 * ...` keeps clippy::erasing_op happy.
                let v_u = gpu_u[y * w * 2 + x * 2 + dx_axis];
                let v_as = gpu_as[y * w * 2 + x * 2 + dx_axis];
                assert!(
                    (v_u - 8.0).abs() < 1e-5,
                    "∇U[(y={y}, x={x}, ax=DX, c=0)] = {v_u}, expected 8.0"
                );
                assert!(
                    (v_as - 8.0).abs() < 1e-5,
                    "∇A_Σ[(y={y}, x={x}, ax=DX)] = {v_as}, expected 8.0"
                );
            }
        }
    }
}
