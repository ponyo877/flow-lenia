//! M6.C-2-4-c ParameterFlowPass — identity-copy infrastructure pass
//! that establishes the binding contract and ping-pong scaffolding
//! for Plantec 2025 Eq. 8 (parameter inheritance during
//! reintegration). The stochastic-sampling algorithm itself is
//! deferred to M5 per Ponyo877-san strategic decision 2026-05-27.
//!
//! See `shaders/parameter_flow.wgsl` for the binding contract and
//! the M5 hook block.

use crate::{globals::GpuGlobals, GpuContext};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

pub const WORKGROUP_X: u32 = 8;
pub const WORKGROUP_Y: u32 = 8;

/// Compiled parameter-flow pass. Single pipeline; the M5 Eq. 8
/// upgrade replaces only the WGSL body, not the bindings.
pub struct ParameterFlowPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl ParameterFlowPass {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/parameter_flow.wgsl"),
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("parameter_flow.wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("parameter_flow bind group layout"),
                    entries: &[
                        // 0: p_in (read)
                        bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                        // 1: p_out (read_write)
                        bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: false }),
                        // 2: matter_flow (read, M5 hook input)
                        bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                        // 3: kernel_routing (read, M5 hook input)
                        bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                        // 4: globals (uniform)
                        bgl_entry(4, wgpu::BufferBindingType::Uniform),
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("parameter_flow pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("parameter_flow compute pipeline"),
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

    /// Allocate an `H * W * K` flat f32 buffer suitable for either
    /// `p_in` (read) or `p_out` (read_write). `STORAGE | COPY_SRC |
    /// COPY_DST` to permit upload (initialisation from
    /// `parameter_map::build_for_patches`), readback (tests), and
    /// ping-pong swaps.
    #[must_use]
    pub fn allocate_p(
        ctx: &GpuContext,
        height: u32,
        width: u32,
        num_kernels: u32,
    ) -> wgpu::Buffer {
        let bytes = u64::from(height) * u64::from(width) * u64::from(num_kernels) * 4;
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("parameter_flow p buffer"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    #[must_use]
    pub fn upload_globals(ctx: &GpuContext, globals: &GpuGlobals) -> wgpu::Buffer {
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("parameter_flow globals"),
                contents: cast_slice(std::slice::from_ref(globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        p_in: &wgpu::Buffer,
        p_out: &wgpu::Buffer,
        matter_flow: &wgpu::Buffer,
        kernel_routing: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("parameter_flow bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: p_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: p_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: matter_flow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: kernel_routing.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: globals.as_entire_binding(),
                },
            ],
        })
    }

    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("parameter_flow compute pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
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
    use crate::{passes::parameter_map, readback::readback_buffer};
    use flow_lenia_core::config::BorderMode;

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    /// M6.C-2-4-c identity property: a single ParameterFlowPass
    /// dispatch must leave the parameter map P unchanged
    /// (`p_out == p_in` bit-equal). This is the entire algorithmic
    /// content of the case-(a) infrastructure; M5 will replace this
    /// invariant with Eq. 8 stochastic sampling.
    #[test]
    fn parameter_flow_identity_p_out_equals_p_in() {
        let (ctx, guard) = headless_ctx();
        let pass = ParameterFlowPass::new(&ctx);

        let h: u32 = 16;
        let w: u32 = 16;
        let c: u32 = 3;
        let k: u32 = 8;

        // Build a non-trivial P map via parameter_map (matches the
        // C-2-4-a / C-2-4-b layout exactly).
        let default_p: Vec<f32> = (0..k).map(|ki| (ki as f32) * 0.01).collect();
        let patches = vec![
            parameter_map::CreaturePatch {
                bbox: (2, 2, 6, 6),
                p_vector: (0..k).map(|ki| (ki as f32) * 0.10 + 1.0).collect(),
            },
            parameter_map::CreaturePatch {
                bbox: (10, 10, 14, 14),
                p_vector: (0..k).map(|ki| (ki as f32) * 0.10 + 2.0).collect(),
            },
        ];
        let p_initial = parameter_map::build_for_patches(w, k, &default_p, &patches);

        let p_in_buf = ParameterFlowPass::allocate_p(&ctx, h, w, k);
        let p_out_buf = ParameterFlowPass::allocate_p(&ctx, h, w, k);
        ctx.queue
            .write_buffer(&p_in_buf, 0, cast_slice(&p_initial));

        // Dummy matter_flow (C, H, W, 2) and kernel_routing (K).
        // Identity copy never reads these.
        let flow_bytes = u64::from(c) * u64::from(h) * u64::from(w) * 2 * 4;
        let matter_flow = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test matter_flow"),
            size: flow_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let routing: Vec<u32> = (0..k).map(|ki| ki % c).collect();
        let kernel_routing =
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test kernel_routing"),
                contents: cast_slice(&routing),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let globals = GpuGlobals::new(h, w, c, k, 1, BorderMode::Torus);
        let globals_buf = ParameterFlowPass::upload_globals(&ctx, &globals);

        let bg = pass.make_bind_group(
            &ctx,
            &p_in_buf,
            &p_out_buf,
            &matter_flow,
            &kernel_routing,
            &globals_buf,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("parameter_flow identity test encoder"),
            });
        pass.record(&mut enc, &bg, h, w);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let p_out = readback_buffer::<f32>(
            &ctx,
            &p_out_buf,
            (h * w * k) as usize,
        );
        assert_eq!(p_out.len(), p_initial.len());
        for (i, (a, b)) in p_out.iter().zip(p_initial.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "identity violated at index {i}: p_out={a} p_in={b}"
            );
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// Ping-pong invariant: after N dispatches alternating
    /// (p_a → p_b) and (p_b → p_a), the final buffer holds the
    /// **bit-equal initial map**. This locks the M5 hook contract —
    /// `parameter_map` initialisation is preserved across step
    /// boundaries until Eq. 8 is wired in.
    #[test]
    fn parameter_flow_pingpong_ten_steps_preserves_initial_map() {
        let (ctx, guard) = headless_ctx();
        let pass = ParameterFlowPass::new(&ctx);

        let h: u32 = 8;
        let w: u32 = 8;
        let c: u32 = 1;
        let k: u32 = 4;
        let steps = 10_u32;

        // Distinct per-cell P so bit-equality across 10 steps is a
        // strong signal that nothing other than identity copy ran.
        let mut p_initial = vec![0.0_f32; (h * w * k) as usize];
        for y in 0..h {
            for x in 0..w {
                for ki in 0..k {
                    let idx = ((y * w + x) * k + ki) as usize;
                    p_initial[idx] =
                        (y as f32) * 100.0 + (x as f32) * 10.0 + (ki as f32) * 0.1;
                }
            }
        }

        let p_a = ParameterFlowPass::allocate_p(&ctx, h, w, k);
        let p_b = ParameterFlowPass::allocate_p(&ctx, h, w, k);
        ctx.queue.write_buffer(&p_a, 0, cast_slice(&p_initial));

        let flow_bytes = u64::from(c) * u64::from(h) * u64::from(w) * 2 * 4;
        let matter_flow = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("test matter_flow"),
            size: flow_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let routing: Vec<u32> = (0..k).map(|_| 0).collect();
        let kernel_routing =
            ctx.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("test kernel_routing"),
                contents: cast_slice(&routing),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            });

        let globals = GpuGlobals::new(h, w, c, k, 1, BorderMode::Torus);
        let globals_buf = ParameterFlowPass::upload_globals(&ctx, &globals);

        let bg_ab = pass.make_bind_group(
            &ctx,
            &p_a,
            &p_b,
            &matter_flow,
            &kernel_routing,
            &globals_buf,
        );
        let bg_ba = pass.make_bind_group(
            &ctx,
            &p_b,
            &p_a,
            &matter_flow,
            &kernel_routing,
            &globals_buf,
        );

        for step in 0..steps {
            let mut enc =
                ctx.device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("parameter_flow ping-pong encoder"),
                    });
            let bg = if step % 2 == 0 { &bg_ab } else { &bg_ba };
            pass.record(&mut enc, bg, h, w);
            ctx.queue.submit([enc.finish()]);
        }
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        // After 10 steps (even count) the final write landed in `p_b`.
        let p_final = readback_buffer::<f32>(&ctx, &p_b, (h * w * k) as usize);
        for (i, (a, b)) in p_final.iter().zip(p_initial.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "ping-pong drift at index {i}: final={a} initial={b}"
            );
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }
}
