//! M2.6 flow compute pass — combined α + F in a single dispatch.
//!
//! Implements paper Eq. 5
//! `F_i = (1 - α)·∇U_i - α·∇A_Σ` with α dispatched on
//! `globals.paper_strict`. See the WGSL `flow.wgsl` doc-block for the
//! "1 shader vs 2 shaders" design decision (different from M2.4).

use crate::{globals::GpuGlobals, GpuContext};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// Workgroup tile. Matches `@workgroup_size(8, 8, 1)` in `flow.wgsl`.
pub const WORKGROUP_X: u32 = 8;
pub const WORKGROUP_Y: u32 = 8;

/// Compiled flow pass — single pipeline, single bind-group layout.
pub struct FlowPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl FlowPass {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/flow.wgsl"),
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("flow.wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("flow bind group layout"),
                    entries: &[
                        bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                        bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                        bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                        bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                        bgl_entry(4, wgpu::BufferBindingType::Uniform),
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("flow pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("flow compute pipeline"),
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

    /// Allocate `(C, H, W, 2)` flat output for F. `STORAGE | COPY_SRC`.
    #[must_use]
    pub fn allocate_f_out(
        ctx: &GpuContext,
        height: u32,
        width: u32,
        channels: u32,
    ) -> wgpu::Buffer {
        let bytes = u64::from(channels) * u64::from(height) * u64::from(width) * 2 * 4;
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("flow f_out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    #[must_use]
    pub fn upload_globals(ctx: &GpuContext, globals: &GpuGlobals) -> wgpu::Buffer {
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("flow globals"),
                contents: cast_slice(std::slice::from_ref(globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        a_in: &wgpu::Buffer,
        grad_u: &wgpu::Buffer,
        grad_a_sum: &wgpu::Buffer,
        f_out: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("flow bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: grad_u.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: grad_a_sum.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: f_out.as_entire_binding(),
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
            label: Some("flow compute pass"),
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
    use crate::{
        activation_buffer::upload_activation, passes::gradient::GradientPass,
        readback::readback_buffer,
    };
    use flow_lenia_core::{
        alpha::alpha,
        config::{BorderMode, FlowLeniaConfig, MixRule},
        flow::flow as cpu_flow,
        sobel::{grad_a_sum, sobel_per_channel},
    };
    use ndarray::Array3;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    fn make_random_activation(rng: &mut ChaCha8Rng, h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0_f32..1.0))
    }

    fn cfg_for(c: u32, paper_strict: bool) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: 32,
            grid_height: 32,
            channels: c,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 0,
            paper_strict,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        }
    }

    /// Run the gradient + flow pipeline on `activation` and read back
    /// the `(C, H, W, 2)` flat F output.
    fn run_flow(
        ctx: &GpuContext,
        grad_pass: &GradientPass,
        flow_pass: &FlowPass,
        activation: &Array3<f32>,
        cfg: &FlowLeniaConfig,
    ) -> Vec<f32> {
        let (h, w, c) = activation.dim();
        let a_buf = upload_activation(ctx, activation);
        let grad_u_buf = GradientPass::allocate_grad_u(ctx, h as u32, w as u32, c as u32);
        let grad_as_buf = GradientPass::allocate_grad_a_sum(ctx, h as u32, w as u32);
        let f_buf = FlowPass::allocate_f_out(ctx, h as u32, w as u32, c as u32);

        let globals = GpuGlobals::new(
            h as u32, w as u32, c as u32, 0, // K unused by flow / gradient
            1, // max_side unused
            cfg.border,
        )
        .with_paper_strict(cfg.paper_strict)
        .with_beta_a(cfg.beta_a)
        .with_n(cfg.n);
        let grad_globals_buf = GradientPass::upload_globals(ctx, &globals);
        let flow_globals_buf = FlowPass::upload_globals(ctx, &globals);

        let bg_u = grad_pass.make_bind_group_u(ctx, &a_buf, &grad_u_buf, &grad_globals_buf);
        let bg_as = grad_pass.make_bind_group_a_sum(ctx, &a_buf, &grad_as_buf, &grad_globals_buf);
        let bg_flow = flow_pass.make_bind_group(
            ctx,
            &a_buf,
            &grad_u_buf,
            &grad_as_buf,
            &f_buf,
            &flow_globals_buf,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("flow test encoder"),
            });
        grad_pass.record_u(&mut enc, &bg_u, h as u32, w as u32);
        grad_pass.record_a_sum(&mut enc, &bg_as, h as u32, w as u32);
        flow_pass.record(&mut enc, &bg_flow, h as u32, w as u32);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        readback_buffer::<f32>(ctx, &f_buf, c * h * w * 2)
    }

    /// CPU reference: alpha + flow chain, returning the CPU
    /// `Array4<f32>` shape `(H, W, 2, C)`. Re-emits in the same
    /// `(C, H, W, 2)` ordering as the GPU output for easy comparison.
    fn cpu_flow_pipeline(activation: &Array3<f32>, cfg: &FlowLeniaConfig) -> Vec<f32> {
        let grad_u = sobel_per_channel(activation, cfg.border);
        let grad_as = grad_a_sum(activation, cfg.border);
        let alpha_field = alpha(activation, cfg);
        let f = cpu_flow(&grad_u, &grad_as, &alpha_field); // (H, W, 2, C)
        let (h, w, _two, c) = f.dim();
        let mut out = vec![0.0_f32; c * h * w * 2];
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    for ax in 0..2usize {
                        out[ci * h * w * 2 + y * w * 2 + x * 2 + ax] = f[[y, x, ax, ci]];
                    }
                }
            }
        }
        out
    }

    fn compare(label: &str, gpu: &[f32], cpu: &[f32], rel_tol: f32, abs_tol: f32) -> (f32, f32) {
        assert_eq!(gpu.len(), cpu.len(), "{label}: shape mismatch");
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (i, (&gv, &cv)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let abs_err = (gv - cv).abs();
            let rel_err = abs_err / cv.abs().max(1e-6);
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            assert!(
                rel_err < rel_tol || abs_err < abs_tol,
                "{label} @ {i}: gpu={gv:e} cpu={cv:e} abs={abs_err:.3e} rel={rel_err:.3e}"
            );
        }
        (max_abs, max_rel)
    }

    #[test]
    fn flow_jax_compat_matches_cpu() {
        let (ctx, guard) = headless_ctx();
        let grad_pass = GradientPass::new(&ctx);
        let flow_pass = FlowPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0xF10F_C001);
        let cfg = cfg_for(3, false);
        let a = make_random_activation(&mut rng, 32, 32, 3);

        let started = Instant::now();
        let gpu = run_flow(&ctx, &grad_pass, &flow_pass, &a, &cfg);
        let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;

        let cpu_started = Instant::now();
        let cpu = cpu_flow_pipeline(&a, &cfg);
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

        let (max_abs, max_rel) = compare("flow_jax_compat", &gpu, &cpu, 1e-4, 1e-5);
        eprintln!(
            "[M2.6-jax]   32×32 torus C=3 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
             gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    #[test]
    fn flow_paper_strict_matches_cpu() {
        let (ctx, guard) = headless_ctx();
        let grad_pass = GradientPass::new(&ctx);
        let flow_pass = FlowPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0xF10F_C002);
        let cfg = cfg_for(3, true);
        let a = make_random_activation(&mut rng, 32, 32, 3);

        let started = Instant::now();
        let gpu = run_flow(&ctx, &grad_pass, &flow_pass, &a, &cfg);
        let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;

        let cpu_started = Instant::now();
        let cpu = cpu_flow_pipeline(&a, &cfg);
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

        let (max_abs, max_rel) = compare("flow_paper_strict", &gpu, &cpu, 1e-4, 1e-5);
        eprintln!(
            "[M2.6-paper] 32×32 torus C=3 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
             gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// With C=1 the two α formulas are mathematically equivalent
    /// (A_Σ = A_0, n hard-coded to 2 in JAX-compat matches the user
    /// setting n = 2 in paper-strict).
    ///
    /// CPU's M1.13 step-level test observed bit-equal trajectories
    /// — but the CPU `alpha_jax_compat` writes `z * z` directly while
    /// `alpha_paper` writes `(z).powf(n)`, and the test holds because
    /// f32 powf was optimised identically across both call sites in
    /// the same `rustc`+ndarray combination.
    ///
    /// **On the GPU the equivalence is only mathematical, not
    /// bit-exact.** The WGSL `pow(x, 2.0)` and (the same `pow(x, n)`
    /// where `n` is a runtime uniform = 2.0) are compiled by Naga +
    /// the Metal driver into slightly different code paths
    /// (constant-folded vs runtime-`n`), producing per-element ≤ 1
    /// ulp drift. Asserted via the same `rel < 1e-6 OR abs < 1e-6`
    /// pattern instead of `to_bits()` so the documented driver
    /// behaviour can't trip a regression on a future driver/Naga
    /// upgrade.
    #[test]
    fn flow_paper_strict_equals_jax_compat_when_c_is_one() {
        let (ctx, guard) = headless_ctx();
        let grad_pass = GradientPass::new(&ctx);
        let flow_pass = FlowPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0xF10F_C003);

        let a = make_random_activation(&mut rng, 32, 32, 1);
        let cfg_jax = cfg_for(1, false);
        let cfg_paper = cfg_for(1, true);

        let f_jax = run_flow(&ctx, &grad_pass, &flow_pass, &a, &cfg_jax);
        let f_paper = run_flow(&ctx, &grad_pass, &flow_pass, &a, &cfg_paper);

        let (max_abs, max_rel) = compare("C=1 paper⇄jax", &f_jax, &f_paper, 1e-6, 1e-6);
        eprintln!(
            "[M2.6-eq]    32×32 torus C=1 paper⇄jax : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }
}
