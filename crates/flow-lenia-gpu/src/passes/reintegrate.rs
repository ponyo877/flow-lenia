//! M2.7 reintegration tracking compute pass (paper Eq. 6).
//!
//! Receiver-side `dd × dd` neighbourhood loop with per-channel
//! accumulation. Reads `A_in` and `F`; writes a fresh `A_out`. The
//! caller manages ping-pong between buffers (M2.10 winit loop).
//!
//! WGSL shader: `shaders/reintegrate.wgsl`. CPU reference:
//! `flow_lenia_core::reintegrate::reintegrate` (M1.11).

use crate::{globals::GpuGlobals, GpuContext};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

pub const WORKGROUP_X: u32 = 8;
pub const WORKGROUP_Y: u32 = 8;

/// Compiled reintegrate pass.
pub struct ReintegratePass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl ReintegratePass {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/reintegrate.wgsl"),
        );
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("reintegrate.wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("reintegrate bind group layout"),
                    entries: &[
                        bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                        bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                        bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: false }),
                        bgl_entry(3, wgpu::BufferBindingType::Uniform),
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("reintegrate pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("reintegrate compute pipeline"),
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

    /// Allocate a fresh `(C, H, W)` channel-major output buffer.
    /// `STORAGE | COPY_SRC | COPY_DST` so it can serve either as the
    /// read-only `A_in` in the *next* step (ping-pong) or be read back
    /// for tests.
    #[must_use]
    pub fn allocate_a(ctx: &GpuContext, height: u32, width: u32, channels: u32) -> wgpu::Buffer {
        let bytes = u64::from(channels) * u64::from(height) * u64::from(width) * 4;
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reintegrate a buffer"),
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
                label: Some("reintegrate globals"),
                contents: cast_slice(std::slice::from_ref(globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        a_in: &wgpu::Buffer,
        flow: &wgpu::Buffer,
        a_out: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("reintegrate bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: a_in.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: flow.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: a_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
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
            label: Some("reintegrate compute pass"),
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
        activation_buffer::{flatten_activation_channel_major, upload_activation},
        readback::readback_buffer,
    };
    use flow_lenia_core::{
        config::{BorderMode, FlowLeniaConfig, MixRule},
        reintegrate::reintegrate as cpu_reintegrate,
        state::{FLOW_DX, FLOW_DY},
    };
    use ndarray::{Array3, Array4};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    fn cfg_torus(c: u32, sigma: f32, dt: f32, dd: u32, h: u32, w: u32) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: w,
            grid_height: h,
            channels: c,
            dt,
            sigma,
            n: 2.0,
            beta_a: 2.0,
            dd,
            num_kernels: 0,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        }
    }

    /// Flatten a CPU `(H, W, 2, C)` flow field into the
    /// `(C, H, W, 2)` channel-major + axis-flow-inner layout the GPU
    /// expects.
    fn flatten_flow_for_gpu(flow: &Array4<f32>) -> Vec<f32> {
        let (h, w, two, c) = flow.dim();
        assert_eq!(two, 2);
        let mut out = vec![0.0_f32; c * h * w * 2];
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    out[ci * h * w * 2 + y * w * 2 + x * 2 + FLOW_DY] = flow[[y, x, FLOW_DY, ci]];
                    out[ci * h * w * 2 + y * w * 2 + x * 2 + FLOW_DX] = flow[[y, x, FLOW_DX, ci]];
                }
            }
        }
        out
    }

    /// Upload `(H, W, 2, C)` flow to a fresh GPU storage buffer.
    fn upload_flow(ctx: &GpuContext, flow: &Array4<f32>) -> wgpu::Buffer {
        let flat = flatten_flow_for_gpu(flow);
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("reintegrate flow"),
                contents: cast_slice(&flat),
                usage: wgpu::BufferUsages::STORAGE
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
            })
    }

    /// Build a `GpuGlobals` from a `FlowLeniaConfig`.
    fn globals_from(cfg: &FlowLeniaConfig) -> GpuGlobals {
        GpuGlobals::new(
            cfg.grid_height,
            cfg.grid_width,
            cfg.channels,
            0,
            1,
            cfg.border,
        )
        .with_dd(cfg.dd)
        .with_sigma(cfg.sigma)
        .with_dt(cfg.dt)
    }

    /// Run one GPU reintegrate step; return the channel-major
    /// `(C, H, W)` flat result.
    fn run_step(
        ctx: &GpuContext,
        pass: &ReintegratePass,
        a: &Array3<f32>,
        flow: &Array4<f32>,
        cfg: &FlowLeniaConfig,
    ) -> Vec<f32> {
        let (h, w, c) = a.dim();
        let a_in = upload_activation(ctx, a);
        let flow_buf = upload_flow(ctx, flow);
        let a_out = ReintegratePass::allocate_a(ctx, h as u32, w as u32, c as u32);
        let globals = globals_from(cfg);
        let globals_buf = ReintegratePass::upload_globals(ctx, &globals);
        let bg = pass.make_bind_group(ctx, &a_in, &flow_buf, &a_out, &globals_buf);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("reintegrate test encoder"),
            });
        pass.record(&mut enc, &bg, h as u32, w as u32);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
        readback_buffer::<f32>(ctx, &a_out, c * h * w)
    }

    fn cmp_flat_vs_array(
        label: &str,
        gpu_flat: &[f32],
        cpu: &Array3<f32>,
        rel_tol: f32,
        abs_tol: f32,
    ) -> (f32, f32) {
        let (h, w, c) = cpu.dim();
        assert_eq!(gpu_flat.len(), c * h * w);
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let gv = gpu_flat[ci * h * w + y * w + x];
                    let cv = cpu[[y, x, ci]];
                    let abs_err = (gv - cv).abs();
                    let rel_err = abs_err / cv.abs().max(1e-6);
                    max_abs = max_abs.max(abs_err);
                    max_rel = max_rel.max(rel_err);
                    assert!(
                        rel_err < rel_tol || abs_err < abs_tol,
                        "{label} @ (y={y}, x={x}, c={ci}): gpu={gv} cpu={cv} \
                         abs={abs_err:.3e} rel={rel_err:.3e}"
                    );
                }
            }
        }
        (max_abs, max_rel)
    }

    /// M1.11 reintegrate_uniform_translation_one_cell_right_torus
    /// ported. `F = (0, 1/dt)` uniform, σ < 0.5, dt = 1. Every cell
    /// should shift exactly one column to the right.
    #[test]
    fn reintegrate_uniform_translation_one_cell_right_torus() {
        let (ctx, guard) = headless_ctx();
        let pass = ReintegratePass::new(&ctx);
        let (h, w, c) = (16usize, 16usize, 1usize);
        let cfg = cfg_torus(c as u32, 0.3, 1.0, 5, h as u32, w as u32);

        // Arbitrary non-trivial initial A.
        let mut rng = ChaCha8Rng::seed_from_u64(0xC011_0001);
        let mut a: Array3<f32> = Array3::zeros((h, w, c));
        for y in 0..h {
            for x in 0..w {
                a[[y, x, 0]] = rng.gen_range(0.0_f32..1.0);
            }
        }
        // dt·F_x = 1 (one cell to the right). FLOW_DY = 0, FLOW_DX = 1.
        let flow: Array4<f32> =
            Array4::from_shape_fn(
                (h, w, 2, c),
                |(_, _, fi, _)| {
                    if fi == FLOW_DX {
                        1.0
                    } else {
                        0.0
                    }
                },
            );

        let gpu_flat = run_step(&ctx, &pass, &a, &flow, &cfg);
        // Each (y, x_new) should equal the value at (y, x_new - 1) (mod W).
        for y in 0..h {
            for x in 0..w {
                let prev_x = (x + w - 1) % w;
                // C=1: channel offset is 0, omitted to keep clippy::erasing_op happy.
                let gpu_v = gpu_flat[y * w + x];
                let expected = a[[y, prev_x, 0]];
                assert!(
                    (gpu_v - expected).abs() < 1e-5,
                    "shift mismatch @ (y={y}, x={x}): gpu={gpu_v} expected={expected}"
                );
            }
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// M1.11 reintegrate_uniform_translation_subcell ported.
    /// `dt·F_x = 0.5` with `σ = 0.5` should split mass 50/50 between
    /// the source cell and the cell one to the right.
    #[test]
    fn reintegrate_uniform_translation_subcell_torus() {
        let (ctx, guard) = headless_ctx();
        let pass = ReintegratePass::new(&ctx);
        let (h, w, c) = (8usize, 8usize, 1usize);
        let cfg = cfg_torus(c as u32, 0.5, 1.0, 5, h as u32, w as u32);

        // A single non-zero cell.
        let mut a: Array3<f32> = Array3::zeros((h, w, c));
        a[[4, 4, 0]] = 1.0;
        let flow: Array4<f32> =
            Array4::from_shape_fn(
                (h, w, 2, c),
                |(_, _, fi, _)| {
                    if fi == FLOW_DX {
                        0.5
                    } else {
                        0.0
                    }
                },
            );

        let gpu_flat = run_step(&ctx, &pass, &a, &flow, &cfg);
        // C=1 → channel offset 0, omitted (clippy::erasing_op).
        let v_same = gpu_flat[4 * w + 4];
        let v_right = gpu_flat[4 * w + 5];
        // With σ = 0.5 + dpmu = ±0.5 the overlap is exactly 0.5 each.
        assert!(
            (v_same - 0.5).abs() < 1e-5,
            "same cell got {v_same}, expected 0.5"
        );
        assert!(
            (v_right - 0.5).abs() < 1e-5,
            "right cell got {v_right}, expected 0.5"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// Random A + random F single-step: GPU vs CPU. Tests both
    /// `dd = 5` (default) and `dd = 7` (max UI slider).
    #[test]
    fn reintegrate_single_step_matches_cpu_dd5_and_dd7() {
        let (ctx, guard) = headless_ctx();
        let pass = ReintegratePass::new(&ctx);
        let (h, w, c) = (32usize, 32usize, 3usize);
        let mut rng = ChaCha8Rng::seed_from_u64(0xC011_DD57);

        for &dd in &[5u32, 7u32] {
            let cfg = cfg_torus(c as u32, 0.65, 0.2, dd, h as u32, w as u32);

            let mut a: Array3<f32> = Array3::zeros((h, w, c));
            for y in 0..h {
                for x in 0..w {
                    for ci in 0..c {
                        a[[y, x, ci]] = rng.gen_range(0.0_f32..1.0);
                    }
                }
            }
            // Modest F (within ma = dd - σ so clip is rarely triggered).
            let flow: Array4<f32> =
                Array4::from_shape_fn((h, w, 2, c), |_| rng.gen_range(-2.0_f32..2.0));

            let started = Instant::now();
            let gpu_flat = run_step(&ctx, &pass, &a, &flow, &cfg);
            let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;

            let cpu_started = Instant::now();
            let cpu_out = cpu_reintegrate(&a, &flow, &cfg);
            let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

            let (max_abs, max_rel) = cmp_flat_vs_array(
                &format!("M2.7-single dd={dd}"),
                &gpu_flat,
                &cpu_out,
                1e-4,
                1e-5,
            );
            eprintln!(
                "[M2.7-single] dd={dd} 32×32 torus C=3 : \
                 max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
                 gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
            );
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// 100-step GPU-only loop, random F (fresh per step) — analogous to
    /// M1.11's `reintegrate_mass_conservation_torus_c{1,3}`. We
    /// allocate two `A` buffers and ping-pong between them.
    ///
    /// M6.C-0: ValidationGuard assertion is performed inside the helper
    /// rather than the callers, since the helper owns the GpuContext
    /// lifetime and the 2 callers (C=1 / C=3 mass-conservation tests)
    /// would otherwise need to duplicate the guard machinery.
    fn run_mass_conservation(channels: u32, seed: u64) -> (f64, f64, f64) {
        let (ctx, guard) = headless_ctx();
        let pass = ReintegratePass::new(&ctx);
        let (h, w) = (32usize, 32usize);
        let c = channels as usize;
        let cfg = cfg_torus(channels, 0.65, 0.2, 5, h as u32, w as u32);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut a: Array3<f32> = Array3::zeros((h, w, c));
        for y in 0..h {
            for x in 0..w {
                for ci in 0..c {
                    a[[y, x, ci]] = rng.gen_range(0.0_f32..1.0);
                }
            }
        }

        // Compute initial mass on the CPU (cheap reduction over the
        // initial activation; the per-step readback would dominate).
        let m0: f64 = a.iter().map(|&v| f64::from(v)).sum();
        assert!(m0 > 0.0);

        // Ping-pong buffers.
        let buf_a = ReintegratePass::allocate_a(ctx_ref(&ctx), h as u32, w as u32, channels);
        let buf_b = ReintegratePass::allocate_a(ctx_ref(&ctx), h as u32, w as u32, channels);
        // Seed buf_a with the initial state via queue.write_buffer.
        let flat_a0 = flatten_activation_channel_major(&a);
        ctx.queue.write_buffer(&buf_a, 0, cast_slice(&flat_a0));

        let globals = globals_from(&cfg);
        let globals_buf = ReintegratePass::upload_globals(&ctx, &globals);

        let mut max_rel = 0.0_f64;
        let started = Instant::now();
        let n_steps = 100u32;

        // Pre-build flow buffer (resized each step); allocate once with
        // STORAGE | COPY_DST so we can re-upload data.
        let flow_bytes = u64::from(channels) * (h as u64) * (w as u64) * 2 * 4;
        let flow_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("reintegrate mass-cons flow"),
            size: flow_bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Pre-build both bind groups; swap by alternating which buffer
        // is read vs written across steps.
        let bg_a_to_b = pass.make_bind_group(&ctx, &buf_a, &flow_buf, &buf_b, &globals_buf);
        let bg_b_to_a = pass.make_bind_group(&ctx, &buf_b, &flow_buf, &buf_a, &globals_buf);

        for step in 0..n_steps {
            // Fresh modest random F (kept small enough that ma clip is
            // rarely active).
            let flow: Array4<f32> =
                Array4::from_shape_fn((h, w, 2, c), |_| rng.gen_range(-0.5_f32..0.5));
            let flow_flat = flatten_flow_for_gpu(&flow);
            ctx.queue.write_buffer(&flow_buf, 0, cast_slice(&flow_flat));

            let mut enc = ctx
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("reintegrate mass-cons encoder"),
                });
            let bg = if step % 2 == 0 {
                &bg_a_to_b
            } else {
                &bg_b_to_a
            };
            pass.record(&mut enc, bg, h as u32, w as u32);
            ctx.queue.submit([enc.finish()]);

            // Measure mass every step. The destination buffer alternates
            // (buf_b on even steps, buf_a on odd steps).
            let dst = if step % 2 == 0 { &buf_b } else { &buf_a };
            let flat = readback_buffer::<f32>(&ctx, dst, c * h * w);
            let m: f64 = flat.iter().map(|&v| f64::from(v)).sum();
            let rel = (m - m0).abs() / m0;
            if rel > max_rel {
                max_rel = rel;
            }
        }
        let elapsed_ms = started.elapsed().as_secs_f64() * 1000.0;
        let per_step_ms = elapsed_ms / f64::from(n_steps);

        if let Some(g) = &guard {
            g.assert_no_errors();
        }

        (max_rel, elapsed_ms, per_step_ms)
    }

    fn ctx_ref(ctx: &GpuContext) -> &GpuContext {
        ctx
    }

    #[test]
    fn reintegrate_mass_conservation_100_steps_c1_torus() {
        let (max_rel, total_ms, per_step_ms) = run_mass_conservation(1, 0xC011_C001);
        eprintln!(
            "[M2.7-mass] C=1 torus 100 steps : max_rel={max_rel:.3e}  \
             total={total_ms:.1}ms  per_step={per_step_ms:.2}ms"
        );
        assert!(max_rel < 1e-3, "C=1 max_rel={max_rel:.3e} exceeds 1e-3");
    }

    #[test]
    fn reintegrate_mass_conservation_100_steps_c3_torus() {
        let (max_rel, total_ms, per_step_ms) = run_mass_conservation(3, 0xC011_C003);
        eprintln!(
            "[M2.7-mass] C=3 torus 100 steps : max_rel={max_rel:.3e}  \
             total={total_ms:.1}ms  per_step={per_step_ms:.2}ms"
        );
        assert!(max_rel < 1e-3, "C=3 max_rel={max_rel:.3e} exceeds 1e-3");
    }
}
