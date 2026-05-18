//! M2.4 affinity-growth compute pass (paper Eq. 3 and Eq. 7).
//!
//! Produces the affinity field `U_j(x)` from the per-(cell, kernel)
//! convolutions `pre_g[x, i]` (M2.3 output). Two pipelines share the
//! same bind-group layout — only the WGSL body differs:
//!
//! - `pipeline_constant`: paper Eq. 3, weight is a per-kernel `h_i`.
//! - `pipeline_localized`: paper Eq. 7, weight is a per-cell map
//!   `P_i(x)`.
//!
//! Output layout: channel-major `(C, H, W)` to match the activation
//! buffer (M2.3 `a_in`) — this lets the downstream Sobel (M2.5) read
//! one channel slab at a time the same way the convolve pass does.

use crate::{globals::GpuGlobals, kernel_buffers::GpuKernelBuffers, GpuContext};
use bytemuck::cast_slice;
use wgpu::util::DeviceExt;

/// Hard cap on `|K|` for the constant-weights buffer. Mirrors
/// `MAX_KERNELS` in `shaders/types.wgsl`. DESIGN.md §7 caps the UI
/// slider at 45.
pub const MAX_KERNELS: usize = 45;

/// Workgroup tile dimensions. Match `@workgroup_size(8, 8, 1)` in
/// both `affinity_growth_*.wgsl` files.
pub const WORKGROUP_X: u32 = 8;
pub const WORKGROUP_Y: u32 = 8;

/// Constant per-kernel weights `h_i` (paper Eq. 3).
///
/// Stored as a fixed `MAX_KERNELS`-element array on both the Rust
/// and the WGSL side so the binding layout doesn't depend on `K`.
/// Unused tail entries are zero-padded and the WGSL loop simply
/// stops at `globals.k`.
///
/// Layout matches WGSL `array<f32>` in `storage` — 4-byte stride,
/// no per-element 16-byte requirement (only `uniform` arrays carry
/// that rule).
#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuConstantWeights {
    pub h: [f32; MAX_KERNELS],
}

impl GpuConstantWeights {
    /// Pack a `&[f32]` of length `≤ MAX_KERNELS` into a
    /// `GpuConstantWeights`. The tail is zero-filled.
    ///
    /// # Panics
    ///
    /// Panics if `h.len() > MAX_KERNELS`.
    #[must_use]
    pub fn from_slice(h: &[f32]) -> Self {
        assert!(
            h.len() <= MAX_KERNELS,
            "h.len() = {} exceeds MAX_KERNELS = {MAX_KERNELS}",
            h.len()
        );
        let mut buf = [0.0_f32; MAX_KERNELS];
        buf[..h.len()].copy_from_slice(h);
        Self { h: buf }
    }
}

/// Upload constant per-kernel weights as a fresh GPU storage buffer.
#[must_use]
pub fn upload_constant_weights(ctx: &GpuContext, weights: &GpuConstantWeights) -> wgpu::Buffer {
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("affinity_growth h_weights (Eq. 3)"),
            contents: cast_slice(std::slice::from_ref(weights)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
}

/// Flatten a CPU `(H, W, K)` `ndarray::Array3<f32>` into the
/// cell-major flat layout used by `affinity_growth_localized.wgsl`:
///
/// ```text
/// p_map[y * W * K + x * K + ki]
/// ```
///
/// Matches the convolve pass's `pre_g` layout (M2.3) exactly so the
/// two buffers can be read with the same `base + ki` indexing in the
/// shader.
#[must_use]
pub fn flatten_p_map_cell_major(p_map: &ndarray::Array3<f32>) -> Vec<f32> {
    let (h, w, k) = p_map.dim();
    let mut flat = vec![0.0_f32; h * w * k];
    for y in 0..h {
        for x in 0..w {
            for ki in 0..k {
                flat[y * w * k + x * k + ki] = p_map[[y, x, ki]];
            }
        }
    }
    flat
}

/// Upload a cell-major `P_i(x)` map as a fresh GPU storage buffer
/// (paper Eq. 7 weights).
#[must_use]
pub fn upload_localized_weights(ctx: &GpuContext, p_map: &ndarray::Array3<f32>) -> wgpu::Buffer {
    let flat = flatten_p_map_cell_major(p_map);
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("affinity_growth p_map (Eq. 7)"),
            contents: cast_slice(&flat),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        })
}

/// Compiled affinity-growth pass — two pipelines sharing one bind-group
/// layout.
///
/// The Eq. 3 and Eq. 7 pipelines have **identical** bind-group layouts;
/// the only thing that changes is the WGSL body interpreting binding 2
/// as either `h_weights` (constant case) or `p_map` (localized case).
/// This lets callers swap pipelines per step without rebuilding bind
/// groups.
pub struct AffinityGrowthPass {
    pub pipeline_constant: wgpu::ComputePipeline,
    pub pipeline_localized: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl AffinityGrowthPass {
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const CONSTANT_SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/affinity_growth_constant.wgsl"),
        );
        const LOCALIZED_SOURCE: &str = concat!(
            include_str!("../shaders/types.wgsl"),
            "\n",
            include_str!("../shaders/affinity_growth_localized.wgsl"),
        );
        let module_constant = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("affinity_growth_constant.wgsl"),
                source: wgpu::ShaderSource::Wgsl(CONSTANT_SOURCE.into()),
            });
        let module_localized = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("affinity_growth_localized.wgsl"),
                source: wgpu::ShaderSource::Wgsl(LOCALIZED_SOURCE.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("affinity_growth bind group layout"),
                    entries: &[
                        // pre_g : storage<read>
                        bgl_entry(0, wgpu::BufferBindingType::Storage { read_only: true }),
                        // meta_arr : storage<read>
                        bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                        // h_weights / p_map : storage<read>
                        bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                        // u_out : storage<read_write>
                        bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: false }),
                        // globals : uniform
                        bgl_entry(4, wgpu::BufferBindingType::Uniform),
                    ],
                });

        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("affinity_growth pipeline layout"),
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

        let make_pipeline = |module: &wgpu::ShaderModule, label: &str| {
            ctx.device
                .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                    label: Some(label),
                    layout: Some(&pipeline_layout),
                    module,
                    entry_point: Some("main"),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    cache: None,
                })
        };

        Self {
            pipeline_constant: make_pipeline(&module_constant, "affinity_growth_constant pipeline"),
            pipeline_localized: make_pipeline(
                &module_localized,
                "affinity_growth_localized pipeline",
            ),
            bind_group_layout,
        }
    }

    /// Allocate a fresh `u_out` storage buffer sized for `(C, H, W)`
    /// channel-major flat. `STORAGE | COPY_SRC` usage so tests can
    /// read it back.
    #[must_use]
    pub fn allocate_u_out(
        ctx: &GpuContext,
        height: u32,
        width: u32,
        channels: u32,
    ) -> wgpu::Buffer {
        let bytes = u64::from(height) * u64::from(width) * u64::from(channels) * 4;
        ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("affinity_growth u_out"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    /// Upload a single [`GpuGlobals`] uniform buffer. Sized 32 bytes;
    /// see `globals.rs` for the layout.
    #[must_use]
    pub fn upload_globals(ctx: &GpuContext, globals: &GpuGlobals) -> wgpu::Buffer {
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("affinity_growth globals"),
                contents: cast_slice(std::slice::from_ref(globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }

    /// Build a bind group for either variant. `weights_or_p` is the
    /// constant-weights buffer for the `_constant` pipeline and the
    /// `p_map` buffer for the `_localized` pipeline.
    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        pre_g: &wgpu::Buffer,
        kernels: &GpuKernelBuffers,
        weights_or_p: &wgpu::Buffer,
        u_out: &wgpu::Buffer,
        globals: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("affinity_growth bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: pre_g.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: kernels.meta.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: weights_or_p.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: u_out.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: globals.as_entire_binding(),
                },
            ],
        })
    }

    /// Append a single `dispatch_workgroups` using the Eq. 3
    /// constant-weights pipeline.
    pub fn record_constant(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        self.record_with(&self.pipeline_constant, encoder, bind_group, height, width);
    }

    /// Append a single `dispatch_workgroups` using the Eq. 7
    /// localised-weights pipeline.
    pub fn record_localized(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        self.record_with(&self.pipeline_localized, encoder, bind_group, height, width);
    }

    fn record_with(
        &self,
        pipeline: &wgpu::ComputePipeline,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        height: u32,
        width: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("affinity_growth compute pass"),
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
    use crate::{
        activation_buffer::upload_activation, kernel_buffers::upload_kernels,
        passes::convolve::ConvolvePass, readback::readback_buffer,
    };
    use flow_lenia_core::{
        affinity::{affinity_with_constant_weights, affinity_with_localized_weights},
        config::BorderMode,
        params::{KernelParams, SamplingSettings},
    };
    use ndarray::{Array3, Axis};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use std::time::Instant;

    fn headless_ctx() -> GpuContext {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        GpuContext::new_blocking(instance, None)
    }

    /// Helper: run convolve + affinity_growth on the GPU, return
    /// channel-major flat `u_out`.
    #[allow(clippy::too_many_arguments)]
    fn run_gpu_affinity_constant(
        ctx: &GpuContext,
        conv_pass: &ConvolvePass,
        ag_pass: &AffinityGrowthPass,
        activation: &Array3<f32>,
        params: &KernelParams,
        h: &[f32],
        border: BorderMode,
    ) -> Vec<f32> {
        let (gh, gw, gc) = activation.dim();
        let a_buf = upload_activation(ctx, activation);
        let kernels = upload_kernels(ctx, params);
        let pre_g_buf = ConvolvePass::allocate_pre_g(ctx, gh as u32, gw as u32, kernels.count);
        let u_out_buf = AffinityGrowthPass::allocate_u_out(ctx, gh as u32, gw as u32, gc as u32);
        let globals = GpuGlobals::new(
            gh as u32,
            gw as u32,
            gc as u32,
            kernels.count,
            kernels.max_side,
            border,
        );
        let conv_globals_buf = ConvolvePass::upload_globals(ctx, &globals);
        let ag_globals_buf = AffinityGrowthPass::upload_globals(ctx, &globals);
        let weights = GpuConstantWeights::from_slice(h);
        let weights_buf = upload_constant_weights(ctx, &weights);

        let conv_bg =
            conv_pass.make_bind_group(ctx, &a_buf, &kernels, &pre_g_buf, &conv_globals_buf);
        let ag_bg = ag_pass.make_bind_group(
            ctx,
            &pre_g_buf,
            &kernels,
            &weights_buf,
            &u_out_buf,
            &ag_globals_buf,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("affinity_growth constant test encoder"),
            });
        conv_pass.record(&mut enc, &conv_bg, gh as u32, gw as u32);
        ag_pass.record_constant(&mut enc, &ag_bg, gh as u32, gw as u32);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::Wait).unwrap();

        readback_buffer::<f32>(ctx, &u_out_buf, gh * gw * gc)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_gpu_affinity_localized(
        ctx: &GpuContext,
        conv_pass: &ConvolvePass,
        ag_pass: &AffinityGrowthPass,
        activation: &Array3<f32>,
        params: &KernelParams,
        p_map: &Array3<f32>,
        border: BorderMode,
    ) -> Vec<f32> {
        let (gh, gw, gc) = activation.dim();
        let a_buf = upload_activation(ctx, activation);
        let kernels = upload_kernels(ctx, params);
        let pre_g_buf = ConvolvePass::allocate_pre_g(ctx, gh as u32, gw as u32, kernels.count);
        let u_out_buf = AffinityGrowthPass::allocate_u_out(ctx, gh as u32, gw as u32, gc as u32);
        let globals = GpuGlobals::new(
            gh as u32,
            gw as u32,
            gc as u32,
            kernels.count,
            kernels.max_side,
            border,
        );
        let conv_globals_buf = ConvolvePass::upload_globals(ctx, &globals);
        let ag_globals_buf = AffinityGrowthPass::upload_globals(ctx, &globals);
        let p_buf = upload_localized_weights(ctx, p_map);

        let conv_bg =
            conv_pass.make_bind_group(ctx, &a_buf, &kernels, &pre_g_buf, &conv_globals_buf);
        let ag_bg = ag_pass.make_bind_group(
            ctx,
            &pre_g_buf,
            &kernels,
            &p_buf,
            &u_out_buf,
            &ag_globals_buf,
        );

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("affinity_growth localized test encoder"),
            });
        conv_pass.record(&mut enc, &conv_bg, gh as u32, gw as u32);
        ag_pass.record_localized(&mut enc, &ag_bg, gh as u32, gw as u32);
        ctx.queue.submit([enc.finish()]);
        ctx.device.poll(wgpu::PollType::Wait).unwrap();

        readback_buffer::<f32>(ctx, &u_out_buf, gh * gw * gc)
    }

    /// Compare GPU channel-major `(C, H, W)` against CPU `(H, W, C)`
    /// `UField`. Returns `(max_abs, max_rel)` and the **worst cell
    /// that violates both bounds** (the "effective" max-error cell).
    ///
    /// Combined `rel < rel_tol  OR  abs < abs_tol` is the standard
    /// pattern for cancellation-prone outputs (M1.6 jax_fixture_smoke
    /// uses the same). Eq. 7 with random `P_i ∈ [-1, 1]` can sum to
    /// values near 0 where relative error explodes despite an
    /// unobjectionable absolute error.
    fn compare_to_cpu(
        gpu_flat: &[f32],
        cpu_u: &Array3<f32>,
        rel_tol: f32,
        abs_tol: f32,
    ) -> (f32, f32) {
        let (h, w, c) = cpu_u.dim();
        assert_eq!(gpu_flat.len(), h * w * c);
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for ci in 0..c {
            for y in 0..h {
                for x in 0..w {
                    let gpu_v = gpu_flat[ci * h * w + y * w + x];
                    let cpu_v = cpu_u[[y, x, ci]];
                    let abs_err = (gpu_v - cpu_v).abs();
                    let rel_err = abs_err / cpu_v.abs().max(1e-6);
                    if abs_err > max_abs {
                        max_abs = abs_err;
                    }
                    if rel_err > max_rel {
                        max_rel = rel_err;
                    }
                    assert!(
                        rel_err < rel_tol || abs_err < abs_tol,
                        "({y}, {x}, c={ci}): gpu={gpu_v:e} cpu={cpu_v:e} \
                         abs={abs_err:.3e} rel={rel_err:.3e}"
                    );
                }
            }
        }
        let _ = Axis(0); // touch import — keeps it warning-free if reshuffled
        (max_abs, max_rel)
    }

    fn make_random_activation(rng: &mut ChaCha8Rng, h: usize, w: usize, c: usize) -> Array3<f32> {
        Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0_f32..1.0))
    }

    fn precompute_cpu_kernels(
        params: &KernelParams,
    ) -> (Vec<ndarray::Array2<f32>>, Vec<flow_lenia_core::KernelMeta>) {
        let kernels: Vec<ndarray::Array2<f32>> = params
            .kernels
            .iter()
            .map(|e| flow_lenia_core::compute_kernel(params.r_global, e))
            .collect();
        let meta: Vec<flow_lenia_core::KernelMeta> = (0..params.kernels.len())
            .map(|i| flow_lenia_core::KernelMeta::from_params(params, i))
            .collect();
        (kernels, meta)
    }

    #[test]
    fn affinity_growth_constant_matches_cpu() {
        let ctx = headless_ctx();
        let conv_pass = ConvolvePass::new(&ctx);
        let ag_pass = AffinityGrowthPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0xAFF1_C042);

        let (h, w, c, num_kernels) = (32_usize, 32_usize, 3_usize, 10_u32);
        let activation = make_random_activation(&mut rng, h, w, c);
        let params = KernelParams::sample_random(
            &mut rng,
            SamplingSettings {
                num_kernels,
                num_channels: c as u32,
            },
        );
        let h_vec: Vec<f32> = params.kernels.iter().map(|e| e.h).collect();

        let started = Instant::now();
        let gpu_flat = run_gpu_affinity_constant(
            &ctx,
            &conv_pass,
            &ag_pass,
            &activation,
            &params,
            &h_vec,
            BorderMode::Torus,
        );
        let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;

        let cpu_started = Instant::now();
        let (cpu_kernels, cpu_meta) = precompute_cpu_kernels(&params);
        let cpu_u = affinity_with_constant_weights(
            &activation,
            &cpu_kernels,
            &cpu_meta,
            &h_vec,
            BorderMode::Torus,
        );
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

        // Eq. 3 tolerance: 1e-4 relative OR 1e-5 absolute. Random h_i
        // is in [0.01, 1) (KernelEntry::sample_random), so cancellation
        // is mild — relative is usually the binding bound.
        let (max_abs, max_rel) = compare_to_cpu(&gpu_flat, &cpu_u, 1e-4, 1e-5);
        eprintln!(
            "[M2.4-Eq3] 32×32 torus C=3 K=10 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
             gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
        );
    }

    #[test]
    fn affinity_growth_localized_matches_cpu() {
        let ctx = headless_ctx();
        let conv_pass = ConvolvePass::new(&ctx);
        let ag_pass = AffinityGrowthPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0xAFF1_C043);

        let (h, w, c, num_kernels) = (32_usize, 32_usize, 3_usize, 10_u32);
        let activation = make_random_activation(&mut rng, h, w, c);
        let params = KernelParams::sample_random(
            &mut rng,
            SamplingSettings {
                num_kernels,
                num_channels: c as u32,
            },
        );
        // P_i(x) = random per cell per kernel.
        let p_map: Array3<f32> = Array3::from_shape_fn((h, w, num_kernels as usize), |_| {
            rng.gen_range(-1.0_f32..1.0)
        });

        let started = Instant::now();
        let gpu_flat = run_gpu_affinity_localized(
            &ctx,
            &conv_pass,
            &ag_pass,
            &activation,
            &params,
            &p_map,
            BorderMode::Torus,
        );
        let gpu_ms = started.elapsed().as_secs_f64() * 1000.0;

        let cpu_started = Instant::now();
        let (cpu_kernels, cpu_meta) = precompute_cpu_kernels(&params);
        let cpu_u = affinity_with_localized_weights(
            &activation,
            &cpu_kernels,
            &cpu_meta,
            &p_map,
            BorderMode::Torus,
        );
        let cpu_ms = cpu_started.elapsed().as_secs_f64() * 1000.0;

        // Eq. 7 tolerance: 1e-4 relative OR 1e-5 absolute. With
        // random P_i ∈ [-1, 1] the sum can cancel to near 0, where
        // relative error is ill-defined; the absolute fallback is
        // what matters in those cells.
        let (max_abs, max_rel) = compare_to_cpu(&gpu_flat, &cpu_u, 1e-4, 1e-5);
        eprintln!(
            "[M2.4-Eq7] 32×32 torus C=3 K=10 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}  \
             gpu={gpu_ms:.2}ms  cpu={cpu_ms:.2}ms"
        );
    }

    /// Locks the Eq. 3 ⊂ Eq. 7 specialisation: setting
    /// `P_i(x) ≡ h_i` everywhere produces the same `u_out` from both
    /// pipelines, **bit-equal** (the two shaders accumulate kernels in
    /// the same order, so f32 ordering matches exactly).
    #[test]
    fn affinity_growth_localized_with_uniform_p_equals_constant() {
        let ctx = headless_ctx();
        let conv_pass = ConvolvePass::new(&ctx);
        let ag_pass = AffinityGrowthPass::new(&ctx);
        let mut rng = ChaCha8Rng::seed_from_u64(0xAFF1_C044);

        let (h, w, c, num_kernels) = (16_usize, 16_usize, 3_usize, 8_u32);
        let activation = make_random_activation(&mut rng, h, w, c);
        let params = KernelParams::sample_random(
            &mut rng,
            SamplingSettings {
                num_kernels,
                num_channels: c as u32,
            },
        );
        let h_vec: Vec<f32> = params.kernels.iter().map(|e| e.h).collect();
        let p_map: Array3<f32> =
            Array3::from_shape_fn((h, w, num_kernels as usize), |(_, _, ki)| h_vec[ki]);

        let gpu_const = run_gpu_affinity_constant(
            &ctx,
            &conv_pass,
            &ag_pass,
            &activation,
            &params,
            &h_vec,
            BorderMode::Torus,
        );
        let gpu_loc = run_gpu_affinity_localized(
            &ctx,
            &conv_pass,
            &ag_pass,
            &activation,
            &params,
            &p_map,
            BorderMode::Torus,
        );
        assert_eq!(gpu_const.len(), gpu_loc.len());
        for (i, (&a, &b)) in gpu_const.iter().zip(gpu_loc.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "divergence at index {i}: constant={a} localized={b}"
            );
        }
    }
}
