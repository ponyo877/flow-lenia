//! Full Flow-Lenia step pipeline on the GPU (M2.8).
//!
//! Composes the five M2.3..M2.7 compute passes into one
//! `record_step` call. Mirrors `flow_lenia_core::step::step` on the
//! CPU side. The per-step buffer set is allocated once at
//! construction; bind groups are pre-built for both ping-pong
//! orientations so `step()` only chooses an index and submits.
//!
//! Pipeline order (DESIGN.md §3, same as CPU step):
//!
//! ```text
//! a_in --convolve--> pre_g
//! pre_g, kernel_meta, h_weights --affinity_growth--> u
//! u                            --gradient_u-------> grad_u
//! a_in                         --gradient_a_sum---> grad_a_sum
//! a_in, grad_u, grad_a_sum     --flow-------------> F
//! a_in, F                      --reintegrate------> a_out
//! ```
//!
//! Each `step()` flips a `ping` bit so the next step reads from what
//! was just written. M2.8 ships the **constant-weights variant**
//! only (`affinity_growth_constant`); the localised variant is wired
//! at M5 when parameter painting lands.

use crate::{
    activation_buffer::{flatten_activation_channel_major, readback_activation, upload_activation},
    globals::GpuGlobals,
    kernel_buffers::{upload_kernels, GpuKernelBuffers},
    passes::{
        affinity_growth::{upload_constant_weights, AffinityGrowthPass, GpuConstantWeights},
        convolve::ConvolvePass,
        flow::FlowPass,
        gradient::GradientPass,
        reintegrate::ReintegratePass,
    },
    GpuContext,
};
use bytemuck::cast_slice;
use flow_lenia_core::{config::FlowLeniaConfig, params::KernelParams, state::ActivationField};
use wgpu::util::DeviceExt;

/// One full Flow-Lenia step on the GPU. Owns every buffer and bind
/// group it needs for steady-state per-step recording.
///
/// Note (`dead_code`): most static-buffer fields are *only* referenced
/// indirectly via the bind groups built in `new` — the bind group
/// already holds an `Arc` to each buffer internally, so the field
/// isn't load-bearing for liveness. We keep them as named fields so
/// future M2.9 visualisation or M5 weights swaps can grab them by
/// name without rebuilding the pipeline; the `dead_code` allow is
/// intentional and not a TODO.
#[allow(dead_code)]
pub struct GpuStepPipeline {
    // Passes.
    convolve_pass: ConvolvePass,
    affinity_pass: AffinityGrowthPass,
    gradient_pass: GradientPass,
    flow_pass: FlowPass,
    reintegrate_pass: ReintegratePass,

    // Static buffers (re-used every step).
    kernel_buffers: GpuKernelBuffers,
    h_weights_buf: wgpu::Buffer,
    pre_g_buf: wgpu::Buffer,
    u_buf: wgpu::Buffer,
    grad_u_buf: wgpu::Buffer,
    grad_a_sum_buf: wgpu::Buffer,
    flow_field_buf: wgpu::Buffer,
    globals_buf: wgpu::Buffer,

    /// Ping-pong A buffers. `a_buffers[ping]` is read each step;
    /// `a_buffers[1 - ping]` is written.
    a_buffers: [wgpu::Buffer; 2],
    ping: usize,

    // Pre-built bind groups for both ping-pong orientations. Some
    // passes don't touch the ping-pong A buffers and need only one
    // bind group each (`affinity_bg`, `gradient_u_bg`).
    convolve_bgs: [wgpu::BindGroup; 2],
    affinity_bg: wgpu::BindGroup,
    gradient_u_bg: wgpu::BindGroup,
    gradient_a_sum_bgs: [wgpu::BindGroup; 2],
    flow_bgs: [wgpu::BindGroup; 2],
    reintegrate_bgs: [wgpu::BindGroup; 2],

    // Shape cache.
    height: u32,
    width: u32,
    channels: u32,
    step_count: u64,
}

impl GpuStepPipeline {
    /// Allocate every buffer + bind group and seed `a_buffers[0]`
    /// with `initial_a`.
    ///
    /// `cfg` and `kernel_params` must agree on `num_kernels` and
    /// `channels` (asserted). `initial_a` must match `cfg`'s grid
    /// shape.
    #[must_use]
    pub fn new(
        ctx: &GpuContext,
        cfg: &FlowLeniaConfig,
        kernel_params: &KernelParams,
        initial_a: &ActivationField,
    ) -> Self {
        let (h, w, c) = initial_a.dim();
        assert_eq!(
            (h, w, c),
            (
                cfg.grid_height as usize,
                cfg.grid_width as usize,
                cfg.channels as usize
            ),
            "initial_a shape mismatch with cfg grid"
        );
        assert_eq!(
            kernel_params.kernels.len(),
            cfg.num_kernels as usize,
            "kernel_params.len() != cfg.num_kernels"
        );

        let height = h as u32;
        let width = w as u32;
        let channels = c as u32;

        // ── Passes ─────────────────────────────────────────────
        let convolve_pass = ConvolvePass::new(ctx);
        let affinity_pass = AffinityGrowthPass::new(ctx);
        let gradient_pass = GradientPass::new(ctx);
        let flow_pass = FlowPass::new(ctx);
        let reintegrate_pass = ReintegratePass::new(ctx);

        // ── Kernel + weights ───────────────────────────────────
        let kernel_buffers = upload_kernels(ctx, kernel_params);
        let h_vec: Vec<f32> = kernel_params.kernels.iter().map(|e| e.h).collect();
        let h_weights = GpuConstantWeights::from_slice(&h_vec);
        let h_weights_buf = upload_constant_weights(ctx, &h_weights);

        // ── Static per-step buffers ────────────────────────────
        let pre_g_buf = ConvolvePass::allocate_pre_g(ctx, height, width, kernel_buffers.count);
        let u_buf = AffinityGrowthPass::allocate_u_out(ctx, height, width, channels);
        let grad_u_buf = GradientPass::allocate_grad_u(ctx, height, width, channels);
        let grad_a_sum_buf = GradientPass::allocate_grad_a_sum(ctx, height, width);
        let flow_field_buf = FlowPass::allocate_f_out(ctx, height, width, channels);

        // ── Globals ────────────────────────────────────────────
        let globals = GpuGlobals::new(
            height,
            width,
            channels,
            kernel_buffers.count,
            kernel_buffers.max_side,
            cfg.border,
        )
        .with_paper_strict(cfg.paper_strict)
        .with_beta_a(cfg.beta_a)
        .with_n(cfg.n)
        .with_dd(cfg.dd)
        .with_sigma(cfg.sigma)
        .with_dt(cfg.dt);
        let globals_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("pipeline globals"),
                contents: cast_slice(std::slice::from_ref(&globals)),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });

        // ── A ping-pong buffers ────────────────────────────────
        // Buffer 0 holds the initial state; buffer 1 starts zero.
        let a_buffer_0 = upload_activation(ctx, initial_a);
        let a_buffer_1 = ReintegratePass::allocate_a(ctx, height, width, channels);
        // Convolve / gradient need their A input as `STORAGE | COPY_*`.
        // `upload_activation` already provides STORAGE | COPY_DST | COPY_SRC;
        // `allocate_a` (M2.7) does the same. Both buffers therefore
        // satisfy every binding contract on either ping-pong side.

        let a_buffers = [a_buffer_0, a_buffer_1];

        // ── Bind groups ────────────────────────────────────────
        // Two-orientation BGs (input A flips between buffers[0] and
        // buffers[1]).
        let convolve_bgs = [
            convolve_pass.make_bind_group(
                ctx,
                &a_buffers[0],
                &kernel_buffers,
                &pre_g_buf,
                &globals_buf,
            ),
            convolve_pass.make_bind_group(
                ctx,
                &a_buffers[1],
                &kernel_buffers,
                &pre_g_buf,
                &globals_buf,
            ),
        ];
        let gradient_a_sum_bgs = [
            gradient_pass.make_bind_group_a_sum(ctx, &a_buffers[0], &grad_a_sum_buf, &globals_buf),
            gradient_pass.make_bind_group_a_sum(ctx, &a_buffers[1], &grad_a_sum_buf, &globals_buf),
        ];
        let flow_bgs = [
            flow_pass.make_bind_group(
                ctx,
                &a_buffers[0],
                &grad_u_buf,
                &grad_a_sum_buf,
                &flow_field_buf,
                &globals_buf,
            ),
            flow_pass.make_bind_group(
                ctx,
                &a_buffers[1],
                &grad_u_buf,
                &grad_a_sum_buf,
                &flow_field_buf,
                &globals_buf,
            ),
        ];
        let reintegrate_bgs = [
            reintegrate_pass.make_bind_group(
                ctx,
                &a_buffers[0],
                &flow_field_buf,
                &a_buffers[1],
                &globals_buf,
            ),
            reintegrate_pass.make_bind_group(
                ctx,
                &a_buffers[1],
                &flow_field_buf,
                &a_buffers[0],
                &globals_buf,
            ),
        ];

        // One-orientation BGs (don't touch ping-pong A).
        let affinity_bg = affinity_pass.make_bind_group(
            ctx,
            &pre_g_buf,
            &kernel_buffers,
            &h_weights_buf,
            &u_buf,
            &globals_buf,
        );
        let gradient_u_bg = gradient_pass.make_bind_group_u(ctx, &u_buf, &grad_u_buf, &globals_buf);

        Self {
            convolve_pass,
            affinity_pass,
            gradient_pass,
            flow_pass,
            reintegrate_pass,
            kernel_buffers,
            h_weights_buf,
            pre_g_buf,
            u_buf,
            grad_u_buf,
            grad_a_sum_buf,
            flow_field_buf,
            globals_buf,
            a_buffers,
            ping: 0,
            convolve_bgs,
            affinity_bg,
            gradient_u_bg,
            gradient_a_sum_bgs,
            flow_bgs,
            reintegrate_bgs,
            height,
            width,
            channels,
            step_count: 0,
        }
    }

    /// Append one full step into `encoder`. **Does not flip `ping`** —
    /// callers must invoke [`swap_buffers`](Self::swap_buffers) after
    /// they submit the encoder for the next step to read the freshly
    /// written buffer.
    pub fn record_step(&self, encoder: &mut wgpu::CommandEncoder) {
        let h = self.height;
        let w = self.width;
        let p = self.ping;
        // Order: convolve → affinity → grad_u → grad_a_sum → flow → reintegrate.
        self.convolve_pass
            .record(encoder, &self.convolve_bgs[p], h, w);
        self.affinity_pass
            .record_constant(encoder, &self.affinity_bg, h, w);
        self.gradient_pass
            .record_u(encoder, &self.gradient_u_bg, h, w);
        self.gradient_pass
            .record_a_sum(encoder, &self.gradient_a_sum_bgs[p], h, w);
        self.flow_pass.record(encoder, &self.flow_bgs[p], h, w);
        self.reintegrate_pass
            .record(encoder, &self.reintegrate_bgs[p], h, w);
    }

    /// Flip the ping-pong index. Call this **after submitting** the
    /// encoder that contains a `record_step` so that the destination
    /// buffer just written becomes the next step's source.
    pub fn swap_buffers(&mut self) {
        self.ping ^= 1;
        self.step_count += 1;
    }

    /// Convenience: record + submit + swap for one step. Holds an
    /// `&GpuContext` only for the submit; no readback is performed.
    pub fn step(&mut self, ctx: &GpuContext) {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GpuStepPipeline step encoder"),
            });
        self.record_step(&mut enc);
        ctx.queue.submit([enc.finish()]);
        self.swap_buffers();
    }

    /// Run `n` steps back-to-back. Blocks on a final `poll(Wait)` so
    /// callers can immediately read back the final state.
    pub fn run_steps(&mut self, ctx: &GpuContext, n: u32) {
        for _ in 0..n {
            self.step(ctx);
        }
        ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    }

    /// Number of steps taken since construction.
    #[must_use]
    pub fn step_count(&self) -> u64 {
        self.step_count
    }

    /// Push fresh `cfg` values (paper_strict / border / dt / dd /
    /// sigma / n / beta_a) into the live uniform buffer without
    /// rebuilding any bind groups. Cheap enough to call every frame
    /// when a UI slider drags. Grid shape / kernel count come from
    /// the existing pipeline state and are intentionally ignored —
    /// changing those needs a full `GpuStepPipeline::new` (see M4.4
    /// "Apply" / "New Seed" paths in `flow-lenia-web`).
    pub fn update_globals(&self, ctx: &GpuContext, cfg: &FlowLeniaConfig) {
        let globals = GpuGlobals::new(
            self.height,
            self.width,
            self.channels,
            self.kernel_buffers.count,
            self.kernel_buffers.max_side,
            cfg.border,
        )
        .with_paper_strict(cfg.paper_strict)
        .with_beta_a(cfg.beta_a)
        .with_n(cfg.n)
        .with_dd(cfg.dd)
        .with_sigma(cfg.sigma)
        .with_dt(cfg.dt);
        ctx.queue.write_buffer(
            &self.globals_buf,
            0,
            cast_slice(std::slice::from_ref(&globals)),
        );
    }

    /// The buffer that holds the current activation state — the one
    /// `record_step` will read from next. Use this when wiring a
    /// downstream visualization pass; for CPU-side comparison call
    /// [`readback_activation`](Self::readback_activation) instead.
    #[must_use]
    pub fn current_activation_buffer(&self) -> &wgpu::Buffer {
        &self.a_buffers[self.ping]
    }

    /// Borrow one of the two ping-pong A buffers by index (0 or 1).
    /// Lets downstream code pre-build a `[BindGroup; 2]` pair indexed
    /// by [`ping_index`](Self::ping_index) so no per-frame bind-group
    /// rebuild is needed when running the visualisation pass on top
    /// of the simulator. Callers should panic on `index >= 2`.
    ///
    /// # Panics
    ///
    /// Panics if `index >= 2`.
    #[must_use]
    pub fn a_buffer(&self, index: usize) -> &wgpu::Buffer {
        &self.a_buffers[index]
    }

    /// Current ping-pong index — `0` or `1`. Equal to the index of
    /// the buffer returned by [`current_activation_buffer`](Self::current_activation_buffer)
    /// and (after each `step`) the buffer that holds the *latest*
    /// activation. Pair with [`a_buffer`](Self::a_buffer) to pick the
    /// right pre-built bind group each frame.
    #[must_use]
    pub fn ping_index(&self) -> usize {
        self.ping
    }

    /// Copy the current activation buffer back to the CPU as a fresh
    /// `(H, W, C)` `ActivationField`.
    #[must_use]
    pub fn readback_activation(&self, ctx: &GpuContext) -> ActivationField {
        readback_activation(
            ctx,
            self.current_activation_buffer(),
            self.height as usize,
            self.width as usize,
            self.channels as usize,
        )
    }

    /// Replace the current activation buffer's contents with
    /// `new_a`. Useful for tests / fixture seeding without
    /// rebuilding the whole pipeline.
    pub fn upload_activation_to_current(&self, ctx: &GpuContext, new_a: &ActivationField) {
        let flat = flatten_activation_channel_major(new_a);
        ctx.queue
            .write_buffer(self.current_activation_buffer(), 0, cast_slice(&flat));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flow_lenia_core::{
        config::{BorderMode, MixRule},
        FlowLeniaSimulator,
    };

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    fn small_cfg(channels: u32, paper_strict: bool, border: BorderMode) -> FlowLeniaConfig {
        FlowLeniaConfig {
            grid_width: 32,
            grid_height: 32,
            channels,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 6,
            paper_strict,
            border,
            mix_rule: MixRule::Stochastic,
        }
    }

    /// Build CPU simulator and GPU pipeline from the same `(cfg, seed)`,
    /// run them for `n` steps, and compare per-cell with the requested
    /// tolerance.
    /// M6.C-0: ValidationGuard assertion is performed inside this
    /// helper rather than the 4 callers (gpu_pipeline_swap /
    /// _ten_steps / _paper_strict / _wall_border), since the helper
    /// owns the GpuContext lifetime and callers would otherwise
    /// duplicate the guard machinery.
    fn compare_run(
        cfg: &FlowLeniaConfig,
        seed: u64,
        n_steps: u32,
        rel_tol: f32,
        abs_tol: f32,
    ) -> (f32, f32) {
        let (ctx, guard) = headless_ctx();
        let mut cpu_sim = FlowLeniaSimulator::new(*cfg, seed);
        let initial_a = cpu_sim.activation().clone();
        let kernel_params = cpu_sim.kernel_params().clone();
        let mut gpu_pipeline = GpuStepPipeline::new(&ctx, cfg, &kernel_params, &initial_a);

        cpu_sim.step_many(n_steps);
        gpu_pipeline.run_steps(&ctx, n_steps);
        let gpu_a = gpu_pipeline.readback_activation(&ctx);
        let cpu_a = cpu_sim.activation();

        let (h, w, c) = cpu_a.dim();
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for ((y, x, ci), &cpu_v) in cpu_a.indexed_iter() {
            let gpu_v = gpu_a[[y, x, ci]];
            let abs_err = (gpu_v - cpu_v).abs();
            let rel_err = abs_err / cpu_v.abs().max(1e-6);
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            assert!(
                rel_err < rel_tol || abs_err < abs_tol,
                "({y}, {x}, c={ci}) after {n_steps} steps: gpu={gpu_v} cpu={cpu_v} \
                 abs={abs_err:.3e} rel={rel_err:.3e}"
            );
        }
        let _ = (h, w, c);

        if let Some(g) = &guard {
            g.assert_no_errors();
        }

        (max_abs, max_rel)
    }

    /// One step: ping-pong index moves from 0 to 1, readback should
    /// reflect the *destination* buffer not the source.
    #[test]
    fn gpu_pipeline_swap_alternates_buffers() {
        let cfg = small_cfg(3, false, BorderMode::Torus);
        // Run 2 steps so we exercise *both* ping-pong orientations
        // (forward then backward) and confirm the per-step output
        // continues the same trajectory the CPU computes.
        let (max_abs, max_rel) = compare_run(&cfg, 0x5_AA_5A, 2, 1e-4, 1e-5);
        eprintln!("[M2.8-swap]  2-step C=3 torus : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}");
    }

    /// 10-step trajectory comparison — exercises ping-pong many times.
    #[test]
    fn gpu_pipeline_ten_steps_match_cpu() {
        let cfg = small_cfg(3, false, BorderMode::Torus);
        let (max_abs, max_rel) = compare_run(&cfg, 0x6_BB_6B, 10, 1e-3, 1e-4);
        eprintln!("[M2.8-10st]  10-step C=3 torus : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}");
    }

    /// `paper_strict = true` path through the full pipeline.
    #[test]
    fn gpu_pipeline_paper_strict_matches_cpu() {
        let cfg = small_cfg(3, true, BorderMode::Torus);
        let (max_abs, max_rel) = compare_run(&cfg, 0x7_CC_7C, 5, 1e-3, 1e-4);
        eprintln!(
            "[M2.8-ps]    5-step C=3 torus paper-strict : \
             max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );
    }

    /// Wall border full-pipeline check.
    #[test]
    fn gpu_pipeline_wall_border_matches_cpu() {
        let cfg = small_cfg(3, false, BorderMode::Wall);
        let (max_abs, max_rel) = compare_run(&cfg, 0x8_DD_8D, 5, 1e-3, 1e-4);
        eprintln!("[M2.8-wall]  5-step C=3 wall : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}");
    }
}
