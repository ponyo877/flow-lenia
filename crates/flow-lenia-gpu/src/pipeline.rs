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
        convolve_fft::ConvolveFftPass,
        fft::{is_fft_pipeline_grid, SUPPORTED_N},
        flow::FlowPass,
        gradient::GradientPass,
        kernel_fft::{precompute_kernel_ffts, KernelFftBuffers},
        parameter_flow::ParameterFlowPass,
        reintegrate::ReintegratePass,
        spectral_multiply::SpectralMultiplyPass,
    },
    GpuContext,
};
use bytemuck::cast_slice;
use flow_lenia_core::{config::FlowLeniaConfig, params::KernelParams, state::ActivationField};
use wgpu::util::DeviceExt;

/// Which convolution algorithm `GpuStepPipeline` uses per step.
/// Default is [`ConvolveMode::Auto`] (M6.C-1-5-b): pick `Fft` when
/// the grid is in `SUPPORTED_N` (= {64, 256}) and `Direct` otherwise,
/// so callers requesting grid 32 / 128 / 512 (mixed-radix sizes the
/// FFT primitive does not yet handle) silently fall back to the
/// direct path. This preserves existing test sweeps + UI grid
/// options while making FFT the primary path for the supported
/// grid sizes.
///
/// **M6.C-1-5-a status**: `Fft` supports `C >= 1` (per-kernel
/// source-channel routing via `ConvolveFftPass::kernel_routing_buf`).
/// Mixed-radix grid sizes (32, 128, 512) remain out of scope per
/// the FFT primitive's pure-radix-4 constraint (M6.C-1-2
/// scope-guardian deferred); the C-1-5-b auto-fallback keeps the
/// direct path for those grids as a deprecation path rather than a
/// regression.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum ConvolveMode {
    /// M2.3 direct convolution pass. **Deprecated as of C-1-5-b**:
    /// kept as the `Auto` fall-back for grid sizes the FFT primitive
    /// does not yet support (32 / 128 / 512). Will be removed once
    /// the mixed-radix FFT lands (M6.C-1 後半 or M5).
    Direct,
    /// FFT-based convolution via [`ConvolveFftPass`] (kernel pre-FFT
    /// + spectral multiply + per-kernel inverse FFT + layout
    /// transpose). C >= 1 with per-kernel `source_channel` routing
    /// (C-1-5-a). Grid must be in `SUPPORTED_N` = {64, 256}.
    Fft,
    /// **Default** (M6.C-1-5-b): pick `Fft` if the grid is in
    /// `SUPPORTED_N`, else fall back to `Direct`. The actual choice
    /// is resolved at `new_with_mode()` time by
    /// [`ConvolveMode::resolve`], so the runtime field always holds
    /// `Direct` or `Fft` (never `Auto`).
    #[default]
    Auto,
}

impl ConvolveMode {
    /// Auto-fallback resolution: `Auto` collapses to `Fft` when
    /// `grid` is FFT-pipeline-capable ({64, 256} radix-4 + 512
    /// mixed-radix, see [`is_fft_pipeline_grid`]), else `Direct`.
    /// Explicit `Direct` / `Fft` are passed through. Callers that
    /// explicitly pick `Fft` for an unsupported grid will hit the
    /// FFT-side assertion at `ConvolveFftPass::new` — that path is
    /// intentional.
    #[must_use]
    pub fn resolve(self, grid: u32) -> ConvolveMode {
        match self {
            ConvolveMode::Auto => {
                if is_fft_pipeline_grid(grid) {
                    ConvolveMode::Fft
                } else {
                    ConvolveMode::Direct
                }
            }
            other => other,
        }
    }
}

/// Which affinity-growth variant the per-step pipeline uses (paper
/// Eq. 3 vs Eq. 7).
///
/// **M6.C-2-4-d**: `Localized` is the production wiring of the
/// existing M2.4 `pipeline_localized` (Eq. 7 per-cell `P_i(x)`)
/// plus the C-2-4-c `ParameterFlowPass` (identity-copy + M5 Eq. 8
/// hook). Multi-creature simulation goes through this path; default
/// remains `Constant` so existing single-creature tests / UI stay
/// on the Eq. 3 path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Default)]
pub enum AffinityMode {
    /// Paper Eq. 3: one constant `h_i` per kernel. M2.4 default;
    /// used by every `gpu_pipeline_*` legacy test and by the
    /// flow-lenia-app single-creature UI.
    #[default]
    Constant,
    /// Paper Eq. 7: per-cell `P_i(x)` map (4 creature M6.C-2-4
    /// scope). Pipeline holds two ping-pong P buffers and runs
    /// `ParameterFlowPass` after `ReintegratePass` each step so
    /// `P` follows the same ping-pong cadence as `A` (identity
    /// copy until M5 wires Eq. 8 stochastic sampling into the WGSL
    /// hook block — see `docs/M6_C2_4_creature_design.md` §"M5
    /// hook specification").
    Localized,
}

/// Localized-mode state bundle. Present iff
/// `affinity_mode == AffinityMode::Localized`.
struct LocalizedState {
    parameter_flow_pass: ParameterFlowPass,
    /// Ping-pong P map buffers (same indexing convention as
    /// `a_buffers`: `p_buffers[ping]` is the *source* each step,
    /// `p_buffers[1 - ping]` is the destination).
    p_buffers: [wgpu::Buffer; 2],
    /// Per-ping affinity bind groups — binding 2 (`p_map`) points
    /// at the matching ping's P buffer. Replaces the
    /// `Constant`-mode single `affinity_bg`.
    affinity_localized_bgs: [wgpu::BindGroup; 2],
    /// Per-ping parameter-flow bind groups — `p_in` =
    /// `p_buffers[ping]`, `p_out` = `p_buffers[1 - ping]`,
    /// `matter_flow` = `flow_field_buf`, `kernel_routing` =
    /// `kernel_routing_buf`, `globals` = `globals_buf`.
    parameter_flow_bgs: [wgpu::BindGroup; 2],
    /// `array<u32>` length K, `kernel_routing[k] = c0` (source
    /// channel of kernel k). Held live so the bind group's internal
    /// buffer reference stays valid; never read directly by Rust
    /// (used only via the bind group's binding 3 in
    /// `parameter_flow.wgsl`). Currently unused inside the WGSL
    /// identity copy; reserved for M5 Eq. 8 creature-competition
    /// semantics (see `parameter_flow.wgsl` header).
    #[allow(dead_code)]
    kernel_routing_buf: wgpu::Buffer,
}

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

    // M6.C-1-4-b: FFT-mode passes + kernel pre-FFT buffer. Both
    // populated only when `convolve_mode == ConvolveMode::Fft`.
    convolve_fft_pass: Option<ConvolveFftPass>,
    kernel_fft: Option<KernelFftBuffers>,

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
    convolve_mode: ConvolveMode,

    // M6.C-2-4-d: Localized (Eq. 7 + ParameterFlowPass) state.
    // `Some` iff `affinity_mode == AffinityMode::Localized`.
    affinity_mode: AffinityMode,
    localized: Option<LocalizedState>,
}

impl GpuStepPipeline {
    /// Allocate every buffer + bind group and seed `a_buffers[0]`
    /// with `initial_a`.
    ///
    /// `cfg` and `kernel_params` must agree on `num_kernels` and
    /// `channels` (asserted). `initial_a` must match `cfg`'s grid
    /// shape.
    /// Default constructor: uses [`ConvolveMode::Auto`] which picks
    /// `Fft` for grid ∈ {64, 256} and `Direct` otherwise. M6.C-1-5-b
    /// switched the default from `Direct` to `Auto` so the FFT path
    /// becomes primary while preserving existing grid sweeps for the
    /// mixed-radix sizes the FFT primitive does not yet handle
    /// (C-1-2 scope-guardian deferred).
    #[must_use]
    pub fn new(
        ctx: &GpuContext,
        cfg: &FlowLeniaConfig,
        kernel_params: &KernelParams,
        initial_a: &ActivationField,
    ) -> Self {
        Self::new_with_modes(
            ctx,
            cfg,
            kernel_params,
            initial_a,
            ConvolveMode::Auto,
            AffinityMode::Constant,
            None,
        )
    }

    /// M6.C-1-4-b explicit-mode constructor. For `ConvolveMode::Fft`
    /// the C=1 + grid ∈ {64, 256} restrictions apply (see
    /// [`ConvolveMode`] rustdoc); violating them panics with a
    /// pointer to the limitation. The bigger startup cost on this
    /// path is the K-kernel forward 2D FFT precompute (M6.C-1-3),
    /// done here so per-step `record_step_fft` issues no allocations
    /// or readbacks of its own (modulo the per-call uniform / bind-
    /// group allocations honestly framed in the helpers' rustdocs).
    #[must_use]
    pub fn new_with_mode(
        ctx: &GpuContext,
        cfg: &FlowLeniaConfig,
        kernel_params: &KernelParams,
        initial_a: &ActivationField,
        convolve_mode: ConvolveMode,
    ) -> Self {
        Self::new_with_modes(
            ctx,
            cfg,
            kernel_params,
            initial_a,
            convolve_mode,
            AffinityMode::Constant,
            None,
        )
    }

    /// M6.C-2-4-d full-mode constructor. Takes both
    /// [`ConvolveMode`] and [`AffinityMode`]. For
    /// `AffinityMode::Localized` the caller must supply
    /// `initial_p_map` (`H * W * K` row-major `(y, x, ki)` flat,
    /// matching [`parameter_map::build_for_patches`] output);
    /// `None` is asserted away. For `AffinityMode::Constant` the
    /// `initial_p_map` argument is ignored.
    ///
    /// Localized state allocated here:
    /// - `[wgpu::Buffer; 2]` ping-pong P maps
    /// - `ParameterFlowPass` + per-ping bind groups
    /// - `kernel_routing` buffer (length-K `u32`, `c0` per kernel)
    /// - Per-ping `affinity_growth.pipeline_localized` bind groups
    ///   (binding 2 = current ping's P buffer)
    #[must_use]
    #[allow(clippy::too_many_arguments)]
    pub fn new_with_modes(
        ctx: &GpuContext,
        cfg: &FlowLeniaConfig,
        kernel_params: &KernelParams,
        initial_a: &ActivationField,
        convolve_mode: ConvolveMode,
        affinity_mode: AffinityMode,
        initial_p_map: Option<&[f32]>,
    ) -> Self {
        // M6.C-1-5-b auto-fallback: `Auto` resolves to `Fft` for
        // supported grids, `Direct` otherwise. The stored field will
        // be the resolved variant; tests can call
        // `pipeline.convolve_mode()` to see what actually ran.
        let convolve_mode = convolve_mode.resolve(cfg.grid_width);
        if convolve_mode == ConvolveMode::Fft {
            // M6.C-1-5-a: channels >= 1 now supported (per-kernel
            // source_channel routing via ConvolveFftPass kernel_routing_buf).
            assert!(
                cfg.channels >= 1,
                "ConvolveMode::Fft requires cfg.channels >= 1 (got {})",
                cfg.channels
            );
            assert!(
                is_fft_pipeline_grid(cfg.grid_width),
                "ConvolveMode::Fft requires cfg.grid_width ∈ {SUPPORTED_N:?} (radix-4) \
                 or 512 (mixed-radix, M6.C-3-2) (got {}); 32/128 mixed-radix are \
                 primitive-capable but kept on the Direct fallback in the pipeline.",
                cfg.grid_width
            );
            assert_eq!(
                cfg.grid_height, cfg.grid_width,
                "ConvolveMode::Fft requires square grid (got {}×{})",
                cfg.grid_height, cfg.grid_width
            );
        }
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

        // M6.C-2-4-d: Localized mode state. Only allocated when
        // `affinity_mode == Localized`; the Constant default leaves
        // every Localized buffer unallocated so existing tests pay
        // zero memory + zero pipeline-construction cost.
        let localized = match affinity_mode {
            AffinityMode::Constant => {
                assert!(
                    initial_p_map.is_none(),
                    "AffinityMode::Constant must not be paired with initial_p_map; \
                     pass None"
                );
                None
            }
            AffinityMode::Localized => {
                let p_initial = initial_p_map.expect(
                    "AffinityMode::Localized requires initial_p_map (H * W * K row-major \
                     flat, matching parameter_map::build_for_patches output)",
                );
                let p_len_expected =
                    (height as usize) * (width as usize) * (kernel_buffers.count as usize);
                assert_eq!(
                    p_initial.len(),
                    p_len_expected,
                    "initial_p_map length {} does not match H*W*K = {p_len_expected}",
                    p_initial.len()
                );

                // Two ping-pong P buffers. Both allocated via
                // `ParameterFlowPass::allocate_p` so they each carry
                // STORAGE | COPY_SRC | COPY_DST and either can be
                // read back by tests at any ping orientation (the
                // identity-copy semantics here mean `p_buffers[0]`
                // and `p_buffers[1]` should hold the same map once
                // any step has run, but allocation symmetry is the
                // contract; see also `upload_parameter_map_buf`
                // standalone helper which is COPY_DST-only and
                // suitable for downstream callers that own their own
                // readback path).
                let p_buffer_0 =
                    ParameterFlowPass::allocate_p(ctx, height, width, kernel_buffers.count);
                let p_buffer_1 =
                    ParameterFlowPass::allocate_p(ctx, height, width, kernel_buffers.count);
                ctx.queue.write_buffer(&p_buffer_0, 0, cast_slice(p_initial));
                let p_buffers = [p_buffer_0, p_buffer_1];

                // kernel_routing buffer (length K, u32, source channel
                // per kernel). Reused for the parameter_flow binding
                // 3 hook (M5 creature competition).
                let routing: Vec<u32> = kernel_params
                    .kernels
                    .iter()
                    .map(|e| e.c0 as u32)
                    .collect();
                let kernel_routing_buf =
                    SpectralMultiplyPass::upload_kernel_routing(ctx, &routing);

                let parameter_flow_pass = ParameterFlowPass::new(ctx);

                // Per-ping affinity bind groups: binding 2 = current
                // ping's P buffer.
                let affinity_localized_bgs = [
                    affinity_pass.make_bind_group(
                        ctx,
                        &pre_g_buf,
                        &kernel_buffers,
                        &p_buffers[0],
                        &u_buf,
                        &globals_buf,
                    ),
                    affinity_pass.make_bind_group(
                        ctx,
                        &pre_g_buf,
                        &kernel_buffers,
                        &p_buffers[1],
                        &u_buf,
                        &globals_buf,
                    ),
                ];

                // Per-ping parameter-flow bind groups: p_in / p_out
                // ping-pong, matter_flow = flow_field_buf, routing
                // buffer + globals shared.
                let parameter_flow_bgs = [
                    parameter_flow_pass.make_bind_group(
                        ctx,
                        &p_buffers[0],
                        &p_buffers[1],
                        &flow_field_buf,
                        &kernel_routing_buf,
                        &globals_buf,
                    ),
                    parameter_flow_pass.make_bind_group(
                        ctx,
                        &p_buffers[1],
                        &p_buffers[0],
                        &flow_field_buf,
                        &kernel_routing_buf,
                        &globals_buf,
                    ),
                ];

                Some(LocalizedState {
                    parameter_flow_pass,
                    p_buffers,
                    affinity_localized_bgs,
                    parameter_flow_bgs,
                    kernel_routing_buf,
                })
            }
        };

        // M6.C-1-4-b: build the FFT-mode passes + kernel pre-FFT
        // buffer iff caller selected FFT mode. The assertions above
        // already guaranteed the cfg shape is FFT-compatible.
        let (convolve_fft_pass, kernel_fft) = match convolve_mode {
            ConvolveMode::Direct => (None, None),
            ConvolveMode::Fft => {
                let n = cfg.grid_width;
                let fft = ConvolveFftPass::new(ctx, n, cfg.channels, kernel_params);
                let kfft =
                    precompute_kernel_ffts(ctx, kernel_params, n, &fft.fft2d, &fft.twiddles);
                (Some(fft), Some(kfft))
            }
            // Auto was resolved earlier in this function.
            ConvolveMode::Auto => unreachable!("Auto resolved at function entry"),
        };

        Self {
            convolve_pass,
            affinity_pass,
            gradient_pass,
            flow_pass,
            reintegrate_pass,
            convolve_fft_pass,
            kernel_fft,
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
            convolve_mode,
            affinity_mode,
            localized,
        }
    }

    /// Append one full step into `encoder`. **Does not flip `ping`** —
    /// callers must invoke [`swap_buffers`](Self::swap_buffers) after
    /// they submit the encoder for the next step to read the freshly
    /// written buffer.
    pub fn record_step(&self, encoder: &mut wgpu::CommandEncoder) {
        assert_eq!(
            self.convolve_mode,
            ConvolveMode::Direct,
            "record_step() supports Direct mode only. For Fft mode use \
             step() (which routes to record_step_fft internally — the FFT \
             path requires a &GpuContext for per-call uniform / bind-group \
             allocations, honestly framed in ConvolveFftPass + scratch \
             helper rustdocs)."
        );
        let h = self.height;
        let w = self.width;
        let p = self.ping;
        // Order: convolve → affinity → grad_u → grad_a_sum → flow →
        //   reintegrate → (Localized only) parameter_flow.
        self.convolve_pass
            .record(encoder, &self.convolve_bgs[p], h, w);
        self.record_affinity(encoder, p, h, w);
        self.gradient_pass
            .record_u(encoder, &self.gradient_u_bg, h, w);
        self.gradient_pass
            .record_a_sum(encoder, &self.gradient_a_sum_bgs[p], h, w);
        self.flow_pass.record(encoder, &self.flow_bgs[p], h, w);
        self.reintegrate_pass
            .record(encoder, &self.reintegrate_bgs[p], h, w);
        self.record_parameter_flow_if_localized(encoder, p, h, w);
    }

    /// M6.C-1-4-b FFT-mode per-step recording. Same downstream
    /// passes as `record_step`, only the convolve sub-step differs:
    /// FFT path = forward 2D + spectral multiply + per-kernel
    /// inverse 2D + layout transpose, writing the same
    /// `pre_g[y * W * K + x * K + ki]` layout the downstream
    /// `affinity_growth` pass expects.
    fn record_step_fft(&self, ctx: &GpuContext, encoder: &mut wgpu::CommandEncoder) {
        let h = self.height;
        let w = self.width;
        let p = self.ping;
        let fft = self.convolve_fft_pass.as_ref().expect(
            "record_step_fft requires ConvolveMode::Fft; \
             convolve_fft_pass populated in new_with_mode",
        );
        let kfft = self
            .kernel_fft
            .as_ref()
            .expect("record_step_fft requires ConvolveMode::Fft; kernel_fft populated in new_with_mode");
        // Convolve: FFT path. input_a is the current ping-pong
        // buffer (channel 0 since C=1 enforced in new_with_mode).
        fft.record(
            ctx,
            encoder,
            &self.a_buffers[p],
            &kfft.buffer,
            &self.pre_g_buf,
        );
        // Downstream passes are identical to Direct mode (modulo the
        // shared Localized branch).
        self.record_affinity(encoder, p, h, w);
        self.gradient_pass
            .record_u(encoder, &self.gradient_u_bg, h, w);
        self.gradient_pass
            .record_a_sum(encoder, &self.gradient_a_sum_bgs[p], h, w);
        self.flow_pass.record(encoder, &self.flow_bgs[p], h, w);
        self.reintegrate_pass
            .record(encoder, &self.reintegrate_bgs[p], h, w);
        self.record_parameter_flow_if_localized(encoder, p, h, w);
    }

    /// Dispatch the per-ping affinity-growth pass (Eq. 3 or Eq. 7
    /// depending on `affinity_mode`). Shared between Direct and FFT
    /// convolve paths.
    fn record_affinity(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ping: usize,
        h: u32,
        w: u32,
    ) {
        match (&self.affinity_mode, &self.localized) {
            (AffinityMode::Constant, _) => {
                self.affinity_pass
                    .record_constant(encoder, &self.affinity_bg, h, w);
            }
            (AffinityMode::Localized, Some(loc)) => {
                self.affinity_pass.record_localized(
                    encoder,
                    &loc.affinity_localized_bgs[ping],
                    h,
                    w,
                );
            }
            (AffinityMode::Localized, None) => {
                unreachable!(
                    "AffinityMode::Localized without LocalizedState — new_with_modes \
                     should have allocated it"
                );
            }
        }
    }

    /// Append one [`ParameterFlowPass`] dispatch iff Localized mode
    /// is active. No-op in `Constant`. **Order**: must run after
    /// `reintegrate` so the freshly-written matter-flow informs M5's
    /// Eq. 8 hook. The identity copy here is order-independent, but
    /// the order is fixed now so M5 can plug in without rewriting
    /// the step pipeline.
    fn record_parameter_flow_if_localized(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        ping: usize,
        h: u32,
        w: u32,
    ) {
        if let Some(loc) = self.localized.as_ref() {
            loc.parameter_flow_pass.record(
                encoder,
                &loc.parameter_flow_bgs[ping],
                h,
                w,
            );
        }
    }

    /// Mode accessor.
    #[must_use]
    pub fn convolve_mode(&self) -> ConvolveMode {
        self.convolve_mode
    }

    /// Affinity-growth mode accessor (M6.C-2-4-d).
    #[must_use]
    pub fn affinity_mode(&self) -> AffinityMode {
        self.affinity_mode
    }

    /// Current ping's parameter-map buffer (M6.C-2-4-d). Returns
    /// `None` outside Localized mode. Mirrors
    /// [`current_activation_buffer`](Self::current_activation_buffer)
    /// for the parameter-map ping-pong cadence.
    #[must_use]
    pub fn current_parameter_map_buffer(&self) -> Option<&wgpu::Buffer> {
        self.localized
            .as_ref()
            .map(|loc| &loc.p_buffers[self.ping])
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
        match self.convolve_mode {
            ConvolveMode::Direct => self.record_step(&mut enc),
            ConvolveMode::Fft => self.record_step_fft(ctx, &mut enc),
            // Auto is resolved at new_with_mode and can't appear here.
            ConvolveMode::Auto => unreachable!("convolve_mode resolved at construction"),
        }
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

    /// M6.C-3-3 profiling: per-pass GPU timing breakdown for the
    /// FFT-mode step. Returns `(label, mean_ns)` averaged over `iters`
    /// timed steps (after 5 warmup steps).
    ///
    /// **CPU-clock based**, not GPU `TIMESTAMP_QUERY`. The timestamp
    /// path (single + multi resolve variants both tried) hung on
    /// wgpu 29 + Metal even at small `iters` — `device.poll(Wait)`
    /// never returned after `write_timestamp` + multiple submits, with
    /// the process sleeping at <1% CPU for minutes. **Root cause
    /// remains unknown** (the suspicion is Metal counter sampling
    /// buffer drain interacting with `TIMESTAMP_QUERY_INSIDE_ENCODERS`
    /// + multiple un-polled submits, but this is unverified). The
    /// CPU variant sidesteps the hang but does NOT guarantee that the
    /// per-pass relative breakdown matches what `TIMESTAMP_QUERY`
    /// would have reported once it works. Tracking that hang's root
    /// cause is a deferred M6.C-3-7 retro item. Per-pass:
    ///
    ///   1. start = Instant::now()
    ///   2. record only that pass into a fresh encoder
    ///   3. submit
    ///   4. device.poll(Wait) — synchronous drain
    ///   5. elapsed = Instant::now() - start
    ///
    /// The submit + drain overhead is added to every pass uniformly
    /// **under the assumption that the per-call `submit + poll(Wait)`
    /// floor is independent of the encoded GPU work size**. That
    /// assumption is unproven; the observed ~1.55 ms minimum across
    /// `affinity / flow / gradient_a_sum` (M6.C-3-3 bench_512_breakdown
    /// output) is consistent with it but is a 3-sample empirical
    /// coincidence, not a measurement of the floor in isolation.
    /// Treat absolute per-pass µs as `(real_gpu_us + submit_floor_us)`
    /// where `submit_floor_us` is bounded above by ~1.55 ms but not
    /// known below. **Relative ordering** (which pass dominates) is
    /// usable for judgement A; **absolute values** are an upper bound
    /// only and must NOT be compared with `bench_c2_configs` ms/step.
    ///
    /// Requires `convolve_mode == Fft`. The convolve entry is the
    /// whole `ConvolveFftPass::record` (forward×C + spectral multiply
    /// + inverse×K); drilling into those sub-dispatches is a separate
    /// follow-up if convolve dominates.
    ///
    /// # Panics
    /// Panics if `convolve_mode != Fft`.
    pub fn profile_passes_fft(
        &mut self,
        ctx: &GpuContext,
        iters: u32,
    ) -> Vec<(&'static str, f64)> {
        use std::time::Instant;
        assert_eq!(
            self.convolve_mode,
            ConvolveMode::Fft,
            "profile_passes_fft requires Fft mode"
        );
        let h = self.height;
        let w = self.width;

        let mut labels: Vec<&'static str> = vec![
            "convolve",
            "affinity",
            "gradient_u",
            "gradient_a_sum",
            "flow",
            "reintegrate",
        ];
        if self.localized.is_some() {
            labels.push("parameter_flow");
        }
        let n_passes = labels.len();

        // Warmup (shader cache + thermal ramp). Drain at the end so the
        // first timed iter starts from a quiet queue.
        for _ in 0..5 {
            self.step(ctx);
        }
        ctx.device
            .poll(wgpu::PollType::Wait { submission_index: None, timeout: None })
            .expect("device.poll(Wait) after warmup failed");

        let mut acc = vec![0.0_f64; n_passes];
        for _ in 0..iters {
            let p = self.ping;
            // Helper: record + submit + drain, returns elapsed ns.
            let measure = |label_idx: usize,
                           record_fn: &mut dyn FnMut(&mut wgpu::CommandEncoder)|
             -> f64 {
                let mut enc = ctx
                    .device
                    .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                        label: Some("profile_passes_fft per-pass enc"),
                    });
                record_fn(&mut enc);
                let start = Instant::now();
                ctx.queue.submit([enc.finish()]);
                ctx.device
                    .poll(wgpu::PollType::Wait {
                        submission_index: None,
                        timeout: None,
                    })
                    .expect("device.poll(Wait) per-pass failed");
                let _ = label_idx; // intentionally unused — caller indexes acc
                start.elapsed().as_nanos() as f64
            };

            let fft = self.convolve_fft_pass.as_ref().expect("Fft mode");
            let kfft = self.kernel_fft.as_ref().expect("Fft mode");
            // Borrow self.convolve_fft_pass through a raw clone of the
            // resource references — Rust's borrow checker would otherwise
            // complain about &mut self in `self.record_affinity` while
            // also holding refs into self. We work around by recording
            // each pass into a closure that captures only the necessary
            // shared resources, not &mut self for those that need it.

            // Pass 0: convolve
            acc[0] += measure(0, &mut |enc| {
                fft.record(ctx, enc, &self.a_buffers[p], &kfft.buffer, &self.pre_g_buf);
            });
            // Pass 1: affinity (Constant only here — Localized adds the
            // sibling parameter_flow pass at the end).
            acc[1] += measure(1, &mut |enc| {
                self.record_affinity(enc, p, h, w);
            });
            // Pass 2: gradient_u
            acc[2] += measure(2, &mut |enc| {
                self.gradient_pass.record_u(enc, &self.gradient_u_bg, h, w);
            });
            // Pass 3: gradient_a_sum
            acc[3] += measure(3, &mut |enc| {
                self.gradient_pass
                    .record_a_sum(enc, &self.gradient_a_sum_bgs[p], h, w);
            });
            // Pass 4: flow
            acc[4] += measure(4, &mut |enc| {
                self.flow_pass.record(enc, &self.flow_bgs[p], h, w);
            });
            // Pass 5: reintegrate
            acc[5] += measure(5, &mut |enc| {
                self.reintegrate_pass
                    .record(enc, &self.reintegrate_bgs[p], h, w);
            });
            // Pass 6 (Localized only): parameter_flow
            if self.localized.is_some() {
                acc[6] += measure(6, &mut |enc| {
                    self.record_parameter_flow_if_localized(enc, p, h, w);
                });
            }
            self.swap_buffers();
        }

        labels
            .into_iter()
            .zip(acc.into_iter().map(|a| a / f64::from(iters)))
            .collect()
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
    ///
    /// **TODO (M6.C-1-5-b M1, M6.C-4 / M5)**: `kernel_routing_buf`
    /// in `ConvolveFftPass` is built once from `kernel_params[k].c0`
    /// in `new_with_mode` and is NOT refreshed by `update_globals`.
    /// If a future UI exposes per-kernel `c0` painting (M5
    /// parameter-painting candidate) the FFT path will silently use
    /// stale routing. Reachable today only via a full pipeline
    /// rebuild; flagged here so the next UI-side change includes a
    /// rebuild trigger.
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

    /// M6.C-1-4-b: Direct mode vs FFT mode in the full pipeline at
    /// N=64 C=1 K=10 Torus. Both modes run from the same initial state
    /// for a small number of steps; the per-cell activation difference
    /// must stay within A.4.5-tiered g64 tolerance (rel < 5e-4),
    /// matching the C-1-3 / C-1-4-a end-to-end FFT-vs-direct
    /// convolution headroom propagated through the affinity / flow /
    /// reintegrate stack.
    ///
    /// **C=1 + grid=64 only** per [`ConvolveMode::Fft`] limitation
    /// (M6.C-1-4-b scope). M6.A `m1_regression_gpu` runs at C=3 and
    /// is therefore NOT a host for this comparison; the FFT-mode
    /// regression target is `tests/diagnose_divergence.rs` (the
    /// existing C=1 testbed) plus this in-place sanity.
    #[test]
    fn gpu_pipeline_fft_mode_matches_direct_n64_c1_short() {
        let (ctx, guard) = headless_ctx();
        // Use the same kernel-radius / sigma defaults that drive
        // diagnose_divergence so the FFT path stays in sync with
        // the M6.A.4.5 C=1 measurement campaign's parameter space.
        let cfg = FlowLeniaConfig {
            grid_width: 64,
            grid_height: 64,
            channels: 1,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 10,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        };
        let seed = 0x4F_FE_64_C1_u64;
        let cpu_init = FlowLeniaSimulator::new(cfg, seed);
        let initial_a = cpu_init.activation().clone();
        let kernel_params = cpu_init.kernel_params().clone();

        let mut direct =
            GpuStepPipeline::new_with_mode(&ctx, &cfg, &kernel_params, &initial_a, ConvolveMode::Direct);
        let mut fft =
            GpuStepPipeline::new_with_mode(&ctx, &cfg, &kernel_params, &initial_a, ConvolveMode::Fft);

        // Short horizon: the chaotic-amplification finding (M6.A.4.5)
        // makes longer C=1 horizons noisy here too, so we keep the
        // comparison at 5 steps (matches gpu_pipeline_wall_border_matches_cpu).
        let n_steps: u32 = 5;
        direct.run_steps(&ctx, n_steps);
        fft.run_steps(&ctx, n_steps);

        let direct_a = direct.readback_activation(&ctx);
        let fft_a = fft.readback_activation(&ctx);

        let (h, w, c) = direct_a.dim();
        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for ((y, x, ci), &d) in direct_a.indexed_iter() {
            let f = fft_a[[y, x, ci]];
            let abs_err = (d - f).abs();
            let rel_err = abs_err / d.abs().max(1e-6);
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            assert!(
                rel_err < 5e-4 || abs_err < 1e-5,
                "({y}, {x}, c={ci}) after {n_steps} steps: direct={d} fft={f} \
                 abs={abs_err:.3e} rel={rel_err:.3e}"
            );
        }
        let _ = (h, w, c);
        eprintln!(
            "[M6.C-1-4-b] pipeline direct vs fft N=64 C=1 K=10 Torus {n_steps}-step : \
             max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// M6.C-1-5-a: Direct mode vs FFT mode at **C=3** N=64 K=10
    /// Torus, 5-step short horizon. multi-channel + per-kernel
    /// source_channel routing が direct と一致を確認。
    /// **chaos amplification** は C=3 で C=1 よりさらに大きい (M2.8
    /// finding)、Layer 3 A.4.5 tiered tolerance g64 = 5e-4 を採用、
    /// C-1-6 long-horizon measurement で sustainability 確認。
    #[test]
    fn gpu_pipeline_fft_mode_matches_direct_n64_c3_short() {
        let (ctx, guard) = headless_ctx();
        let cfg = FlowLeniaConfig {
            grid_width: 64,
            grid_height: 64,
            channels: 3,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 10,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        };
        let seed = 0x4F_FE_64_C3_u64;
        let cpu_init = FlowLeniaSimulator::new(cfg, seed);
        let initial_a = cpu_init.activation().clone();
        let kernel_params = cpu_init.kernel_params().clone();

        let mut direct = GpuStepPipeline::new_with_mode(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Direct,
        );
        let mut fft = GpuStepPipeline::new_with_mode(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Fft,
        );

        let n_steps: u32 = 5;
        direct.run_steps(&ctx, n_steps);
        fft.run_steps(&ctx, n_steps);

        let direct_a = direct.readback_activation(&ctx);
        let fft_a = fft.readback_activation(&ctx);

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for ((y, x, ci), &d) in direct_a.indexed_iter() {
            let f = fft_a[[y, x, ci]];
            let abs_err = (d - f).abs();
            let rel_err = abs_err / d.abs().max(1e-6);
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            assert!(
                rel_err < 5e-4 || abs_err < 1e-5,
                "({y}, {x}, c={ci}) after {n_steps} steps C=3: direct={d} fft={f} \
                 abs={abs_err:.3e} rel={rel_err:.3e}"
            );
        }
        eprintln!(
            "[M6.C-1-5-a] pipeline direct vs fft N=64 C=3 K=10 Torus {n_steps}-step : \
             max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    // M6.C-1-4-b の gpu_pipeline_fft_mode_rejects_multi_channel は
    // M6.C-1-5-a で multi-channel 対応により obsolete、削除済。
    // C=3 + grid=64 で FFT mode は正常に構築・動作するようになった
    // (gpu_pipeline_fft_mode_matches_direct_n64_c3_short が新規 coverage)。

    /// M6.C-3-2: end-to-end FFT-mode equivalence at **N=512** (mixed-
    /// radix). Direct path is the trusted reference (validated vs CPU
    /// at M2.8); this confirms the mixed-radix FFT pipeline
    /// (ConvolveFftPass + Fft2dPass mixed H/V + FftInvToPreGPass mixed
    /// + spectral multiply) produces the same field as Direct at the
    /// 512 hi-end grid. **C=1, 2-step short horizon** — Direct at 512
    /// is ~190 ms/step (BENCH §15) so we keep the step count minimal;
    /// short horizon also keeps FFT-vs-Direct chaotic divergence below
    /// the tolerance (A.4.5: chaos amplification grows with grid AND
    /// horizon, so 2 steps at 512 is the analog of 5 steps at 64).
    #[test]
    fn gpu_pipeline_fft_mode_matches_direct_n512_c1_short() {
        let (ctx, guard) = headless_ctx();
        let cfg = FlowLeniaConfig {
            grid_width: 512,
            grid_height: 512,
            channels: 1,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: 10,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        };
        let seed = 0x512_FE_C1_u64;
        let cpu_init = FlowLeniaSimulator::new(cfg, seed);
        let initial_a = cpu_init.activation().clone();
        let kernel_params = cpu_init.kernel_params().clone();

        let mut direct = GpuStepPipeline::new_with_mode(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Direct,
        );
        let mut fft = GpuStepPipeline::new_with_mode(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Fft,
        );
        assert_eq!(fft.convolve_mode(), ConvolveMode::Fft, "512 must route to Fft");

        let n_steps: u32 = 2;
        direct.run_steps(&ctx, n_steps);
        fft.run_steps(&ctx, n_steps);

        let direct_a = direct.readback_activation(&ctx);
        let fft_a = fft.readback_activation(&ctx);

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for ((y, x, ci), &d) in direct_a.indexed_iter() {
            let f = fft_a[[y, x, ci]];
            let abs_err = (d - f).abs();
            let rel_err = abs_err / d.abs().max(1e-6);
            max_abs = max_abs.max(abs_err);
            max_rel = max_rel.max(rel_err);
            // 512 short-horizon tolerance: A.4.5 tiered bound scaled up
            // from g256 (2.5e-3). 2-step keeps actual divergence far
            // below this; the bound is the regression ceiling, not the
            // observed value (printed below).
            assert!(
                rel_err < 5e-3 || abs_err < 1e-5,
                "({y}, {x}, c={ci}) after {n_steps} steps N=512: direct={d} fft={f} \
                 abs={abs_err:.3e} rel={rel_err:.3e}"
            );
        }
        eprintln!(
            "[M6.C-3-2] pipeline direct vs fft N=512 C=1 K=10 Torus {n_steps}-step : \
             max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// M6.C-1-4-b: ConvolveMode::Fft must reject unsupported grid
    /// sizes (only {64, 256} pass; 32 / 128 / 512 are mixed-radix,
    /// deferred to C-1-5+).
    /// **C=1 base cfg** to isolate the grid assert.
    #[test]
    #[should_panic(expected = "ConvolveMode::Fft requires cfg.grid_width")]
    fn gpu_pipeline_fft_mode_rejects_unsupported_grid() {
        let (ctx, _guard) = headless_ctx();
        // Build a C=1 cfg with grid=32 (rejected by SUPPORTED_N=[64, 256]).
        let mut cfg = small_cfg(1, false, BorderMode::Torus);
        cfg.grid_width = 32;
        cfg.grid_height = 32;
        let kernel_params = FlowLeniaSimulator::new(cfg, 0).kernel_params().clone();
        let initial_a = FlowLeniaSimulator::new(cfg, 0).activation().clone();
        let _ = GpuStepPipeline::new_with_mode(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Fft,
        );
    }

    /// M6.C-2-4-d: production-pipeline 4-creature smoke test.
    ///
    /// Wires the C-2-4-a parameter map + C-2-4-c ParameterFlowPass
    /// into `GpuStepPipeline.new_with_modes(_, _, _, _, Auto,
    /// Localized, Some(p_initial))` at N=64 C=3 K=8, runs 10 steps,
    /// and verifies:
    ///
    /// 1. Construction succeeds (Auto-resolves to Fft for N=64,
    ///    Localized allocates the LocalizedState bundle)
    /// 2. WebGPU validation is clean over the full 10-step run
    /// 3. Mass conservation: total `A` drift < 10% (ReintegratePass
    ///    anchors mass; the relaxed bound absorbs Localized-mode
    ///    flux into the background P region)
    /// 4. **P identity invariant**: bit-equal between the initial
    ///    `p_initial` and the current ping's P buffer after 10
    ///    steps — this is the case-(a) C-2-4-c contract carried
    ///    through the full per-step pipeline, validating the
    ///    parameter_flow_bgs ping-pong wiring
    /// 5. **4 creature alive**: per-creature mass within a 24×24
    ///    neighborhood remains > 20% of the initial blob mass after
    ///    10 steps (Plantec §4.3.2 layout: corners disjoint at
    ///    N=64; the threshold is the headless equivalent of the
    ///    "screenshot shows 4 distinct creatures" visual smoke test
    ///    in M6.C-2-4 plan)
    #[test]
    fn gpu_pipeline_localized_four_creatures_alive_after_10_steps() {
        use crate::{
            passes::parameter_map::{build_for_patches, CreaturePatch},
            readback::readback_buffer,
        };
        use ndarray::Array3;
        use rand::{Rng, SeedableRng};
        use rand_chacha::ChaCha8Rng;

        let (ctx, guard) = headless_ctx();

        let n: u32 = 64;
        let c: u32 = 3;
        let k: u32 = 8;
        let cfg = FlowLeniaConfig {
            grid_width: n,
            grid_height: n,
            channels: c,
            dt: 0.2,
            sigma: 0.65,
            n: 2.0,
            beta_a: 2.0,
            dd: 5,
            num_kernels: k,
            paper_strict: false,
            border: BorderMode::Torus,
            mix_rule: MixRule::Stochastic,
        };
        // Reuse the seed that the existing FFT-N=64-C=3 K=10 test
        // (`gpu_pipeline_fft_mode_matches_direct_n64_c3_short`) uses;
        // it's known to sample KernelEntries with radii that fit
        // inside an N=64 grid. Some seeds pick r_global large enough
        // to overflow the grid (kernel_fft.rs:117 padded-kernel
        // assertion).
        let seed: u64 = 0x4F_FE_64_C3_u64;

        // Borrow KernelParams from a CPU-side sim build so we get the
        // same KernelEntry sampling distribution that the rest of the
        // pipeline tests use.
        let cpu_init = FlowLeniaSimulator::new(cfg, seed);
        let kernel_params = cpu_init.kernel_params().clone();
        let h_base: Vec<f32> = kernel_params.kernels.iter().map(|e| e.h).collect();
        assert_eq!(h_base.len(), k as usize);

        // 4-creature initial A: corner-placed 16×16 blobs with random
        // [0.3, 0.8) mass per cell per channel. Centers at (12, 12),
        // (12, 52), (52, 12), (52, 52) keep the four neighborhoods
        // disjoint on torus N=64 (40-cell spacing > 16-cell blob).
        let blob_size: i32 = 16;
        let half = blob_size / 2;
        let centers: [(i32, i32); 4] = [(12, 12), (12, 52), (52, 12), (52, 52)];
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let mut initial_a = Array3::<f32>::zeros((n as usize, n as usize, c as usize));
        let mut initial_creature_mass = [0.0_f32; 4];
        for (idx, &(cy, cx)) in centers.iter().enumerate() {
            for dy in -half..half {
                for dx in -half..half {
                    let y = ((cy + dy).rem_euclid(n as i32)) as usize;
                    let x = ((cx + dx).rem_euclid(n as i32)) as usize;
                    for ci in 0..c as usize {
                        let v = rng.gen_range(0.3_f32..0.8);
                        initial_a[[y, x, ci]] = v;
                        initial_creature_mass[idx] += v;
                    }
                }
            }
        }
        let initial_total: f32 = initial_a.iter().sum();
        assert!(initial_total > 0.0);

        // 4-creature P map: each creature scales h_base by a per-
        // creature factor so the P vectors are visibly distinct.
        // Background = h_base so outside-patch cells behave like
        // Constant mode (Eq. 3 fallback) for the "wider neighborhood
        // alive" check — without this fallback, background G(0) ≈ -1
        // would kill all background mass and the per-creature
        // window check would become trivial.
        let patches: Vec<CreaturePatch> = centers
            .iter()
            .enumerate()
            .map(|(idx, &(cy, cx))| {
                let y0 = ((cy - half).rem_euclid(n as i32)) as u32;
                let x0 = ((cx - half).rem_euclid(n as i32)) as u32;
                let p_vector: Vec<f32> = h_base
                    .iter()
                    .enumerate()
                    .map(|(ki, &h)| h * (1.0 + 0.05 * idx as f32 + 0.01 * ki as f32))
                    .collect();
                CreaturePatch {
                    bbox: (y0, x0, y0 + blob_size as u32, x0 + blob_size as u32),
                    p_vector,
                }
            })
            .collect();
        let p_initial = build_for_patches(n, k, &h_base, &patches);
        assert_eq!(p_initial.len(), (n * n * k) as usize);

        // Pipeline construction.
        let mut pipeline = GpuStepPipeline::new_with_modes(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Auto,
            AffinityMode::Localized,
            Some(&p_initial),
        );
        assert_eq!(
            pipeline.convolve_mode(),
            ConvolveMode::Fft,
            "Auto must resolve to Fft for N=64"
        );
        assert_eq!(pipeline.affinity_mode(), AffinityMode::Localized);

        // Run 10 steps.
        let n_steps: u32 = 10;
        pipeline.run_steps(&ctx, n_steps);

        // (1) Mass conservation.
        let final_a = pipeline.readback_activation(&ctx);
        let final_total: f32 = final_a.iter().sum();
        let mass_drift = (final_total - initial_total).abs() / initial_total.max(1e-3);
        assert!(
            mass_drift < 0.1,
            "mass drift {:.3e} > 10% after {n_steps} steps \
             (initial={initial_total:.3e} final={final_total:.3e})",
            mass_drift
        );

        // (2) P identity bit-equal after 10 ping-pong steps. Reads
        // the buffer the **next** step would consume (current ping);
        // for the case-(a) identity-copy contract that buffer must
        // hold the initial map verbatim.
        let p_buf = pipeline
            .current_parameter_map_buffer()
            .expect("Localized mode pipeline must expose P buffer");
        let p_final = readback_buffer::<f32>(&ctx, p_buf, p_initial.len());
        for (i, (a, b)) in p_final.iter().zip(p_initial.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "P identity violated at index {i}: final={a} initial={b}"
            );
        }

        // (3) 4 creature alive: each 24×24 neighborhood retains > 20%
        // of its initial blob mass. Neighborhoods are disjoint at
        // N=64 (centers 12 / 52, half-width 12 → ranges 0..24 and
        // 40..64, no overlap).
        let neighborhood_half: i32 = 12;
        for (idx, &(cy, cx)) in centers.iter().enumerate() {
            let mut final_mass = 0.0_f32;
            for dy in -neighborhood_half..=neighborhood_half {
                for dx in -neighborhood_half..=neighborhood_half {
                    let y = ((cy + dy).rem_euclid(n as i32)) as usize;
                    let x = ((cx + dx).rem_euclid(n as i32)) as usize;
                    for ci in 0..c as usize {
                        final_mass += final_a[[y, x, ci]];
                    }
                }
            }
            assert!(
                final_mass > initial_creature_mass[idx] * 0.2,
                "creature {idx} at ({cy}, {cx}) died: final neighborhood \
                 mass {final_mass:.3e} < 20% of initial {:.3e}",
                initial_creature_mass[idx]
            );
            eprintln!(
                "[C-2-4-d] creature {idx} at ({cy}, {cx}): initial \
                 mass {:.3e}, final {final_mass:.3e} (retained {:.1})",
                initial_creature_mass[idx],
                100.0 * final_mass / initial_creature_mass[idx].max(1e-6)
            );
        }

        eprintln!(
            "[C-2-4-d] Localized N=64 C=3 K=8 4-creature {n_steps}-step : \
             total mass drift {:.3e}, P identity preserved",
            mass_drift
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// M6.C-2-4-d: assertion that AffinityMode::Constant rejects an
    /// `initial_p_map` argument (defensive: the type's docs say
    /// Constant ignores it, but we asserted it must be None to catch
    /// caller bugs where a Localized-intended call accidentally
    /// picked Constant).
    #[test]
    #[should_panic(expected = "AffinityMode::Constant must not be paired with initial_p_map")]
    fn gpu_pipeline_constant_mode_rejects_initial_p_map() {
        let (ctx, _guard) = headless_ctx();
        let cfg = small_cfg(3, false, BorderMode::Torus);
        let cpu_init = FlowLeniaSimulator::new(cfg, 0);
        let initial_a = cpu_init.activation().clone();
        let kernel_params = cpu_init.kernel_params().clone();
        let dummy_p_map = vec![0.0_f32; 8]; // wrong shape on purpose, must panic before length check
        let _ = GpuStepPipeline::new_with_modes(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Direct,
            AffinityMode::Constant,
            Some(&dummy_p_map),
        );
    }

    /// M6.C-2-4-d: assertion that AffinityMode::Localized rejects a
    /// missing `initial_p_map`.
    #[test]
    #[should_panic(expected = "AffinityMode::Localized requires initial_p_map")]
    fn gpu_pipeline_localized_mode_requires_initial_p_map() {
        let (ctx, _guard) = headless_ctx();
        let cfg = small_cfg(3, false, BorderMode::Torus);
        let cpu_init = FlowLeniaSimulator::new(cfg, 0);
        let initial_a = cpu_init.activation().clone();
        let kernel_params = cpu_init.kernel_params().clone();
        let _ = GpuStepPipeline::new_with_modes(
            &ctx,
            &cfg,
            &kernel_params,
            &initial_a,
            ConvolveMode::Direct,
            AffinityMode::Localized,
            None,
        );
    }
}
