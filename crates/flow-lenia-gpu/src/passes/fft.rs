//! M6.C-1-1 / M6.C-1-2 — 1D + 2D radix-4 Cooley-Tukey FFT.
//!
//! C-1-1 introduced [`FftPass`] (1D forward, N=256 fixed). C-1-2
//! extends this to:
//!
//! - **Dynamic N** ∈ {64, 256} via WGSL pipeline-override constants
//!   on `WORKGROUP_X`. Mixed-radix sizes (32, 128, 512) are out of
//!   scope for M6.C-1 — those need a radix-2 fall-out stage and are
//!   deferred to a later sub-step (per M6.B literature survey §2.5
//!   + scope-guardian approval for C-1-2: pure radix-4 first, 512
//!   add-on if the main goal demands it).
//! - **Forward / inverse direction** via a runtime `direction` flag
//!   in `FftParams`. The same WGSL pipeline serves both directions;
//!   inverse re-uses the forward twiddle table by conjugating
//!   in-shader and applies the 1/N normalisation at store time
//!   (rustfft / NumPy convention: forward unnormalised, inverse
//!   divides by N).
//! - **2D separable transform** via [`Fft2dPass`], which composes a
//!   per-row pass (`fft_1d_radix4.wgsl`) with a per-column pass
//!   (`fft_1d_radix4_v.wgsl`). 2 dispatches per direction, no
//!   transpose pass. See `fft_1d_radix4_v.wgsl` header for the
//!   column-stride trade-off (memory bandwidth vs avoiding a
//!   transpose pass with its own dispatch overhead).
//!
//! Hot-path usage (C-1-4 will wire this into `ConvolvePass`):
//!
//! ```text
//! // Startup
//! let pass2d   = Fft2dPass::new(&ctx, 256);
//! let twiddles = precompute_twiddles_1d(&ctx, 256);
//!
//! // Per step:
//! //  - upload params with direction=Forward
//! //  - bind_group: input=real f32 H×W, twiddles, intermediate vec2 H×W, params
//! //  - record_axis(H, num_rows=H)  → intermediate has H-axis spectrum
//! //  - re-bind: input=intermediate (as flat f32), output=spectrum
//! //  - record_axis(V, num_rows=W)  → spectrum is the 2D FFT
//! //  - (spectral multiply by per-kernel pre-FFT  — C-1-3 scope)
//! //  - mirror inverse path with direction=Inverse to recover the field
//! ```
//!
//! `record_axis` only appends the dispatch; submission and
//! synchronisation are the caller's responsibility, matching the
//! M2.3+ pattern used by every other pass in this crate.

use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Pure-radix-4 sizes supported by the C-1-2 dispatch shape. Adding
/// 32 / 128 / 512 requires a radix-2 fall-out stage in the WGSL —
/// deferred per scope-guardian approval, will be revisited when the
/// Flow-Lenia grid sweep needs them (currently the main goal is
/// 256×256×4creature + 512×512 hi-end mode).
pub const SUPPORTED_N: &[u32] = &[64, 256];

/// Direction of a 1D pass. Same WGSL pipeline runs both; the runtime
/// flag flips twiddle conjugation and per-cell 1/N normalisation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftDirection {
    Forward = 0,
    Inverse = 1,
}

/// Axis selector for 2D dispatches. The two axes use different WGSL
/// pipelines (row-stride vs column-stride load/store), assembled into
/// a single [`Fft2dPass`] for the caller's convenience.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftAxis {
    H,
    V,
}

/// Mirror of the WGSL `struct FftParams`. The `_pad` field exists
/// to round to 16-byte uniform alignment per the WGSL spec.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FftParams {
    pub n: u32,
    pub num_rows: u32, // V-axis pass: column count
    pub direction: u32,
    pub _pad: u32,
}

impl FftParams {
    /// Build a `FftParams` with `direction` typed and `_pad` zeroed.
    /// `num_rows` for H is row count, for V is column count.
    #[must_use]
    pub fn new(n: u32, num_rows: u32, direction: FftDirection) -> Self {
        Self {
            n,
            num_rows,
            direction: direction as u32,
            _pad: 0,
        }
    }
}

/// Check whether `n` is a supported pure-radix-4 transform length.
#[must_use]
pub fn is_supported_n(n: u32) -> bool {
    SUPPORTED_N.contains(&n)
}

/// M6.C-3-1 mixed-radix sizes: `N = 2 × 4^k` (= {8, 32, 128, 512}).
/// These need exactly one radix-2 stage on top of the radix-4
/// machinery (`fft_1d_radix2x4.wgsl`). Distinct from the pure-radix-4
/// [`SUPPORTED_N`]; the FFT-mode pipeline does **not** route these
/// through `ConvolveFftPass` yet (that wiring is M6.C-3-2).
pub const SUPPORTED_MIXED_N: &[u32] = &[8, 32, 128, 512];

/// Check whether `n = 2 × 4^k` (i.e. `n/2` is a power of 4 and `n`
/// is even). The mixed-radix primitive [`FftPass::new_h_mixed`]
/// handles these.
#[must_use]
pub fn is_supported_mixed_n(n: u32) -> bool {
    SUPPORTED_MIXED_N.contains(&n)
}

/// Grids the FFT-mode **pipeline** (`ConvolveFftPass` + `GpuStepPipeline`
/// `Auto`) routes through end-to-end: pure radix-4 {64, 256} plus
/// **512** (mixed-radix, wired in M6.C-3-2 for the 512×512 hi-end
/// goal). The mixed primitive *also* compiles for 8/32/128, but the
/// pipeline intentionally keeps those on the Direct fallback — routing
/// them through FFT would change the existing M6.A snapshot /
/// regression baselines generated with Direct, and they are not a
/// performance target. Extend this set only with a deliberate
/// baseline-regeneration plan.
#[must_use]
pub fn is_fft_pipeline_grid(n: u32) -> bool {
    is_supported_n(n) || n == 512
}

/// Compiled 1D radix-4 FFT pass for a single axis. C-1-1 callers
/// instantiate this directly via [`FftPass::new_h`]; C-1-2 callers
/// should prefer [`Fft2dPass`] which owns both H and V passes.
pub struct FftPass {
    pub n: u32,
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl FftPass {
    /// Compile the H-axis (row-stride) variant. Convenience for the
    /// C-1-1 1D-only tests; new C-1-2 callers should use
    /// [`Fft2dPass::new`] instead.
    #[must_use]
    pub fn new_h(ctx: &GpuContext, n: u32) -> Self {
        assert!(
            is_supported_n(n),
            "C-1-2 FftPass supports N ∈ {SUPPORTED_N:?} (got {n}); \
             mixed-radix sizes (32/128/512) require a radix-2 fall-out \
             stage — use FftPass::new_h_mixed for N = 2 × 4^k."
        );
        Self::compile(
            ctx,
            n,
            include_str!("../shaders/fft_1d_radix4.wgsl"),
            "fft_1d_radix4",
            "fft_1d_radix4.wgsl",
        )
    }

    /// Compile the V-axis (column-stride) variant.
    #[must_use]
    pub fn new_v(ctx: &GpuContext, n: u32) -> Self {
        assert!(
            is_supported_n(n),
            "C-1-2 FftPass supports N ∈ {SUPPORTED_N:?} (got {n}); \
             mixed-radix sizes (32/128/512) require a radix-2 fall-out \
             stage — use FftPass::new_h_mixed for N = 2 × 4^k."
        );
        Self::compile(
            ctx,
            n,
            include_str!("../shaders/fft_1d_radix4_v.wgsl"),
            "fft_1d_radix4_v",
            "fft_1d_radix4_v.wgsl",
        )
    }

    /// M6.C-3-1 mixed-radix H-axis variant for `N = 2 × 4^k`
    /// ({8, 32, 128, 512}). Uses `fft_1d_radix2x4.wgsl` (4^k radix-4
    /// stages + 1 radix-2 combine). Binding contract is identical to
    /// the pure-radix-4 pass, so [`make_bind_group`](Self::make_bind_group)
    /// / [`record`](Self::record) / [`upload_params`](Self::upload_params)
    /// all apply unchanged. The V-axis mixed variant + 2D + ConvolveFftPass
    /// wiring land in M6.C-3-2.
    #[must_use]
    pub fn new_h_mixed(ctx: &GpuContext, n: u32) -> Self {
        assert!(
            is_supported_mixed_n(n),
            "FftPass::new_h_mixed supports N ∈ {SUPPORTED_MIXED_N:?} \
             (= 2 × 4^k) (got {n}); pure powers of 4 use FftPass::new_h."
        );
        Self::compile(
            ctx,
            n,
            include_str!("../shaders/fft_1d_radix2x4.wgsl"),
            "fft_1d_radix2x4",
            "fft_1d_radix2x4.wgsl",
        )
    }

    /// M6.C-3-2 mixed-radix V-axis (column-stride) variant for
    /// `N = 2 × 4^k`. Pairs with [`new_h_mixed`](Self::new_h_mixed)
    /// inside [`Fft2dPass::new`] for 512×512 2D transforms.
    #[must_use]
    pub fn new_v_mixed(ctx: &GpuContext, n: u32) -> Self {
        assert!(
            is_supported_mixed_n(n),
            "FftPass::new_v_mixed supports N ∈ {SUPPORTED_MIXED_N:?} \
             (= 2 × 4^k) (got {n}); pure powers of 4 use FftPass::new_v."
        );
        Self::compile(
            ctx,
            n,
            include_str!("../shaders/fft_1d_radix2x4_v.wgsl"),
            "fft_1d_radix2x4_v",
            "fft_1d_radix2x4_v.wgsl",
        )
    }

    fn compile(
        ctx: &GpuContext,
        n: u32,
        source: &str,
        entry: &str,
        label: &str,
    ) -> Self {
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some(&format!("{entry} bind group layout")),
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
                label: Some(&format!("{entry} pipeline layout")),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        // Pipeline-override constant: pin WORKGROUP_X to N. This lets
        // one WGSL source produce a different `@workgroup_size(N,1,1)`
        // per pipeline without text-template hacks. See WGSL spec
        // "pipeline-overridable constants". wgpu 29 takes the
        // overrides as `&[(&str, f64)]`.
        let constants: [(&str, f64); 1] = [("WORKGROUP_X", f64::from(n))];

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&format!("{entry} pipeline (N={n})")),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });

        Self {
            n,
            pipeline,
            bind_group_layout,
        }
    }

    /// Assemble a bind group for one (input, twiddles, output, params)
    /// quadruple.
    #[must_use]
    pub fn make_bind_group(
        &self,
        ctx: &GpuContext,
        input: &wgpu::Buffer,
        twiddles: &wgpu::Buffer,
        output: &wgpu::Buffer,
        params: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("fft_1d_radix4 bind group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: input.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: twiddles.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: params.as_entire_binding(),
                },
            ],
        })
    }

    /// Append one dispatch processing `num_rows` rows (or `num_cols`
    /// columns for the V variant) of `n` complex samples each.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        num_rows_or_cols: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fft_1d_radix4 pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(num_rows_or_cols, 1, 1);
    }

    /// Convenience: upload an `FftParams` value as a uniform buffer.
    ///
    /// Enforces the `params.n == self.n` invariant at the helper
    /// boundary so the WGSL `digit_reverse_4_dynamic` walk and the
    /// stage loop match the pipeline-baked `WORKGROUP_X`. Callers
    /// constructing their own uniform buffer bypass this gate (see
    /// crate rustdoc caveat).
    #[must_use]
    pub fn upload_params(&self, ctx: &GpuContext, params: FftParams) -> wgpu::Buffer {
        assert_eq!(
            params.n, self.n,
            "FftParams.n={got} must equal FftPass.n={expected} \
             (the WGSL workgroup_size and stage loop were baked at \
             pipeline construction).",
            got = params.n,
            expected = self.n,
        );
        assert!(
            params.num_rows >= 1,
            "FftParams.num_rows must be ≥ 1 (got {})",
            params.num_rows
        );
        ctx.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fft_1d_radix4 params uniform"),
                contents: bytemuck::bytes_of(&params),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            })
    }
}

/// Compiled 2D radix-4 FFT pass: owns both H and V passes for a
/// given `n`. Forward and inverse share pipelines via the runtime
/// `direction` flag in `FftParams`.
pub struct Fft2dPass {
    pub n: u32,
    pub h: FftPass,
    pub v: FftPass,
}

impl Fft2dPass {
    /// Build both axis passes for transform length `n`. `n` must be
    /// in [`SUPPORTED_N`] (pure radix-4: 64/256) **or**
    /// [`SUPPORTED_MIXED_N`] (mixed-radix 2×4^k: 8/32/128/512). The
    /// mixed sizes (M6.C-3-2) route to `fft_1d_radix2x4{,_v}.wgsl`;
    /// pure powers of 4 keep the original radix-4 shaders. All
    /// downstream `Fft2dPass` methods are shader-agnostic (they only
    /// call `self.h` / `self.v`).
    #[must_use]
    pub fn new(ctx: &GpuContext, n: u32) -> Self {
        let (h, v) = if is_supported_n(n) {
            (FftPass::new_h(ctx, n), FftPass::new_v(ctx, n))
        } else if is_supported_mixed_n(n) {
            (FftPass::new_h_mixed(ctx, n), FftPass::new_v_mixed(ctx, n))
        } else {
            panic!(
                "Fft2dPass::new: N={n} not in SUPPORTED_N {SUPPORTED_N:?} \
                 nor SUPPORTED_MIXED_N {SUPPORTED_MIXED_N:?}"
            );
        };
        Self { n, h, v }
    }

    /// Pick the per-axis sub-pass.
    #[must_use]
    pub fn axis(&self, axis: FftAxis) -> &FftPass {
        match axis {
            FftAxis::H => &self.h,
            FftAxis::V => &self.v,
        }
    }

    /// M6.C-1-4 caller-supplied-scratch hot-path 2D forward FFT.
    /// Same algorithm as [`forward_2d`] but the caller owns the input,
    /// output, and one complex-sized scratch buffer. The large
    /// `n²`-complex data buffers do not reallocate per call; appends
    /// 2 dispatches to the supplied encoder and does not submit or
    /// poll. Designed for `ConvolveFftPass` per-step orchestration
    /// where the same data scratch buffers persist across thousands
    /// of frames.
    ///
    /// **Honest framing** (Round 2 review NC1): per-call this helper
    /// still creates two `FftParams` uniform buffers (one each for
    /// the H and V axes) and two bind groups, because `upload_params`
    /// and `make_bind_group` are called inside. Hoisting these is
    /// straightforward (the buffer identities are caller-fixed) but
    /// is deferred to C-1-4-b / C-1-5 perf phase together with the
    /// matching hoist on the `ConvolveFftPass` side. Callers
    /// integrating this into a tight per-frame loop should expect
    /// ~4 small wgpu object creations per call until that lands.
    ///
    /// Buffer contract:
    /// - `input_complex`: bound as `array<f32>`. The H-axis shader
    ///   addresses only the first `n²` f32 slots
    ///   (`input[row * n + col]`) on the forward path, so a
    ///   real-valued input may be a tightly-packed `4n²`-byte buffer;
    ///   a complex-sized `8n²`-byte buffer also works (the second
    ///   half is simply not read on the forward path). (Round 2
    ///   review NC2 rewrite — the previous "upper half" wording was
    ///   left over from an in-place-FFT draft.)
    /// - `scratch_complex`: `n²` complex (`8n²` bytes), scratch
    ///   between H and V.
    /// - `output_complex`: `n²` complex (`8n²` bytes), receives the
    ///   2D spectrum.
    pub fn forward_2d_with_scratch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        twiddles: &wgpu::Buffer,
        input_complex: &wgpu::Buffer,
        scratch_complex: &wgpu::Buffer,
        output_complex: &wgpu::Buffer,
    ) {
        let n = self.n;
        let params_h = self
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let params_v = self
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let bg_h = self.h.make_bind_group(
            ctx,
            input_complex,
            twiddles,
            scratch_complex,
            &params_h,
        );
        let bg_v = self.v.make_bind_group(
            ctx,
            scratch_complex,
            twiddles,
            output_complex,
            &params_v,
        );
        self.h.record(encoder, &bg_h, n);
        self.v.record(encoder, &bg_v, n);
    }

    /// M6.C-1-4 caller-supplied-scratch 2D inverse FFT. Mirror of
    /// [`forward_2d_with_scratch`] but with `direction = Inverse` and
    /// V-then-H ordering (matching the round-trip layout used by
    /// `ConvolveFftPass`).
    ///
    /// Same per-call allocation caveat as the forward helper (Round 2
    /// review NC1): two `FftParams` uniform buffers + two bind groups
    /// are created per call; hoist deferred to C-1-4-b / C-1-5.
    /// `input_complex` is always treated as complex `vec2<f32>` on the
    /// inverse path (no real-input branch).
    pub fn inverse_2d_with_scratch(
        &self,
        ctx: &GpuContext,
        encoder: &mut wgpu::CommandEncoder,
        twiddles: &wgpu::Buffer,
        input_complex: &wgpu::Buffer,
        scratch_complex: &wgpu::Buffer,
        output_complex: &wgpu::Buffer,
    ) {
        let n = self.n;
        let params_v = self
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Inverse));
        let params_h = self
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Inverse));
        let bg_v = self.v.make_bind_group(
            ctx,
            input_complex,
            twiddles,
            scratch_complex,
            &params_v,
        );
        let bg_h = self.h.make_bind_group(
            ctx,
            scratch_complex,
            twiddles,
            output_complex,
            &params_h,
        );
        self.v.record(encoder, &bg_v, n);
        self.h.record(encoder, &bg_h, n);
    }

    /// M6.C-1-3 standalone 2D forward FFT helper. Runs the two-axis
    /// dispatch on a real `n × n` input and returns the complex
    /// `n × n` spectrum as `Vec<[f32; 2]>` (.0 = real, .1 = imag).
    ///
    /// Allocates two scratch buffers per call — fine for the C-1-3
    /// startup kernel pre-FFT path (one call per kernel) and for
    /// integration tests. The per-step hot path that C-1-4 will wire
    /// up should manage its own buffers and call into the
    /// `axis(...) + record(...)` API directly to avoid the per-call
    /// allocation. `TODO(M6.C-1-4)`: replace the per-call buffers
    /// with caller-supplied scratch when integrating into the
    /// convolution step.
    ///
    /// Round 1 review #4: the resize-copy + H/V dispatch are folded
    /// into a single command encoder + a single submit (was two
    /// submits in the initial commit — this saves K queue submits at
    /// kernel-precompute startup).
    #[must_use]
    pub fn forward_2d(
        &self,
        ctx: &GpuContext,
        twiddles: &wgpu::Buffer,
        input_real: &[f32],
    ) -> Vec<[f32; 2]> {
        let n = self.n;
        let total_cells = (n * n) as usize;
        assert_eq!(
            input_real.len(),
            total_cells,
            "Fft2dPass::forward_2d: input must be N×N reals (got {} for N={})",
            input_real.len(),
            n
        );

        // Staging: real input (N×N reals = 4N² bytes), only used as
        // a COPY_SRC for the encoder-internal copy into buf_a below.
        let staging = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fft2d::forward_2d staging real"),
                contents: bytemuck::cast_slice(input_real),
                usage: wgpu::BufferUsages::COPY_SRC,
            });
        // buf_a: complex-sized (8N² bytes) — receives the H-axis
        // shader's real-input read (which addresses the first 4N²
        // bytes) and, after the V-axis pass, the complex spectrum.
        let buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft2d::forward_2d buf_a (complex-sized)"),
            size: (total_cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft2d::forward_2d buf_b"),
            size: (total_cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_h = self
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let params_v = self
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let bg_h = self
            .h
            .make_bind_group(ctx, &buf_a, twiddles, &buf_b, &params_h);
        let bg_v = self
            .v
            .make_bind_group(ctx, &buf_b, twiddles, &buf_a, &params_v);

        // Single encoder = single submit: staging→buf_a copy, then
        // H+V dispatch. wgpu inserts implicit barriers between the
        // copy and the compute pass.
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft2d::forward_2d encoder"),
            });
        enc.copy_buffer_to_buffer(&staging, 0, &buf_a, 0, (total_cells * 4) as u64);
        self.h.record(&mut enc, &bg_h, n);
        self.v.record(&mut enc, &bg_v, n);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let flat = crate::readback::readback_buffer::<f32>(ctx, &buf_a, total_cells * 2);
        flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    }
}

/// Precompute the full N forward-FFT twiddle factors
/// `W_N^k = exp(-2πi k / N)` for k ∈ [0, N) and upload them as a
/// storage buffer.
///
/// The WGSL kernel uses three twiddle reads per butterfly (`w_idx`,
/// `2*w_idx`, `3*w_idx` for k, 2k, 3k). While `w_idx` itself lives
/// in [0, N/4) at every stage, `2*w_idx` reaches up to ~N/2 and
/// `3*w_idx` up to ~3N/4, so the buffer **must** cover the full
/// [0, N) range. A first attempt at C-1-1 stored only N/4 entries
/// (the radix-4 "first quadrant" economy from fgiesen 2023) and
/// silently OOB-read 0 for the upper-quadrant indices — caught by
/// the random-input rustfft test (the impulse-response test passed
/// regardless, because for a δ input every butterfly already has
/// p1=p2=p3=0 and the twiddle multiplies don't affect the output).
///
/// Inverse FFT re-uses this same table by conjugating in-shader
/// (`twiddle_for_direction`), so we keep one table per N — not two.
///
/// Built on CPU once at pipeline construction; uploaded as STORAGE
/// (not UNIFORM) because the buffer is read by stride-pattern indices
/// rather than as a fixed-size struct.
#[must_use]
pub fn precompute_twiddles_1d(ctx: &GpuContext, n: u32) -> wgpu::Buffer {
    assert!(
        n.is_power_of_two() && n >= 4,
        "FFT length must be a power of two ≥ 4 (got n = {n})"
    );
    let count = n as usize;
    let mut data: Vec<[f32; 2]> = Vec::with_capacity(count);
    let two_pi_over_n = -2.0_f64 * std::f64::consts::PI / f64::from(n);
    for k in 0..count {
        let theta = two_pi_over_n * (k as f64);
        // Compute in f64 then cast to f32 — the ulp difference vs
        // f32 cos/sin is small but recoverable for free at startup.
        data.push([theta.cos() as f32, theta.sin() as f32]);
    }
    ctx.device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("fft_1d_radix4 twiddle buffer"),
            contents: bytemuck::cast_slice(&data),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::readback::readback_buffer;
    use crate::GpuContext;
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;
    use rustfft::{num_complex::Complex32, FftPlanner};
    use wgpu::util::DeviceExt;

    fn headless_ctx() -> (GpuContext, Option<crate::validation::ValidationGuard>) {
        crate::validation::test_ctx_for_lib()
    }

    // ─── 1D tests (unchanged from C-1-1, retained for regression) ───────

    fn run_forward_1d(
        ctx: &GpuContext,
        pass: &FftPass,
        twiddles: &wgpu::Buffer,
        input_real: &[f32],
    ) -> Vec<[f32; 2]> {
        let n = pass.n as usize;
        assert!(
            input_real.len() % n == 0,
            "input length must be a multiple of N={n}"
        );
        let num_rows = (input_real.len() / n) as u32;
        let n_complex = (num_rows as usize) * n;

        let input_buf = ctx
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("fft test input"),
                contents: bytemuck::cast_slice(input_real),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            });
        let output_buf = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("fft test output"),
            size: (n_complex * 8) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let params_buf = pass.upload_params(
            ctx,
            FftParams::new(pass.n, num_rows, FftDirection::Forward),
        );
        let bg = pass.make_bind_group(ctx, &input_buf, twiddles, &output_buf, &params_buf);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("fft test encoder"),
            });
        pass.record(&mut enc, &bg, num_rows);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let flat = readback_buffer::<f32>(ctx, &output_buf, n_complex * 2);
        flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    }

    fn cpu_reference_fft_1d(n: usize, input: &[f32]) -> Vec<[f32; 2]> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);
        let mut buf: Vec<Complex32> = input.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        for chunk in buf.chunks_mut(n) {
            fft.process(chunk);
        }
        buf.iter().map(|c| [c.re, c.im]).collect()
    }

    /// **Observed on M1 mini (2026-05-25)**: max_abs ≈ 2.86e-6,
    /// max_rel ≈ 1.06e-6. Tolerance carries ~100× headroom.
    #[test]
    fn fft_1d_matches_rustfft() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new_h(&ctx, 256);
        let twiddles = precompute_twiddles_1d(&ctx, 256);

        let mut rng = ChaCha8Rng::seed_from_u64(0xFF7_1D42);
        let n_rows: usize = 4;
        let input: Vec<f32> = (0..(n_rows * 256))
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let gpu = run_forward_1d(&ctx, &pass, &twiddles, &input);
        let cpu = cpu_reference_fft_1d(256, &input);

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let abs_re = (g[0] - c[0]).abs();
            let abs_im = (g[1] - c[1]).abs();
            let mag = (c[0] * c[0] + c[1] * c[1]).sqrt().max(1e-6);
            let rel = abs_re.max(abs_im) / mag;
            max_abs = max_abs.max(abs_re.max(abs_im));
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1e-4 || abs_re.max(abs_im) < 1e-5,
                "bin {i}: rel={rel:.3e} abs_re={abs_re:.3e} abs_im={abs_im:.3e}"
            );
        }
        eprintln!("[M6.C-1-1] fft_1d vs rustfft N=256 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// 1D GPU forward + GPU inverse round-trip — isolates whether the
    /// inverse path is correct without involving 2D orchestration.
    /// Implementation hoisted into `run_1d_gpu_round_trip` so the N=64
    /// sibling shares the same code path (Round 2 review NC1).
    #[test]
    fn fft_1d_gpu_inverse_round_trip() {
        let max_abs = run_1d_gpu_round_trip(0xFF7_1D_67, 256);
        eprintln!("[M6.C-1-2] fft_1d GPU round-trip N=256 : max_abs={max_abs:.3e}");
    }

    /// Impulse response — see C-1-1 rustdoc for the coverage caveat
    /// (impulse passes regardless of twiddle bugs because p1=p2=p3=0).
    #[test]
    fn fft_1d_impulse_response_is_all_ones() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new_h(&ctx, 256);
        let twiddles = precompute_twiddles_1d(&ctx, 256);

        let mut input = vec![0.0_f32; 256];
        input[0] = 1.0;
        let gpu = run_forward_1d(&ctx, &pass, &twiddles, &input);
        for (i, c) in gpu.iter().enumerate() {
            let abs_re = (c[0] - 1.0).abs();
            let abs_im = c[1].abs();
            assert!(
                abs_re < 1e-5 && abs_im < 1e-5,
                "bin {i}: gpu=({:e}, {:e})",
                c[0],
                c[1]
            );
        }
        eprintln!("[M6.C-1-1] fft_1d impulse → all-ones : pass");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    // ─── 2D tests (C-1-2 new) ───────────────────────────────────────────

    /// Run a 2D forward (real H×N → complex H×N spectrum) followed by
    /// a 2D inverse (complex H×N → complex H×N with imag ≈ 0). Returns
    /// the inverse output `.x` field, which should equal the input
    /// modulo numerical drift.
    fn run_round_trip_2d(
        ctx: &GpuContext,
        pass2d: &Fft2dPass,
        twiddles: &wgpu::Buffer,
        input_real: &[f32],
        n: u32,
    ) -> Vec<f32> {
        let total_cells = (n * n) as usize;
        assert_eq!(input_real.len(), total_cells, "input must be H×N=N×N");

        // Buffer A: real input. Read by H-axis forward; rewritten by
        // H-axis inverse (as complex, so 2× capacity needed).
        let buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d round-trip buf_a (real/complex)"),
            // 2 × f32 per cell to fit complex on the way back.
            size: (total_cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        // Upload only the first N*N f32 (the real-input layout). The
        // remaining slots are unused on the forward path.
        ctx.queue
            .write_buffer(&buf_a, 0, bytemuck::cast_slice(input_real));

        // Buffer B: complex intermediate + final spectrum + inverse
        // intermediate, all reused. Sized for complex.
        let buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d round-trip buf_b (complex)"),
            size: (total_cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_fwd_h = pass2d
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let params_fwd_v = pass2d
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let params_inv_v = pass2d
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Inverse));
        let params_inv_h = pass2d
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Inverse));

        // Forward: A (real) → B (complex)  via H,  then B → A (complex) via V
        let bg_h_fwd =
            pass2d
                .h
                .make_bind_group(ctx, &buf_a, twiddles, &buf_b, &params_fwd_h);
        let bg_v_fwd =
            pass2d
                .v
                .make_bind_group(ctx, &buf_b, twiddles, &buf_a, &params_fwd_v);
        // Inverse: A (complex) → B (complex) via V, then B → A (complex, imag≈0) via H
        let bg_v_inv =
            pass2d
                .v
                .make_bind_group(ctx, &buf_a, twiddles, &buf_b, &params_inv_v);
        let bg_h_inv =
            pass2d
                .h
                .make_bind_group(ctx, &buf_b, twiddles, &buf_a, &params_inv_h);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("2d round-trip encoder"),
            });
        pass2d.h.record(&mut enc, &bg_h_fwd, n);
        pass2d.v.record(&mut enc, &bg_v_fwd, n);
        pass2d.v.record(&mut enc, &bg_v_inv, n);
        pass2d.h.record(&mut enc, &bg_h_inv, n);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        // Read back as complex; keep only the real part (.x of each
        // vec2). After 2× forward + 2× inverse with each axis
        // normalised by 1/N, the imag part should be < 1e-4.
        let flat = readback_buffer::<f32>(ctx, &buf_a, total_cells * 2);
        flat.chunks_exact(2).map(|c| c[0]).collect()
    }

    /// Run only the forward 2D, return the complex spectrum. Used by
    /// the rustfft 2D comparison test.
    fn run_forward_2d(
        ctx: &GpuContext,
        pass2d: &Fft2dPass,
        twiddles: &wgpu::Buffer,
        input_real: &[f32],
        n: u32,
    ) -> Vec<[f32; 2]> {
        let total_cells = (n * n) as usize;
        assert_eq!(input_real.len(), total_cells);

        let buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d forward buf_a"),
            size: (total_cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&buf_a, 0, bytemuck::cast_slice(input_real));

        let buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("2d forward buf_b"),
            size: (total_cells * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_fwd_h = pass2d
            .h
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));
        let params_fwd_v = pass2d
            .v
            .upload_params(ctx, FftParams::new(n, n, FftDirection::Forward));

        let bg_h_fwd =
            pass2d
                .h
                .make_bind_group(ctx, &buf_a, twiddles, &buf_b, &params_fwd_h);
        let bg_v_fwd =
            pass2d
                .v
                .make_bind_group(ctx, &buf_b, twiddles, &buf_a, &params_fwd_v);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("2d forward encoder"),
            });
        pass2d.h.record(&mut enc, &bg_h_fwd, n);
        pass2d.v.record(&mut enc, &bg_v_fwd, n);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let flat = readback_buffer::<f32>(ctx, &buf_a, total_cells * 2);
        flat.chunks_exact(2).map(|c| [c[0], c[1]]).collect()
    }

    fn cpu_reference_fft_2d(n: usize, input: &[f32]) -> Vec<[f32; 2]> {
        // Row-by-row 1D, then column-by-column 1D — separable.
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(n);

        let mut buf: Vec<Complex32> = input.iter().map(|&v| Complex32::new(v, 0.0)).collect();

        // H-axis rows
        for row in 0..n {
            let start = row * n;
            fft.process(&mut buf[start..start + n]);
        }
        // V-axis columns (gather into temp, transform, scatter back)
        let mut col_buf: Vec<Complex32> = vec![Complex32::new(0.0, 0.0); n];
        for col in 0..n {
            for r in 0..n {
                col_buf[r] = buf[r * n + col];
            }
            fft.process(&mut col_buf);
            for r in 0..n {
                buf[r * n + col] = col_buf[r];
            }
        }

        buf.iter().map(|c| [c.re, c.im]).collect()
    }

    /// 2D round-trip at N=256: forward → inverse → original input.
    /// Both axes normalised at 1/N in the inverse, total factor 1/N²
    /// matches rustfft's two-pass separable inverse.
    #[test]
    fn fft_2d_round_trip_n256() {
        let (ctx, guard) = headless_ctx();
        let pass2d = Fft2dPass::new(&ctx, 256);
        let twiddles = precompute_twiddles_1d(&ctx, 256);

        let mut rng = ChaCha8Rng::seed_from_u64(0xF7_2D_5C56);
        let n: u32 = 256;
        let input: Vec<f32> = (0..(n * n) as usize)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let roundtrip = run_round_trip_2d(&ctx, &pass2d, &twiddles, &input, n);

        let mut max_abs = 0.0_f32;
        for (i, (&r, &orig)) in roundtrip.iter().zip(input.iter()).enumerate() {
            let abs_err = (r - orig).abs();
            max_abs = max_abs.max(abs_err);
            assert!(
                abs_err < 1e-4,
                "2D round-trip mismatch at idx {i}: roundtrip={r} orig={orig} abs={abs_err:.3e}"
            );
        }
        eprintln!("[M6.C-1-2] fft_2d round-trip N=256 : max_abs={max_abs:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// 2D round-trip at N=64 — smaller-grid sanity for the dynamic-N
    /// pipeline-override path (different `WORKGROUP_X` from N=256).
    #[test]
    fn fft_2d_round_trip_n64() {
        let (ctx, guard) = headless_ctx();
        let pass2d = Fft2dPass::new(&ctx, 64);
        let twiddles = precompute_twiddles_1d(&ctx, 64);

        let mut rng = ChaCha8Rng::seed_from_u64(0xF7_2D_5C64);
        let n: u32 = 64;
        let input: Vec<f32> = (0..(n * n) as usize)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let roundtrip = run_round_trip_2d(&ctx, &pass2d, &twiddles, &input, n);

        let mut max_abs = 0.0_f32;
        for (i, (&r, &orig)) in roundtrip.iter().zip(input.iter()).enumerate() {
            let abs_err = (r - orig).abs();
            max_abs = max_abs.max(abs_err);
            assert!(
                abs_err < 1e-4,
                "2D round-trip mismatch at idx {i}: roundtrip={r} orig={orig} abs={abs_err:.3e}"
            );
        }
        eprintln!("[M6.C-1-2] fft_2d round-trip N=64 : max_abs={max_abs:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// M6.C-3-2: 2D round-trip at **N=512** (mixed-radix). Exercises
    /// the full `Fft2dPass::new(512)` → mixed H + mixed V forward,
    /// then mixed V + mixed H inverse, column-stride and row-stride
    /// both. This is the 2D-orchestration correctness anchor for the
    /// 512 hi-end mode before the ConvolveFftPass / pipeline wiring.
    #[test]
    fn fft_2d_round_trip_n512() {
        let (ctx, guard) = headless_ctx();
        let pass2d = Fft2dPass::new(&ctx, 512);
        let twiddles = precompute_twiddles_1d(&ctx, 512);

        let mut rng = ChaCha8Rng::seed_from_u64(0xF7_2D_512);
        let n: u32 = 512;
        let input: Vec<f32> = (0..(n * n) as usize)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let roundtrip = run_round_trip_2d(&ctx, &pass2d, &twiddles, &input, n);

        let mut max_abs = 0.0_f32;
        for (i, (&r, &orig)) in roundtrip.iter().zip(input.iter()).enumerate() {
            let abs_err = (r - orig).abs();
            max_abs = max_abs.max(abs_err);
            assert!(
                abs_err < 1e-4,
                "2D round-trip N=512 mismatch at idx {i}: roundtrip={r} orig={orig} \
                 abs={abs_err:.3e}"
            );
        }
        eprintln!("[M6.C-3-2] fft_2d round-trip N=512 : max_abs={max_abs:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// 2D forward output matches rustfft's separable two-pass output.
    /// This is the canonical correctness gate; the round-trip tests
    /// catch sign-convention bugs but cannot distinguish "GPU and
    /// CPU agree on the wrong answer" from "both are right".
    ///
    /// Parametrised over `SUPPORTED_N` so the dynamic-N path
    /// (pipeline-override `WORKGROUP_X`) and the `digit_reverse_4_dynamic`
    /// log₄(n)-digit loop are exercised at both N=64 and N=256.
    /// Round 1 review M1: without the N=64 leg, a hypothetical
    /// off-by-one in either the digit reversal or the stage loop
    /// would be absorbed by round-trip alone and miss the rustfft
    /// witness.
    fn run_forward_vs_rustfft_2d(seed: u64, n: u32) -> (f32, f32) {
        let (ctx, guard) = headless_ctx();
        let pass2d = Fft2dPass::new(&ctx, n);
        let twiddles = precompute_twiddles_1d(&ctx, n);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let input: Vec<f32> = (0..(n * n) as usize)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let gpu = run_forward_2d(&ctx, &pass2d, &twiddles, &input, n);
        let cpu = cpu_reference_fft_2d(n as usize, &input);

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let abs_re = (g[0] - c[0]).abs();
            let abs_im = (g[1] - c[1]).abs();
            let mag = (c[0] * c[0] + c[1] * c[1]).sqrt().max(1e-6);
            let rel = abs_re.max(abs_im) / mag;
            max_abs = max_abs.max(abs_re.max(abs_im));
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1e-3 || abs_re.max(abs_im) < 1e-4,
                "2D bin {i} (N={n}): gpu=({:e},{:e}) cpu=({:e},{:e}) rel={rel:.3e}",
                g[0],
                g[1],
                c[0],
                c[1]
            );
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
        (max_abs, max_rel)
    }

    #[test]
    fn fft_2d_forward_matches_rustfft_n256() {
        let (max_abs, max_rel) = run_forward_vs_rustfft_2d(0xF7_2D_F256, 256);
        eprintln!(
            "[M6.C-1-2] fft_2d vs rustfft N=256 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );
    }

    /// Round 1 review M1: dynamic-N path witness at N=64.
    #[test]
    fn fft_2d_forward_matches_rustfft_n64() {
        let (max_abs, max_rel) = run_forward_vs_rustfft_2d(0xF7_2D_F064, 64);
        eprintln!(
            "[M6.C-1-2] fft_2d vs rustfft N=64  : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );
    }

    /// M6.C-3-2: **independent 2D forward witness at N=512** vs
    /// rustfft. This is the canonical correctness gate (the round-trip
    /// test only proves GPU forward/inverse are mutual inverses — it
    /// cannot catch a forward-only V-mixed column-stride bug that the
    /// inverse exactly undoes). The mixed-H forward is witnessed at 1D
    /// (`fft_1d_mixed_matches_rustfft_n512`); this closes the gap for
    /// the mixed-V column-stride forward + the full 2D composition.
    /// (adversarial-reviewer C-3-2 required measurement.)
    #[test]
    fn fft_2d_forward_matches_rustfft_n512() {
        let (max_abs, max_rel) = run_forward_vs_rustfft_2d(0xF7_2D_F512, 512);
        eprintln!(
            "[M6.C-3-2] fft_2d vs rustfft N=512 : max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );
    }

    /// Round 1 review M2: 1D GPU forward + GPU inverse round-trip at
    /// N=64 — the dynamic-N path needs an inverse witness too, not
    /// just at N=256 (forward+inverse separately could each be wrong
    /// in cancelling ways the N=256 test happens to miss).
    fn run_1d_gpu_round_trip(seed: u64, n: u32) -> f32 {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new_h(&ctx, n);
        let twiddles = precompute_twiddles_1d(&ctx, n);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let input: Vec<f32> = (0..n as usize)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("1d gpu round-trip A (parametric)"),
            size: (n as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&buf_a, 0, bytemuck::cast_slice(&input));

        let buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("1d gpu round-trip B (parametric)"),
            size: (n as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_fwd = pass.upload_params(&ctx, FftParams::new(n, 1, FftDirection::Forward));
        let params_inv = pass.upload_params(&ctx, FftParams::new(n, 1, FftDirection::Inverse));
        let bg_fwd = pass.make_bind_group(&ctx, &buf_a, &twiddles, &buf_b, &params_fwd);
        let bg_inv = pass.make_bind_group(&ctx, &buf_b, &twiddles, &buf_a, &params_inv);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("1d gpu round-trip encoder (parametric)"),
            });
        pass.record(&mut enc, &bg_fwd, 1);
        pass.record(&mut enc, &bg_inv, 1);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let flat = readback_buffer::<f32>(&ctx, &buf_a, n as usize * 2);
        let recovered: Vec<f32> = flat.chunks_exact(2).map(|c| c[0]).collect();

        let mut max_abs = 0.0_f32;
        for (i, (&r, &orig)) in recovered.iter().zip(input.iter()).enumerate() {
            let abs_err = (r - orig).abs();
            max_abs = max_abs.max(abs_err);
            assert!(
                abs_err < 1e-4,
                "1D GPU round-trip (N={n}) mismatch at idx {i}: \
                 recovered={r} orig={orig} abs={abs_err:.3e}"
            );
        }
        if let Some(g) = &guard {
            g.assert_no_errors();
        }
        max_abs
    }

    #[test]
    fn fft_1d_gpu_inverse_round_trip_n64() {
        let max_abs = run_1d_gpu_round_trip(0xFF7_1D_64, 64);
        eprintln!("[M6.C-1-2] fft_1d GPU round-trip N=64  : max_abs={max_abs:.3e}");
    }

    // ─── M6.C-3-1 mixed-radix (N = 2 × 4^k) tests ──────────────────────

    /// Forward mixed-radix N=512 vs rustfft, **random input** (4 rows).
    /// Random input is essential here: it exercises every twiddle
    /// (impulse passes regardless of twiddle bugs — see C-1-1 rustdoc).
    /// This is the primary correctness anchor for the radix-2 combine
    /// stage at the target size.
    #[test]
    fn fft_1d_mixed_matches_rustfft_n512() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new_h_mixed(&ctx, 512);
        let twiddles = precompute_twiddles_1d(&ctx, 512);

        let mut rng = ChaCha8Rng::seed_from_u64(0x512_FF7);
        let n_rows: usize = 4;
        let input: Vec<f32> = (0..(n_rows * 512))
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let gpu = run_forward_1d(&ctx, &pass, &twiddles, &input);
        let cpu = cpu_reference_fft_1d(512, &input);

        let mut max_abs = 0.0_f32;
        let mut max_rel = 0.0_f32;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let abs_re = (g[0] - c[0]).abs();
            let abs_im = (g[1] - c[1]).abs();
            let mag = (c[0] * c[0] + c[1] * c[1]).sqrt().max(1e-6);
            let rel = abs_re.max(abs_im) / mag;
            max_abs = max_abs.max(abs_re.max(abs_im));
            max_rel = max_rel.max(rel);
            assert!(
                rel < 1e-4 || abs_re.max(abs_im) < 1e-5,
                "bin {i}: rel={rel:.3e} abs_re={abs_re:.3e} abs_im={abs_im:.3e}"
            );
        }
        eprintln!(
            "[M6.C-3-1] fft_1d_mixed vs rustfft N=512 : max_abs={max_abs:.3e}  \
             max_rel={max_rel:.3e}"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// Forward mixed-radix **N=8** (= 2 × 4) vs rustfft, random input.
    /// Small-case cross-check of the radix-2 combine stage (1 radix-4
    /// stage + 1 radix-2 stage). N=8 = M=4 case: digit_reverse over a
    /// single base-4 digit, then the even/odd radix-2 combine. Catches
    /// reversal / combine-index bugs that N=512 might mask via averaging.
    #[test]
    fn fft_1d_mixed_matches_rustfft_n8() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new_h_mixed(&ctx, 8);
        let twiddles = precompute_twiddles_1d(&ctx, 8);

        let mut rng = ChaCha8Rng::seed_from_u64(0x8_FF7_42);
        // 3 rows to exercise the row_base offset path too.
        let n_rows: usize = 3;
        let input: Vec<f32> = (0..(n_rows * 8))
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let gpu = run_forward_1d(&ctx, &pass, &twiddles, &input);
        let cpu = cpu_reference_fft_1d(8, &input);

        let mut max_abs = 0.0_f32;
        for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
            let abs_re = (g[0] - c[0]).abs();
            let abs_im = (g[1] - c[1]).abs();
            max_abs = max_abs.max(abs_re.max(abs_im));
            assert!(
                abs_re.max(abs_im) < 1e-5,
                "bin {i}: gpu=({:e},{:e}) cpu=({:e},{:e})  abs={:.3e}",
                g[0],
                g[1],
                c[0],
                c[1],
                abs_re.max(abs_im)
            );
        }
        eprintln!("[M6.C-3-1] fft_1d_mixed vs rustfft N=8 : max_abs={max_abs:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// Impulse response N=512 → all-ones (sanity; passes regardless of
    /// twiddle bugs per C-1-1 caveat, but catches dispatch-shape /
    /// store-index errors).
    #[test]
    fn fft_1d_mixed_impulse_n512() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new_h_mixed(&ctx, 512);
        let twiddles = precompute_twiddles_1d(&ctx, 512);

        let mut input = vec![0.0_f32; 512];
        input[0] = 1.0;
        let gpu = run_forward_1d(&ctx, &pass, &twiddles, &input);
        for (i, c) in gpu.iter().enumerate() {
            let abs_re = (c[0] - 1.0).abs();
            let abs_im = c[1].abs();
            assert!(
                abs_re < 1e-5 && abs_im < 1e-5,
                "bin {i}: gpu=({:e}, {:e})",
                c[0],
                c[1]
            );
        }
        eprintln!("[M6.C-3-1] fft_1d_mixed impulse N=512 → all-ones : pass");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// GPU forward + GPU inverse round-trip for mixed-radix N. Isolates
    /// the inverse path (Method B conjugate-load/store) at the target
    /// size without 2D orchestration.
    fn run_1d_gpu_round_trip_mixed(seed: u64, n: u32) -> f32 {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new_h_mixed(&ctx, n);
        let twiddles = precompute_twiddles_1d(&ctx, n);

        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let input: Vec<f32> = (0..n as usize)
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let buf_a = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("1d mixed round-trip A"),
            size: (n as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        ctx.queue
            .write_buffer(&buf_a, 0, bytemuck::cast_slice(&input));

        let buf_b = ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("1d mixed round-trip B"),
            size: (n as usize * 2 * 4) as u64,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let params_fwd = pass.upload_params(&ctx, FftParams::new(n, 1, FftDirection::Forward));
        let params_inv = pass.upload_params(&ctx, FftParams::new(n, 1, FftDirection::Inverse));
        let bg_fwd = pass.make_bind_group(&ctx, &buf_a, &twiddles, &buf_b, &params_fwd);
        let bg_inv = pass.make_bind_group(&ctx, &buf_b, &twiddles, &buf_a, &params_inv);

        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("1d mixed round-trip encoder"),
            });
        pass.record(&mut enc, &bg_fwd, 1);
        pass.record(&mut enc, &bg_inv, 1);
        ctx.queue.submit([enc.finish()]);
        ctx.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: None,
            })
            .unwrap();

        let flat = readback_buffer::<f32>(&ctx, &buf_a, n as usize * 2);
        let recovered: Vec<f32> = flat.chunks_exact(2).map(|c| c[0]).collect();

        let mut max_abs = 0.0_f32;
        for (i, (&r, &orig)) in recovered.iter().zip(input.iter()).enumerate() {
            let abs_err = (r - orig).abs();
            max_abs = max_abs.max(abs_err);
            assert!(
                abs_err < 1e-4,
                "1D mixed GPU round-trip (N={n}) mismatch at idx {i}: \
                 recovered={r} orig={orig} abs={abs_err:.3e}"
            );
        }
        if let Some(g) = &guard {
            g.assert_no_errors();
        }
        max_abs
    }

    #[test]
    fn fft_1d_mixed_round_trip_n512() {
        let max_abs = run_1d_gpu_round_trip_mixed(0x512_1D_67, 512);
        eprintln!("[M6.C-3-1] fft_1d_mixed GPU round-trip N=512 : max_abs={max_abs:.3e}");
    }

    #[test]
    fn fft_1d_mixed_round_trip_n8() {
        let max_abs = run_1d_gpu_round_trip_mixed(0x8_1D_67, 8);
        eprintln!("[M6.C-3-1] fft_1d_mixed GPU round-trip N=8 : max_abs={max_abs:.3e}");
    }

    /// Forward mixed-radix vs rustfft for the intermediate declared
    /// sizes N=32 (2 radix-4 stages + radix-2) and N=128 (3 stages +
    /// radix-2). Closes the "declared in SUPPORTED_MIXED_N but
    /// untested" gap (adversarial-reviewer C-3-1): these exercise the
    /// non-top-stage twiddle stride `local_idx · N/stage_size` against
    /// the full-N table that the N=8/N=512 endpoints alone don't fully
    /// cover. Random input (impulse would mask twiddle bugs).
    #[test]
    fn fft_1d_mixed_matches_rustfft_n32_n128() {
        let (ctx, guard) = headless_ctx();
        for (n, seed) in [(32u32, 0x32_FF7_u64), (128u32, 0x128_FF7_u64)] {
            let pass = FftPass::new_h_mixed(&ctx, n);
            let twiddles = precompute_twiddles_1d(&ctx, n);

            let mut rng = ChaCha8Rng::seed_from_u64(seed);
            let n_rows: usize = 3;
            let input: Vec<f32> = (0..(n_rows * n as usize))
                .map(|_| rng.gen_range(-1.0_f32..1.0))
                .collect();

            let gpu = run_forward_1d(&ctx, &pass, &twiddles, &input);
            let cpu = cpu_reference_fft_1d(n as usize, &input);

            let mut max_abs = 0.0_f32;
            let mut max_rel = 0.0_f32;
            for (i, (g, c)) in gpu.iter().zip(cpu.iter()).enumerate() {
                let abs_re = (g[0] - c[0]).abs();
                let abs_im = (g[1] - c[1]).abs();
                let mag = (c[0] * c[0] + c[1] * c[1]).sqrt().max(1e-6);
                let rel = abs_re.max(abs_im) / mag;
                max_abs = max_abs.max(abs_re.max(abs_im));
                max_rel = max_rel.max(rel);
                assert!(
                    rel < 1e-4 || abs_re.max(abs_im) < 1e-5,
                    "N={n} bin {i}: rel={rel:.3e} abs_re={abs_re:.3e} abs_im={abs_im:.3e}"
                );
            }
            eprintln!(
                "[M6.C-3-1] fft_1d_mixed vs rustfft N={n} : max_abs={max_abs:.3e}  \
                 max_rel={max_rel:.3e}"
            );
        }

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }
}
