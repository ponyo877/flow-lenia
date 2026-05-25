//! M6.C-1-1 — 1D radix-4 Cooley-Tukey FFT compute pass.
//!
//! Compiles `src/shaders/fft_1d_radix4.wgsl` (see that file for the
//! algorithm and binding contract) and exposes:
//!
//! - [`FftPass`] — pipeline + bind-group layout, one-shot construction
//!   per `GpuContext`. Currently locked to N=256; the assertion lives
//!   in [`FftPass::upload_params`] so the WGSL invariants documented
//!   at the top of `fft_1d_radix4.wgsl` cannot be silently violated
//!   on the standard build path (callers constructing their own
//!   uniform buffer bypass this gate; C-1-2 will generalise the WGSL
//!   to runtime-N and the assertion lifts with it).
//! - [`precompute_twiddles_1d`] — uploads the N/4 forward-FFT
//!   twiddle factors `W_N^k = exp(-2πi k / N)` once at startup, used
//!   for every subsequent dispatch.
//! - [`FftParams`] — `vec2<u32> = (n, num_rows)` uniform, matching
//!   the WGSL `struct FftParams`.
//!
//! Per-step usage (once C-1-4 wires this into `ConvolvePass`):
//!
//! ```text
//! let pass = FftPass::new(&ctx);                      // startup
//! let twiddles = precompute_twiddles_1d(&ctx, 256);   // startup
//! let bg = pass.make_bind_group(&ctx, &in_buf, &twiddles, &out_buf, &params_buf);
//! let mut enc = ctx.device.create_command_encoder(...);
//! pass.record(&mut enc, &bg, num_rows);
//! ctx.queue.submit([enc.finish()]);
//! ```
//!
//! `record` only appends the dispatch; submission and synchronisation
//! are the caller's responsibility, matching the M2.3+ pattern used
//! by every other pass in this crate.

use crate::GpuContext;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

/// Fixed 1D transform length for the C-1-1 primitive. Generalising
/// to runtime-chosen N (32 / 64 / 128 / 256 / 512) is C-1-2 scope.
pub const FFT_N: u32 = 256;

/// Workgroup width matches `@workgroup_size(256, 1, 1)` in
/// `fft_1d_radix4.wgsl` and equals `FFT_N` by construction. Exposed
/// so downstream tests can compute the dispatch count without
/// re-deriving the constant.
pub const WORKGROUP_X: u32 = FFT_N;

/// Mirror of the WGSL `struct FftParams` — push-constants would be
/// nicer but wgpu's `Features::PUSH_CONSTANTS` is not part of the
/// WebGPU 1.0 spec, so a uniform buffer keeps the path web-compatible.
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct FftParams {
    pub n: u32,
    pub num_rows: u32,
}

/// Compiled 1D radix-4 FFT pass.
pub struct FftPass {
    pub pipeline: wgpu::ComputePipeline,
    pub bind_group_layout: wgpu::BindGroupLayout,
}

impl FftPass {
    /// Compile `fft_1d_radix4.wgsl` and build the pipeline. Call once
    /// per `GpuContext`.
    #[must_use]
    pub fn new(ctx: &GpuContext) -> Self {
        const SOURCE: &str = include_str!("../shaders/fft_1d_radix4.wgsl");
        let shader = ctx
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("fft_1d_radix4.wgsl"),
                source: wgpu::ShaderSource::Wgsl(SOURCE.into()),
            });

        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("fft_1d_radix4 bind group layout"),
                    entries: &[
                        // input: storage<read>, real f32
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
                        // twiddles: storage<read>, vec2<f32>
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
                        // output: storage<read_write>, vec2<f32>
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
                        // params: uniform
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
                label: Some("fft_1d_radix4 pipeline layout"),
                bind_group_layouts: &[Some(&bind_group_layout)],
                immediate_size: 0,
            });

        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("fft_1d_radix4 pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("fft_1d_radix4"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });

        Self {
            pipeline,
            bind_group_layout,
        }
    }

    /// Assemble a bind group for one (input, twiddles, output, params)
    /// quadruple. Hot-loop callers (post C-1-4) should reuse a single
    /// bind group across submissions when the buffer identities don't
    /// change — only `params_buf`'s `num_rows` field varies.
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

    /// Append one dispatch processing `num_rows` rows of `FFT_N=256`
    /// samples each. The caller is responsible for setting
    /// `params.num_rows` in the uniform buffer before submitting.
    pub fn record(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        bind_group: &wgpu::BindGroup,
        num_rows: u32,
    ) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("fft_1d_radix4 pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&self.pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        // dispatch (num_rows, 1, 1) workgroups; each is 256 threads
        // processing one row.
        pass.dispatch_workgroups(num_rows, 1, 1);
    }

    /// Convenience: upload an `FftParams` value as a uniform buffer.
    ///
    /// Enforces the C-1-1 invariant `params.n == FFT_N` at the helper
    /// boundary (Round 1 review M1: the rustdoc at the top of this
    /// file promised `record()`-side enforcement that did not exist).
    /// Lifting this assertion lives at C-1-2, when the WGSL gains
    /// dynamic-N support.
    #[must_use]
    pub fn upload_params(ctx: &GpuContext, params: FftParams) -> wgpu::Buffer {
        assert_eq!(
            params.n, FFT_N,
            "FftParams.n must equal FFT_N={FFT_N} for the C-1-1 primitive \
             (the WGSL digit_reverse_4 and stage loop bake N=256). \
             Dynamic-N support lands in C-1-2."
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
/// The 4× memory cost over the "first quadrant" trick is 256 ×
/// 8 byte = 2 KB at N=256, dwarfed by the per-row workgroup-memory
/// scratch and not worth the index-folding complexity it would
/// require. A future RFFT-packed variant (deferred) may revisit.
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

    /// Helper: run forward FFT on `num_rows × FFT_N` real input,
    /// return the full `num_rows × FFT_N` complex output flat.
    fn run_forward(
        ctx: &GpuContext,
        pass: &FftPass,
        twiddles: &wgpu::Buffer,
        input_real: &[f32],
    ) -> Vec<[f32; 2]> {
        assert!(
            input_real.len() % FFT_N as usize == 0,
            "input length must be a multiple of FFT_N=256 (got {})",
            input_real.len()
        );
        let num_rows = (input_real.len() / FFT_N as usize) as u32;
        let n_complex = (num_rows * FFT_N) as usize;

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
        let params_buf = FftPass::upload_params(
            ctx,
            FftParams {
                n: FFT_N,
                num_rows,
            },
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

    fn cpu_reference_fft(input: &[f32]) -> Vec<[f32; 2]> {
        let mut planner = FftPlanner::<f32>::new();
        let fft = planner.plan_fft_forward(FFT_N as usize);
        let mut buf: Vec<Complex32> = input.iter().map(|&v| Complex32::new(v, 0.0)).collect();
        // rustfft processes in chunks of FFT_N
        for chunk in buf.chunks_mut(FFT_N as usize) {
            fft.process(chunk);
        }
        buf.iter().map(|c| [c.re, c.im]).collect()
    }

    /// Compare GPU FFT against rustfft on a deterministic random
    /// signal. rustfft is the canonical CPU reference; relative
    /// tolerance 1e-4 absorbs the radix-4 vs split-radix accumulation-
    /// order f32 drift while still catching algorithmic bugs.
    ///
    /// **Observed on M1 mini (2026-05-25)**: `max_abs ≈ 2.86e-6`,
    /// `max_rel ≈ 1.06e-6`. The committed tolerance (`rel < 1e-4 ||
    /// abs < 1e-5`) carries ~100× headroom over both measurements —
    /// loose enough to survive driver / Naga upgrades, tight enough
    /// that a 100× regression from a real bug trips immediately.
    /// Tighten the screw if a future commit shows the observed
    /// values stable at 10× headroom across multiple runs.
    #[test]
    fn fft_1d_matches_rustfft() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new(&ctx);
        let twiddles = precompute_twiddles_1d(&ctx, FFT_N);

        let mut rng = ChaCha8Rng::seed_from_u64(0xFF7_1D42);
        let n_rows: usize = 4;
        let input: Vec<f32> = (0..(n_rows * FFT_N as usize))
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let gpu = run_forward(&ctx, &pass, &twiddles, &input);
        let cpu = cpu_reference_fft(&input);

        assert_eq!(gpu.len(), cpu.len(), "output length mismatch");
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
                "bin {i} (row {row}, freq {freq}): gpu=({g0:e}, {g1:e}) cpu=({c0:e}, {c1:e}) \
                 abs_re={abs_re:.3e} abs_im={abs_im:.3e} rel={rel:.3e}",
                row = i / FFT_N as usize,
                freq = i % FFT_N as usize,
                g0 = g[0],
                g1 = g[1],
                c0 = c[0],
                c1 = c[1],
            );
        }
        eprintln!(
            "[M6.C-1-1] fft_1d vs rustfft, {n_rows} × {FFT_N} : \
             max_abs={max_abs:.3e}  max_rel={max_rel:.3e}"
        );

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// Round-trip: forward(GPU) → inverse(rustfft) ≈ original input.
    /// Inverse is intentionally CPU-only in C-1-1; the GPU inverse
    /// is its own deliverable in a later sub-step.
    ///
    /// **Observed on M1 mini (2026-05-25)**: `max_abs ≈ 2.38e-7`.
    /// Tolerance 1e-4 carries ~400× headroom.
    #[test]
    fn fft_1d_round_trip_via_rustfft_inverse() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new(&ctx);
        let twiddles = precompute_twiddles_1d(&ctx, FFT_N);

        let mut rng = ChaCha8Rng::seed_from_u64(0xFF7_1D43);
        let n_rows: usize = 2;
        let input: Vec<f32> = (0..(n_rows * FFT_N as usize))
            .map(|_| rng.gen_range(-1.0_f32..1.0))
            .collect();

        let gpu_spectrum = run_forward(&ctx, &pass, &twiddles, &input);

        // rustfft inverse expects Complex32; rustfft does not include
        // the 1/N normalisation, so we divide manually.
        let mut planner = FftPlanner::<f32>::new();
        let ifft = planner.plan_fft_inverse(FFT_N as usize);
        let mut roundtrip: Vec<f32> = Vec::with_capacity(input.len());
        for row_chunk in gpu_spectrum.chunks_exact(FFT_N as usize) {
            let mut buf: Vec<Complex32> = row_chunk
                .iter()
                .map(|c| Complex32::new(c[0], c[1]))
                .collect();
            ifft.process(&mut buf);
            let scale = 1.0_f32 / FFT_N as f32;
            for c in &buf {
                roundtrip.push(c.re * scale);
            }
        }

        let mut max_abs = 0.0_f32;
        for (i, (&r, &orig)) in roundtrip.iter().zip(input.iter()).enumerate() {
            let abs_err = (r - orig).abs();
            max_abs = max_abs.max(abs_err);
            assert!(
                abs_err < 1e-4,
                "round-trip mismatch at idx {i}: roundtrip={r} orig={orig} abs={abs_err:.3e}"
            );
        }
        eprintln!("[M6.C-1-1] fft_1d round-trip (rustfft inverse) : max_abs={max_abs:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }

    /// Impulse response: δ at idx=0 → spectrum is all (1, 0). This is
    /// the simplest possible FFT sanity check, catching gross issues
    /// like the wrong twiddle sign or a bit-reversal mismatch that
    /// the random-input test might absorb into a "small" rel error.
    ///
    /// **Coverage caveat** (Round 1 review concern #3): for δ at
    /// idx=0, every butterfly has p1=p2=p3=0, so the twiddle
    /// multiplies are silently bypassed and a twiddle-buffer bug does
    /// NOT trip this test. The C-1-1 development cycle hit exactly
    /// this: the initial N/4-entry twiddle buffer passed impulse but
    /// failed `fft_1d_matches_rustfft` — see `precompute_twiddles_1d`
    /// rustdoc for the post-mortem. Treat impulse as a "shader
    /// loads" sanity, not a correctness gate; correctness lives in
    /// the random-vs-rustfft comparison.
    ///
    /// **Observed on M1 mini (2026-05-25)**: `max_abs = 0.0` (the
    /// trivial sum/diff arithmetic of all-zero p1/p2/p3 cancels
    /// exactly). Tolerance 1e-5 is purely defensive.
    #[test]
    fn fft_1d_impulse_response_is_all_ones() {
        let (ctx, guard) = headless_ctx();
        let pass = FftPass::new(&ctx);
        let twiddles = precompute_twiddles_1d(&ctx, FFT_N);

        let mut input = vec![0.0_f32; FFT_N as usize];
        input[0] = 1.0;

        let gpu = run_forward(&ctx, &pass, &twiddles, &input);
        assert_eq!(gpu.len(), FFT_N as usize);

        // Every bin should be (1, 0). f32 round-trip through cos/sin
        // and 4 stages of accumulation gives ulp-scale drift; 1e-5
        // is a generous bound.
        let mut max_abs = 0.0_f32;
        for (i, c) in gpu.iter().enumerate() {
            let abs_re = (c[0] - 1.0).abs();
            let abs_im = c[1].abs();
            max_abs = max_abs.max(abs_re.max(abs_im));
            assert!(
                abs_re < 1e-5 && abs_im < 1e-5,
                "impulse bin {i}: gpu=({:e}, {:e}), expected (1, 0)",
                c[0],
                c[1]
            );
        }
        eprintln!("[M6.C-1-1] fft_1d impulse → all-ones : max_abs={max_abs:.3e}");

        if let Some(g) = &guard {
            g.assert_no_errors();
        }
    }
}
