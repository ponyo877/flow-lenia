#![deny(warnings)]
//! M2.11 step-time benchmark.
//!
//! Runs three measurement sections and prints results in a form
//! suitable for direct transcription into BENCH.md:
//!
//! 1. **Full-step matrix**: per-step time CPU vs GPU across
//!    `(grid, C) ∈ {32, 64, 128, 256} × {1, 3}` (and 512 if it
//!    fits the adapter limits). |K| = 10 fixed, seed = 1729.
//! 2. **Per-pass breakdown** (GPU only) on the 64×64 / C=3 config:
//!    measures each of the 5 compute passes plus the visualize
//!    render pass in isolation by submitting 1000 dispatches.
//! 3. **Pipeline init + memory** at the same grid sizes —
//!    `GpuStepPipeline::new()` wall time and the host-side count of
//!    bytes allocated to GPU buffers.
//!
//! Usage:
//!
//! ```text
//! cargo run --release --bin bench_step
//! ```
//!
//! The visualize pass is **excluded** from the full-step matrix —
//! we want the pure simulator throughput, not the render cost. The
//! per-pass section measures it separately for reference.

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{
    activation_buffer::upload_activation,
    kernel_buffers::upload_kernels,
    passes::{
        affinity_growth::{upload_constant_weights, AffinityGrowthPass, GpuConstantWeights},
        convolve::ConvolvePass,
        flow::FlowPass,
        gradient::GradientPass,
        reintegrate::ReintegratePass,
        visualize::VisualizePass,
    },
    GpuContext, GpuGlobals, GpuStepPipeline,
};
use std::time::Instant;

const SEED: u64 = 1729;
const NUM_KERNELS: u32 = 10;

fn cfg(grid: u32, channels: u32) -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: grid,
        grid_height: grid,
        channels,
        dt: 0.2,
        sigma: 0.65,
        n: 2.0,
        beta_a: 2.0,
        dd: 5,
        num_kernels: NUM_KERNELS,
        paper_strict: false,
        border: BorderMode::Torus,
        mix_rule: MixRule::Stochastic,
    }
}

/// Measurement step count is grid-dependent so each case takes a
/// reasonable wall-clock time (~10..30 s in the worst case).
fn measure_steps(grid: u32) -> u32 {
    match grid {
        0..=64 => 1000,
        65..=128 => 300,
        129..=256 => 100,
        _ => 50,
    }
}

fn warmup_steps(grid: u32) -> u32 {
    (measure_steps(grid) / 10).max(10)
}

fn headless_ctx() -> GpuContext {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    GpuContext::new_blocking(instance, None)
}

// ────────────────────────────────────────────────────────────────────
// Section 1 — full-step matrix
// ────────────────────────────────────────────────────────────────────

// `StepResult` fields are only read indirectly through the
// table-printing eprintln! above; they're retained for future
// programmatic re-use (e.g. emitting a CSV).
#[allow(dead_code)]
struct StepResult {
    grid: u32,
    channels: u32,
    cpu_us: f64,
    gpu_us: f64,
    warmup: u32,
    measure: u32,
}

fn section_full_matrix(ctx: &GpuContext, grids: &[u32], channels_list: &[u32]) -> Vec<StepResult> {
    eprintln!("\n## Section 1 — full-step matrix (CPU vs GPU, |K|=10, seed=1729)\n");
    eprintln!(
        "| grid | C | warmup | measure | cpu us/step | gpu us/step | cpu ms/step | gpu ms/step | GPU/CPU | step rate (CPU, GPU) |\n\
         |-----:|--:|-------:|--------:|------------:|------------:|------------:|------------:|--------:|----------------------|"
    );
    let mut results = Vec::new();
    for &grid in grids {
        for &channels in channels_list {
            let w = warmup_steps(grid);
            let m = measure_steps(grid);
            let cfg = cfg(grid, channels);

            // CPU
            let mut cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
            cpu_sim.step_many(w);
            let cpu_started = Instant::now();
            cpu_sim.step_many(m);
            let cpu_us = cpu_started.elapsed().as_secs_f64() * 1e6 / f64::from(m);

            // GPU
            let setup_sim = FlowLeniaSimulator::new(cfg, SEED);
            let initial_a = setup_sim.activation().clone();
            let kernel_params = setup_sim.kernel_params().clone();
            let mut gpu_pipeline = GpuStepPipeline::new(ctx, &cfg, &kernel_params, &initial_a);
            gpu_pipeline.run_steps(ctx, w);

            let gpu_started = Instant::now();
            for _ in 0..m {
                gpu_pipeline.step(ctx);
            }
            ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
            let gpu_us = gpu_started.elapsed().as_secs_f64() * 1e6 / f64::from(m);

            let ratio = gpu_us / cpu_us;
            let cpu_rate = 1e6 / cpu_us;
            let gpu_rate = 1e6 / gpu_us;
            eprintln!(
                "| {grid:4} | {channels} | {w:6} | {m:7} | {cpu_us:11.1} | {gpu_us:11.1} | {:11.3} | {:11.3} | {ratio:7.2} | {cpu_rate:7.1} sps, {gpu_rate:7.1} sps |",
                cpu_us / 1000.0,
                gpu_us / 1000.0,
            );
            results.push(StepResult {
                grid,
                channels,
                cpu_us,
                gpu_us,
                warmup: w,
                measure: m,
            });
        }
    }
    results
}

// ────────────────────────────────────────────────────────────────────
// Section 2 — per-pass breakdown at 64×64 / C=3
// ────────────────────────────────────────────────────────────────────

fn measure_pass<F: FnMut(&mut wgpu::CommandEncoder)>(
    ctx: &GpuContext,
    label: &str,
    iters: u32,
    mut record: F,
) -> f64 {
    // Single warmup submit, then `iters` submits inside the timed
    // region. Submit + poll(Wait) per iteration gives an upper-bound
    // wall-clock for each pass's GPU + queue cost.
    let mut warm_enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
    record(&mut warm_enc);
    ctx.queue.submit([warm_enc.finish()]);
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

    let started = Instant::now();
    for _ in 0..iters {
        let mut enc = ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some(label) });
        record(&mut enc);
        ctx.queue.submit([enc.finish()]);
    }
    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
    started.elapsed().as_secs_f64() * 1e6 / f64::from(iters)
}

fn section_per_pass(ctx: &GpuContext) {
    eprintln!("\n## Section 2 — per-pass breakdown (64×64 / C=3 / K=10)\n");
    eprintln!(
        "Each pass is dispatched in isolation 1000× with submit+poll(Wait) per iteration.\n\
         Wall-clock per call therefore *includes* command-encoder + queue overhead.\n"
    );

    let grid: u32 = 64;
    let channels: u32 = 3;
    let cfg = cfg(grid, channels);
    let setup_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = setup_sim.activation().clone();
    let kernel_params = setup_sim.kernel_params().clone();

    // Build everything needed for each pass in isolation.
    let a_buf = upload_activation(ctx, &initial_a);
    let kernels = upload_kernels(ctx, &kernel_params);

    let convolve_pass = ConvolvePass::new(ctx);
    let pre_g = ConvolvePass::allocate_pre_g(ctx, grid, grid, kernels.count);

    let affinity_pass = AffinityGrowthPass::new(ctx);
    let h_vec: Vec<f32> = kernel_params.kernels.iter().map(|e| e.h).collect();
    let h_buf = upload_constant_weights(ctx, &GpuConstantWeights::from_slice(&h_vec));
    let u_buf = AffinityGrowthPass::allocate_u_out(ctx, grid, grid, channels);

    let gradient_pass = GradientPass::new(ctx);
    let grad_u_buf = GradientPass::allocate_grad_u(ctx, grid, grid, channels);
    let grad_a_sum_buf = GradientPass::allocate_grad_a_sum(ctx, grid, grid);

    let flow_pass = FlowPass::new(ctx);
    let flow_buf = FlowPass::allocate_f_out(ctx, grid, grid, channels);

    let reintegrate_pass = ReintegratePass::new(ctx);
    let a_out_buf = ReintegratePass::allocate_a(ctx, grid, grid, channels);

    let visualize_pass = VisualizePass::new(ctx, wgpu::TextureFormat::Rgba8Unorm, 8);
    let viz_globals = visualize_pass.upload_globals(ctx, grid, grid, channels);
    let viz_texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("bench viz target"),
        size: wgpu::Extent3d {
            width: grid * 8,
            height: grid * 8,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    let viz_view = viz_texture.create_view(&wgpu::TextureViewDescriptor::default());

    let globals = GpuGlobals::new(
        grid,
        grid,
        channels,
        kernels.count,
        kernels.max_side,
        cfg.border,
    )
    .with_paper_strict(cfg.paper_strict)
    .with_beta_a(cfg.beta_a)
    .with_n(cfg.n)
    .with_dd(cfg.dd)
    .with_sigma(cfg.sigma)
    .with_dt(cfg.dt);
    let globals_buf = ConvolvePass::upload_globals(ctx, &globals);

    let convolve_bg = convolve_pass.make_bind_group(ctx, &a_buf, &kernels, &pre_g, &globals_buf);
    let affinity_bg =
        affinity_pass.make_bind_group(ctx, &pre_g, &kernels, &h_buf, &u_buf, &globals_buf);
    let grad_u_bg = gradient_pass.make_bind_group_u(ctx, &u_buf, &grad_u_buf, &globals_buf);
    let grad_as_bg =
        gradient_pass.make_bind_group_a_sum(ctx, &a_buf, &grad_a_sum_buf, &globals_buf);
    let flow_bg = flow_pass.make_bind_group(
        ctx,
        &a_buf,
        &grad_u_buf,
        &grad_a_sum_buf,
        &flow_buf,
        &globals_buf,
    );
    let reintegrate_bg =
        reintegrate_pass.make_bind_group(ctx, &a_buf, &flow_buf, &a_out_buf, &globals_buf);
    let viz_bg = visualize_pass.make_bind_group(ctx, &a_buf, &viz_globals);

    let iters: u32 = 1000;
    let convolve_us = measure_pass(ctx, "convolve", iters, |enc| {
        convolve_pass.record(enc, &convolve_bg, grid, grid);
    });
    let affinity_us = measure_pass(ctx, "affinity", iters, |enc| {
        affinity_pass.record_constant(enc, &affinity_bg, grid, grid);
    });
    let grad_u_us = measure_pass(ctx, "grad_u", iters, |enc| {
        gradient_pass.record_u(enc, &grad_u_bg, grid, grid);
    });
    let grad_as_us = measure_pass(ctx, "grad_a_sum", iters, |enc| {
        gradient_pass.record_a_sum(enc, &grad_as_bg, grid, grid);
    });
    let flow_us = measure_pass(ctx, "flow", iters, |enc| {
        flow_pass.record(enc, &flow_bg, grid, grid);
    });
    let reintegrate_us = measure_pass(ctx, "reintegrate", iters, |enc| {
        reintegrate_pass.record(enc, &reintegrate_bg, grid, grid);
    });
    let visualize_us = measure_pass(ctx, "visualize", iters, |enc| {
        visualize_pass.record(enc, &viz_bg, &viz_view, None);
    });
    let pass_sum = convolve_us + affinity_us + grad_u_us + grad_as_us + flow_us + reintegrate_us;

    eprintln!(
        "| pass               | per-call (μs) | share of step |\n\
         |--------------------|--------------:|--------------:|"
    );
    let print = |name: &str, us: f64| {
        eprintln!(
            "| {name:<18} | {us:13.1} | {:13.1}% |",
            100.0 * us / pass_sum,
        );
    };
    print("convolve", convolve_us);
    print("affinity_growth", affinity_us);
    print("gradient_u", grad_u_us);
    print("gradient_a_sum", grad_as_us);
    print("flow", flow_us);
    print("reintegrate", reintegrate_us);
    eprintln!("| **step sum**       | **{pass_sum:.1}** | 100.0% |");
    eprintln!("| visualize (render) | {visualize_us:.1} | n/a (not in step) |\n");
}

// ────────────────────────────────────────────────────────────────────
// Section 3 — pipeline init + memory accounting
// ────────────────────────────────────────────────────────────────────

fn pipeline_buffer_bytes(grid: u32, channels: u32, max_side: u32) -> u64 {
    // 2 ping-pong A: 2 · C·H·W·4
    let a_buffers = 2 * u64::from(channels) * u64::from(grid) * u64::from(grid) * 4;
    // pre_g: H·W·K·4
    let pre_g = u64::from(grid) * u64::from(grid) * u64::from(NUM_KERNELS) * 4;
    // u: C·H·W·4
    let u = u64::from(channels) * u64::from(grid) * u64::from(grid) * 4;
    // grad_u: C·H·W·2·4
    let grad_u = u64::from(channels) * u64::from(grid) * u64::from(grid) * 2 * 4;
    // grad_a_sum: H·W·2·4
    let grad_a_sum = u64::from(grid) * u64::from(grid) * 2 * 4;
    // flow_field: C·H·W·2·4
    let flow_field = u64::from(channels) * u64::from(grid) * u64::from(grid) * 2 * 4;
    // kernels: K·max_side²·4
    let kernels = u64::from(NUM_KERNELS) * u64::from(max_side) * u64::from(max_side) * 4;
    // meta: K · 16
    let meta = u64::from(NUM_KERNELS) * 16;
    // h_weights: 45·4 = 180
    let h_weights = 180;
    // globals: 64
    let globals = 64;
    a_buffers + pre_g + u + grad_u + grad_a_sum + flow_field + kernels + meta + h_weights + globals
}

fn section_init_memory(ctx: &GpuContext, grids: &[u32], channels_list: &[u32]) {
    eprintln!("\n## Section 3 — `GpuStepPipeline::new()` init time + GPU memory\n");
    eprintln!(
        "| grid | C | init (ms) | A buf (2×) | pre_g | u | grad_u | grad_a_sum | flow | kernels | total (KB) |\n\
         |-----:|--:|----------:|-----------:|------:|--:|-------:|-----------:|-----:|--------:|-----------:|"
    );

    for &grid in grids {
        for &channels in channels_list {
            let cfg = cfg(grid, channels);
            let setup_sim = FlowLeniaSimulator::new(cfg, SEED);
            let initial_a = setup_sim.activation().clone();
            let kernel_params = setup_sim.kernel_params().clone();
            let started = Instant::now();
            let pipeline = GpuStepPipeline::new(ctx, &cfg, &kernel_params, &initial_a);
            let init_ms = started.elapsed().as_secs_f64() * 1000.0;
            // Drop the pipeline to release GPU memory before next case
            drop(pipeline);

            // Compute max_side for accounting
            let max_side = {
                let mut ms = 0_u32;
                for entry in &kernel_params.kernels {
                    let er = flow_lenia_core::effective_radius(kernel_params.r_global, entry.r);
                    ms = ms.max(2 * er + 1);
                }
                ms
            };

            let a_b = 2 * u64::from(channels) * u64::from(grid) * u64::from(grid) * 4;
            let pre_g = u64::from(grid) * u64::from(grid) * u64::from(NUM_KERNELS) * 4;
            let u_b = u64::from(channels) * u64::from(grid) * u64::from(grid) * 4;
            let grad_u = u64::from(channels) * u64::from(grid) * u64::from(grid) * 2 * 4;
            let grad_as = u64::from(grid) * u64::from(grid) * 2 * 4;
            let flow_b = u64::from(channels) * u64::from(grid) * u64::from(grid) * 2 * 4;
            let kbuf = u64::from(NUM_KERNELS) * u64::from(max_side) * u64::from(max_side) * 4;
            let total = pipeline_buffer_bytes(grid, channels, max_side);

            eprintln!(
                "| {grid:4} | {channels} | {init_ms:9.1} | {:>10} | {:>5} | {:>2} | {:>6} | {:>10} | {:>4} | {:>7} | {:>10} |",
                kb(a_b),
                kb(pre_g),
                kb(u_b),
                kb(grad_u),
                kb(grad_as),
                kb(flow_b),
                kb(kbuf),
                kb(total),
            );
        }
    }
}

fn kb(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{bytes} B")
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.2} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}

// ────────────────────────────────────────────────────────────────────
// Driver
// ────────────────────────────────────────────────────────────────────

// ────────────────────────────────────────────────────────────────────
// Section 4 — 500-step mass conservation (DESIGN.md §8 M2 criterion)
// ────────────────────────────────────────────────────────────────────

fn section_mass_conservation_500(ctx: &GpuContext) {
    eprintln!("\n## Section 4 — 500-step mass conservation (DESIGN.md §8)\n");
    eprintln!(
        "Same 8-case matrix (paper_strict × border × C) as M1.15 baseline,\n\
         32×32 grid, K=10, seed=42. Tolerance: 1e-3 (torus), 1e-2 (wall).\n"
    );
    eprintln!(
        "| paper_strict | border | C | max_rel | total ms | per-step ms |\n\
         |--------------|--------|--:|--------:|---------:|------------:|"
    );

    const N_STEPS: u32 = 500;
    for paper_strict in [false, true] {
        for border in [BorderMode::Torus, BorderMode::Wall] {
            for channels in [1_u32, 3_u32] {
                let cfg = FlowLeniaConfig {
                    grid_width: 32,
                    grid_height: 32,
                    channels,
                    dt: 0.2,
                    sigma: 0.65,
                    n: 2.0,
                    beta_a: 2.0,
                    dd: 5,
                    num_kernels: NUM_KERNELS,
                    paper_strict,
                    border,
                    mix_rule: MixRule::Stochastic,
                };
                let setup_sim = FlowLeniaSimulator::new(cfg, 42);
                let initial_a = setup_sim.activation().clone();
                let kernel_params = setup_sim.kernel_params().clone();
                let m0: f64 = initial_a.iter().map(|&v| f64::from(v)).sum();
                let mut pipeline = GpuStepPipeline::new(ctx, &cfg, &kernel_params, &initial_a);

                let started = Instant::now();
                let mut max_rel = 0.0_f64;
                for _ in 0..N_STEPS {
                    pipeline.step(ctx);
                    ctx.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();
                    let a = pipeline.readback_activation(ctx);
                    let m: f64 = a.iter().map(|&v| f64::from(v)).sum();
                    let rel = (m - m0).abs() / m0;
                    if rel > max_rel {
                        max_rel = rel;
                    }
                }
                let total_ms = started.elapsed().as_secs_f64() * 1000.0;
                let per_step_ms = total_ms / f64::from(N_STEPS);
                eprintln!(
                    "| {paper_strict:<12} | {border:?} | {channels} | {max_rel:.3e} | {total_ms:8.0} | {per_step_ms:11.2} |"
                );
            }
        }
    }
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("warn")).init();

    let ctx = headless_ctx();
    eprintln!(
        "Adapter: {} ({:?}) backend={:?}",
        ctx.adapter.get_info().name,
        ctx.adapter.get_info().device_type,
        ctx.adapter.get_info().backend
    );

    let grids = [32_u32, 64, 128, 256];
    let channels = [1_u32, 3];

    let _results = section_full_matrix(&ctx, &grids, &channels);
    section_per_pass(&ctx);
    section_init_memory(&ctx, &grids, &channels);
    section_mass_conservation_500(&ctx);

    eprintln!("\nbench complete.");
}
