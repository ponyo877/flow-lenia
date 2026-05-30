#![deny(warnings)]
//! M6.C-3 Stage 2 final 512 ms/step measurement.
//!
//! **M6.C-3-5 usage**: paired bisect for the reintegrate workgroup-
//! tiling effect (constant mode only, N=3 trials, ~30 s).
//!
//! **M6.C-3-6 usage**: 4-creature Localized AffinityMode for the final
//! Stage 2 verdict (60 FPS judgment C). Production target is `4×
//! creature × Localized × 512` so this is the only configuration that
//! decides whether Stage 2 is "achieved" / "essentially achieved" /
//! "40+ fps confirmed".
//!
//! ```text
//! cargo run --release --bin bench_512_reintegrate
//! ```

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    ActivationField, FlowLeniaSimulator,
};
use flow_lenia_gpu::{
    passes::{build_for_patches, CreaturePatch},
    pipeline::{AffinityMode, ConvolveMode},
    GpuContext, GpuStepPipeline,
};
use std::time::Instant;

const SEED: u64 = 1729;
const WARMUP: u32 = 20;
const STEPS: u32 = 50;
const TRIALS: usize = 3;

fn cfg() -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: 512,
        grid_height: 512,
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
    }
}

fn measure_constant(ctx: &GpuContext) -> f64 {
    let c = cfg();
    let cpu_init = FlowLeniaSimulator::new(c, SEED);
    let initial_a = cpu_init.activation().clone();
    let kernel_params = cpu_init.kernel_params().clone();
    let mut pipeline =
        GpuStepPipeline::new_with_mode(ctx, &c, &kernel_params, &initial_a, ConvolveMode::Fft);
    pipeline.run_steps(ctx, WARMUP);
    let started = Instant::now();
    pipeline.run_steps(ctx, STEPS);
    started.elapsed().as_secs_f64() / f64::from(STEPS) * 1000.0
}

/// Build the same 4-creature initial state + per-cell P map that
/// `bench_c2_configs.rs:build_four_creature_state` uses so the
/// numbers are directly comparable with BENCH.md §15-17.
fn build_four_creature_state(
    grid: u32,
    channels: u32,
    kernel_params: &flow_lenia_core::params::KernelParams,
) -> (ActivationField, Vec<f32>) {
    let n = grid as usize;
    let cc = channels as usize;
    let k = kernel_params.kernels.len();
    let h_base: Vec<f32> = kernel_params.kernels.iter().map(|e| e.h).collect();

    let blob: i32 = (grid / 4) as i32;
    let half = blob / 2;
    let q = grid as i32 / 4;
    let centers: [(i32, i32); 4] = [
        (q, q),
        (q, 3 * q),
        (3 * q, q),
        (3 * q, 3 * q),
    ];

    let mut initial_a = ActivationField::zeros((n, n, cc));
    for &(cy, cx) in &centers {
        for dy in -half..half {
            for dx in -half..half {
                let y = ((cy + dy).rem_euclid(grid as i32)) as usize;
                let x = ((cx + dx).rem_euclid(grid as i32)) as usize;
                let r2 = (dy * dy + dx * dx) as f32;
                let sigma2 = (half * half) as f32 * 0.5;
                let v = (-r2 / sigma2).exp() * 0.6;
                for ci in 0..cc {
                    initial_a[[y, x, ci]] = v;
                }
            }
        }
    }

    let patches: Vec<CreaturePatch> = centers
        .iter()
        .enumerate()
        .map(|(idx, &(cy, cx))| {
            let y0 = ((cy - half).rem_euclid(grid as i32)) as u32;
            let x0 = ((cx - half).rem_euclid(grid as i32)) as u32;
            let p_vector: Vec<f32> = h_base
                .iter()
                .enumerate()
                .map(|(ki, &h)| h * (1.0 + 0.05 * idx as f32 + 0.01 * ki as f32))
                .collect();
            CreaturePatch {
                bbox: (y0, x0, y0 + blob as u32, x0 + blob as u32),
                p_vector,
            }
        })
        .collect();
    let p_map = build_for_patches(grid, k as u32, &h_base, &patches);
    (initial_a, p_map)
}

fn measure_localized_four_creature(ctx: &GpuContext) -> f64 {
    let c = cfg();
    let cpu_init = FlowLeniaSimulator::new(c, SEED);
    let kernel_params = cpu_init.kernel_params().clone();
    let (initial_a, p_map) = build_four_creature_state(512, 3, &kernel_params);
    let mut pipeline = GpuStepPipeline::new_with_modes(
        ctx,
        &c,
        &kernel_params,
        &initial_a,
        ConvolveMode::Auto,
        AffinityMode::Localized,
        Some(&p_map),
    );
    pipeline.run_steps(ctx, WARMUP);
    let started = Instant::now();
    pipeline.run_steps(ctx, STEPS);
    started.elapsed().as_secs_f64() / f64::from(STEPS) * 1000.0
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);
    let info = ctx.adapter.get_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);

    eprintln!("\n=== 512 C=3 K=10 dd=5 ms/step (warmup {WARMUP}, measured {STEPS}, ×{TRIALS}) ===");
    let mut const_ms = Vec::with_capacity(TRIALS);
    let mut loc_ms = Vec::with_capacity(TRIALS);
    for t in 0..TRIALS {
        let cst = measure_constant(&ctx);
        let loc = measure_localized_four_creature(&ctx);
        eprintln!(
            "  trial {t}: constant={cst:.3} ms ({csps:.1} sps)  4-creature={loc:.3} ms ({lsps:.1} sps)",
            csps = 1000.0 / cst,
            lsps = 1000.0 / loc,
        );
        const_ms.push(cst);
        loc_ms.push(loc);
    }
    let cm = median(&mut const_ms);
    let lm = median(&mut loc_ms);
    eprintln!(
        "\n=== SUMMARY (median of {TRIALS}) ===\n  constant     : {cm:.3} ms ({csps:.1} sps)\n  4-creature loc: {lm:.3} ms ({lsps:.1} sps)\n  60 FPS budget : 16.667 ms (60.0 sps)\n  4-creature gap: {gap:+.3} ms ({gap_pct:+.1}%)",
        csps = 1000.0 / cm,
        lsps = 1000.0 / lm,
        gap = lm - 16.667,
        gap_pct = 100.0 * (lm - 16.667) / 16.667,
    );
}
