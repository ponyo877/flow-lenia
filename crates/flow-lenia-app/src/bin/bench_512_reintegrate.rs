#![deny(warnings)]
//! M6.C-3-5 focused 512 ms/step measurement at the **current** code
//! state. Use to bisect the reintegrate workgroup-tiling effect
//! against a known-good baseline:
//!
//! 1. checkout the C-3-4 revert commit (`341e4f6`) — measure baseline
//! 2. checkout the C-3-5 implementation — measure under tiled
//! 3. ratio = baseline_ms / tiled_ms
//!
//! Single config (N=512 C=3 FFT-mode Constant + Localized 4-creature),
//! N=3 trials, median ms/step. The bench_c2_configs full sweep
//! includes 5 configs and takes minutes; this one focuses only on the
//! Stage 2 target and finishes in ~30s.
//!
//! ```text
//! cargo run --release --bin bench_512_reintegrate
//! ```

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{pipeline::ConvolveMode, GpuContext, GpuStepPipeline};
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

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);
    let info = ctx.adapter.get_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);

    eprintln!("\n=== 512 C=3 K=10 dd=5 constant ms/step (warmup {WARMUP}, measured {STEPS}, ×{TRIALS}) ===");
    let mut const_ms = Vec::with_capacity(TRIALS);
    for t in 0..TRIALS {
        let cst = measure_constant(&ctx);
        eprintln!(
            "  trial {t}: constant={cst:.3} ms ({csps:.1} sps)",
            csps = 1000.0 / cst,
        );
        const_ms.push(cst);
    }
    let cm = median(&mut const_ms);
    eprintln!(
        "\n=== SUMMARY (median of {TRIALS}) ===\n  constant : {cm:.3} ms ({csps:.1} sps)",
        csps = 1000.0 / cm,
    );
}
