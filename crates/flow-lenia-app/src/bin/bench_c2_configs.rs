#![deny(warnings)]
//! M6.C-2-5 paired-run measurement: 5 configs for the M6.C-2
//! milestone + Stage 1 中間評価 input.
//!
//! ## What C-2 changed (and why the methodology works)
//!
//! M6.C-2 perf changes (C-2-1-a fused inverse FFT + transpose,
//! C-2-2 spectral-multiply 2-cell unroll) are baked into the FFT
//! path and are NOT toggleable in the shipped code. The **Direct
//! path is unchanged by C-2**, so it serves as a same-session
//! normalisation anchor:
//!
//! ```text
//! C-2 FFT speedup vs C-1  ≈  ratio_C2 / ratio_C1
//!   where ratio = direct_ms_per_step / fft_ms_per_step
//! ```
//!
//! Because both ratios divide by their *own session's* Direct
//! measurement, the cross-session thermal / cold-boot difference
//! between this run and the BENCH §14 C-1 baseline cancels out (the
//! Direct path code is identical). This is the cleanest C-2-vs-C-1
//! comparison achievable without a git-checkout paired build.
//!
//! BENCH §14 C-1 FFT-vs-Direct baselines (N=64, K=10, Torus, N=3
//! median, quiesced):
//! - C=1: ratio 8.206× (direct 13.31 → fft 1.62 ms/step)
//! - C=3: ratio 8.655× (direct 16.33 → fft 1.89 ms/step)
//!
//! ## Configs (M6.C-2-5 plan)
//!
//! | # | grid | C | mode | purpose |
//! |---|------|---|------|---------|
//! | 1 | 64   | 1 | fft (+ direct) | C-2 ratio vs C-1 §14 8.206× |
//! | 2 | 64   | 3 | fft (+ direct) | C-2 ratio vs C-1 §14 8.655× |
//! | 3 | 256  | 1 | fft (+ direct) | **first N=256 FFT data** |
//! | 4 | 256  | 3 | fft (+ direct) | **撤退ライン判定 (constant)** |
//! | 5 | 256  | 3 | localized 4-creature fft | **Stage 1 核心** |
//!
//! Config 5's localised overhead is reported relative to config 4
//! (same grid / channels, constant vs localized + ParameterFlowPass).
//!
//! ## Measurement protocol (CLAUDE.md §測定プロトコル準拠)
//!
//! - Paired interleave (D F D F … per config) to absorb thermal
//!   drift; N=3 trials, median reported.
//! - 20-step warmup per trial (shader cache + dispatch ladder),
//!   matching bench_step / bench_fft_vs_direct.
//! - N=256 uses 50 measured steps (per-step time is large enough
//!   that 50 steps is stable); N=64 uses 100.
//! - Quiesced state is the caller's responsibility (Ponyo877-san
//!   confirmed trunk serve / cargo / browser stopped for this run).
//! - Honest framing: printed ratios are observed-at-this-run.
//!
//! ## Gate (Stage 1 中間評価 prep)
//!
//! - 30 FPS 撤退ライン = 33.3 ms/step. Config 5 (N=256 C=3 4-creature)
//!   absolute ms/step is the go/no-go input.
//! - C-2 FFT speedup gate: ≥ 1.5× 順調 / < 1.5× 早期撤退ゲート検討 /
//!   < 1.0× 退行 即時停止 (the binary prints the verdict but the
//!   strategic call is Ponyo877-san's).
//!
//! ## Usage
//!
//! ```text
//! cargo run --release --bin bench_c2_configs
//! ```

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    state::ActivationField,
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{
    passes::{build_for_patches, CreaturePatch},
    pipeline::{AffinityMode, ConvolveMode},
    GpuContext, GpuStepPipeline,
};
use std::time::Instant;

const SEED: u64 = 1729;
const NUM_KERNELS: u32 = 10;
const N_TRIALS: usize = 3;
const WARMUP: u32 = 20;

// BENCH §14 C-1 FFT-vs-Direct baselines (N=64).
const C1_RATIO_N64_C1: f64 = 8.206;
const C1_RATIO_N64_C3: f64 = 8.655;

fn cfg_for(grid: u32, channels: u32) -> FlowLeniaConfig {
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

fn n_steps_for(grid: u32) -> u32 {
    if grid >= 256 {
        50
    } else {
        100
    }
}

fn median(samples: &mut [f64]) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    samples[samples.len() / 2]
}

/// Build a 4-creature localized initial state: corner-placed blobs
/// + a per-cell P map (background = constant h, patches = scaled h).
/// Returns `(initial_a, p_map_flat)`.
fn build_four_creature_state(
    grid: u32,
    channels: u32,
    kernel_params: &flow_lenia_core::params::KernelParams,
) -> (ActivationField, Vec<f32>) {
    let n = grid as usize;
    let c = channels as usize;
    let k = kernel_params.kernels.len();
    let h_base: Vec<f32> = kernel_params.kernels.iter().map(|e| e.h).collect();

    // Blob size scales with grid: N/4 wide blobs at the 4 quadrant
    // centers keep the four creatures disjoint on the torus.
    let blob: i32 = (grid / 4) as i32;
    let half = blob / 2;
    let q = grid as i32 / 4;
    let centers: [(i32, i32); 4] = [
        (q, q),
        (q, 3 * q),
        (3 * q, q),
        (3 * q, 3 * q),
    ];

    let mut initial_a = ActivationField::zeros((n, n, c));
    // Deterministic fill (no RNG dep): smooth radial bump per blob.
    for &(cy, cx) in &centers {
        for dy in -half..half {
            for dx in -half..half {
                let y = ((cy + dy).rem_euclid(grid as i32)) as usize;
                let x = ((cx + dx).rem_euclid(grid as i32)) as usize;
                let r2 = (dy * dy + dx * dx) as f32;
                let sigma2 = (half * half) as f32 * 0.5;
                let v = (-r2 / sigma2).exp() * 0.6;
                for ci in 0..c {
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

/// Measure ms/step for one (grid, channels, convolve_mode) in
/// Constant affinity mode.
fn measure_constant(ctx: &GpuContext, grid: u32, channels: u32, mode: ConvolveMode) -> f64 {
    let cfg = cfg_for(grid, channels);
    let cpu_init = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_init.activation().clone();
    let kernel_params = cpu_init.kernel_params().clone();
    let mut pipeline = GpuStepPipeline::new_with_mode(ctx, &cfg, &kernel_params, &initial_a, mode);
    let n_steps = n_steps_for(grid);
    pipeline.run_steps(ctx, WARMUP);
    let started = Instant::now();
    pipeline.run_steps(ctx, n_steps);
    started.elapsed().as_secs_f64() / f64::from(n_steps) * 1000.0
}

/// Measure ms/step for a localized 4-creature run (FFT mode).
fn measure_localized_four_creature(ctx: &GpuContext, grid: u32, channels: u32) -> f64 {
    let cfg = cfg_for(grid, channels);
    let cpu_init = FlowLeniaSimulator::new(cfg, SEED);
    let kernel_params = cpu_init.kernel_params().clone();
    let (initial_a, p_map) = build_four_creature_state(grid, channels, &kernel_params);
    let mut pipeline = GpuStepPipeline::new_with_modes(
        ctx,
        &cfg,
        &kernel_params,
        &initial_a,
        ConvolveMode::Auto,
        AffinityMode::Localized,
        Some(&p_map),
    );
    let n_steps = n_steps_for(grid);
    pipeline.run_steps(ctx, WARMUP);
    let started = Instant::now();
    pipeline.run_steps(ctx, n_steps);
    started.elapsed().as_secs_f64() / f64::from(n_steps) * 1000.0
}

struct RatioResult {
    direct_median: f64,
    fft_median: f64,
    ratio: f64,
}

/// Paired D F D F measurement for (grid, channels) in Constant mode.
fn measure_ratio(ctx: &GpuContext, grid: u32, channels: u32) -> RatioResult {
    let mut direct: Vec<f64> = Vec::with_capacity(N_TRIALS);
    let mut fft: Vec<f64> = Vec::with_capacity(N_TRIALS);
    eprintln!("\n=== N={grid} C={channels} paired D F (×{N_TRIALS}) ===");
    for trial in 0..N_TRIALS {
        let d = measure_constant(ctx, grid, channels, ConvolveMode::Direct);
        direct.push(d);
        let f = measure_constant(ctx, grid, channels, ConvolveMode::Fft);
        fft.push(f);
        eprintln!(
            "  trial {trial}: direct={d:.3} ms  fft={f:.3} ms  ratio={r:.3}×",
            r = d / f
        );
    }
    let direct_median = median(&mut direct);
    let fft_median = median(&mut fft);
    let ratio = direct_median / fft_median;
    eprintln!(
        "  median: direct={direct_median:.3} ms ({dsps:.1} sps)  \
         fft={fft_median:.3} ms ({fsps:.1} sps)  ratio={ratio:.3}×",
        dsps = 1000.0 / direct_median,
        fsps = 1000.0 / fft_median,
    );
    RatioResult {
        direct_median,
        fft_median,
        ratio,
    }
}

fn main() {
    eprintln!(
        "M6.C-2-5 bench_c2_configs\n\
         K={NUM_KERNELS}, Torus, warmup {WARMUP}, N={N_TRIALS} trials\n\
         N=64: 100 measured steps; N=256: 50 measured steps"
    );

    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);
    let info = ctx.adapter.get_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);

    // Configs 1-4: FFT-vs-Direct ratio + absolute.
    let c1 = measure_ratio(&ctx, 64, 1);
    let c2 = measure_ratio(&ctx, 64, 3);
    let c3 = measure_ratio(&ctx, 256, 1);
    let c4 = measure_ratio(&ctx, 256, 3);

    // Config 5: localized 4-creature, paired against config-4-style
    // constant FFT to expose the localized + ParameterFlowPass
    // overhead at the same grid / channels.
    eprintln!("\n=== N=256 C=3 localized 4-creature vs constant (×{N_TRIALS}) ===");
    let mut const_fft: Vec<f64> = Vec::with_capacity(N_TRIALS);
    let mut loc_fft: Vec<f64> = Vec::with_capacity(N_TRIALS);
    for trial in 0..N_TRIALS {
        let cst = measure_constant(&ctx, 256, 3, ConvolveMode::Fft);
        const_fft.push(cst);
        let loc = measure_localized_four_creature(&ctx, 256, 3);
        loc_fft.push(loc);
        eprintln!(
            "  trial {trial}: constant={cst:.3} ms  localized={loc:.3} ms  \
             overhead={ov:.3}×",
            ov = loc / cst
        );
    }
    let const_fft_median = median(&mut const_fft);
    let loc_fft_median = median(&mut loc_fft);
    let loc_overhead = loc_fft_median / const_fft_median;
    eprintln!(
        "  median: constant={const_fft_median:.3} ms ({csps:.1} sps)  \
         localized={loc_fft_median:.3} ms ({lsps:.1} sps)  overhead={loc_overhead:.3}×",
        csps = 1000.0 / const_fft_median,
        lsps = 1000.0 / loc_fft_median,
    );

    // ── M6.C-3-2 Stage 2: naive 512 FFT-only (Direct at 512 is
    //    ~930 ms/step = unusable, so we measure FFT absolute only).
    //    config 6 = N=512 C=3 constant, config 7 = N=512 C=3
    //    4-creature localized (Stage 2 核心). ──────────────────────
    eprintln!("\n=== N=512 C=3 FFT-only (naive, Stage 2) (×{N_TRIALS}) ===");
    let mut c512_const: Vec<f64> = Vec::with_capacity(N_TRIALS);
    let mut c512_loc: Vec<f64> = Vec::with_capacity(N_TRIALS);
    for trial in 0..N_TRIALS {
        let cst = measure_constant(&ctx, 512, 3, ConvolveMode::Fft);
        c512_const.push(cst);
        let loc = measure_localized_four_creature(&ctx, 512, 3);
        c512_loc.push(loc);
        eprintln!(
            "  trial {trial}: constant={cst:.3} ms ({csps:.1} sps)  \
             localized={loc:.3} ms ({lsps:.1} sps)",
            csps = 1000.0 / cst,
            lsps = 1000.0 / loc,
        );
    }
    let c512_const_median = median(&mut c512_const);
    let c512_loc_median = median(&mut c512_loc);

    // ── Summary ───────────────────────────────────────────────
    eprintln!("\n========== SUMMARY ==========");
    eprintln!(
        "config 1  N=64  C=1 fft : {:.3} ms ({:.1} sps)  [FFT/Direct {:.3}×]",
        c1.fft_median,
        1000.0 / c1.fft_median,
        c1.ratio
    );
    eprintln!(
        "config 2  N=64  C=3 fft : {:.3} ms ({:.1} sps)  [FFT/Direct {:.3}×]",
        c2.fft_median,
        1000.0 / c2.fft_median,
        c2.ratio
    );
    eprintln!(
        "config 3  N=256 C=1 fft : {:.3} ms ({:.1} sps)  [FFT/Direct {:.3}×, direct {:.3} ms]",
        c3.fft_median,
        1000.0 / c3.fft_median,
        c3.ratio,
        c3.direct_median,
    );
    eprintln!(
        "config 4  N=256 C=3 fft : {:.3} ms ({:.1} sps)  [FFT/Direct {:.3}×, direct {:.3} ms]",
        c4.fft_median,
        1000.0 / c4.fft_median,
        c4.ratio,
        c4.direct_median,
    );
    eprintln!(
        "config 5  N=256 C=3 4-creature localized : {:.3} ms ({:.1} sps)  \
         [localized/constant {:.3}×]",
        loc_fft_median,
        1000.0 / loc_fft_median,
        loc_overhead
    );
    eprintln!(
        "config 6  N=512 C=3 fft (constant)       : {:.3} ms ({:.1} sps)",
        c512_const_median,
        1000.0 / c512_const_median,
    );
    eprintln!(
        "config 7  N=512 C=3 4-creature localized : {:.3} ms ({:.1} sps)  ← Stage 2 核心",
        c512_loc_median,
        1000.0 / c512_loc_median,
    );

    // ── C-2 FFT speedup vs C-1 (Direct-anchored) ──────────────
    eprintln!("\n========== C-2 vs C-1 (Direct-anchored ratio-of-ratios) ==========");
    let c2_speedup_c1 = c1.ratio / C1_RATIO_N64_C1;
    let c2_speedup_c3 = c2.ratio / C1_RATIO_N64_C3;
    eprintln!(
        "N=64 C=1: current FFT/Direct {:.3}× ÷ C-1 §14 {C1_RATIO_N64_C1:.3}× \
         = C-2 FFT speedup {c2_speedup_c1:.3}×",
        c1.ratio
    );
    eprintln!(
        "N=64 C=3: current FFT/Direct {:.3}× ÷ C-1 §14 {C1_RATIO_N64_C3:.3}× \
         = C-2 FFT speedup {c2_speedup_c3:.3}×",
        c2.ratio
    );
    eprintln!(
        "  (N=256 has no C-1 FFT baseline — config 3/4 are first-ever \
         N=256 FFT data, absolute only)"
    );

    // ── Stage 1 撤退ライン verdict ─────────────────────────────
    eprintln!("\n========== Stage 1 撤退ライン (30 FPS = 33.3 ms/step) ==========");
    let retreat_ms = 33.3;
    let verdict = |ms: f64| {
        if ms <= 16.7 {
            "60 FPS clear"
        } else if ms <= retreat_ms {
            "30-60 FPS (撤退ライン上)"
        } else {
            "< 30 FPS (撤退ライン未達)"
        }
    };
    eprintln!(
        "config 4 (N=256 C=3 constant)      : {:.3} ms → {}",
        c4.fft_median,
        verdict(c4.fft_median)
    );
    eprintln!(
        "config 5 (N=256 C=3 4-creature)    : {:.3} ms → {}  ← Stage 1 核心",
        loc_fft_median,
        verdict(loc_fft_median)
    );

    // ── M6.C-3-2 Stage 2 中間評価 (naive 512、追加最適化前) ──────
    eprintln!("\n========== M6.C-3-2 Stage 2 中間評価 (naive 512) ==========");
    let sps_512_loc = 1000.0 / c512_loc_median;
    let stage2_verdict = if sps_512_loc >= 40.0 {
        "≥40 sps: subgroup + mixed-precision で 60 FPS 確実"
    } else if sps_512_loc >= 30.0 {
        "30-40 sps: 全 deferred 手法必要、続行"
    } else if sps_512_loc >= 20.0 {
        "20-30 sps: 1.85× 境界、慎重続行"
    } else {
        "<20 sps: mixed-radix FFT 実装に問題、要調査 (Phase 3 条件3)"
    };
    eprintln!(
        "config 6 (N=512 C=3 constant)      : {:.3} ms ({:.1} sps)",
        c512_const_median,
        1000.0 / c512_const_median
    );
    eprintln!(
        "config 7 (N=512 C=3 4-creature)    : {:.3} ms ({sps_512_loc:.1} sps) → {stage2_verdict}",
        c512_loc_median,
    );
    eprintln!(
        "  最終ゴール 512×512×4creature×60FPS (16.7 ms) まで残り {:.2}× 必要",
        c512_loc_median / 16.67
    );

    eprintln!("\nNote: strategic call (撤退 / 継続 / 縮小 / 目標再評価) は Ponyo877 さん責任。");
}
