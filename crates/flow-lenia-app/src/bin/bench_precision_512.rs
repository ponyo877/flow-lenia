#![deny(warnings)]
//! M6.C-3-8 Step 1 precision report (案 Y) — **post-revert artifact**.
//!
//! This bench was built during the M6.C-3-8 Phase A trial (kernel_fft
//! + channel_spectra + k_spectra → f16 packed). The implementation was
//! reverted after the precision report (this exact bench) measured
//! the f16 contribution at **148× / 385× / 256× over the f32 Direct
//! baseline** for 1/5/10 step horizons at N=512, violating the user
//! STOP condition "256 Stage 1 数値が regression (絶対死守)".
//!
//! Run on HEAD post-revert, the bench measures the **Direct vs CPU**
//! baseline alone — the truth line that the f16 path would have had
//! to stay near. Future C-3-8 attempts (partial f16, f16 storage +
//! f32 accumulator, FFT-domain saturation guard, …) should keep this
//! bench around and re-add the FFT(f16) leg of the 3-way comparison
//! to confirm Δ rel = `FFT vs CPU − Direct vs CPU` stays within the
//! user-acceptable budget.
//!
//! ```text
//! cargo run --release --bin bench_precision_512
//! ```

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{pipeline::ConvolveMode, GpuContext, GpuStepPipeline};

const SEED: u64 = 1729;
const HORIZONS: &[u32] = &[1, 5, 10];

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

/// Measure Direct vs CPU at one horizon. With both sides f32 the rel
/// stays in the A.4.5 baseline regime (g256 10-step = 2.17e-4 at
/// HEAD 78db272, scaled up to g512 by chaos amplification). This is
/// the line any future f16 attempt has to stay close to.
fn measure_direct_vs_cpu(ctx: &GpuContext, horizon: u32) {
    let c = cfg();

    let mut cpu_sim = FlowLeniaSimulator::new(c, SEED);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    cpu_sim.step_many(horizon);
    let cpu_a = cpu_sim.activation().clone();

    let mut gpu_direct = GpuStepPipeline::new_with_mode(
        ctx,
        &c,
        &kernel_params,
        &initial_a,
        ConvolveMode::Direct,
    );
    gpu_direct.run_steps(ctx, horizon);
    let gpu_direct_a = gpu_direct.readback_activation(ctx);

    let (max_abs_d_cpu, max_rel_d_cpu) = compare(&cpu_a, &gpu_direct_a);

    let m_cpu: f64 = cpu_a.iter().map(|&v| f64::from(v)).sum();
    let m_direct: f64 = gpu_direct_a.iter().map(|&v| f64::from(v)).sum();
    let m_init: f64 = initial_a.iter().map(|&v| f64::from(v)).sum();

    eprintln!(
        "\n=== N=512 C=3 K=10 horizon={horizon} (post-revert baseline) ===\n  \
         Direct vs CPU     : max_abs = {max_abs_d_cpu:.3e}  max_rel = {max_rel_d_cpu:.3e}\n  \
         mass init   : {m_init:.6e}\n  \
         mass CPU    : {m_cpu:.6e}  drift = {dc:+.3e}\n  \
         mass Direct : {m_direct:.6e}  drift = {dd:+.3e}",
        dc = (m_cpu - m_init) / m_init,
        dd = (m_direct - m_init) / m_init,
    );
}

fn compare(
    a: &flow_lenia_core::ActivationField,
    b: &flow_lenia_core::ActivationField,
) -> (f32, f32) {
    assert_eq!(a.dim(), b.dim());
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (&va, &vb) in a.iter().zip(b.iter()) {
        let abs = (va - vb).abs();
        let mag = va.abs().max(vb.abs()).max(1e-6);
        let rel = abs / mag;
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
    }
    (max_abs, max_rel)
}

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let ctx = GpuContext::new_blocking(instance, None);
    let info = ctx.adapter.get_info();
    eprintln!("adapter: {} ({:?})", info.name, info.backend);
    eprintln!(
        "config: N=512 C=3 K=10 dd=5 BorderMode::Torus MixRule::Stochastic seed={SEED}"
    );
    eprintln!("Layer 3 tolerance reference (g=512, A.4.5 tiered): rel < ~5e-3");

    for &h in HORIZONS {
        measure_direct_vs_cpu(&ctx, h);
    }
}
