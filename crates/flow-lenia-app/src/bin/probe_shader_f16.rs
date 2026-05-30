#![deny(warnings)]
//! M6.C-3-4-a probe: does the M1 Metal adapter expose
//! `wgpu::Features::SHADER_F16` (= WebGPU `shader-f16`)?
//!
//! M6.C-3-4 (mixed-precision) depends on being able to declare
//! `enable f16;` in WGSL and use `f16` / `vec2<f16>` types in the
//! `spectral_multiply.wgsl` + `kernel_fft` precompute shaders. The
//! wgpu equivalent is the `SHADER_F16` device feature; without it the
//! Metal backend rejects the shader at compilation time.
//!
//! This probe just inspects the adapter's `Features` bitset and prints
//! whether SHADER_F16 is present. Run after `bench_512_breakdown` to
//! decide whether C-3-4 can proceed natively or needs a Direct-mode
//! fallback.
//!
//! ```text
//! cargo run --release --bin probe_shader_f16
//! ```

fn main() {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .expect("no adapter");
    let info = adapter.get_info();
    let feats = adapter.features();
    eprintln!("adapter: {} (backend {:?})", info.name, info.backend);
    eprintln!("SHADER_F16 supported: {}", feats.contains(wgpu::Features::SHADER_F16));
    eprintln!(
        "TIMESTAMP_QUERY supported: {}",
        feats.contains(wgpu::Features::TIMESTAMP_QUERY)
    );
    eprintln!(
        "TIMESTAMP_QUERY_INSIDE_ENCODERS supported: {}",
        feats.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS)
    );
    eprintln!(
        "SUBGROUP supported: {}",
        feats.contains(wgpu::Features::SUBGROUP)
    );
}
