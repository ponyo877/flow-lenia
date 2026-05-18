#![deny(warnings)]
//! Flow-Lenia GPU: `wgpu` compute pipeline implementation.
//!
//! Implements the WebGPU/wgpu side of the design specified in `DESIGN.md`
//! (Rev. 4.1), sections §3 (data layout) and §4 (shader pipeline). Built
//! incrementally across M2 (`DESIGN.md` §8). M2.1 lands the GPU context
//! plumbing only — no compute passes yet.

use std::future::Future;

pub mod activation_buffer;
pub mod config_border;
pub mod globals;
pub mod kernel_buffers;
pub mod passes;
pub mod readback;

pub use activation_buffer::{
    flatten_activation_channel_major, readback_activation, unflatten_activation_channel_major,
    upload_activation,
};
pub use config_border::BorderCode;
pub use globals::GpuGlobals;
pub use kernel_buffers::{
    readback_kernels, readback_meta, upload_kernels, GpuKernelBuffers, GpuKernelMeta,
};
pub use passes::{
    upload_constant_weights, upload_localized_weights, AffinityGrowthPass, ConvolvePass, FlowPass,
    GpuConstantWeights, GradientPass, ReintegratePass, MAX_KERNELS,
};
pub use readback::readback_buffer;

/// Owns the four core `wgpu` handles the rest of the crate (and the
/// downstream binary) needs.
///
/// Held as a plain struct because every M2 compute pass will borrow
/// `device` and `queue` immutably — putting them behind a richer
/// abstraction now would be premature.
pub struct GpuContext {
    /// The `Instance` is the entry point to the wgpu API and is reused
    /// for everything (`create_surface`, future debug tooling, etc.).
    pub instance: wgpu::Instance,
    /// Selected adapter. Backend (`Metal` / `Vulkan` / `DX12` / `GL`) is
    /// determined here; consult `adapter.get_info()` after construction.
    pub adapter: wgpu::Adapter,
    /// Logical device — the handle used to create buffers, textures,
    /// bind groups, and pipelines.
    pub device: wgpu::Device,
    /// Submission queue for command buffers (`queue.submit(...)`).
    pub queue: wgpu::Queue,
}

impl GpuContext {
    /// Async constructor.
    ///
    /// `compatible_surface` should be `Some(&surface)` when the caller
    /// owns a window-bound surface and needs an adapter that can render
    /// to it (the M2.1 `native_gpu` binary). Pass `None` for a
    /// headless / compute-only context (future M2.x off-screen tests).
    ///
    /// The instance is supplied by the caller so that `create_surface`
    /// can happen *before* adapter selection — the surface and adapter
    /// must share the same instance, and the `compatible_surface`
    /// argument needs the surface object.
    ///
    /// # Panics
    ///
    /// Panics if no suitable adapter or device can be obtained. Both
    /// failures indicate a fundamentally unusable environment (no GPU,
    /// driver mismatch, etc.) rather than user error, so a panic is the
    /// right escalation here — there is no caller-actionable recovery.
    pub async fn new(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Self {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface,
                force_fallback_adapter: false,
            })
            .await
            .expect("no suitable wgpu adapter found");

        let info = adapter.get_info();
        log::info!(
            "wgpu adapter: {} ({:?}) backend={:?}",
            info.name,
            info.device_type,
            info.backend
        );

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("flow-lenia-gpu::Device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .expect("failed to request wgpu device");

        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }

    /// Convenience: block on [`Self::new`] using `pollster`. Useful for
    /// non-async call sites (e.g. winit's synchronous
    /// `ApplicationHandler::resumed`).
    #[must_use]
    pub fn new_blocking(
        instance: wgpu::Instance,
        compatible_surface: Option<&wgpu::Surface<'_>>,
    ) -> Self {
        pollster::block_on(Self::new(instance, compatible_surface))
    }
}

/// Returned alongside [`GpuContext`] when building a windowed renderer
/// in one shot via [`GpuContext::with_surface`]. Kept separate so the
/// context itself stays surface-agnostic for future headless use.
pub struct SurfaceState<'window> {
    pub surface: wgpu::Surface<'window>,
    pub config: wgpu::SurfaceConfiguration,
}

impl GpuContext {
    /// One-shot helper: build a `(GpuContext, SurfaceState)` pair from
    /// a window-like surface target and the current physical size.
    ///
    /// The size arguments are clamped to at least 1 internally, since
    /// some platforms (e.g. minimised winit windows) report `(0, 0)`
    /// and a zero-sized surface configuration is rejected by wgpu.
    pub async fn with_surface<'w>(
        target: impl Into<wgpu::SurfaceTarget<'w>>,
        width: u32,
        height: u32,
    ) -> (Self, SurfaceState<'w>) {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance
            .create_surface(target)
            .expect("failed to create wgpu surface");
        let ctx = Self::new(instance, Some(&surface)).await;
        let config = surface
            .get_default_config(&ctx.adapter, width.max(1), height.max(1))
            .expect("surface has no default configuration on this adapter");
        surface.configure(&ctx.device, &config);
        (ctx, SurfaceState { surface, config })
    }
}

/// Re-exported `pollster::block_on` so downstream binaries do not need
/// to depend on `pollster` directly for the typical
/// `pollster::block_on(GpuContext::new(...))` pattern.
pub fn block_on<F: Future>(f: F) -> F::Output {
    pollster::block_on(f)
}
