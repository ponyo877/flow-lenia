#![deny(warnings)]
// The whole module body uses `winit::platform::web` and `web_sys`,
// which only build on `wasm32-unknown-unknown`. Gating at the crate
// root keeps `cargo clippy --workspace --all-targets` (native) green
// — when compiled for the host the crate is an empty rlib.
#![cfg(target_arch = "wasm32")]
//! Flow-Lenia browser entry point (M3.2 — Hello WebGPU).
//!
//! This crate builds the `cdylib` that `trunk` bundles into the
//! `dist/` directory. The current scope is **only** the plumbing
//! to land a wgpu surface on a `<canvas>` element and clear it to
//! blue every frame — i.e. the M2.1 milestone, repeated on the
//! browser side. No compute passes, no Flow-Lenia state.
//!
//! Boot sequence on `wasm_bindgen(start)`:
//!
//! 1. Install `console_error_panic_hook` so Rust panics show a
//!    proper JS stack trace in DevTools.
//! 2. Initialise `console_log` at `Info` so `log::info!` reaches the
//!    browser console.
//! 3. Build a `winit::EventLoop<AppEvent>` with a user-event channel,
//!    capture its `EventLoopProxy`, and `run_app` an [`App`] whose
//!    state begins as `None`.
//! 4. On `resumed`, attach a `Window` to `<canvas id="flow-lenia-canvas">`
//!    via `WindowAttributesExtWebSys::with_canvas` and `spawn_local`
//!    the async `wgpu` initialisation. Once the adapter + device +
//!    surface are ready, the proxy delivers an
//!    `AppEvent::GpuReady(...)` back to the event loop.
//! 5. `user_event(AppEvent::GpuReady)` installs the renderer state
//!    and requests the first redraw.
//! 6. `WindowEvent::RedrawRequested` clears to blue, presents, then
//!    requests the next frame.
//!
//! The same pattern will scale up in M3.3+ when we wire the M2.8
//! `GpuStepPipeline` behind this surface.

use std::sync::Arc;

use wasm_bindgen::prelude::*;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowId};

const CANVAS_ID: &str = "flow-lenia-canvas";
const CANVAS_W: u32 = 512;
const CANVAS_H: u32 = 512;

/// Custom event delivered from a `wasm_bindgen_futures::spawn_local`
/// task back to the winit event loop. The async wgpu initialisation
/// produces a [`Renderer`]; the event loop installs it on receipt.
enum AppEvent {
    GpuReady(Renderer),
}

/// All runtime state the redraw handler needs. Only constructed
/// once the async wgpu setup completes.
struct Renderer {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    device: wgpu::Device,
    queue: wgpu::Queue,
    adapter_info: String,
}

struct App {
    state: Option<Renderer>,
    proxy: EventLoopProxy<AppEvent>,
    /// Set to `true` once `resumed` has fired so the async init isn't
    /// re-spawned on duplicate resumes (some platforms re-emit).
    init_started: bool,
}

impl App {
    fn new(proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            state: None,
            proxy,
            init_started: false,
        }
    }
}

impl ApplicationHandler<AppEvent> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.init_started {
            return;
        }
        self.init_started = true;

        // Look up the host-page canvas. The HTML supplies a fixed-id
        // 512×512 element so the page layout is deterministic.
        let canvas = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id(CANVAS_ID))
            .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("flow-lenia-canvas element not found in document");

        // winit 0.30 web: without an explicit `inner_size`, the
        // `WindowAttributesExtWebSys::with_canvas` path resizes the
        // canvas's `width` / `height` DOM attributes to `2^25` and the
        // page layout collapses. The HTML side already supplies a
        // 512×512 element, so we restate the same size here.
        let attrs = Window::default_attributes()
            .with_title("Flow-Lenia (M3.2)")
            .with_inner_size(LogicalSize::new(CANVAS_W, CANVAS_H))
            .with_canvas(Some(canvas));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create winit window"),
        );

        let proxy = self.proxy.clone();
        let window_for_async = Arc::clone(&window);

        wasm_bindgen_futures::spawn_local(async move {
            let renderer = build_renderer(window_for_async).await;
            // `send_event` only fails when the event loop has been
            // dropped; on the web that means the user closed the tab
            // mid-init, so silently swallow.
            let _ = proxy.send_event(AppEvent::GpuReady(renderer));
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::GpuReady(renderer) => {
                log::info!(
                    "wgpu ready: {} surface_format={:?}",
                    renderer.adapter_info,
                    renderer.surface_config.format,
                );
                renderer.window.request_redraw();
                self.state = Some(renderer);
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                log::info!("close requested — exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                if size.width == 0 || size.height == 0 {
                    return;
                }
                state.surface_config.width = size.width;
                state.surface_config.height = size.height;
                state
                    .surface
                    .configure(&state.device, &state.surface_config);
            }
            WindowEvent::RedrawRequested => {
                render_blue_frame(state);
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

async fn build_renderer(window: Arc<Window>) -> Renderer {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let surface = instance
        .create_surface(Arc::clone(&window))
        .expect("failed to create wgpu surface");
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("no suitable wgpu adapter found");

    let info = adapter.get_info();
    let adapter_info = format!(
        "{} ({:?}) backend={:?}",
        info.name, info.device_type, info.backend
    );
    log::info!("wgpu adapter: {adapter_info}");

    let (device, queue) = adapter
        .request_device(&wgpu::DeviceDescriptor {
            label: Some("flow-lenia-web::Device"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                .using_resolution(adapter.limits()),
            memory_hints: wgpu::MemoryHints::default(),
            trace: wgpu::Trace::Off,
        })
        .await
        .expect("failed to request wgpu device");

    let caps = surface.get_capabilities(&adapter);
    let format = caps
        .formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
        .unwrap_or(caps.formats[0]);
    let size = window.inner_size();
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: size.width.max(1),
        height: size.height.max(1),
        present_mode: caps.present_modes[0],
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    };
    log::info!(
        "surface format: {format:?} size: {}x{}",
        surface_config.width,
        surface_config.height
    );
    surface.configure(&device, &surface_config);

    Renderer {
        window,
        surface,
        surface_config,
        device,
        queue,
        adapter_info,
    }
}

fn render_blue_frame(state: &mut Renderer) {
    let frame = match state.surface.get_current_texture() {
        Ok(f) => f,
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            state
                .surface
                .configure(&state.device, &state.surface_config);
            return;
        }
        Err(e) => {
            log::warn!("get_current_texture: {e:?}");
            return;
        }
    };
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = state
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M3.2 clear-blue encoder"),
        });
    {
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("M3.2 clear-blue pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.05,
                        g: 0.15,
                        b: 0.85,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
        });
    }
    state.queue.submit([encoder.finish()]);
    frame.present();
}

// ─────────────────────────────────────────────────────────────────────
// M3.3 — convolve sanity check
//
// Independently of the render-loop App, kick off an async task that
// runs the M2.3 `ConvolvePass` on a headless `GpuContext` and compares
// the output to CPU `convolve2d`. Result is printed to the browser
// console.
//
// "Headless" here means we re-use `GpuContext::new(instance, None)`
// (no `compatible_surface`) so the test is independent of the canvas /
// surface lifecycle managed by the App above.
// ─────────────────────────────────────────────────────────────────────

/// Drain a `wgpu::Buffer` to the CPU as `Vec<T>` via the M3.3 async
/// readback pattern (oneshot + browser event loop), since
/// `Device::poll(Wait)` deadlocks on the JS main thread.
async fn readback_async<T: bytemuck::Pod>(
    ctx: &flow_lenia_gpu::GpuContext,
    source: &wgpu::Buffer,
    element_count: usize,
) -> Vec<T> {
    let bytes = (element_count * std::mem::size_of::<T>()) as wgpu::BufferAddress;
    let staging = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("M3.3 web readback staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M3.3 web readback encoder"),
        });
    encoder.copy_buffer_to_buffer(source, 0, &staging, 0, bytes);
    ctx.queue.submit([encoder.finish()]);

    let (tx, rx) = futures_channel::oneshot::channel();
    staging
        .slice(..)
        .map_async(wgpu::MapMode::Read, move |result| {
            // Receiver might be dropped if the test future was abandoned.
            let _ = tx.send(result);
        });
    // No `device.poll(Wait)` — the browser event loop drives queue
    // progress while we `.await` the channel. WebGPU implementations
    // run the map callback once the GPU has produced the data.
    rx.await
        .expect("readback channel dropped")
        .expect("buffer map_async reported failure");

    let view = staging.slice(..).get_mapped_range();
    let out: Vec<T> = bytemuck::cast_slice(&view).to_vec();
    drop(view);
    staging.unmap();
    out
}

async fn run_convolve_test() {
    use flow_lenia_core::{
        config::BorderMode,
        convolve::convolve2d,
        kernel::compute_kernel,
        params::{KernelParams, SamplingSettings},
    };
    use flow_lenia_gpu::{
        activation_buffer::upload_activation, kernel_buffers::upload_kernels,
        passes::convolve::ConvolvePass, GpuContext, GpuGlobals,
    };
    use ndarray::{Array3, Axis};
    use rand::{Rng, SeedableRng};
    use rand_chacha::ChaCha8Rng;

    log::info!("[M3.3] starting convolve CPU vs WASM-WebGPU comparison …");

    // Headless wgpu context — same `GpuContext::new` we use natively,
    // just with no surface compatibility constraint.
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
    let ctx = GpuContext::new(instance, None).await;
    let info = ctx.adapter.get_info();
    log::info!(
        "[M3.3] headless adapter: {} ({:?}) backend={:?}",
        info.name,
        info.device_type,
        info.backend
    );

    // Test setup (mirrors M2.3 `convolve_pass_matches_cpu_reference`):
    // 32×32 torus, C = 3, K = 4, deterministic seed.
    let (h, w, c, num_kernels) = (32_usize, 32_usize, 3_usize, 4_u32);
    let mut rng = ChaCha8Rng::seed_from_u64(0xC0FF_EE42_u64);
    let a: Array3<f32> = Array3::from_shape_fn((h, w, c), |_| rng.gen_range(0.0_f32..1.0));
    let params = KernelParams::sample_random(
        &mut rng,
        SamplingSettings {
            num_kernels,
            num_channels: c as u32,
        },
    );

    // ── GPU side ──────────────────────────────────────────────────
    let pass = ConvolvePass::new(&ctx);
    let a_buf = upload_activation(&ctx, &a);
    let kernels = upload_kernels(&ctx, &params);
    let pre_g = ConvolvePass::allocate_pre_g(&ctx, h as u32, w as u32, kernels.count);
    let globals = GpuGlobals::new(
        h as u32,
        w as u32,
        c as u32,
        kernels.count,
        kernels.max_side,
        BorderMode::Torus,
    );
    let globals_buf = ConvolvePass::upload_globals(&ctx, &globals);
    let bind_group = pass.make_bind_group(&ctx, &a_buf, &kernels, &pre_g, &globals_buf);
    let mut enc = ctx
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M3.3 convolve dispatch encoder"),
        });
    pass.record(&mut enc, &bind_group, h as u32, w as u32);
    ctx.queue.submit([enc.finish()]);
    let gpu_flat: Vec<f32> = readback_async(&ctx, &pre_g, h * w * kernels.count as usize).await;

    // ── CPU side ──────────────────────────────────────────────────
    let cpu_kernels: Vec<ndarray::Array2<f32>> = params
        .kernels
        .iter()
        .map(|e| compute_kernel(params.r_global, e))
        .collect();
    let mut cpu_per_kernel: Vec<ndarray::Array2<f32>> = Vec::with_capacity(cpu_kernels.len());
    for (entry, k_arr) in params.kernels.iter().zip(cpu_kernels.iter()) {
        let src_c = entry.c0 as usize;
        let a_src = a.index_axis(Axis(2), src_c).to_owned();
        cpu_per_kernel.push(convolve2d(&a_src, k_arr, BorderMode::Torus));
    }

    // ── Compare in the M2.3 cell-major order: pre_g[y, x, ki] ─────
    let k_count = kernels.count as usize;
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for y in 0..h {
        for x in 0..w {
            for (ki, per_k) in cpu_per_kernel.iter().enumerate().take(k_count) {
                let gpu_v = gpu_flat[y * w * k_count + x * k_count + ki];
                let cpu_v = per_k[[y, x]];
                let abs_err = (gpu_v - cpu_v).abs();
                let rel_err = abs_err / cpu_v.abs().max(1e-6);
                if abs_err > max_abs {
                    max_abs = abs_err;
                }
                if rel_err > max_rel {
                    max_rel = rel_err;
                }
            }
        }
    }

    log::info!(
        "[M3.3] convolve test: max_abs = {:.3e}, max_rel = {:.3e} \
         (budget: rel < 1e-4 OR abs < 1e-5)",
        max_abs,
        max_rel
    );
    if max_rel < 1e-4 || max_abs < 1e-5 {
        log::info!("[M3.3] ✅ convolve PASS");
    } else {
        log::error!("[M3.3] ❌ convolve FAIL");
    }
}

/// `wasm-bindgen` entry point — invoked by the generated JS glue
/// when the WASM module is instantiated. Trunk wires this up via
/// the `<link data-trunk rel="rust" />` tag.
#[wasm_bindgen(start)]
pub fn run() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("console_log init failed");

    log::info!("flow-lenia-web booting (M3.3 — Hello WebGPU + convolve sanity)");

    // Spin the convolve sanity check off concurrently with the App's
    // own async wgpu init. The two GpuContexts are independent (the
    // M3.3 one is headless), so there's no contention.
    wasm_bindgen_futures::spawn_local(async {
        run_convolve_test().await;
    });

    let event_loop = EventLoop::<AppEvent>::with_user_event()
        .build()
        .expect("failed to create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let proxy = event_loop.create_proxy();
    let mut app = App::new(proxy);
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
