#![deny(warnings)]
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

/// `wasm-bindgen` entry point — invoked by the generated JS glue
/// when the WASM module is instantiated. Trunk wires this up via
/// the `<link data-trunk rel="rust" />` tag.
#[wasm_bindgen(start)]
pub fn run() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("console_log init failed");

    log::info!("flow-lenia-web booting (M3.2 — Hello WebGPU)");

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
