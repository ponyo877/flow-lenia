#![deny(warnings)]
// The whole module body uses `winit::platform::web` and `web_sys`,
// which only build on `wasm32-unknown-unknown`. Gating at the crate
// root keeps `cargo clippy --workspace --all-targets` (native) green
// — when compiled for the host the crate is an empty rlib.
#![cfg(target_arch = "wasm32")]
//! Flow-Lenia browser entry point (M3.4 — full pipeline in Chrome).
//!
//! Boot path:
//!
//! 1. `wasm_bindgen(start)` → `run()`
//!    - install panic hook + console logger
//!    - build `EventLoop<AppEvent>` with a user-event channel
//!    - `event_loop.run_app(&mut App { state: None, ... })`
//! 2. `ApplicationHandler::resumed`
//!    - look up `<canvas id="flow-lenia-canvas">`, build a winit
//!      `Window` with `.with_canvas(canvas)` + the explicit
//!      `with_inner_size(512, 512)` workaround for the 2^25 trap
//!    - `spawn_local`s an async task that builds a wgpu instance +
//!      window-bound surface, requests an adapter compatible with
//!      the surface, constructs [`flow_lenia_gpu::GpuStepPipeline`]
//!      from the same `(cfg, seed)` the M2.10 native binary uses,
//!      constructs [`flow_lenia_gpu::VisualizePass`] + two pre-built
//!      bind groups (ping/pong over the pipeline's A buffers), and
//!      proxies `AppEvent::GpuReady(AppState)` back
//! 3. `ApplicationHandler::user_event(GpuReady)`
//!    - install `AppState`, request first redraw
//! 4. `WindowEvent::RedrawRequested`
//!    - if `running`, advance the pipeline by `steps_per_frame`
//!    - `visualize_pass.record(...)` reads
//!      `pipeline.a_buffer(pipeline.ping_index())` via the
//!      pre-built `visualize_bind_groups[ping]` — no per-frame
//!      bind-group rebuild
//!    - submit + present, FPS tick, request next redraw
//! 5. `WindowEvent::KeyboardInput`
//!    - Space toggles `running`
//!    - r/R rebuilds the pipeline from a fresh `FlowLeniaSimulator`
//!    - q/Q calls `event_loop.exit()` (may be ignored by the browser
//!      tab; that's expected — see DESIGN.md M3.4 notes)
//!
//! No per-step CPU readback. `device.poll(Wait)` is **never** invoked
//! on the web — the surface's `present()` and the browser's redraw
//! cycle drive queue progress instead.

use std::sync::Arc;

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{GpuContext, GpuStepPipeline, VisualizePass};
use wasm_bindgen::prelude::*;
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowId};

const CANVAS_ID: &str = "flow-lenia-canvas";
// Canvas CSS size in logical pixels. Hard-coded for M3 — M4 UI work
// will turn these into a resizable / grid-size-tunable value, at
// which point the `<= 1` fallback heuristic in
// `resolve_physical_canvas_size` also needs revisiting (it currently
// assumes "tiny reading = winit init race", which only holds while
// the displayed canvas is locked to these constants).
const CANVAS_W: u32 = 512;
const CANVAS_H: u32 = 512;
const GRID_W: u32 = 64;
const GRID_H: u32 = 64;
const SEED: u64 = 1729;

fn demo_cfg() -> FlowLeniaConfig {
    FlowLeniaConfig {
        grid_width: GRID_W,
        grid_height: GRID_H,
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

/// One full Flow-Lenia run state — everything `RedrawRequested` and
/// keyboard handlers need.
struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    pipeline: GpuStepPipeline,
    visualize: VisualizePass,
    visualize_globals_buf: wgpu::Buffer,
    /// Pre-built bind groups for the two ping-pong A buffers.
    /// Indexed by `pipeline.ping_index()` each frame.
    visualize_bind_groups: [wgpu::BindGroup; 2],
    running: bool,
    fps: FpsCounter,
    steps_per_frame: u32,
}

struct FpsCounter {
    frames: u32,
    last_report: Instant,
}

impl FpsCounter {
    fn new() -> Self {
        Self {
            frames: 0,
            last_report: Instant::now(),
        }
    }

    fn tick(&mut self, step_count: u64, running: bool) {
        self.frames += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed.as_secs() >= 1 {
            let fps = f64::from(self.frames) / elapsed.as_secs_f64();
            let pause = if running { "" } else { " [paused]" };
            log::info!(
                "step={step_count} fps={fps:.1}  ({} frames in {:.3}s){pause}",
                self.frames,
                elapsed.as_secs_f64()
            );
            self.frames = 0;
            self.last_report = Instant::now();
        }
    }
}

enum AppEvent {
    GpuReady(Box<AppState>),
}

struct App {
    state: Option<AppState>,
    proxy: EventLoopProxy<AppEvent>,
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

        let canvas = web_sys::window()
            .and_then(|w| w.document())
            .and_then(|d| d.get_element_by_id(CANVAS_ID))
            .and_then(|el| el.dyn_into::<web_sys::HtmlCanvasElement>().ok())
            .expect("flow-lenia-canvas element not found in document");

        // winit 0.30 web: pair `with_canvas` with an explicit
        // `with_inner_size` or the DOM `width`/`height` get reset to
        // 2^25 (M3.2 pitfall — kept for posterity here too).
        let attrs = Window::default_attributes()
            .with_title("Flow-Lenia (M3.4)")
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
            let state = build_app_state(window_for_async).await;
            // Boxed because `AppState` is fairly large; passing it
            // through the user-event channel as a Box keeps the
            // event enum compact (winit copies it on every dispatch).
            let _ = proxy.send_event(AppEvent::GpuReady(Box::new(state)));
        });
    }

    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppEvent) {
        match event {
            AppEvent::GpuReady(state) => {
                log::info!(
                    "flow-lenia-web ready: adapter={} surface_format={:?}",
                    state.gpu.adapter.get_info().name,
                    state.surface_config.format
                );
                state.window.request_redraw();
                self.state = Some(*state);
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
                log::info!("close requested at step={}", state.pipeline.step_count());
                event_loop.exit();
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key,
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => match logical_key.as_ref() {
                Key::Named(NamedKey::Space) => {
                    state.running = !state.running;
                    log::info!(
                        "{} at step={}",
                        if state.running { "resumed" } else { "paused" },
                        state.pipeline.step_count()
                    );
                }
                Key::Character("r") | Key::Character("R") => {
                    reset_simulation(state);
                }
                Key::Character("q") | Key::Character("Q") => {
                    // Closing the tab from WASM requires a user gesture
                    // the browser may not grant; we still call exit()
                    // and let the user / DESIGN.md M3.4 notes handle
                    // the rest.
                    log::info!("q pressed — requesting exit");
                    event_loop.exit();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                if state.running {
                    for _ in 0..state.steps_per_frame {
                        state.pipeline.step(&state.gpu);
                    }
                }
                render_frame(state);
                state.fps.tick(state.pipeline.step_count(), state.running);
                state.window.request_redraw();
            }
            _ => {}
        }
    }
}

/// Resolve the canvas's physical pixel size at this moment.
///
/// Why this exists: on winit/web `window.inner_size()` can briefly
/// return `(1, 1)` before the canvas CSS layout settles (caught during
/// M3.5 stability testing on a backgrounded tab — the surface then
/// sticks at 1×1, the visualize `upscale = surface_w / GRID_W` rounds
/// down to 0, and `VisualizePass::new`'s `assert!(upscale > 0)` fires).
///
/// We treat any reading where either dimension is `<= 1` as the
/// init-time race and fall back to `CANVAS_W/H × devicePixelRatio` —
/// the size the canvas will end up at once layout settles. The
/// `WindowEvent::Resized` handler still reconfigures the surface on
/// later genuine resizes, so a real (e.g. user-driven) tiny resize
/// would still be respected in subsequent frames.
///
/// TODO(M4): when UI introduces a resizable canvas or a grid-size
/// dropdown, the `<= 1` heuristic and the `CANVAS_W/H` constants both
/// become dynamic. Revisit this helper at that point — it currently
/// assumes the displayed canvas is locked to those compile-time sizes.
fn resolve_physical_canvas_size(window: &Window) -> (u32, u32) {
    let raw = window.inner_size();
    if raw.width <= 1 || raw.height <= 1 {
        let dpr = web_sys::window()
            .map(|w| w.device_pixel_ratio())
            .unwrap_or(1.0);
        let expected_w = ((f64::from(CANVAS_W) * dpr).round() as u32).max(1);
        let expected_h = ((f64::from(CANVAS_H) * dpr).round() as u32).max(1);
        log::warn!(
            "window.inner_size() = {}x{} (init race); falling back to dpr-based {}x{} (dpr={})",
            raw.width,
            raw.height,
            expected_w,
            expected_h,
            dpr
        );
        (expected_w, expected_h)
    } else {
        (raw.width, raw.height)
    }
}

async fn build_app_state(window: Arc<Window>) -> AppState {
    // ─── wgpu context bound to the canvas surface ─────────────────
    // wgpu 29: `InstanceDescriptor::default()` was removed in favour
    // of explicit display-handle policy. The browser canvas surface
    // doesn't need a raw display handle (the browser provides one
    // implicitly), so use the no-handle variant.
    let instance =
        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    let surface = instance
        .create_surface(Arc::clone(&window))
        .expect("failed to create wgpu surface");
    let gpu = GpuContext::new(instance, Some(&surface)).await;

    let caps = surface.get_capabilities(&gpu.adapter);
    // On Chrome WebGPU the available formats are typically `Bgra8Unorm`
    // / `Rgba8Unorm`, *not* their sRGB variants. The compositor still
    // does the sRGB encode at present time, so the result matches the
    // M2.10 native sRGB path visually (just on a different code path).
    let format = caps
        .formats
        .iter()
        .copied()
        .find(wgpu::TextureFormat::is_srgb)
        .unwrap_or(caps.formats[0]);
    let (physical_w, physical_h) = resolve_physical_canvas_size(&window);
    let surface_config = wgpu::SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format,
        width: physical_w,
        height: physical_h,
        present_mode: caps.present_modes[0],
        desired_maximum_frame_latency: 2,
        alpha_mode: caps.alpha_modes[0],
        view_formats: vec![],
    };
    surface.configure(&gpu.device, &surface_config);
    log::info!(
        "surface format: {format:?} size: {}x{}",
        surface_config.width,
        surface_config.height
    );

    // ─── CPU-side initial state (same recipe as M2.10) ─────────────
    let cfg = demo_cfg();
    let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    let pipeline = GpuStepPipeline::new(&gpu, &cfg, &kernel_params, &initial_a);

    // ─── Visualize pass ────────────────────────────────────────────
    // `upscale` is the surface-pixel-to-grid ratio. We derive it from
    // the same `resolve_physical_canvas_size` helper as the surface
    // itself, so the two are guaranteed to agree even when the helper
    // takes the dpr-based fallback path (a `surface_config.width / 64`
    // computation would otherwise round to 0 during the 1×1 init race
    // and panic in `VisualizePass::new`'s `assert!(upscale > 0)`).
    let upscale = (physical_w / GRID_W).max(1);
    log::info!("visualize upscale = {upscale} (physical canvas = {physical_w}x{physical_h})");
    let visualize = VisualizePass::new(&gpu, format, upscale);
    let visualize_globals_buf = visualize.upload_globals(&gpu, GRID_H, GRID_W, cfg.channels);
    // Pre-build a bind group for each of the two ping-pong A buffers.
    let visualize_bind_groups = [
        visualize.make_bind_group(&gpu, pipeline.a_buffer(0), &visualize_globals_buf),
        visualize.make_bind_group(&gpu, pipeline.a_buffer(1), &visualize_globals_buf),
    ];

    AppState {
        window,
        surface,
        surface_config,
        gpu,
        pipeline,
        visualize,
        visualize_globals_buf,
        visualize_bind_groups,
        running: true,
        fps: FpsCounter::new(),
        steps_per_frame: 1,
    }
}

fn reset_simulation(state: &mut AppState) {
    let cfg = demo_cfg();
    let cpu_sim = FlowLeniaSimulator::new(cfg, SEED);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    state.pipeline = GpuStepPipeline::new(&state.gpu, &cfg, &kernel_params, &initial_a);
    state.visualize_bind_groups = [
        state.visualize.make_bind_group(
            &state.gpu,
            state.pipeline.a_buffer(0),
            &state.visualize_globals_buf,
        ),
        state.visualize.make_bind_group(
            &state.gpu,
            state.pipeline.a_buffer(1),
            &state.visualize_globals_buf,
        ),
    ];
    state.running = true;
    log::info!("simulation reset (seed = {SEED})");
}

fn render_frame(state: &mut AppState) {
    // wgpu 29 replaced `Result<SurfaceTexture, SurfaceError>` with the
    // `CurrentSurfaceTexture` enum — see the native_gpu binary for the
    // full rationale comment. Suboptimal carries a usable texture too,
    // so we render with it and reconfigure for the next frame.
    let frame = match state.surface.get_current_texture() {
        wgpu::CurrentSurfaceTexture::Success(f) => f,
        wgpu::CurrentSurfaceTexture::Suboptimal(f) => {
            state
                .surface
                .configure(&state.gpu.device, &state.surface_config);
            f
        }
        wgpu::CurrentSurfaceTexture::Lost | wgpu::CurrentSurfaceTexture::Outdated => {
            state
                .surface
                .configure(&state.gpu.device, &state.surface_config);
            return;
        }
        wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => return,
        wgpu::CurrentSurfaceTexture::Validation => {
            log::warn!("get_current_texture: validation error");
            return;
        }
    };
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    let bg = &state.visualize_bind_groups[state.pipeline.ping_index()];
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M3.4 frame encoder"),
        });
    state.visualize.record(&mut encoder, bg, &view);
    state.gpu.queue.submit([encoder.finish()]);
    frame.present();
}

/// `wasm-bindgen` entry point — invoked by the generated JS glue
/// when the WASM module is instantiated. Trunk wires this up via
/// the `<link data-trunk rel="rust" />` tag.
#[wasm_bindgen(start)]
pub fn run() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("console_log init failed");

    log::info!("flow-lenia-web booting (M3.4 — full pipeline + visualize)");

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
