#![deny(warnings)]
//! Native GPU binary for Flow-Lenia (M2.10 full implementation).
//!
//! Opens a 512×512 winit window backed by a wgpu sRGB surface and
//! runs the M2.8 `GpuStepPipeline` continuously, rendering each
//! frame via the M2.9 `VisualizePass`. Keyboard:
//!
//! ```text
//!   Space : pause / resume simulation
//!   r     : reset to fresh `(cfg, seed)` initial state
//!   q     : quit
//! ```
//!
//! Usage:
//!
//! ```text
//! cargo run --release --bin native_gpu -- [steps_per_frame=1] [seed=1729]
//! ```
//!
//! Defaults: `steps_per_frame = 1`, `seed = 1729` (the M1.14 / M2.9
//! visualisation reference seed). Logs `step / fps` every second via
//! `env_logger` (initialised at `info`).

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{GpuContext, GpuStepPipeline, VisualizePass};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{Key, NamedKey};
use winit::window::{Window, WindowAttributes, WindowId};

const WINDOW_W: u32 = 512;
const WINDOW_H: u32 = 512;
const GRID_W: u32 = 64;
const GRID_H: u32 = 64;
const UPSCALE: u32 = WINDOW_W / GRID_W; // = 8

fn default_cfg() -> FlowLeniaConfig {
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

/// All state that depends on a window existing.
struct AppState {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
    pipeline: GpuStepPipeline,
    visualize: VisualizePass,
    visualize_globals_buf: wgpu::Buffer,
    cfg: FlowLeniaConfig,
    seed: u64,
    running: bool,
    fps: FpsCounter,
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

struct App {
    state: Option<AppState>,
    steps_per_frame: u32,
    seed: u64,
}

impl App {
    fn new(steps_per_frame: u32, seed: u64) -> Self {
        Self {
            state: None,
            steps_per_frame,
            seed,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return; // Mobile platforms can re-fire `resumed`.
        }

        // 1. Window — fixed size, non-resizable (M4 will add resize support).
        let attrs = WindowAttributes::default()
            .with_title("Flow-Lenia (M2.10)")
            .with_inner_size(LogicalSize::new(WINDOW_W, WINDOW_H))
            .with_resizable(false);
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );

        // 2. Manual surface + adapter selection so we can pick an sRGB
        //    swapchain format (gamma correction comes free, no manual
        //    `pow(c, 1/2.2)` in the fragment shader).
        // wgpu 29: explicit display-handle policy. Native window
        // needs the display handle to pick the right Vulkan/Metal/DX
        // surface, so pull it from the env (winit/raw-window-handle).
        let instance =
            wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = instance
            .create_surface(window.clone())
            .expect("failed to create wgpu surface");
        let gpu = GpuContext::new_blocking(instance, Some(&surface));

        let caps = surface.get_capabilities(&gpu.adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(wgpu::TextureFormat::is_srgb)
            .unwrap_or(caps.formats[0]);
        log::info!("surface format: {format:?}");
        let size = window.inner_size();
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0], // Fifo by default
            desired_maximum_frame_latency: 2,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
        };
        surface.configure(&gpu.device, &surface_config);

        // 3. CPU-side initial state. The simulator does the central-
        //    patch seeding from `(cfg, seed)` for us; we reuse its
        //    activation + kernel_params to build the GPU pipeline.
        let cfg = default_cfg();
        let cpu_sim = FlowLeniaSimulator::new(cfg, self.seed);
        let initial_a = cpu_sim.activation().clone();
        let kernel_params = cpu_sim.kernel_params().clone();
        let pipeline = GpuStepPipeline::new(&gpu, &cfg, &kernel_params, &initial_a);

        // 4. VisualizePass — sized for the swapchain format with the
        //    fixed `UPSCALE` factor.
        let visualize = VisualizePass::new(&gpu, format, UPSCALE);
        let visualize_globals_buf = visualize.upload_globals(&gpu, GRID_H, GRID_W, cfg.channels);

        self.state = Some(AppState {
            window,
            surface,
            surface_config,
            gpu,
            pipeline,
            visualize,
            visualize_globals_buf,
            cfg,
            seed: self.seed,
            running: true,
            fps: FpsCounter::new(),
        });
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
                log::info!(
                    "close requested at step={} — exiting",
                    state.pipeline.step_count()
                );
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
                    log::info!("q pressed — exiting");
                    event_loop.exit();
                }
                _ => {}
            },
            WindowEvent::RedrawRequested => {
                if state.running {
                    for _ in 0..self.steps_per_frame {
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

/// Re-build the GPU pipeline from a fresh `(cfg, seed)` simulator.
/// Simpler than tracking individual buffer resets — pipeline
/// construction is fast (≈ tens of ms; M2.7 measured 10 ms for the
/// fixture generator across 8 cases).
fn reset_simulation(state: &mut AppState) {
    let cpu_sim = FlowLeniaSimulator::new(state.cfg, state.seed);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    state.pipeline = GpuStepPipeline::new(&state.gpu, &state.cfg, &kernel_params, &initial_a);
    state.running = true;
    log::info!("simulation reset (seed = {})", state.seed);
}

fn render_frame(state: &mut AppState) {
    // wgpu 29 replaced `Result<SurfaceTexture, SurfaceError>` with an
    // enum that distinguishes "got a usable texture" (Success /
    // Suboptimal — both carry one) from a handful of recoverable /
    // unrecoverable non-frame states. Treat Suboptimal as success
    // (render but reconfigure for next frame), Lost / Outdated as
    // "reconfigure and skip", Timeout / Occluded as "skip silently",
    // Validation as a warn-and-skip.
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

    // The bind group is rebuilt every frame because
    // `pipeline.current_activation_buffer()` alternates between the
    // two ping-pong A buffers — wgpu bind groups are immutable, so we
    // must build the one pointing at the buffer we actually want to
    // read this frame. Bind-group construction is ~µs on Apple M1.
    let bind_group = state.visualize.make_bind_group(
        &state.gpu,
        state.pipeline.current_activation_buffer(),
        &state.visualize_globals_buf,
    );

    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M2.10 frame encoder"),
        });
    // None: draw across the whole window — native_gpu has no UI panel
    // overlay (egui is web-only in M4).
    state.visualize.record(&mut encoder, &bind_group, &view, None);
    state.gpu.queue.submit([encoder.finish()]);
    frame.present();
}

fn main() {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let steps_per_frame: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1729);
    log::info!("steps_per_frame={steps_per_frame}  seed={seed}");

    let event_loop = EventLoop::new().expect("failed to create event loop");
    // `Poll` (not `Wait`) so we drive a continuous redraw cycle — the
    // present mode (Fifo / vsync) provides natural throttling.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(steps_per_frame, seed);
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
