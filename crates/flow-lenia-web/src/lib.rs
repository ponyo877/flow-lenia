#![deny(warnings)]
// The whole module body uses `winit::platform::web` and `web_sys`,
// which only build on `wasm32-unknown-unknown`. Gating at the crate
// root keeps `cargo clippy --workspace --all-targets` (native) green
// — when compiled for the host the crate is an empty rlib.
#![cfg(target_arch = "wasm32")]
//! Flow-Lenia browser entry point (M4.1 — egui side panel over wgpu).
//!
//! Layout (logical pixels): the canvas is `CANVAS_W` × `CANVAS_H`. egui
//! claims a fixed `SIDE_PANEL_W`-wide column on the right via
//! `egui::SidePanel::right`, leaving the Flow-Lenia visualisation in
//! the remaining `CentralPanel` rect on the left.
//!
//! Render pipeline (per frame):
//!
//! 1. step the GPU pipeline `steps_per_frame` times (M3.4 logic).
//! 2. begin one wgpu encoder.
//! 3. `VisualizePass::record(..., Some((x, y, w, h)))` — Flow-Lenia
//!    draws into the CentralPanel sub-rect of the surface; the
//!    `LoadOp::Clear` inside `record` clears the full attachment first
//!    so the SidePanel side stays black until egui paints over it.
//! 4. egui-wgpu `Renderer::update_buffers` populates per-frame
//!    vertex / uniform buffers.
//! 5. a second render pass with `LoadOp::Load` runs egui's draw
//!    commands over the same surface texture (panel background +
//!    text + future controls).
//! 6. submit, present, request the next redraw.
//!
//! Event routing:
//!
//! - `egui_winit::State::on_window_event` is called first; if it
//!   reports `consumed`, the app's keyboard branch is skipped. This
//!   lets future text inputs / sliders absorb their key strokes
//!   without firing the Space/r/q shortcuts.
//! - `EventResponse::repaint` requests an extra redraw (egui needs
//!   one even when the simulation is paused).
//!
//! No per-step CPU readback. `device.poll(Wait)` is **never** invoked
//! on the web — the surface's `present()` and the browser's redraw
//! cycle drive queue progress instead.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Arc;

use flow_lenia_core::{
    config::{BorderMode, FlowLeniaConfig, MixRule},
    FlowLeniaSimulator,
};
use flow_lenia_gpu::{GpuContext, GpuStepPipeline, VisualizePass};
use wasm_bindgen::prelude::*;
use wasm_bindgen::JsCast;
use web_time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, EventLoopProxy};
use winit::keyboard::{Key, NamedKey};
use winit::platform::web::WindowAttributesExtWebSys;
use winit::window::{Window, WindowId};

// M4.5.1 — global state cell so both the winit `ApplicationHandler`
// callbacks (keyboard / window events / async readback completions)
// and the `requestAnimationFrame`-driven `tick()` function can reach
// the same `AppState` without threading it through `App`. WASM is
// single-threaded so `RefCell` is sufficient; every borrow scope is
// kept tight and never crosses an `await` / `spawn_local`.
thread_local! {
    static APP_STATE: RefCell<Option<AppState>> = const { RefCell::new(None) };
}

const CANVAS_ID: &str = "flow-lenia-canvas";
// Canvas CSS size in logical pixels. M4.1 widened to 768×512 so the
// 250-px SidePanel fits next to a 512×512 Flow-Lenia square. Both
// dimensions are still compile-time constants — the `<= 1` fallback
// heuristic in `resolve_physical_canvas_size` assumes they don't
// change at runtime (TODO(M4): revisit if/when UI adds a resize handle
// or a grid-size dropdown).
// Canvas CSS size in logical pixels. M4.1 widened to 768×512 so the
// 250-px SidePanel fits next to a 512×512 Flow-Lenia square. Both
// dimensions are still compile-time constants — the `<= 1` fallback
// heuristic in `resolve_physical_canvas_size` assumes they don't
// change at runtime (TODO(M4): revisit if/when UI adds a resize handle
// or a grid-size dropdown).
const CANVAS_W: u32 = 768;
const CANVAS_H: u32 = 512;
/// Width of the egui SidePanel on the right edge, in logical pixels.
/// The Flow-Lenia visualisation gets `CANVAS_W - SIDE_PANEL_W` logical
/// pixels of horizontal room (matches the `egui::SidePanel::right`
/// call below; keep them in sync).
const SIDE_PANEL_W: u32 = 250;
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

/// One full Flow-Lenia run state — everything the render loop and
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
    /// Physical-pixel viewport allocated to the Flow-Lenia render.
    /// `(CANVAS_W - SIDE_PANEL_W) × dpr`; visualize draws into this
    /// sub-rect and egui paints the complement.
    flow_lenia_viewport: (f32, f32, f32, f32),
    running: bool,
    fps: FpsCounter,
    steps_per_frame: u32,
    /// M4.3 stats — the SidePanel reads these every frame; they're
    /// updated lazily so the readback / FPS computation doesn't stall
    /// the render loop. `current_fps` ticks once per second via
    /// `FpsCounter::tick`'s side effect; `cached_mass` ticks every
    /// `MASS_READBACK_INTERVAL` frames via `trigger_mass_readback`.
    current_fps: f32,
    cached_mass: Vec<f32>,
    frames_until_mass_readback: u32,
    mass_readback_in_flight: bool,
    /// Held inside the state so the async mass-readback task can
    /// post `AppEvent::MassReadbackDone(...)` back into the event
    /// loop without borrowing `App`.
    proxy: EventLoopProxy<AppEvent>,
    /// M4.4 — live config that the SidePanel sliders write into.
    /// Lightweight changes (paper_strict / border / dt / dd) push
    /// straight into the uniform via `pipeline.update_globals`;
    /// num_kernels / `New Seed` rebuild the pipeline from scratch.
    cfg: FlowLeniaConfig,
    /// Current seed used to sample `KernelParams`. Mutated only by
    /// `New Seed` (the M4.2 `Reset` button keeps `SEED`).
    seed: u64,
    /// M4.5 — staged values for the controls that only commit on
    /// "Apply". Living on `AppState` so the ComboBox / Slider widgets
    /// can `&mut` a stable target — without this the ComboBox
    /// selection resets between frames because the egui closure
    /// always sees `state.cfg.*`, never the user's pending pick.
    pending_grid: u32,
    pending_channels: u32,
    pending_num_kernels: u32,
    /// Set by the Apply / New Seed buttons to defer the heavy
    /// pipeline rebuild outside the egui closure.
    pending_rebuild: Option<RebuildRequest>,
    // ─── egui ──────────────────────────────────────────────────────
    egui_ctx: egui::Context,
    egui_winit_state: egui_winit::State,
    egui_renderer: egui_wgpu::Renderer,
    /// M4.5.1 — TEMPORARY. Remove after stutter root-cause fix.
    frame_diag: FrameTimingDiag,
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

    /// Returns `Some(fps)` on the seconds where the per-second log
    /// fires; `None` otherwise. Callers can use the value to update
    /// the UI without computing FPS twice.
    fn tick(&mut self, step_count: u64, running: bool) -> Option<f32> {
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
            Some(fps as f32)
        } else {
            None
        }
    }
}

enum AppEvent {
    GpuReady(Box<AppState>),
    /// M4.3 — async readback of the per-channel activation sum.
    /// Posted from `trigger_mass_readback` after `map_async` resolves.
    MassReadbackDone(Vec<f32>),
}

/// M4.5.1 — TEMPORARY frame-timing histogram. Drives a per-frame
/// stutter diagnosis. Remove after the M4.5.1 root cause is fixed.
///
/// Captures two series in parallel:
/// - `intervals_ms` — wall-clock between consecutive `about_to_wait`
///   entries. This is the actual *displayed* frame cadence.
/// - `render_durations_ms` — time spent inside the
///   step + render_frame block (CPU side, excluding any vsync wait
///   that `Surface::present` may impose).
///
/// Every 300 samples we log percentile stats and clear, so a steady
/// 60fps run reports once every 5 seconds.
struct FrameTimingDiag {
    last_frame_start: Option<Instant>,
    intervals_ms: Vec<f64>,
    render_durations_ms: Vec<f64>,
}

impl FrameTimingDiag {
    fn new() -> Self {
        Self {
            last_frame_start: None,
            intervals_ms: Vec::with_capacity(400),
            render_durations_ms: Vec::with_capacity(400),
        }
    }

    fn tick(&mut self, now: Instant) {
        if let Some(prev) = self.last_frame_start {
            let elapsed_ms = now.duration_since(prev).as_secs_f64() * 1000.0;
            // M4.5.1.1 — drop intervals that are obviously
            // `document.visibilityState === "hidden"` pauses. RAF is
            // suspended while the tab is hidden, and the elapsed gap
            // when it resumes can easily run into tens of seconds,
            // which would dominate `mean` / `p99` and hide actual
            // stutter. Real frame stutters max out in the few-hundred-
            // ms range, so 500 ms is well above any real signal but
            // far below any plausible hide window.
            const HIDE_GAP_THRESHOLD_MS: f64 = 500.0;
            if elapsed_ms <= HIDE_GAP_THRESHOLD_MS {
                self.intervals_ms.push(elapsed_ms);
            }
        }
        self.last_frame_start = Some(now);
    }

    fn record_render(&mut self, dur_ms: f64) {
        self.render_durations_ms.push(dur_ms);
    }

    fn maybe_report(&mut self) {
        if self.intervals_ms.len() < 300 {
            return;
        }
        let intervals = std::mem::take(&mut self.intervals_ms);
        let renders = std::mem::take(&mut self.render_durations_ms);
        log_frame_stats("interval", &intervals);
        log_frame_stats("render", &renders);
    }
}

fn log_frame_stats(label: &str, samples: &[f64]) {
    if samples.is_empty() {
        return;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let min = sorted[0];
    let max = sorted[n - 1];
    let pick = |q: f64| sorted[((n as f64) * q).min((n - 1) as f64) as usize];
    let slow_30 = samples.iter().filter(|&&x| x > 30.0).count();
    let slow_50 = samples.iter().filter(|&&x| x > 50.0).count();
    let slow_100 = samples.iter().filter(|&&x| x > 100.0).count();
    log::info!(
        "[frame_diag/{label}] n={n} mean={mean:.2} p50={:.2} p95={:.2} p99={:.2} min={min:.2} max={max:.2} slow>30={slow_30} >50={slow_50} >100={slow_100}",
        pick(0.50),
        pick(0.95),
        pick(0.99),
    );
}

/// M4.4 — pipeline rebuild requests parked on the AppState so the
/// SidePanel closure can flag them without borrowing the heavy parts
/// of state. Processed once per frame, after the egui closure returns.
#[derive(Copy, Clone)]
enum RebuildRequest {
    /// "Apply" button after a num_kernels change. Keeps the current
    /// seed, resamples kernels at the new count.
    ApplyKernelCount,
    /// "New Seed" button. Advances the seed and resamples kernels at
    /// the current count.
    NewSeed,
    /// M4.5 "Apply Grid/Channels" button. Resamples kernels at the
    /// new grid shape *and* tears down the visualize pass so a fresh
    /// `upscale` factor is picked.
    ApplyGridAndChannels,
}

/// Frames between successive `trigger_mass_readback` calls. 30 ≈
/// 0.5 s at the ~60 Hz polling cycle, fast enough for the UI label
/// to feel live but slow enough that the readback queue never
/// backs up on its own.
const MASS_READBACK_INTERVAL: u32 = 30;

struct App {
    proxy: EventLoopProxy<AppEvent>,
    init_started: bool,
}

impl App {
    fn new(proxy: EventLoopProxy<AppEvent>) -> Self {
        Self {
            proxy,
            init_started: false,
        }
    }
}

impl ApplicationHandler<AppEvent> for App {
    /// M4.5.1 — render is now driven from JavaScript
    /// `requestAnimationFrame` via the exported [`tick`] function,
    /// pacing the simulation to the compositor instead of to the raw
    /// `ControlFlow::Poll` cadence (which on Chrome 148 WebGPU runs
    /// uncapped at ~130 fps and produces visible per-frame step
    /// jitter — see `docs/known-issues.md` #4).
    ///
    /// M4.5.1.2 — re-arm `ControlFlow::WaitUntil(now + 16 ms)` on every
    /// `about_to_wait` so the winit event loop ticks at ~60 Hz instead
    /// of the ~250–1000 Hz busy-loop that pure `Poll` produces on web.
    /// `WaitUntil` still wakes immediately on a DOM event (so discrete
    /// keydown / mouseup / `proxy.send_event` deliveries arrive without
    /// extra latency); the change just reclaims the CPU that the
    /// MessageChannel pump was burning between events. Without this
    /// throttle Ponyo877's M1 mini measured 90 % renderer CPU; with it
    /// the wake-up rate drops to roughly the display refresh.
    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        // winit 0.30 on wasm32 stores deadlines as `web_time::Instant`
        // (so `performance.now()` is the backing clock); using the
        // already-imported `web_time::Instant` here avoids the
        // `std::time::Instant` vs. `web_time::Instant` distinct-types
        // compile error.
        event_loop.set_control_flow(ControlFlow::WaitUntil(
            Instant::now() + std::time::Duration::from_millis(16),
        ));
    }

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
            .with_title("Flow-Lenia (M4.1)")
            .with_inner_size(LogicalSize::new(CANVAS_W, CANVAS_H))
            .with_canvas(Some(canvas));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create winit window"),
        );

        let proxy = self.proxy.clone();
        let proxy_for_state = self.proxy.clone();
        let window_for_async = Arc::clone(&window);

        wasm_bindgen_futures::spawn_local(async move {
            let state = build_app_state(window_for_async, proxy_for_state).await;
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
                // The RAF loop is already scheduled (see `start_raf_loop`
                // in `run()`); it has been ticking as a no-op waiting
                // for `APP_STATE` to populate. From the next animation
                // frame onwards it will start advancing the simulation.
                APP_STATE.with(|cell| *cell.borrow_mut() = Some(*state));
            }
            AppEvent::MassReadbackDone(mass) => {
                APP_STATE.with(|cell| {
                    // M4.5.1.1 — `try_borrow_mut` over `borrow_mut`. If
                    // a RAF tick is mid-flight, dropping this readback
                    // is acceptable (next 30-frame cycle will trigger
                    // another one); a panic here would tear down the
                    // whole event loop.
                    let Ok(mut binding) = cell.try_borrow_mut() else {
                        log::warn!("user_event(MassReadbackDone): APP_STATE busy, dropping result");
                        return;
                    };
                    if let Some(state) = binding.as_mut() {
                        state.cached_mass = mass;
                        state.mass_readback_in_flight = false;
                    }
                });
            }
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        APP_STATE.with(|cell| {
            // M4.5.1.1 — see the same comment on tick() / user_event.
            // Losing a single keystroke is unfortunate but the user can
            // press the key again; a panic would be permanent.
            let Ok(mut binding) = cell.try_borrow_mut() else {
                log::warn!("window_event: APP_STATE busy, dropping event");
                return;
            };
            let Some(state) = binding.as_mut() else {
                return;
            };

            // Hand every WindowEvent to egui first so future text inputs
            // / sliders absorb their key strokes before our Space/r/q
            // shortcuts fire. `EventResponse::consumed` says "this event
            // was for me, don't double-handle it".
            let egui_response = state
                .egui_winit_state
                .on_window_event(&state.window, &event);

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
                } if !egui_response.consumed => match logical_key.as_ref() {
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
                // The M4.1 winit-0.30.13 + wgpu-29 regression that made
                // `RedrawRequested` undeliverable is irrelevant under
                // the M4.5.1 RAF model (render is no longer requested
                // through winit). Left here so that if upstream does
                // restore delivery we still match the arm cleanly.
                WindowEvent::RedrawRequested => {}
                _ => {}
            }
        });
    }
}

/// M4.5.1 — exported tick called from `requestAnimationFrame` in the
/// host page (scheduled by [`start_raf_loop`]). This is now the *only*
/// place that steps the simulation and renders, so the cadence matches
/// what the compositor will actually display.
///
/// No-op until `APP_STATE` is populated by `AppEvent::GpuReady`. RAF is
/// kicked off as soon as `run()` returns control to JS, so the first
/// dozen or so animation frames are no-ops while WASM finishes async
/// GPU init.
///
/// M4.5.1.1 — uses `try_borrow_mut` rather than `borrow_mut`. WASM is
/// single-threaded so a conflicting borrow shouldn't happen in normal
/// flow, but this is the one entry point that *can* fire mid-event-
/// dispatch (a `device.poll` inside the previous frame can synchronously
/// fire a `map_async` callback that calls a `proxy.send_event` whose
/// `user_event` handler is then drained on the very next microtask).
/// Skipping a single frame is cheap (next RAF is 16 ms away) and far
/// better than the alternative of a `RefCell` double-borrow panic.
#[wasm_bindgen]
pub fn tick() {
    APP_STATE.with(|cell| {
        let Ok(mut binding) = cell.try_borrow_mut() else {
            // Another path is already inside APP_STATE — skip this frame
            // and let RAF re-fire. This is rare enough that logging on
            // every miss would be noise; the `frame_diag/interval` max
            // will surface it if it ever becomes frequent.
            return;
        };
        let Some(state) = binding.as_mut() else {
            return;
        };

        let frame_start = Instant::now();
        // M4.5.1 — frame-timing diag (interval is the gap between two
        // RAF-driven tick() entries, i.e. exactly what the user sees).
        state.frame_diag.tick(frame_start);

        // M4.4 — pipeline rebuild requests parked by the SidePanel
        // closure are picked up here, before stepping, so the
        // "step → render" pair always sees a consistent pipeline.
        if let Some(req) = state.pending_rebuild.take() {
            match req {
                RebuildRequest::ApplyKernelCount => {
                    log::info!(
                        "rebuild: apply num_kernels = {} (seed = {})",
                        state.cfg.num_kernels,
                        state.seed
                    );
                }
                RebuildRequest::NewSeed => {
                    state.seed = state.seed.wrapping_add(1);
                    log::info!("rebuild: new seed = {}", state.seed);
                }
                RebuildRequest::ApplyGridAndChannels => {
                    log::info!(
                        "rebuild: grid = {}x{} channels = {} (seed = {})",
                        state.cfg.grid_width,
                        state.cfg.grid_height,
                        state.cfg.channels,
                        state.seed
                    );
                }
            }
            rebuild_pipeline(state);
        }
        if state.running {
            for _ in 0..state.steps_per_frame {
                state.pipeline.step(&state.gpu);
            }
        }
        render_frame(state);

        let render_end = Instant::now();
        let render_ms =
            render_end.duration_since(frame_start).as_secs_f64() * 1000.0;
        state.frame_diag.record_render(render_ms);
        state.frame_diag.maybe_report();

        if let Some(fps) = state.fps.tick(state.pipeline.step_count(), state.running) {
            state.current_fps = fps;
        }
        if state.running {
            state.frames_until_mass_readback =
                state.frames_until_mass_readback.saturating_sub(1);
            if state.frames_until_mass_readback == 0 && !state.mass_readback_in_flight {
                trigger_mass_readback(state);
                state.frames_until_mass_readback = MASS_READBACK_INTERVAL;
            }
        }
        // wgpu 29 (web) needs an explicit poll for `map_async` callbacks
        // to fire — see `docs/known-issues.md` #3.
        let _ = state.gpu.device.poll(wgpu::PollType::Poll);
    });
}

/// M4.5.1 — install the RAF self-rescheduling closure. Uses the canonical
/// `Rc<RefCell<Option<Closure>>>` pattern so the same closure handle can
/// be re-passed to `requestAnimationFrame` each frame (without leaking
/// a fresh `Closure` per call).
///
/// Calling this once during `run()` is enough; the closure keeps itself
/// scheduled forever and only stops if `tick()` panics, which the
/// `console_error_panic_hook` will then surface in the JS console.
fn start_raf_loop() {
    let raf_holder: Rc<RefCell<Option<Closure<dyn FnMut()>>>> =
        Rc::new(RefCell::new(None));
    let raf_holder_clone = raf_holder.clone();
    *raf_holder.borrow_mut() = Some(Closure::new(move || {
        tick();
        // Re-schedule ourselves for the next compositor frame. The
        // borrow is released before we hand the Function back to
        // `requestAnimationFrame`, so the JS engine is free to call us
        // again on the next frame without any re-entrancy concerns.
        let win = web_sys::window().expect("no window");
        let cb = raf_holder_clone
            .borrow()
            .as_ref()
            .expect("raf closure missing")
            .as_ref()
            .unchecked_ref::<js_sys::Function>()
            .clone();
        win.request_animation_frame(&cb)
            .expect("requestAnimationFrame failed");
    }));
    let win = web_sys::window().expect("no window");
    let cb = raf_holder
        .borrow()
        .as_ref()
        .expect("raf closure missing")
        .as_ref()
        .unchecked_ref::<js_sys::Function>()
        .clone();
    win.request_animation_frame(&cb)
        .expect("initial requestAnimationFrame failed");
    // Intentionally leak the holder: the closure must outlive `run()`
    // (which never returns on web because winit suspends via throw).
    std::mem::forget(raf_holder);
}

/// M4.5 — pick the integer `upscale` and the matching square viewport
/// that fits the current grid into the CentralPanel rect.
///
/// Returns `(upscale, viewport_side)` in physical pixels. The viewport
/// is `(0, 0, viewport_side, viewport_side)` (square, top-left of the
/// surface) — egui paints the SidePanel and any leftover space.
fn compute_visualize_layout(grid_w: u32, grid_h: u32) -> (u32, u32) {
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0);
    let central_logical_w = CANVAS_W - SIDE_PANEL_W;
    let central_physical_w = ((f64::from(central_logical_w) * dpr).round() as u32).max(grid_w);
    let central_physical_h = ((f64::from(CANVAS_H) * dpr).round() as u32).max(grid_h);
    let upscale = (central_physical_w.min(central_physical_h) / grid_w.max(grid_h)).max(1);
    let viewport_side = upscale * grid_w.max(grid_h);
    (upscale, viewport_side)
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

async fn build_app_state(window: Arc<Window>, proxy: EventLoopProxy<AppEvent>) -> AppState {
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
        // `COPY_SRC` is needed so the M4.2 screenshot button can
        // capture the surface via `canvas.toBlob` — without it the
        // browser hands back a black frame (the swap chain texture
        // can't be sampled).
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
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
    // M4.1: Flow-Lenia shares the canvas with egui's SidePanel. The
    // `compute_visualize_layout` helper picks the integer upscale +
    // square viewport for the current grid; M4.5 rebuild paths reuse
    // the same helper after a grid resize.
    let dpr = web_sys::window()
        .map(|w| w.device_pixel_ratio())
        .unwrap_or(1.0);
    let (upscale, viewport_side) =
        compute_visualize_layout(cfg.grid_width, cfg.grid_height);
    let flow_lenia_viewport = (0.0, 0.0, viewport_side as f32, viewport_side as f32);
    log::info!(
        "visualize upscale = {upscale} (grid {}x{}, viewport side = {viewport_side})",
        cfg.grid_width,
        cfg.grid_height,
    );
    let visualize = VisualizePass::new(&gpu, format, upscale);
    let visualize_globals_buf =
        visualize.upload_globals(&gpu, cfg.grid_height, cfg.grid_width, cfg.channels);
    // Pre-build a bind group for each of the two ping-pong A buffers.
    let visualize_bind_groups = [
        visualize.make_bind_group(&gpu, pipeline.a_buffer(0), &visualize_globals_buf),
        visualize.make_bind_group(&gpu, pipeline.a_buffer(1), &visualize_globals_buf),
    ];

    // ─── egui ──────────────────────────────────────────────────────
    // `Context` is cheap to clone (Arc inside) — egui-winit needs its
    // own handle. The renderer lives on the wgpu side and shares the
    // surface format so its draws blend correctly with the visualize
    // output already on the surface texture.
    let egui_ctx = egui::Context::default();
    let egui_winit_state = egui_winit::State::new(
        egui_ctx.clone(),
        egui::ViewportId::ROOT,
        &*window,
        Some(dpr as f32),
        None,
        Some(8192),
    );
    let egui_renderer =
        egui_wgpu::Renderer::new(&gpu.device, format, egui_wgpu::RendererOptions::default());

    AppState {
        window,
        surface,
        surface_config,
        gpu,
        pipeline,
        visualize,
        visualize_globals_buf,
        visualize_bind_groups,
        flow_lenia_viewport,
        running: true,
        fps: FpsCounter::new(),
        steps_per_frame: 1,
        current_fps: 0.0,
        cached_mass: vec![0.0; cfg.channels as usize],
        // First readback fires on the very next frame so the panel
        // shows a real value almost immediately after init.
        frames_until_mass_readback: 1,
        mass_readback_in_flight: false,
        proxy,
        cfg,
        seed: SEED,
        pending_grid: cfg.grid_width,
        pending_channels: cfg.channels,
        pending_num_kernels: cfg.num_kernels,
        pending_rebuild: None,
        egui_ctx,
        egui_winit_state,
        egui_renderer,
        frame_diag: FrameTimingDiag::new(),
    }
}

fn reset_simulation(state: &mut AppState) {
    // The keyboard Reset / `r` shortcut returns to the original
    // SEED + cfg defaults; M4.4 "New Seed" advances `state.seed`
    // and reuses the same rebuild path through `pending_rebuild`.
    state.seed = SEED;
    state.cfg = demo_cfg();
    state.pending_grid = state.cfg.grid_width;
    state.pending_channels = state.cfg.channels;
    state.pending_num_kernels = state.cfg.num_kernels;
    rebuild_pipeline(state);
    log::info!("simulation reset (seed = {})", state.seed);
}

/// M4.4 — rebuild the GPU pipeline against `state.cfg` and
/// `state.seed`. Shared by Reset, "Apply Kernels", "New Seed", and
/// M4.5 "Apply Grid/Channels". `step_count` is reset to 0 as a
/// side-effect of constructing a fresh `GpuStepPipeline`.
///
/// VisualizePass is *also* recreated here so a grid-size change picks
/// up a fresh `upscale` (M4.5). It's cheap — the WGSL is cached by
/// the wgpu device and the bind-group layout is a one-shot.
fn rebuild_pipeline(state: &mut AppState) {
    let cpu_sim = FlowLeniaSimulator::new(state.cfg, state.seed);
    let initial_a = cpu_sim.activation().clone();
    let kernel_params = cpu_sim.kernel_params().clone();
    state.pipeline = GpuStepPipeline::new(&state.gpu, &state.cfg, &kernel_params, &initial_a);

    let (upscale, viewport_side) =
        compute_visualize_layout(state.cfg.grid_width, state.cfg.grid_height);
    state.flow_lenia_viewport = (0.0, 0.0, viewport_side as f32, viewport_side as f32);
    state.visualize = VisualizePass::new(&state.gpu, state.surface_config.format, upscale);
    state.visualize_globals_buf = state.visualize.upload_globals(
        &state.gpu,
        state.cfg.grid_height,
        state.cfg.grid_width,
        state.cfg.channels,
    );
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
    state.cached_mass = vec![0.0; state.cfg.channels as usize];
    state.running = true;
    state.frames_until_mass_readback = 1;
}

/// Copy the current activation buffer into a staging buffer, then
/// post `AppEvent::MassReadbackDone(per_channel_sum)` once the
/// `map_async` callback resolves. Cheap enough at our default 64×64×3
/// = 49 KB readback that the frame budget is unaffected — see the
/// M4.3 FPS measurements in the commit message.
///
/// Marks `state.mass_readback_in_flight = true`; the user_event
/// handler clears the flag when it installs the new value.
fn trigger_mass_readback(state: &mut AppState) {
    state.mass_readback_in_flight = true;
    let channels = state.cfg.channels as usize;
    let plane = (state.cfg.grid_width as usize) * (state.cfg.grid_height as usize);
    let element_count = plane * channels;
    let bytes = (element_count * std::mem::size_of::<f32>()) as wgpu::BufferAddress;

    let device = state.gpu.device.clone();
    let queue = state.gpu.queue.clone();
    let activation_buf = state.pipeline.a_buffer(state.pipeline.ping_index()).clone();
    let proxy = state.proxy.clone();

    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("mass readback staging"),
        size: bytes,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mass readback encoder"),
    });
    encoder.copy_buffer_to_buffer(&activation_buf, 0, &staging, 0, bytes);
    queue.submit([encoder.finish()]);

    wasm_bindgen_futures::spawn_local(async move {
        // Keep the device/queue/buffer handles alive across the await.
        let _hold_activation = activation_buf;
        let _hold_queue = queue;
        let _hold_device = device;
        let (sender, receiver) = futures_channel::oneshot::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        match receiver.await {
            Ok(Ok(())) => {}
            other => {
                log::warn!("mass map_async failed: {other:?}");
                let _ = proxy.send_event(AppEvent::MassReadbackDone(vec![0.0; channels]));
                return;
            }
        }
        let mass: Vec<f32> = {
            let mapped = staging.slice(..).get_mapped_range();
            let activation: &[f32] = bytemuck::cast_slice(&mapped);
            // GPU layout is `[C][H * W]` (M2.2 plan A); take per-channel
            // sums by slicing into plane-sized chunks.
            let mut out = Vec::with_capacity(channels);
            for c in 0..channels {
                let sum: f32 = activation[c * plane..(c + 1) * plane].iter().sum();
                out.push(sum);
            }
            out
        };
        staging.unmap();
        let _ = proxy.send_event(AppEvent::MassReadbackDone(mass));
    });
}

/// 256-byte row alignment required by `copy_texture_to_buffer`. We
/// pad to this on copy and strip the padding back out after readback.
const COPY_BYTES_PER_ROW_ALIGNMENT: u32 = 256;

fn align_to(x: u32, alignment: u32) -> u32 {
    x.div_ceil(alignment) * alignment
}

/// Render-and-download the current creature as a PNG.
///
/// Approach (chosen after `canvas.toBlob` on the live WebGPU surface
/// returned a black frame): build a fresh offscreen `Rgba8Unorm`
/// texture, render the *same* visualize pass into it at a
/// screenshot-friendly resolution, copy the texture into a staging
/// buffer, and `map_async` it. The mapped bytes are PNG-encoded by
/// the `image` crate and handed to the browser via a synthesised
/// `<a download>` link.
///
/// We **do not** capture the egui side panel — the PNG is pure
/// Flow-Lenia output, so SNS shares are unobstructed.
fn trigger_screenshot(state: &AppState, step: u64) {
    // Target a fixed maximum dimension, then pick the largest integer
    // upscale that fits. M4.5 made the grid size runtime-configurable,
    // so read the live cfg instead of the compile-time constant.
    const PNG_MAX_DIM: u32 = 1024;
    let grid_w = state.cfg.grid_width;
    let grid_h = state.cfg.grid_height;
    let upscale = (PNG_MAX_DIM / grid_w.max(grid_h)).max(1);
    let png_w = grid_w * upscale;
    let png_h = grid_h * upscale;

    // Offscreen visualize at a known sRGB-correct format. Rgba8Unorm
    // is what the `image` PNG encoder expects bit-for-bit, so we
    // skip any sRGB conversion on the readback path.
    let device = state.gpu.device.clone();
    let queue = state.gpu.queue.clone();
    let activation_buf = state.pipeline.a_buffer(state.pipeline.ping_index()).clone();
    let visualize = VisualizePass::new(&state.gpu, wgpu::TextureFormat::Rgba8Unorm, upscale);
    let globals_buf =
        visualize.upload_globals(&state.gpu, grid_h, grid_w, state.cfg.channels);
    let bind_group = visualize.make_bind_group(&state.gpu, &activation_buf, &globals_buf);

    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("screenshot offscreen"),
        size: wgpu::Extent3d {
            width: png_w,
            height: png_h,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    let bytes_per_pixel = 4u32;
    let padded_bytes_per_row = align_to(png_w * bytes_per_pixel, COPY_BYTES_PER_ROW_ALIGNMENT);
    let buffer_size = u64::from(padded_bytes_per_row) * u64::from(png_h);
    let staging = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("screenshot staging"),
        size: buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("screenshot encoder"),
    });
    visualize.record(&mut encoder, &bind_group, &view, None);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &staging,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded_bytes_per_row),
                rows_per_image: Some(png_h),
            },
        },
        wgpu::Extent3d {
            width: png_w,
            height: png_h,
            depth_or_array_layers: 1,
        },
    );
    queue.submit([encoder.finish()]);

    let filename = format!("flow-lenia-step{step}-seed{SEED}.png");
    wasm_bindgen_futures::spawn_local(async move {
        // Move-keep the heavy resources alive through the await; they
        // drop at the end of this future after the download fires.
        let _hold_texture = texture;
        let _hold_view = view;
        let _hold_visualize = visualize;
        let _hold_globals = globals_buf;
        let _hold_bind = bind_group;
        let _hold_activation = activation_buf;

        let (sender, receiver) = futures_channel::oneshot::channel();
        staging
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        match receiver.await {
            Ok(Ok(())) => {}
            other => {
                log::warn!("screenshot map_async failed: {other:?}");
                return;
            }
        }

        let bytes: Vec<u8> = {
            let mapped = staging.slice(..).get_mapped_range();
            // Strip the 256-byte row alignment padding.
            let mut tight = Vec::with_capacity((png_w * png_h * bytes_per_pixel) as usize);
            let row_bytes = (png_w * bytes_per_pixel) as usize;
            for row in 0..png_h as usize {
                let start = row * padded_bytes_per_row as usize;
                tight.extend_from_slice(&mapped[start..start + row_bytes]);
            }
            tight
        };
        staging.unmap();
        drop(queue);
        drop(device);

        let mut png_buf: Vec<u8> = Vec::new();
        {
            use image::codecs::png::PngEncoder;
            use image::{ColorType, ImageEncoder};
            let encoder = PngEncoder::new(&mut png_buf);
            if let Err(e) = encoder.write_image(&bytes, png_w, png_h, ColorType::Rgba8) {
                log::warn!("screenshot PNG encode failed: {e:?}");
                return;
            }
        }

        // Hand the encoded bytes to the browser as an `image/png` Blob.
        let uint8 = js_sys::Uint8Array::from(&png_buf[..]);
        let array = js_sys::Array::new();
        array.push(&uint8.buffer());
        let bag = web_sys::BlobPropertyBag::new();
        bag.set_type("image/png");
        let Ok(blob) =
            web_sys::Blob::new_with_u8_array_sequence_and_options(&array, &bag)
        else {
            log::warn!("screenshot Blob construction failed");
            return;
        };
        let Ok(url) = web_sys::Url::create_object_url_with_blob(&blob) else {
            log::warn!("screenshot createObjectURL failed");
            return;
        };
        let Some(document) = web_sys::window().and_then(|w| w.document()) else {
            return;
        };
        let Ok(anchor_el) = document.create_element("a") else {
            return;
        };
        let Ok(anchor) = anchor_el.dyn_into::<web_sys::HtmlAnchorElement>() else {
            return;
        };
        anchor.set_href(&url);
        anchor.set_download(&filename);
        anchor.click();
        let _ = web_sys::Url::revoke_object_url(&url);
        log::info!("screenshot saved: {filename} ({png_w}×{png_h})");
    });
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

    // ─── 1. egui pass: build UI shapes from this frame's input ─────
    let raw_input = state.egui_winit_state.take_egui_input(&state.window);
    // Snapshot the bits the closure needs to read; the closure then
    // signals back through these locals so we can mutate `state`
    // outside the closure (the closure can't borrow `state` because
    // `state.egui_ctx.run` already holds the egui-side borrow).
    let running_before = state.running;
    let step_before = state.pipeline.step_count();
    let fps_before = state.current_fps;
    let mass_before = state.cached_mass.clone();
    let seed_before = state.seed;
    let mut pause_clicked = false;
    let mut reset_clicked = false;
    let mut screenshot_clicked = false;
    let mut apply_clicked = false;
    let mut new_seed_clicked = false;
    let mut apply_grid_clicked = false;
    // Live "lightweight" cfg field snapshots — written by the egui
    // closure, applied to `state.cfg` + the GPU uniform after the
    // closure returns. The grid/channels/num_kernels controls write
    // to `state.pending_*` directly so the ComboBox / Slider state
    // persists across frames (a frame-local mirror would reset to
    // `state.cfg.*` on each redraw and snap the dropdown back to its
    // previous value).
    let mut live_paper_strict = state.cfg.paper_strict;
    let mut live_border = state.cfg.border;
    let mut live_dt = state.cfg.dt;
    let mut live_dd = state.cfg.dd;
    // Mirror the AppState pending fields into closure-local vars so
    // the egui closure can `&mut` them without colliding with the
    // outer `state.egui_ctx.run` borrow. Written back to `state.*`
    // immediately after the closure returns — same frame, no UX lag.
    let mut local_grid = state.pending_grid;
    let mut local_channels = state.pending_channels;
    let mut local_kernels = state.pending_num_kernels;
    // egui 0.34 deprecated `SidePanel::show(&Context, ...)` in favour
    // of a Ui-centric API (PR #5659 unifies SidePanel / TopBottomPanel
    // under a single `Panel` and pushes everyone toward
    // `show_inside(&mut Ui, ...)`). The Context-based call still works
    // at runtime — slated for a follow-up clean-up sub-step inside M4.
    #[allow(deprecated)]
    let full_output = state.egui_ctx.run(raw_input, |ctx| {
        // Only the SidePanel is shown — egui leaves the CentralPanel
        // region untouched so the visualize pass's pixels stay visible
        // underneath. (Adding a CentralPanel here, even with
        // `Frame::NONE`, paints a fullscreen background quad that
        // covers the creature.)
        egui::SidePanel::right("controls")
            .resizable(false)
            .exact_width(SIDE_PANEL_W as f32)
            .show(ctx, |ui| {
                // SidePanel grew enough in M4.4 that the bottom rows can
                // run off the canvas at small viewports — wrap the body
                // in a ScrollArea so users can still reach Pause /
                // Screenshot when the window is short.
                egui::ScrollArea::vertical().show(ui, |ui| {
                    ui.heading("Flow-Lenia");
                    ui.separator();
                    // ─── Stats (M4.3) ──────────────────────────────
                    ui.label(format!("FPS: {fps_before:.1}"));
                    ui.label(format!("Step: {step_before}"));
                    ui.label(format!("Seed: {seed_before}"));
                    ui.add_space(4.0);
                    ui.label("Mass:");
                    for (c, &mass) in mass_before.iter().enumerate() {
                        ui.label(format!("  C{c}: {mass:.2}"));
                    }
                    ui.separator();
                    // ─── Parameters (M4.4, live) ───────────────────
                    ui.checkbox(&mut live_paper_strict, "Paper strict");
                    ui.label("Border:");
                    ui.horizontal(|ui| {
                        ui.radio_value(&mut live_border, BorderMode::Torus, "Torus");
                        ui.radio_value(&mut live_border, BorderMode::Wall, "Wall");
                    });
                    ui.add(egui::Slider::new(&mut live_dt, 0.05..=0.5).text("dt"));
                    ui.add(egui::Slider::new(&mut live_dd, 3..=7).text("dd"));
                    ui.separator();
                    // ─── Kernel set (M4.4, deferred / Apply) ───────
                    ui.add(egui::Slider::new(&mut local_kernels, 1..=45).text("Kernels"));
                    ui.horizontal(|ui| {
                        if ui.button("Apply").clicked() {
                            apply_clicked = true;
                        }
                        if ui.button("New Seed").clicked() {
                            new_seed_clicked = true;
                        }
                    });
                    ui.separator();
                    // ─── Grid / Channels (M4.5, deferred) ──────────
                    ui.horizontal(|ui| {
                        ui.label("Grid:");
                        egui::ComboBox::from_id_salt("grid_combo")
                            .selected_text(format!("{0}×{0}", local_grid))
                            .show_ui(ui, |ui| {
                                for &size in &[32u32, 64, 128, 256] {
                                    ui.selectable_value(
                                        &mut local_grid,
                                        size,
                                        format!("{size}×{size}"),
                                    );
                                }
                            });
                    });
                    ui.horizontal(|ui| {
                        ui.label("Channels:");
                        egui::ComboBox::from_id_salt("channels_combo")
                            .selected_text(format!("{}", local_channels))
                            .show_ui(ui, |ui| {
                                for &c in &[1u32, 2, 3] {
                                    ui.selectable_value(
                                        &mut local_channels,
                                        c,
                                        format!("{c}"),
                                    );
                                }
                            });
                    });
                    if ui.button("Apply Grid/Channels").clicked() {
                        apply_grid_clicked = true;
                    }
                    ui.separator();
                    // ─── Controls (M4.2) ───────────────────────────
                    let pause_label = if running_before { "Pause" } else { "Resume" };
                    if ui.button(pause_label).clicked() {
                        pause_clicked = true;
                    }
                    if ui.button("Reset").clicked() {
                        reset_clicked = true;
                    }
                    if ui.button("Screenshot").clicked() {
                        screenshot_clicked = true;
                    }
                    ui.add_space(8.0);
                    ui.label("Keys: Space / R / Q");
                });
            });
    });
    // Apply button effects outside the closure so we have unrestricted
    // access to `state`. Mirrors the keyboard shortcut branch behaviour.
    if pause_clicked {
        state.running = !state.running;
        log::info!(
            "{} at step={}",
            if state.running { "resumed" } else { "paused" },
            state.pipeline.step_count()
        );
    }
    if reset_clicked {
        reset_simulation(state);
    }
    if screenshot_clicked {
        let step = state.pipeline.step_count();
        trigger_screenshot(state, step);
    }

    // ─── M4.4 lightweight parameter updates ─────────────────────────
    // paper_strict / border / dt / dd: push straight into the uniform.
    let lightweight_changed = live_paper_strict != state.cfg.paper_strict
        || live_border != state.cfg.border
        || (live_dt - state.cfg.dt).abs() > f32::EPSILON
        || live_dd != state.cfg.dd;
    state.cfg.paper_strict = live_paper_strict;
    state.cfg.border = live_border;
    state.cfg.dt = live_dt;
    state.cfg.dd = live_dd;
    if lightweight_changed {
        state.pipeline.update_globals(&state.gpu, &state.cfg);
    }
    // ─── M4.5 pending control values (persist across frames) ────────
    // The ComboBox / Slider widgets mutate `local_*`; mirror them back
    // into `state.pending_*` so the next frame's closure sees the
    // user's choice. Only `state.cfg` changes on Apply / New Seed.
    state.pending_grid = local_grid;
    state.pending_channels = local_channels;
    state.pending_num_kernels = local_kernels;
    if apply_grid_clicked {
        // Grid is always square; ComboBox edits a single value, mirror
        // it into both width / height so the GpuStepPipeline assertion
        // passes.
        state.cfg.grid_width = state.pending_grid;
        state.cfg.grid_height = state.pending_grid;
        state.cfg.channels = state.pending_channels;
        state.cfg.num_kernels = state.pending_num_kernels;
        state.pending_rebuild = Some(RebuildRequest::ApplyGridAndChannels);
    } else if apply_clicked {
        state.cfg.num_kernels = state.pending_num_kernels;
        state.pending_rebuild = Some(RebuildRequest::ApplyKernelCount);
    } else if new_seed_clicked {
        state.pending_rebuild = Some(RebuildRequest::NewSeed);
    }
    state
        .egui_winit_state
        .handle_platform_output(&state.window, full_output.platform_output);
    let paint_jobs = state
        .egui_ctx
        .tessellate(full_output.shapes, full_output.pixels_per_point);
    let screen_descriptor = egui_wgpu::ScreenDescriptor {
        size_in_pixels: [state.surface_config.width, state.surface_config.height],
        pixels_per_point: full_output.pixels_per_point,
    };

    // ─── 2. wgpu encoder: visualize → egui → present ───────────────
    let bg = &state.visualize_bind_groups[state.pipeline.ping_index()];
    let mut encoder = state
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M4.1 frame encoder"),
        });

    for (id, image_delta) in &full_output.textures_delta.set {
        state
            .egui_renderer
            .update_texture(&state.gpu.device, &state.gpu.queue, *id, image_delta);
    }
    let egui_pre_cmds = state.egui_renderer.update_buffers(
        &state.gpu.device,
        &state.gpu.queue,
        &mut encoder,
        &paint_jobs,
        &screen_descriptor,
    );

    state
        .visualize
        .record(&mut encoder, bg, &view, Some(state.flow_lenia_viewport));

    {
        let egui_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("egui render pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Load,
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        state.egui_renderer.render(
            &mut egui_pass.forget_lifetime(),
            &paint_jobs,
            &screen_descriptor,
        );
    }

    for id in &full_output.textures_delta.free {
        state.egui_renderer.free_texture(id);
    }

    let mut to_submit = egui_pre_cmds;
    to_submit.push(encoder.finish());
    state.gpu.queue.submit(to_submit);
    frame.present();
}

/// `wasm-bindgen` entry point — invoked by the generated JS glue
/// when the WASM module is instantiated. Trunk wires this up via
/// the `<link data-trunk rel="rust" />` tag.
#[wasm_bindgen(start)]
pub fn run() {
    console_error_panic_hook::set_once();
    console_log::init_with_level(log::Level::Info).expect("console_log init failed");

    log::info!("flow-lenia-web booting (M4.5.1 — RAF-driven render)");

    // M4.5.1 — install the RAF-driven render loop BEFORE entering
    // `event_loop.run_app`. The latter never returns on web (winit
    // suspends via JS throw), so any setup after it would be unreachable.
    // The RAF callback is a no-op while `APP_STATE` is empty; once
    // `AppEvent::GpuReady` populates it, the next animation frame starts
    // stepping + rendering.
    start_raf_loop();

    let event_loop = EventLoop::<AppEvent>::with_user_event()
        .build()
        .expect("failed to create event loop");
    // M4.5.1.2 — initial control flow is `Poll` so the first
    // iteration of the loop runs immediately; `about_to_wait` then
    // re-arms `ControlFlow::WaitUntil(now + 16 ms)` on every wake,
    // which is the steady-state throttle. See the comment on
    // `about_to_wait` for the full rationale (M4.5.1's pure `Wait`
    // dropped discrete events; pure `Poll` burned 90 % CPU on M1).
    event_loop.set_control_flow(ControlFlow::Poll);
    let proxy = event_loop.create_proxy();
    let mut app = App::new(proxy);
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
