#![deny(warnings)]
//! Native GPU binary for Flow-Lenia.
//!
//! M2.1 lands the minimum viable wgpu pipeline: a winit window with a
//! wgpu surface that clears to a flat blue colour every frame. No
//! compute passes, no Flow-Lenia state on the GPU yet — those land
//! in M2.2 .. M2.10.
//!
//! Usage:
//!
//! ```text
//! RUST_LOG=info cargo run --release --bin native_gpu
//! ```
//!
//! Logs the selected adapter and the rolling 1-second FPS to stderr.
//! Window closes via the usual platform shortcuts (Cmd-Q / Alt-F4 /
//! close button).

use flow_lenia_gpu::{GpuContext, SurfaceState};
use std::sync::Arc;
use std::time::Instant;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowAttributes, WindowId};

/// All state that depends on a window existing.
///
/// Held inside `App` as an `Option` because winit's lifecycle only
/// guarantees a window after the `resumed` callback fires.
struct Render {
    window: Arc<Window>,
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    gpu: GpuContext,
}

struct App {
    render: Option<Render>,
    fps_counter: FpsCounter,
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

    fn tick(&mut self) {
        self.frames += 1;
        let elapsed = self.last_report.elapsed();
        if elapsed.as_secs_f64() >= 1.0 {
            log::info!(
                "fps={:.1} ({} frames in {:.3}s)",
                f64::from(self.frames) / elapsed.as_secs_f64(),
                self.frames,
                elapsed.as_secs_f64()
            );
            self.frames = 0;
            self.last_report = Instant::now();
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        // `resumed` can fire multiple times on some platforms (mobile);
        // guard so we only build the renderer once.
        if self.render.is_some() {
            return;
        }

        let attrs = WindowAttributes::default()
            .with_title("flow-lenia native_gpu (M2.1)")
            .with_inner_size(LogicalSize::new(640u32, 480u32));
        let window = Arc::new(
            event_loop
                .create_window(attrs)
                .expect("failed to create window"),
        );
        let size = window.inner_size();

        let (gpu, SurfaceState { surface, config }) = flow_lenia_gpu::block_on(
            GpuContext::with_surface(window.clone(), size.width, size.height),
        );

        self.render = Some(Render {
            window,
            surface,
            config,
            gpu,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(render) = self.render.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => {
                log::info!("close requested — exiting");
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                // Skip 0-sized surfaces (minimised windows) — they panic in `configure`.
                if size.width == 0 || size.height == 0 {
                    return;
                }
                render.config.width = size.width;
                render.config.height = size.height;
                render.surface.configure(&render.gpu.device, &render.config);
            }
            WindowEvent::RedrawRequested => {
                render_frame(render);
                self.fps_counter.tick();
                // Continuous redraw — request the next frame.
                render.window.request_redraw();
            }
            _ => {}
        }
    }
}

fn render_frame(render: &mut Render) {
    let frame = match render.surface.get_current_texture() {
        Ok(frame) => frame,
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            // Re-configure and try again next frame.
            render.surface.configure(&render.gpu.device, &render.config);
            return;
        }
        Err(e) => {
            log::warn!("get_current_texture error: {e:?}");
            return;
        }
    };
    let view = frame
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());
    let mut encoder = render
        .gpu
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("M2.1 clear-blue encoder"),
        });
    {
        // Scoped so the borrow on `encoder` drops before `encoder.finish()`.
        let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("M2.1 clear-blue pass"),
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
    render.gpu.queue.submit([encoder.finish()]);
    frame.present();
}

fn main() {
    // Default to "info" so the adapter and FPS log lines actually appear.
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let event_loop = EventLoop::new().expect("failed to create event loop");
    // Poll (not Wait) so we drive continuous redraws — vsync inside the
    // surface's present blocks naturally.
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App {
        render: None,
        fps_counter: FpsCounter::new(),
    };
    event_loop
        .run_app(&mut app)
        .expect("event loop terminated with error");
}
