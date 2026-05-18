# Known Issues

## 1. winit 0.30.13 (web) + wgpu 29: `WindowEvent::RedrawRequested` is never delivered

### Symptom

On the `wasm32-unknown-unknown` target with Chrome's WebGPU backend,
`winit::window::Window::request_redraw()` no longer triggers a
`WindowEvent::RedrawRequested` event in `ApplicationHandler::window_event`.
The EventLoop itself remains healthy — `ApplicationHandler::about_to_wait`
fires ~1000×/sec under `ControlFlow::Poll` — but the redraw event is
never queued, so the M3.5-style "advance the simulation inside the
`RedrawRequested` arm" render loop stalls after its first frame.

### How it was diagnosed

| Probe | Observation |
|---|---|
| Step counter logged from `RedrawRequested` arm | `step=1 fps=0.0 (1 frames in 23.768s)` — only the initial frame |
| Counter inside `about_to_wait` | Logs every frame, ~1380 fires/sec |
| Counter inside `window_event` (any variant) | Zero entries after the first burst of init events |
| Removing egui entirely (M4.1 diag-1) | Identical symptom — egui is **not** the cause |
| Reverting `CANVAS_W` 768 → 512 | Same symptom — layout size is not the cause |
| `about_to_wait` forced `state.window.request_redraw()` | Still no `RedrawRequested` delivery |

The same code path animated correctly on M3.5 (wgpu 25). Re-reading
the M3.5 → M4.0.5 web console reveals the regression had been present
since the M4.0.5 commit (`d16abd0`, wgpu 25 → 29 upgrade); we initially
mistook the single initial frame for normal animation because the
creature image was on screen.

### Workaround

`crates/flow-lenia-web/src/lib.rs` drives `render_frame` from
`ApplicationHandler::about_to_wait` instead of from
`WindowEvent::RedrawRequested`. `ControlFlow::Poll` makes
`about_to_wait` the right hook for "advance one frame each iteration".

```rust
fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
    if let Some(state) = self.state.as_mut() {
        if state.running {
            for _ in 0..state.steps_per_frame {
                state.pipeline.step(&state.gpu);
            }
        }
        render_frame(state);
        state.fps.tick(state.pipeline.step_count(), state.running);
    }
}
```

The `WindowEvent::RedrawRequested` arm is **kept as an empty no-op**
so the moment upstream restores delivery we can move rendering back
there without re-architecting the loop.

### Scope of impact

- Affects only `flow-lenia-web` (Chrome WebGPU + wgpu 29 + winit 0.30.13).
- `native_gpu` is unaffected — the native backend still delivers
  `RedrawRequested` normally and uses the M3.5 loop unchanged.

### Upstream status

Not yet investigated. The root cause sits somewhere in winit's web
backend's interaction with wgpu 29's `Surface::present` path. Filing
an upstream issue is queued for after M4 lands (we want a minimal
repro that doesn't drag in our pipeline).

### When to revisit

- Any `winit` / `wgpu` / Chrome WebGPU upgrade — re-test by moving
  the render call back into `WindowEvent::RedrawRequested` and
  watching for the step counter to advance.
- If `egui-wgpu` releases a version targeting a different wgpu
  major, that bump may incidentally fix or change the symptom.
