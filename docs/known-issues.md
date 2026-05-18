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

## 2. Chrome WebGPU canvas: `canvas.toBlob` returns a black image

### Symptom

Calling `HTMLCanvasElement::to_blob` (Rust web-sys) or
`canvas.toBlob()` (JS) on the canvas backing the WebGPU surface
returns a `Blob` of the expected size but the decoded PNG is solid
black.

### Cause

`GPUCanvasContext::getCurrentTexture` hands back a swap-chain
texture that lives outside the 2D-canvas pipeline; Chrome's
`toBlob` / `toDataURL` reach into the 2D-canvas backbuffer and
therefore see nothing. Adding `wgpu::TextureUsages::COPY_SRC` to
the surface configuration does not change the behaviour — the
limit is on the read side, not on the wgpu side.

### Workaround

Render the same `VisualizePass` into a fresh `Rgba8Unorm` offscreen
texture, `copy_texture_to_buffer` into a staging buffer with
256-byte row alignment, `map_async` the buffer, strip the padding
on the CPU, and feed the bytes to the `image` crate's PNG encoder.
The encoded bytes go to the browser as a `Blob` and a synthesised
`<a download>` triggers the file save. See
`crates/flow-lenia-web/src/lib.rs::trigger_screenshot` for the full
M4.2 implementation.

A side benefit: the offscreen path captures the Flow-Lenia field
*without* the egui SidePanel, which is exactly what we want for
SNS sharing.

### Scope of impact

`flow-lenia-web` only. native binaries take screenshots through
the existing `texture_readback` path with no surface involved.

## 3. wgpu 29 (web) needs an explicit `Device::poll` to advance `map_async`

### Symptom

A `Buffer::map_async` callback never fires after the work is
`queue.submit()`-ted — the awaiting `oneshot` receiver hangs
forever, the readback path stalls, and (from the M4.2 / M4.3
investigation) the screenshot / mass UI never refreshes.

### Cause

On native, the driver progresses the wgpu queue implicitly. On the
browser the wgpu 29 runtime only advances its internal queue when
the caller asks it to via `Device::poll`. Without the poll the
internal "submitted → completed → fire map callback" pipeline
never moves past the first step.

### Workaround

`flow-lenia-web` calls
`state.gpu.device.poll(wgpu::PollType::Poll)` once per frame
inside `ApplicationHandler::about_to_wait` (the same hook that
already drives the workaround for issue #1). The cost is
negligible — `PollType::Poll` is non-blocking and returns
immediately when no submission is in flight.

### Scope of impact

Web only. Native readback paths still use `PollType::Wait` and
block synchronously.

### When to revisit

If a wgpu release notes "implicit web queue progression", the
explicit poll can move into the readback paths themselves (so it
only costs CPU when there's an actual `map_async` in flight) or
disappear entirely.
