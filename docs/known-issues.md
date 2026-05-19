# Known Issues

## 1. winit 0.30.13 (web) + wgpu 29: `WindowEvent::RedrawRequested` is never delivered

**Status (M4.5.1):** obsoleted by issue #4 below — the M4.5.1 RAF model
no longer routes rendering through winit, so the RedrawRequested
delivery bug no longer affects us. Documented here for the historical
record and so that any future refactor that brings rendering back to a
winit-driven path knows what to expect.

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

## 4. Chrome 148 WebGPU: `Surface::present` does not block on vsync

### Symptom

A render loop that calls `step + render + present` inside
`ApplicationHandler::about_to_wait` (which under `ControlFlow::Poll`
fires ~1000×/sec) runs uncapped at ~130 fps on Chrome 148 instead of
clamping at the 60 Hz display refresh. The browser compositor still
displays only ~60 frames/sec, so each visible frame corresponds to
1–3 simulation steps with significant jitter — perceived as
"creature teleports" or motion stutter even though the SidePanel
FPS counter shows 40+.

### How it was diagnosed

`crates/flow-lenia-web/src/lib.rs` instruments two histograms
(`frame_diag/interval` and `frame_diag/render`) and emits percentile
stats every 300 frames. During the M4.5.1 measurement on Claude in
Chrome (Chrome 148, hidden tab):

| Metric | Value | Interpretation |
|---|---|---|
| interval mean | 7.62 ms | ~131 fps loop |
| interval p99 | 12.80 ms | very consistent — no idle gap |
| render mean | 7.30 ms | ≈ interval mean → idle is ~0.3 ms |
| `FpsCounter` peak | 167 fps (1 s window) | uncapped burst |

If `present()` were vsync-gated, interval would clamp at 16.7 ms
(60 fps). It does not — so `present()` is returning before vblank.

### Workaround

M4.5.1 moves rendering off `about_to_wait` and onto a JavaScript
`requestAnimationFrame` loop that drives an exported `tick()` function
in WASM. RAF is naturally paced to the compositor's display rate, so
every visible frame corresponds to exactly one simulation step on a
60 Hz monitor. As a side benefit:

- the winit event loop switches to `ControlFlow::Wait` and only wakes
  on real input, recovering the CPU previously burned by the
  ~1000 Hz `Poll` cadence;
- hidden tabs auto-throttle (Chrome suspends RAF when
  `document.visibilityState === 'hidden'`) which means we no longer
  spin GPU work when the user isn't looking;
- the obsoleted RedrawRequested-non-delivery issue (#1) becomes
  irrelevant — we don't request redraws through winit anymore.

The `start_raf_loop` helper in `crates/flow-lenia-web/src/lib.rs` uses
the canonical `Rc<RefCell<Option<Closure>>>` pattern so the same
closure re-schedules itself frame after frame without leaking a fresh
`Closure` per call. `mem::forget` on the holder is intentional —
`run()` never returns on web (winit suspends via JS `throw`), so we
need the closure to outlive its containing stack frame.

### Scope of impact

`flow-lenia-web` only. Native (`native_gpu`) is unaffected and keeps
its existing `about_to_wait`-driven render loop — the native winit
backend honours vsync through the windowing system, so the bug never
reproduces there.

### When to revisit

If a future Chrome release vsync-blocks `Surface::present` on the
WebGPU path (or wgpu starts forcing a present pacing primitive), the
`about_to_wait` model could come back without the 130 fps explosion.
At that point the trade-off is "saved JS↔WASM round-trips per frame"
vs. "simpler architecture" — measure first.

### Verification residue (left in place)

The `FrameTimingDiag` struct and its hooks (`tick()` in `lib.rs`) are
kept under debug-build so future performance work can rely on them.
They emit one log line every 300 frames (≈ 5 s at 60 fps), which is
quiet enough not to clutter the console but loud enough to catch a
regression at a glance.
