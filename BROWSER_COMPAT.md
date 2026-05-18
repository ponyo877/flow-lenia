# Browser Compatibility — Flow-Lenia (M3.5)

Status as of M3.5 (2026-05-18).

The web build (`crates/flow-lenia-web`) targets `wasm32-unknown-unknown`
with the `wgpu` `webgpu` backend. There is no WebGL2 fallback — Flow-Lenia
needs storage buffers + compute, so WebGPU is a hard requirement.

The `webgpu` backend on each browser is also the only thing being tested
here. The Rust code is the same; what differs is the browser's Tint/Naga
shader compiler, surface format set, present-mode policy, and how it
handles `devicePixelRatio` on HiDPI canvases.

## Reference platform — Chrome stable (measured)

| Field              | Value                                                       |
| ------------------ | ----------------------------------------------------------- |
| OS                 | macOS 26.3.1 (Apple silicon)                                |
| Browser            | Google Chrome 148.0.7778.168 (stable channel)               |
| WebGPU backend     | `wgpu::Backends::BROWSER_WEBGPU` (Tint shader compiler)     |
| Surface format     | `Bgra8Unorm` (no sRGB variant is offered — see note below)  |
| Canvas CSS size    | 512×512 (logical)                                           |
| Canvas physical    | 1024×1024 (DPR=2)                                           |
| Grid               | 64×64 × C=3                                                 |
| Visualize upscale  | 16× (= physical 1024 / grid 64)                             |
| Steady-state FPS   | 45–60 (steps_per_frame=1, `Bgra8Unorm` present, vsync)      |
| Init time          | < 1 s from page load to first frame                         |
| Long-run stability | 2000+ steps with no panic, NaN, or queue stall              |
| JS heap            | ~8–12 MB steady (no leak across 2000+ steps)                |

### Notes

* **No sRGB surface variant.** Chrome WebGPU advertises `Bgra8Unorm` and
  `Rgba8Unorm` but *not* their sRGB-encoded variants. The compositor
  performs the sRGB encode itself, so output matches the M2.10 native
  sRGB-surface path visually — just on a different code path. The
  surface-format selection in `lib.rs` therefore falls through to the
  first non-sRGB format Chrome offers.
* **Init-time `inner_size` race.** winit's `window.inner_size()` returns
  `0×0` briefly during init before the canvas CSS layout settles. The
  surface (and visualize `upscale`) are derived from
  `CANVAS_W/H × devicePixelRatio` whenever that happens — see
  `resolve_physical_canvas_size` in `crates/flow-lenia-web/src/lib.rs`
  for the full rationale and the M4 TODO note.
* **Long-run dynamics.** With the demo seed (1729, kernels = 10,
  `paper_strict=false`) the creature visibly dissipates around step
  ~2300. This is expected Lenia behaviour for an unconstrained random
  kernel set — not a stability bug. Mass conservation across the same
  trajectory was previously verified to `rel < 1e-3` (torus) in
  BENCH.md §4.

## Other browsers (not yet verified on real hardware)

| Browser           | Expected status    | Notes                                            |
| ----------------- | ------------------ | ------------------------------------------------ |
| Safari 26+        | Should work        | WebGPU enabled by default in Safari 26 (2025).   |
| Firefox 147+      | Should work        | WebGPU shipped stable in Firefox 147 (2025).     |
| Chromium-based\*  | Same as Chrome     | Edge / Brave / Arc all reuse Blink + Tint.       |

\* Vivaldi and Opera ship Chromium but with their own WebGPU flag
defaults; treat as "should work" pending real check.

These have not been exercised yet during M3 — the in-IDE browser
automation harness only attaches to Chrome. Their status will be updated
during M5 (deploy / GitHub Pages) when the bundle goes live and the
maintainer can hit each browser directly.

## Known limitations

* **No tab close from WASM.** The `q` key calls `event_loop.exit()`,
  which on web stops the winit event loop (so `step` and `render`
  halt and the canvas freezes at its last frame). The browser tab
  itself stays open — closing it requires a user gesture the page
  cannot synthesise. This is documented behaviour and not a bug.
* **Fixed canvas size.** `CANVAS_W` / `CANVAS_H` are compile-time
  constants. Window resize events from the browser are not yet wired
  back into a `surface.configure(...)` re-call, and `resolve_physical_canvas_size`'s
  `<= 1` heuristic assumes the canvas stays at its CSS-pixel default.
  Both are M4 UI work.
* **Linux / Android / Chrome OS, mobile Safari.** Untested. WebGPU
  shipped on Chrome / Android in 2023 and on iOS 26 Safari in 2025, so
  basic functionality is expected, but performance characteristics
  (especially on Android tile-based GPUs) and DPR handling may differ.
  Plan to revisit during M5 deploy.

## Keyboard

Confirmed in Chrome 148 (M3.5-b):

| Key        | Action                                                          |
| ---------- | --------------------------------------------------------------- |
| `Space`    | Toggle running. Pause: step counter freezes, `[paused]` appears |
|            | in the per-second FPS log, redraw continues at ~60 fps.         |
| `r` / `R`  | Rebuild the pipeline from `FlowLeniaSimulator(seed=1729)` and   |
|            | resume. Verifies determinism (same creature reappears).         |
| `q` / `Q`  | `event_loop.exit()`. Stops stepping and redrawing; canvas       |
|            | freezes at the last frame. Tab stays open (see above).          |
