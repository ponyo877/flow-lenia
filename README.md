# flow-lenia

Rust + WebAssembly + WebGPU reimplementation of **Flow-Lenia**
(Plantec et al. 2025, *Artificial Life journal*,
[arXiv:2506.08569v1](https://arxiv.org/abs/2506.08569)).

Educational and research-quality reproduction of the 2025 paper's mass-conservative
continuous cellular automaton, with both a CPU reference implementation and a
WebGPU compute pipeline target. Multi-species (Eq. 7/8 parameter embedding) and
mutation beams are supported in later milestones.

## Documents

- [`DESIGN.md`](DESIGN.md) — Authoritative design specification (currently **Rev. 4.1**)
- [`BENCH.md`](BENCH.md) — Measured CPU vs GPU per-step times, per-pass
  breakdown, init time, GPU memory (as of M2.11)
- [`references/JAX_NOTES.md`](references/JAX_NOTES.md) — Annotated reading of the
  official JAX implementation (`erwanplantec/FlowLenia`, commit `dce428c`)
- [`JAX_NOTES.md`](JAX_NOTES.md) — Symlink to the above for convenient access from
  the workspace root
- `papers/` — Source PDFs (2025 + 2023 Flow-Lenia, Lenia 2019)
- `references/FlowLenia-jax/` — Read-only JAX reference (excluded from the repo via
  `.gitignore` because the upstream lacks an explicit license file). To reproduce
  the exact source consulted by `references/JAX_NOTES.md`:

  ```sh
  git clone https://github.com/erwanplantec/FlowLenia.git references/FlowLenia-jax
  cd references/FlowLenia-jax && git checkout dce428c6b0c5079a06e5606fb7b5ac1fe1323bc5
  ```

## Workspace layout

| Crate | Purpose | Status |
|---|---|---|
| `crates/flow-lenia-core/` | Platform-independent CA logic and CPU reference | **M1 complete** |
| `crates/flow-lenia-gpu/` | wgpu compute pipeline | **M2 complete** |
| `crates/flow-lenia-ui/` | egui controls / statistics panels | M1.1 skeleton |
| `crates/flow-lenia-app/` | Native binaries (`native_cpu`, `native_gpu`, `bench_step`, `generate_m1_fixtures`) | **M2 complete** |

## Build / Run

```sh
# Verify the workspace compiles
cargo check --workspace

# Run all tests (release; the integration tests are GPU-touching)
cargo test --release --workspace

# Native CPU reference — ANSI terminal animation (M1.14)
cargo run --release --bin native_cpu -- [seed] [steps] [render_every]

# Native GPU — 512×512 winit window with real-time Flow-Lenia (M2.10)
#   Space: pause / resume    r: reset to seed    q: quit
cargo run --release --bin native_gpu -- [steps_per_frame=1] [seed=1729]

# Per-step benchmark — CPU vs GPU across grid sizes (M2.11)
cargo run --release --bin bench_step
```

### Web build (M3.5 — Chrome WebGPU)

```sh
# Build + serve crates/flow-lenia-web at http://localhost:8080/
cd crates/flow-lenia-web && trunk serve
```

Requires a WebGPU-capable browser. Chrome stable (148+) is the reference
target — see [`BROWSER_COMPAT.md`](BROWSER_COMPAT.md) for measured behaviour
on Chrome and the current status of Safari / Firefox.

Keyboard (same as the native binary):

| Key       | Action                                                                   |
| --------- | ------------------------------------------------------------------------ |
| `Space`   | Toggle running. Pause freezes the step counter; redraw continues.        |
| `r` / `R` | Reset the pipeline from `seed=1729` (verifies determinism).              |
| `q` / `Q` | `event_loop.exit()`. Canvas freezes at the last frame; the tab stays    |
|           | open (WASM cannot close a browser tab without a user gesture).           |

Toolchain: Rust **1.95.0** stable (pinned via `rust-toolchain.toml`).

## Regression test running guide

The flow-lenia regression suite splits into two tiers so day-to-day
`cargo test` stays fast (~45 s) while the heavy long-horizon and
large-grid checks live behind `--include-ignored`.

### Day-to-day (lightweight, ~45 s)

```sh
cargo test --release -p flow-lenia-core
```

Runs the 32×32 and 64×64 bit-equal regressions (`m1_regression_g32`,
`_g64`) plus all other non-`#[ignore]` tests. The 128×128 and 256×256
bit-equal regressions, all five `mass_conservation_*` tests, and the
two one-off measurement tests are skipped.

### Full pre-push verification (heavy, ~1.5 h)

```sh
cargo test --release -p flow-lenia-core -- --include-ignored
```

Runs the full set:

- `m1_regression_g{32,64,128,256}` — bit-equal CPU regression across
  all M6.A.1 fixtures (~13 min).
- `mass_conservation_g{32,64,128,256,512}` — mass-conservation matrix
  with tiered step counts (~46 min total, see
  `crates/flow-lenia-core/tests/mass_conservation_1k.rs` for the
  per-grid step / case selection).
- `drift_vs_grid_size_100step` — one-off measurement for
  `BENCH.md` Section 5 (~42 min).
- `baseline_64x64_1000step` — one-off measurement for `BENCH.md`
  Section 6 (~7.7 min).

Run the heavy set before pushing any commit that could touch the
simulator's numerical path (kernel sampling, convolve, growth,
reintegrate, gradient). The two one-off measurement tests can be
omitted when their previously-captured BENCH.md values are still
known to apply.

### GPU regression tolerance (M6.A.4 + A.4.5)

The four `gpu_field_regression_g{N}` tests use **tiered tolerances**
(1e-4 / 5e-4 / 1e-3 / 2.5e-3 from g32 to g256), not a uniform
`rel < 1e-3`. The tolerance scales because the underlying
Flow-Lenia dynamics is **chaotic** at C=1 / grid ≥ 64 — an
ε = 1e-6 perturbation saturates to O(0.8) in *one step* at g64
on CPU-only simulation, so the grid-independent per-cell f32
add-order delta between CPU and GPU (~ 1e-5) gets amplified
grid-dependently over a few steps.

This is **dynamics intrinsic**, not a GPU implementation defect.
The full investigation, with per-step rel tables and the CPU
Lyapunov measurement, is in `BENCH.md` §8 "A.4.5 GPU regression
tolerance". The takeaway: a future M6 step that pushes a
`gpu_field_regression_g{N}` test rel above its tolerance is
either a real regression in the simulator, or chaos has shifted
— the BENCH §8 baseline tells you which.

### Partial execution

Every regression test name embeds its grid, so a single-grid run is
one flag away:

```sh
cargo test --release -p flow-lenia-core m1_regression_g64
cargo test --release -p flow-lenia-core mass_conservation_g256 \
    -- --include-ignored
```

## Milestone status

- [x] **M1.1** — Project skeleton
- [x] **M1.2** — Common type definitions (`FlowLeniaConfig`, mode enums, `KernelParams`)
- [x] **M1.3** — Parameter sampling (JAX `flowlenia.py:55-64` ranges)
- [x] **M1.4** — Kernel generation (JAX form, paper Eq. 1 mapped)
- [x] **M1.5** — Growth function `G_i` (paper Eq. 2)
- [x] **M1.6** — Direct convolution (CPU, torus/wall, per-kernel radius)
- [x] **M1.7** — Sobel filter (no normalization, JAX `utils.py:16-37`)
- [x] **M1.8** — α computation (both modes per `DESIGN.md` §4.1.5)
- [x] **M1.9** — Flow `F` (paper Eq. 5)
- [x] **M1.10** — Overlap area (with `min(1, 2σ)` clip per JAX `utils.py:57-58`)
- [x] **M1.11** — Reintegration tracking (paper Eq. 6, 11×11 neighborhood)
- [x] **M1.12** — Affinity `U` with parameter embedding (paper Eq. 7)
- [x] **M1.13** — One-step integration
- [x] **M1.14** — Terminal visualization
- [x] **M1.15** — Mass conservation across all mode combinations
- [x] **M2.1** — wgpu init + winit blue-screen baseline
- [x] **M2.2** — Kernel-bank GPU buffer upload (Plan A: fixed stride, zero-pad)
- [x] **M2.3** — Convolve compute shader (`pre_g = K_i ∗ A_{c_i^0}`)
- [x] **M2.4** — Affinity-growth compute shader (paper Eq. 3 + Eq. 7)
- [x] **M2.5** — Gradient shaders (`∇U` per-channel, `∇A_Σ` on-the-fly)
- [x] **M2.6** — Flow shader (combined α + F per paper Eq. 5, both modes)
- [x] **M2.7** — Reintegration tracking (paper Eq. 6, dd-neighbourhood loop)
- [x] **M2.8** — `GpuStepPipeline` full step + M1.15 fixture regression
- [x] **M2.9** — Visualisation render pass (storage buffer → sRGB target)
- [x] **M2.10** — Native binary with winit event loop + keyboard control
- [x] **M2.11** — Performance benchmarks ([BENCH.md](BENCH.md))
- [x] **M3.1** — `wasm32-unknown-unknown` build green
- [x] **M3.2** — Hello WebGPU in Chrome (blue-screen baseline)
- [x] **M3.3** — Convolve pass through `wgpu` on the browser
- [x] **M3.4** — Full pipeline animation in the canvas
- [x] **M3.5** — Chrome stability + keyboard verify; Safari 26.3.1 and
  Firefox 150.0.3 basic-functionality verified
  ([BROWSER_COMPAT.md](BROWSER_COMPAT.md))

See `DESIGN.md` §8 for milestone definitions and completion criteria.

## M2 completion evidence

| DESIGN.md §8 criterion | Evidence |
|---|---|
| All 6 compute passes + visualize | `crates/flow-lenia-gpu/src/passes/{convolve, affinity_growth, gradient, flow, reintegrate, visualize}.rs` |
| Reference vs GPU agreement (single-step) | `pipeline::tests::gpu_pipeline_*_matches_cpu` (M2.8) — `rel < 1e-4 OR abs < 1e-5` |
| Mass conservation (32×32, 500 steps, rel < 1e-3 torus / 1e-2 wall) | `bench_step` Section 4 — see [BENCH.md](BENCH.md) |
| 4 modes (`paper_strict × border`) tested | `m1_regression_gpu::gpu_pipeline_mass_conservation_100_steps` — 8 cases pass |
| Regression fixture committed | `tests/regression_fixtures/m1_baseline/` (8 cases, M1.15 generator) |
| Real-time animation in native window | `native_gpu` at 55+ FPS on 64×64 / C=3 (M2.10) |

## JAX fixture re-generation (optional, M1.6 L2)

The L2 smoke test (`crates/flow-lenia-core/tests/jax_fixture_smoke.rs`)
compares Rust convolution output against JAX `jax.scipy.signal.convolve2d`
fixtures committed to `tests/fixtures/`. Re-generate only if `CASES` in the
script changes:

```sh
python3 -m venv .venv-fixtures
.venv-fixtures/bin/pip install -U pip 'jax[cpu]' numpy
.venv-fixtures/bin/python scripts/generate_jax_fixtures.py
```

`.venv-fixtures/` is gitignored. CI does not need a JAX environment.

## M1 regression fixtures

`crates/flow-lenia-core/tests/m1_regression.rs` asserts bit-equality
between the simulator output and the 32 baseline fixtures committed
under `tests/regression_fixtures/m1_baseline/` — 4 grid sizes (32, 64,
128, 256) × 8 `paper_strict × border × C` cases each, 100 steps from
`seed=42`. (M6.A.1 expanded the fixture set; 512×512 is intentionally
skipped from bit-equal regression to keep the fixture footprint at
5.4 MB on disk and the regeneration time at ~17 min; the
mass-conservation suite covers 512.) Re-generate only when the
dynamics or supporting infra intentionally changes:

```sh
cargo run --release --bin generate_m1_fixtures
```

Bit-equality requires the **same Rust toolchain (1.95.0, pinned via
`rust-toolchain.toml`)** and the **same resolved `ndarray` version**
(see `manifest.json`); regenerate after either upgrade. The full
mass-conservation matrix (5 grids, tiered step counts) lives in
`tests/mass_conservation_1k.rs` and runs under the heavy
`--include-ignored` flow described in
*Regression test running guide* above.

## License

Dual-licensed under either Apache-2.0 or MIT, at your option.
