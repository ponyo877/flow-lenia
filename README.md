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

Toolchain: Rust **1.87.0** stable (pinned via `rust-toolchain.toml`).

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
between the simulator output and the 8 baseline fixtures committed under
`tests/regression_fixtures/m1_baseline/` (one per `paper_strict × border × C`
combination, 100 steps from `seed=42`). Re-generate only when the dynamics
or supporting infra intentionally changes:

```sh
cargo run --release --bin generate_m1_fixtures
```

Bit-equality requires the **same Rust toolchain (1.87.0, pinned via
`rust-toolchain.toml`)** and the **same resolved `ndarray` version**
(see `manifest.json`); regenerate after either upgrade. The 1000-step
mass-conservation matrix lives in `tests/mass_conservation_1k.rs` and
runs under `cargo test --release -p flow-lenia-core --test
mass_conservation_1k -- --ignored`.

## License

Dual-licensed under either Apache-2.0 or MIT, at your option.
