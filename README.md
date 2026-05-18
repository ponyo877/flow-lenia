# flow-lenia

Rust + WebAssembly + WebGPU reimplementation of **Flow-Lenia**
(Plantec et al. 2025, *Artificial Life journal*,
[arXiv:2506.08569v1](https://arxiv.org/abs/2506.08569)).

Educational and research-quality reproduction of the 2025 paper's mass-conservative
continuous cellular automaton, with both a CPU reference implementation and a
WebGPU compute pipeline target. Multi-species (Eq. 7/8 parameter embedding) and
mutation beams are supported in later milestones.

## Documents

- [`DESIGN.md`](DESIGN.md) — Authoritative design specification (currently **Rev. 4**)
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
| `crates/flow-lenia-core/` | Platform-independent CA logic and CPU reference | M1 complete |
| `crates/flow-lenia-gpu/` | wgpu compute pipeline | M1.1 skeleton |
| `crates/flow-lenia-ui/` | egui controls / statistics panels | M1.1 skeleton |
| `crates/flow-lenia-app/` | Native binaries (`native_cpu`, `native_gpu`, `generate_m1_fixtures`) | M1.14 native_cpu live |

## Build / Run

```sh
# Verify the workspace compiles
cargo check --workspace

# Run the (placeholder) native CPU binary
cargo run -p flow-lenia-app --bin native_cpu

# Run the (placeholder) native GPU binary
cargo run -p flow-lenia-app --bin native_gpu
```

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
- [ ] **M2.1–M2.11** — GPU pipeline

See `DESIGN.md` §8 for milestone definitions and completion criteria.

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
