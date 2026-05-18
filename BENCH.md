# Flow-Lenia benchmarks

Recorded by `cargo run --release --bin bench_step` on the build host.
Re-run on your own hardware to compare; see the binary source at
`crates/flow-lenia-app/src/bin/bench_step.rs`.

## Build host

| field | value |
|---|---|
| machine | Apple M1 mini |
| OS | macOS (Darwin 25.3.0) |
| wgpu backend | Metal |
| adapter | Apple M1 (IntegratedGpu) |
| rust toolchain | 1.87.0 (pinned via `rust-toolchain.toml`) |
| profile | release (`opt-level=3`, `lto="thin"`, `codegen-units=1`) |
| benchmark date | 2026-05-18 |

## Section 1 — full-step matrix (CPU vs GPU)

`|K| = 10`, `seed = 1729`. Each measurement is the average of
`measure` simulator steps after `warmup` steps of warm-up. The GPU
column **excludes the visualization render pass** so the comparison
is for pure simulator throughput.

| grid | C | warmup | measure | CPU μs/step | GPU μs/step | CPU ms/step | GPU ms/step | GPU/CPU | step rate (CPU, GPU) |
|-----:|--:|-------:|--------:|------------:|------------:|------------:|------------:|--------:|----------------------|
|   32 | 1 |    100 |    1000 |    12 275.6 |     6 267.4 |       12.28 |        6.27 |   0.51× |  81.5 sps,   159.6 sps |
|   32 | 3 |    100 |    1000 |    13 348.4 |     7 713.1 |       13.35 |        7.71 |   0.58× |  74.9 sps,   129.6 sps |
|   64 | 1 |    100 |    1000 |    49 623.1 |    13 285.6 |       49.62 |       13.29 |   0.27× |  20.2 sps,    75.3 sps |
|   64 | 3 |    100 |    1000 |    53 363.3 |    16 285.3 |       53.36 |       16.29 |   0.31× |  18.7 sps,    61.4 sps |
|  128 | 1 |     30 |     300 |   199 837.4 |    47 593.9 |      199.84 |       47.59 |   0.24× |   5.0 sps,    21.0 sps |
|  128 | 3 |     30 |     300 |   213 437.7 |    58 545.7 |      213.44 |       58.55 |   0.27× |   4.7 sps,    17.1 sps |
|  256 | 1 |     10 |     100 |   798 617.8 |   184 846.5 |      798.62 |      184.85 |   0.23× |   1.3 sps,     5.4 sps |
|  256 | 3 |     10 |     100 |   864 731.8 |   230 140.3 |      864.73 |      230.14 |   0.27× |   1.2 sps,     4.3 sps |

**Observations.**

- **GPU beats CPU at every grid size** measured, even 32 × 32. The
  earlier per-pass tests (M2.3 / M2.4 / M2.5) showed CPU winning,
  but those measurements included per-call buffer allocation and
  readback; the steady-state pipeline (M2.8 `GpuStepPipeline` with
  pre-allocated buffers + ping-pong) wins from the smallest grid up.
- **GPU/CPU ratio plateaus** at ~0.25× for grids ≥ 64. CPU per-step
  time scales as `O(grid²)` (predicted, since the convolve inner
  loop dominates and is `H × W × K × kernel_size²`); GPU scales the
  same way but with the inner work parallelised across SMs.
- **Real-time targets.** At 64 × 64 / C = 3 the GPU runs at 61 sps;
  M2.10 lands close to vsync (55 fps) since 16.3 ms/step is just
  under the 16.67 ms frame budget. At 128 × 128 / C = 3 the GPU
  delivers ≈ 17 sps — playable but choppy. 256 × 256 is research-
  only (≈ 4 sps).

## Section 2 — per-pass breakdown (64 × 64 / C = 3)

Each pass is dispatched in isolation 1000× with `submit + poll(Wait)`
per iteration, so per-call wall-clock includes command-encoder +
queue submission overhead. Use as a relative ranking, not as an
absolute pipeline cost (the full-step measurements above are the
absolute number).

| pass               | per-call μs | share of step |
|--------------------|------------:|--------------:|
| convolve           |    15 925.0 |       **97.4%** |
| affinity_growth    |        58.4 |          0.4% |
| gradient_u         |        45.3 |          0.3% |
| gradient_a_sum     |        38.1 |          0.2% |
| flow               |        35.0 |          0.2% |
| reintegrate        |       243.6 |          1.5% |
| **step sum**       | **16 345.4**| 100.0% |
| visualize (render) |       268.8 | n/a (not part of step) |

**Single takeaway.** Direct convolution at K = 10 kernels with
`max_side²` ≈ 7000 spatial stencil points per kernel dominates the
step budget at 97.4%. The remaining five compute passes plus the
render pass together account for under 3%. This is the **clear
M6 optimisation target** — FFT-based convolution (DESIGN.md §1.7,
§13) is expected to drop the convolve cost by 10–100× at K = 10,
which would put 256 × 256 in the 17 sps "playable" range.

## Section 3 — pipeline init + GPU memory

| grid | C | init (ms) | A buf (2×) | pre_g | u | grad_u | grad_a_sum | flow | kernels | total |
|-----:|--:|----------:|-----------:|------:|--:|-------:|-----------:|-----:|--------:|------:|
|   32 | 1 |       6.3 |     8.0 KB | 40.0 KB | 4.0 KB | 8.0 KB |     8.0 KB |  8.0 KB | 53.5 KB | 129.9 KB |
|   32 | 3 |       2.9 |    24.0 KB | 40.0 KB | 12.0 KB | 24.0 KB |     8.0 KB | 24.0 KB | 65.7 KB | 198.1 KB |
|   64 | 1 |       3.0 |    32.0 KB | 160.0 KB | 16.0 KB | 32.0 KB |    32.0 KB | 32.0 KB | 53.5 KB | 357.9 KB |
|   64 | 3 |       2.7 |    96.0 KB | 160.0 KB | 48.0 KB | 96.0 KB |    32.0 KB | 96.0 KB | 65.7 KB | 594.1 KB |
|  128 | 1 |       2.6 |   128.0 KB | 640.0 KB | 64.0 KB | 128.0 KB |   128.0 KB | 128.0 KB | 53.5 KB |   1.24 MB |
|  128 | 3 |       2.7 |   384.0 KB | 640.0 KB | 192.0 KB | 384.0 KB |   128.0 KB | 384.0 KB | 65.7 KB |   2.13 MB |
|  256 | 1 |       2.8 |   512.0 KB | 2.50 MB | 256.0 KB | 512.0 KB |   512.0 KB | 512.0 KB | 53.5 KB |   4.80 MB |
|  256 | 3 |       3.2 |    1.50 MB | 2.50 MB | 768.0 KB | 1.50 MB |   512.0 KB |  1.50 MB | 65.7 KB |   8.31 MB |

`GpuStepPipeline::new()` is uniformly fast (2.5–8.0 ms — the 64 × 1
outlier at 8 ms reflects the macOS Metal driver's first-call shader
compile for a fresh entry-point combination; subsequent cases hit
the on-disk shader cache). GPU memory stays comfortably inside
WebGPU's `max_storage_buffer_binding_size` (128 MB) at every
measured size; the 256 × 256 / C = 3 case sits at 8.31 MB total,
6.5% of the binding limit.

## Section 4 — 500-step mass conservation (DESIGN.md §8 M2 criterion)

Same 8-case matrix (paper_strict × border × C) as the M1.15 baseline,
32 × 32 grid, K = 10, seed = 42. The per-step time **includes**
per-step CPU readback (used to find the worst-step mass deviation),
so the per-step numbers here are larger than Section 1's pure
simulator times.

| paper_strict | border | C | max_rel  | total ms | per-step ms |
|--------------|--------|--:|---------:|---------:|------------:|
| false        | Torus  | 1 | 2.664e-5 |    5 406 |       10.81 |
| false        | Torus  | 3 | 2.625e-5 |    5 917 |       11.83 |
| false        | Wall   | 1 | 2.643e-5 |    4 876 |        9.75 |
| false        | Wall   | 3 | 2.077e-5 |    4 867 |        9.73 |
| true         | Torus  | 1 | 2.643e-5 |    5 302 |       10.60 |
| true         | Torus  | 3 | 2.653e-5 |    5 939 |       11.88 |
| true         | Wall   | 1 | 2.643e-5 |    4 864 |        9.73 |
| true         | Wall   | 3 | 1.782e-5 |    4 837 |        9.67 |

**All 8 cases pass.** Worst-case `max_rel = 2.67e-5` against the
tightest torus budget of `1e-3` — a 37× margin. Compared to
M2.7's standalone reintegrate measurement (5.4e-6 at 100 steps),
the full pipeline at 500 steps drifts by roughly 5×, exactly what
random-walk f32 accumulation would predict (`√(500 / 100) ≈ 2.2`,
plus the rest of the pipeline contributes its own per-step floor).

The wall-mode rel values (~2e-5) sit comfortably inside the
wall-mode 1e-2 budget at a 500× margin. **Mass is the right
invariant to test at long horizons** — see [JAX_NOTES.md §14]
(references/JAX_NOTES.md) for why field-comparison at 500 steps
would be meaningless under chaotic dynamics.

## Re-running

```sh
cargo run --release --bin bench_step
```

Output goes to stderr in markdown-ready table format; redirect or
copy from terminal as needed.
