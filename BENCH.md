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

## Section 5 — drift vs grid size at 100 steps (M6.A.3)

Each cell is the worst-case `max_rel = |m(t) - m(0)| / m(0)` across
100 simulator steps from `seed=42`. 4 cases per grid
(`paper_strict=false`, Torus/Wall × C=1/3) keep the table readable.
Generator: `mass_conservation_1k::drift_vs_grid_size_100step`.

| grid | Torus C=1 | Torus C=3 | Wall C=1 | Wall C=3 |
|-----:|----------:|----------:|---------:|---------:|
|   32 |  4.121e-6 |  4.296e-6 | 4.189e-6 | 2.811e-6 |
|   64 |  4.327e-6 |  3.953e-6 | 4.327e-6 | 3.953e-6 |
|  128 |  4.258e-6 |  3.885e-6 | 4.258e-6 | 3.885e-6 |
|  256 |  4.258e-6 |  3.976e-6 | 4.258e-6 | 3.976e-6 |
|  512 |  4.327e-6 |  3.930e-6 | 4.327e-6 | 3.930e-6 |

**Key observation: drift is essentially grid-independent.** Every
row hovers at `≈ 4e-6`. The simulator's f32 accumulation error per
step is set by the per-cell arithmetic (a fixed-size growth-and-
reintegrate sequence per cell), so summing over more cells just
adds more iid-ish drift contributions that cancel out under the
relative-error normalisation. This rules out the "drift grows with
grid²" concern in DESIGN.md §1.4 and validates the M6.A.3 choice of
tiered step counts: a 50-step run at 512×512 carries the same
detection power as a 50-step run at 32×32.

## Section 6 — drift growth with step count (M6.A.3)

64×64 baseline at 1000 steps. Generator:
`mass_conservation_1k::baseline_64x64_1000step`. Compare against
Section 5 row "grid = 64" (same configuration, 100 steps).

| paper_strict | border | C | max_rel (1000 step) |
|--------------|--------|--:|--------------------:|
| false        | Torus  | 1 | 4.327e-5 |
| false        | Torus  | 3 | 3.940e-5 |
| false        | Wall   | 1 | 4.327e-5 |
| false        | Wall   | 3 | 3.930e-5 |
| true         | Torus  | 1 | 4.327e-5 |
| true         | Torus  | 3 | 4.150e-5 |
| true         | Wall   | 1 | 4.327e-5 |
| true         | Wall   | 3 | 4.157e-5 |

**Drift scales linearly with step count, not as the √N a pure
random-walk model would predict.** 1000 steps gives ~10× the drift
of 100 steps (4.3e-5 vs 4.3e-6), not the ~3.16× √10 prediction.
The per-step floor is `≈ 4.3 × 10⁻⁸` and stays constant across
grid sizes and step counts:

- 100 step: 4.3e-6 / 100 = 4.3e-8 per step
- 200 step: 8.6e-6 / 200 = 4.3e-8 per step (g128 / g256 cases)
- 500 step: 2.2e-5 / 500 = 4.4e-8 per step (g64 cases)
- 1000 step: 4.3e-5 / 1000 = 4.3e-8 per step
- 50 step: 2.1e-6 / 50 = 4.2e-8 per step (g512 cases)

The linear-not-√N scaling means the error per step is *biased*
(same sign every step) rather than zero-mean random. The bias is
small enough that even 1000 steps stays 23× under the Torus 1e-3
budget; M6 changes to the convolve / reintegrate path should
preserve this floor unless they intentionally trade precision for
throughput (e.g. f16 accumulation in a future FFT shader). The
regression-detection mechanism here is: if drift per step jumps
above ~10⁻⁷, the corresponding `mass_conservation_g*` test will
fail well before the simulator visibly misbehaves.

## Section 7 — full mass-conservation matrix at tiered step counts (M6.A.3)

Each grid runs the `paper_strict × border × C` matrix at the step
count chosen by `mass_conservation_1k::mass_conservation_g{N}`.
512 is restricted to `paper_strict=false` to cap the heavy-run
budget; 32 keeps the M1.15 1000-step baseline.

| grid | steps | cases | max max_rel | within Torus 1e-3 | within Wall 1e-2 |
|-----:|------:|------:|------------:|:-----------------:|:----------------:|
|   32 |  1000 |     8 |    4.210e-5 |  ✓ (23× margin)   |  ✓ (237× margin) |
|   64 |   500 |     8 |    2.170e-5 |  ✓ (46×)          |  ✓ (460×)        |
|  128 |   200 |     8 |    8.653e-6 |  ✓ (115×)         |  ✓ (1.15k×)      |
|  256 |   200 |     8 |    8.653e-6 |  ✓ (115×)         |  ✓ (1.15k×)      |
|  512 |    50 |     4 |    2.129e-6 |  ✓ (470×)         |  ✓ (4.7k×)       |

Total runtime for all five `mass_conservation_g*` tests (CPU
release, M1 mini, `--test-threads=1`): ~46 min. Adding the two
one-off tests (`drift_vs_grid_size_100step`,
`baseline_64x64_1000step`) brings the full `--include-ignored`
sweep to ~86 min.

## Section 8 — A.4.5 GPU regression tolerance (chaos amplification)

M6.A.4 split `gpu_pipeline_matches_m1_baseline_fixtures_c1` into one
test per grid (g32/g64/g128/g256, C=1, 10 step) and initially set
`rel < 1e-3` for all of them. The g256 case failed at
`max_rel = 1.136e-3`; investigation A.4.5 traced the cause to
**intrinsic chaos in the Flow-Lenia dynamics, not a GPU bug**, and
the tolerances below were chosen accordingly.

### Empirical chain (A.4.5 experiments, all `#[ignore]` tests
under `crates/flow-lenia-gpu/tests/diagnose_divergence.rs`):

1. **Experiment 4** (`m6a45_per_step_rel_growth_by_grid_c1`)
   — GPU vs CPU `max_rel` at every step 1..10 for grids 32–256,
   C=1, paper_strict=false, Torus.

   | grid | step 1 max_rel | step 10 max_rel | growth factor / step |
   |-----:|---------------:|----------------:|---------------------:|
   |   32 |       1.15e-5  |        1.93e-5  |               ×1.06 |
   |   64 |       9.95e-5  |        1.06e-4  |               ×1.01 |
   |  128 |       1.22e-5  |        2.04e-4  |               **×1.36** |
   |  256 |       1.40e-5  |        4.73e-4  |               **×1.48** |

   Step 1 rel is roughly grid-independent (~1e-5), confirming the
   per-cell per-pass error is the same regardless of grid. Step 2+
   amplification grows with grid (chaos signature).

2. **Experiment 3 / shader audit**
   — `grep` of all WGSL shaders found zero `atomicAdd`,
   `workgroupBarrier`, `workgroupUniformLoad`. Every thread writes
   one output cell with a deterministic per-cell reduction order, so
   the per-cell f32 result is bit-identical between any two grid
   sizes given the same inputs. Rules out "grid-dependent
   accumulation order" as a GPU implementation defect.

3. **Experiment 5** (`m6a45_cpu_lyapunov_by_grid_c1`)
   — CPU-only baseline + ε = 1e-6 perturbation, two trajectories
   diverge per the dynamics' Lyapunov rate. C=1 only, Torus only.

   | grid | step 1 max_rel | step 5 | step 10 | step 20 |
   |-----:|---------------:|-------:|--------:|--------:|
   |   32 |       6.0e-5   | 1.3e-5 |  3.4e-5 |  1.6e-4 |
   |   64 |       **8.5e-1** | 5.9e-1 |  6.1e-1 |  6.7e-1 |
   |  128 |       **8.8e-1** | 6.6e-1 |  6.2e-1 |  4.9e-1 |
   |  256 |       **9.7e-1** | 7.3e-1 |  6.1e-1 |  6.0e-1 |

   At grid ≥ 64 the ε = 1e-6 perturbation reaches O(0.8) within
   *one step* and stays saturated. This is the smoking gun: C=1
   Flow-Lenia is **strongly chaotic at non-trivial spatial extents**,
   contrary to the common "C=1 collapses the dynamics" intuition.
   The per-cell f32 add-order delta between CPU and GPU (≈ 1e-5,
   grid-independent) is amplified grid-dependently because the
   *dynamics itself* has grid-dependent Lyapunov.

4. **Experiment 6** (`m6a45_chaos_nondeterminism_g256_c1`)
   — Same g256 / C=1 / 10-step run repeated 5 times in one process,
   reading the same `max_rel` each pass.

   | min | max | mean | std | max/min |
   |---:|---:|---:|---:|---:|
   | 1.136e-3 | 1.136e-3 | 1.136e-3 | 0.0 | **1.0000×** |

   The GPU pipeline is fully bit-deterministic across re-runs in one
   process, so the tolerance margin only needs to absorb future M6
   numerical-path drift, not run-to-run noise.

### Tolerance choice

Based on Experiment 6's exact determinism, a 2–5× safety margin
over the deterministic 10-step rel is sufficient.

| grid | measured rel (step 10) | tolerance | margin |
|-----:|-----------------------:|----------:|-------:|
|   32 |              3.6e-5    |    1e-4   |   2.8× |
|   64 |              1.1e-4    |    5e-4   |   4.5× |
|  128 |              4.5e-4    |    1e-3   |   2.2× |
|  256 |              1.1e-3    |    2.5e-3 |   2.2× |

Encoded as the per-test `rel_tolerance` argument inside
`crates/flow-lenia-gpu/tests/m1_regression_gpu.rs`. Reads: if M6
work pushes the GPU's 10-step output more than ~2× farther from CPU
than the current `case_g{N}_psX_btX_c1.bin` baseline, the
corresponding `gpu_field_regression_g{N}` will catch it.

### What this finding implies beyond M6.A

The "Flow-Lenia is chaotic at C=1 / grid ≥ 64" result is
material to:

- **M6.C** — convolve / reintegrate rewrites can't be validated by
  field comparison at chaotic scales; they need the mass + sanity
  layer of the regression pyramid (see CPU §5–§7).
- **M5** — the "same seed produces different visible creatures
  under tiny pipeline changes" symptom isn't a bug, it's chaos.
  Useful framing for the eventual flow-lenia.com FAQ.
- **Future research** — most prior Lenia work implicitly assumes
  small-grid stability transfers to large grids. The g32 / g64
  Lyapunov gap here suggests that assumption needs a closer look.

## Section 9 — A.6 perf-regression baselines and machine-state drift

`crates/flow-lenia-gpu/tests/perf_regression.rs` anchors a small
8-case `(grid, channels)` table to step-rates that any subsequent
M6.C commit must stay within ±20 % of (the ±5 % warning is an
early-warning band). The baselines look noticeably lower than the
M6.0 Section 1 numbers, and that's intentional — the test is
*intended* to detect commit-to-commit drift, not to enforce
fresh-machine throughput. The table below documents the
relationship.

| grid | C | M6.0 cpu sps (§1) | A.6 cpu sps | Δ | M6.0 gpu sps | A.6 gpu sps | Δ |
|-----:|--:|------------------:|------------:|---|-------------:|------------:|---|
|   32 | 1 |              80.8 |       69.14 | −14.4% |        159.5 |      117.67 | −26.2% |
|   32 | 3 |              76.6 |       64.90 | −15.3% |        129.5 |      106.28 | −17.9% |
|   64 | 1 |              19.2 |       17.16 | −10.6% |         75.3 |       55.35 | −26.5% |
|   64 | 3 |              17.1 |       15.90 |  −7.0% |         61.4 |       50.65 | −17.5% |
|  128 | 1 |               5.1 |        4.30 | −15.6% |         20.8 |       15.25 | −26.7% |
|  128 | 3 |               4.8 |        4.02 | −16.3% |         17.1 |       14.01 | −18.0% |
|  256 | 1 |               1.3 |        1.05 | −18.9% |          5.4 |        3.92 | −27.4% |
|  256 | 3 |               1.2 |        1.01 | −15.7% |          4.4 |        3.60 | −18.1% |

### Why the gap

Both columns came from the same host (Apple M1 mini), the same
Rust toolchain (1.95.0), the same wgpu (29.0.3) and the same
code path (`bench_step`-style timing — warmup, then a measure-
loop terminated by `device.poll(Wait)`). The only differences:

- **M6.0** was the first `bench_step` run after a clean session
  start. The machine was cool, no other CPU/GPU consumers were
  active, and no shader caches were yet populated.
- **A.6** ran inside a long M6.A session during which the GPU
  has been thermally seasoned by the M6.A.4 / A.4.5 / A.5 test
  suites (multiple hours of `device.poll(Wait)` and per-step
  readback workloads) and the host had accumulated background
  load from the M6 development tooling.

The slow-down splits cleanly by axis:

- **CPU sps** lands at 0.81–0.93 × the M6.0 number — typical
  background-load + thermal-cap dynamics on M1, with the worst
  case at large grids (more time, more thermal opportunity).
- **GPU sps at C = 1** is consistently 0.73–0.74 × M6.0 (very
  uniform across grid sizes), while **GPU sps at C = 3** lands
  at 0.82–0.83 ×. The bigger gap at C = 1 is consistent with a
  fixed per-step overhead growing relative to compute: at C = 1
  each step has less work, so overhead dominates more.

### What the test is actually checking

After re-anchoring, the regression test measures **commit-to-commit
drift** rather than absolute hardware throughput. An M6.C change
that genuinely makes GPU faster will register as a positive Δ on
the GPU sps column (above the ±5 % warning band) and on the
`gpu_sps / cpu_sps` ratio (above the ±30 % ratio band). An M6.C
change that accidentally slows the simulator down — kernel
recompile, lost LTO, unfortunate workgroup-size pick — registers
as a negative Δ.

Re-anchor when:

- The host hardware changes (different M-series chip, different
  thermal envelope, different driver version).
- An M6.C step intentionally shifts the baseline (e.g. the
  convolve FFT migration lands and the GPU sps numbers jump by
  3–4 ×; *that* run becomes the new baseline for downstream M6.C
  iterations).

`perf_regression_full_matrix` prints `cpu = …, gpu = …` numbers
ready to paste into the `BASELINES` table.

## Section 10 — A.7 WebGPU validation overhead

`crates/flow-lenia-gpu/src/validation.rs` adds a `ValidationGuard`
that installs `Device::on_uncaptured_error` and collects messages
into an `Arc<Mutex<Vec<String>>>`. Tests opt in via
`FLOW_LENIA_VALIDATE=1`; production callers (`flow-lenia-app`,
`flow-lenia-web`) never opt in and stay on the default zero-overhead
path. This section documents the perf cost of validation when it
*is* enabled, so M6.C developers know what they're paying for the
extra safety net.

### Method

`crates/flow-lenia-gpu/tests/perf_regression.rs` was run twice on
the same M1 host:

1. `cargo test --release -p flow-lenia-gpu --test perf_regression
   -- --include-ignored --nocapture` (validation off).
2. `FLOW_LENIA_VALIDATE=1 cargo test ...` (same command, validation
   guard installed on the test's `GpuContext`).

Both pass the test's 3-run-median variance mitigation, so the
single-run numbers below carry the same ±2-3 % run-to-run jitter
band as A.6.

### Measurement attempt #1 — host with trunk serve restarting in parallel

Single-pair (off vs on) against the M6.A.6 BASELINES, validation
on run with `trunk serve` bring-up happening concurrently on the
same host:

| grid | C | cpu off | cpu on | cpu Δ | gpu off | gpu on | gpu Δ |
|-----:|--:|--------:|-------:|------:|--------:|-------:|------:|
|   32 | 1 |   69.14 |  67.51 | −2.4% |  117.67 | 119.38 | +1.5% |
|   32 | 3 |   64.90 |  64.30 | −0.9% |  106.28 | 108.43 | +2.0% |
|   64 | 1 |   17.16 |  17.10 | −0.3% |   55.35 |  56.35 | +1.8% |
|   64 | 3 |   15.90 |  16.15 | +1.6% |   50.65 |  51.31 | +1.3% |
|  128 | 1 |    4.30 |   4.31 | +0.3% |   15.25 |  15.69 | +2.9% |
|  128 | 3 |    4.02 |   4.00 | −0.4% |   14.01 |  14.28 | +2.0% |
|  256 | 1 |    1.05 |   0.90 | −14% |    3.92 |   3.16 | −19% |
|  256 | 3 |    1.01 |   0.61 | −40% |    3.60 |   2.90 | −19% |

Grids 32–128 showed apparent overhead < 3 %, but 256 looked terrible.
The reviewer flagged this as "assumption-based dismissal of the 256
row" — fair, and verified next.

### Measurement attempt #2 — quiesced host (trunk stopped), thermal-matched off+on pair

After stopping `trunk serve` and running validation-on and
validation-off back-to-back (~25 min total), comparing the two runs
against each other (not against the cold-boot BASELINES) cancels
ambient drift to the extent possible without a real cooldown wait:

| grid | C | cpu off | cpu on | cpu Δ | gpu off | gpu on | gpu Δ |
|-----:|--:|--------:|-------:|------:|--------:|-------:|------:|
|   32 | 1 |   59.93 |  52.62 | −12.2% |  53.26 |  35.04 | −34.2% |
|   32 | 3 |   52.48 |  47.76 |  −9.0% |  45.05 |  56.37 | **+25.1%** |
|   64 | 1 |   12.03 |  11.56 |  −3.9% |  24.61 |  24.83 |  +0.9% |
|   64 | 3 |   12.04 |  12.07 |  +0.2% |  23.37 |  22.07 |  −5.6% |
|  128 | 1 |    3.62 |   3.24 | −10.5% |   8.70 |   7.20 | −17.2% |
|  128 | 3 |    3.33 |   2.95 | −11.4% |   9.31 |   6.07 | −34.8% |
|  256 | 1 |    0.90 |   0.79 | −12.2% |   2.67 |   1.80 | −32.6% |
|  256 | 3 |    0.85 |   0.76 | −10.6% |   2.65 |   1.74 | −34.3% |

### What the two attempts actually say

Both runs show **CPU sps drifting −10 % to −12 % between paired off
and on runs** — but the CPU simulator never touches wgpu, so
`Device::on_uncaptured_error` cannot slow it by even a percent. The
CPU column is therefore measuring pure host-state noise: a paired
~15 min off / ~15 min on cadence on this M1 mini shows ±10–12 %
run-to-run on CPU sps after three consecutive `perf_regression`
runs in the session (M6.A.6 commit + two A.7 attempts) push the
host into thermal accumulation that doesn't recover inside a 30 s
gap.

That noise floor sits on the GPU column too. The GPU Δ at 64×1 is
+0.9 % — well inside noise — and at 64×3 is −5.6 %, also inside
the CPU-derived ±12 % noise band. The larger 128 / 256 GPU drops
(−17 % to −35 %) could be real validation overhead, could be a
disproportionate thermal hit on larger compute work, or could be
both — the experiment as run cannot decompose them.

What is verifiable:

- **Validation overhead is at most O(machine-noise) under
  comparable conditions** — under the small-grid quiet
  point (64 / 32) the off / on gap is in the same range as the
  CPU-only noise floor.
- **The first attempt's "< 3 %" small-grid numbers were a
  best-case anchor**, not a contradiction: with the host less
  thermally loaded, validation cost is small enough to disappear
  into noise; with the host loaded, validation may compound the
  load, but we have not measured "validation only".
- **A cleaner overhead number would require a thermally-controlled
  rig** (cooldown gaps between runs, per-case isolation, more
  repetitions). That's research-level effort, out of scope for
  M6.A.7.

### Coverage scope

**Updated by M6.C-0** (= task #132 / M6.A.7.1 backlog completion).
`FLOW_LENIA_VALIDATE=1` now covers **43 of 47** tests reported by
`cargo test -p flow-lenia-gpu`. Per binary:

| binary | tests | reaches a validation guard |
|---|---:|:---:|
| `gpu_snapshot_regression` | 4 | ✓ (`common::test_ctx`) |
| `m1_regression_gpu` | 8 | ✓ (`common::test_ctx`) |
| `perf_regression` | 1 | ✓ (`common::test_ctx`) |
| `validation_smoke` | 1 | always-on (built-in guard) |
| `visualize_test` | 3 | ✓ (`common::test_ctx`) |
| `heap_regression` | 1 | ✓ (`common::test_ctx`, since M6.A.8) |
| `diagnose_divergence` | 6 | 4 ✓ (`common::test_ctx`) / 2 CPU-only (no GPU surface) |
| lib unit tests (`src/passes/*`, `pipeline.rs`, `kernel_buffers.rs`, `activation_buffer.rs`) | 23 | 21 ✓ (`validation::test_ctx_for_lib`) / 2 CPU-only (`gpu_kernel_meta_layout` size_of/align_of; `activation_buffer::flatten_unflatten_round_trip` pure CPU) |
| **total covered** | **43** | (every test that touches `wgpu::Device`) |
| **CPU-only (validation N/A)** | **4** | |

The previously-uncovered 29 (= 23 lib + 6 diagnose) were the per-pass
WGSL surface M6.C will rewrite — that gap is now closed. The 4
CPU-only tests exercise layout / flatten arithmetic with no
`wgpu::Device` use, so there is nothing for the validation callback
to surface.

**Helper architecture** (post M6.C-0): the integration-test side
uses `tests/common/mod.rs::test_ctx()`; the lib-unit-test side uses
`flow-lenia-gpu/src/validation.rs::test_ctx_for_lib()` (the
`#[cfg(test)] pub(crate)` counterpart). Both helpers share the same
8-line body intentionally — see `test_ctx_for_lib` rustdoc for the
decision rationale (single-source consolidation deferred until a
third consumer appears, per the `flow-lenia-testkit` note below).

### Interpretation (with the scope caveat above)

The two attempts above don't agree on a single overhead number,
and the disagreement traces to host-state drift between paired
runs, not to validation cost. What is defensible:

- **Small grids (32 / 64): cost likely small, with one
  attribution caveat.** Attempt #1 saw 32–128 stay inside ±3 %
  against a cold-state baseline; attempt #2 at grid 64 shows
  GPU Δ +0.9 % / −5.6 % — inside the paired CPU drift (−3.9 % /
  +0.2 %) of the same paired runs. Both data points are
  consistent with "validation cost ≤ small-grid CPU noise floor".
  *Caveat*: attempt #2 at grid 32 shows GPU −34.2 % / +25.1 %
  paired drift — the only positive Δ in the matrix and a large
  excursion in both directions; the table attributes this to
  host-state noise but an independent measurement at grid 32
  would be needed to confirm validation cost stays small there
  specifically.
- **Large grids (128 / 256): not measurable from these two
  attempts.** Attempt #2 shows GPU drops of 17–35 % at 128 / 256
  while the CPU column drops only 10–12 %. We don't have a
  physical model for why a 10 % CPU thermal hit would coincide
  with a 35 % GPU hit purely from thermal — that ratio might be
  driven by different DVFS curves on the Apple Silicon GPU vs.
  CPU side, or it might be real validation cost, or both. The
  experiment as run cannot decompose them. Treat the 35 % figure
  as the **upper bound** for "validation + the thermal cost of
  one extra ~15 min run", not as validation overhead alone.
- **Do not use `FLOW_LENIA_VALIDATE=1` together with
  `perf_regression` as a regression signal.** The
  perf-regression ±5 % warn / ±20 % err bands were anchored to
  the no-validation BASELINES; under host-state noise alone they
  can trip, and adding validation cost on top makes the signal
  worse. Run perf with validation off; run other integration
  tests with validation on if live coverage of new shaders is
  desired.
- **A thermally-controlled rig** (cooldown gaps between runs,
  per-case isolation, more repetitions) would be needed to pin
  down a true validation-only overhead at 128 / 256. Out of
  scope for M6.A.7.

The validation guard caught **zero errors across the integration-
test sweep** on the M6.A.7 commit — the integration-test surface of
flow-lenia-gpu is validation-clean. A future M6.C shader change
that trips the guard will surface immediately at the integration
test where the bad command is issued, with the wgpu `Debug` chain
(message + source location) in the panic output via
`ValidationGuard::assert_no_errors`. **As of M6.C-0**, the lib-unit-
test surface (21 of 23 tests; 2 are CPU-only) plus
`diagnose_divergence` (4 of 6 tests; 2 are CPU-only) are also
covered — a validation error in WGSL rewritten under M6.C-1+ will
now surface at the closest lib unit test (e.g.
`src/passes/convolve.rs::tests`) rather than only at
`m1_regression_gpu`, one step removed.

### Resolved footgun

`generate_gpu_snapshots` (M6.A.5) was `#[ignore]`-gated to prevent
accidental snapshot overwrites, but `--include-ignored` still
triggered it. The A.7 validation-on sweep tripped this once
(manifest.json date + commit_sha got rewritten — the snapshot
binaries themselves stayed bit-identical, as expected from A.4.5 /
A.5 GPU determinism). M6.A.7 adds a `FLOW_LENIA_REGEN_SNAPSHOTS`
env-var gate inside `generate_gpu_snapshots` so even
`--include-ignored` no-ops by default; intentional regenerations
now require `FLOW_LENIA_REGEN_SNAPSHOTS=1 cargo test ... --
generate_gpu_snapshots --include-ignored`.

## Section 11 — A.8 heap-leak regression (10 000 step, CPU heap only)

`crates/flow-lenia-gpu/tests/heap_regression.rs` runs a
`GpuStepPipeline` for 10 000 steps at 64×64 / C=3 / Torus,
brackets the loop with `peak_alloc::PeakAlloc` snapshots at three
points (baseline → mid-loop → post-loop), and asserts on both a
current-allocation delta (catches accumulating leaks in `current`)
and a peak drift between mid and post (catches slow leaks visible
only in `peak`, where a one-time transient would already have
settled by the mid sample).

### Methodology

1. 100-step warmup, drain wgpu queue, read baseline `current` and
   `peak`.
2. Run 5 000 steps, drain queue, read mid `peak`.
3. Run another 5 000 steps, drain queue, read post `current` and
   `peak`.
4. Assert `Δ current = post − baseline < CURRENT_DELTA_LIMIT_KB`
   (signed — leaks are positive growth).
5. Assert `peak drift = (post peak − baseline) − (mid peak − baseline)
   < PEAK_DRIFT_LIMIT_KB` — a one-shot transient settles by step 5 K,
   so a post-loop peak that climbs further is a slow growth signal.

The two `device.poll(Wait)` calls are deliberate — they let any
deferred drop-on-fence work run before each measurement so the
comparison is between fully-quiesced states.

### Tolerance derivation

| const | value | basis |
|---|---:|---|
| `CURRENT_DELTA_LIMIT_KB` | 500 KB | ~ 2 × the observed transient (270 KB). Halved from the M6.A.8 initial 1 MB after adversarial-reviewer flagged that 1 MB hid a 50 B/step leak across 10 K steps. The current 500 KB band catches anything > 50 B/step sustained while still absorbing the one-time wgpu drop/poll transient. |
| `PEAK_DRIFT_LIMIT_KB` | 256 KB | Discriminates transient (settles in first 5 K steps) from slow leak (keeps growing). Threshold sized for the per-half allocator noise band; tighter would trip on benign wgpu submission-ring resizing. |

### Detection floor

`CURRENT_DELTA_LIMIT_KB / N_STEPS_TOTAL` = 500 KB / 10 000 = **50
B/step**. A sustained leak below that average rate doesn't push
`Δ current` over the limit and slips through. To translate that
floor into the operational regime:

| leak rate | daily growth (60 fps `flow-lenia-app`) | when it surfaces in Activity Monitor |
|---|---:|---|
| 50 B/step (at floor) | 259 MB/day | first 8-hour session: ~ 86 MB |
| 25 B/step (½ floor) | 130 MB/day | first multi-day session |
| 5 B/step (1/10 floor) | 26 MB/day | days |
| 1 B/step (1/50 floor) | 5 MB/day | week+ |

The take-away: leaks the test *misses* are well below the rate
that would damage an interactive session. Anything operationally
material (≥ tens of MB per day of typical use) sits at or above
the floor and surfaces. The earlier M6.A.8 draft of this paragraph
claimed the test caught what it actually misses — corrected here
after the adversarial-reviewer pass.

### Measurement (M6.A.8 commit, M1 baseline)

Two runs documented: the **initial run** (1 MB tolerance, no
mid-loop sample) before the adversarial-reviewer pass, and the
**review-driven run** (500 KB tolerance, 5 K mid + 5 K post split)
after addressing the reviewer's concerns.

Initial run (commit-pre-review snapshot):

| reading | KB |
|---|---:|
| baseline current | 851.39 |
| post current | 1 122.09 |
| Δ current | **+270.70** |
| baseline peak | 1 248.20 |
| post peak | 2 390.25 |
| Δ peak | **+1 142.05** |

The +270 KB `Δ current` sits inside the 500 KB rewritten band.
The +1.1 MB `Δ peak` was the trigger for adding the mid-sample
discriminator — single-point peak measurement can't tell whether
that climb settled early or kept going.

Review-driven run with mid-loop sample (M6.A.8 commit, same host
thermal state):

| reading | KB | Δ from baseline |
|---|---:|---:|
| baseline current | 851.39 | — |
| mid current | 1 122.09 | +270.70 |
| post current | 1 122.09 | +270.70 |
| baseline peak | 1 248.20 | — |
| mid peak | 2 390.25 | +1 142.05 |
| post peak | 2 396.25 | +1 148.05 |
| **peak drift mid → post** | | **+6.00** (limit ±256) |

Two observations that the single-point measurement could not have
produced:

- **`current` reached steady-state by step 5 K** (mid = post,
  both 1 122.09 KB). The transient is fully held by step 5 K, no
  ongoing per-step Rust allocation in the back half.
- **Peak climbed +1 142 KB in the first half and only +6 KB in
  the second.** This is the transient-vs-leak discriminator
  paying off: the +1.1 MB peak observed in the initial single-
  point run is now confirmed as a one-shot allocation that
  settles inside the first 5 K steps, not a slow accumulation.

Both assertions pass with margin: current Δ at 54 % of the 500 KB
limit, peak drift at 2 % of the 256 KB limit. An M6.C refactor
that introduces even a 100 B/step leak (1 MB across 10 K steps)
would trip the current assertion well before any RSS growth shows
up in interactive use.

### Why one grid is sufficient

Steady-state per-step Rust allocation in `GpuStepPipeline::step`
is grid-independent by construction: one command-encoder build,
six pass `record(...)` calls (no per-step bind-group reallocation
— all bind groups are pre-built in `GpuStepPipeline::new`, see
the M6.0 §3 bind-group audit), one queue submit, one ping-index
swap. None of these per-step allocations scale with grid. The
wgpu buffers that *do* scale with grid² are constructed once
inside `new()` (outside the timed loop) and freed when the
pipeline drops (outside the post-loop reading). A leak that
survives the per-step loop will surface at 64×64 just as it would
at 256×256, so the single grid choice rests on the architecture,
not on grid-specific empirical evidence. If a future M6.C lands
grid-dependent per-step allocation (unlikely but not impossible —
e.g. a working buffer sized to N_kernels × grid²), this test
would need a multi-grid extension.

### Wall-clock

The commit-time run measured **197 s** on a host thermally
degraded by three prior `perf_regression` sweeps in the same
session (M6.A.6 commit + two A.7 attempts). A cold-boot
extrapolation from Section 1's `16.29 ms/step` × 10 000 steps
puts the theoretical lower bound near 165 s, with the 100-step
warmup + two drains adding under 5 s. No independent cold-boot
run was taken in M6.A.8 — the "~ 80 s" wording in an earlier
draft of this section was extrapolated from `bench_step`'s 1000-
step measure, not from a 10 K-step cold run, and was deleted.
The "M1 baseline" reference for §11 is therefore "the M6.A.8
commit thermal envelope", not "cold-boot M1".

### Scope: CPU-side heap only

`peak_alloc` wraps `std::alloc::System` and reports only Rust
allocations: `Vec`, `Box`, `String`, owned `Array3`, etc. **GPU
memory** (wgpu buffers, textures, command encoders) is managed by
the Metal driver outside the Rust allocator and does not surface
in `current_usage_as_kb()`. M6.A.8 deliberately leaves GPU-side
leak detection out (CLAUDE.md "Scope 制約" addition) — recoverable-
from-Rust leaks are cheap to catch in CI, while wgpu-side leaks
would typically appear as `Device::poll(Wait)` failures or test-
time OOMs that the other regression layers already trip on.
Manual GPU-memory monitoring via macOS Activity Monitor is the
M6.A.9 recipe for the GPU side.

## Section 12 — M6.A anchor for downstream comparison

M6.A closed with two distinct sets of perf baselines on file. This
section is the canonical record of which set anchors what.

### M6.0 cold-boot baseline (Section 1)

Captured in `BENCH.md §1` at the M6.A.0 commit (`aed05f0`), on a
freshly-restarted M1 mini with no other GPU/CPU consumers. These
are the **best-case** numbers — useful as a historical reference
("on a cold boot, the simulator can do this") but **not** the
baseline against which M6.C regression should compare. Running
`bench_step` immediately after a `cargo build` of a release binary
will reproduce them only if the host hasn't yet thermally
accumulated load from another perf run.

### M6.A.6 typical-development-state baseline (`perf_regression::BASELINES`)

Captured at the M6.A.6 commit (`f872f43`), with the same M1 mini
already in its M6.A working-session thermal envelope (i.e. after
M6.A.0-A.5 had run various test sweeps). These are the numbers
hardcoded into `BASELINES` inside
`crates/flow-lenia-gpu/tests/perf_regression.rs` and reproduced
here for the record:

| grid | C | cpu sps | gpu sps |
|-----:|--:|--------:|--------:|
|   32 | 1 |   69.14 |  117.67 |
|   32 | 3 |   64.90 |  106.28 |
|   64 | 1 |   17.16 |   55.35 |
|   64 | 3 |   15.90 |   50.65 |
|  128 | 1 |    4.30 |   15.25 |
|  128 | 3 |    4.02 |   14.01 |
|  256 | 1 |    1.05 |    3.92 |
|  256 | 3 |    1.01 |    3.60 |

### Which baseline applies when

| use case | baseline | rationale |
|---|---|---|
| `perf_regression_full_matrix` ±5/±20 % bands | **A.6** | regression detection wants commit-to-commit drift, not cold-boot delta. Anchoring to the typical dev state means the test trips on a real change, not on whether the host has been hot for ten minutes. |
| BENCH.md historical narrative | **M6.0** | reader wants to know "what was the cold-boot performance ceiling at M6 start". |
| M6.C "did the FFT migration speed us up?" | **A.6** | the question is "did this commit improve over the last commit", and the last commit was anchored to A.6. |
| M6.C "are we close to the 60 FPS goal at 512²?" | **new measurement at the M6.C step in question** | both M6.0 and A.6 are below 60 FPS at 256² (5.4 sps cold / 3.92 sps warm) and 512² isn't even in the BASELINES table, so the gap-to-goal question is dominated by whatever the current M6.C optimisation pass produces, not by which baseline you compare against. |
| post-M6.C re-anchor | **new run** | once an M6.C step lands a numerical-path change with positive Δ, that run becomes the new anchor. Update `BASELINES` + this section + §9 in the same commit. |

The Section 9 drift table (M6.0 → A.6 per-cell delta) remains the
explanation for why these two baselines differ; this section is the
operational mapping of "which one do I cite when".

## Section 13 — M6.A retrospective

This section is a single-stop summary of M6.A — what was built,
what was found, what the M6.B / M6.C / M5 inheritors need to know.
The narrative form complements the per-section measurements above;
read the relevant Section 5-11 for the raw numbers and Section 12
for the anchor mapping.

### Sub-step inventory

| sub-step | commit | one-line summary |
|---|---|---|
| M6.A.0 | `aed05f0` | `default-run = "native_gpu"` on flow-lenia-app |
| M6.A.1 | `778aa24` | extend regression fixtures to grids {32, 64, 128, 256} |
| M6.A.2 | `f0fe683` | split m1_regression into one #[test] per grid |
| M6.A.2.1 | `4ee0079` | mark m1_regression_g128 / _g256 as #[ignore] |
| M6.A.3 + A.10 + A.11 | `7f2d5d9` | mass conservation g32-g512 tiered + creature-alive sanity |
| M6.A.4 + A.4.5 | `fb9ccdb` | GPU regression per grid + chaos-amplification root-cause |
| M6.A.5 | `f5c5fe3` | GPU snapshot regression for pre/post M6.C comparison |
| M6.A.6 | `f872f43` | perf_regression ±5 % warn / ±20 % err + GPU/CPU ratio |
| M6.A.7 | `7053166` | WebGPU validation error assertion (test-only opt-in) |
| M6.A.8 | `4efec2e` | native heap leak 10k-step regression (CPU heap only) |
| CLAUDE.md measurement protocol | `7b71816` | paired run / quiesced / honest framing from A.6/A.7/A.8 |
| M6.A.9 | *this commit* | BENCH §12/§13 + README regen procedures + DESIGN Rev.4.6 |

Backlog (logged but deferred past M6.A close):

- **M6.A.7.1** (task #132): ~~extend `ValidationGuard` coverage to
  the 23 lib unit tests in `src/passes/*` + `src/pipeline.rs` and
  to `tests/diagnose_divergence.rs`~~ — **completed in M6.C-0**
  (post-M6.B, pre-M6.C-1 milestone). Coverage now 43 of 47;
  see §10 Coverage scope above.

### Key empirical findings

1. **C=1 dynamics is strongly chaotic at grid ≥ 64** (Section 8).
   ε = 1e-6 initial perturbation saturates to O(0.8) within one
   step at g64 / g128 / g256, against the common "C=1 collapses
   the chaos" intuition. Operationally: GPU vs CPU field rel
   scales with grid² (not from any GPU implementation defect, but
   because the same per-cell f32 add-order delta gets amplified by
   the dynamics). M6.C field-level tolerance must be grid-tiered;
   `rel < 1e-3` holds through 128 (measured 4.5e-4) but breaks at
   256 (measured 1.136e-3). The committed tiered tolerances are
   1e-4 / 5e-4 / 1e-3 / 2.5e-3 for g32 / g64 / g128 / g256.
2. **GPU pipeline is bit-deterministic across processes and days**
   (Section 8 Experiment 6 + M6.A.5 + M6.A.7 generate_gpu_snapshots
   round-trip). Same code → byte-identical snapshot binaries. This
   is what makes `gpu_snapshot_regression` viable as a regression
   anchor; any non-byte-identical regeneration is a real signal.
3. **Cold-boot vs warm-state perf differs 7-27 %** (Section 9).
   M1 mini in a working M6 session accumulates thermal load that
   the M6.0 cold-boot numbers can't represent. Section 12's
   anchor-mapping rule (M6.C uses A.6, not M6.0) follows directly.
4. **Validation overhead is undecomposable from thermal noise at
   the M6.A measurement budget** (Section 10). Small grids (32-64)
   show off/on Δ inside the CPU-derived ±12 % noise floor — the
   data is consistent with "overhead ≤ noise floor" but does not
   prove "overhead is small". At 128/256 the paired off/on Δ on
   GPU sps is 17-35 %, while CPU sps drifts only 10-12 % between
   the same paired runs; that ratio cannot be attributed to
   validation alone without a thermally-controlled rig. Treat as
   "not measured" for grids ≥ 128, not as "free". Coverage as of
   M6.C-0 is **43 of 47** (4 tests are CPU-only and have no GPU
   surface to validate); see §10 Coverage scope for the per-binary
   breakdown.
5. **Steady-state Rust allocation in the GPU step path is grid-
   independent** (Section 11). 10 K steps at 64×64 / C=3 leave
   `current` flat after step 5 K; peak's +1.1 MB transient settles
   in the first half (+6 KB drift in the second). The detection
   floor is 50 B/step (CURRENT_DELTA_LIMIT_KB / N_STEPS_TOTAL); a
   sustained 50 B/step leak would consume 259 MB/day at 60 sps
   interactive use, so the test catches anything operationally
   material.

### Methodology that's now standing infrastructure

The patterns below were developed during M6.A and are intended to
carry forward unchanged through M6.B / M6.C / M5. CLAUDE.md
"レビュー手順" / "Scope 制約" / "測定プロトコル" sections are the
canonical text; this list is the index.

- **5-layer numerical regression** (chaos test strategy memory):
  bit-equal CPU / mass / CPU-GPU C=1 short / GPU pre-post / sanity.
  Each layer has its own tolerance scenario; uniform "rel < X" is
  not a viable model for chaotic dynamics at large grids.
- **Subagent review workflow** (CLAUDE.md "レビュー手順"):
  scope-guardian before implementation, adversarial-reviewer after.
  Phase 2 from A.8 onwards: subagent approve → commit + push
  without separate human pre-commit gate.
- **Paired-run measurement protocol** (CLAUDE.md "測定プロトコル"):
  off/on same machine state, quiesced host, N=3 median, honest
  framing of noise-band vs signal, cold-boot vs warm-state
  distinction. Lifted from A.6/A.7/A.8 measurement struggles into
  a forward-acting rule.
- **Tolerance derivation from observation, not from intuition**
  (A.7 round 1, A.8 round 1): tolerances are sized at small
  integer multiples of measured transients, with the detection
  floor documented in the same place as the constant. "1 MB feels
  safe" is not a derivation; "2 × observed 270 KB transient gives
  500 KB band, floor 50 B/step" is.

### Hand-off to M6.B / M6.C / M5

**For M6.B literature survey** (next sub-stage, ~1 week):
- Read BENCH §2 per-pass breakdown: convolve at 97.4 % is the
  unambiguous M6.C target.
- Read §8 chaos finding before assuming "FFT vs direct match
  bit-perfectly". They will not; the M6.C field-level regression
  must be A.4.5-tolerance-aware.
- The literature survey itself decides the M6.C algorithm choice
  (FFT family, input-format optimisation, batching, GPU
  primitives, MPS-style reference). Bookmark target is
  `docs/M6_literature_survey_draft.md`, created in a follow-up
  commit (M6.B preparation, not part of M6.A.9).

**For M6.C per-pass optimization** (after M6.B):
- **Pre-condition completed in M6.C-0** (post-M6.B, pre-M6.C-1):
  M6.A.7.1 lib-unit-test + diagnose_divergence validation coverage
  extension landed. A validation error in WGSL rewritten under
  M6.C-1+ now surfaces at the closest lib unit test
  (`src/passes/convolve.rs::tests` etc.) instead of only at
  `m1_regression_gpu`. See §10 Coverage scope (updated table) for
  the 43 / 47 breakdown.
- Each M6.C-N sub-step must pass `m1_regression_g*` (CPU bit-
  equal), `mass_conservation_g*` (5-layer), `gpu_field_regression_g*`
  (A.4.5 tiered), `gpu_snapshot_regression` (A.5 pre/post), and
  not regress `perf_regression` by more than ±20 %.
- The convolve FFT migration is expected to land a 3-4 × GPU sps
  jump; treat *that* commit as the new perf anchor (see §12
  "post-M6.C re-anchor" row).

**For M5 evolutionary search** (after M6.C completes):
- The "same seed produces different visible creatures under tiny
  pipeline changes" symptom is the chaos finding from §8 in
  visible form, not a bug. Useful framing for an eventual
  flow-lenia.com FAQ.
- Mass conservation is the right invariant for long-horizon
  validation under chaos (the field comparison is meaningless past
  the Lyapunov saturation timescale).

## Section 14 — M6.C-1 retro + Stage 1 中間評価入力 (Phase 3 改訂条件 1)

M6.C-1 (WGSL FFT 実装) 全体完了の retrospective。**Ponyo877 さん
Stage 1 中間評価判断のための input section** — 戦略判断 (撤退 / 継続 /
縮小 / 目標再評価) は Ponyo877 さん責任、本 section は Claude Code
からの measured data + assumptions の honest summary に留める。

### Sub-step inventory (commit SHA)

| sub-step | commit | 内容 |
|---|---|---|
| C-1-1 | `cf61b92` | 1D Cooley-Tukey radix-4 FFT primitive (N=256 固定) |
| C-1-2 | `8363a5f` | 2D RFFT separable + 動的 N {64, 256} + GPU 完結 inverse |
| C-1-3 | `24cec08` | kernel pre-FFT 永続化 + spectral multiply pass |
| C-1-4-a | `6f27df1` | ConvolveFftPass primitive (C=1 only) + scratch helpers |
| C-1-4-b | `91a59f7` | GpuStepPipeline ConvolveMode 統合 + 早期撤退ゲート PASS (8.697×) |
| C-1-5-a | `3da2359` | ConvolveFftPass multi-channel (C ≥ 1) + per-kernel routing |
| C-1-5-b | `cf61645` | ConvolveMode::Auto default + bench C=1/C=3 (8.2× / 8.7×) |
| C-1-6-α | `05b94ea` | bench_long_horizon_fft binary (horizon 10/50/100 sweep) |
| C-1-6-β | *this commit* | BENCH retro + DESIGN Rev.4.7 + Stage 1 input |

### 主要観察

1. **FFT 化 end-to-end speedup**: bench_fft_vs_direct paired-run N=3 median,
   N=64 K=10 Torus, quiesced state:
   - **C=1**: direct 13.31 ms/step (75.1 sps) → fft 1.62 ms/step (616.4 sps)、
     **ratio 8.206×**
   - **C=3**: direct 16.33 ms/step (61.2 sps) → fft 1.89 ms/step (529.9 sps)、
     **ratio 8.655×**
   - 当初 BENCH §13 line 924 predict "3-4×" を **2× 超過**、特に C=3 で
     direct の per-kernel × per-channel inner loop が FFT path の K kernels
     共有 + per-channel forward 1 回 / channel より C scaling 大幅悪い
   - **N=256 extrapolation**: per-pass breakdown 未測定 (C-1-6-β scope
     creep に該当、C-2 perf phase に defer)。BENCH §1 の N=256/C=3 direct
     230.14 ms/step (4.3 sps) に対し FFT で同 8.7× ratio を仮定すると
     ~26.5 ms/step ≈ 38 sps、ただし: (a) per-channel forward の C scaling
     は dispatch overhead でなく FFT compute cost (O(N² log N))、N が
     増えると total FFT cost に占める share 大 → ratio が下がる方向、(b)
     dispatch overhead は per-pass overhead で N と無関係 → small N で
     ratio 大、大 N で convolve compute が dominant で ratio 小、合 (a)+(b)
     で N=256 の actual ratio は **3-5×** が現実圏内 (M6.B literature
     survey §7.1 C-1 predict 3-4× と整合)

2. **long-horizon stability** (bench_long_horizon_fft、N=64 K=10 Torus
   single-trial diagnostic):
   - horizon **10 step で既に tolerance violation 開始**: C=1 random max_rel
     8.30e-4 vs A.4.5 tiered g64 = 5e-4 → **1.66× 超過**
   - horizon 50-100 で **chaotic saturation**: max_rel ≥ O(1)、direct と
     FFT は same Lyapunov attractor 上の different trajectories
   - per-step amplification factor (geom over horizon 10→100): 1.11-1.24×
     (random vs identical kernels で 約 1.1× 差、saturation 込み geom mean
     は representative value ではない)
   - **identical-kernels controlled experiment** (C-1-4-b S-2 deferred 解析):
     kernel parameter scaling は **寄与あるが dominant ではない**、FFT inject
     が主因 (random vs identical の max_rel は order 同等、saturation pattern
     同等)

3. **Layer 4 (snapshot regression) の意義 redefinition**:
   - Layer 4 snapshot は短期 horizon (≤ 5-10 step) でのみ meaningful、長期
     horizon は Lenia chaos の性質上 physically impossible
   - 本 sub-step では Direct path baseline を維持 (M6.A.5 で生成)、FFT path
     baseline は Stage 1 中間評価で Ponyo877 さん判断後 (採用 確定後) 再生成

4. **multi-channel + per-kernel source_channel routing** (C-1-5-a):
   - 案 a (WGSL indirect routing) 採用、case b (per-channel SM batch K×C
     dispatch) は overhead 増で却下
   - C=3 5-step direct vs fft max_rel 2.094e-4 (tolerance 5e-4 で 2.4×
     headroom、C=1 の 1.675e-4 より 25% 大、multi-channel coupling 影響)
   - per-channel forward × C 回 + copy_buffer_to_buffer × 2C の overhead
     はあるが、C=3 で end-to-end 8.7× speedup を達成

5. **Auto fallback (C-1-5-b)**:
   - grid 32/128/512 mixed-radix は FFT primitive 未対応、direct fallback
     で feature regression 回避
   - mixed-radix 対応は M6.C-1 後半 or M5 で判定 (今は scope out)
   - 既存 UI / test sweep を破壊せず、grid 64/256 で FFT 化を享受

### Stage 1 中間評価への input (Ponyo877 さん判断材料)

CLAUDE.md 撤退ライン "256×256×3×4creature で 30 FPS なら M5 へ"
= 33 ms/step。

**直接測定** (single-trial diagnostic、CLAUDE.md §測定プロトコル準拠の
paired-run + N=3 median は bench_fft_vs_direct のみ):
- N=64 / C=1 / K=10 / fft: 1.62 ms/step (616 sps)、目標 60 FPS 余裕度 10×
- N=64 / C=3 / K=10 / fft: 1.89 ms/step (530 sps)、目標 60 FPS 余裕度 8.8×
- N=64 / C=3 / direct: 16.33 ms/step (61 sps)、目標 60 FPS 余裕度 1.0×

**Amdahl extrapolation** (理論、未測定):
- N=256 / C=3 / direct (BENCH §1 line 35): 230.14 ms/step (4.3 sps)、目標
  60 FPS 余裕度 0.07× → unacceptable for production
- N=256 / C=3 / fft: **3-5× speedup 推定** = 46-77 ms/step (13-22 sps)、
  60 FPS 未達、撤退ライン 30 FPS (33 ms/step) も marginal
- M6.C-2 (kernel fusion + subgroup) で **追加 1.5-2× 期待** (M6.B literature
  survey §7.1)、N=256 で 23-50 ms/step (20-43 sps) → 撤退ライン clearに上

**4 creature 影響**:
- Plantec 2025 paper の multi-creature 表現方式 (additive channel か独立
  グリッド か) は M6.B survey §9.1 で **未確認** (本文 PDF 未読)
- 1 グリッド additive channel なら C=12 (3 ch × 4 creature) 相当 = per-channel
  forward FFT 12 回、per-step time ~ C scaling の 12/3 = 4× で 92-200 ms/step
  → 撤退ライン未達 (Stage 1 撤退 判断材料)
- 独立グリッド × 4 なら GPU pipeline 4 並列 = throughput 1/4 = 92-200 ms/step
  同じ
- M6.C-4 (4 creature 実装 + Stage 2 動作確認) で実測必須

**まとめ** (Claude Code 判断ではなく measured data の framing):
- N=64 / C=3 で FFT 化により 8.7× speedup 達成、60 FPS 目標 余裕度 8.8×
- N=256 / C=3 / 4 creature の Stage 1 ターゲットは extrapolation でも未達、
  C-2 fusion + C-4 4-creature 実測で正式判断
- Long-horizon (horizon ≥ 10) で direct と FFT は chaotic separation、
  これは Lenia の inherent dynamics、FFT path 固有の bug ではない
- Stage 1 判断 (撤退 / 継続 / 縮小 / 目標再評価) は Ponyo877 さん責任、
  C-1-6 commit + push 後 Phase 3 改訂条件 1 で Claude Web 送信

## Section 15 — M6.C-2-5 paired-run measurement (5 configs) + Stage 1 中間評価入力

M6.C-2 (kernel fusion + parameter map P infra) 完了直前の measurement。
**Stage 1 中間評価の核心 input section** — 戦略判断は Ponyo877 さん責任、
本 section は measured data + assumptions の honest framing に留める。

### 測定環境

- Apple M1 (Metal)、quiesced state (Ponyo877 さん trunk serve / cargo /
  browser 停止確認済)
- `bench_c2_configs` binary、CLAUDE.md §測定プロトコル準拠
- paired interleave (D F D F …)、N=3 trials median、warmup 20 step
- N=64: 100 measured steps、N=256: 50 measured steps
- K=10、Torus、seed=1729

### 測定結果 (median)

| # | config | ms/step | sps | FFT/Direct ratio | direct ms |
|---|---|---|---|---|---|
| 1 | N=64  C=1 fft | 1.574 | 635.4 | 8.507× | 13.390 |
| 2 | N=64  C=3 fft | 1.934 | 517.1 | 8.573× | 16.580 |
| 3 | N=256 C=1 fft | 5.180 | 193.1 | **36.495×** | 189.032 |
| 4 | N=256 C=3 fft | 6.861 | 145.7 | **33.917×** | 232.716 |
| 5 | N=256 C=3 4-creature localized | **6.840** | **146.2** | localized/constant 1.063× | — |

各 trial の variance は ±5-10% (例: config 1 ratio = 9.17 / 8.51 /
8.42×)、median で吸収。

**config 5 の 1.063× overhead の読み方** (paired-ratio vs 絶対値):
1.063× は paired interleave の per-trial `localized/constant` ratio の
median (bench_c2_configs.rs)。一方 summary 表の config 4 (6.861) と
config 5 (6.840) は **独立 median** で config 5 が僅かに速く見えるが、
両者は thermal noise band 内 (差 0.3%)。6.3% overhead は interleaved
pairing から得た値で、独立 median の表差分とは別物 (同一 trial 内で
localized が constant より遅い分を捉えている)。

### 主要観察

1. **Stage 1 撤退ライン圧倒的クリア** (config 5 = 核心):
   - 撤退ライン "256×256×C3×4creature で 30 FPS" = 33.3 ms/step
   - 実測 **6.840 ms/step (146 sps)** = 撤退ライン **4.87× 上回る**、
     60 FPS 目標 (16.7 ms) も **2.44× 上回る**
   - §14 の Amdahl extrapolation "N=256/C=3/fft 46-77 ms/step (13-22
     sps)、撤退ライン marginal" は **大幅に悲観的**だった (後述 2)

2. **N=256 FFT-vs-Direct ratio = 34× が §14 予測 (3-5×) を約 6.8-11× 超過**:
   - §14 は「N 増で convolve compute が dominant → FFT-vs-Direct ratio
     が下がる」と仮定 (line 974-978)、actual は **逆**
   - Direct は per-cell O(kernel_area × K × C)、N=256 で kernel ~33² ≈
     1089 cell × K=10 × C=3 の inner loop が catastrophic (232 ms/step)
   - FFT は O(N² log N × K)、log N scaling で N 増の劣化が緩やか
     (5.18 / 6.86 ms/step)
   - 結果 ratio は N=64 の 8.5× → N=256 の 34× へ **増加**
   - 教訓: §14 の extrapolation は Direct の kernel-area scaling を
     過小評価。実測が必須だった (extrapolation の honest framing の
     正しさを裏付け)

3. **C-2 perf micro-opt (C-2-1-a fused inverse + C-2-2 SM unroll) の
   end-to-end 効果 ≈ ゼロ (thermal noise band 内)**:
   - 方法論: Direct path は C-2 で不変 → 同一セッションの FFT/Direct
     ratio を §14 C-1 baseline ratio で割れば、Direct を anchor として
     cross-session thermal がキャンセル (ratio-of-ratios)
   - N=64 C=1: current 8.507× ÷ §14 8.206× = **C-2 speedup 1.037×**
   - N=64 C=3: current 8.573× ÷ §14 8.655× = **C-2 speedup 0.991×**
   - 両者とも ±10% thermal noise band 内 = **有意な end-to-end 改善なし**
   - 原因分析: C-2-1-a が削減した 11 dispatch/step の大半は
     `copy_buffer_to_buffer` (10 copies) + 1 transpose dispatch。Metal
     上で buffer copy は数 μs と安価で、Maczan 32-71μs/dispatch は
     compute dispatch の数字。FFT path は元々 dispatch-bound でなく
     compute-bound (forward/inverse FFT 自体) のため、dispatch 削減の
     限界効用が低い
   - **C-2 ratio gate (≥ 1.5× 順調 / < 1.5× 早期撤退検討) に対し、
     C-2-1-a + C-2-2 のみでは 1.0× = gate 未達**。ただし M6.B literature
     survey §7.1 の「1.5-2×」予測は kernel fusion + **subgroup reduction**
     (C-2-3、未実装) を含む想定。実装した micro-opt 2 つだけでは届かない
     のは整合的

4. **localized 4-creature overhead = 1.063× (6.3%)**:
   - config 5 (localized) vs config 4 (constant) 同 N=256 C=3
   - parameter map P (2.5 MB at N=256 K=10) read + ParameterFlowPass
     identity-copy dispatch の追加コストが 6.3% のみ
   - Plantec §3.1 parameter map P infra は実用上 negligible overhead
     で 4 creature を表現可能 (case δ paper-faithful 設計の妥当性裏付け)

### Stage 1 中間評価への input (Ponyo877 さん判断材料)

**撤退ライン判定** (CLAUDE.md "256×256×3×4creature で 30 FPS なら M5 へ"):
- **PASS 圧倒的** — 146 sps (6.84 ms/step) は 30 FPS の 4.87×、60 FPS の
  2.44×。M5 進行は measured data 上明確に正当

**最終ゴール (512×512×4creature 60 FPS) への含意**:
- 512 は radix-4 FFT 非対応 (SUPPORTED_N = {64, 256}、512 = 2^9 は
  power-of-4 でない)。現状 512 は Direct fallback = 推定 ~930 ms/step
  (256 Direct 232ms × 4× cells) で全く届かない
- 512 で 60 FPS を狙うには **mixed-radix FFT (radix-2 fall-out stage)**
  が必須。これは M6.C-1-2 で scope-guardian deferred、別 work item
- N=256 で 146 sps の余裕があるため、512 (4× cells) で FFT 化できれば
  naive には ~37 sps 圏内、最終ゴール 60 FPS は mixed-radix + C-2/C-3
  追加最適化で射程に入る可能性

**C-2 残 sub-step の限界効用評価**:
- C-2-1-a + C-2-2 が end-to-end ~0× だった事実から、残る
  C-2-1-b (forward H+V fusion) も同様に dispatch 削減系で限界効用低い見込み
- C-2-3 (subgroup reduction) は spectral multiply の reduction を
  subgroup intrinsics 化する別系統で、効果は未知数だが、**N=256 で既に
  撤退ライン 4.87× クリア済みのため緊急性は低い**
- mixed-radix FFT (512 対応) の方が最終ゴールへの寄与が大きい可能性

**まとめ** (Claude Code 判断ではなく measured data の framing):
- N=256/C=3/4creature = 146 sps、撤退ライン 4.87× クリア → M5 進行正当
- C-2 perf micro-opt (C-2-1-a/2) は end-to-end ~0× (FFT が dispatch-bound
  でないため)、ただし機能 (C-2-4 parameter map P) は 6.3% 安価で動作
- N=256 FFT-vs-Direct = 34× は §14 予測を大きく上回る正の驚き
- 最終ゴール 512 は mixed-radix FFT 必須 (現状 scope 外)
- Stage 1 判断 (M5 進行 / C-2 残継続 / mixed-radix 優先 / 目標再評価) は
  Ponyo877 さん責任、本 measurement を Phase 3 改訂条件 2 (早期撤退ゲート:
  C-2 ratio < 1.5×) で Claude Web 送信

## Section 16 — M6.C-2 retrospective + Stage 1 中間評価結果 (主目標達成)

M6.C-2 (kernel fusion + parameter map P infrastructure) milestone
close-out。**Stage 1 中間評価: 主目標達成**を Ponyo877 さんが 2026-05-28
判断、512 高性能エンジン (M6.C-3) へ continue 決定。

### Sub-step inventory (commit SHA)

| sub-step | commit | 内容 |
|---|---|---|
| C-2-2 | `3777394` | spectral multiply 2-cell loop unroll |
| C-2-4 戦略確定 | `fbc7ed2` | case δ paper-faithful 確定 + CLAUDE.md subagent verification retro |
| C-2-4-a | `afa7259` | parameter map P storage + build_for_patches + 4 unit tests |
| C-2-1-a | `2a6d026` | kernel fusion case c (fused inverse FFT + transpose-to-pre_g) |
| C-2-4-b | `566654c` | parameter_map → affinity_localized bridge test (4 creature) |
| C-2-4-c | `ff52f1c` | ParameterFlowPass identity-copy + M5 Eq. 8 hook |
| C-2-4-d | `5d9cc30` | AffinityMode::Localized 配線 + ParameterFlowPass step + 4 creature smoke |
| C-2-5 | `1e7009a` | bench_c2_configs 5-config paired-run + BENCH §15 |
| C-2-6 | *this commit* | retro + §16 + DESIGN Rev.4.8 + §14 extrapolation 修正 |

### Stage 1 中間評価結果: 主目標達成 ✅ (Ponyo877 さん 2026-05-28 判断)

CLAUDE.md 撤退ライン "256×256×3×4creature で 30 FPS なら M5 へ" に対し:
- 実測 **6.84 ms/step (146 sps)** = 撤退ライン (33.3ms) を **4.87×**、
  60 FPS 目標 (16.7ms) を **2.44×** 上回る (BENCH §15 config 5)
- 当初 M6.B Amdahl extrapolation (13-22 sps、§14) を **6-11× 上回る**

**主目標 (256×256×4creature×60FPS) 達成済み**と確定。

**主目標達成宣言の correctness caveat** (honest framing): 本宣言は
**性能 (FPS)** の達成であり、N=256 FFT path の **long-horizon 数値
同値性は未検証**。§14 obs 2 の long-horizon chaos divergence は N=64
のみ測定 (horizon 10 で既に A.4.5 tolerance violation 開始)。N=256
FFT path の数値正確性は **short-horizon** のみ検証済 (C-1-5-a で C=3
5-step direct vs fft max_rel 2.094e-4、§14 obs 4)。Lenia の inherent
chaos のため long-horizon の bit 一致は physically impossible (§14
obs 3、Layer 4 redefinition) であり、これは FFT path 固有の bug では
ないが、「主目標達成」は perf 達成であって long-horizon correctness
guarantee ではないことを明記する。N=256 long-horizon の chaos
amplification 実測は M6.C-3 で 512 5-layer test 拡張時に併せて anchor
予定。

### 主要 retrospective 観察

1. **C-2 perf micro-opt (C-2-1-a + C-2-2) の end-to-end 効果 ≈ ゼロ
   — 原因究明済**:
   - measured: N=64 C-2 speedup 1.037× (C=1) / 0.991× (C=3)、±10%
     thermal noise band 内 (BENCH §15 §3)
   - 原因: C-2-1-a が削減した 11 dispatch/step の大半は安価な
     `copy_buffer_to_buffer` (10 copies)、Metal 上で数 μs。FFT path は
     元々 **dispatch-bound でなく compute-bound** (forward/inverse FFT
     自体が支配的) のため dispatch 削減の限界効用が低い
   - これは CLAUDE.md 開発原則 1 (観察した現象は対症療法せず原因究明)
     に従った結論: 「micro-opt が効かない」を tolerance 緩和等で
     糊塗せず、compute-bound という構造的理由を特定
   - M6.B literature survey §7.1 の「C-2 で 1.5-2×」予測は **subgroup
     reduction (C-2-3、未実装)** 込みの想定で、実装した micro-opt 2 つ
     だけで届かないのは整合的

2. **N=256 FFT-vs-Direct = 34× が §14 extrapolation (3-5×) を約 6.8-11×
   超過 (想定外の正の発見)**:
   - §14 (line 974-978) は「N 増で convolve compute が dominant →
     FFT-vs-Direct ratio が下がる」と仮定、actual は **逆**
   - 計算量理論: Direct は per-cell O(kernel_area × K × C)、N=256 で
     kernel ~33² ≈ 1089 cell の inner loop が catastrophic (232 ms)。
     FFT は O(N² log N × K)、log N scaling で N 増の劣化が緩やか
     (6.86 ms)
   - **§14 extrapolation の誤り訂正** (下記「§14 訂正」):
     ratio は N=64 の 8.5× → N=256 の 34× へ **増加**する (§14 の
     減少仮定は誤り)。これは Direct の kernel-area scaling を §14 が
     過小評価したため
   - 教訓: extrapolation は実測で覆りうる。§14 が honest framing で
     「extrapolation (理論、未測定)」と明記していたのは正しい姿勢

3. **case δ paper-faithful infrastructure 完成 (C-2-4-a〜d)**:
   - Plantec 2025 §3.1 parameter map P (per-cell K-vector) を CPU build
     + GPU storage + AffinityGrowthPass localized (Eq. 7) + ParameterFlowPass
     (Eq. 8 M5 hook) で実装
   - localized 4-creature overhead = **1.063× (6.3%)** のみ (BENCH §15
     §4)。2.5 MB parameter map (N=256 K=10) read + identity-copy dispatch
     が実用上 negligible
   - Eq. 8 stochastic sampling は M5 hook として docs に specification
     明文化 (`docs/M6_C2_4_creature_design.md` §"M5 hook specification")

4. **honest framing 検証** (Ponyo877 さん明示要求):
   - 146 sps は誇張ではない: quiesced state (trunk serve / cargo /
     browser 停止確認済)、paired interleave D F、N=3 median、warmup 20、
     50 measured steps (N=256) の CLAUDE.md §測定プロトコル準拠測定
   - 各 trial variance ±5-10% を median で吸収、min/max range も §15 表に
     未記載だが bench stdout に出力 (再現は `bench_c2_configs`)
   - "4.87× クリア" は config 5 median 6.84 ms ÷ 撤退ライン 33.3ms の
     観測値であり long-term guarantee ではない (single quiesced session)

### §14 extrapolation の訂正

§14 line 969-978 の N=256 extrapolation は本 §15 実測で覆った:

| 項目 | §14 予測 (extrapolation) | §15 実測 |
|---|---|---|
| N=256/C=3/fft ms/step | 46-77 ms (13-22 sps) | **6.86 ms (146 sps)** |
| N=256 FFT-vs-Direct ratio | 3-5× (N 増で減少と仮定) | **33.9× (N 増で増加)** |

§14 の誤りは「N 増で FFT compute share 増 → ratio 減」という仮定。
実際は Direct の O(kernel_area × K × C) per-cell cost が N 増で
catastrophic に効くため、FFT (O(N² log N)) との ratio は **N 増で増加**。
§14 は「extrapolation (理論、未測定)」と honest framing していたため、
実測による訂正は想定内の運用 (CLAUDE.md 測定プロトコル §4)。

### M6.C-3 (512 高性能エンジン) への引き継ぎ

Ponyo877 さん戦略決定 (2026-05-28):「理論値の超高性能エンジンを完成
させてから M5 進化的探索へ」。256 で over-engineering だった
subgroup / mixed-precision を 512 で「60 FPS 達成に必要」に転用:

- C-2-3 (subgroup reduction) → M6.C-3-3 へ転用
- C-3 (mixed-precision) → M6.C-3-4 へ転用
- 512 = 2^9 は radix-4 非対応 → M6.C-3-1 で **mixed-radix FFT
  (radix-4 × 4 + radix-2 × 1)** 実装が技術的核心
- 512 naive 外挿: N=256 6.84 ms → O(N² log N) で 4.5× = ~30.8 ms
  (32 sps)、60 FPS まで追加 1.85× 必要。deferred 手法積 (subgroup
  1.5-2× × mixed-precision 1.3-1.5× × workgroup tuning 1.2-1.5× =
  2.34-4.5×) で射程内

## Section 17 — M6.C-3-2 Stage 2 中間評価 (naive 512 mixed-radix FFT)

M6.C-3-1 (N=512 mixed-radix FFT primitive) を 2D + ConvolveFftPass +
pipeline に配線し (C-3-2)、**追加最適化前 (naive) の 512 性能**を測定。
512 高性能エンジン (最終ゴール 512×512×4creature×60FPS) の到達可能性
判定が目的。

### 測定環境

- Apple M1 (Metal)、`bench_c2_configs` (512 section 追加)、CLAUDE.md
  §測定プロトコル準拠 (paired/N=3 median/warmup 20)。512 は Direct が
  ~930 ms/step で実用外のため **FFT-only 測定** (Direct paired なし)
- N=512 は 50 measured steps

### 測定結果 (median、FFT mode)

| config | grid | C | mode | ms/step | sps |
|---|---|---|---|---|---|
| 6 | 512 | 3 | fft constant | 23.31 | 42.9 |
| 7 | 512 | 3 | fft **4-creature localized** | **24.68** | **40.5** |

(同セッションの 256 再測定: config 4 N=256/C=3 = 6.53 ms/153 sps、
config 5 localized = 6.64 ms/151 sps、§15 と整合。C-2 FFT speedup
再確認 N=64 = 1.007×/1.003× = noise band 内、§16 の ~0× 結論を再追認)

### Stage 2 中間評価 (Ponyo877 さん判断材料)

判定基準 (DESIGN Rev.4.8 §8 M6.C-3-2):
- ≥ 40 sps → subgroup + mixed-precision で 60 FPS 確実
- 30-40 sps → 全 deferred 手法必要、続行
- 20-30 sps → 1.85× 境界、慎重続行
- < 20 sps → mixed-radix FFT 実装に問題、要調査

**結果: config 7 = 40.5 sps → 最良バケット (≥ 40 sps)**。最終ゴール
60 FPS (16.7 ms) まで残り **1.48×**。

**1.48× 達成の見通し (honest framing — これは見積もりであり実測でない)**:
deferred 手法の高速化係数 (subgroup 1.5-2× / mixed-precision 1.3-1.5× /
workgroup tuning 1.2-1.5×) は **いずれも 512 で未実測**。M6.B 文献
survey の一般的レンジで、本プロジェクトの 512 spectral-multiply /
FFT に適用した時の実効値は C-3-3〜C-3-5 で測定するまで不明。さらに
C-2-1-a/C-2-2 が「FFT は compute-bound で dispatch 削減の限界効用低」
(§16) だった前例から、subgroup reduction も期待下限に留まる可能性は
ある。よって「1.48× は射程内の蓋然性が高い」が正確な表現で、確定では
ない。各手法適用ごとに実測し、届いた最高 FPS で確定する (案 a)。

### 主要観察

1. **mixed-radix FFT が naive で 40.5 sps 達成**: 512² Direct は
   ~930 ms/step (256 Direct 230ms × 4 cells) で全く不可能だった領域。
   radix-4×4 + radix-2×1 の mixed-radix が 512 を FFT 化し、512² で
   40 sps を実現
2. **512 naive 外挿の検証 (provisional)**: §16 で「256 6.84 ms → 512
   O(N²logN) で 4.5× = ~30.8 ms (32 sps)」と予測。実測 24.68 ms
   (40.5 sps) は予測より良い (256 localized 6.64ms の 3.7×)。考えられる
   要因は mixed-radix radix-2 段の効率性 or N=256→512 per-pass overhead
   増が予想より小だが、**これは N=3 median 1 データ点・Direct paired
   なし (Direct 512 が ~930ms で実用外のため)** なので provisional。
   C-3-3 以降の measurement で安定性を確認する
3. **localized overhead at 512 = 1.059×** (24.68/23.31): 256 の 1.062×
   と同等、parameter map P infra は 512 でも安価
4. **数値正確性**: 512 mixed-radix は 2D round-trip max_abs 4.8e-7、
   pipeline FFT-vs-Direct (C=1 2-step) max_rel 1.1e-3 (A.4.5 tiered
   512 tolerance 5e-3 内) で Direct と一致確認 (Layer 3 相当)

### 残り (C-3-3 以降)

40.5 sps → 60 fps は subgroup reduction (C-3-3) + mixed-precision
(C-3-4) + workgroup tuning (C-3-5) で 1.48× を獲得。各手法適用ごとに
512 FPS 測定、60 fps 達成時点で残り skip (案 a、届いた最高 FPS で確定)。

## Section 18 — M6.C-3 Stage 2 final (C-3-6 / C-3-3〜C-3-5 全 ≤1.1× → 届いた最高で確定)

C-3-3 per-pass breakdown による原因究明、C-3-4 mixed-precision、
C-3-5 reintegrate workgroup tiling を全 paired N=3 median で測定し、
**いずれも judgment B「<1.1× 即捨て」該当**でした。 Stage 2 最終 sps
は §17 中間値からほぼ変動なし。詳細経緯は `docs/overnight_log.md`
Entry 1-6。

### 測定環境

- Apple M1 (Metal)、`bench_512_reintegrate` (新規 focused bench)、
  CLAUDE.md §測定プロトコル準拠 (paired/N=3 median/warmup 20、
  Ponyo877 さん睡眠中の quiesced state、CPU で他プロセスなし)
- N=512 grid、ITERS=50 measured steps

### 結果 (median、FFT mode、N=3 paired-run)

| config | grid | C | mode | ms/step | sps | 60 FPS budget | gap |
|---|---|---|---|---|---|---|---|
| §17 baseline | 512 | 3 | fft constant | 23.31 | 42.9 | 16.67 ms | +6.6 ms |
| §17 baseline | 512 | 3 | fft 4-creature loc | 24.68 | 40.5 | 16.67 ms | +8.0 ms |
| §18 final | 512 | 3 | fft constant | **23.02** | **43.4** | 16.67 ms | +6.4 ms |
| §18 final | 512 | 3 | fft **4-creature loc** | **24.19** | **41.3** | 16.67 ms | **+7.5 ms (+45.2%)** |

§17→§18 の改善 (constant 42.9→43.4 sps, localized 40.5→41.3 sps)
は **C-3-3〜C-3-5 の合計** だが、いずれも noise band ±2% 内で
**観測される改善は thermal recovery / measurement noise の可能性が
高い**。手法別の効果は別途記録:

### Per-method effect (C-3-3〜C-3-5)

| 手法 | 期待 | 実測 ratio | 判定 |
|---|---|---|---|
| C-3-3 per-pass breakdown | (調査のみ、性能影響なし) | — | infra deliverable |
| C-3-4 f16 kernel_fft | ~1.18× | **1.019×** (29.84/30.42 µs total) | <1.1× 即捨て revert |
| C-3-5 reintegrate tiling | ~1.30× | **0.999×** (23.02/23.00 ms) | <1.1× 即捨て revert |

C-3-4 + C-3-5 のいずれも overnight session で実装+検証+測定が完了
したが、**判断 B (<1.1×) で revert**。

### judgment C (60 FPS 判定) 適用

ユーザー指示の閾値:
- ≥ 60 sps → 達成
- 50-60 sps → 実質達成、最高 FPS 確定
- 40-50 sps → 512 は 40+ fps で確定、深追いせず

**結果: 4-creature localized 41.3 sps → 40-50 バケット → 「40+ fps
で確定、深追いせず」**。

### Stage 2 final 確定

- **512×512×4creature×60FPS は未達** (41.3 sps、gap +45.2%)
- 届いた最高: **41.3 sps (24.19 ms/step)**、Constant mode は 43.4 sps
- 案 a (届いた最高で確定) に従い M6.C-3 を close、ここから先は
  M6.C-3 範囲では追わない

### 60 FPS への gap 残り解析 (honest framing)

`docs/overnight_log.md` Entry 2 の per-pass breakdown (real % 補正後):
reintegrate 51.5% + convolve 43.6% = 95% を 2 大 pass が占める。

C-3-5 で reintegrate 51.5% を 23× cache reuse で削減する設計は **M1
Apple Silicon の large L1 cache が既に gather pattern を吸収して**
おり、shared memory 経由でも同等 (Entry 5 詳細)。
C-3-4 で convolve 43.6% を mixed-precision で削減する設計は **本物の
speedup には FFT 全 intermediate buffer f16 化が必要** (kernel_fft
だけだと SM pass の 1ms にしか効かない) で overnight 範囲外。

つまり残る 1.45× は:
- discrete GPU では memory-bound 構造を最適化できる場面でも、Apple
  Silicon ではすでに architecture が cache 階層で吸収済み
- f16 全層化 (FFT intermediate を含む) という大規模 shader 改修が
  唯一の有効な path、これは M5 hook or 別 milestone 案件

Web target (Chrome WebGPU) で f16 が動作するかは未検証 (probe で
adapter feature SHADER_F16=true だが intermediate buffer f16 化に
SHADER_F16 feature が必要、これも overnight 検証外)。

### 残された手段 (future work、本 milestone close 後)

1. **FFT 内 mixed-precision** (channel_spectra / scratch_complex /
   k_spectra を u32-packed f16 で扱い、unpack2x16float で復元):
   想定 1.3-1.5× convolve speedup → total 1.13-1.22× → 47-50 sps
   bucket。実装は WGSL 5+ files、layout 全更新、tolerance 物理根拠
   再評価
2. **convolve FFT butterfly subgroup barrier elision**: stage_size ≤
   32 の butterfly stage で workgroupBarrier → subgroupBarrier に
   置換、想定 ~1.05× convolve → total ~1.02×。Chrome 限定 (案 P)
3. **larger workgroup for FFT** (512 → ? thread): 占有率改善、想定
   ~1.05×。ただし shared memory 増加で M1 register pressure 影響あり

これらは合算しても 1.3× 程度、60 sps (1.45×) に届かせるには **GPU
architecture を変える (M2 Ultra / discrete GPU) または algorithm
を変える (Direct path で大 kernel)** が必要、という physical
constraint がある。

### 主要観察

1. **C-3-3 breakdown insight**: 95% を reintegrate + convolve の 2
   pass で占める、という構造が明らかになった。これは M5 + 別 milestone
   で再アタックする時の出発点
2. **Apple Silicon architecture insight**: M1 の大 L1 cache が
   memory-bound pass の tile optimization を実質無効化する。これは
   discrete GPU では効くアプローチが Apple では効かない、という
   transferable な知見
3. **判断 B の有効性**: ≥1.2× 採用 / <1.1× 即捨て という Phase 3
   早期撤退ロジックで時間を溶かさず止められた (C-3-4 + C-3-5 合計で
   ~1h、各手法 30 分以内で判定)
4. **judgment C の有効性**: 60 FPS 未達でも届いた最高 (41.3 sps) で
   確定するルールが、深追いを防止して milestone を close できた

## Re-running

```sh
cargo run --release --bin bench_step
cargo run --release --bin bench_fft_vs_direct
cargo run --release --bin bench_long_horizon_fft
cargo run --release --bin bench_c2_configs   # 512 Stage 2 を含む
```

Output goes to stderr in markdown-ready table format; redirect or
copy from terminal as needed.
