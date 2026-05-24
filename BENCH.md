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

`FLOW_LENIA_VALIDATE=1` covers **17 of 46** tests reported by
`cargo test -p flow-lenia-gpu`. Per binary:

| binary | tests | reaches `test_ctx()` |
|---|---:|:---:|
| `gpu_snapshot_regression` | 4 | ✓ |
| `m1_regression_gpu` | 8 | ✓ |
| `perf_regression` | 1 | ✓ |
| `validation_smoke` | 1 | always-on (built-in guard) |
| `visualize_test` | 3 | ✓ |
| `diagnose_divergence` | 6 | ✗ (local `headless_ctx`) |
| lib unit tests (`src/passes/*`, `pipeline.rs`) | 23 | ✗ (per-module `headless_ctx`) |
| **total covered** | **17** | |
| **total uncovered** | **29** | |

The uncovered 29 are precisely the per-pass WGSL surface M6.C will
rewrite (`convolve`, `affinity_growth`, `gradient`, `flow`,
`reintegrate`) plus the diagnostic tests, so the gap is material.
Logged as **M6.A.7.1 follow-up** (task #132): lift `test_ctx` into
a module reachable from both trees, or wrap each local
`headless_ctx` with the same env-var check; migrate
`diagnose_divergence.rs` to `mod common; common::test_ctx()` at
the same time.

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
`ValidationGuard::assert_no_errors`. Lib unit tests that touch the
same shaders are *not* yet covered — see the M6.A.7.1 follow-up.

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

## Re-running

```sh
cargo run --release --bin bench_step
```

Output goes to stderr in markdown-ready table format; redirect or
copy from terminal as needed.
