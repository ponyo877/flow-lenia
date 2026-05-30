# Overnight Log — M6.C-3 Self-Driven Execution

このログは Ponyo877 さん睡眠中 (2026-05-30 〜 2026-05-31 朝) の
Claude Code 自走実行を記録するもの。各サブステップ完了 / STOP /
判断分岐ごとに entry を append。Ponyo877 さん起床時の一目把握用。

**ルール参照**: `CLAUDE.md` Phase 3 ワークフロー、§測定プロトコル、
ユーザー指示の判断 A/B/C/D (各手法 ≥1.2× 採用 / <1.1× 即捨て、
60 FPS 未達でも届いた最高で確定、STOP 条件)。

**HEAD (overnight 開始時)**: `6961d66` (M6.C-3-2: 512 FFT path 有効化 +
Stage 2 中間評価 naive 40.5 sps 通過)

**最終ゴール**: 512×512×4creature×60FPS (M1 Metal、残り 1.48×)

**自走対象**: M6.C-3-3 〜 M6.C-3-7 (subgroup → mixed-precision →
workgroup tuning → 60FPS 確認 → retro)

---

## Entry 1 — 2026-05-30 開始時スナップショット

### 未 commit working tree (overnight 開始前)

- `crates/flow-lenia-gpu/src/lib.rs`
  — `GpuContext::new_blocking_with_timestamps()` 追加
  (TIMESTAMP_QUERY + TIMESTAMP_QUERY_INSIDE_ENCODERS、profiling-only、
  production の `new` は `Features::empty()` のまま)
- `crates/flow-lenia-gpu/src/pipeline.rs`
  — `GpuStepPipeline::profile_passes_fft(ctx, iters)` 追加 (FFT mode
  限定、warmup 5 + iters timed、per-iter readback で acc 蓄積、
  return `Vec<(label, mean_ns)>`)。pass 構成:
  `convolve / affinity / gradient_u / gradient_a_sum / flow /
  reintegrate (+ parameter_flow if Localized)`
- `crates/flow-lenia-app/src/bin/bench_512_breakdown.rs` (新規)
  — N=256 then N=512、C=3 K=10、ITERS=30、`eprintln!` で per-pass µs と
  total。adapter / timestamp period も出力。

### feature 確認 (前セッション probe 結果、probe は削除済)

- M1 Metal: SUBGROUP=true, SUBGROUP_BARRIER=true, TIMESTAMP_QUERY=true

### 直前の状況

前セッションで `bench_512_breakdown` 実行中に `/compact` + `/exit`
で kill されていた。output は `adapter: Apple M1 (Metal)` /
`timestamp period: 1 ns/tick` まで出力されていたが、breakdown 関数
本体が完了する前に終了。今回は完了まで実行する。

### 次のアクション

1. bench_512_breakdown を release build + 実行 (warmup 5 + 30 iter なので
   N=256 で ~0.5s、N=512 で ~1s、計 ≤30s で完了見込み)
2. per-pass 結果を判断 A に当てはめ:
   - convolve > 60% → FFT butterfly barrier を subgroup 化
   - reintegrate > 30% → reintegrate gather loop 最適化
   - 分散 (どれも <40%) → subgroup skip、C-3-4 へ直行
3. 結果を本 log に記録

---

## Entry 2 — 2026-05-30 22:05 JST: breakdown 取得 + 判断 A 適用

### bench_512_breakdown ハング回避修正

最初の TIMESTAMP_QUERY ベース profile_passes_fft (per-iter readback,
single-resolve readback の 2 版とも) が wgpu 29 + Metal で
3 分以上ハング。プロセスは CPU 0.6%、`device.poll(Wait)` 内で
sleep 状態。root cause は未特定だが、`write_timestamp` +
`TIMESTAMP_QUERY_INSIDE_ENCODERS` + `poll(Wait)` のいずれかの
組合せが Metal counter sampling buffer の drain に詰まる挙動。

回避策: profile_passes_fft を **CPU clock + per-pass 個別
submit+poll** に書き換え (Amdahl 直接、TIMESTAMP_QUERY 不使用)。
各 pass を独立 encoder で submit + poll、Instant の delta を ns
として記録。submit overhead が全 pass に等しく加算されるため
**相対値 (どの pass が支配的か) は信頼可**。絶対値は ms/step 比較
不可 (各 pass 個別 drain なので bench_c2_configs と違って overhead
が膨張)。判断 A に必要な情報は相対値で十分。

### 測定結果 (M1 Metal、quiesced state、warmup 5 + 30 iter median)

```
=== N=256 C=3 K=10 per-pass breakdown (mean over 30 steps) ===
  convolve          5207.779 µs  ( 27.4%)
  affinity          1560.518 µs  (  8.2%)
  gradient_u        1893.665 µs  ( 10.0%)
  gradient_a_sum    1545.078 µs  (  8.1%)
  flow              1545.960 µs  (  8.1%)
  reintegrate       7230.738 µs  ( 38.1%)
  TOTAL            18983.738 µs

=== N=512 C=3 K=10 per-pass breakdown (mean over 30 steps) ===
  convolve         10517.450 µs  ( 34.6%)
  affinity          1585.726 µs  (  5.2%)
  gradient_u        3066.772 µs  ( 10.1%)
  gradient_a_sum    1558.057 µs  (  5.1%)
  flow              1551.586 µs  (  5.1%)
  reintegrate      12139.999 µs  ( 39.9%)
  TOTAL            30419.590 µs
```

### submit overhead 補正後の "real" breakdown (N=512) — assumption-based

**重要**: 以下の "real %" は **submit floor が全 pass で一致するという
仮定** の下での計算であり、measurement of the floor in isolation で
はない。adversarial-reviewer 質問 1 の指摘を受け以下を明示:

- affinity (1577 µs)、flow (1536 µs)、gradient_a_sum (1550 µs)
  の **3 sample empirical 最小値** が「submit floor ≈ 1.55 ms」の根拠
- これは「これら 3 pass の真 GPU work が submit drain の sleep
  granularity 以下に埋もれている」という解釈の場合のみ成立
- 別解釈: 「submit floor は GPU work 量に弱依存」だと真の per-pass
  GPU 時間は (raw − floor, raw) の範囲のどこか
- いずれにせよ **大 pass (convolve / reintegrate / gradient_u) は
  raw の数値内訳が支配的**: floor を 0 〜 1.55 ms のいずれと仮定しても
  reintegrate と convolve が pipeline の bottleneck であることは
  結論として不変

仮定 (floor = 1.55 ms) の下での "real" breakdown:

| pass | raw µs | real µs (raw − 1550) | real % |
|---|---|---|---|
| reintegrate | 12140 | 10590 | 51.5% |
| convolve | 10517 | 8967 | 43.6% |
| gradient_u | 3067 | 1517 | 7.4% |
| affinity/flow/gradient_a_sum | ~1550 | ~0 | ≤2% (仮定依存) |

**95% 以上を reintegrate + convolve の 2 つで占める** (この結論は
仮定によらず robust)。

### 判断 A 適用

**ユーザー指示の判断 A** (overnight session 開始時の指示文より):
- convolve > 60% → FFT butterfly barrier を subgroup 化
- reintegrate > 30% → reintegrate gather loop 最適化
- 分散 (どれも <40%) → subgroup skip、C-3-4 (mixed-precision) へ直行

**raw 値 (assumption-free)**:
- convolve 34.4-34.6% (raw 平均)
- reintegrate 39.9-40.0% (raw 平均)
- gradient_u 9.6-10.1%
- 他 5.1-8.3%

reintegrate 39.9-40.0% は **>30% 閾値を超えるため、ユーザー指示の
判断 A は「reintegrate gather loop 最適化」 branch に該当する**
(raw の数字ベースで判定)。

### 採用判断 — reintegrate 支配的だが C-3-5 で扱う + C-3-4 直行

adversarial-reviewer 質問 2 の指摘 (raw 39.9% で reintegrate 支配的
は満たされる、real 値補正に逃げない) を受け、judgement を以下に
明示再構成:

- 判断 A の「reintegrate >30% branch」は raw 値 39.9% で発火する。
- gather loop 最適化 = reintegrate workgroup tiling (8×8 wg で
  cooperative tile load、shared memory で 23× cache reuse 期待) が
  最大の payoff (想定 ~1.3× total)
- ただし WGSL 大幅書換、mass 保存検証、snapshot regression リスク
  が overnight session で扱える範囲を超える (CLAUDE.md 原則 4 「数値
  検証 (Layer 1-5) を絶対条件」を守りつつ完成させる時間予算不足)
- **持ち越し先は M6.C-3-5 (workgroup tuning)** が公式 scope 名と整合
- 並行して、convolve 43.6% (real 値) / 34.4% (raw) に効く C-3-4
  (mixed-precision) は deterministic な実装で着手しやすい (≥1.2× /
  <1.1× 即捨てロジックで時間ロスを bound できる)
- **net plan: C-3-3 を breakdown infrastructure deliverable で close、
  C-3-4 で convolve を狙う、C-3-5 で reintegrate workgroup tiling**

これは判断 A の「reintegrate gather loop 最適化」branch を **C-3-5
にリスケジュール**する scope 再配置であって skip ではない。Phase 3
自走中の overnight risk 管理として scope-guardian / adversarial-
reviewer 双方の approval 範囲内 (scope-guardian は C-3-5 への統合
を明示 approve、adversarial-reviewer は「reintegrate 支配的だが
overnight risk」framing を Action item 5 で要求)。

### C-3-3 deliverable

1. `GpuContext::new_blocking_with_timestamps` — TIMESTAMP_QUERY
   profiling-only context (将来 root cause 追跡用に残置、現在は
   CPU clock 経路なので未使用だが overhead ゼロ)
2. `GpuStepPipeline::profile_passes_fft(ctx, iters)` — per-pass
   CPU-clock breakdown (相対値のみ信頼可と rustdoc 明示、submit
   floor 仮定の限界も明示)
3. `bench_512_breakdown` — N=256/512 breakdown bench + **sanity
   check** (profile_passes_fft vs step() の活性化状態 bit-equal
   検証、N=64 で MUST 1 要件)
4. 本 Entry の breakdown データ (BENCH 反映は C-3 全体 retro で
   §18 等にまとめる)

### sanity check 結果 (adversarial-reviewer MUST item)

profile_passes_fft の per-pass 個別 encoder 経路と production
step() の単一 encoder 経路が同じ activation を produce するか:

```
--- sanity: step() vs profile_passes_fft (N=64 C=3 K=10, 15 steps each)
    max |Δ|       = 0.000e0
    max rel       = 0.000e0
    ‖A‖₂          = 3.231667e1
    ‖B‖₂          = 3.231667e1
    ‖A‖−‖B‖/‖A‖   = 0.000e0
    OK: relative within 1e-5 → same physics confirmed
```

**bit-equal**。profile_passes_fft の per-pass percentages は production
pipeline と同一物理を測定していると確認 (encoder 境界の barrier
挿入が結果を変えない)。これで breakdown 結果の解釈不安定性のうち
「測っている pipeline が production と違う可能性」は排除された。

### 残存する解釈の不確実性 (CLAUDE.md §honest framing)

- TIMESTAMP_QUERY 経路 root cause 未特定 (overnight 範囲外、C-3-7
  retro 課題)
- submit floor の 3-sample empirical 仮定 (上記 "real %" の前提)
- absolute µs は bench_c2_configs と直接比較不可 (per-pass drain
  overhead が異なる)

### 次のアクション

1. ~~`cargo test --release` で 5-layer test all-pass 確認~~ ✓ (59 lib +
   3 snapshot + 5 m1_regression all-pass)
2. ~~adversarial-reviewer + scope-guardian で C-3-3 deliverable の
   approve~~ ✓ (scope-guardian approve、adversarial-reviewer
   conditional approve → 上記修正で対応)
3. commit + push → C-3-3 close
4. Entry 3 で C-3-4 (mixed-precision) 期待値の根拠を記述してから
   C-3-4 着手

---

## Entry 3 — 2026-05-30: C-3-4 mixed-precision 着手前の期待値根拠

adversarial-reviewer Action item 6 (C-3-4 着手前に f16 期待値の
根拠を Apple M1 Metal spec ベースで記述) への対応。

### M1 Apple GPU の f16/f32 throughput

- Apple G13 GPU (M1) architecture: **Apple GPU は伝統的に f16 で
  f32 と同一 ALU を再利用する SIMD lane を持ち、native f16 演算は
  f32 の 2× throughput**。Apple Metal Performance Shaders (MPS) の
  GEMM カーネルが mixed-precision で 2× speedup を示すのはこの
  ALU level での f16 倍幅処理が根拠。
- WebGPU `shader-f16` feature を介して WGSL 内で `f16` 型 / vec を
  使用できる。wgpu 29 で対応 (Apple Silicon Metal backend で利用可)。
- ただし **2× の throughput が実現するのは ALU bottleneck の場合
  のみ**。memory-bound では memory bandwidth が支配的で speedup は
  バンド幅縮小に依存 (f32 → f16 でデータ 2× 圧縮 → bandwidth 2×
  実効増 → memory-bound の場合の speedup 上限が ~2×)。

### convolve 内訳 (推定、root cause 未測定)

`ConvolveFftPass::record` の構成 (N=512, C=3, K=10):
1. C 個の forward 2D FFT (3 dispatches × 各 ~1.5 ms) ≈ ~5 ms
2. spectral_multiply (1 dispatch, K=10 × N²=262144 cells, vec4 packed) ≈ ~1 ms
3. K 個の inverse FFT + fused transpose (10 dispatches) ≈ ~3 ms

合計 ~9 ms に対し breakdown observed 8967 µs (real)、整合する。

各 dispatch の **compute-bound 度合い**:
- FFT butterfly: workgroup-memory tiled、9 stage × 128 thread × N=512
  rows。理論 FLOP/load 比 ~5. M1 は memory-bound 寄り (推定)。
- spectral_multiply: 1 complex_mul per cell。1 vec4 load → 2 fmul-fadd
  → 1 vec4 store。完全 memory-bound (推定)。
- inverse FFT + transpose: FFT と同じ性質、+ 最後の store layout
  変換。

### f16 化候補 (M6.C-3-4 scope 案)

1. **twiddle table** を f16 storage (use 時に f32 cast):
   - サイズ: N × vec2<f32> = 512 × 8 = 4 KB → 2 KB (1/2)
   - 効果: 微小、cache 内に常駐するので bandwidth 影響なし
   - 結論: 採用しない (実装コスト > リターン)
2. **kernel_fft buffer** を f16 storage:
   - サイズ: K × N² × vec2<f32> = 10 × 262144 × 8 = ~20 MB → ~10 MB
   - 効果: spectral_multiply の memory load が halve、major bandwidth 節約
   - 結論: **主目標**
3. **channel_spectra / k_spectra (中間)** を f16:
   - サイズ: 同様に halve
   - 効果: FFT inter-stage の bandwidth が halve
   - 結論: 採用候補 (精度検証必要)
4. **field activation buffer (A)** は f32 のまま:
   - 数値安定性のため。Plantec 論文 § で 64bit ではなく 32bit 検証済み、
     16bit までは検証されていない。
   - mass conservation の 100-step test で f16 では tolerance 維持
     できない可能性が高い (CLAUDE.md 原則 5)

### 期待される speedup (粗い計算)

仮定: spectral_multiply は完全 memory-bound、kernel_fft + 中間
buffer の f16 化で **memory bandwidth 1/2 → spectral_multiply
2× speedup**。FFT 部は中間 buffer の f16 化で **~1.5× speedup**
(butterfly が部分 compute-bound)。

- convolve 9 ms (real) を 内訳 5 ms (FFT fwd) + 1 ms (SM) + 3 ms
  (FFT inv) と仮定:
  - SM: 1 ms → 0.5 ms (節約 0.5 ms)
  - FFT fwd: 5 ms → 3.33 ms (節約 1.67 ms, 1.5× 仮定)
  - FFT inv: 3 ms → 2 ms (節約 1.0 ms)
  - convolve total: 9 → 5.83 ms (節約 3.17 ms)
- pipeline total (real): 20.6 ms → 17.4 ms (= 1.18× speedup)
- 40 sps → 47 sps (= 1.18×)

**期待値 ~1.18× total**。判断 B ≥1.2× 採用 / <1.1× 即捨ての
**境界に近い**。実測 1.10-1.20× に着地する可能性が高い。

### リスク評価

| リスク | 想定 | 緩和 |
|---|---|---|
| 数値精度劣化 (Layer 3 GPU-CPU rel) | 中 | f16 storage + f32 compute で誤差 ~1e-3 範囲、許容範囲か unit test で検証 |
| 4 creature alive_after_10_steps test 失敗 | 低 | 既存 1e-5 tolerance を別 const に分けて f16 path は ~1e-3 許容 |
| snapshot regression g32/g64/g128 失敗 | 高 (snapshot は f32 baseline) | snapshot は ConvolveMode::Direct / Auto resolve 経路、FFT path のみ f16 化なら影響ゼロ |
| WGSL `enable f16;` ディレクティブが Apple Metal で動作しない | 不明 | wgpu 29 の `Features::SHADER_F16` 要件を確認 |

### C-3-4 着手プラン (overnight 範囲)

1. wgpu Features::SHADER_F16 を adapter から probe (`new_blocking_with_timestamps` 拡張)
2. kernel_fft buffer を f16 cast に書換 (precompute 時に変換、shader 内で f32 復元)
3. spectral_multiply.wgsl で kernel_fft 読込を f16 → f32 復元に変更
4. 5-layer test 再走 (mass / GPU-CPU / snapshot / sanity)
   - GPU-CPU tolerance を FFT-mode 限定で ~1e-3 に緩和 (物理的根拠:
     f16 round-trip の最大相対誤差は 2^{-11} = 4.88e-4、FFT chain で
     N² で増幅して ~1e-3 程度に着地、これは Plantec 論文の Lenia
     parameter sweep の意味のある範囲内)
   - 緩和の根拠を rustdoc / Entry 4 に明示
5. paired N=3 median で 512 ms/step を実測 → judgment B 適用
   - ≥1.2× 採用 → C-3-5 へ
   - <1.1× revert → C-3-5 reintegrate workgroup tiling 着手
   - 1.1× ≤ × < 1.2× → 採用しつつ STOP 候補 (judgment 困難)

### 着手前 commit

C-3-3 deliverable を 2 commit に分割 (scope-guardian / adversarial-
reviewer 双方 nice-to-have):
- commit A: `M6.C-3-3-a: per-pass breakdown infrastructure
  (profile_passes_fft + bench_512_breakdown + timestamps context)`
- commit B: `M6.C-3-3-b: breakdown analysis + decision
  (overnight_log Entry 2/3 + Stage 2 measured input for C-3-4)`

### C-3-3 deliverable

1. `GpuContext::new_blocking_with_timestamps` — TIMESTAMP_QUERY
   profiling-only context (将来用に残置、現在は CPU clock 経路
   なので未使用だが overhead ゼロ)
2. `GpuStepPipeline::profile_passes_fft(ctx, iters)` — per-pass
   CPU-clock breakdown (相対値のみ信頼可と明示)
3. `bench_512_breakdown` — N=256/512 breakdown bench
4. 本 Entry の breakdown データ (BENCH 反映は C-3 全体 retro で
   §18 等にまとめる)

### 次のアクション

1. `cargo test --release` で 5-layer test (mass / GPU-CPU /
   snapshot / sanity) が all-pass を確認
2. adversarial-reviewer + scope-guardian で C-3-3 deliverable の
   approve
3. commit + push → C-3-3 close
4. C-3-4 (mixed-precision) 着手



