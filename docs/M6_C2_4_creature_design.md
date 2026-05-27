# M6.C-2-4 4 creature 実装方式 — case δ 採用 (paper-faithful + Eq. 8 defer to M5)

**Status**: ✅ **戦略判断確定** (Ponyo877 さん 2026-05-27)。
case δ (paper-faithful parameter map P infrastructure) 採用、Eq. 8
stochastic sampling は M5 に defer、creature 数 4 維持。

## 経緯

scope-guardian C-2 approve 時に web fetch で Plantec 2025 paper を
確認 (metadata のみ) → case α (additive channel C=12) approve。
本 study (M6.C-2-4a、commit `3fefd44`) で paper PDF 直接読了 → case α
は paper §3.1 + §4.3 と不一致が判明、Ponyo877 さん戦略相談 (Phase 3
改訂条件 3) trigger。

## 戦略判断確定事項 (Ponyo877 さん 2026-05-27)

### 判断 1: 実装方式 = case δ (paper-faithful)

採用理由:
- 当初志向「論文を読み込んで革新的」と整合
- Flow-Lenia の本質 (parameter inheritance) を実装する責任
- M5 進化的探索 (Eq. 8) への接続準備
- 工数差 2-3 日は M6 全体の数 % で値する投資

### 判断 2: M6.C-2-4 scope = parameter map P infrastructure のみ + Eq. 8 defer to M5

C-2-4 で実装:
- parameter map P infrastructure (per-cell vector size |K|)
- AffinityGrowthPass 改修 (constant h → per-cell P)
- ReintegratePass infra (matter + P 同時 flow の枠組み)
- 初期 state: 4 patches × P vectors

M5 に defer:
- Eq. 8 stochastic sampling (parameter inheritance during reintegration)
- creature 同士の parameter 競争メカニズム

理由:
- Stage 1 性能評価には P infrastructure があれば十分
- Eq. 8 は M5 進化的探索の核心、文脈として M5 で実装が自然
- 工数 3-4 日、M6.C-2 1 週間想定内

### 判断 3: creature 数 = 4 維持

理由:
- 計算量の予測可能性 (64 は性能 unknown)
- Visual の見やすさ
- SNS 公開時の見栄え
- 後で拡張容易 (slider 化等は M6.C-5 or M5)

## M6.C-2-4 サブステップ計画 (確定)

case δ + Eq. 8 defer に基づく:

### C-2-4-a: parameter map P storage + 初期化 (1 日)
- per-cell vector size |K| storage buffer (H × W × K × 4 byte)
- 4 patches で異なる P vector を初期配置
- 既存 ConvolveFftPass の `kernel_routing_buf` (K × u32 = source_channel)
  とは別 buffer、`parameter_map_p_buf` (H × W × K × f32) として
  AffinityGrowthPass に bind
- N=256/K=10 で ~2.5 MB、N=64/K=10 で ~160 KB

### C-2-4-b: AffinityGrowthPass 改修 (1 日)
- 既存 `affinity_growth_constant.wgsl`: U_j(x) = Σ h_i · G_i(...) · [c_1^i=j]
- 新規 `affinity_growth_localized.wgsl` (or 既存改修): U_j(x) = Σ
  P_i(x) · G_i(...) · [c_1^i=j] (Eq. 7)
- WGSL binding に新 `parameter_map_p: array<f32>` 追加、per-cell に
  K 個の f32 を読む
- AffinityGrowthPass に new mode (constant vs localized)、default
  constant で backward-compat

### C-2-4-c: ReintegratePass infrastructure 改修 (1-2 日)
- 既存: matter のみ flow (per-cell A → 別 cell)
- 新規: matter + P 同時 flow (per-cell A + P → 別 cell)
- Eq. 8 stochastic sampling は **M5 defer**、constant P で flow
  (= cell に複数 P 候補が arrive した場合、最初の 1 つ採用 or 単純
  average)
- M5 で Eq. 8 を実装する hook point を WGSL コメント + Rust API で明示

### C-2-4-d: 4 creature 動作確認 + visual smoke test (0.5 日)
- 4 patches × distinct P vectors で動作
- assert_creature_alive 全 creature 生存確認
- screenshot で 4 creature visual 確認
- M6.A.11 sanity check (creature alive) を 4-creature 拡張

**C-2-4 合計**: 3.5-4.5 日

## C-2-5 measurement 計画 (Stage 1 入力)

C-2-4 完了後の paired-run measurement (5 configs、CLAUDE.md §測定
プロトコル準拠):

| # | config | 主目的 |
|---|---|---|
| 1 | N=64 / C=1 / fft + C-2 | C-1 baseline からの C-2 ratio |
| 2 | N=64 / C=3 / fft + C-2 | C-1 baseline からの C-2 ratio |
| 3 | N=256 / C=1 / fft + C-2 | Stage 1 評価の核心 |
| 4 | N=256 / C=3 / fft + C-2 | 撤退ライン判定 |
| 5 | N=256 / C=3 / **4 creature with parameter map P** / fft + C-2 | Stage 1 中間評価の主要入力 |

5 番目は case δ infrastructure (Eq. 8 なし、constant P flow) で測定。

## Stage 1 中間評価への影響 (honest framing)

- case δ で per-step overhead +20-50% 推定 (AffinityGrowthPass per-cell
  P load + ReintegratePass P flow infrastructure)
- 当初 N=256/C=3/4creature 30 FPS 撤退ラインは case α 前提では marginal、
  case δ で **更に厳しくなる可能性**
- M6.0 期待 speedup 計算は case α 前提だった、case δ では撤退ライン到達
  も慎重判断
- C-3 (mixed-precision) 採用判断が Stage 1 で重要に
- 60 FPS 達成は **困難度上昇**
- Stage 1 中間評価で実測ベース正式判断

## 並行進行 (Ponyo877 さん承認)

C-2-1-a (kernel fusion case c) と C-2-2 (SM vec4 packing) は parameter
map P と独立、自走継続。

## Phase 3 自走復帰

判断 1-3 確定で C-2-4 scope 整理完了。C-2-1-a, C-2-2, C-2-4-a〜d を
Phase 3 自走実施。

次の Ponyo877 さん介在は M6.C-2 milestone 完了 (Phase 3 改訂条件 1)
= Stage 1 中間評価のみ。

## paper 引用箇所 (case δ 設計の primary source)

- §3.1 Parameter map P : L → Θ where Θ ≡ ℝ^|K|
- Eq. 7: U_j^t(x) = Σ P_i^t(x) · G_i(K_i ∗ A^t_{c_0^i})(x) · [c_1^i = j]
- 制約: "changing the kernels parameters dynamically would make ...
  fast-Fourier convolution impossible" → kernel parameters 固定、`h`
  weights のみ per-cell variable、FFT 互換性維持
- Eq. 8: stochastic sampling — **M5 defer**
- §4.3.2 Vanilla: "20×20 square patch ... P sampled following a normal
  distribution and set identically for all cells in a patch"

## 関連 commit

- `3fefd44` (M6.C-2-4a): Plantec paper PDF 直接読了、case α 不一致発覚
- 本 commit (M6.C-2-4a 戦略確定): Ponyo877 さん戦略判断確定 + 計画更新
