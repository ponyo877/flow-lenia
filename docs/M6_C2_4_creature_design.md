# M6.C-2-4 4 creature 実装方式 — Plantec 2025 paper §3.1 + §4.3 study

**Status**: ⚠ **戦略判断要請** (新ルール 3、想定外発見)。当初 case α
(additive channel C=12) は **paper と不一致**、parameter map P
infrastructure 新規実装が paper-faithful 実装には必要。

## Paper 本文確認 (papers/2506.08569v1.pdf §3.1 + §4.3)

### §3.1 Flow-Lenia with parameters embedding

- **Parameter map** `P : L → Θ` where Θ ≡ ℝ^|K| (per-cell vector of
  kernel weighting `h`)
- Eq. 7 (with parameters):
  ```
  U_j^t(x) = Σ_{i=1}^{|K|} P_i^t(x) · G_i(K_i ∗ A^t_{c_0^i})(x) · [c_1^i = j]
  ```
  — 既存 Eq. 3 (`Σ h_i · G_i(...)`) の `h_i` を per-cell `P_i^t(x)` に
  置換、つまり **kernel weighting が cell ごとに variable**
- **重要な制約 (paper 直接引用、15 word 以下)**:
  > "changing the kernels parameters dynamically would make ... fast-
  > Fourier convolution impossible"

  → paper は **kernel parameters (a, b, w, r) は固定**、`h` weights のみ
  per-cell variable とすることで FFT 利用を可能にした。本 M6.C-1 FFT
  実装も この制約下で valid (kernel pre-FFT 永続化が OK)
- Eq. 8 (stochastic sampling for reintegration with parameters):
  ```
  P[P^{t+dt}(x) = P^t(x')] = exp(A^t(x') · I(x', x)) /
                              Σ_{x''} exp(A^t(x'') · I(x'', x))
  ```
  — parameter map が matter と一緒に flow、cell に到着した複数 source
  の parameter から softmax で 1 つ選択

### §4.3 Intrinsic evolution experiments (multi-species setup)

- **§4.3.2 Vanilla**:
  > "environment is simply initialized with **64 creatures**"

  → "creatures" = 64 個の **20×20 patch**、各 patch が異なる P vector
  (parameter set) で初期化
- "Matter concentrations (A) ... sampled uniformly in [0, 1] and
  parameter (P) is sampled following a normal distribution and **set
  identically for all cells in a patch**"
- "**3 channels** and 5 kernels per channel pair making a total of 45
  kernels"

  → **A は C=3 (固定)**、creatures は patches + P vectors で表現 (NOT
  channels)、creature 間競合は per-cell parameter selection (Eq. 8)
  で実現
- 500,000 steps simulation、interesting behaviors emerge ~100 steps

## 当初仮説 (scope-guardian C-2-4 approve 時) との不一致

scope-guardian は web fetch で paper を読んだとして "case α (additive
channel C=12 = 3 ch × 4 creature) は paper と整合" と approve した。
本 turn の **PDF 本文直接確認** で:

- **case α は paper と不一致**:
  - paper actual: A は C=3 単一 grid、P (parameter map) で creature 区別
  - case α 当初想定: 4 creature × 3 ch = 12 channels で creature 区別
  - case α は paper Eq. 7 の per-cell P semantics を実装せず、A の
    channel 数 拡張で代替するアプローチで paper-faithful ではない
- web fetch (前 scope-guardian round) は paper metadata のみで本文未確認、
  case α 確定は誤情報ベース
- **本 turn は paper PDF 直接読了 (papers/2506.08569v1.pdf:4-9)**、上記
  finding は paper §3.1 + §4.3 直接引用

## Paper-faithful 実装案 (case δ: parameter map P infrastructure)

case δ = paper Eq. 7 + Eq. 8 の正規実装:

1. **A は C=3 維持** (既存 ConvolveFftPass + 既存 ConvolvePass で動作)
2. **新規 parameter map P**: `array<f32>` 形状 `(H, W, K)`、per-cell に
   kernel weighting vector h を保持
3. **AffinityGrowthPass 改修**:
   - 既存: `U_j(x) = Σ h_i · G_i(...)·[c_1^i=j]` (constant h)
   - 新規: `U_j(x) = Σ P_i(x) · G_i(...)·[c_1^i=j]` (per-cell P from map)
   - WGSL `affinity_growth_localized.wgsl` (既存 file?) を有効化 / 改修
4. **ReintegratePass 改修**:
   - 既存: matter flow のみ
   - 新規: matter + parameter map P 同時 flow + Eq. 8 stochastic
     sampling for parameter selection
   - softmax sampling は per-cell に random number 必要 (stochastic)
5. **Initial state**: 64 (or N) patches × P vectors を patch ごとに
   uniform、grid 全体には P "0" default
6. **GPU memory**: P map = H × W × K × 4 byte、N=256/K=10 で ~2.5 MB
   追加 (acceptable)
7. **5-layer test 拡張**: parameter map embedding mode、既存 constant-h
   mode と並行運用

## 戦略判断要請事項 (Ponyo877 さん介在)

### 判断 1: case α (簡略実装) vs case δ (paper-faithful)

- **case α (当初仮説 = 簡略)**:
  - 実装工数: 1-1.5 日 (multi-channel infra 流用、既存 ConvolveFftPass
    の C=16 upper bound 内)
  - paper 不一致だが visual 的に 4 creature が動く
  - Stage 1 中間評価で perf 数値取得には十分
  - **paper Eq. 7 / Eq. 8 を実装しない**、Plantec evolution experiment
    再現は不可
- **case δ (paper-faithful)**:
  - 実装工数: 3-5 日 (parameter map P + AffinityGrowthPass 改修 +
    ReintegratePass stochastic sampling + WGSL 改修 + test 拡張)
  - paper Eq. 7 (per-cell h) + Eq. 8 (stochastic sampling) 完全実装
  - M5 evolutionary experiment への直結基盤
  - M6.C-2 工数 1 週間想定を **超過** (case δ で 3-5 日 + 既存 C-2-1/2/5/6
    で 3-4 日 = 計 6-9 日)

### 判断 2: M6.C-2-4 scope に case δ を含めるか defer か

- 案 (i): case δ を M6.C-2-4 内で実装 (工数 6-9 日、1 週間超過リスク)
- 案 (ii): C-2 では case α 簡略実装、case δ は新 milestone (M6.C-7 or
  M6.D parameter embedding) で別途
- 案 (iii): C-2 では parameter map P 実装スキップ、4 creature は無し、
  Stage 1 中間評価は C=3 single creature 数値で判断 (case α も skip)
- 案 (iv): C-2 では parameter map P **infrastructure のみ** 実装、
  AffinityGrowthPass per-cell h 対応、stochastic sampling Eq. 8 は
  defer (constant per-patch initial P で competition なし)。case δ
  の partial 実装で 4 creature は visual 的に動くが competition なし

### 判断 3: creature 数: 4 (Ponyo877 さん指示) vs 64 (paper default)

- Ponyo877 さん M6.C-1 計画書: "256×256×4creature 60FPS 主目標"
- Plantec paper actual: 64 creatures (3 ch、5 kernels/pair = 45 kernels)
- 4 = M6.C-1 計画の number (Ponyo877 さん戦略選定、Stage 1 評価対象)
- 64 = paper default、より dense な multi-creature dynamics
- どちらに合わせて Stage 1 中間評価を行うか

## Phase 3 改訂条件 trigger

- **条件 3**: 戦略判断 (想定外発見: scope-guardian の paper 案内不一致、
  paper Eq. 7/Eq. 8 が当初仮説より広い scope 必要)
- → **Ponyo877 さん経由 Claude Web 経由で戦略相談**

## 次の action

1. 本 doc を commit + push (doc-only、self-judgment OK)
2. Ponyo877 さんに状況報告 + 上記 3 判断要請
3. Ponyo877 さん判断を受けて C-2-4 scope を確定、その後 C-2-1-a / C-2-2 等
   並行可能な sub-step は自走着手

## paper 引用箇所 (re-reading 用)

- §3.1 line 1-10: "parameter map P : L → Θ where Θ ≡ ℝ^|K|"
- §3.1 line 11-14: "this would come with high memory and computational
  costs ... fast-Fourier convolution impossible"
- Eq. 7: per-cell P_i(x) version of Eq. 3
- Eq. 8: stochastic sampling for parameter inheritance
- §4.3.2 Vanilla: "64 creatures ... 20×20 square patch ... P sampled
  following a normal distribution and set identically for all cells in a
  patch"
- §4.3.1: "3 channels and 5 kernels per channel pair making a total of
  45 kernels"
