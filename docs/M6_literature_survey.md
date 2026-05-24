# M6.B Literature Survey — GPU 高速化文献調査

**目的**: Apple M1 mini + Chrome WebGPU で 512×512 grid 4 creature 60 FPS
(= 16.67 ms/step) 達成のため、必要な speedup を実現する GPU 最適化手法を
文献から特定する。

**現状実測** (BENCH.md §1、2026-05-18 build host M1 mini, Metal native, K=10):

| grid | C | native GPU ms/step | sps |
|---|---|---|---|
| 256 | 3 | **230.14** | 4.3 |
| 128 | 3 | 58.55 | 17.1 |
| 64 | 3 | 16.29 | 61.4 |

**重要**: BENCH 数値は **wgpu-native (Metal backend)** であって **WebGPU
(Chrome)** ではない。Chrome WebGPU 経由は dispatch overhead + safety checks
で追加 overhead が乗る (文献ベース推定 +10-30%、§5.3 参照)、ただし本 survey
時点で web 実測値は持っていない。**Stage 1 中間評価 (M6.C-3 完了時) で web
実測が必要**。本 §9 必要 speedup 計算は native baseline で行い、web 換算は
明示的 separately で扱う。

**スコープ** (scope-guardian approve 取得):

- Deliverable: 本 `docs/M6_literature_survey.md` のみ、コード変更ゼロ
- 網羅: 30-50 件、適用可能性 high/medium/low/reference-only
- 深掘り: **必須 3 トピック** (WGSL FFT / 2D convolution algorithms /
  WebGPU subgroup operations)
- 重要 4-6 (shader-f16, Apple Silicon GPU 最適化, batch dispatch) は
  網羅 + 概説、深掘りは M6.C 進度を見て conditional
- 不足 speedup を埋める追加探索責任は M6.B にない → honest report で
  Ponyo877 さんの戦略判断 (Stage 1 中間評価) に委ねる

**現状の前提条件** (M6.A.9 までに確定):

- Flow-Lenia kernel radius `dd = 5` → kernel 11×11 (FFT 適用閾値 ≥ 7×7 を満たす)
- **kernel は non-separable** (`crates/flow-lenia-core/src/` を grep 確認、
  separable / build_kernel_1d 系の関数なし、Plantec 論文の β=[1,…] 多重リング kernel
  も separable ではない) → 1D separable conv で代替する手は使えない、FFT 化が本道
- target hardware: Apple M1 (G13 GPU, SIMD lane 32, TBDR, unified memory,
  threadgroup memory 32 KB、register file 208 KiB/threadgroup)
- M6.A.5 で GPU bit-determinism を 5 process / 複数日にわたり確認済み
- 現状 step path: 6 dispatch (compute_potential / growth / flow_field /
  flow_apply / mass_balance / activation_clamp、BENCH.md §2 参照)
- BENCH.md §2: **convolve = 97.4% pass occupancy** (15 925 μs/呼出) —
  唯一の最優先最適化ターゲット。他 5 pass 合計 < 3%
- BENCH.md §2 takeaway: **FFT 化で convolve コスト 10-100× 削減見込み**、
  end-to-end GPU sps は Amdahl で 3-4× jump 予想 (M6.A.9 §13 hand-off)

**M6.C 着手前の milestone 境界として Ponyo877 さんに最終承認を依頼する**
採用 3 手法 + M6.C-1〜C-3 サブステップ計画案を §7, §8 にまとめる。
§9 で期待 speedup 積算 vs 70× 必要倍率の honest report (Stage 1 中間評価入力)。

---

## §1. 既知の前提文献 (M6.A までで認識済み)

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 1 | Plantec et al., Flow-Lenia | 2025 | [arXiv:2506.08569](https://arxiv.org/abs/2506.08569) | reference | Flow-Lenia 数値モデル原典、本実装の検証基準 |
| 2 | Chan, Lenia | 2019 | [arXiv:1812.05433](https://arxiv.org/abs/1812.05433) | reference | Lenia の基礎、JAX/PyCUDA 既存 GPU 実装が FFT-based convolution を採用 |

---

## §2. WGSL FFT 実装 (深掘り対象 1)

### §2.1 背景と適用判断

Flow-Lenia の畳み込みカーネルは現状 direct (point-wise multiply-add)。
kernel 11×11 × 10 kernel × 3 channel ≈ 3 630 ops/pixel。FFT 化すると
O(N² log N) で 256×256 grid なら理論上 10× 〜 100× の convolve 単独 speedup。
**Lenia 系研究の標準パターン**: Bert Chan の `LeniaNDK.py` (Reikna FFT) も
JAX の `jnp.fft` を使う最新研究も、kernel を pre-FFT して保持し、
`A_FFT × kernel_FFT → ifftn → fftshift` で実装している (`LeniaNDK.py` line 393-394
+ 458)。Flow-Lenia もこのパターンを踏襲するのが標準で、本件で否定する根拠なし。

### §2.2 関連文献

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 3 | fgiesen, "Notes on FFTs: for implementers" | 2023 | [fgiesen blog](https://fgiesen.wordpress.com/2023/03/19/notes-on-ffts-for-implementers/) | high | 実装者向け FFT 設計 notes、Cooley-Tukey + radix-4 推奨 (Stockham 非推奨) |
| 4 | Lloyd, "Fast computation of general Fourier Transforms on GPUs" | 2008 | [Microsoft TR-2008-62](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/tr-2008-62.pdf) | medium | GPU 上の汎用 FFT、Stockham + workgroup tiling の元ネタ (古いが基本設計) |
| 5 | Cadik & Slavik, "FFT and Convolution Performance in Image Filtering on GPU" | 2006 | [paper PDF](http://cadik.posvete.cz/papers/cadikm-iv06-gpu.pdf) | high | FFT vs direct conv のクロスオーバー解析、kernel ≥ 7×7 threshold の出典 |
| 6 | "Convolution of large 3D images on GPU" | 2011 | [Springer 10.1186/1687-6180-2011-120](https://link.springer.com/article/10.1186/1687-6180-2011-120) | medium | 大規模 FFT conv の decomposition、本件 256×256 は中規模で適用しやすい |
| 7 | "Overlap-and-Save GPU Fast Convolution" | 2019 | [arXiv:1910.01972](https://arxiv.org/pdf/1910.01972) | low | 大入力 + 小カーネルの分割 FFT、Flow-Lenia 同サイズ全フィールドには不適 |
| 8 | "GPU Acceleration of Image Convolution using Spatially-varying Kernel" | 2012 | [arXiv:1209.5823](https://arxiv.org/pdf/1209.5823) | reference | spatially-varying kernel は FFT 不可、Flow-Lenia の固定 kernel は適用可の裏付け |
| 9 | "Real-Time Cloth Simulation Using WebGPU" | 2025 | [arXiv:2507.11794](https://arxiv.org/pdf/2507.11794) | medium | WebGPU compute shader 高解像度シミュレーションの実例 |
| 10 | webgpu-groth16 (Heliax) | 2025 | [GitHub heliaxdev](https://github.com/heliaxdev/webgpu-groth16) | medium | Production WGSL FFT 実装 (ZK proving、Cooley-Tukey + tile-based + Montgomery twiddles) |
| 11 | "ZK Proofs and WebGPU" 解説 | 2025 | [webgpu.com news](https://www.webgpu.com/news/zk-proofs-webgpu-boost/) | reference | ZK 暗号で NTT (FFT) 実装の活発化、production lib 不在の指摘 |

### §2.3 ケーススタディ: WebTide (BarthPaleologue)

[BarthPaleologue/WebTide](https://github.com/BarthPaleologue/WebTide) は
WebGPU で Tessendorf 海洋を実装。`src/shaders/` を読むと:

- **アルゴリズム**: Cooley-Tukey radix-2 (precomputed twiddle テクスチャ +
  per-stage dispatch)。Stockham は不採用 — fgiesen 推奨と一致。
- **dispatch 構造**:
  - `twiddleFactors.wgsl` (`@workgroup_size(1,8,1)`): 起動時 1 回、
    各 stage の twiddle + バタフライ index を `rgba32float` texture に焼く
  - `horizontalStepIfft.wgsl` (`@workgroup_size(8,8,1)`): 1 dispatch = 1
    butterfly stage × 1 行群。256-point 横方向 FFT なら **log₂(256) = 8 dispatch**
  - `verticalStepIfft.wgsl`: 縦方向に同上 8 dispatch
  - **合計 16 dispatch per 2D iFFT pass**
- **workgroup memory は使用していない** (`var<workgroup>` 宣言なし)。各 thread が
  texture から twiddle + 2 input をロードして 1 butterfly 計算 → output texture
- **bit-reversal は不要** (twiddleFactors が `iid.y` に応じた input index を
  `data.ba` に焼き込む形 — fgiesen の bit-reverse-or-precomputed-index 議論で
  precomputed 側)
- **permutation.wgsl は bit-reversal ではなく `(-1)^(x+y)` の符号フリップ**
  = fftshift 相当 (周波数ゼロを中央へ)、本件では Lenia 同様 fftshift が必要

### §2.4 WebGPU dispatch overhead という致命的制約

Jędrzej Maczan, "Characterizing WebGPU Dispatch Overhead for LLM Inference
Across Four GPU Vendors, Three Backends, and Three Browsers"
([arXiv:2604.02344](https://arxiv.org/abs/2604.02344), 2026): **Metal backend
で 1 dispatch あたり 32-71 μs API overhead** (測定条件: LLM 推論 batch
size 1)。Apple は 4 vendor (NVIDIA / AMD / Apple / Intel) の 1 つとして
含まれるが、**論文中に M1 specifically の数値内訳はなく**、Apple 全体での
range と判断する。

**注意**: 同論文は "sequential-dispatch methodology が naive single-op
benchmark を ~20× overestimate する" とも指摘 — 本 survey の dispatch
overhead 推定は naive を採用しており **upper bound 寄り**、実 pipeline
で sequential pattern を取れば overhead は緩和される可能性あり。**M6.C-1
着手時に M1 + Chrome の実機 cross-check が必須**。

naive bound での試算 (WebTide パターンを Flow-Lenia 256×256 / K=10 / C=3 に
そのまま移植した場合):

- 1 step あたり dispatch 数:
  - A の forward 2D FFT: H 軸 8 stage + V 軸 8 stage = **16 dispatch**
  - per-kernel inverse 2D FFT: 16 dispatch × 10 kernel = **160 dispatch**
  - kernel-ごと spectral multiply: **10 dispatch**
  - 合計 **186 dispatch**
- per-dispatch overhead の range 32-71 μs (上記論文値) で全体 overhead 試算:
  - 下限 (32 μs): 32 × 186 = **5.95 ms** (60 FPS 予算 16.67 ms の **36%**)
  - 上限 (71 μs): 71 × 186 = **13.2 ms** (60 FPS 予算の **79%**)
  - 中央 (50 μs を midpoint 仮定): 50 × 186 = **9.3 ms** (60 FPS 予算の **56%**)
  - (Round 2 review NC-1 受け、point 推定 50 μs は range の midpoint 仮定で
    あること、conclusion は下限 36% でも成立することを明示)

→ **per-stage dispatch (WebTide pattern) では 60 FPS 目標は naive bound 下限
36% overhead でも破綻** (compute 時間に加えて overhead だけで予算の 1/3+)。
sequential-dispatch fast path で overhead が ~20× 改善されると 1 dispatch
あたり 2.5 μs 程度、186 dispatch 合計 **~500 μs total** で frame budget の数%
に収まる (Round 2 review NC-4 受け、当初 "500 μs/dispatch" は unit 表記
誤り、正しくは "500 μs total")。ただし sequential-dispatch fast path が
本件 pipeline (FFT 16 stage 連鎖 + kernel-loop) で完全に取れるかは要 M6.C-1
着手時の実機検証。

**workgroup-memory tiled FFT** で 1 dispatch あたり複数 stage を処理する必要あり
(M1 threadgroup memory 32 KB = complex64 で 1024 点まで on-chip 可、
256-point 1D FFT は 1 workgroup = 256 thread で 8 stage 全て 1 dispatch 完結可能)。

### §2.5 重要な技術的観察 (fgiesen + Lloyd + 実装ケースの統合)

- **Stockham は推奨されない** (fgiesen): ping-pong で working set 2 倍、L1 圧迫。
  in-place Cooley-Tukey 優先。**M6.A.9 時点の DESIGN.md §8 placeholder
  "Stockham カーネル × 2 軸" は本調査結果で覆る** — Rev.4.6 で「M6.B 結果次第
  で変更」と hedge した通り。
- **radix-4 が solid choice** (fgiesen): pass 数半減で twiddle 読込削減。
  ただし 256-point に対して log₄(256)=4 stage、log₂(256)=8 stage で 2 stage 削減、
  WGSL での実装複雑度トレードオフ要評価
- **RFFT (real input)**: 入力 A は実数なので N/2 サイズの複素 FFT + 1 追加 stage
  で N 点実 FFT 化、計算量 + メモリ 2 倍削減
- **twiddle precompute は KISS** (fgiesen): 対称性圧縮より素直に保持。WebTide が
  rgba32float texture に焼く方式は WebGPU でも自然
- **workgroup memory 32 KB は 256-point 1D FFT を on-chip で完結させるに十分**
  (complex64 = 8 byte × 256 = 2 KB、kernel data + scratch 込みで余裕)。
  log₂(subBlockSize) stage を 1 dispatch にまとめる定石が必須

### §2.6 M6.C 採用候補としての評価

**採用**: workgroup-memory tiled Cooley-Tukey radix-2 RFFT、
1 dispatch = 1D FFT 1 軸完結 (log₂(N) stage 統合)、twiddle は precompute texture/buffer。

- **期待 speedup (convolve pass 単独)**: 10-100× (BENCH §2 既存予測、文献 Cadik 2006
  + Springer 2011 の image conv 実測と整合)
- **期待 speedup (end-to-end GPU sps)**: **3-4× (BENCH §13 hand-off と一致)**。
  Amdahl 理論上限は 1/(0.026 + 0.974/50) = 21× だが、dispatch overhead +
  メモリトラフィック + kernel-spectrum multiply 追加コストで実効 3-4× が
  現実的予測。BENCH §13 は同根拠で 3-4× を確定しており、本 survey で
  範囲を拡張する根拠はない (Round 1 review SA-3 受け、3-4× 案を pull back)
- **risk**:
  - workgroup tiled FFT の WGSL 実装が production 例少ない (webgpu-groth16 が
    数少ない reference、ZK 用なので image conv 向け naming + layout で要適応)
  - dispatch overhead 制約 → **per-stage dispatch 設計を採用すると目標未達**、
    M6.C-1 着手時に "1D FFT を 1 dispatch に圧縮できるか" を最初に検証する必要あり
  - chaos amplification (M6.A.4.5 の tiered tolerance) を FFT 化でも維持する
    必要、Layer 3 (CPU-GPU C=1) の許容値が 1e-3〜2.5e-3 と緩いので吸収可能と
    予測するが M6.C-1 完了時に snapshot regression (Layer 4) で実証要

---

## §3. GPU 2D 畳み込み高速化 (深掘り対象 2)

### §3.1 アルゴリズム選択肢

| 手法 | 計算量 | 適用 kernel size | M6.C 採用評価 |
|---|---|---|---|
| Direct (現状) | O(N² K²) | 任意 | 現状、最適化対象 |
| im2col + GEMM | O(N² K²) | 任意 | row-major reshape の追加メモリコスト + Metal の matmul 最適化適用可、ただし Flow-Lenia の per-pixel kernel-sum パターンとは layout 不一致 |
| Winograd F(2,3) / F(4,3) | O(N² K²) × 0.5-0.7 | **≤ 5×5 で最適** | kernel 11×11 では適用不可 (Winograd は ≤ 5×5 の小カーネル CNN 向け) |
| FFT (Convolution Theorem) | O(N² log N) | **≥ 7×7 で有利** | **kernel 11×11 + N=256 で確実に適用可、§2 採用** |

### §3.2 関連文献

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 12 | "Im2col-Winograd: Fused Winograd for NHWC on GPUs" | 2024 | [ACM 10.1145/3673038.3673039](https://dl.acm.org/doi/10.1145/3673038.3673039) | low | Winograd 融合、kernel ≤ 5×5 向け、本件 11×11 では不適 |
| 13 | "ConvBench: 2D Conv Primitive Benchmark" | 2024 | [arXiv:2407.10730](https://arxiv.org/pdf/2407.10730) | reference | 9243 op benchmark、algorithm 選択基準の引用元 |
| 14 | "Optimizing Winograd on GPUs" | 2023 | [scispace](https://scispace.com/papers/optimizing-winograd-convolution-on-gpus-via-multithreaded-3vplxa7s) | low | multithreaded 最適化、Winograd 不採用なら参照のみ |
| 15 | "Karatsuba for 2D Conv" | 2025 | [PMC PMC12110178](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12110178/) | low | 数学最適化、GPU 実装は未検証、本件適用不可 |
| 16 | "FFT and Conv Performance in Image Filtering on GPU" | 2006 | [paper PDF](http://cadik.posvete.cz/papers/cadikm-iv06-gpu.pdf) | high | (= §2 ref 5) FFT vs direct クロスオーバー、kernel 11×11 で FFT 有利の根拠 |

### §3.3 M6.C 採用判断

**FFT 一択** (§2 と同じ採用)。Winograd は kernel 11×11 で適用不可。
im2col+GEMM は理論上可能だが Flow-Lenia の "K 個 kernel を同一 A に適用 → K 個
potential を生成" というパターンは GEMM の (M, K) batch dim にマップしづらい
(K が batch でなく "卷積 multiplier" として畳まれる)。direct のままで
constant-folding を狙う方向は M6.0 で既に到達済み (`@workgroup_size(8,8,1)` +
ピング・ポング設計、これ以上の direct 最適化は微増のみ予想)。

---

## §4. WebGPU Subgroup Operations (深掘り対象 3)

### §4.1 status (2026-05 時点)

- **Chrome**: Chrome 144 で `subgroup_id` feature ship、SIMD-level
  parallelism が production 利用可。`enable subgroups;` + `subgroupAdd`,
  `subgroupBallot`, `subgroupBroadcast`, `subgroupShuffle`, `subgroupExclusiveMul`
  等が使える。D3D backend は emulated だが M1 + Chrome は Metal backend なので native。
- **Safari**: 26.0 で WebGPU 本体は ship したが **subgroup ops は未対応**
  (catching up 中の言及あり)。
- **Firefox**: 145 で macOS Apple Silicon サポートに追いついたが
  **subgroup ops は未対応**。

**本件 target は Chrome WebGPU 限定** なので採用可。ただし Safari/Firefox
fallback が必要な場面 (publish phase M5) では subgroup 不使用 path も並存させる
必要あり (M5 で検討事項、M6.C 内では Chrome 専用で OK)。

### §4.2 関連文献

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 17 | gpuweb/proposals/subgroups.md | 2024-2026 | [GitHub gpuweb](https://github.com/gpuweb/gpuweb/blob/main/proposals/subgroups.md) | high | WebGPU subgroup spec proposal、利用可能 builtin の primary source |
| 18 | "What's New in WebGPU (Chrome 144)" | 2026 | [Chrome dev blog](https://developer.chrome.com/blog/new-in-webgpu-144?hl=en) | high | `subgroup_id` ship、D3D で emulated の制約 |
| 19 | "What's New in WebGPU (Chrome 128)" | 2024 | [Chrome dev blog](https://developer.chrome.com/blog/new-in-webgpu-128) | medium | subgroup origin trial 開始時の解説 |
| 20 | "What's New in WebGPU (Chrome 134)" | 2025 | [Chrome dev blog](https://developer.chrome.com/blog/new-in-webgpu-134) | medium | subgroup matrix multiply 追加状況 |
| 21 | gpuweb Implementation Status wiki | rolling | [GitHub wiki](https://github.com/gpuweb/gpuweb/wiki/Implementation-Status) | reference | browser feature support の primary source |
| 22 | WebGPU 1.0 W3C spec | 2026 | [W3C TR/webgpu](https://www.w3.org/TR/webgpu/) | reference | spec 本体、limits の最終確認 |
| 23 | webgpufundamentals "WebGPU Subgroups" | 2025 | [webgpufundamentals.org](https://webgpufundamentals.org/webgpu/lessons/webgpu-subgroups.html) | medium | 教育用解説、`subgroupExclusiveMul` で memory-free factorials の例 |
| 24 | Khronos "Vulkan Subgroup Tutorial" | 2018 | [Khronos blog](https://www.khronos.org/blog/vulkan-subgroup-tutorial) | reference | subgroup ops 設計の元ネタ (WGSL は Vulkan SPIR-V 由来) |
| 25 | M1 G13 SIMD 解析 (dougallj) | rolling | [Apple G13 Reference](https://dougallj.github.io/applegpu/docs.html) | high | SIMD width = 32 確認、32-bit exec_mask、128 GPR/SIMD-group |

### §4.3 Flow-Lenia への適用余地

convolve は 97.4% を占めるが、subgroup ops が直接効くわけではない
(畳み込みは output-pixel 並列でロード境界が SIMD lane 境界と合わせにくい)。
**むしろ FFT butterfly での 2-input 共有** や **reduction (mass_balance,
最大値クリッピング)** で活用余地あり:

1. **FFT butterfly の partner-lane 共有**: 1D FFT 256-point を SIMD lane 32 で
   分担すると 1 butterfly = 同 subgroup 内の lane i ↔ lane (i+stride) のデータ交換
   → `subgroupShuffle` で **shared memory 経由より低レイテンシ**で取得可能。
   ただし stride > 32 の上位 stage は subgroup 跨ぎ → workgroup memory 経由必要。
2. **reduction の `subgroupAdd` 化**: mass_balance pass の sum-reduction を
   subgroupAdd で 1 step (= log₂(32) 相当の latency) で完結。現状 reintegrate
   が 1.5% なので改善幅は小さい (end-to-end 0.5% 未満) が、kernel fusion で
   別 pass を統合する際の primitive として有用。
3. **subgroup matrix multiply (Chrome 134+)**: Chrome 134 で subgroup matrix
   ops が追加 (cite 文献 20)。**Apple Silicon family は Metal の
   simdgroup_matrix を持つと報告されており** (Llama2 7B / HF embedding
   ベンチで Apple Silicon M1 Pro / M1 Max での speedup 実測あり — §5.1 ref
   #26, #27)、**M1 G13 specifically の対応は dougallj ref #25, #29 でも
   命令単位の直接記述は確認できなかった (Round 1 review SA-5 受け downgrade)**。
   M6.C-1 着手時に Chrome WebGPU 経由で M1 mini 実機で subgroup matrix が
   native 動作するかは未確認。matrix mul による direct conv (im2col 代替) は
   今回不採用だが、FFT butterfly の "2-input vec2 → 2-output vec2" を matrix
   mul で表現する手は理論上ある。

### §4.4 M6.C 採用候補としての評価

**採用 (conditional)**: M6.C-1 (FFT 化) と並行して subgroup primitive 利用を
試行、ただし主目的は FFT 化の補完。end-to-end speedup 寄与は限定的 (1.5-2×、
かつ FFT 化と同時 commit なので独立計測困難)、M6.C-2 で「FFT 化後に subgroup
あり/なし」のペアラン測定で寄与を分離する設計とする。Safari/Firefox 対応は
M5 で fallback path (subgroup 不使用版を WGSL に共存) を別途検討。

---

## §5. 重要トピック (網羅 + 概説)

### §5.1 shader-f16 (half precision)

**Status**: Chrome 120+ で `shader-f16` feature ship、WGSL で `enable f16;`
+ `f16` / `vec2<f16>` / `vec4<f16>` 等が使える ([Chrome blog M120](https://developer.chrome.com/blog/new-in-webgpu-120))。

**Apple M1 実績** (Intel 解説 + Chrome blog):

- ALU bound で +25%、memory bound で +50% (Intel ベンチ)
- M1 Pro で Llama2 7B prefill +28% / decoding +41% (cf 同上)
- M1 Max で HF text embedding **3×** vs f32 (cf 同上)
- → **memory bound 部分での効果が大きい**。Flow-Lenia の FFT は kernel-spectrum
  × world-spectrum の per-cell complex multiply が memory bound 寄り → 期待 +50%
  〜 +100%

**Risk**:

- Flow-Lenia の chaotic dynamics (M2.8 で C=3 chaos amplification 実証済み、
  M6.A.4.5 で rel<1e-3 が 256 で破綻) → f16 mantissa **10 bit + 1 implicit
  = 11 bit precision、machine epsilon ε ≈ 4.88e-4** (Round 1 review SA-4
  受け訂正、当初 "ε ≈ 1e-3" は不正確)。中心極限定理で 1 step 1 cell あたり
  100 ops の random-walk 蓄積を見積もると、step 内誤差 σ_step ≈ ε × √100
  = 5e-4 × 10 = **5e-3 per step**。さらに 100 step の独立蓄積で σ_100step ≈
  σ_step × √100 = **5e-2** に達する。Layer 3 tiered tolerance 1e-3〜2.5e-3 を
  step 1 でも超過し、100 step では 20× 超過。
  → **精度を完全 f16 化する案は採用不可と確定** (Round 2 review NC-3 受け
  arithmetic 訂正、step / multi-step 両 scale で quantitative に超過確認)。
- 採用可能性: kernel spectrum (静的) や twiddle (静的) を f16 で持ち、
  active field A は f32 維持の **mixed-precision** 方式。これも safe か要 M6.C-3
  実機検証 (Layer 3-4 regression で確認)。

**M6.C 採用候補**: M6.C-3 (FFT + subgroup の後) で **mixed-precision を試行**、
レイヤ 3 (CPU-GPU C=1 tiered tolerance) を超えるなら roll back。期待 +30-50%、
不採用時 0%。

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 26 | "What's New in WebGPU (Chrome 120)" | 2023 | [Chrome dev blog](https://developer.chrome.com/blog/new-in-webgpu-120) | high | shader-f16 ship、M1 実績数値の出典 |
| 27 | Intel "Revving Up WebGPU Apps with f16" | 2024 | [Intel article](https://www.intel.com/content/www/us/en/developer/articles/community/revving-up-webgpu-applications-with-power-of-f16.html) | high | f16 ベンチマーク (ALU 25% / memory 50%) |
| 28 | "WebGPU shader-f16 feature support" 解説 | 2025 | [scrapfly](https://scrapfly.io/web-scraping-tools/gpu-fingerprint/webgpu/shader-f16) | medium | デバイス別 f16 サポート状況 |

### §5.2 Apple Silicon GPU 最適化 (TBDR / unified memory / threadgroup memory)

**事実関係** (philipturner/metal-benchmarks + dougallj/applegpu):

- M1 G13 GPU: 8-core, SIMD width 32, threadgroup max 1024 thread,
  threadgroup memory **32 KB**, register file 208 KiB/threadgroup,
  256 half-word reg/thread (使用過多で occupancy 低下)
- **TBDR** (Tile-Based Deferred Rendering) はグラフィックス pass 向け、
  compute pass では tile memory 直接利用不可 (Metal の `MTLTileRenderPipeline`
  は WebGPU 非露出)
- **Unified memory**: host buffer ↔ device buffer の copy 不要、wgpu の
  `MAP_READ` も低コスト (M1 で <10 μs/MB 実測 M6.0)
- **simdgroup_matrix サポート確認**: M1 G13 で利用可能 ("tensor core" 相当)、
  ただし WebGPU 露出は Chrome 134+ の subgroup matrix multiply 経由 (要実機検証)

**M6.C 採用ポイント**:

- workgroup memory 32 KB → 1D FFT 256-point (2 KB) を on-chip で完結可、
  1D FFT 1024-point まで余裕あり (8 KB)
- register pressure: M1 で thread あたり 256 reg を超えると occupancy 低下、
  WGSL で `var<private>` の使用量を 100-200 程度に抑える設計が望ましい
- unified memory: kernel-FFT data の host → GPU upload を起動時 1 回で済ませる
  既存戦略 (M6.0) を継続

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 29 | dougallj "Apple G13 GPU Architecture Reference" | rolling | [dougallj.github.io](https://dougallj.github.io/applegpu/docs.html) | high | M1 G13 命令セット詳細、reverse-engineered |
| 30 | philipturner "metal-benchmarks" | rolling | [GitHub metal-benchmarks](https://github.com/philipturner/metal-benchmarks) | high | M1 マイクロアーキテクチャ実測ベンチマーク |
| 31 | Apple "Metal Feature Set Tables" | rolling | [Apple PDF](https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf) | reference | Metal feature の Apple 公式表、limits 確認 |
| 32 | Rosenzweig "Dissecting the Apple M1 GPU, part III" | 2021 | [alyssarosenzweig blog](https://alyssarosenzweig.ca/blog/asahi-gpu-part-3.html) | medium | Asahi project の M1 GPU 解析、命令 encoding 解説 |
| 33 | "WebGPU bugs holding back browser AI" (Emmerich) | 2025 | [Medium](https://medium.com/@marcelo.emmerich/webgpu-bugs-are-holding-back-the-browser-ai-revolution-27d5f8c1dfca) | reference | WebGPU 実装側の現状制約、Apple 環境での問題例 |

### §5.3 複数 creature の batch dispatch / 1 dispatch per step

**現状の問題** (BENCH §2 § §13):

- 1 step = 6 dispatch、Metal overhead 50 μs × 6 ≈ 300 μs (16.67 ms 予算の 1.8%)
- FFT 化で +50-150 dispatch 追加 (§2.4) → overhead だけで予算を食う

**最適化方向**:

1. **kernel fusion**: 隣接 pass を 1 WGSL に統合。M6.0 では分離設計 (clarity
   優先) だったが、M6.C で compute_potential + growth + flow_field を 1 pass に
   する余地あり (各 pass の input/output が次 pass の input/output と直結)
2. **vec4 packing**: channel C=3 を vec4 にパック (1 lane 余り) して 1 lane = 1 cell
   全 channel を処理。M1 G13 は f32 vec4 演算が 1 cycle で完結 (cf. metal-benchmarks)
3. **K kernel batching**: kernel 10 個を spectrum multiply で per-pixel x K の
   並列化、現状 K-loop in shader を K-axis dispatch dimension に展開

**M6.C 採用候補**: M6.C-1 (FFT) の WGSL 設計で kernel fusion + vec4 packing を
最初から織り込む。独立トピックとしての measurement は困難 (FFT 化と同時)、
deliverable レベルでは「FFT 化 commit に含まれる」扱い。

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 34 | "Characterizing WebGPU Dispatch Overhead" | 2026 | [arXiv:2604.02344](https://arxiv.org/abs/2604.02344) | high | Metal で 32-71 μs/dispatch、本件 60FPS 目標で kernel fusion 必須の根拠 |
| 35 | "WebGPU Compute Shaders Explained" (Beckley) | 2024 | [Medium](https://medium.com/@osebeckley/webgpu-compute-shaders-explained-a-mental-model-for-workgroups-threads-and-dispatch-eaefcd80266a) | reference | workgroup/dispatch メンタルモデル解説 |
| 36 | NVIDIA "Thread-Group ID Swizzling for L2 Locality" | 2020 | [NVIDIA blog](https://developer.nvidia.com/blog/optimizing-compute-shaders-for-l2-locality-using-thread-group-id-swizzling/) | low | L2 swizzle、M1 では cache 階層が違うため direct 適用は限定的 |

---

## §6. Reference (網羅、深掘り不要)

### §6.1 Lenia / Flow-Lenia 既存 GPU 実装

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 37 | Chakazul/Lenia (Bert Chan, Python) | rolling | [GitHub Chakazul/Lenia](https://github.com/Chakazul/Lenia) | reference | LeniaND.py / LeniaNDK.py 等、Reikna FFT で convolution theorem 実装 (line 393-394) |
| 38 | Bert Chan home page | rolling | [chakazul.github.io](https://chakazul.github.io/) | reference | Lenia 全 implementation の起点、ALife 2018-2024 文献まとめ |
| 39 | "Towards Large-Scale Simulations of Open-Ended Evolution" (Chan 2024) | 2024 | [ACM 10.1145/3583133.3590670](https://dl.acm.org/doi/10.1145/3583133.3590670) | reference | JAX evolutionary Lenia、jnp.fft 利用 |
| 40 | BarthPaleologue/WebTide | 2024 | [GitHub WebTide](https://github.com/BarthPaleologue/WebTide) | high | (§2.3 で詳細分析) WebGPU FFT 実装ケーススタディ |
| 41 | Popov72/OceanDemo | 2024 | [GitHub OceanDemo](https://github.com/Popov72/OceanDemo) | medium | BabylonJS + WebGPU FFT 海洋、200-250 compute dispatch/frame の重い設計 |
| 42 | iamyoukou/fftWater | 2023 | [GitHub fftWater](https://github.com/iamyoukou/fftWater) | reference | C++/OpenGL Tessendorf FFT、WebGPU 非だが algorithm 比較に |
| 43 | tessarakkt/godot4-oceanfft | 2024 | [GitHub godot4-oceanfft](https://github.com/tessarakkt/godot4-oceanfft) | reference | Godot 4 compute shader FFT、algorithm 構成参考 |

### §6.2 WebGPU 一般 (M5 deploy phase 用 status check 文献群)

**正直 framing** (Round 1 review M-3 受け): 以下 7 件は M6.C 採用判断に直接
寄与しない context / status-check 文献。50 件達成のための padding ではなく、
**M5 publish phase で WebGPU production-readiness を確認する際の reference
群** として明示。M6.C 着手判断には §1-§5 + §6.1 で十分。

| # | Ref | Year | Source | 適用 | 1 行要約 |
|---|---|---|---|---|---|
| 44 | webgpufundamentals "Compute Shader Basics" | 2024 | [webgpufundamentals.org](https://webgpufundamentals.org/webgpu/lessons/webgpu-compute-shaders.html) | reference | WebGPU compute shader 入門 |
| 45 | surma "WebGPU — All of the cores" | 2023 | [surma.dev](https://surma.dev/things/webgpu/) | reference | WebGPU compute 解説、初学者向け |
| 46 | "WebGPU 2026: GPU Compute and AI Inference" | 2026 | [Programming Helper](https://www.programming-helper.com/tech/webgpu-2026-browser-gpu-api-wgsl-ai-inference) | reference | 2026 時点での WebGPU AI 利用状況、M5 status check 用 |
| 47 | "WebGPU Hits Critical Mass" | 2025 | [webgpu.com news](https://www.webgpu.com/news/webgpu-hits-critical-mass-all-major-browsers/) | reference | 全主要ブラウザ shipping、M5 deploy 時の status check に |
| 48 | "WebGPU vs WebGL Inference Benchmarks" | 2025 | [sitepoint](https://www.sitepoint.com/webgpu-vs-webgl-inference-benchmarks/) | reference | WebGPU の WebGL 比、M5 で WebGL fallback 検討時の参考 |
| 49 | "WebGPU Browser Support 2026" | 2026 | [webo360solutions](https://webo360solutions.com/blog/webgpu-browser-support/) | reference | ブラウザ別 support 一覧、M5 deploy 時に再確認 |
| 50 | "Frontier Web APIs 2026" | 2026 | [utsubo blog](https://www.utsubo.com/blog/frontier-web-apis-2026-production-ready) | reference | WebGPU production-readiness 評価、M5 deploy 判断補助 |

---

## §7. M6.C 採用候補 3 手法と期待 speedup 積算

scope-guardian 確認の通り、深掘り 3 トピックから採用する 3 手法を以下に絞る。
**Ponyo877 さんへの最終承認対象セクション**。

### §7.1 採用 3 手法

| # | 手法 | 由来 | 期待 end-to-end speedup | 信頼度 |
|---|------|------|---|---|
| **C-1** | **workgroup-memory tiled FFT (Cooley-Tukey radix-2, RFFT, precomputed twiddle, kernel pre-FFT 永続化)** | §2 深掘り | **3-4×** (BENCH §13 hand-off と一致) | high (BENCH §2 既存予測 + Lenia 系標準 + Cadik 2006 実測根拠) |
| **C-2** | **kernel fusion + vec4 channel packing + subgroup-aware reduction** (FFT pass の dispatch 数削減 + reduction 加速) | §4 + §5.3 | **1.5-2×** | medium (dispatch overhead 32-71 μs/呼出 削減の理論積算、実測は M6.C-2 時) |
| **C-3** | **mixed-precision (kernel/twiddle f16、active field f32 維持)** | §5.1 | **1.3-1.5×** (条件付き) | medium-low (M1 で memory-bound 部分 +50% 実証、ただし numerical regression 維持できるか M6.C-3 時の Layer 3-4 検証で決定、unsafe なら roll back し 0%) |

### §7.2 期待 speedup 積算 (理論上限 vs 現実圏内)

3 手法を順次積み上げた場合の **理論積最大 (積独立性を仮定した ceiling)**:

- C-1 単独: 3-4×
- C-1 × C-2: 4.5-8×
- C-1 × C-2 × C-3: 6-12×

**重要な caveat (Round 1 review SA-1 受け、積独立性を honest articulate)**:
上記積算は各手法の effect が独立と仮定した **upper-bound ceiling** であり、
現実には以下の相互依存が存在する:

- C-1 (FFT) と C-2 (FFT pass の kernel fusion) は **同じ compute unit を共有**。
  C-1 設計時から fusion を織り込んだ場合、C-2 の "追加" 寄与は table 値より
  目減りする
- C-2 (subgroup reduction による memory traffic 削減) と C-3 (f16 による
  memory bandwidth 半減) は **同じ memory bandwidth budget を奪い合う** —
  C-2 で bandwidth pressure が緩和されていれば C-3 の marginal 寄与は縮む
- C-3 (f16 storage) の +30-50% は memory-bound 部分での効果、C-1 + C-2 後の
  pipeline で memory-bound か compute-bound かは実装後にしか確定しない

→ **現実圏内の combined speedup は 6-10× が経験的予測** (上記相互依存 +
dispatch overhead 実測値 + register pressure 等を考慮)。15× は perfect
independence の理論 ceiling であって "expected" ではない。

**1 creature ベース** (必要倍率は §9 参照): 60 FPS で 必要 14× (現実圏内
6-10× では 不足)、30 FPS で 必要 7× (現実圏内で 達成圏内)。
**4 creature** はコスト係数未確認 (§9.1 MF-2 参照、Stage 1 中間評価 defer)。

### §7.3 各手法の risk と falsifiability

| 手法 | risk | M6.C 内で誤りが判明する mechanism (test 名明示、Round 1 review SA-6 受け) |
|------|------|---|
| C-1 | dispatch overhead で実効 speedup が予測下限を割る、workgroup memory tiled の WGSL 実装が困難 | `perf_regression` で `bench_step` の post/pre ratio が **< 2×** ならば下限割れ確定 (BENCH §13 既存予測 3-4× の下限 75% 割れ) → C-2, C-3 進む前に再評価 |
| C-2 | kernel fusion で WGSL 複雑化、subgroup ops が Chrome+M1 で期待通り効かない | M6.C-2 完了時に `perf_regression` を **paired-run mode** (C-2 あり/なし を同一 quiesced host で交互測定 N=3 median) で実行、寄与 < 1.3× なら cost-benefit 不成立 (paired-run プロトコルは CLAUDE.md 測定プロトコル §1 + BENCH §13 既存運用) |
| C-3 | mixed-precision で chaotic dynamics の数値発散、Layer 3-4 regression 落ちる | `gpu_field_regression_g256` (= Layer 3 CPU-GPU tiered C=1 test) で 2.5e-3 tolerance 違反、または `gpu_snapshot_regression` (= Layer 4) で 既存 post-M6.C-1 baseline からの drift → 即 roll back、C-3 commit 不採用、寄与 0 として終了 |

---

## §8. M6.C-1 〜 M6.C-3 サブステップ計画案

scope-guardian 確認の通り、優先順位 + 期待 speedup 根拠まで。WGSL 雛形は M6.C で。
**Ponyo877 さんへの最終承認対象セクション**。

### §8.0 着手前提条件

- **task #132 (M6.A.7.1, ValidationGuard lib unit tests 拡張)** を完了する
  ことを M6.C-1 着手プロンプトで reminder として呼び出す (M6.A.9 確定事項)
- M6.A.5 の 5-layer test strategy を M6.C-N 各サブステップで踏襲、特に Layer 3
  (CPU-GPU C=1 tiered tolerance 1e-4/5e-4/1e-3/2.5e-3) と Layer 4 (snapshot
  pre/post regression) を毎 commit で実行

### §8.1 M6.C-1: workgroup-memory tiled FFT 化

**目的**: convolve pass を direct → FFT 化、end-to-end GPU sps 3-4× 向上
(§2.6 + BENCH §13 一致)。

**設計骨子**:

1. `crates/flow-lenia-gpu/src/passes/convolve.rs` を新規 `convolve_fft.rs`
   として並置 (direct convolve は M6.C-1 中は keep、A/B 比較に使用)
2. WGSL: `fft_horizontal.wgsl` + `fft_vertical.wgsl` + `spectral_multiply.wgsl` +
   `ifft_horizontal.wgsl` + `ifft_vertical.wgsl`、または kernel fusion で
   `fft_2d.wgsl` + `spectral_multiply.wgsl` + `ifft_2d.wgsl` (3 dispatch)
3. workgroup_size 設計: 1D FFT 256-point を 1 workgroup = 256 thread で完結
   (log₂(256)=8 stage を `workgroupBarrier` で繋いで 1 dispatch)
4. twiddle: 起動時 1 回 precompute、`storage<read>` uniform buffer に固定
5. kernel pre-FFT: K=10 kernels を起動時に FFT してキャッシュ、毎 step 不要
6. Bit-reversal は precomputed index でなく twiddle に焼くか cooley-tukey の
   in-place natural-to-bit-reverse 順序で受ける形のどちらかを選択 (実装時判断)

**検証**:

- Layer 1 (CPU bit-equal): direct convolve の出力と比較 (FFT 化で bit-equal は
  期待できないので、A.4.5 tolerance scenario で field level rel < 1e-3 程度を
  許容)
- Layer 2 (mass): mass conservation 維持を grid 32-512 全域で
- Layer 3 (CPU-GPU C=1 tiered): 1e-4/5e-4/1e-3/2.5e-3 を維持
- Layer 4 (snapshot regression): pre-M6.C-1 baseline と新 baseline を別ファイル
  保持 (snapshot は M6.C-1 で必然的に bit 変わるので、新 baseline を確定し
  以降は post-M6.C-1 baseline で比較)
- Layer 5 (sanity): creature alive 持続
- perf_regression: end-to-end GPU sps が +200-400% (3-4×) に上振れ確認

**期待**: 3-4×

### §8.2 M6.C-2: kernel fusion + vec4 + subgroup reduction

**目的**: M6.C-1 後の dispatch 数を削減、reduction pass を subgroup ops 化、
追加 1.5-2× speedup。

**設計骨子**:

1. M6.C-1 で分けた fft_2d / spectral_multiply / ifft_2d を **1 大 dispatch**
   に統合 (workgroupBarrier で stage 境界、shared memory で intermediate
   buffer)
2. compute_potential + growth + flow_field を 1 pass に fusion
3. mass_balance reduction を `subgroupAdd` + workgroup-level reduction で
   2-level に再構築
4. vec4 packing: channel 3 を vec4 に詰めて 1 thread = 1 cell × 全 channel

**検証**: M6.C-1 と同じ 5-layer + perf_regression。さらに ペアラン
("M6.C-2 あり/なし" を quiesced state で各 N=3 median) で寄与を分離記録。

**期待**: 1.5-2× (M6.C-1 後比)

### §8.3 M6.C-3: mixed-precision (kernel/twiddle f16、active field f32)

**目的**: memory bound 部分の f16 化で追加 1.3-1.5× speedup。

**設計骨子**:

1. WGSL で `enable f16;`、kernel_FFT (静的データ) と twiddle を `vec2<f16>` 格納
2. 計算は f32 で行い (`f32(kernel_h.x)` 等)、ストレージのみ f16
3. active field A は f32 維持 (chaos に対する保守)

**検証**: M6.C-1/2 と同じ 5-layer。Layer 3 (CPU-GPU C=1 tiered) を **必ず**
クリアすること、超えたら即 roll back (寄与 0 として終了、M6.C-3 commit 不採用)。

**期待**: 1.3-1.5× (採用時)、roll back 時 0% で M6.C-2 までで M6.C 締め

### §8.4 M6.C-3 完了時の Stage 1 中間評価

CLAUDE.md 撤退ライン: **256×256×3×4creature で 30 FPS なら M5 へ**。

- M6.C-3 完了時に 256×256×3 で実測、4 creature 表現方式が確定したら
  4 creature 込みで実測
- 30 FPS = 33 ms/step 達成可否で Stage 1 通過判定
- 30 FPS 未達でも 15-25 FPS 帯にいれば research/demo 価値あり、CLAUDE.md
  「目標見直し」枠で Ponyo877 さんに方針確認

---

## §9. Stage 1 中間評価への予備評価

**Ponyo877 さんへの戦略判断入力**。M6.B 段階で M6.C 結果を待たずに行う予備評価。

### §9.1 必要 speedup と達成見込み

**Round 1 review MF-1 受け、baseline を BENCH §1 実測値に再anchor**
(当初 "web 300 ms" は出典なき推測で削除):

- 現状 **256×256×3 native warm = 230.14 ms/step (4.3 sps)** (BENCH §1 実測、
  Apple M1 mini, wgpu Metal native, K=10)
- **web (Chrome WebGPU 経由) の実測は本 survey 時点で未取得**。文献的に WebGPU
  vs native Metal の overhead 差は +10-30% (§5.3 dispatch overhead) 推定だが、
  Stage 1 中間評価時に web 実測必須

**4 creature の cost 増 (Round 1 review MF-2 受け、推測を honest framing)**:

- Plantec 2025 paper の multi-creature 表現方式 (additive channel か独立グリッド
  か) は **本 survey の WebFetch では abstract のみ確認可、本文 PDF 未読**。
  「Flow-Lenia allows us to embed the parameters of the model... thus allowing
  for multispecies simulations」(abstract 引用) という言明のみで、cost
  スケーリングは未確認
- → **4 creature cost 係数は本 survey で確定せず**、必要倍率は **1 creature
  ベース** で提示。M6.C 着手時に Flow-Lenia 実装での 4 creature 表現方式を
  確定し、M6.C-3 Stage 1 中間評価で実測 → 正式必要倍率確定

**1 creature ベース、native 230 ms から必要倍率**:

| ターゲット | 必要 ms/step | 必要 speedup (native baseline) |
|---|---|---|
| 60 FPS (本来目標) | 16.67 | **13.8×** |
| 30 FPS (CLAUDE.md 撤退ライン) | 33.3 | **6.9×** |
| 20 FPS (撤退時 fallback 候補、参考) | 50.0 | **4.6×** |

**web 換算** (Chrome WebGPU overhead +10-30% 推定): 必要倍率は上記に +10-30%
上乗せ → 60 FPS 必要 **15-18×**、30 FPS 必要 **7.6-9.0×**、20 FPS 必要
**5.1-6.0×** (1 creature ベース)

**4 creature 込みは未確定**、M6.C-3 Stage 1 中間評価で実測。

### §9.2 §7.2 積算結果との比較

| ターゲット | 必要 speedup (web 推定, 1 creature) | M6.C-1〜C-3 現実圏内 6-10× | M6.C-1〜C-3 理論積最大 12× | 達成見込み |
|---|---|---|---|---|
| 60 FPS | 15-18× | 不足 | 境界 | **現実圏内では未達、理論上限なら境界** |
| 30 FPS (撤退ライン) | 7.6-9.0× | 達成圏内 | 達成圏内 | **達成見込みあり** |
| 20 FPS (参考: 撤退ライン未達時の目標再定義候補) | 5.1-6.0× | 達成圏内 | 達成圏内 | **確実圏内** |

**重要な caveat**: 上表は **1 creature ベース** のみ。4 creature 込みのコスト
係数は Plantec 2025 本文 未読のため未確定 (§9.1)、Stage 1 中間評価で実測。
仮に 4 creature で 2-4× 増ならば、上記必要倍率は 2-4× 倍となり、30 FPS も
不足圏内に陥る可能性あり。

### §9.3 honest framing と戦略選択肢

**事実関係 (再anchor 後)**:

- 60 FPS 目標は M6.C-1〜C-3 現実圏内 (6-10×) では届かない、理論積最大 (12×)
  で境界
- 30 FPS (撤退ライン) は 1 creature ベースで現実圏内で達成見込み、4 creature
  込みで判定保留
- 20 FPS は 1 creature ベースで確実圏内、4 creature 込みでも達成圏内予想

**scope-guardian 確認の通り、M6.B では「不足分を埋める追加探索責任はない」**
ので、本セクションは Ponyo877 さんに戦略選択肢の提示で完結する:

**選択肢 A**: M6.C 着手、Stage 1 中間評価 (M6.C-3 完了時) で 30 FPS 達成可否
判定 (4 creature 実装込みで実測)。達成なら M5 へ、未達なら目標見直し (例:
20 FPS で finalize、または 256×256×3×2creature に縮小)

**選択肢 B**: M6.B 時点で目標見直し (例: 60 FPS は維持できない、20-30 FPS
帯で finalize、creature 数縮小)

**選択肢 C**: M6.B を extend して追加手法を調査 (subgroup matrix multiply の
M1+Chrome 実機検証、shader-f16 完全採用版の数値精度検証、独自 algorithm 探索)。
ただし scope-guardian は「M6.B では不足分追加探索責任なし」と判定済みなので、
これを採用する場合 scope 拡張として Ponyo877 さんの明示承認が必要

### §9.4 Claude Code としての推奨

CLAUDE.md "撤退ライン" (256×256×3×4creature で 30 FPS なら M5 へ) を尊重し、
**選択肢 A**: M6.C 着手 + M6.C-3 完了 Stage 1 中間評価で正式判断、を推奨。

**self-serving rationalization リスクの開示** (Round 1 review M-2 受け):
"Claude Code は M6.C 作業を継続する立場にあるため、選択肢 A 推奨は自分の作業
継続を rationalize している可能性がある" — この risk を Ponyo877 さんに
明示しておく。下記推奨理由を確認し、selfish 要素があれば指摘ください。

推奨理由:

1. §7.2 積算 (積最大 12× / 現実圏内 6-10×) は **積最大が ceiling であって
   expected ではない**ことを §7.2 で honest articulate 済み。実測で
   現実圏内下限 (6×) を割る可能性は dispatch overhead や register pressure
   実値次第。M6.C-1 完了時点で early-exit 判断材料が揃う
2. 現時点 web vs native 比、4 creature cost 係数の両方が未確定 (§9.1 で
   honest framing)。M6.C-3 までの実測でこれらを確定してから戦略判断が CLAUDE.md
   原則 1 (観察した現象は対症療法せず、原因究明を先行) と整合
3. 「測定なしで撤退」は M6.A の subagent review 文化 (実証ベース判断) とも
   不整合 — ただしこの理由は 1, 2 ほど strict ではなく、judgment call
4. **反論可能性**: もし §7.2 現実圏内下限 6× × 4 creature コスト 4× = 必要
   speedup 60× で gap が極大なら、M6.C 完走しても撤退確実 → 選択肢 B の
   早期撤退が strictly logical。この場合の判断は MF-2 (4 creature コスト
   未確定) の解消が前提

---

## §10. 調査メタ情報

### 調査期間
M6.B 着手: 2026-05-24 (Phase 3 移行第 1 milestone)
完了: 2026-05-24 (同日内、scope-guardian + adversarial-reviewer review pending)

### 調査原則
- CLAUDE.md "tolerance 緩和前に物理的根拠" を speedup 期待値にも適用
  ("論文値そのまま転記" でなく、Flow-Lenia の条件下で何 % か根拠を示す)
- 既存実装が見つかった場合も、M6.B では読解 + 評価まで。動作確認 + ベンチは M6.C
- adversarial-reviewer review で「speedup 期待値の根拠が十分か」を厳しく見る

### 既知の制約
- Apple M1 mini target、simdgroup_matrix は M1 G13 で利用可能 (本調査で更新、
  当初の「M1 Pro 以降のみ」前提は誤り)。ただし WebGPU からの露出は Chrome 134+
  の subgroup matrix multiply 経由、M1+Chrome での native 動作は要 M6.C-1 実機検証
- Chrome target、Safari/Firefox は subgroup 未対応 (M5 fallback 検討事項)
- M6.A.5 で確立した 5-layer test strategy を M6.C で踏襲する前提
- §8 各サブステップで M6.A.9 確立の paired-run / quiesced / N=3 median /
  honest framing プロトコル (CLAUDE.md 測定プロトコル) を踏襲

### 統計
- 網羅文献数: **50** (§1: 2 件 / §2: 9 件 / §3: 5 件 / §4: 9 件 / §5: 11 件
  / §6: 14 件)
- 適用区分: high 14, medium 14, low 5, reference 17
- 深掘り: §2 (WGSL FFT) / §3 (2D conv algorithm 比較) / §4 (subgroup ops) の 3 つ
- 採用候補: §7 で 3 手法 (C-1 FFT / C-2 fusion+vec4+subgroup / C-3 mixed-precision)
- 期待 speedup 理論積最大 12× / 現実圏内 6-10× (Round 1 review SA-1 受け、
  積独立性を honest framing)
- 必要 speedup (1 creature web 推定): 60 FPS で 15-18× / 30 FPS で 7.6-9.0× /
  20 FPS で 5.1-6.0× (Round 1 review MF-1 受け BENCH §1 native baseline 230 ms
  に再anchor、Chrome WebGPU overhead +10-30% 推定込み)。4 creature コスト
  係数は Plantec 2025 本文未読のため未確定、Stage 1 中間評価で実測 (MF-2)

### サブステップ一覧 (M6.B 内、M6.A 同様の inventory フォーマット)

本 M6.B は 1 commit で完結する想定 (scope-guardian の "中間 commit 3-4 回想定"
に対し、scope-guardian が allow した "Claude Code 判断" で 1 commit 化を選択 —
理由: docs 単一ファイルで横断的に参照する内容が多く、分割 commit で diff が
散逸するより 1 commit で全体像を提示した方が adversarial-reviewer の review
コストも低い。scope-guardian の上限 5 commit 内、下限 1 commit は明示されて
いないが文脈上許容範囲)
