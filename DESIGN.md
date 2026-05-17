# Flow-Lenia WebGPU Visualizer — 設計書

本書は Rust + WebAssembly + WebGPU で **Flow-Lenia (Plantec et al., 2025, Artificial Life journal, arXiv:2506.08569v1)** を厳密に再現し、ブラウザ上でリアルタイム可視化する実装の設計書である。

**実装の正典**は `papers/2506.08569v1.pdf` (2025年版) であり、Equation 番号は同論文を指す。副参照として `papers/2212.07906v2.pdf` (2023年版) と Moroz, 2020 "Reintegration tracking" を用いる。

設計合意フェーズの成果物であり、本書が承認されるまで**実装コードは一行も書かない**。末尾の「未確定事項と質問」セクションをまず承認願いたい。

---

## 0. 用語と論文式の早見表

| 記号 | 意味 | 論文 |
|---|---|---|
| `L` | 2次元格子 (CA の支持集合) | §2 |
| `C` | チャンネル数 | §2 |
| `A^t : L → R^C_{≥0}` | 時刻 t での活性度 (matter 濃度)。**unit range には縛られない** | §3 末 |
| `A_Σ^t(x) = Σ_i A_i^t(x)` | セル x の総質量 | Eq. 5 |
| `K_i` | 同心ガウスバンプ和カーネル | Eq. 1 |
| `G_i` | Gaussian growth function (range [-1, 1]) | Eq. 2 |
| `h ∈ R^|K|` | カーネル重み (パラメータ埋め込みなし版) | Eq. 3 |
| `U^t` | アフィニティマップ (旧 growth) | Eq. 3 / Eq. 7 |
| `α^t(x) = [(A_Σ/β_A)^n]_[0,1]` | 拡散項の重み | Eq. 5 |
| `F^t : L → (R^2)^C` | フロー (物質の瞬間速度) | Eq. 5 |
| `D(m, s)` | m 中心・側辺 2s の一様正方分布 | Eq. 6 |
| `Ω(x)` | セル x の領域 (側辺1) | Eq. 6 |
| `I_i(x', x) = ∫_Ω(x) D(x'+dt·F, s)` | x' から x への移動比率 | Eq. 6 |
| `P^t : L → R^|K|` | 各セルに局在化したカーネル重み | §3.1 / Eq. 7 |
| `s` | リインテグレーション分布の幅 (温度) | Eq. 6 |
| `dt` | 時間刻み (Table 1 で 0.2) | Table 1 |
| `n` | 拡散項立ち上がりの鋭さ (Table 1 で 2) | Table 1 |
| `β_A` | 臨界質量 (公式 JAX 実装の慣例値) | Eq. 5 |

**重要な計算量削減** (Eq. 6 周辺、§3 後半):
> we do not look at all cells to compute incoming matter as described by equation 6 but only at the neighborhood composed of cells whose **Chebyshev distance to the target cell is less than 5** (extended Moore neighborhood)

つまり、リインテグレーション・トラッキングの送受信は **11×11 = 121 セルの拡張 Moore 近傍**だけで完結する。これは設計の前提とする。

---

## 1. 技術スタック確定

### 1.1 Rust toolchain

- **edition: 2021** (2024 はまだ不要、依存クレートとの互換性最大化)
- **toolchain: stable (1.83+)** — 本設計が想定する `wgpu 29.x` は stable で動く。nightly は不要。
  - 理由: `wgpu` も `egui` も stable サポート。教育的価値を重視するため nightly を要求しない。
- `rust-toolchain.toml` をリポジトリに置き、コンポーネント `rustfmt`, `clippy`, `rust-src` を指定する。

### 1.2 wgpu / wasm 関連バージョン

`cargo search wgpu` 同等の確認 (Web 検索: docs.rs/wgpu, github.com/gfx-rs/wgpu releases) で **2026-03-26 リリースの `wgpu 29.0.3`** が最新安定版であることを確認した。

| crate | バージョン | 用途 |
|---|---|---|
| `wgpu` | `29.0` | GPU API |
| `winit` | `0.30` | ネイティブウィンドウ |
| `wasm-bindgen` | `0.2` (最新) | WASM ⇄ JS 境界 |
| `web-sys` | `0.3` | DOM / Canvas / WebGPU 型 |
| `console_error_panic_hook` | `0.1` | ブラウザでのパニック表示 |
| `bytemuck` | `1` | バッファシリアライズ |
| `glam` | `0.29` | ベクトル/行列 |
| `egui` / `eframe` | `0.32` 相当 (wgpu 29 対応版) | UI |
| `ndarray` | `0.16` | CPU 参照実装の補助 (CPU 版だけで利用) |
| `rand` / `rand_chacha` | `0.8` | 再現可能な乱数 (seed-able) |

**注意**: `egui` / `eframe` の最新リリース版が `wgpu 29` に追従しているかは実装着手時に確認する。追従が遅れていれば、次節 §1.4 の代替案として `egui + wgpu` をフレームワーク経由せず直接統合する経路に切り替える。

### 1.3 WebGPU 対応ブラウザの最低要件 (2026-05 時点)

Web 検索で確認した結果:

| ブラウザ | ステータス | 備考 |
|---|---|---|
| Chrome 113+ / Edge 113+ | ✅ Stable (デスクトップ・Android 12+) | 主ターゲット |
| Safari 26.0+ (macOS Tahoe 26 / iOS 26 / visionOS 26) | ✅ Stable | 主ターゲット |
| Firefox 141+ (Windows) / 145+ (macOS Apple Silicon) | ✅ Stable | 主ターゲット (Linux/Android は未対応) |

**Linux 系**は依然として穴がある (Chrome は driver-specific roll-out、Firefox は Nightly のみ) ため、起動時に `navigator.gpu` の存在チェックと「お使いのブラウザでは WebGPU が無効です」エラーパネル表示を必ず実装する (UI §7)。

### 1.4 ビルドツール選定

| 候補 | 採用判断 |
|---|---|
| **trunk** | ✅ 採用。`eframe` 公式テンプレートが trunk を使う。`trunk serve` でホットリロード可、CSS/HTML/WASM 一式の bundling が単一コマンドで済む。 |
| wasm-pack | △ 不採用 (補助)。trunk が内部で `wasm-bindgen` を呼ぶので wasm-pack の直叩きは不要。CI から `wasm-bindgen-cli` だけ直接呼ぶ可能性は残す。 |
| cargo-leptos | × leptos を使わないので不要。 |

### 1.5 UI フレームワーク選定

| 候補 | 評価 |
|---|---|
| **egui (+ eframe)** | ✅ **採用**。即時モード、WASM 対応成熟、`egui-wgpu` バックエンドで本プロジェクトの compute pass と同一 `wgpu::Device` を共有可能 (テクスチャを `register_native_texture` 経由で egui の `Image` として直接表示できる)。スライダ・グラフ・チェックボックスが揃っており、本設計の §7 UI 仕様を最短で満たす。 |
| leptos / yew | × フィット度低。reactive な DOM-first フレームワークで、低レベル wgpu 描画との同居が逆に煩雑になる。 |
| 素の JS + HTML + Rust core | × 二言語管理コスト。教育的価値の観点でも Rust 一本化を優先。 |

**結論**: `eframe { features = ["wgpu"] }` を採用し、ネイティブ・Web 両ターゲットを単一バイナリで賄う。compute は `eframe::App::update` 内で自前の wgpu パイプラインを `egui-wgpu` の `CallbackTrait` 経由でディスパッチする。

### 1.6 数値計算ライブラリ

- **CPU 参照実装** (`core/cpu_reference.rs`): `ndarray` を採用。`rustfft` の必要性は §4 で判断するが、当面は使わない (直接畳み込みで充分小さい)。
- **GPU 側**: `glam` のみ。`ndarray` は GPU バッファに触らない。

### 1.7 FFT の扱い

論文の Table 1 では `R ∈ [2, 25]` で、最大半径 25 → カーネル直径 51。256×256 グリッドで kernel が 45 個ある場合、直接畳み込みは `256·256·51·51·45 ≈ 1.5 × 10^10 ops / step` でフルチャンネル想定では重い。一方 FFT 畳み込みなら 1 step あたり `O(C · N log N + K · N log N)` で済む。

しかし WGSL での 2D FFT 実装は非自明 (Stockham カーネル × 2 軸 × 複素対応 × 多チャンネル) で、教育的価値もぼやける。

**初期方針**:
- **M2 まで: 直接畳み込み**。中間目標の 128×128 / R≤13 / 10 kernels では 1 step 約 `1.7×10^9 ops` で 30 FPS は射程内。GPU の並列性で実測する。
- **M6 性能チューニング段階で FFT 化**を検討。Stockham FFT を別 module 化して差し替え可能にする。
- **カーネル `K_i` 自体は CPU で 1 回だけ計算**して GPU バッファに転送する。パラメータ再ランダム化時のみ再計算。

---

## 2. アーキテクチャ図

```
flow-lenia/
├── crates/
│   ├── flow-lenia-core/      # プラットフォーム非依存
│   │   ├── src/
│   │   │   ├── params.rs           # KernelParams, FlowParams, Table 1 サンプリング
│   │   │   ├── kernel.rs           # Eq. 1 の K_i を CPU で precompute
│   │   │   ├── state.rs            # A, U, F, P の論理レイアウト定義
│   │   │   ├── cpu_reference.rs    # 参照実装 (ndarray, 単スレッド, 正確性検証専用)
│   │   │   ├── shaders/            # WGSL ソース (str! で埋め込み)
│   │   │   │   ├── convolve.wgsl
│   │   │   │   ├── affinity.wgsl
│   │   │   │   ├── gradient.wgsl
│   │   │   │   ├── flow.wgsl
│   │   │   │   ├── reintegrate.wgsl
│   │   │   │   ├── params_update.wgsl
│   │   │   │   └── visualize.wgsl
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   │
│   ├── flow-lenia-gpu/       # wgpu セットアップ・パイプライン構築・ディスパッチ
│   │   ├── src/
│   │   │   ├── context.rs    # Instance/Adapter/Device/Queue 取得 (native + web)
│   │   │   ├── buffers.rs    # ダブルバッファ管理 (A_in/A_out, P_in/P_out)
│   │   │   ├── pipelines.rs  # 7 つの compute pipeline + 1 render pipeline
│   │   │   ├── dispatch.rs   # 1 step = 6 compute passes をエンコード
│   │   │   ├── readback.rs   # 質量検証用に総和を CPU に取り戻す path
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   │
│   ├── flow-lenia-ui/        # egui スライダ・コントロール・グラフ
│   │   └── src/lib.rs
│   │
│   └── flow-lenia-app/       # eframe アプリケーション。bin ターゲット
│       ├── src/
│       │   ├── app.rs        # eframe::App 実装
│       │   ├── main.rs       # ネイティブエントリ
│       │   └── lib.rs        # WASM 用エントリ (wasm_bindgen(start))
│       ├── index.html
│       └── Trunk.toml
│
├── tests/                    # ワークスペース横断のインテグレーションテスト
│   ├── mass_conservation.rs
│   ├── reference_vs_gpu.rs
│   └── regression_fixtures/  # 黄金パラメータ・既知出力 (npy or bin)
│
├── papers/                   # 既存
├── DESIGN.md                 # 本書
├── Cargo.toml                # workspace
└── rust-toolchain.toml
```

### 2.1 依存方向

```
flow-lenia-app  ──→  flow-lenia-ui  ──→  flow-lenia-gpu  ──→  flow-lenia-core
        │                                                            ↑
        └────────────────────────────────────────────────────────────┘
                       (App は core の型/パラメータも直接使う)
```

`core` は `wgpu` には依存しない (WGSL ソースは `&'static str` として保持するだけ)。`gpu` は `wgpu` に依存し `core` の型を使う。これにより `core` を `no_std`-friendly に保ち、CPU 参照実装をネイティブテストで独立に走らせやすくする。

### 2.2 ランタイム実行ループ (簡略)

```
┌─────────────── eframe::App::update (毎フレーム) ───────────────┐
│                                                                  │
│  1. UI: スライダ更新 → SimulationParams 反映                     │
│  2. もし running なら steps_per_frame 回 dispatch_step を呼ぶ    │
│  3. 1 frame に 1 回 readback (オプション: 質量検証 ON 時のみ)    │
│  4. egui で 状態テクスチャを Image として描画 (visualize.wgsl    │
│     の出力テクスチャを共有)                                      │
│  5. 統計 (FPS, mass error) を表示                                │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. データレイアウト設計

### 3.1 GPU バッファ vs テクスチャの選定

| 物理量 | 種別 | 形式 | 理由 |
|---|---|---|---|
| `A^t` 活性度 | **Storage Texture 2D Array** | `Rgba16Float` (4 ch まで束ねる) または `R32Float` Array | テクスチャサンプリングと相性が良く、可視化パスでそのまま `texture_2d` として読める。質量保存の数値検証のため f32 が望ましい→**R32Float の `texture_2d_array<f32>` を採用**。`C` チャンネルを array layer に割り当て。 |
| `U^t` アフィニティ | Storage Texture 2D Array (`R32Float`) | 中間値、次パスで読み戻す | 同上 |
| `F^t` フロー | Storage Texture 2D Array (`Rg32Float`) | 2D ベクトル、C 個 | 各チャンネル独立 |
| `∇U`, `∇A_Σ` | Storage Texture (`Rg32Float`) | 中間 | 一時的でも別テクスチャを確保 (in-place gradient はやらない) |
| `P^t` パラメータマップ | **Storage Buffer** `array<f32>` (size: W·H·|K|) | バインディング数を抑え、kernel index 軸に沿ったストライドで素直にインデックスできる | テクスチャの max array layers (通常 256) を超える可能性 (例: 45 kernels × 別 entry → OK だが、可搬性のため buffer 採用) |
| `K_i` カーネル | **Storage Buffer** `array<f32>` (kernel side² × |K|) | 形状が不均一 (半径ごとに違うが最大半径でパディング)。CPU で 1 回計算して固定。 |
| カーネル meta (radius, c0_i, c1_i) | Uniform Buffer `array<KernelMeta>` (|K| 個) | 16-byte align で並べる | |
| 成長関数 μ, σ | Uniform Buffer `array<vec2<f32>>` (|K| 個) | 軽量 |
| グローバルパラメータ (`dt`, `s`, `n`, `β_A`, `W`, `H`, `C`, `|K|`) | Uniform Buffer (32 bytes 程度) | 毎フレーム書き換える可能性 (UI スライダ) |

### 3.2 ダブルバッファリング

各 frame で次の物理量だけ `_in` / `_out` のペアを持つ:
- `A` (Eq. 6 で前ステップを読みつつ次ステップを書く)
- `P` (Eq. 8 で同じく)

`U`, `F`, `∇U`, `∇A_Σ` は同 step 内で生成・消費される中間値なので 1 個でよい。

ピンポンは `step_parity: bool` を CPU 側で持ち、`dispatch_step` 内で `bind_group_a`, `bind_group_b` を切り替える。

### 3.3 ビット幅の選定

- **GPU 計算は f32 で統一**する (`R32Float`)。
- 質量保存の検証精度を保つため f16 は使わない (F16 は WebGPU では `shader-f16` feature 拡張、ブラウザ間差異あり)。
- 可視化テクスチャ (画面に出す最終 RGBA) のみ `Rgba8Unorm` に変換する。

### 3.4 バインディング表 (case: 拡張版 Eq. 7 で全パス共用するつもりの初期案)

| パス | group | binding | リソース | アクセス |
|---|---|---|---|---|
| convolve | 0 | 0 | `A_in` (R32Float arr) | read |
| convolve | 0 | 1 | `U_out` (R32Float arr) | write |
| convolve | 0 | 2 | `kernels` buffer | read |
| convolve | 0 | 3 | `kernel_meta` uniform | read |
| convolve | 0 | 4 | `globals` uniform | read |
| convolve | 0 | 5 | `P_in` buffer (Eq. 7) | read |
| affinity_growth | 1 | 0..3 | `U_inout`, `growth_params`, `globals`, `kernel_meta` | rw / read |
| gradient | 2 | 0..3 | `U_in`, `A_in`, `gradU_out`, `gradAsum_out` | rw |
| flow | 3 | 0..4 | `gradU`, `gradAsum`, `A_in`, `F_out`, `globals` | rw |
| reintegrate | 4 | 0..4 | `A_in`, `F_in`, `A_out`, `P_in`, `globals` | rw |
| params_update | 5 | 0..3 | `A_in`, `P_in`, `P_out`, `globals` (+ RNG state) | rw |
| visualize | 6 | 0..2 | `A_in`, `screen_tex`, `globals` | r → w |

詳細な struct レイアウトは実装着手前に WGSL prelude として 1 ファイルに集約する (`shaders/types.wgsl`)。

---

## 4. シェーダパイプライン設計

### 4.1 1 step あたりのパス構成

論文の更新規則をパスに分解する:

| # | 名前 | 入力 | 出力 | 対応式 | dispatch サイズ |
|---|---|---|---|---|---|
| 1 | **convolve** | A_in, K, P_in | U_tmp_pre_growth (Σ_i P_i · (K_i ∗ A_{c0_i})) | Eq. 7 の畳み込み部 | (W/8, H/8, |K|) — 1 invocation = 1 セル × 1 カーネル |
| 1b | **accumulate_growth** | U_tmp_pre_growth, growth μ,σ | U (channel-major) | Eq. 7 の G_i 適用 + [c1_i=j] 集約 | (W/8, H/8, C) |
| 2 | **gradient** | U, A_in | ∇U, ∇A_Σ (Sobel) | Eq. 5 の Sobel 部 | (W/8, H/8, max(C, 1)) |
| 3 | **flow** | ∇U, ∇A_Σ, A_in | F | Eq. 5 全体 (α 適用込み) | (W/8, H/8, C) |
| 4 | **reintegrate** | A_in, F, (P_in for Eq. 8 sampling stats) | A_out, P_out | Eq. 6 + Eq. 8 | (W/8, H/8) — 各セルが Chebyshev≤5 範囲を**受信側で集計** |
| 5 | **(swap)** | — | — | ダブルバッファのインデックス反転 (CPU 側) | — |
| R | **visualize** | A_in (next step), P_in | screen_tex (Rgba8Unorm) | — | (W/8, H/8) |

**設計判断 (重要)**:

- **Convolution は spatial direct で実装** (M2 まで)。各 invocation は 1 セル × 1 カーネルを担当し、半径 R 以内の近傍を全探索。`R_max=25` 時は `(2R+1)² = 2601` MAD per kernel-cell。WGSL の `for` ループで素朴に書く。
- **`accumulate_growth` を別パスに分けた理由**: Eq. 7 では「カーネル i ごとに `P_i(x)` で重み付けし、生成チャンネル `c1_i` に加算」が必要。1 のパスで `c1_i` 別の出力にアトミック加算するのは f32 atomic が WebGPU では `shader-atomic-float` の拡張で未保証。よって i 軸を畳んだ中間 (K 個分のスカラー出力 per cell) を作り、次パスで C チャンネル別に集約する。**メモリ消費は W·H·|K|·4 byte**。256² × 45 × 4 = ~12 MB。許容範囲。
- **Reintegration を受信側で実装する理由**: Eq. 6 は「各ターゲット x が、x からチェビシェフ距離≤5 の x' に対し `A_i(x') · I_i(x', x)` を集計」と書ける。これは送信側ループ(各 x' が周辺に書き込み)では race condition が出るので、**受信側で 11×11 ループを回す**形に再定式化する。これは公式 JAX 実装と同じパターン。
- **Eq. 8 (パラメータ更新) も reintegrate パスに統合**できるが、責務分離のため別パスにする。同じ 11×11 近傍を 2 回回るが、教育的価値を優先。M5 で性能チューニング時に結合検討。

### 4.2 各パスの WGSL スケルトン(設計レベル、コード本体ではない)

各 WGSL ファイル冒頭に必ず以下のコメントを書く:

```wgsl
// Flow-Lenia <pass_name> shader
// Implements <equation reference, e.g. "Eq. 6 (reintegration tracking)"> of
// Plantec et al. 2025, "Flow-Lenia: Emergent Evolutionary Dynamics in Mass
// Conservative Continuous Cellular Automata" (arXiv:2506.08569v1).
//
// References:
//   - Eq. 6: A^{t+dt}_i(x) = Σ_{x'} A^t_i(x') · I_i(x', x)
//   - Eq. 6: I_i(x', x) = ∫_Ω(x) D(x' + dt·F^t_i(x'), s)
//   - Chebyshev distance ≤ 5 optimization (Section 3, paragraph after Eq. 6)
//   - Reintegration tracking: Moroz 2020, https://michaelmoroz.github.io/Reintegration-Tracking/
```

### 4.3 リインテグレーション・トラッキングの具体実装方針

「`I_i(x', x) = ∫_Ω(x) D(x' + dt·F_i(x'), s)`」を計算する。`D` は一様正方分布 (中心 `m = x'+dt·F`、側辺 `2s`)、`Ω(x)` も側辺 1 の正方形。

→ **2 つの軸並行な正方形の重なり面積を `2s` で正規化**したものが `I`。

```text
overlap_x = max(0, min(m.x + s, x + 0.5) - max(m.x - s, x - 0.5))
overlap_y = max(0, min(m.y + s, y + 0.5) - max(m.y - s, y - 0.5))
I = (overlap_x * overlap_y) / (2s * 2s)
```

`s` は通常 `[0.1, 2.0]` の範囲。`s + dt·|F|` がチェビシェフ距離 5 を超えるとカットされる(物理的に意味のある近似)。境界条件は **トロイダル (周期境界)** を採用する (論文と公式実装に合わせる)。

### 4.4 数値演算順序の固定

GPU 浮動小数演算は順序で結果が変わる。テストでビット完全一致を狙わず「相対誤差 < 1e-4」を許容する。ただし参照実装と GPU 実装で **加算順序を揃える**よう、Eq. 3 / Eq. 6 のループ順を仕様化する:
- Eq. 3: i (カーネルインデックス) 昇順
- Eq. 6: dy, dx の二重ループ(dy 外、dx 内、いずれも -5..=5)

---

## 5. 数値的正しさの検証戦略

### 5.1 ユニットテスト (`crates/flow-lenia-core/tests/`)

- `test_kernel_normalization`: `Σ_x K_i(x) ≈ 1` (Eq. 1 の制約)
- `test_growth_range`: `G_i(x) ∈ [-1, 1]` for x ∈ [0, 1] (Eq. 2)
- `test_overlap_area`: §4.3 の重なり面積関数の単体テスト (8 ケース: 完全重なり、無重なり、部分重なり、ゼロ幅エッジ等)

### 5.2 参照実装との一致 (`tests/reference_vs_gpu.rs`)

- 16×16 グリッド、C=1、|K|=3 で固定シードランダムパラメータ
- 50 ステップ走らせ、CPU 参照実装と GPU 実装の `A` を比較
- 許容誤差: 各セル `|a_cpu - a_gpu| / (|a_cpu| + 1e-6) < 1e-4`
- これにより、convolve / gradient / flow / reintegrate の各パスが正しく結合しているか保証

### 5.3 質量保存プロパティテスト (`tests/mass_conservation.rs`)

- ランダム初期状態 (32×32 / 64×64 / 128×128 × seeds × C ∈ {1,2,3}) で 1000 ステップ
- **チャンネルごとの** 総質量 `Σ_x A_i(x)` の初期値からの相対誤差 `< 1e-3`
- パラメータ埋め込み版 (Eq. 7/8) でも同様にテスト

注: Eq. 6 の重なり面積を WGSL の `f32` で計算するため、1000 ステップで蓄積する誤差は素朴には `O(N·step·ε_machine)` だが、再分配の合計は `Σ I = 1` が解析的に成立する設計なので、誤差は概ね step 数比例の `O(step · ε)` に留まる想定。実測で 1e-3 を超えるなら double 精度参照実装で再評価。

### 5.4 回帰テスト

- 論文 Figure 4 のパターンに近づくシード/パラメータ集合 (発見次第 `tests/regression_fixtures/` に格納) で固定 step 後の状態を `npy` 形式で保存。
- CI で diff チェック (許容誤差は §5.2 と同じ)。
- 初期 fixture は M2 完了時に生成して固定する。

### 5.5 ベンチマーク (`benches/`)

- `criterion` でネイティブ CPU 参照実装の 1 step 時間を測る
- WASM 側は手動 (`performance.now()`) で FPS を画面表示

---

## 6. パラメータの初期化

### 6.1 サンプリング (Table 1 準拠)

`flow-lenia-core/src/params.rs::sample_random(rng, settings)`:

```text
settings = { num_kernels, num_channels, grid_size }
output: KernelParams = {
  R:           sample u ∈ [2, 25]              (グローバル 1 個)
  for each kernel i in 0..num_kernels:
    c0_i:      sample uniform from 0..num_channels
    c1_i:      sample uniform from 0..num_channels
    r_i:       sample ∈ [0.2, 1.0]
    a_i:       [sample ∈ [0,1]; 3]   (k=3 rings)
    b_i:       [sample ∈ [0,1]; 3]
    w_i:       [sample ∈ [0.01, 0.5]; 3]
    h_i:       sample ∈ [0, 1]
    μ_i:       sample ∈ [0.05, 0.5]
    σ_i:       sample ∈ [0.001, 0.2]
  s:           default 0.65        (UI で可変)
  n:           default 2
  dt:          default 0.2
  β_A:         default 4.0         (公式実装の慣例値、UI で可変)
}
```

### 6.2 シード再現性

- `rand_chacha::ChaCha8Rng::seed_from_u64(seed)` を採用 (再現可能・スレッド無関係)
- UI に `seed: u64` 入力欄。空欄なら `getrandom` で生成し、その値を UI に表示してコピペ可能にする。
- パラメータ JSON エクスポート/インポート (M4 以降)。

### 6.3 初期状態 `A^0`

論文 (§4.1, §4.3.2 vanilla) に合わせ、グリッド中央に 20×20 ~ 40×40 のパッチを置き、各セル `A_i ∈ U(0, 1)` を独立サンプル。残りは 0。
- UI で patch サイズと creature 数 (1 vs 多体) を選べるようにする (M5 で multi-species 用)。

---

## 7. UI 仕様

`flow-lenia-ui` に `ControlsPanel` を実装し、左サイドバーに配置:

| グループ | コントロール | デフォルト | 範囲 |
|---|---|---|---|
| Grid | `grid_size: enum` | 128 | {64, 128, 256, 512} |
| Grid | `channels: u32` | 3 | 1..=3 |
| Grid | `num_kernels: enum` | 10 | {5, 10, 20, 45} |
| Sim | ▶ / ⏸ ボタン | 停止 | — |
| Sim | ⏭ 1-step ボタン | — | — |
| Sim | ⟲ Reset (現パラメータで状態のみ初期化) | — | — |
| Sim | 🎲 Re-randomize (パラメータ + 状態を再サンプル) | — | — |
| Sim | `seed: u64` 入力 + Copy | random | — |
| Sim | `steps_per_frame: u32` (速度) | 1 | 1..=32 |
| Phys | `dt` スライダ | 0.2 | [0.01, 0.5] |
| Phys | `s` (temperature) スライダ | 0.65 | [0.1, 2.0] |
| Phys | `n` スライダ | 2.0 | [0.5, 8.0] |
| Phys | `β_A` (critical mass) スライダ | 4.0 | [0.5, 20.0] |
| Mode | チェック: parameter embedding (Eq. 7) | OFF | bool |
| Mode | チェック: stochastic sampling (Eq. 8) | ON | bool — embedding ON 時のみ |
| Viz | チャンネル → RGB マッピング | C1→R, C2→G, C3→B | — |
| Viz | gamma スライダ | 1.0 | [0.2, 3.0] |
| Stats | 現在総質量 (per channel) | — | — |
| Stats | 初期からの相対誤差 (%) | — | — |
| Stats | FPS / step 時間 (ms) | — | — |

**起動時の WebGPU 非対応エラー**:
- `wgpu::Instance::new` → `request_adapter` が失敗した場合、egui で全画面エラーパネルを出し、対応ブラウザリンク (https://caniuse.com/webgpu) を表示。

---

## 8. 段階的実装計画

### M1: CPU 版 Flow-Lenia がネイティブで動く
- **成果物**: `flow-lenia-core` の `cpu_reference.rs` で Eq. 1, 2, 3, 5, 6 を ndarray で素朴実装
- **完了条件**:
  - `cargo test -p flow-lenia-core` グリーン
  - 64×64 / C=1 / |K|=5 で 100 steps 走り、総質量誤差 < 1e-6
  - 簡易 dump (ppm) で目視確認

### M2: wgpu compute shader 版がネイティブで動く
- **成果物**: `flow-lenia-gpu`、winit ネイティブウィンドウで描画
- **完了条件**:
  - 全 6 compute pass + visualize 動作
  - `tests/reference_vs_gpu.rs` グリーン (16×16, 50 steps, 相対誤差 < 1e-4)
  - `tests/mass_conservation.rs` グリーン (32×32, 500 steps, 相対誤差 < 1e-3)
  - 回帰 fixture を 1 つ確定

### M3: WASM ビルド + ブラウザ WebGPU 動作
- **成果物**: `trunk build --release` で `dist/` に成果物、ローカルサーバで開ける
- **完了条件**:
  - Chrome 最新版で WebGPU 経由で動く
  - M2 と同じシードで出力が一致 (visual diff + 質量チェック)
  - WebGPU 非対応ブラウザでフォールバックエラー表示

### M4: UI 統合・リアルタイム可視化
- **成果物**: egui コントロール一式、状態テクスチャ表示、統計
- **完了条件**:
  - §7 の全コントロール動作
  - 128×128 / C=3 / |K|=45 で 30 FPS 以上 (実測ベンチに合格)
  - 最低 1 つ「明らかに動く creature」が見える固定シードのデモを README に掲載

### M5: パラメータ埋め込み・マルチ種シミュレーション
- **成果物**: Eq. 7, Eq. 8 (stochastic & deterministic) の有効化、初期状態の multi-patch 配置
- **完了条件**:
  - parameter embedding ON で `tests/mass_conservation.rs` グリーン
  - UI から多体配置 (64 creatures × ランダム seeds) を生成できる
  - PCA 可視化はスコープ外 (将来) 、ここでは「色の異なる creature が共存する」状態の動作確認

### M6: 性能チューニング・ドキュメント整備
- **成果物**:
  - 256×256 / C=3 / |K|=45 で 60 FPS 達成 (Apple M1 以上想定)
  - 必要なら FFT 畳み込み実装
  - README、CHANGELOG、シェーダ内コメント整備
  - `cargo doc` で公開可能なドキュメント
- **完了条件**:
  - ベンチ表 (各 grid_size × C × |K| × backend での FPS) を README に掲載
  - シェーダ内の論文式番号コメントが全パスに揃っている

---

## 9. 守る原則の運用

| 原則 | 運用 |
|---|---|
| 論文との対応をコメントに残す | 全 WGSL ファイル冒頭 (§4.2) + Eq. を実装したループ直上 |
| WGSL は型と束縛に厳格 | §3.4 表に従い、`shaders/types.wgsl` を prelude として全パス共有 |
| wgpu バージョン互換性 | Cargo.toml で `wgpu = "=29.0.3"` のように pin。CI で `cargo update --dry-run` 警告化 |
| ブラウザ WebGPU 互換 | §1.3 + §7 起動時チェック + README に対応表 |
| 巨大シェーダ非作成 | §4.1 の 7 ファイル分割を厳守 |

---

## 10. 未確定事項と質問

実装着手前に確定すべき項目を **Q1〜Q9** として列挙する。回答を待って本書を更新する。

### Q1. 主ターゲットブラウザの優先度

Chrome 最新 / Firefox 最新 / Safari 最新 / Edge 最新 の **どれを最優先**で動作確認しますか? また Firefox は Linux で WebGPU 未対応のため、Linux ユーザは Chrome に誘導してよいですか?

> **設計上の仮置き**: 第一優先 = Chrome (macOS / Windows / Linux)、第二優先 = Safari 26 (macOS)、ベストエフォート = Firefox 141+ (Win/mac)。

### Q2. ネイティブビルドのターゲット OS / GPU

開発機は Apple Silicon (Metal) と想定してよいですか? それとも Linux/Vulkan や Windows/DirectX 12 でも動作検証が必要ですか?

> **設計上の仮置き**: M2/M6 性能基準は **Apple M1 (Metal)** を基準にします。Linux/Windows は CI で `cargo check --target` レベルの確認のみ。

### Q3. β_A (臨界質量) のデフォルト値

論文本文では β_A の数値は明示されていません (Eq. 5 で文字定義のみ)。公式 JAX 実装の慣例値 `4.0` を採用する仮置きで良いですか? それとも公式コードを当たって正確な値を反映しますか?

> **設計上の仮置き**: 4.0。公式実装を参照次第アップデート。

### Q4. 食物 / 散逸モデルの実装

論文 §4.3.2 では `vanilla` / `dissipative` / `food` の 3 モデルがあります。**vanilla だけ**を実装範囲にしますか? それとも 3 つ全部?

> **設計上の仮置き**: M5 までは vanilla のみ。dissipative/food は M6 以降の stretch goal。

### Q5. 進化的最適化 (Evolutionary Strategies)

論文 §4.2 では OpenES でパラメータ最適化を行います。これは可視化ツールの範囲外と理解してよいですか? (実装規模が大きすぎる)

> **設計上の仮置き**: スコープ外。ランダムサンプリングと UI からの手動編集のみ。

### Q6. 境界条件

論文では明示されないがコードはトロイダル (周期境界) です。本実装も周期境界で良いですか? それとも反射壁 / 吸収壁などの切替を入れますか?

> **設計上の仮置き**: トロイダルのみ。

### Q7. FFT 畳み込みの実装スコープ

§1.7 / §4.1 の通り、初期は直接畳み込みで実装し、性能が不足したら FFT 化を検討するアプローチで良いですか? それとも初期から FFT を入れる方が望ましいですか?

> **設計上の仮置き**: 直接畳み込みから開始、M6 で必要なら FFT 化。

### Q8. 量子化 / f16 の使用可否

WebGPU の `shader-f16` 拡張は Chrome では使えるが Safari/Firefox では未保証です。**f32 一本**で行く方針 (質量保存の精度優先) で確定して良いですか?

> **設計上の仮置き**: f32 のみ。

### Q9. ライブデモのホスティング

最終成果物をどこかに公開する想定はありますか? (GitHub Pages, Cloudflare Pages, etc.) これによって `index.html` の base path や trunk 設定が変わります。

> **設計上の仮置き**: ローカルのみ前提 (`trunk serve`)。デプロイは別タスク。

---

## 11. 承認後にやること (このフェーズでは実行しない)

承認をいただいたら、以下の順で実装に着手します:

1. `Cargo.toml` workspace と 4 つの crate の雛形作成
2. `rust-toolchain.toml` 配置
3. `flow-lenia-core` の Eq.1/2 + パラメータサンプリングを TDD で
4. `flow-lenia-core::cpu_reference` で Eq.3 → Eq.5 → Eq.6 を順に
5. M1 完了テスト → M2 着手 (wgpu pipeline、最小ループ)
6. 以降は §8 のマイルストーンに従う

---

**設計レビューをお願いします。** 特に §10 の Q1〜Q9 について回答をいただきたいです。仮置きのまま進めてよいなら「仮置き OK」とご返答ください。
