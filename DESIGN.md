# Flow-Lenia WebGPU Visualizer — 設計書 (Rev. 4.1)

本書は Rust + WebAssembly + WebGPU で **Flow-Lenia (Plantec et al., 2025, Artificial Life journal, arXiv:2506.08569v1)** を厳密に再現し、ブラウザ上でリアルタイム可視化する実装の設計書である。

**実装の正典**は `papers/2506.08569v1.pdf` (2025年版) であり、Equation 番号は同論文を指す。副参照として `papers/2212.07906v2.pdf` (2023年版)、Moroz, 2020 "Reintegration tracking"、**公式 JAX 実装** `references/FlowLenia-jax/` (commit `dce428c`, 2024-02-08) を用いる。JAX 実装の精読結果は `references/JAX_NOTES.md` を参照。

## Rev. 履歴

### Rev.4.1 (2026-05、M1.1 実装時の指摘)

- Rust toolchain: 1.76 → **1.87** (transitive 依存の edition 2024 要求対応、wgpu 29 移行余地)
- Cargo.lock 運用方針を §1.8 として新規追加
- 設計書バグ認識: 設計合意フェーズ Rev.4 までで transitive 依存の MSRV 影響を読み切れなかった

### Rev.4 (2026-05、ユーザー承認条件の反映)

- **Q3 デフォルト反転**: デフォルト = **JAX 互換** (per-channel α、β_A=2.0、n=2.0)。`paper_strict` チェックボックス ON で論文 Eq. 5 (全 C 共通 α、A_Σ ベース) に切替。`β_A` と `n` は両モードで UI スライダ可変
- **Q3c 拡充**: JAX 式採用は維持。WGSL カーネルファイル冒頭に**論文式 / JAX 式 / 3 つの差異とその意図**を 10 行以上のコメントブロックで残す (§4.6 にテンプレ提示)
- **Q3d 明記**: 提案通り、デフォルト論文 / `jax_compat` で JAX 式。WGSL コメントに **JAX 実装の `jr.PRNGKey(42)` 固定シードバグを明記** (§4.6 にテンプレ提示)、`references/JAX_NOTES.md` §11 NOTE-11-A / NOTE-11-B を引用
- **Q6 補強**: 提案通り (torus デフォルト + wall 切替 + Sobel も border 追従)。加えて **M4 UI 仕様で「初期パッチの配置は画面中心 ± grid_size/4 の範囲に制限」** (§7)
- **Q7 別案採用**: **M1 では CPU 参照も直接畳み込みで実装** (`rustfft` 依存削除)。許容誤差は CPU 直接畳み込み vs GPU 直接畳み込みで相対誤差 < 1e-3。**M6 で torus 境界限定の FFT 化**を **optional** ターゲットとして追加、JAX とのビット精度近似テストを試金石にする
- **§3.4 補強**: `F` (フロー) は reintegrate / params_update の両方で使うため、**両パスで読み取り専用に共有する単一バッファ**で運用する設計を明示
- **§4.4 検証**: WGSL Sobel の符号が JAX と一致することを保証する **`test_sobel_sign` ユニットテストを M1 完了条件に追加**

### Rev.3 (2026-05、JAX 公式実装 commit `dce428c` 精読後)

- **Q1A 確定**: Chebyshev = **11×11** (`-5..=5`、121 セル)。JAX `reintegration_tracking.py:39` で確認。論文 "less than 5" は実装上 "≤ 5"
- **Q3b 確定**: Sobel = **正規化なし**、`[[1,0,-1],[2,0,-2],[1,0,-1]]` を生で使用 (JAX `utils.py:16-37`)
- **Q3 拡張**: 論文 Eq. 5 と JAX 実装 (`A_c / C` or `A_c / 2`、`n=2` 固定) が**乖離**。論文準拠 + JAX 互換切替で実装
- **新 Q3c**: カーネル定義 (Eq. 1) も論文と JAX で差異 ((R+15)*r スケール + sigmoid マスク + `w` 解釈)
- **新 Q3d**: Eq. 8 softmax も論文と JAX で差異 (`exp(A·I)` vs `A·I` そのまま)
- **新規パラメータ `dd`**: UI スライダで近傍範囲を 3..=7 で可変、デフォルト 5
- **`area` 上限 clip `min(1, 2σ)`** を §4.3 に追記
- **`ma = dd - sigma` でフロー振幅 clip** を §4.3 に追記
- **mutation patch size**: UI スライダで可変、JAX デフォルト 20、論文準拠 10
- **パラメータ範囲微修正**: `s ∈ [0.001, 0.18]`, `h ∈ [0.010, 1.00]`, `b ∈ [0.001, 1.00]^3`

### Rev.2 (2026-05、査読指摘 14 件)

- Eq. 7 のパス分割における数学的順序の修正 (§4.1)
- Eq. 8 が総質量 A_Σ を使うことの明記
- wgpu / eframe バージョン整合性の見直し
- 変異ビームを M5 完了条件に追加
- メモリ予算の上限分析
- WebGPU device features / limits 宣言
- 演算順序の修正

### Rev.1 (初版)

設計合意フェーズの成果物であり、本書 Rev.4 が承認されるまで**実装コードは一行も書かない**。

---

## 0. 用語と論文式の早見表

| 記号 | 意味 | 論文 |
|---|---|---|
| `L` | 2次元格子 (CA の支持集合) | §2 |
| `C` | チャンネル数 (UI 制約: 1〜3、アルゴリズム上は任意) | §2 |
| `A^t : L → R^C_{≥0}` | 時刻 t での活性度 (matter 濃度)。**unit range には縛られない** | §3 末 |
| `A_Σ^t(x) = Σ_i A_i^t(x)` | セル x の総質量 (全チャンネル和) | Eq. 5 |
| `K_i` | 同心ガウスバンプ和カーネル | Eq. 1 |
| `G_i` | Gaussian growth function (range [-1, 1]) | Eq. 2 |
| `h ∈ R^|K|` | カーネル重み (パラメータ埋め込みなし版) | Eq. 3 |
| `U^t` | アフィニティマップ (旧 growth) | Eq. 3 / Eq. 7 |
| `α^t(x)` | 拡散項の重み (詳細はモード依存、§4.1.5 参照) | Eq. 5 |
| `F^t : L → (R^2)^C` | フロー (物質の瞬間速度) | Eq. 5 |
| `D(m, s)` | m 中心・側辺 2s の一様正方分布 | Eq. 6 |
| `Ω(x)` | セル x の領域 (側辺1) | Eq. 6 |
| `I_i(x', x) = ∫_Ω(x) D(x'+dt·F, s)` | x' から x への移動比率 | Eq. 6 |
| `P^t : L → R^|K|` | 各セルに局在化したカーネル重み | §3.1 / Eq. 7 |
| `s` (= σ) | リインテグレーション分布の幅 (温度) | Eq. 6 |
| `dt` | 時間刻み (Table 1 で 0.2) | Table 1 |
| `n` | 拡散項立ち上がりの鋭さ (Table 1 で 2) | Table 1 |
| `β_A` | 臨界質量 (デフォルト 2.0、UI 可変) | Eq. 5 |
| `r_i` | カーネル i のスケール係数 ∈ [0.2, 1.0] | Eq. 1 / Table 1 |
| `R` | 全カーネル共通の最大半径 ∈ [2, 25] | Eq. 1 / Table 1 |
| `dd` | リインテグレーション近傍範囲、デフォルト 5、UI で 3..=7 (Rev.3 追加) | JAX 実装 |

**重要な計算量削減** (Eq. 6 周辺、§3 後半 / 2025 論文原文):
> "we do not look at all cells to compute incoming matter as described by equation 6 but only at the neighborhood composed of cells whose **Chebyshev distance to the target cell is less than 5** (extended Moore neighborhood)"

**Q1A 確定 (Rev.3)**: JAX 公式実装 `reintegration_tracking.py:39` で `for dx in range(-dd, dd+1):` (`dd=5`) を確認。範囲は **`-5..=5` の 11×11 = 121 セル**。論文の "less than 5" は「≤ 5」の意。物理的にも、フロー振幅の上限 `|dt·F| ≤ dd - σ`、分布幅 `σ` を足すと最遠到達点 `= dd`。`dd` は UI パラメータとして公開し、デフォルト 5、可変範囲 3..=7。

---

## 1. 技術スタック確定

### 1.1 Rust toolchain

- **edition: 2021** (2024 は依存クレートとの互換性を理由に保留)
- **toolchain: stable 1.87.0** で pin
  - 当初 1.76 を検討したが、transitive 依存 (`indexmap 2.14+` 等) が
    `edition = "2024"` を要求しており、これは Rust 1.85+ で stable 化された
    cargo 機能のため、1.76 では cargo check が通らない
  - 1.87 を採用する理由:
    - edition 2024 を要求する transitive 依存への対応 (現在 + 将来)
    - 将来 wgpu 29 系へ移行する選択肢を残す (wgpu 29 の MSRV = 1.87)
    - 2025-05 リリースで1年経過、十分 stable
- `rust-toolchain.toml` をリポジトリに置き、コンポーネント `rustfmt`, `clippy`, `rust-src` を指定
- nightly 不要

### 1.2 wgpu / wasm 関連バージョン

調査結果 (2026-05 時点):

- `wgpu` 最新安定版: **29.0.3** (2026-03 リリース、MSRV 1.87)
- `egui` / `eframe` 最新安定版: **0.34.1** (2026-03)、**wgpu 25 にバンドル**

eframe は伝統的に wgpu のメジャーバージョンを 2〜4 リリース遅れて追従する。`wgpu 29 + eframe 0.34` の併用は依存解決失敗または実行時不整合を起こす。

**採用**: **(A) wgpu 25 + eframe 0.34**。教育的価値と段階的実装の見通しを優先。MSRV 1.76 で足りる。

最終バージョン pin (Cargo.toml で `=` 固定、実装着手時の最新パッチを使用):

| crate | バージョン | 用途 |
|---|---|---|
| `wgpu` | `=25.0.x` | GPU API (MSRV は 1.76 だが本プロジェクトは依存の都合で 1.87 採用) |
| `winit` | `=0.30.x` (eframe 0.34 要求) | ネイティブウィンドウ |
| `wasm-bindgen` | `=0.2.x` | WASM ⇄ JS 境界 |
| `web-sys` | `=0.3.x` | DOM / Canvas / WebGPU 型 |
| `console_error_panic_hook` | `=0.1.x` | ブラウザでのパニック表示 |
| `bytemuck` | `=1.x` | バッファシリアライズ |
| `glam` | `=0.29.x` | ベクトル/行列 |
| `egui` / `eframe` | `=0.34.x` (`features = ["wgpu"]`) | UI |
| `ndarray` | `=0.16.x` | CPU 参照実装の補助 |
| `rand` / `rand_chacha` | `=0.8.x` | 再現可能な乱数 (CPU 側) |
| `approx` | `=0.5.x` | テスト用浮動小数比較 |

**着手時の手順**: `cargo new` 後に `cargo add` で最新パッチを取得、Cargo.lock を `git add`。`cargo update --dry-run` 警告は安易に上げず、明確な必要があるときのみ更新。

**Rev.4 で削除**: `rustfft` は M1 では使わない (Q7 別案採用)。

### 1.3 WebGPU 対応ブラウザの最低要件 (2026-05 時点)

| ブラウザ | ステータス | 備考 |
|---|---|---|
| Chrome 113+ / Edge 113+ | ✅ Stable (デスクトップ・Android 12+) | 主ターゲット |
| Firefox 141+ (Windows) / 145+ (macOS Apple Silicon) / 147+ (NVIDIA Linux Wayland) | ✅ Stable | Linux 一般・Android は依然対応外 |
| Safari 26.0+ (macOS Tahoe 26 / iOS 26 / iPadOS 26 / visionOS 26) | ✅ Stable | 主ターゲット |

**Linux 系**は依然として穴がある。起動時に `navigator.gpu` の存在チェックと「お使いのブラウザでは WebGPU が無効です」エラーパネル表示を必ず実装 (§7)。

### 1.3.1 WebGPU 必要 device features / limits

`request_device` で以下を要求し、不足時はフォールバックエラーを表示:

| 種別 | 名前 | 要件 | 用途 |
|---|---|---|---|
| Feature | (なし — 標準機能のみ) | — | 移植性最大化 |
| Limit | `max_storage_buffer_binding_size` | ≥ 64 MB | P バッファ (512² の場合は 128 MB 必要、§3.3) |
| Limit | `max_compute_workgroup_size_x` | ≥ 256 | 8×8×4 workgroup でも余裕 |
| Limit | `max_compute_invocations_per_workgroup` | ≥ 256 | 同上 |
| Limit | `max_storage_textures_per_shader_stage` | ≥ 8 | 多パスでテクスチャ多用 |
| Limit | `max_bind_groups` | ≥ 4 | デフォルト値 |

**`shader-f16` は使わない** (Q8 確定済み)。f32 一本。

### 1.4 ビルドツール

**trunk** 採用。eframe テンプレート公式採用、`trunk serve` で開発、`trunk build --release` で本番ビルド。

### 1.5 UI フレームワーク

**eframe + egui** 採用。`features = ["wgpu"]` で wgpu バックエンド有効化。`egui-wgpu` 経由で自前の wgpu compute pipeline と同じ `wgpu::Device` を共有、`egui::Image::from_texture` で自前テクスチャを直接描画可能。

### 1.6 数値計算ライブラリ

- **CPU 参照実装** (`core/cpu_reference.rs`): `ndarray` のみ。**Rev.4 で `rustfft` 削除** (Q7 別案採用、CPU も直接畳み込み)
- **GPU 側**: `glam` のみ

### 1.7 FFT の扱い — Rev.4 別案採用

JAX 公式実装は全畳み込みを FFT で行うが、本実装では **教育的価値・段階的実装・依存削減**を優先:

- **M1 (CPU 参照)**: **直接畳み込み**。`ndarray` で素朴な O(W·H·K²·R²) 実装。グリッド 16〜64 までの正確性検証専用なので速度問題なし
- **M2 (GPU 直接畳み込み)**: 同じく直接畳み込み (effective_radius で枝刈り)。128² までは 30 FPS 射程
- **テスト判定**: CPU 参照 vs GPU の出力一致テストは **相対誤差 < 1e-3** (両者ともに直接畳み込みなので、差はほぼ GPU 浮動小数の演算順序由来)
- **M6 optional**: torus 境界限定の GPU FFT 化を **stretch goal** とする
  - Stockham カーネル × 2 軸、複素 f32
  - 完了条件: 256² で 60 FPS 達成 (`paper_strict=OFF` モード)
  - **JAX とのビット精度近似テスト** (相対誤差 < 1e-4) を回帰 fixture として追加

カーネル `K_i` 自体は CPU で precompute して GPU 転送 (両モード共通)。JAX 式 `sigmoid(-(D-1)·10)` マスクで実効半径を持つので、直接畳み込みは `effective_radius = ⌈(R+15)·r_i⌉` で枝刈り可能 (§4.1.2)。

### 1.8 Cargo.lock 運用方針

- **Cargo.lock は必ずコミット**。`.gitignore` から除外されていることを確認
- `cargo update` は週次の手動操作とする(自動化しない)
- transitive 依存が MSRV 引き上げを要求した場合、設計書 §1.1 を更新する
  PR レビューを必須とする(勝手に上げない)
- 同種の問題は 2026 年中に再発が予想される (edition 2024 移行期のため)

---

## 2. アーキテクチャ図

```
flow-lenia/
├── crates/
│   ├── flow-lenia-core/      # プラットフォーム非依存
│   │   ├── src/
│   │   │   ├── params.rs           # KernelParams, FlowParams, サンプリング
│   │   │   ├── kernel.rs           # K_i を CPU で precompute (JAX 式)
│   │   │   ├── state.rs            # A, U, F, P の論理レイアウト定義
│   │   │   ├── cpu_reference.rs    # 参照実装 (ndarray, 直接畳み込み)
│   │   │   ├── shaders/            # WGSL ソース (include_str! で埋め込み)
│   │   │   │   ├── types.wgsl              # 共通 struct / bind group 宣言
│   │   │   │   ├── convolve.wgsl           # Pass 1: K_i ∗ A (生)
│   │   │   │   ├── affinity_growth.wgsl    # Pass 1b: G_i + P_i + c1_i 集約
│   │   │   │   ├── gradient.wgsl           # Pass 2: ∇U, ∇A_Σ (Sobel)
│   │   │   │   ├── flow.wgsl               # Pass 3: F = (1-α)∇U - α∇A_Σ
│   │   │   │   ├── reintegrate.wgsl        # Pass 4: A 更新 (Eq. 6)
│   │   │   │   ├── params_update.wgsl      # Pass 5: P 更新 (Eq. 8)
│   │   │   │   ├── mutation.wgsl           # M5: 変異ビーム
│   │   │   │   └── visualize.wgsl          # R: 状態 → 画面
│   │   │   └── lib.rs
│   │   └── Cargo.toml
│   ├── flow-lenia-gpu/       # wgpu セットアップ・パイプライン・ディスパッチ
│   │   └── src/...
│   ├── flow-lenia-ui/        # egui スライダ・コントロール・統計
│   │   └── src/lib.rs
│   └── flow-lenia-app/       # eframe アプリケーション (bin)
│       ├── src/...
│       ├── index.html
│       └── Trunk.toml
├── tests/                    # ワークスペース横断
│   ├── mass_conservation.rs
│   ├── reference_vs_gpu.rs
│   ├── sobel_sign.rs                       # Rev.4 追加 (M1 完了条件)
│   └── regression_fixtures/
├── papers/                   # 既存
├── references/               # 既存 (FlowLenia-jax/, JAX_NOTES.md)
├── DESIGN.md                 # 本書
├── Cargo.toml                # workspace
└── rust-toolchain.toml
```

依存方向: `app → ui → gpu → core` (app は core にも直接アクセス)。`core` は wgpu に依存せず、WGSL を `&'static str` として保持。

実行ループ (eframe::App::update): UI 更新 → params 反映 → running 時は steps_per_frame 回 `dispatch_step` → optional readback (質量検証 ON 時) → visualize テクスチャを egui Image として描画 → 統計表示。

---

## 3. データレイアウト設計

### 3.1 GPU バッファ vs テクスチャ

| 物理量 | 種別 | 形式 | 理由 |
|---|---|---|---|
| `A^t` 活性度 | Storage Texture 2D Array | `R32Float` | チャンネル単位の array layer、可視化パスでもサンプル可、質量検証のため f32 |
| `U^t` アフィニティ | Storage Texture 2D Array | `R32Float` | 中間値、Pass 1b → Pass 2 |
| `F^t` フロー | Storage Texture 2D Array | `Rg32Float` | 2D ベクトル、C 個 |
| `∇U`, `∇A_Σ` | Storage Texture | `Rg32Float` | 中間 |
| `P^t` パラメータマップ | Storage Buffer `array<f32>` | size: `W·H·|K|` | バインディング数を抑える、kernel index 軸が素直 |
| `K_i` カーネル | Storage Buffer | (kernel side² × |K|) | CPU で 1 回計算して固定 |
| `kernel_meta` (R, r_i, c0_i, c1_i, eff_radius_i) | Uniform Buffer | 16-byte align | |
| 成長関数 μ, σ | Uniform Buffer `array<vec2<f32>>` | |K| 個 | 軽量 |
| `globals` (dt, s, n, β_A, W, H, C, |K|, dd, mode flags) | Uniform Buffer | 64 byte 程度 | UI スライダで毎フレーム書き換え可 |

### 3.2 Pass 1 中間バッファ `pre_G`

| 物理量 | 種別 | 形式 | サイズ | 説明 |
|---|---|---|---|---|
| `pre_G[x, i]` | Storage Buffer `array<f32>` | `W·H·|K|` × 4 byte | 256² × 45 = 11.8 MB | Pass 1 で `K_i ∗ A_{c0_i}` の per-i 結果。Pass 1b で消費 |

これは G_i や P_i を一切適用していない**生の畳み込み結果**。G_i が非線形なので、後段で per-i 適用が必要。

### 3.3 メモリ予算分析

| グリッド | A (ピンポン) | P (ピンポン) | pre_G | F (単一、§3.4) | 合計 | 備考 |
|---|---|---|---|---|---|---|
| 64² × C=3 | 0.1 MB | 0.7 MB | 0.7 MB | 0.1 MB | ~ 2 MB | 問題なし |
| 128² × C=3 | 0.4 MB | 2.9 MB | 2.9 MB | 0.4 MB | ~ 10 MB | 問題なし |
| 256² × C=3 | 1.6 MB | 11.8 MB | 11.8 MB | 1.6 MB | ~ 40 MB | 問題なし |
| 512² × C=3 / |K|=45 | 6.3 MB | **94.4 MB** | 47.2 MB | 6.3 MB | ~ 200 MB | ⚠️ adapter 制限注意 |

**対策**: 起動時の adapter limits 取得結果から、UI のグリッドサイズ選択肢を動的に絞る。`max_storage_buffer_binding_size ≥ 128 MB` のとき以外は 512² を無効化。

### 3.4 バインディング表 (Rev.4 補強: F 共有戦略を明示)

#### F (フロー) の共有戦略

**`F` は同 step 内で複数パスから読まれる中間値である**:
- Pass 3 (flow) で書き込み (write-only)
- Pass 4 (reintegrate) で読み取り (read-only)
- Pass 5 (params_update) で読み取り (read-only)、`x'' = x' + dt·F[x']` の計算に必要

**ピンポンは不要**。Pass 4 と Pass 5 はいずれも `F` を入力としてのみ使い、`F` を書き換えない。次 step の Pass 3 が再度上書きするまで `F` は不変。したがって **`F` は単一バッファで運用**する。

**実装上の保証**:
- Pass 3 → Pass 4 → Pass 5 は同じ `CommandEncoder` に順序付きで `ComputePass` として記録され、`begin_compute_pass` の境界でメモリバリアが入る
- これにより Pass 3 の write が Pass 4 / Pass 5 の read に観測可能 (wgpu の保証)
- Pass 4 と Pass 5 の同時実行(`Pass 5 が Pass 4 の `A_out` を読む等)は**避ける**。両者は順序付き別 `ComputePass` として記録

#### バインディング表 (パス名統一)

| パス | group | binding | リソース | アクセス |
|---|---|---|---|---|
| **convolve** | 0 | 0 | `A_in` (R32Float texture array) | read |
| convolve | 0 | 1 | `pre_G` storage buffer (W·H·|K|) | write |
| convolve | 0 | 2 | `kernels` storage buffer | read |
| convolve | 0 | 3 | `kernel_meta` uniform | read |
| convolve | 0 | 4 | `globals` uniform | read |
| **affinity_growth** | 1 | 0 | `pre_G` storage buffer | read |
| affinity_growth | 1 | 1 | `growth_params` (μ_i, σ_i) uniform | read |
| affinity_growth | 1 | 2 | `P_in` storage buffer (W·H·|K|) | read |
| affinity_growth | 1 | 3 | `kernel_meta` uniform | read |
| affinity_growth | 1 | 4 | `U_out` (R32Float texture array) | write |
| affinity_growth | 1 | 5 | `globals` uniform | read |
| **gradient** | 2 | 0 | `U_in` | read |
| gradient | 2 | 1 | `A_in` | read |
| gradient | 2 | 2 | `gradU_out` (Rg32Float array) | write |
| gradient | 2 | 3 | `gradAsum_out` (Rg32Float) | write |
| gradient | 2 | 4 | `globals` uniform | read |
| **flow** | 3 | 0 | `gradU` | read |
| flow | 3 | 1 | `gradAsum` | read |
| flow | 3 | 2 | `A_in` (for α 計算: A_Σ or A_c) | read |
| flow | 3 | 3 | **`F` (Rg32Float array、単一バッファ)** | **write** |
| flow | 3 | 4 | `globals` uniform | read |
| **reintegrate** | 4 | 0 | `A_in` | read |
| reintegrate | 4 | 1 | **`F` (上記と同じ単一バッファ)** | **read** |
| reintegrate | 4 | 2 | `A_out` | write |
| reintegrate | 4 | 3 | `globals` uniform | read |
| **params_update** | 5 | 0 | `A_in` | read |
| params_update | 5 | 1 | **`F` (上記と同じ単一バッファ)** | **read** |
| params_update | 5 | 2 | `P_in` storage buffer | read |
| params_update | 5 | 3 | `P_out` storage buffer | write |
| params_update | 5 | 4 | `rng_state` storage buffer (per-cell u32) | rw |
| params_update | 5 | 5 | `globals` uniform | read |
| **mutation** (M5) | 6 | 0 | `P_inout` storage buffer | rw |
| mutation | 6 | 1 | `mutation_beam_params` uniform | read |
| **visualize** | 7 | 0 | `A_in` | read |
| visualize | 7 | 1 | `P_in` (オプション色付け) | read |
| visualize | 7 | 2 | `screen_tex` (Rgba8Unorm) | write |
| visualize | 7 | 3 | `globals` uniform | read |

`shaders/types.wgsl` を prelude として全パスで `include`。

### 3.5 ダブルバッファ対象まとめ

- **ダブルバッファ**: `A` (Eq. 6 で読み書き別)、`P` (Eq. 8 で読み書き別)
- **単一バッファ**: `U`, `F`, `∇U`, `∇A_Σ`, `pre_G` (いずれも同 step 内で生成・消費)

ピンポンは `step_parity: bool` を CPU 側で持ち、`dispatch_step` 内で `bind_group_even`, `bind_group_odd` を切り替える。

---

## 4. シェーダパイプライン設計

### 4.1 1 step あたりのパス構成

| # | 名前 | 入力 | 出力 | 対応式 | dispatch サイズ |
|---|---|---|---|---|---|
| 1 | **convolve** | `A_in`, `K`, `kernel_meta` | `pre_G[x, i] = (K_i ∗ A_{c_i^0})(x)` | Eq. 7 / Eq. 3 畳み込み部のみ | (W/8, H/8, ⌈|K|/4⌉) |
| 1b | **affinity_growth** | `pre_G`, `growth_params`, `P_in`, `kernel_meta` | `U_j(x) = Σ_i P_i(x) · G_i(pre_G[x, i]) · [c_i^1 = j]` | Eq. 7 全体 | (W/8, H/8, C) |
| 2 | **gradient** | `U`, `A_in` | `∇U` (C 個), `∇A_Σ` (1 個) | Eq. 5 Sobel 部 | (W/8, H/8, C+1) |
| 3 | **flow** | `∇U`, `∇A_Σ`, `A_in` | `F` (単一バッファ) | Eq. 5 全体 (α 適用込み、§4.1.5) | (W/8, H/8, C) |
| 4 | **reintegrate** | `A_in`, `F` | `A_out` | Eq. 6 (受信側、11×11 近傍) | (W/8, H/8) |
| 5 | **params_update** | `A_in`, `F`, `P_in`, `rng_state` | `P_out` | Eq. 8 (§4.1.6) | (W/8, H/8) |
| 5m | **mutation** (M5+) | `P_inout` | `P_inout` | §4.1.7 | (patch, patch) per beam |
| swap | — | — | — | ダブルバッファのインデックス反転 | CPU |
| R | **visualize** | `A_in` (next step) | `screen_tex` | — | (W/8, H/8) |

#### 4.1.1 Pass 1 の出力は「生の畳み込み結果」

Eq. 7 では `U_j(x) = Σ_i P_i(x) · G_i((K_i ∗ A)(x)) · [c_i^1 = j]`。`G_i` は非線形なので、i について先に和を取ってから G を適用するのは別関数。Pass 1 は **G も P も乗じず**、`pre_G[x, i] = (K_i ∗ A_{c_i^0})(x)` だけを書く。Pass 1b で G_i と P_i を適用し、`c_i^1` で集約。これにより Eq. 7 と数学的に等価。

#### 4.1.2 カーネルごとに有効半径 `(R+15)·r_i` を使う

JAX 式に従い、effective_radius = `⌈(R+15)·r_i⌉`:

```wgsl
let er = kernel_meta[i].effective_radius;  // 例: r_i=0.4, R=25 → ⌈16⌉=16
for (var dy: i32 = -er; dy <= er; dy = dy + 1) {
    for (var dx: i32 = -er; dx <= er; dx = dx + 1) {
        // sigmoid(-(D-1)·10) マスクを適用 (D = sqrt(dx²+dy²) / ((R+15)·r_i))
        // 詳細は §4.6 のテンプレ参照
        ...
    }
}
```

#### 4.1.3 Reintegration: 受信側、11×11 近傍 (Q1A 確定)

dy, dx ∈ {-5, ..., 5} の二重ループ (121 セル / 受信側)。送信側ループは race condition を起こすため不採用。JAX `reintegration_tracking.py:39` の `dd=5` で確認済み。

#### 4.1.4 Eq. 8 (params_update) は総質量 A_Σ ベース

論文 Eq. 8 の softmax 分布は per-channel ではなく **総質量 A_Σ**:

```wgsl
// For target cell x, find x' in 11x11 neighborhood
var weights: array<f32, 121>;
for (var idx = 0u; idx < 121u; idx = idx + 1u) {
    let xprime = neighbor_of(x, idx);
    let mass = A_Σ_at(xprime);          // 全 C 和
    let I = overlap_area(xprime + dt * F[xprime], s, x);
    weights[idx] = mass * I;
}

// paper_strict mode: softmax(weights) → categorical sample
//   logit = weights, probability = exp(weights) / Σ exp(weights)
// jax_compat mode (default):
//   probability = weights / Σ weights  (softmax を省略、論文と異なる)
//   詳細は §4.6 と JAX_NOTES.md §11 NOTE-11-B
```

#### 4.1.5 α (拡散項重み) の式 — Rev.4 デフォルト反転

**デフォルト (`paper_strict=OFF`、JAX 互換)**:
```
α(x, c) = clip((A_c(x) / β_A)^2, 0, 1)        # per-channel、n=2 固定
```
- `β_A = 2.0` (UI 可変)
- per-channel: α は channel ごとに別値、`F_c` 計算時に対応する `α_c` を使う

**`paper_strict=ON` (論文 Eq. 5)**:
```
α(x) = clip((A_Σ(x) / β_A)^n, 0, 1)            # 全 C 共通、n 可変
A_Σ(x) = Σ_c A_c(x)
```
- `β_A` と `n` 両方 UI 可変

両モード共通で `β_A`, `n` は UI スライダ可変 (Rev.4 ユーザー指定)。`n` は `paper_strict=ON` 時のみ意味を持つ (OFF 時は 2 固定だが、UI 値は保持しておく)。

#### 4.1.6 Eq. 8 (params_update) の softmax 切替 — Rev.4 明記

**デフォルト (`paper_strict=OFF`、JAX 互換)**:
```
probability(x' → x) = (A_Σ(x') · I(x', x)) / Σ_{x''} (A_Σ(x'') · I(x'', x))
```
- 論文 Eq. 8 の `exp(...)` を **省略**した形 (JAX 実装と一致)
- 参照: `JAX_NOTES.md` §11 NOTE-11-B、`reintegration_tracking.py:127` の `jnp.log(nA.sum(...))` を categorical の logit に渡す動作と等価

**`paper_strict=ON` (論文 Eq. 8)**:
```
probability(x' → x) = exp(A_Σ(x') · I(x', x)) / Σ_{x''} exp(A_Σ(x'') · I(x'', x))
```
- 論文式に厳密準拠

**両モード共通**: per-cell PRNG state (xoshiro128++、`u32` 4 個 × W·H) を保持。**JAX 実装の `jr.PRNGKey(42)` 固定シードバグは採用しない** (`JAX_NOTES.md` §11 NOTE-11-A、`reintegration_tracking.py:127`)。

#### 4.1.7 Mutation (M5+)

論文 §4.3.1 / JAX `flowlenia_params.py:153-162`:
- patch サイズ `sz`: UI で {10, 20, 30} 切替、デフォルト 20 (JAX) / 10 (論文)
- 確率 `p_mut` / step: 0..=1、デフォルト 0.01
- 適用: ランダム位置に `sz × sz` パッチ、各カーネルインデックスごとに 1 個の `N(0, 1)` サンプルを broadcast (patch 全体で同じ値を加算)

### 4.2 各パスの WGSL コメントテンプレート (共通)

各 WGSL ファイル冒頭:

```wgsl
// Flow-Lenia <pass_name> shader
// Implements: <equation reference>
// Reference: Plantec et al. 2025 (arXiv:2506.08569v1)
// References (cross):
//   - DESIGN.md §<section>
//   - references/JAX_NOTES.md §<section> (commit dce428c)
//
// Workgroup: (8, 8, 1)  (or as specified)
// Dispatch: see DESIGN.md §4.1
// Bindings: see DESIGN.md §3.4
```

### 4.3 リインテグレーション・トラッキング — 重なり面積の具体実装

JAX `reintegration_tracking.py:46-60` を WGSL 化:

```text
# Pre-step: フロー振幅を物理上限 ma = dd - sigma で clip (JAX:62)
ma = dd - sigma
F_clipped = clip(F, -ma, ma)

# For each target cell x:
sum = 0
for dx in -dd..=dd:
    for dy in -dd..=dd:
        xp = (x.x + dx) mod W   # トロイダル時 (border=torus)
        yp = (x.y + dy) mod H
        m = (xp + 0.5, yp + 0.5) + dt * F_clipped[xp, yp]

        # トロイダル時は dpmu を 9 通り (±W, ±H) から最小値
        dpmu.x = min(|d|, W - |d|)
        dpmu.y = min(|d|, H - |d|)

        # JAX area 計算 (utils.py:57-58)
        # sz = 0.5 - dpmu + sigma (各成分)
        # 上限 clip min(1, 2σ) は σ > 0.5 (高温) で必須
        sz_x = clamp(0.5 - dpmu.x + sigma, 0, min(1, 2*sigma))
        sz_y = clamp(0.5 - dpmu.y + sigma, 0, min(1, 2*sigma))
        I = (sz_x * sz_y) / (4 * sigma * sigma)

        sum += A[xp, yp] * I

A_new[x] = sum
```

**`border=wall` 時**: dpmu は単純な `|x - m|`、追加で `mu = clip(mu, σ, W-σ)` で境界 clip (JAX:64-65)。

### 4.4 Sobel フィルタ — Rev.4 検証テスト追加

JAX 実装 (`utils.py:16-37`) はカーネルを**正規化なし**で `jsp.signal.convolve2d` (mode='same') に渡す:

```python
kx = jnp.array([
    [1., 0., -1.],
    [2., 0., -2.],
    [1., 0., -1.]
])
ky = jnp.transpose(kx)
```

`convolve2d` は数学的畳み込み (カーネル 180° 回転後の相関)。**WGSL では相関で実装するため、カーネルの符号を反転**する:

```wgsl
// 相関ベース (符号反転後の実効 Sobel)
let sx: array<array<f32, 3>, 3> = array(
    array(-1.0,  0.0,  1.0),
    array(-2.0,  0.0,  2.0),
    array(-1.0,  0.0,  1.0),
);
// sy = sx の転置
```

境界処理: `border=torus` なら周期境界、`border=wall` ならゼロパディング (JAX 既定)。

#### Rev.4 新規: `test_sobel_sign` ユニットテスト (M1 完了条件)

**目的**: WGSL と CPU 参照実装の Sobel 符号が JAX と一致することを保証。
**配置**: `tests/sobel_sign.rs`
**検証手順**:
1. `A(x, y) = x as f32` で初期化された 16×16 グリッドを作成 (左から右へ +1 ずつ増加する勾配場)
2. CPU 参照 Sobel と GPU Sobel の `∂A/∂x` を計算
3. **すべてのセル**で `∂A/∂x ≈ +1.0` (`/ 0.125` のような正規化係数を掛けない生 Sobel では値は **+4.0** = `(1·1 + 2·1 + 1·1) = 4` を期待)
4. `∂A/∂y` がほぼ 0 (浮動小数誤差内) であること
5. 境界セルは torus 補正後の値を期待値とする
6. `A(x, y) = y as f32` でも同様に `∂A/∂y ≈ +4.0`、`∂A/∂x ≈ 0`

これにより、Sobel カーネル符号の方向 (左右どちらが +)、転置軸 (x と y の取り違え)、正規化の有無 を全て検出できる。

### 4.5 数値演算順序の固定

- Eq. 7 (affinity_growth): **i (カーネルインデックス) 昇順**
- Eq. 6 (reintegrate): **dy 外、dx 内、`-5..=5`** (JAX `range(-dd, dd+1)` と一致)
- Sobel (gradient): 固定 3×3 なので順序は自然に決まる

### 4.6 カーネル・softmax の WGSL コメントテンプレ — Rev.4 拡充

#### convolve.wgsl 冒頭 (カーネル定義の 3 つの差異を明示)

```wgsl
// Flow-Lenia convolve shader — Pass 1
// Implements: K_i ∗ A_{c0_i}, the raw convolution part of Eq. 7 (= Eq. 3 numerator).
//
// =================================================================
// ⚠️ KERNEL DEFINITION: paper Eq. 1 vs JAX implementation differ
// =================================================================
//
// Paper Eq. 1 (Plantec et al. 2025, arXiv:2506.08569v1):
//   K_i(x) = Σ_j b_{i,j} · exp( -((r/(r_i·R) - a_{i,j})^2) / (2 w_{i,j}^2) )
//   where r is the cell-center distance, r_i·R is the kernel-i effective radius.
//
// JAX implementation (utils.py:41-59, ker_f at utils.py:9):
//   D = r / ((R + 15) · r_i)             # scaling differs from paper
//   K_i(x) = sigmoid(-(D - 1) · 10) · Σ_j b_{i,j} · exp( -(D - a_{i,j})^2 / w_{i,j} )
//   ↑ extra mask                ↑ different denominator: paper has 2·w^2, JAX has w
//
// Three differences and their intent:
//   (1) R → R+15 in the scaling denominator. Likely an empirical correction
//       to avoid pixel-grid artifacts at small R (R ∈ [2, 25] in Table 1).
//       References/JAX_NOTES.md §7 "推測 1: 最小 R=2 のとき..."
//   (2) Gaussian denominator: paper uses 2·w^2, JAX uses w. The numerical
//       parameter ranges (w ∈ [0.01, 0.5]) happen to overlap, so JAX's "w"
//       can be read as paper's "2·w^2" without changing the parameter range.
//   (3) sigmoid(-(D-1)·10) mask: forces K_i to vanish smoothly for D > 1,
//       i.e., outside the effective radius. Paper does not mention this;
//       it is a numerical regularizer.
//
// Decision: this implementation follows the JAX form (1)–(3) so that JAX
// parameter sets reproduce the same creatures. The mapping to paper Eq. 1
// is documented above for educational purposes.
//
// Workgroup: (8, 8, 4)  — 4 kernels per workgroup along z
// Dispatch: see DESIGN.md §4.1
// Bindings: see DESIGN.md §3.4
```

#### params_update.wgsl 冒頭 (Eq. 8 切替 + JAX バグ明記)

```wgsl
// Flow-Lenia params_update shader — Pass 5
// Implements: Eq. 8 of Plantec et al. 2025 (parameter mixing rule).
//
// =================================================================
// ⚠️ Eq. 8 SOFTMAX: paper vs JAX implementation differ
// =================================================================
//
// Paper Eq. 8:
//   P[P^{t+dt}(x) = P^t(x')] = exp(A_Σ(x') · I(x', x))
//                             / Σ_{x''} exp(A_Σ(x'') · I(x'', x))
//
// JAX implementation (reintegration_tracking.py:125-133):
//   probability = (A_Σ · I) / Σ (A_Σ · I)        # the exp(...) is dropped
//   (specifically: categorical(logits = log(A_Σ · I)) is equivalent to
//    sampling proportional to A_Σ · I; the softmax exp cancels with the log)
//   See references/JAX_NOTES.md §11 NOTE-11-B for the full annotation.
//
// Behavior selection:
//   - paper_strict = OFF (default, JAX compat): use the JAX form (no exp).
//   - paper_strict = ON: use the paper Eq. 8 form (with exp).
//
// =================================================================
// 🐛 JAX BUG: fixed PRNG seed (NOT replicated)
// =================================================================
//
// The JAX implementation calls jax.random.PRNGKey(42) inside the per-step
// __call__ function (reintegration_tracking.py:127), meaning every step
// uses the SAME random seed. This is almost certainly an oversight.
// See references/JAX_NOTES.md §11 NOTE-11-A.
//
// This implementation uses per-cell xoshiro128++ state (u32×4 per cell,
// stored in rng_state storage buffer) advanced every step. This is the
// correct behavior for stochastic sampling and is independent of the
// paper_strict toggle above.
//
// Workgroup: (8, 8, 1)
// Dispatch: see DESIGN.md §4.1
// Bindings: see DESIGN.md §3.4
```

#### flow.wgsl 冒頭 (α 切替)

```wgsl
// Flow-Lenia flow shader — Pass 3
// Implements: F = (1-α)·∇U - α·∇A_Σ  (Eq. 5)
//
// =================================================================
// ⚠️ α (diffusion weight): paper vs JAX implementation differ
// =================================================================
//
// Paper Eq. 5:
//   α(x) = clip( (A_Σ(x) / β_A)^n , 0, 1 )       # whole-channel scalar
//   F_c  = (1 - α) · ∇U_c - α · ∇A_Σ
//
// JAX implementation (flowlenia.py:98 / flowlenia_params.py:101):
//   α(x, c) = clip( (A_c(x) / β_A)^2 , 0, 1 )    # per-channel, n=2 fixed
//   β_A = C (flowlenia.py) or 2.0 (flowlenia_params.py)
//
// Behavior selection:
//   - paper_strict = OFF (default, JAX compat): per-channel α with n=2.
//     β_A defaults to 2.0 but is UI-tunable.
//   - paper_strict = ON: whole-channel α(x) using A_Σ. Both β_A and n
//     are UI-tunable.
//
// Note: even in paper_strict mode, n is UI-tunable (paper does not fix it).
// In JAX mode, the n slider is grayed out (effectively 2).
//
// See DESIGN.md §4.1.5 and references/JAX_NOTES.md §2.
```

---

## 5. 数値的正しさの検証戦略

### 5.1 ユニットテスト (`crates/flow-lenia-core/tests/`)

- `test_kernel_normalization`: Σ K_i ≈ 1 (Eq. 1 / JAX `nK = K / sum(K)`)
- `test_growth_range`: G_i ∈ [-1, 1] for x ∈ [0, 1] (Eq. 2)
- `test_overlap_area`: §4.3 の重なり面積の 8 ケース (完全重なり / 無 / 部分 / σ>0.5 高温 / 境界)
- `test_softmax_sampling`: Eq. 8 が A_Σ を使うこと (paper_strict / jax_compat 両モード)
- `test_per_kernel_radius`: 異なる r_i で正しい effective_radius が使われること
- `test_mutation_beam`: sz×sz 領域に N(0,1) ノイズ、領域外は無変化
- **`test_sobel_sign` (Rev.4 追加、M1 完了条件)**: §4.4 詳細手順、∂A/∂x = +4 on ramp A(x,y)=x、∂A/∂y on ramp A(x,y)=y

### 5.2 参照実装 vs GPU 一致テスト (`tests/reference_vs_gpu.rs`)

- 16×16 グリッド、C=1、|K|=3 で固定シード
- 50 ステップ走らせ CPU 参照と GPU 出力比較
- 許容誤差: `|a_cpu - a_gpu| / (|a_cpu| + 1e-6) < 1e-3` (Rev.4: 直接畳み込み同士のため 1e-3 で十分)

### 5.3 質量保存プロパティテスト (`tests/mass_conservation.rs`)

- ランダム初期状態 (32×32 / 64×64 / 128×128 × seeds × C ∈ {1,2,3}) で 1000 step
- **チャンネルごとの**総質量 `Σ_x A_i(x)` の初期値からの相対誤差 < 1e-3
- 4 モード全部 (`paper_strict × jax_compat × border × embedding`) でテスト

### 5.4 回帰テスト

論文 Figure 4 のパターンに近づくシード/パラメータを `tests/regression_fixtures/` に格納。M2 完了時に初期 fixture を生成して固定。M6 で FFT 化したら JAX とのビット精度近似 fixture を追加 (optional)。

### 5.5 ベンチマーク

`criterion` で CPU 参照の 1 step 時間を測る。WASM 側は手動 `performance.now()` で FPS を画面表示。

---

## 6. パラメータの初期化

JAX 実装 `flowlenia.py:55-64` の範囲に合わせる:

```text
settings = { num_kernels, num_channels, grid_size }
output: KernelParams = {
  R:       uniform ∈ [2.0, 25.0]
  for each kernel i:
    c0_i:  uniform from 0..C
    c1_i:  uniform from 0..C
    r_i:   uniform ∈ [0.20, 1.00]
    a_i:   [uniform ∈ [0.00, 1.00]; 3]
    b_i:   [uniform ∈ [0.001, 1.00]; 3]    # 論文 [0, 1]、JAX が 0 を除外
    w_i:   [uniform ∈ [0.010, 0.50]; 3]
    h_i:   uniform ∈ [0.010, 1.00]         # 論文 [0, 1]、JAX が 0 を除外
    μ_i:   uniform ∈ [0.05, 0.50]
    σ_i:   uniform ∈ [0.001, 0.18]         # 論文 [0.001, 0.2] よりわずかに狭い
  s:           0.65    (UI で [0.1, 2.0])
  n:           2.0     (UI で [0.5, 8.0]、paper_strict=OFF 時は 2 固定)
  dt:          0.2     (UI で [0.01, 0.5])
  dd:          5       (UI で 3..=7)
  β_A:         2.0     (UI で [0.5, 20.0])  # JAX 互換のデフォルト
  border:      "torus" (UI で {torus, wall})
  mix_rule:    "stoch" (UI で {stoch, det})
  paper_strict: false  (UI チェックボックス)  # Rev.4: デフォルトは JAX 互換
  embedding:   false   (UI チェックボックス)  # M5 で ON
}
```

シード再現性: `rand_chacha::ChaCha8Rng::seed_from_u64(seed)`。UI に `seed: u64` 入力、空欄は `getrandom` 自動生成 + 表示。

---

## 7. UI 仕様

`ControlsPanel` 左サイドバー:

| グループ | コントロール | デフォルト | 範囲 | 備考 |
|---|---|---|---|---|
| Grid | `grid_size: enum` | 128 | {64, 128, 256, 512} | 512 は adapter limits 次第で無効化 |
| Grid | `channels: u32` | 3 | 1..=3 | UI 制約 (内部は任意 C) |
| Grid | `num_kernels: enum` | 10 | {5, 10, 20, 45} | |
| Sim | ▶ / ⏸ / ⏭ / ⟲ / 🎲 | — | — | 再生・1step・Reset・Re-randomize |
| Sim | `seed: u64` 入力 + Copy | random | — | |
| Sim | `steps_per_frame: u32` | 1 | 1..=32 | 速度 |
| Phys | `dt` | 0.2 | [0.01, 0.5] | |
| Phys | `s` (温度) | 0.65 | [0.1, 2.0] | |
| Phys | `n` | 2.0 | [0.5, 8.0] | `paper_strict=OFF` 時はグレーアウト (2 固定) |
| Phys | `β_A` | 2.0 | [0.5, 20.0] | **両モードで可変** (Rev.4 ユーザー指定) |
| Phys | `dd` | 5 | 3..=7 | リインテグレーション近傍 |
| Mode | `border`: {torus, wall} | torus | — | Sobel も追従 |
| Mode | **`paper_strict`** | **OFF** | bool | **Rev.4: ON で論文 Eq. 5/8 厳密 (デフォルト JAX 互換)** |
| Mode | `embedding` (Eq. 7) | OFF | bool | M5 |
| Mode | `mix_rule`: {stoch, det} | stoch | — | `embedding=ON` 時のみ、Eq. 8 |
| M5 | `mutation_patch_size` | 20 | {10, 20, 30} | 論文準拠 10、JAX 20 |
| M5 | `mutation_rate p_mut` | 0.01 | [0.0, 1.0] | |
| Init | `init_patch_size` | 40 | [10, 80] | 初期パッチ辺長 |
| Init | **`init_patch_center_range`** | **画面中心 ± grid_size/4** | — | **Rev.4: 配置範囲制限** (壁境界での即時消失を避ける) |
| Init | `num_creatures` (M5+) | 1 | [1, 64] | multi-patch 配置 |
| Viz | C → RGB マッピング | C1→R, C2→G, C3→B | — | |
| Viz | gamma | 1.0 | [0.2, 3.0] | |
| Viz | パラメータ色 (P を hue 可視化) | OFF | bool | M5 |
| Stats | 総質量 (per channel) | — | — | |
| Stats | 初期からの相対誤差 (%) | — | — | |
| Stats | FPS / step ms | — | — | |

### Rev.4 追加: 初期パッチ配置範囲制限

`init_patch_center_range` は **画面中心 ± grid_size/4** の領域に制限する:
- 128² グリッドなら中心 64,64 を中心に ±32 (= 32〜96 の範囲)
- 理由: `border=wall` モードでパッチが境界に近い位置で生成されると、最初の数 step で質量が境界で clip され creature が即時に縮退する事故を避ける
- `border=torus` モードでも同様の制限を適用 (UX 一貫性のため)
- multi-patch 配置時 (M5) は各 patch 中心を独立にこの範囲からサンプル

起動時 WebGPU 非対応エラー: egui で全画面エラーパネル、対応ブラウザ一覧 ([caniuse.com/webgpu](https://caniuse.com/webgpu)) リンク。

---

## 8. 段階的実装計画

### M1: CPU 版 Flow-Lenia がネイティブで動く

**成果物**: `flow-lenia-core` の `cpu_reference.rs` で Eq. 1, 2, 3, 5, 6 を ndarray の**直接畳み込み**で素朴実装 (Rev.4: rustfft 不使用)

**完了条件**:
- `cargo test -p flow-lenia-core` グリーン
- 64×64 / C=1 / |K|=5 で 100 steps 走り、総質量誤差 < 1e-6
- 簡易 dump (ppm) で目視確認
- 全ユニットテストグリーン (§5.1):
  - `test_kernel_normalization`
  - `test_growth_range`
  - `test_overlap_area`
  - `test_softmax_sampling`
  - `test_per_kernel_radius`
  - `test_mutation_beam` (M5 で使うが M1 時点で実装)
  - **`test_sobel_sign` (Rev.4 追加)**

### M2: wgpu compute shader 版がネイティブで動く

**成果物**: `flow-lenia-gpu`、winit ネイティブウィンドウで描画

**完了条件**:
- 全 6 compute pass + visualize 動作
- `tests/reference_vs_gpu.rs` グリーン (16×16, 50 steps, 相対誤差 < 1e-3)
- `tests/mass_conservation.rs` グリーン (32×32, 500 steps, 相対誤差 < 1e-3)
- 4 モード組合せ (`paper_strict × border`) すべてでテスト通過
- 回帰 fixture を 1 つ確定 (CPU 参照を黄金として保存)

### M3: WASM ビルド + ブラウザ WebGPU 動作

**成果物**: `trunk build --release` で `dist/` に成果物

**完了条件**:
- Chrome 最新版で WebGPU 経由で動く
- M2 と同じシードで出力が一致 (visual diff + 質量チェック)
- WebGPU 非対応ブラウザでフォールバックエラー表示
- adapter limits 取得 + UI 動的制限 (512² 無効化等)

### M4: UI 統合・リアルタイム可視化

**成果物**: egui コントロール一式、状態テクスチャ表示、統計

**完了条件**:
- §7 の全コントロール動作 (`paper_strict`, `border`, `dd`, init patch range 含む)
- 128×128 / C=3 / |K|=45 で 30 FPS 以上 (実測ベンチ)
- 最低 1 つ「明らかに動く creature」が見える固定シードのデモを README に掲載

### M5: パラメータ埋め込み・マルチ種シミュレーション

**成果物**: Eq. 7, Eq. 8 (stochastic & deterministic) の有効化、multi-patch 初期配置、変異ビーム

**完了条件**:
- parameter embedding ON で `tests/mass_conservation.rs` グリーン
- UI から多体配置 (64 creatures × ランダム seeds) を生成可能
- 変異ビーム実装で `test_mutation_beam` グリーン (M1 から既存だが、ここで実機統合)
- 数千ステップ走らせて、視覚的に「種が交代する」様子が観察可能

### M6: 性能チューニング・ドキュメント整備

**成果物**:
- 256×256 / C=3 / |K|=45 で 60 FPS 達成 (Apple M1 想定)
- README、CHANGELOG、シェーダ内コメント整備
- `cargo doc` 公開可能

**完了条件**:
- ベンチ表 (grid_size × C × |K| × backend × paper_strict での FPS) を README に
- シェーダ内の論文式番号コメントが全パスに揃っている

#### M6 optional stretch (Rev.4 追加)

- **GPU FFT 化** (torus 境界限定): Stockham カーネル × 2 軸、複素 f32
  - 256² で 60 FPS を非 FFT で達成できない場合に実装
  - 完了条件: 256² / `paper_strict=OFF` で 60 FPS 達成
  - **JAX とのビット精度近似テスト**: 固定パラメータで JAX と本実装 (FFT モード) の `A^100` を比較、相対誤差 < 1e-4
- 散逸モデル / 食物モデル (論文 §4.3.2) は M6 stretch のままに維持

---

## 9. 守る原則の運用

| 原則 | 運用 |
|---|---|
| 論文との対応をコメントに残す | 全 WGSL ファイル冒頭 (§4.2) + Eq. を実装したループ直上 |
| 数式の差異 (paper vs JAX) を明示 | §4.6 のテンプレを convolve.wgsl / params_update.wgsl / flow.wgsl で使用 |
| JAX 公式実装との挙動一致 | デフォルト `paper_strict=OFF` で JAX 互換、M6 optional で FFT 化により JAX ビット精度近似 |
| WGSL は型と束縛に厳格 | `shaders/types.wgsl` を prelude として全パス共有 |
| wgpu バージョン互換性 | Cargo.toml で `wgpu = "=25.0.x"` 等 pin。CI で `cargo update --dry-run` 警告化 |
| ブラウザ WebGPU 互換 | §1.3 + §7 起動時チェック + README 対応表 |
| 巨大シェーダ非作成 | §2 の 9 ファイル分割を厳守 |
| メモリ予算チェック | 起動時に adapter limits 取得、UI の選択肢を絞る |

---

## 10. 確定事項一覧 (要承認事項はすべて Rev.4 ユーザー判断で確定)

| Q# | 内容 | Rev.4 確定値 |
|---|---|---|
| Q1 | ブラウザ優先度 | Chrome 最優先、Safari 次点、Firefox 145+ ベストエフォート |
| Q1A | Chebyshev 距離 | **11×11** (`-5..=5`)、`dd` UI 可変 (3..=7) |
| Q2 | ネイティブ基準 OS/GPU | Apple Silicon (Metal) |
| Q3 | α/β_A/n の式 | **デフォルト JAX 互換** (per-channel α, β_A=2.0, n=2 固定)、`paper_strict=ON` で論文 Eq. 5 (全 C 共通 α、A_Σ ベース、n 可変)。`β_A`, `n` は両モードで UI 可変 |
| Q3b | Sobel 正規化 | 正規化なし、`test_sobel_sign` で符号検証 |
| Q3c | カーネル定義 | JAX 式採用、convolve.wgsl 冒頭に 10 行以上のコメントブロック (§4.6) |
| Q3d | Eq. 8 softmax | デフォルト JAX 式 (`(A·I)/Σ(A·I)`)、`paper_strict=ON` で論文 (`exp(A·I)/Σexp`)。JAX `PRNGKey(42)` バグは採用せず、per-cell xoshiro128++ で修正、params_update.wgsl にバグ明記 (§4.6) |
| Q4 | 食物/散逸モデル | M5 まで vanilla のみ、dissipative/food は M6 stretch |
| Q5 | 進化的最適化 (ES) | スコープ外 |
| Q6 | 境界条件 | デフォルト `torus`、UI で `wall` 切替、Sobel も border 追従。**初期パッチは画面中心 ± grid_size/4 に配置制限** |
| Q7 | FFT | **M1/M2 は CPU/GPU とも直接畳み込み**、`rustfft` 削除。**M6 optional で torus 限定 GPU FFT 化**、JAX ビット精度近似テスト追加 |
| Q8 | 量子化/f16 | f32 のみ |
| Q9 | デプロイ | ローカル `trunk serve` のみ |

---

## 11. 承認後にやること

承認をいただいたら:

1. `Cargo.toml` workspace と 4 crate の雛形作成
2. `rust-toolchain.toml` 配置 (Rust 1.76)
3. `flow-lenia-core` の Eq. 1/2 + パラメータサンプリングを TDD で
4. `flow-lenia-core::cpu_reference` で Eq. 3 → Eq. 5 → Eq. 6 を順に (直接畳み込み)
5. **`test_sobel_sign` を含む全ユニットテスト**を M1 完了条件として実装
6. M1 完了テスト → M2 着手 (wgpu pipeline、最小ループ)
7. 以降は §8 のマイルストーンに従う

**Rev.4 のレビューをお願いします。** 「Rev.4 OK」のご返答をいただいた時点で初めて M1 実装に着手します。それまでコードを一行も書きません。

---

## 付録 A: Rev.3 → Rev.4 の変更要約 (差分)

ユーザー指示 7 件をすべて反映:

1. **Q3 デフォルト反転** → §4.1.5 / §6 / §7 / §10
   - デフォルト `paper_strict=OFF` (JAX 互換)、ON で論文 Eq. 5
   - `β_A` と `n` を両モードで UI 可変 (ただし `n` は OFF 時グレーアウト = 2 固定)
2. **Q3c コメント拡充** → §4.6 convolve.wgsl テンプレ (約 25 行)
3. **Q3d バグ明記** → §4.6 params_update.wgsl テンプレに JAX `PRNGKey(42)` バグ言及、`JAX_NOTES.md` §11 NOTE-11-A/B 引用
4. **Q6 初期パッチ補強** → §7 Init グループに `init_patch_center_range = 画面中心 ± grid_size/4`
5. **Q7 別案採用** → §1.6 (`rustfft` 削除)、§1.7 (M1 直接畳み込み、M6 optional FFT)、§5.2 (許容誤差 1e-3 維持)、§8 M6 optional に JAX ビット精度近似テスト
6. **§3.4 F 共有戦略** → §3.4 冒頭に `F` 単一バッファ運用の根拠 (wgpu の compute pass バリア保証)
7. **§4.4 Sobel 符号検証** → §4.4 / §5.1 / §8 M1 完了条件に `test_sobel_sign` 追加 (検証手順込み)

加えて、`references/JAX_NOTES.md` §11 を Q3d WGSL コメントから引用しやすいよう NOTE-11-A / NOTE-11-B アンカー付きに更新済み。
