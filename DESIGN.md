# Flow-Lenia WebGPU Visualizer — 設計書 (Rev. 4.10)

本書は Rust + WebAssembly + WebGPU で **Flow-Lenia (Plantec et al., 2025, Artificial Life journal, arXiv:2506.08569v1)** を厳密に再現し、ブラウザ上でリアルタイム可視化する実装の設計書である。

**実装の正典**は `papers/2506.08569v1.pdf` (2025年版) であり、Equation 番号は同論文を指す。副参照として `papers/2212.07906v2.pdf` (2023年版)、Moroz, 2020 "Reintegration tracking"、**公式 JAX 実装** `references/FlowLenia-jax/` (commit `dce428c`, 2024-02-08) を用いる。JAX 実装の精読結果は `references/JAX_NOTES.md` を参照。

## Rev. 履歴

### Rev.4.10 (2026-05-31、M6.C-3 正式締め + web 動作検証 + Chrome 5 罠記録)

- **M6.C-3-8 follow-up 8 commits で web 動作確認 完了**:
  - Apple Metal native で通る code が Chrome Tint/Dawn 厳密 validator で
    5 件の壁、`515465a` 〜 `f81e7d4` で順次解消 (BENCH §19 詳細)
  - Chrome WebGPU 上 **512 Constant FFT mode = 46.5 fps 実機確認**、
    Stage 2 が browser でも reproducible
  - 128 grid 16fps → 60fps (mixed-radix FFT routing 副産物、UX 改善)
  - 5 罠を memory に保存 (`feedback_web_chrome_webgpu_pitfalls`)、
    将来の web build で先回り checklist
- **Stage 2 final 数値 (Rev.4.9 確認、推定値追記)**:
  - 512×Constant FFT: **41-46 sps** (native 43.5 / Chrome 46.5)
  - 512×4creature Localized FFT: **41 sps native / ~44 sps Chrome 推定値**
    (Localized overhead 1.05× 既知、実機未測定 — web app に 4 creature
    UI 追加が必要、M6.C-3 締めスコープ外で後日 separate commit 可)
  - judgment C (40-50 sps バケット → 案 a 確定) Rev.4.9 と同一、native
    と Chrome 両方で再現可能と裏付け
- **60 fps 達成は案 Y で f16 NG 判明 (Entry 7)、Stage 1 die-line 保護
  優先で 41-46 sps 案 a 確定**:
  - 将来 60 fps path: f16 全層化は per-step truncation の chaos
    amplification で Stage 1 g256 を 916× regression、structural に
    NG
  - 残された path: subgroup matrix (Chrome 限定、案 P)、Apple Silicon
    の register/cache architecture を活かす別アプローチ、HW 更新待ち
- **web 修正の数値検証 (adversarial-reviewer 重点 review 反映)**:
  - `1afd656` CPU rustfft 化: GPU radix-4 vs rustfft envelope は
    N=64 7.1e-6 / N=128 3.967e-5 / N=256 3.759e-5 / N=512 5.012e-5
    (実測、M6.C-3-8 で N=128 witness 新規追加で完備)
  - g256 10-step rel: 2.174e-4 → 4.397e-4 (Lyapunov amplification of
    static seed perturbation)、BENCH §8 design intent baseline 1.1e-3 を
    依然下回り、tolerance 2.5e-3 内 safe
  - `e371dfd` uniform-CF barrier wrap: 数値完全不変 (production WG
    size == n で dead code、reviewer 直接実機確認)
  - case Y との structural difference: 本件は once-at-startup static
    seed perturbation、case Y は per-step f16 truncation で fresh
    divergence 累積、質的に異なる
- **g128 m1_regression disposition**:
  - `4593291` で production Auto routing が g128 Direct → FFT
  - `gpu_field_regression_g128` test を `ConvolveMode::Direct` で pin
    (snapshot_regression.rs:195 と同 pattern)、A.4.5 baseline
    4.460e-4 を保護、production routing は変更維持
  - FFT 経路 g128 envelope は `fft_2d_forward_matches_rustfft_n128`
    (新規) + 既存 mixed-radix bit-equal で cover
- **§8 マイルストーン状態**: M6.A/B/C-1/C-2/C-3 = ✅、**Stage 1 主目標
  達成 (256, 146 sps)、Stage 2 案 a (512, 41-46 sps native + Chrome)
  で確定**、次は M5 (進化的探索 + Eq. 8 stochastic sampling) へ
- **CLAUDE.md updates**: Chrome WebGPU 5 罠 (Apple Metal で通ったが
  Chrome で壊れる) を memory に保存 + BENCH §19 に詳細記録、今後の
  web build で check 必須

### Rev.4.9 (2026-05-30、M6.C-3 完了 + Stage 2 final 案 a 確定)

- **M6.C-3 全体完了** (overnight self-driven、Phase 3 ワークフロー):
  C-3-1 mixed-radix FFT / C-3-2 512 wiring / **C-3-3 per-pass
  breakdown infrastructure + 判断** / **C-3-4 f16 試行 → 即捨て revert** /
  **C-3-5 reintegrate workgroup tiling → 0× 即捨て revert** /
  **C-3-6 Stage 2 final 測定** / C-3-7 retro (本 Rev)。SHA 一覧は
  BENCH.md §18 sub-step inventory 参照。
- **Stage 2 final 確定 (案 a 適用)**:
  - **512×512×3ch×4creature×Localized = 24.19 ms/step (41.3 sps)**
    (BENCH §18 最終 N=3 median)
  - 60 FPS budget (16.67 ms) に対し **+45.2% over** = 60 FPS **未達**
  - judgment C「40-50 sps バケット → 40+ fps で確定、深追いせず」
    適用、**届いた最高 41.3 sps で確定し M5 へ進む**
- **主要 measured 観察** (BENCH §18 / overnight_log.md):
  - **C-3-3 per-pass breakdown**: TIMESTAMP_QUERY 経路が wgpu 29 +
    Metal でハング → CPU clock variant に切替、sanity check で bit-equal
    検証済み。real % 補正後 reintegrate 51.5% + convolve 43.6% = 95%
    の 2 大 pass 構造を確定
  - **C-3-4 f16 kernel_fft**: 実測 1.019× total (期待 ~1.18×)、
    convolve の SM pass の 1ms 部分にしか効かず、FFT 内部 (8ms) には
    影響なし → 即捨て revert
  - **C-3-5 reintegrate workgroup tiling**: 実測 0.999× total (期待
    1.30×)、M1 Apple Silicon の大 L1 cache が naive gather を既に
    吸収しており shared memory 経路でも同等 → 即捨て revert
- **Apple Silicon GPU architecture 知見** (BENCH §18 残された手段
  + Entry 5):
  - discrete GPU で効く memory-bound tile optimization が Apple
    Silicon では大 L1 cache に吸収されて効かない、という transferable
    な知見
  - 60 FPS gap 1.45× を埋めるには **FFT 全 intermediate buffer の
    f16 化** (channel_spectra / scratch_complex / k_spectra を u32
    packed f16 + unpack2x16float decode) が唯一の有効 path、これは
    M5 hook or 別 milestone で再アタック
- **profiling infrastructure 残置** (C-3-3 deliverable):
  - `GpuContext::new_blocking_with_timestamps` (TIMESTAMP_QUERY ctx、
    将来 root cause 追跡用に残置)
  - `GpuStepPipeline::profile_passes_fft` (CPU clock per-pass timing、
    relative breakdown のみ信頼可と rustdoc 明示)
  - `bench_512_breakdown` + `bench_512_reintegrate` (focused Stage 2
    bench)
  - `probe_shader_f16` (将来 FFT 全 f16 化を再検討する時に再利用)
- **判断 B (≥1.2× 採用 / <1.1× 即捨て) の有効性**: Phase 3 早期撤退
  ロジックで時間を溶かさず止められた (C-3-4 + C-3-5 合計 ~3h、各
  手法 約 1h 以内で判定 + revert + commit)
- **§8 M6 セクション更新**: M6.A/B/C-1/C-2/C-3 = ✅、**Stage 1 主目標
  達成 (256, 146 sps)、Stage 2 案 a (512, 41.3 sps) で確定**、次は
  M5 (進化的探索 + Eq. 8 stochastic sampling) へ
- **未達 60 FPS の future work 候補**: BENCH §18 残された手段 1-3 に
  記録。M5 開始時の reassess で扱うか、別 milestone (M6.D?) として
  scope 切り直すかは Ponyo877 さん判断

### Rev.4.8 (2026-05、M6.C-2 完了 + Stage 1 主目標達成 + M6.C-3 計画)

- **M6.C-2 (kernel fusion + parameter map P infrastructure) 全体完了**。
  C-2-2 / C-2-4 戦略確定 / C-2-4-a〜d / C-2-1-a / C-2-5 / C-2-6 の 9
  commits。SHA 一覧と内容は BENCH.md §16 sub-step inventory 参照。
- **Stage 1 中間評価: 主目標達成 ✅ (Ponyo877 さん 2026-05-28 判断)**:
  - **256×256×C3×4creature = 6.84 ms/step (146 sps)** (BENCH §15 config 5)
  - 撤退ライン (30 FPS = 33.3ms) を **4.87×**、主目標 60 FPS (16.7ms) を
    **2.44×** 上回る
  - 当初 M6.B Amdahl extrapolation (13-22 sps) を **6-11× 上回る**圧倒的
    好結果
  - **主目標 (256×256×4creature×60FPS) 達成済みと確定**
- **主要 measured 観察** (BENCH §15 / §16):
  - **N=256 FFT-vs-Direct = 34×** が §14 extrapolation (3-5×) を約
    6.8-11× 超過。Direct の per-cell O(kernel_area × K × C) が N=256 で
    catastrophic (232 ms)、FFT は O(N² log N) で緩やか → ratio は N 増で
    **増加** (§14 の減少仮定は誤り、§16 で訂正)
  - **C-2 perf micro-opt (C-2-1-a fused inverse + C-2-2 SM unroll) の
    end-to-end 効果 ≈ ゼロ** (1.04× / 0.99×、thermal noise band 内)。
    原因: 削減した dispatch の大半が安価な copy、FFT path は
    compute-bound で dispatch 削減の限界効用低い (CLAUDE.md 原則 1 に
    従い原因究明、対症療法せず)
  - **localized 4-creature overhead = 1.063× (6.3%)** のみ。case δ
    paper-faithful parameter map P infra は実用上 negligible overhead
- **case δ infrastructure 完成** (C-2-4-a〜d): Plantec 2025 §3.1
  parameter map P (per-cell K-vector) + AffinityGrowthPass localized
  (Eq. 7) + ParameterFlowPass (Eq. 8 M5 hook、identity-copy)。Eq. 8
  stochastic sampling は `docs/M6_C2_4_creature_design.md` §"M5 hook
  specification" に明文化
- **戦略決定: 512 高性能エンジン (M6.C-3) を取りに行く** (Ponyo877 さん
  2026-05-28、「理論値の超高性能エンジンを完成させてから M5 進化的探索
  へ」):
  - 256 で over-engineering だった subgroup / mixed-precision を 512 で
    「60 FPS 達成に必要」に転用
  - 512 = 2^9 は radix-4 非対応 → **mixed-radix FFT (radix-4 × 4 +
    radix-2 × 1)** が技術的核心 (M6.C-3-1)
  - 512 naive 外挿 ~32 sps、60 FPS まで追加 1.85× → deferred 手法積
    (subgroup × mixed-precision × workgroup tuning = 2.34-4.5×) で射程内
- **§8 M6 セクション更新**: M6.A/B/C-1/C-2 = ✅、**256 主目標達成**、
  M6.C-3 (512 高性能 FFT エンジン、7 sub-step) を正式計画。旧 M6.C-3
  (mixed-precision) / M6.C-4 (4 creature) は M6.C-2 (4creature 完了) +
  新 M6.C-3 (512 + subgroup + mixed-precision) に再編
- **subgroup ops は Chrome 限定 (案 P)**: M6.C-3-3 subgroup reduction
  は Chrome のみ、Safari/Firefox は subgroup なし fallback path (M5 で
  完成、M6.C-3 では Chrome path 優先)。SNS 公開時「512 ハイエンドは
  Chrome 推奨」運用
- **60 FPS 未達時の方針 (案 a)**: M6.C-3-6 で 512 が 60 FPS 未達でも
  「届いた最高 FPS で確定」(例 45 FPS でも 512 ハイエンドとして価値)、
  最後の極限最適化に固執せず M5 へ進む

### Rev.4.7 (2026-05、M6.C-1 WGSL FFT 実装完了)

- **M6.C-1 (WGSL FFT 実装) 全体完了**。M6.C-1-1 から M6.C-1-6 まで 9
  commits (内部分割 6 sub-step、C-1-4/C-1-5/C-1-6 はそれぞれ -a/-b に
  分割)。SHA 一覧と各 sub-step 内容は BENCH.md §14 sub-step inventory
  参照。
- **主要成果**:
  - **FFT 化 end-to-end speedup** (paired-run N=3 median quiesced、N=64):
    - C=1: direct 13.31 ms (75.1 sps) → fft 1.62 ms (616.4 sps)、ratio **8.206×**
    - C=3: direct 16.33 ms (61.2 sps) → fft 1.89 ms (529.9 sps)、ratio **8.655×**
    - C-1-4-b 早期撤退ゲート (≥ 2.0×) 両 channel count PASS、自走 commit
  - **Cooley-Tukey radix-4 algorithm** 採用 (M6.B 文献調査 §2.5 fgiesen
    推奨)。Stockham + ping-pong は L1 圧迫で却下。動的 N ∈ {64, 256}
    は WGSL pipeline-override constant で実現
  - **Method B inverse** (`IDFT(y) = (1/N) conj(DFT(conj(y)))`): conjugate-
    twiddle 単独は radix-4 butterfly の `complex_mul_i` (= `i*(q1-q3)`)
    も符号反転必要で fail (C-1-2 で N=4 手計算で発覚)、Method B で fix
  - **kernel pre-FFT** (起動時 1 回、K × N×N complex を 1 storage buffer
    に concat、5.24 MB at N=256/K=10)
  - **ConvolveFftPass** (kernel routing で multi-channel + per-kernel
    source_channel 対応): direct ConvolvePass の `meta_arr[ki].source_channel`
    semantics を WGSL `kernel_routing` storage buffer + indirect read で
    実現、single SM dispatch 維持
  - **ConvolveMode::Auto fallback**: grid 64/256 で fft、grid 32/128/512
    は direct fallback (mixed-radix 未対応、scope-guardian C-1-2 で defer)
- **物理的観察**:
  - **C=3 で end-to-end 8.7× speedup** (当初 BENCH §13 predict 3-4× を 2×
    超過): direct path は per-kernel × per-channel inner loop で C scaling
    重い、FFT path は K kernels 共有 + per-channel forward 1/channel で
    C scaling 軽い
  - **long-horizon stability** (bench_long_horizon_fft):
    - horizon 10 で既に A.4.5 tolerance violation 開始 (C=1 random max_rel
      8.30e-4 vs tolerance 5e-4)
    - horizon 50-100 で chaotic saturation (max_rel ≥ O(1))、direct と FFT
      は same Lyapunov attractor 上の different trajectories
    - per-step amplification factor 1.11-1.24× (geom over 10→100、
      saturation 込みなので representative ではない)
  - **identical-kernels controlled experiment**: kernel parameter scaling
    は寄与あるが dominant ではない、FFT inject が主因
  - **Layer 4 (snapshot regression) は短期 horizon (≤ 5-10 step) でのみ
    meaningful**、長期 horizon は Lenia chaos の性質上 physically impossible
- **方法論的成果** (M6.C-2 以降で再利用):
  - **early-exit gate**: bench_fft_vs_direct paired-run で ratio < 2× なら
    Phase 3 改訂条件 2 で Ponyo877 さん戦略相談、≥ 2× で自走 commit
    (C-1-4-b で確立、C-1-5-b で C=3 拡張)
  - **scope-guardian re-consultation pattern**: Option β3 (元 approve) →
    Option β2 (本実装で feature regression 発覚 → 再 approve) の自走判断
    (Rule 3 trigger なし) を C-1-5-b で確立、subagent 介在で戦略 escalation
    回避
  - **honest framing**: ratio claims に "当初予想超過は C scaling 効果で
    legitimate" 等 commit body で reasoning 提示、CLAUDE.md 原則 4 遵守
- **§8 M6 セクション**: M6.A = ✅、M6.B = ✅、M6.C-1 = ✅、M6.C-2 (kernel
  fusion + subgroup ops) 次着手、M6.C-3 (mixed-precision) は Stage 1 で
  採用判断、M6.C-4 (4 creature 実装) は M6.C-2 後
- **Stage 1 中間評価入力**: BENCH §14 で raw measurement + Amdahl
  extrapolation の framing。**戦略判断 (撤退 / 継続 / 縮小 / 目標再評価)
  は Ponyo877 さん責任**、本 Rev 4.7 は input 提供
- **M6.C-2 以降へ引き継ぐ未解決事項**:
  - bench_fft_breakdown (per-pass timing for fft mode) は C-1-6-β scope
    creep 回避で defer、C-2 perf phase で実施
  - perf_regression baseline update も C-1-6-β code-only commit に絞り、
    BENCH §1 数値の手動 update は C-2 全 grid benchmark 後に纏める
  - N=256 long-horizon 実測は cold-boot quiesced session 必要、Stage 1
    後の C-2 perf phase で実施 (本 Rev 4.7 は Amdahl extrapolation のみ)
  - kernel routing UI invalidate (param-painting M5 candidate) は M6.C-4 /
    M5 で対応 (`update_globals` rustdoc TODO 既記載)
- **M6.C-2-4 4 creature 実装方式確定 (戦略判断 2026-05-27)**:
  - **case δ (paper-faithful parameter map P infrastructure)** 採用
  - **Eq. 8 stochastic sampling は M5 defer** (進化的探索の核心、文脈
    として M5 で実装が自然)
  - creature 数 4 維持 (M6.C-1 計画書通り、64 は計算量 unknown)
  - 詳細は `docs/M6_C2_4_creature_design.md` 参照
  - scope-guardian 元 approve (case α) は Plantec paper PDF 本文直接読了
    で覆った retro: `CLAUDE.md` "Subagent 判定の事後検証 (M6.C-2-4 retro)"
    節に教訓記録

### Rev.4.6 (2026-05、M6.A 検証基盤完了)

- **M6.A (validation infrastructure expansion) 全体完了**。M6.A.0
  から M6.A.9 まで（中間の A.2.1 / A.4.5 / A.10 / A.11 を含む）。
  A.0-A.8 + CLAUDE.md 測定プロトコル commit までの SHA は
  BENCH.md §13 sub-step inventory 表に列挙。A.9 commit の SHA は
  本ファイルがその commit に含まれるため `git log --grep 'M6.A.9'`
  で参照可能。**M6.A.7.1 backlog (task #132) は M6.C-0** (post-M6.B、
  pre-M6.C-1 milestone) **で完了**: ValidationGuard 適用範囲が lib unit
  tests 21 + diagnose_divergence 4 にも拡張、coverage 43/47
- **M6.A の主要成果**:
  - **5-layer 数値回帰**: bit-equal CPU (m1_regression g32-g256) /
    mass conservation (5 grids 32-512) / CPU-GPU C=1 short-horizon /
    GPU pre-post snapshot (A.5) / creature-alive sanity (A.11)
  - **perf regression**: ±5 % warn / ±20 % err、3-run median、A.6
    re-anchored baselines（cold-boot vs warm-state drift を明文化）
  - **WebGPU validation guard**: `FLOW_LENIA_VALIDATE=1` opt-in、
    M6.A.7 で integration tests 17/46 をカバー、**M6.C-0 で 43/47**
    へ拡張済 (lib unit tests 21 + diagnose_divergence 4 追加、BENCH §10
    参照、4 件の CPU-only test は wgpu surface 不在で N/A)
  - **heap leak regression**: 10K-step CPU heap delta < 500 KB、
    mid-loop sample で transient/leak 弁別
- **M6.A で確定した重要な物理的観察**:
  - **C=1 でも grid ≥ 64 で強カオス** (M6.A.4.5)。ε = 1e-6 摂動が
    1 step で O(0.8) に飽和、GPU vs CPU rel が grid² 系で増幅。
    chaos limit 内で tolerance を grid-tiered 設定
  - **GPU bit-determinism は process / 日数を跨いで保持**（M6.A.5 / A.7
    で 3 日跨ぎ同 byte 再現を確認）。snapshot regression が成立する基盤
  - **Cold-boot vs warm-state perf 差 7-27 %** (M6.A.6)。M1 thermal
    accumulation で同 session 連続 perf 測定は信頼できず、commit-to-
    commit drift detection には typical-state baseline を anchor
- **方法論的成果** (M6.B / M6.C / M5 で再利用):
  - **Layered tolerance**: 数値比較は scenario ごとに budget を変える
    (bit-equal / mass / GPU-CPU C=1 / GPU-snapshot / creature sanity)
  - **Honest framing**: noise band 以下の改善は overclaim せず、
    measurement と extrapolation を明示的に区別 (A.6/A.7/A.8 で実践)
  - **Subagent review workflow**: scope-guardian + adversarial-reviewer
    で pre/post-implementation の品質ゲート。A.7-A.9 で実用化
  - **Paired-run measurement protocol** (`CLAUDE.md` 測定プロトコル
    節): off/on 同 thermal state、quiesced host、N=3 median
- **§8 M6 セクション**: M6.A = ✅ 完了、M6.B (文献調査) 着手準備、
  M6.C (per-pass optimization / FFT 化) は M6.B 後着手
- **M6.B / M6.C / M5 へ引き継ぐ未解決事項**:
  - ~~M6.A.7.1 (task #132): lib unit tests への validation 拡張~~
    **→ M6.C-0 で完了** (post-M6.B、pre-M6.C-1 milestone)
  - GPU memory monitoring 自動化: 現状 Activity Monitor 手動、
    M6.A スコープ外（将来の M6.C/M5 で必要性再評価）
  - FFT 設計詳細: BENCH §1/§2 の per-pass 占有率 (convolve 97.4 %)
    を踏まえ M6.B 文献調査 → M6.C で実装計画

### Rev.4.5 (2026-05、M4 全体完了)

- **M4 (UI 統合・リアルタイム可視化) 全体完了**。M4.1 から M4.6 まで 11 サブステップで実装、commit は M4.0.5 〜 M4.6 で計 12 個 (origin/main HEAD = `59943b8`)。Chrome 148 WebGPU での実機検証も完了し、Flow-Lenia ブラウザインタラクティブ版が完成形に到達
- **完成形の達成内容**:
  - 64×64 / C=3 / |K|=10 で 40+ fps、CPU 22% (M2.11 native ベンチと整合)
  - egui SidePanel の 6 セクション (Stats / Parameters / Kernels / Grid / Actions / Keyboard) で全コントロール (paper_strict、border、dt、dd、num_kernels、grid_size、channels、Pause/Reset/Screenshot) を提供
  - キーボード Space / R / Q + マウスインタラクション両対応
  - Screenshot PNG download (offscreen Rgba8Unorm + wgpu readback + image crate)
  - per-channel mass realtime readback (30 frame 周期、async map_async + AppEvent dispatch)
- **主要な技術的判断**:
  - **render loop の RAF 移行** (M4.5.1, commit `c535f6a`): `about_to_wait` 駆動の uncapped ループが Chrome 148 WebGPU の `Surface::present` 非 vsync block と相まって 131 fps 暴走 → 各 visible frame の sim step 数が不均一になりカクつき発生。`requestAnimationFrame` から WASM の `tick()` を呼ぶ JS-driven 構造に切替えて compositor pace と一致させ、解消
  - **`ControlFlow::WaitUntil(16ms)` で event loop を 60Hz wake に**(M4.5.1.2, commit `d633dbd`): `Poll` は web で MessageChannel busy-loop となり M1 mini で 90% CPU、`Wait` は discrete event drop。`WaitUntil` で妥協点 (22% CPU + discrete event 即時応答) に到達
  - **`thread_local!` + `RefCell<Option<AppState>>` + `try_borrow_mut` 防御** (M4.5.1, M4.5.1.1): RAF tick と winit event handler が同一 `AppState` を共有する必要から導入。`device.poll` の同期 callback 経路での double-borrow を防ぐため `tick()` / `window_event` / `user_event` で `try_borrow_mut` skip on contention
  - **offscreen Rgba8Unorm + 256-byte row alignment readback** (M4.2, commit `2197c13`): Chrome WebGPU の `canvas.toBlob` が swap-chain texture を読めず黒画像になるバグを回避。`VisualizePass` を別 RT に再生成 → `copy_texture_to_buffer` → `map_async` → `image` crate PNG エンコード経路で実装
  - **`pending_*` field + `rebuild_pipeline()` 統一 helper** (M4.4 / M4.5): egui closure 内の `&mut state` 借用制約を回避しつつ Reset / Apply Kernels / New Seed / Apply Grid を単一 rebuild path で処理。`pending_rebuild: Option<RebuildRequest>` を closure 外で消費する flag pattern
  - **`FrameTimingDiag` 一時計測コード** (M4.5.1): カクつき診断時に `about_to_wait` (後に `tick()`) 内で interval / render duration ヒストグラムを 300 サンプル毎にログ出力。M6 性能調査でも再利用するため debug build に常設、release では log level で抑制 (M6.0 で扱い検討)
- **`docs/known-issues.md` 更新**: #1 (RedrawRequested 非到達) を M4.5.1 RAF 移行で obsolete としてマーク、#4 (Chrome 148 WebGPU `Surface::present` 非 vsync block) を新規追加。診断方法と回避策の根拠を記録
- **§8 マイルストーン状態**: M4 = ✅ 完了、M6 = 着手準備 (M6.0 性能調査) へ。M5 (creature 探索) は M6 完了後に着手予定 (Rev.4.2 の順序維持)
- **残った既知の課題 (M6 / M5 で対応)**:
  - **`wasm-opt -Oz` 適用 + bundle size 計測**: dev build 26 MB → release で大幅縮小見込み (現在 trunk 0.21.14 はデフォルトでは wasm-opt を呼ばない、Rev.4.4 で既述)
  - **Safari 26 / Firefox 150 での詳細 perf 測定**: M3.5 で動作確認のみ、各 grid size の FPS 比較は未取得
  - **egui `SidePanel::show(&Context, ...)` の Ui-centric API 移行**: egui 0.34 で deprecated、現在 `#[allow(deprecated)]` で抑制中、egui 0.35+ への upgrade と合わせて対応
  - **Paused 時のフレーム cap**: `state.running = false` でも render_frame は走り続けるので CPU 数% 消費が残る、2Hz wake 等への throttle を検討
  - **60Hz 表示で実測 45 fps の調査**: native 55+ fps との差の原因究明 (M6.5 で予定)
  - **winit / wgpu の `RedrawRequested` 上流修正の追跡**: 修正されれば RAF パスを残したまま winit 駆動経路も復活可能
  - **`FrameTimingDiag` の release ビルド扱い**: log level 抑制 (現在 INFO) で済むか、conditional compilation で完全に削るかを M6 着手時に判断

### Rev.4.4 (2026-05、M4.1 着手前の rust-toolchain 上げ)

- **`rust-toolchain.toml` の channel pin を `1.87.0` → `1.95.0`** (M4.0.6)。理由: M4.1 で採用予定の `egui 0.34.2` (+ transitive `epaint`, `vello_cpu`) が MSRV 1.92 を要求、Rev.4.1 で採用した 1.87 ではビルド不可。1.95.0 は 2026-05 時点の Rust stable 最新版
- **§1.1 (Rust toolchain) と §1.2 (wgpu / wasm 関連) の表記**を 1.95.0 へ追従
- **M1.15 fixture は再生成不要**: rustc 1.87 → 1.95 で `m1_regression_matches_baseline_fixtures` が **bit-identical** で通った (CPU 参照実装の直接畳み込み + reintegration が LLVM 最適化レベルで同じ命令列に落ちている)。manifest.json の "rustc 1.87" 表記は据え置き — 1.95 でも同じ bit パターンを再現できることが反証検証で示された
- **回帰テスト結果**: cargo test --workspace --release で 138 tests 全 pass / 3 ignored、M2.8 GPU 比較も bit-identical 維持、clippy clean。native_gpu 5374 step / 42-43 fps、Chrome WebGPU で creature 表示確認 (M4.0.5 と同じ same-seed リング creature を再現)
- **wasm-opt 適用問題**: M4.0.6 で trunk release build の wasm が 2.7 MB と判明。M3.5 報告の 515 KB は wasm-opt 適用後の値で、trunk 0.21.14 が wasm-opt をデフォルトで呼ばないことを確認。サイズ最適化は M4 完了後 / デプロイ前 (M5 後) で集中対応

### Rev.4.3 (2026-05、M4 着手前の依存版上げ)

- **§1.2 wgpu pin を `=25.0.x` → `29` に変更**。理由: M4.0 調査で `egui-wgpu 0.34.2` (M4 で採用予定の最新 egui 統合 crate) が wgpu 29 系に固定されていることを確認、Rev.4.2 時点で「eframe 0.34 = wgpu 25 バンドル」を前提とした採用判断 (Rev.4 当初の「eframe を使う」想定) が **M4 で eframe を使わず `egui` + `egui-wgpu` + `egui-winit` を直叩きする方針** によって失効したため
- **§1.2 表内の `winit` pin を `=0.30.x` → `=0.30.13`** に変更 (`egui-winit 0.34.x` の optional dep)
- **§1.2 table の備考**: 「MSRV は 1.76」→「MSRV は 1.87 (wgpu 29 が要求)」
- M4.0.5 として独立コミットで実施。回帰検証: cargo test --workspace --release で 138 tests 全 pass、M1.15 fixture (mass conservation 8 ケース) max_rel ≈ 2e-5 で M2 baseline と整合、bench_step 64×64 / C=3 で GPU/CPU = 0.34 (M2.11 = 0.31、+9.7% 微増、許容範囲内)、native_gpu / Chrome WebGPU 両方で creature 動作確認済み
- API surface の主な機械的変更点 (DEV メモ): `Instance::new` が値渡しかつ `InstanceDescriptor::new_without_display_handle()` 経由、`PipelineLayoutDescriptor` の `bind_group_layouts` が `&[Option<&BindGroupLayout>]`、`push_constant_ranges` 廃止 → `immediate_size: 0` (wgpu 28 rename)、`RenderPassColorAttachment` に `depth_slice: None` 追加 (wgpu 26)、`RenderPipelineDescriptor::multiview` → `multiview_mask: Option<NonZero<u32>>` (wgpu 28)、`PollType::Wait` が `{ submission_index, timeout }` 形式 (wgpu 27)、`Surface::get_current_texture` が `CurrentSurfaceTexture` enum (wgpu 29)、`DeviceDescriptor` に `experimental_features` field 追加 (wgpu 27)

### Rev.4.2 (2026-05、M2 完了時の方針変更)

- **§8 マイルストーン順序を入れ替え**: M3 → M4 → **M6 → M5** に変更 (旧: M3 → M4 → M5 → M6)。理由: 「ブラウザで高速で最適化された Flow-Lenia」というゴールから逆算し、M5 進化探索を最適化済み GPU pipeline で実行する方が効率的。M5 完了後に creature 発見 + SNS 公開という流れに統一
- **M3 サブステップ M3.1〜M3.5 を明記**: M3 を「最小 WASM ビルド → Hello WebGPU → 単一 compute pass → 全パイプライン統合 → ブラウザ互換性確認」の 5 段階に分解
- **M3 完了条件を縮小**: Chrome stable で 64×64 / C=3 / |K|=10 動作確認のみ。Safari / Firefox は **状況報告のみ**。デプロイは外し M5 完了後に別途実施
- **§10 Q9 (デプロイ) を更新**: 「ローカル `trunk serve` のみ」→「M5 完了後に GitHub Pages or Cloudflare Pages にデプロイ + SNS 公開」
- **§10 Q1 (ブラウザ優先度) を補強**: M3 段階では Chrome stable のみ、Safari / Firefox は M5 後デプロイ時に再評価
- **M6 内容を M5 前に前倒し**: FFT convolution (DESIGN.md §8 M6 optional stretch) + bind-group caching などの最適化を M5 進化探索の前段で実装。これにより M5 探索が高速化された pipeline 上で走るため、より多くの creature を時間内に発見可能
- **M2 完了時点の実測値を §8 M6 の前提に反映**: 32×32 で GPU/CPU = 0.51×, 64×64 で 0.27×, 256×256 で 0.27× (BENCH.md 参照)。convolve が per-step 97.4% を占有することを発見、M6 FFT 化の量的根拠として記録

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
- **toolchain: stable 1.95.0** で pin (Rev.4.4 で 1.87.0 → 1.95.0)
  - 1.87 を選んだ Rev.4.1 当時の理由 (edition 2024 transitive 依存、wgpu 29 の MSRV) は変わらず満たされる
  - Rev.4.4 で 1.95 へ更新した理由:
    - M4.1 で採用する `egui 0.34.2` (および transitive `epaint` / `vello_cpu`) が MSRV 1.92 を要求、1.87 では cargo check が通らない
    - 1.95.0 は 2026-05 時点の stable 最新版で十分 stable
  - 反証検証: rustc 1.87 → 1.95 で M1.15 baseline fixture は **bit-identical** で通った。CPU 参照実装が直接畳み込み中心で LLVM 最適化に左右されにくいため
- `rust-toolchain.toml` をリポジトリに置き、コンポーネント `rustfmt`, `clippy`, `rust-src` を指定。channel は具体的なパッチバージョン (例: `1.95.0`) で pin して `stable` rolling channel は使わない (M1 fixture の bit-identical 再現性のため)
- nightly 不要

### 1.2 wgpu / wasm 関連バージョン

調査結果 (2026-05 時点):

- `wgpu` 最新安定版: **29.0.3** (2026-03 リリース、MSRV 1.87)
- `egui` / `egui-wgpu` 最新安定版: **0.34.2** (2026-05)、**wgpu 29 系に直接対応**

**採用 (Rev.4.3 で更新)**: **wgpu 29 + `egui` 0.34.x (eframe を使わず `egui` + `egui-wgpu` + `egui-winit` を直叩き)**。Rev.4.2 までは「eframe 0.34 が wgpu 25 をバンドルする」前提で wgpu 25 を採用していたが、M4 着手時の調査で `eframe` 抜きの `egui-wgpu` 直叩きが wgpu 29 と整合することが判明、`eframe` 自体のアップグレード待ちを介さず最新 wgpu を採用できるため移行。MSRV は wgpu 29 要求の **1.87** で従来 (Rev.4.1) 採用バージョンと同じ。

最終バージョン pin (Cargo.toml で `=` 固定、実装着手時の最新パッチを使用):

| crate | バージョン | 用途 |
|---|---|---|
| `wgpu` | `29` | GPU API (MSRV 1.87、Rev.4.3 で 25 → 29、ただし Rev.4.4 で toolchain 自体は 1.95 に追従) |
| `winit` | `=0.30.13` (`egui-winit 0.34.x` の optional dep) | ネイティブウィンドウ |
| `wasm-bindgen` | `=0.2.x` | WASM ⇄ JS 境界 |
| `web-sys` | `=0.3.x` | DOM / Canvas / WebGPU 型 |
| `console_error_panic_hook` | `=0.1.x` | ブラウザでのパニック表示 |
| `bytemuck` | `=1.x` | バッファシリアライズ |
| `glam` | `=0.29.x` | ベクトル/行列 |
| `egui` / `egui-wgpu` / `egui-winit` | `=0.34.x` (M4.1 で追加) | UI、eframe 抜きで直叩き |
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

### プロジェクト完了像 (Rev.4.2 で再定義)

すべての milestone (M1–M6) 完了時点で達成される最終像:

- **ブラウザ完結**: Chrome stable で URL を踏むと即起動、WASM + WebGPU で動作
- **最適化済み**: FFT-based convolution と GPU pipeline 最適化で
  64×64 / C=3 で 60 FPS、128×128 / C=3 で 30 FPS
- **インタラクティブ**: egui UI で grid size、mode、kernel 数等が動的変更可能
- **creature 発見済み**: 進化的探索で論文 Figure 4 相当の creature を
  発見、URL parameter でその creature を再現可能
- **共有可能**: SNS で URL を共有すると、開いた人が同じ creature を観察できる

M3 → M4 → M6 → M5 の順序で進め、M5 完了後に GitHub Pages or
Cloudflare Pages にデプロイ。

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

### M3: WASM ビルド + ブラウザ WebGPU 動作 (動作確認のみ、デプロイは M5 後)

**成果物**: `trunk build --release` で `dist/` にローカル実行可能な成果物。**デプロイは M5 完了後に別途実施**。

**完了条件 (Rev.4.2 で縮小)**:
- **Chrome stable** で 64×64 / C=3 / |K|=10 が動作 (M1.14 / M2.10 の seed=1729 が同じ rotating petal pattern を見せる)
- **Safari / Firefox は状況報告のみ** (動けば良し、動かなければ M5 後デプロイ時に再対応)
- WebGPU 非対応ブラウザでフォールバックエラー表示
- adapter limits 取得 + UI 動的制限 (512² 無効化等)
- 検証方針 (M2 と同じ二重ガード): **5-10 step trajectory CPU vs WASM-GPU で rel < 1e-4 OR abs < 1e-5**、**100 step mass conservation で rel < 1e-3 (torus) / 1e-2 (wall)**。Chrome の WebGPU 実装が native Metal と異なる加算順序で chaotic drift を起こすのは正常 (M2.8 verify 参照)

#### M3 サブステップ (Rev.4.2 追加)

- **M3.1 最小 WASM ビルド**: `flow-lenia-core` の純粋計算部分 (compute_kernel, growth, sum_channels 等の non-GPU 部分) を `wasm32-unknown-unknown` でビルド可能にし、Node.js / wasm-pack でユニットテストが通る
- **M3.2 Hello WebGPU**: M2.1 の `native_gpu` 相当 (青い画面のみ) を Chrome stable で表示。`web-sys` + `wasm-bindgen` + `wgpu` の wasm32 ターゲット組合せの sanity check
- **M3.3 単一 compute pass WASM**: M2.3 convolve pass を WASM 上で走らせ、readback 結果を CPU 参照と相対誤差 < 1e-4 で一致確認
- **M3.4 全パイプライン + visualize WASM 統合**: M2.10 の `native_gpu` 相当 (seed=1729 で creature animation) を Chrome stable で動作確認
- **M3.5 Chrome stable 動作確認 + Safari/Firefox 状況報告**: M3.4 完了状態を Chrome stable で 30 秒以上走らせて mass conservation を確認、Safari/Firefox は試行のみ (動作・非動作・部分動作を README で報告)

### M4: UI 統合・リアルタイム可視化 ✅ (Rev.4.5 で完了)

**成果物**: egui コントロール一式、状態テクスチャ表示、統計

**完了条件**:
- §7 の全コントロール動作 (`paper_strict`, `border`, `dd`, init patch range 含む) — ✅ 達成 (M4.4 / M4.5)
- 128×128 / C=3 / |K|=45 で 30 FPS 以上 (実測ベンチ) — ⚠ 64×64 は 40 fps 達成、128×128 は M2.11 実測で 17 sps、M6 FFT 化後に再評価
- 最低 1 つ「明らかに動く creature」が見える固定シードのデモを README に掲載 — ⚠ creature 表示は確認済 (Chrome 148 / Safari 26 / Firefox 150)、README 掲載は M5 デプロイ時に実施

**実装サブステップ** (Rev.4.5):
- M4.0.5 (`d16abd0`): wgpu 25 → 29 + winit 0.30.13 upgrade
- M4.0.6 (`b8a5134`): rustc 1.87 → 1.95 (egui 0.34 MSRV)
- M4.1 (`8e7eaa7`): egui + egui-wgpu 最小統合 + about_to_wait workaround
- M4.2 (`2197c13`): Pause / Reset / Screenshot ボタン (PNG offscreen readback)
- M4.3 (`4606191`): realtime FPS / step / per-channel mass 表示
- M4.4 (`78fae7c`): live parameter sliders + kernel Apply / New Seed
- M4.5 (`8b17b54`): grid-size + channels dropdown
- M4.5.1 (`c535f6a`): render を RAF 駆動化 (カクつき修正)
- M4.5.1.1 (`1b2933a`): discrete-event delivery + RefCell 防御 + diag filter
- M4.5.1.2 (`d633dbd`): WaitUntil(16ms) で CPU 90% → 22%
- M4.6 (`59943b8`): SidePanel polish + 残存日本語英語化

### M6: 性能チューニング (Rev.4.2 で M5 の前に前倒し)

**進捗** (Rev.4.8 更新):
- ✅ M6.A 検証基盤 (validation infrastructure): A.0-A.9 完了。
  詳細は BENCH.md §5-§13 / Rev.4.6 ヘッダー
- ✅ M6.B 文献調査: WGSL FFT 実装方式確定 (Cooley-Tukey radix-4、
  workgroup-memory tiled)。BENCH §13 / Rev.4.7
- ✅ M6.C-1 WGSL FFT 実装: 9 commits、N=64 で 8.2-8.7× speedup。
  BENCH §14 / Rev.4.7
- ✅ **M6.C-2 kernel fusion + parameter map P infrastructure**: 9
  commits、case δ paper-faithful 4 creature infra 完成。BENCH §15-§16
  / Rev.4.8
- ✅ **M6.D Stage 1 中間評価: 主目標達成** (2026-05-28)。
  256×256×C3×4creature = **146 sps** (撤退ライン 4.87× / 60 FPS 2.44×
  クリア)。**主目標 (256×256×4creature×60FPS) 達成確定**
- 🔜 **M6.C-3 512 高性能 FFT エンジン**: 最終ゴール (512×512×4creature×
  60FPS) 用。mixed-radix FFT + subgroup + mixed-precision。7 sub-step
  (下記)。Stage 2 中間評価は C-3-2 完了時
- 🔜 M6.E 最終評価 + ドキュメント

#### M6.C-3: 512 高性能 FFT エンジン (Rev.4.8 で正式計画)

512 = 2^9 は radix-4 非対応 (4^4=256, 4^5=1024)。256 主目標達成済の
余力を最終ゴール 512×512×4creature×60FPS に投入。256 で
over-engineering だった subgroup / mixed-precision が 512 では 60 FPS
達成に必要に転じる。

| sub-step | 内容 |
|---|---|
| C-3-1 | **mixed-radix FFT** (radix-4 × 4 + radix-2 × 1) for N=512。技術的核心 |
| C-3-2 | 512 で FFT path 有効化 + 動作確認 (naive ~32 sps 目標)。**Stage 2 中間評価** |
| C-3-3 | subgroup reduction (Chrome 限定、SIMD lane=32)。Safari/Firefox は fallback path |
| C-3-4 | mixed-precision (kernel/twiddle f16、field f32) |
| C-3-5 | workgroup tuning (512 tile 最適化) |
| C-3-6 | 512×512×4creature 60 FPS 達成確認 + Stage 2 final。**届いた最高 FPS で確定** (案 a) |
| C-3-7 | retro + BENCH + DESIGN Rev.4.9 |

**C-3-1 mixed-radix FFT 検証**:
- rustfft (既存 dev-dep) と rel < 1e-4 一致 (N=512)
- round-trip (forward → inverse) で原 field 復元
- 既存 N=64/256 path は無変更 (regression なし)
- 5-layer test に N=512 追加 (CPU bit-equal / mass / GPU-CPU / snapshot /
  creature alive)。512 chaos tolerance は A.4.5 grid 依存から外挿
  (256 で 2.5e-3 → 512 で ~5e-3、実測で確定)

**Stage 2 中間評価 (C-3-2 完了時、Ponyo877 さん介在ポイント)**:
naive 512 FPS で判定:
- ≥ 40 sps → subgroup + mixed-precision で 60 FPS 確実
- 30-40 sps → 全 deferred 手法必要、続行
- 20-30 sps → 1.85× 境界、慎重続行
- < 20 sps → mixed-radix FFT 実装に問題、Phase 3 条件 3 で Claude Web 相談

**subgroup ops Chrome 限定 (案 P)**: C-3-3 は Chrome のみ有効、
Safari/Firefox は subgroup なし低速 fallback。runtime feature detection
で切替。SNS 公開時「512 ハイエンドは Chrome 推奨」運用。

**60 FPS 未達時 (案 a)**: 届いた最高 FPS で確定 (45 FPS でも 512
ハイエンドとして価値)、極限最適化に固執せず M5 へ。

**動的切替**: M4.5 grid dropdown を 512 対応に拡張 (256 主目標 ↔ 512
ハイエンド)、512 で重い場合「Chrome 推奨」表示。

**成果物**:
- 256×256 / C=3 / |K|=45 で 60 FPS 達成 (Apple M1 想定)
- README、CHANGELOG、シェーダ内コメント整備
- `cargo doc` 公開可能

**完了条件**:
- ベンチ表 (grid_size × C × |K| × backend × paper_strict での FPS) を README に
- シェーダ内の論文式番号コメントが全パスに揃っている

**M2.11 実測の前提** (Rev.4.2 追加):
- 32×32 / C=3 で GPU/CPU = 0.58× (GPU 1.7× 速い)
- 64×64 / C=3 で 16.3 ms/step ≈ 61 sps
- 128×128 / C=3 で 58.5 ms/step ≈ 17 sps
- 256×256 / C=3 で 230 ms/step ≈ 4.3 sps
- **convolve pass が per-step 97.4% を占有** (BENCH.md §2) → FFT 化が M6 最重要ターゲット

#### M6 optional stretch (Rev.4 追加 / Rev.4.2 で前倒し)

- **GPU FFT 化** (torus 境界限定): Stockham カーネル × 2 軸、複素 f32
  - 256² で 60 FPS を非 FFT で達成できない場合に実装 (M2.11 実測で必須)
  - 完了条件: 256² / `paper_strict=OFF` で 60 FPS 達成
  - **JAX とのビット精度近似テスト**: 固定パラメータで JAX と本実装 (FFT モード) の `A^100` を比較、相対誤差 < 1e-4
- **bind-group caching** (M2.10 の per-frame visualize BG 再構築コスト削減)
- 散逸モデル / 食物モデル (論文 §4.3.2) は M6 stretch のままに維持

### M5: パラメータ埋め込み・マルチ種シミュレーション + creature 発見 (Rev.4.2 で M6 の後に移動)

**成果物**: Eq. 7, Eq. 8 (stochastic & deterministic) の有効化、multi-patch 初期配置、変異ビーム、**最適化済み GPU pipeline 上での進化探索**

**完了条件**:
- parameter embedding ON で `tests/mass_conservation.rs` グリーン
- UI から多体配置 (64 creatures × ランダム seeds) を生成可能
- 変異ビーム実装で `test_mutation_beam` グリーン (M1 から既存だが、ここで実機統合)
- 数千ステップ走らせて、視覚的に「種が交代する」様子が観察可能
- **進化探索で論文 Figure 4 相当の creature を少なくとも 1 個発見**

### [M5 後] デプロイ + SNS 公開 (Rev.4.2 追加)

- **GitHub Pages or Cloudflare Pages** に WASM 成果物をデプロイ
- 発見した creature を固定 seed として URL parameter で再現可能に
- SNS (X / Bluesky) で公開、reproduction URL 付き

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
| Q1 | ブラウザ優先度 | **M3 段階: Chrome stable のみ**、Safari/Firefox は状況報告のみ (Rev.4.2)。M5 後デプロイ時に再評価 |
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
| Q9 | デプロイ | **Rev.4.2**: M3 段階はローカル `trunk serve` のみ、**M5 完了後**に GitHub Pages or Cloudflare Pages にデプロイ + SNS 公開 |

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
