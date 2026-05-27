# Flow-Lenia M6 開発方針

## 最終ゴール
Apple M1 mini Chrome WebGPU で 512×512 grid に 4 体 creature (相互作用あり) 60 FPS

## 撤退ライン
- 256×256×3×4creature で 30 FPS なら成果として M5 へ
- Stage 1 中間評価で「不可能」と判明したら撤退、目標見直し

## 開発原則
1. 観察した現象は対症療法せず、原因究明を先行
2. tolerance を緩める前に物理的根拠を確認
3. 「動くものを動かす」より「原因を理解した上で動かす」
4. 数値検証 (Layer 1-5) を絶対条件として保持
5. M2.8 で C=3 chaotic divergence を 3 実験で確定した姿勢を継承

## 各サブステップの完了基準
- 数値回帰テスト pass (Layer 1: bit-equal, Layer 2: mass, Layer 3: GPU-CPU, Layer 4: snapshot, Layer 5: sanity)
- 性能回帰テスト pass (±5% warning, ±20% error)
- WebGPU validation error free
- README/BENCH への根拠記載
- adversarial-reviewer subagent による approval

## Scope 制約
- 1 commit 1 関心事
- subroutine 完了報告に書かれた以上のスコープ拡大は scope-guardian に確認
- M6.B (文献調査) の並行作業は A.6-A.9 中の余力でのみ
- 「これは M6.C で扱う」「これは M5 で扱う」を明示
- **GPU memory monitoring は M6.A スコープ外** (CPU heap regression は A.8、wgpu/Metal の GPU memory は Activity Monitor 等で手動確認、自動化は将来課題)

## レビュー手順
1. サブステップ実装完了時、adversarial-reviewer subagent を起動
2. 議論ポイントは scope-guardian にも諮る
3. 両 subagent から approval が出たら commit + push
4. Ponyo877 さんへの確認は方針変更が必要な時のみ

## 完了報告の送信ルール (Phase 3 改訂)

C-1-1 / C-1-2 / C-1-3 で実証された通り、各サブステップ完了報告に
対する Claude Web 経由のコメントは「自走継続 OK」のみで戦略判断を
含まない状況が続いた。Ponyo877 さんのコピペ媒介として無駄な往復を
避けるため、以下のルールで運用する。

### Ponyo877 さん経由 Claude Web に送信する条件

以下のいずれかを満たす場合のみ送信:

1. **milestone 完了** (M6.C-1 全体、M6.C-2 全体、M6.A 全体 等の大区切り)
2. **早期撤退ゲート** (C-1-4 等の事前定義タイミング)
3. **戦略判断が必要** (撤退、目標変更、想定外発見、scope creep)
4. **Claude Code が判断に迷う** (自発的に呼ぶ)

### 上記に該当しない場合

各サブステップ完了は:

- subagent (scope-guardian + adversarial-reviewer) approve → commit + push
- Ponyo877 さんへの事後通知も Claude Web 介在も不要
- 次サブステップに直接進む
- 進捗追跡は `git log` + BENCH.md / DESIGN.md / docs/ で可能

### Ponyo877 さんが進捗確認したい時

任意のタイミングで `git log` や docs を見て確認可。Claude Code に
「現状報告」依頼も可 (この場合は Claude Web 介在なし、直接 conversation 内で完結)。

### 適用範囲

- M6.C-1 残作業 (C-1-4 は早期撤退ゲートで送信、C-1-5 は自走、C-1-6
  で全体報告として送信)
- M6.C-2 以降のサブステップも同方針
- Stage 1 中間評価は milestone 完了 (条件 1) で当然送信

## Subagent 判定の事後検証 (M6.C-2-4 retro)

scope-guardian や adversarial-reviewer の判定が、後の精査で覆る
可能性がある。特に paper や spec 等の primary source を参照する
場合は、subagent の web fetch metadata レベルではなく PDF / source
直接読了が必要。

判定例:
- M6.C-2-4 scope-guardian: paper §3.1+§4.3 を metadata で確認 →
  case α approve
- M6.C-2-4 PDF 直接読了: case α と paper 不一致発覚 → case δ 採用

教訓: spec/paper 依存の判定は subagent metadata でなく PDF 直接
読了が必須。

## 測定プロトコル

性能測定や副作用測定 (overhead, validation cost, ...) を行う場合:

1. **Paired run**: off / on の両条件を **同じ machine state** で
   交互測定。thermal drift を吸収。M6.A.7 で「validation-on を先に
   1 回、後で validation-off を比較」の order でやってしまい、
   thermal envelope shift で comparison が confounded した実例あり。
2. **Quiesced state**: trunk serve、別 cargo build、ブラウザ等の
   background process を止めた状態で測定。M6.A.7 の最初の
   validation-on run が trunk serve 起動と並走してしまい、256 grid
   の数値が信頼できなくなった実例あり。
3. **Multiple runs**: N=3 以上の median 採用。perf_regression が
   既にこの pattern を default で踏襲している。
4. **Honest framing**: noise band を超えない場合は overclaim せず
   「noise band 以下」と記録。"≤ 3%" のような数字が「観測値か推測か」
   を読者が判別できる形で書く (BENCH.md §10 / §11 を参考)。
5. **Cold-boot vs warm-state は別物**: 同 session で連続 perf 測定を
   何度も行うと M1 は thermal accumulation で 7-27% 遅くなる
   (BENCH.md §9 参照)。「cold-boot 数値」を測りたいなら独立 session、
   それが不可なら extrapolation と明示。

これらは M6.A.6 / A.7 / A.8 で順次表面化した知見で、M6.C の per-pass
optimization 測定で同じ罠を踏まないための予防則。