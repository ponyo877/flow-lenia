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

## レビュー手順
1. サブステップ実装完了時、adversarial-reviewer subagent を起動
2. 議論ポイントは scope-guardian にも諮る
3. 両 subagent から approval が出たら commit + push
4. Ponyo877 さんへの確認は方針変更が必要な時のみ