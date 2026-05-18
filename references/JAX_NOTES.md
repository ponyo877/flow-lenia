# JAX 公式実装 (`erwanplantec/FlowLenia`) 精読ノート

**目的**: M2 着手前の事前確認 (DESIGN.md Q1A, Q3, Q3b)、および追加で発見した論文との食い違いを記録する。
**参照リポジトリ**: `references/FlowLenia-jax/` (commit `dce428c6b0c5079a06e5606fb7b5ac1fe1323bc5`, "import alias + default configs", 2024-02-08)
**精読ファイル**:
- `flowlenia/flowlenia.py` (Eq. 7 なし版、`h` で重み)
- `flowlenia/flowlenia_params.py` (Eq. 7 あり版、`P` でセル局在重み)
- `flowlenia/reintegration_tracking.py` (Eq. 6 + Eq. 8 の混合則)
- `flowlenia/utils.py` (Sobel, growth, kernel precompute)

論文式番号は **Plantec et al. 2025, Artificial Life journal, arXiv:2506.08569v1** に従う。

---

## 1. **Q1A 確定**: Chebyshev 距離 = **11×11 (`-5..=5`)**

### 該当コード (`reintegration_tracking.py:39-44`)

```python
dd = self.dd  # Config default = 5
for dx in range(-dd, dd+1):       # range(-5, 6) = {-5,-4,...,5}
    for dy in range(-dd, dd+1):
        dxs.append(dx)
        dys.append(dy)
```

### 結論

- `dd=5` の場合、近傍は **`(-5..=5) × (-5..=5)` = 11 × 11 = 121 セル**
- 論文本文の "less than 5" は **literal な解釈では誤り**。実装は **"less than or equal to 5" = 9 近傍 (Moore distance ≤ 5)** を意味する
- **DESIGN.md の仮置き「9×9」は誤り。11×11 に改める**

### 物理的な裏付け (`reintegration_tracking.py:62`)

```python
ma = self.dd - self.sigma  # upper bound of the flow magnitude
mu = pos[..., None] + jnp.clip(self.dt * F, -ma, ma)
```

- 1 ステップで動ける最大距離 `|dt·F|` を `dd - sigma` に clip
- `dd=5, sigma=0.65` → `ma = 4.35`
- 分布 `D` の幅は `2·sigma = 1.3`、つまり中心から `±sigma = ±0.65` 広がる
- 最遠到達点 = `|dt·F| + sigma = 4.35 + 0.65 = 5.0`
- つまり **`dd=5` の近傍は「物質が 1 ステップで届きうる範囲ピッタリ」** を覆っており、`dd` 個分の余裕は物理的必然
- **`dd` をパラメータとして UI で公開する余地あり** (DESIGN.md に追加検討)

### 実装への影響

- WGSL の reintegrate / params_update パスのループは **`dy, dx ∈ {-5, ..., 5}` の 121 セル**
- DESIGN.md §3.3, §4.1, §4.1.3, §4.5 の「9×9 / 81 セル」を 11×11 / 121 セルに修正

---

## 2. **Q3 確定**: `β_A` ── **論文と JAX 実装で式そのものが異なる**

### 該当コード

`flowlenia.py:98`:
```python
alpha = jnp.clip((A[:, :, None, :] / self.cfg.C) ** 2, .0, 1.)
```

`flowlenia_params.py:101`:
```python
alpha = jnp.clip((A[:, :, None, :] / 2) ** 2, .0, 1.)
```

### 論文 Eq. 5 (2025 版)

```
α^t(x) = [(A_Σ^t(x) / β_A) ^ n] _ [0,1]
A_Σ^t(x) = Σ_i A_i^t(x)
```

### 比較表

| 項目 | 論文 Eq. 5 | JAX `flowlenia.py` | JAX `flowlenia_params.py` |
|---|---|---|---|
| α は per-channel か | **全チャンネル共通** (A_Σ ベース) | **per-channel** (A_c ベース) | per-channel (A_c ベース) |
| 臨界質量 β_A | 文字定義のみ | `= C` (チャンネル数) | `= 2` (リテラル) |
| 指数 n | 任意パラメータ | `2` 固定 | `2` 固定 |

### 結論

- **β_A の "デフォルト値" を論文から確定することは不可能** (論文に数値がない)
- **JAX 実装は論文の Eq. 5 そのものを実装していない**:
  - per-channel に分解 (A_Σ → A_c)
  - β_A を「2」または「チャンネル数 C」に固定 (パラメータではない)
  - n も 2 に固定
- これは **教育的価値の観点で見過ごせない非互換**

### 設計判断 (要承認)

DESIGN.md の「2025 年版論文の Equation 5, 6, 7, 8 を数学的に厳密に再現する」を満たすため、**実装は論文 Eq. 5 に準拠**する:
- `α(x) = clip((A_Σ(x) / β_A)^n, 0, 1)` (A_Σ = 全 C 和、α は全チャンネル共通)
- `β_A` を UI スライダ化、**デフォルト値 `2.0`** (JAX `flowlenia_params.py` 慣例値に揃える)
- `n` を UI スライダ化、デフォルト `2.0`

加えて、**「JAX 互換モード」チェックボックス**を UI に用意し、ON のときは:
- `α(x, c) = clip((A_c(x) / β_A)^2, 0, 1)` (per-channel)
- `β_A` を `C` (vanilla) または `2` (params 埋め込み版) にハードコード

これにより、論文厳密モードと JAX 公式互換モードを切替えてベンチマーク比較できる。

→ **Q3 を「論文準拠 + JAX 互換切替」で確定したい。承認を得る (DESIGN.md Q3 更新)**

---

## 3. **Q3b 確定**: Sobel = **正規化なし** (生の係数)

### 該当コード (`utils.py:16-37`)

```python
kx = jnp.array([
    [1., 0., -1.],
    [2., 0., -2.],
    [1., 0., -1.]
])
ky = jnp.transpose(kx)

def sobel_x(A):
    return jnp.dstack([jsp.signal.convolve2d(A[:, :, c], kx, mode='same')
                       for c in range(A.shape[-1])])
```

### 結論

- **正規化係数なし**。`/8` や `/4` を掛けない
- カーネル符号: 標準的な Sobel `[[-1,0,1],[-2,0,2],[-1,0,1]]` を **横方向に反転** した形 (`[[1,0,-1],...]`)
  - `jsp.signal.convolve2d` は数学的畳み込み (カーネルを 180° 反転して相関) なので、実効的には **標準 Sobel 相関** と同じ向きの出力
- 境界処理は `mode='same'` → **ゼロパディング**
  - **トロイダル境界の Eq. 5 と矛盾**する可能性がある。詳細は §6 で再論

### 実装への影響

- WGSL `gradient.wgsl` での Sobel カーネル係数は **`[[1,0,-1],[2,0,-2],[1,0,-1]]` をそのまま使用**
- DESIGN.md §4.4 「Sobel `/8`」を **正規化なし** に修正
- 境界処理は **トロイダル周期境界** に統一する (詳細 §6)

---

## 4. **追加発見**: 畳み込みは FFT ベース

### 該当コード (`flowlenia.py:82-86`)

```python
fA = jnp.fft.fft2(A, axes=(0,1))           # (x,y,c)
fAk = fA[:, :, self.cfg.c0]                # (x,y,k)
U = jnp.real(jnp.fft.ifft2(state.fK * fAk, axes=(0,1)))  # (x,y,k)
```

### `utils.py:41-59` kernel precompute

```python
def get_kernels_fft(X, Y, k, R, r, a, w, b):
    mid = X//2
    Ds = [np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / ((R+15) * r[k]) for k in range(k)]
    K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, a[k], w[k], b[k]) for k, D in zip(range(k), Ds)])
    nK = K / jnp.sum(K, axis=(0,1), keepdims=True)
    fK = jnp.fft.fft2(jnp.fft.fftshift(nK, axes=(0,1)), axes=(0,1))
    return fK
```

### 結論

- JAX 実装は **すべてのカーネル畳み込みを FFT で実行**
- カーネルは **CPU で 1 回 precompute → FFT 化 → GPU に保持**
- `K_i` の半径は **グリッド全体** に展開される (sigmoid マスク `sigmoid(-(D-1)*10)` で D > 1 を急減衰させて「実効半径」を作る)

### 実装への影響

DESIGN.md §1.7, §4.1 で「直接畳み込みから開始」としていたが、これは **JAX 実装と数値的に一致させたい場合は不利**:
- 直接畳み込み (空間領域、有効半径で枝刈り) と FFT 畳み込み (周波数領域、全体) は周期境界条件の扱いが微妙に異なる
- 質量保存テストは両者でほぼ同じ結果になるはずだが、参照実装 vs GPU の **ビット精度比較**は無理

### 設計判断 (要承認)

- **CPU 参照実装 (M1)** は **FFT を使う** (JAX と完全一致狙い、`rustfft` または `realfft` クレートを採用)
- **GPU 実装 (M2)** は **直接畳み込みから開始**、ただし参照実装との比較は「相対誤差 < 1e-3」許容に緩和
  - 直接畳み込みでも質量保存は厳密に保たれるはず (Eq. 6 部分が保存則の本質なので)
- **M6 性能チューニング段階**で GPU 側も FFT 化を検討 (Stockham カーネル × 2 軸)

→ DESIGN.md §1.6, §1.7, §5.2 を更新する。**承認後**に着手。

---

## 5. **追加発見**: デフォルト境界条件は `"wall"`、論文/直感はトロイダル

### 該当コード

`flowlenia.py:24`:
```python
class Config(NamedTuple):
    ...
    border: str="wall"
```

`reintegration_tracking.py:64-65`:
```python
if self.border == "wall":
    mu = jnp.clip(mu, self.sigma, self.SX-self.sigma)
```

### 結論

- **JAX デフォルトは `wall`**: 壁境界 (中心位置 mu を sigma..SX-sigma に clip)
- `border='torus'` も実装あり: 9 通りの位置で最小距離を取る (周期境界相当)
- Sobel 部分の `convolve2d(mode='same')` は **ゼロパディング**、つまり gradient 計算は **トロイダルではない**
- → JAX 実装は **gradient: zero-pad、reintegration: wall or torus** という混在
- 論文には境界条件の明記なし

### 設計判断

DESIGN.md Q6 で「トロイダルのみ」と仮置きしていたが、JAX 実装と一致させたい場合は wall を選ぶ必要がある。

**提案**: 設計範囲を以下のように拡張:
- **UI に border 切替: `{torus, wall}` を選べる**
- gradient (Sobel) も **境界処理を統一**:
  - `torus` モード時: Sobel もトロイダル境界
  - `wall` モード時: Sobel もゼロパディング
- デフォルトは **`torus`** (論文の「マスは保存される」直感に最も合う、wall は境界 clip で質量が失われる可能性がある)

→ DESIGN.md Q6 を「torus デフォルト + wall 切替可」に更新したい。**承認を得る**。

---

## 6. **追加発見**: パラメータ範囲が論文 Table 1 と微妙に違う

### JAX `flowlenia.py:55-64`

| 変数 | 論文 Table 1 | JAX 実装 (`uniform`) |
|---|---|---|
| `R` | `[2, 25]` | `[2.0, 25.0]` ✅ |
| `r` | `[0.2, 1.0]` | `[0.20, 1.00]` ✅ |
| `m` (= μ in 論文) | `[0.05, 0.5]` | `[0.050, 0.50]` ✅ |
| `s` (= σ in 論文) | `[0.001, 0.2]` | **`[0.001, 0.18]`** ⚠️ 微差 |
| `h` | `[0, 1]` | **`[0.010, 1.00]`** ⚠️ 0 を除外 |
| `a` | `[0, 1]^3` | `[0.000, 1.00]` ✅ |
| `b` | `[0, 1]^3` | **`[0.001, 1.00]^3`** ⚠️ 0 を除外 |
| `w` | `[0.01, 0.5]^3` | `[0.010, 0.50]` ✅ |

### 結論

`s`, `h`, `b` の最小値が JAX では微妙に大きい (0 を避けて 0.001 や 0.01 から)。
- 数値安定性のため (0 だとカーネル正規化や対数で問題)
- 論文 Table 1 はおそらく「概略の範囲」、実装は「実用上の範囲」

### 設計判断

DESIGN.md §6.1 のサンプリング範囲を **JAX 実装の値に合わせる**:
- `s ∈ [0.001, 0.18]`
- `h ∈ [0.010, 1.00]`
- `b ∈ [0.001, 1.00]^3`

→ DESIGN.md §6.1 を更新 (Q なし、自明な修正)。

---

## 7. **追加発見**: カーネルのスケーリング `(R+15) * r[k]` (論文 Eq. 1 と異なる)

### 該当コード (`utils.py:51-54`)

```python
mid = X // 2
Ds = [np.linalg.norm(np.mgrid[-mid:mid, -mid:mid], axis=0) / ((R + 15) * r[k]) for k in range(k)]
K = jnp.dstack([sigmoid(-(D-1)*10) * ker_f(D, a[k], w[k], b[k]) for k, D in zip(range(k), Ds)])
```

### 論文 Eq. 1

```
K_i(x) = Σ_j b_{i,j} · exp( -((r/(r_i·R) - a_{i,j})^2) / (2 w_{i,j}^2) )
```

ここで `r` はセル中心からの距離 (グリッド単位)、`r_i·R` は **カーネル i の実効半径** (グリッド単位)。

### JAX 実装での違い

- スケーリング分母が **`(R + 15) * r[k]`**、論文より大きい
- なぜ `+15`?
  - 推測 1: `R ∈ [2, 25]` で最小 `R=2` のとき、`R * r ∈ [0.4, 2.0]` → カーネルが 1-2 ピクセル幅にしかならず numerical artifact が出る。`+15` で「最小でも 17 ピクセル幅」を保証する経験則
  - 推測 2: グリッド境界アーティファクトを避けるためのマージン
- **論文には記載なし**。明らかな実装独自の修正

### sigmoid マスク `sigmoid(-(D-1)*10)`

- `D` は normalized distance (1 でカーネル境界)
- `sigmoid(-(D-1)*10)` は `D < 1` で ≈ 1、`D > 1` で ≈ 0 (急峻な切り替え)
- つまり **`D = 1` (= `(R+15)*r[k]` ピクセル) で実効半径を切る**
- 論文 Eq. 1 にはこのマスクの記載はない

### `ker_f` (`utils.py:9`)

```python
ker_f = lambda x, a, w, b : (b * jnp.exp( - (x[..., None] - a)**2 / w)).sum(-1)
```

- 論文 Eq. 1 は `exp(-(x - a)^2 / (2 w^2))`
- JAX 実装は `exp(-(x - a)^2 / w)`
- **`w` の解釈が違う**: 論文の `2 w^2` に対し、JAX は `w` 単体
- これは「実装上の `w` は論文の `2 w^2` に相当」ということ
- パラメータ範囲 `w ∈ [0.01, 0.5]` を考えると、論文の `w^2 ∈ [1e-4, 0.25]` × 2 = `[2e-4, 0.5]` で重なる
- **同じ数値範囲を保つには「JAX 実装上の w」をそのまま使うのが自然**

### 設計判断

論文と JAX 実装でカーネル定義が違う。教育目的で「論文の Eq. 1 を厳密に実装」しても、JAX で見つかった魅力的パラメータをコピペしても動かない (ker shape が違うため)。

**提案**:
- **JAX 実装の式を採用** (`(R+15)*r[k]` スケーリング + `sigmoid` マスク + `w` 解釈)
- WGSL コメントで **「論文 Eq. 1 と JAX 実装の対応関係」** を明示
  - `w_jax ↔ 2 * w_paper^2`
  - `(R+15) ↔ R` の差分は経験則として記載
  - `sigmoid(-(D-1)*10)` マスクの存在を明示

これにより、JAX のパラメータセットをコピペして同じ creature が観察できる。これは教育的価値・回帰テストの両面で重要。

→ Q3c (新) として確認したい。

---

## 8. **追加発見**: Mutation patch サイズが論文 (10×10) と JAX (20×20) で違う

### 該当コード (`flowlenia_params.py:153-162`)

```python
def beam_mutation(state: State, key: jax.Array, sz: int=20, p: float=0.01):
    kmut, kloc, kp = jr.split(key, 3)
    P = state.P
    k = P.shape[-1]
    mut = jnp.ones((sz,sz,k)) * jr.normal(kmut, (1,1,k))  # patch 全体で同じ N(0,1) サンプル
    loc = jr.randint(kloc, (3,), minval=0, maxval=P.shape[0]-sz).at[-1].set(0)
    dP = jax.lax.dynamic_update_slice(jnp.zeros_like(P), mut, loc)
    m = (jr.uniform(kp, ()) < p).astype(float)
    P = P + dP*m
    return state._replace(P=P)
```

### 論文 §4.3.1

> we introduce mutations in the form of square "beams" affecting a random **10 × 10** patch in the grid

### 結論

- 論文: **10×10**
- JAX: **デフォルト 20×20**、ただし可変 (`sz=20` パラメータ)
- 実装としては可変なので、UI で `mutation_patch_size ∈ {10, 20, ...}` を選べる形が望ましい

### 設計判断

DESIGN.md §4.1 / §7 の mutation 仕様を **可変サイズ (default 20、論文準拠なら 10)** に更新。UI スライダ追加。

---

## 9. **追加発見**: ker_f の `b` 重みは「リング」を選択的に有効化する

(これは §7 とも関連)

`b ∈ [0.001, 1.00]^3` (3 リングそれぞれ)。論文では `b ∈ [0, 1]^3`。

JAX 実装ではガウシアン和:
```python
ker_f(x, a, w, b) = Σ_j b_j * exp(-(x - a_j)^2 / w_j)
```

`b_j ≈ 0` のリングは事実上「消える」。JAX が下限を 0.001 にしている理由は数値安定性 (`K / sum(K)` の分母が 0 にならないように)。

### 設計判断

DESIGN.md §6.1 で `b ∈ [0.001, 1.00]^3` を採用 (§6 のサマリで既に修正済み)。追加対応不要。

---

## 10. **追加発見**: Reintegration tracking の `nA = step(A, mu, dxs, dys).sum(0)` は **送信側ループ** を `vmap` 化して **受信側和を取る** 設計

### コード再掲 (`reintegration_tracking.py:31-69`)

```python
@partial(jax.vmap, in_axes=(None, None, 0, 0))
def step(A, mu, dx, dy):
    Ar = jnp.roll(A, (dx, dy), axis=(0, 1))      # A をシフト → 「(x+dx, y+dy) からの寄与を (x, y) で受信」と等価
    mur = jnp.roll(mu, (dx, dy), axis=(0, 1))
    ...
    sz = .5 - dpmu + self.sigma
    area = jnp.prod(jnp.clip(sz, 0, min(1, 2*self.sigma)), axis=2) / (4 * self.sigma**2)
    nA = Ar * area
    return nA

nA = step(A, mu, dxs, dys).sum(0)  # 121 個のシフトを和
```

### 結論

JAX 実装は概念的に「**受信側で 11×11 の範囲を集計**」と等価。各 (dx, dy) について `jnp.roll` でグリッド全体をシフトし、各受信セル `(x, y)` で重なり面積を計算 → 121 シフトの結果を sum。

これは **DESIGN.md §4.1.3 の「受信側ループ」の方針と一致**。WGSL では `jnp.roll` の代わりに `(x + dx) mod W` で読みに行く形にすればよい。

### `area` 計算の `min(1, 2*self.sigma)` の意味

- `sz = .5 - dpmu + sigma` の各成分
- `clip(sz, 0, min(1, 2*sigma))` の上限 `min(1, 2*sigma)`:
  - `sigma <= 0.5` のとき: 上限 = `2*sigma` (分布の幅自体)
  - `sigma > 0.5` のとき: 上限 = `1` (セル幅自体)
- 「重なり区間の長さは、分布全幅 (`2*sigma`) も、セル幅 (`1`) も超えない」という物理的制約
- 正規化 `/ (4 * sigma^2)` で「重なり面積を分布全面積で割る」 = 確率

→ DESIGN.md §4.3 の overlap 計算式に **上限 clip `min(1, 2*sigma)`** を追加する必要あり。

---

## 11. **追加発見**: stochastic 混合則は `log(nA.sum(axis=-1))` を logit にする

### 該当箇所 (JAX 引用用アンカー: **NOTE-11-A** / **NOTE-11-B**)

ソース: `references/FlowLenia-jax/flowlenia/reintegration_tracking.py:125-133` (commit `dce428c`)

```python
elif self.mix == "stoch":
    categorical = jax.random.categorical(
        jax.random.PRNGKey(42),       # ← NOTE-11-A: 固定シードバグ
        jnp.log(nA.sum(axis=-1, keepdims=True)),  # ← NOTE-11-B: logit = log(A_Σ)
        axis=0)
    mask = jax.nn.one_hot(categorical, num_classes=(2*self.dd+1)**2, axis=-1)
    mask = jnp.transpose(mask, (3,0,1,2))
    nH = jnp.sum(nH * mask, axis=0)
    nA = jnp.sum(nA, axis=0)
```

WGSL コメントからの引用フォーマット例:
```wgsl
// JAX 互換モードの logit 計算は categorical(log(A_Σ·I)) で実装される
// (JAX_NOTES.md §11、NOTE-11-B、reintegration_tracking.py:127 参照)。
// 数学的には softmax(log(x)) = x / Σx なので、これは「正規化された A_Σ·I を確率に」と等価。
//
// なお JAX 実装には PRNGKey(42) を毎ステップ再生成する固定シードバグ
// (JAX_NOTES.md §11、NOTE-11-A、reintegration_tracking.py:127) がある。
// 本実装では per-cell xoshiro128++ state を用いて修正する。
```

### 結論

- **`logit = log(A_Σ)`** → softmax は `exp(log(A_Σ)) = A_Σ` → 確率は `A_Σ(x') / Σ A_Σ(x'')` (※`I` で重み付けは内包済み、`nA = A_r * area = A·I`)
- これは論文 Eq. 8 `P[P^{t+dt}(x) = P^t(x')] ∝ exp(A^t(x') * I(x', x))` と **数学的に異なる**:
  - 論文: softmax over **`A · I`** (積を logit)
  - JAX: categorical over **`A · I` 自体** (積を確率) ← **softmax の指数 `exp` を省略している**!
- 数値的にはどちらも「質量×到達確率」が大きいほど採用されやすい点で類似だが、**重みの強度が違う**
- **論文の式と JAX 実装が乖離している (3 番目の食い違い)**

### ⚠️ 固定シード問題

`jax.random.PRNGKey(42)` が毎ステップ同じ! これは JAX のステートレス RNG をうっかり書いた実装ミスの可能性が高い (まじめな再現性研究なら問題)。設計では **毎ステップ独立な PRNG state** を必須にする。

### 設計判断

- **論文 Eq. 8 を厳密に実装** (`softmax(A_Σ(x') · I(x', x))`)
- JAX 互換モードでは **「正規化された A·I を確率として使う」**
- WGSL 側で per-cell に PRNG state (`u32`) を持ち、xoshiro128++ など軽量実装
- → Q3d (新) として確認したい

---

## 12. まとめ: 確定事項と要承認事項

### ✅ 即時確定 (DESIGN.md を機械的に修正してよい)

1. **Q1A**: Chebyshev = **11×11** (`-5..=5`、121 セル) — DESIGN.md §0, §3.3, §4.1, §4.1.3, §4.5
2. **Q3b**: Sobel = **正規化なし** (生係数)、`[[1,0,-1],[2,0,-2],[1,0,-1]]` を直接使う — DESIGN.md §4.4
3. **パラメータ範囲微差**: `s ∈ [0.001, 0.18]`, `h ∈ [0.010, 1.00]`, `b ∈ [0.001, 1.00]^3` — DESIGN.md §6.1
4. **`ma = dd - sigma` でフロー振幅 clip** — DESIGN.md §4.3 に追加
5. **`area` 上限 clip `min(1, 2σ)`** — DESIGN.md §4.3 に追加

### ❓ 要承認 (設計判断が必要)

6. **Q3 (β_A)**: 「**論文 Eq. 5 (A_Σ ベース) を実装 + UI スライダ化 + JAX 互換モード切替**」で確定したい
7. **Q3c (新)**: カーネル定義 — **JAX 実装式 (`(R+15)*r[k]` + sigmoid マスク + `w` 解釈) を採用、論文 Eq. 1 との対応関係をコメントで明記**
8. **Q3d (新)**: Eq. 8 softmax — **論文式 `softmax(A_Σ·I)` を採用、JAX 互換モードでは `normalize(A_Σ·I)` を選択**
9. **Q6 (境界条件)**: **デフォルト `torus`、UI で `wall` 切替可** に拡張
10. **Q7 (FFT)**: **CPU 参照実装は FFT 化** (JAX と数値一致狙い)、GPU は直接畳み込みから開始
11. **Mutation patch size**: **UI スライダで 10 or 20 or ...**、デフォルト 20 (JAX 慣例)

### 🚫 既存設計から削るもの

- DESIGN.md §0 の literal「9×9」解釈 → 11×11 に修正
- DESIGN.md §4.4 「`/8` 正規化」 → 正規化なし
- DESIGN.md §1.6 の「FFT 不使用」 → CPU 参照実装は FFT 採用

---

## 13. 残課題

- **simutils.py / examples 未読**: 初期パターンの設定方法 (`patch_size`, `creatures_per_grid`)、`vizutils.py` のカラーマッピングなど、UI 仕様の参考になる箇所
- **clone した repo の commit hash を固定** (`git rev-parse HEAD` の結果を本書に追記)
- **論文 Figure 4 のパラメータ**: examples ディレクトリにあるか確認 → 回帰 fixture の元ネタにする

---

## 14. Empirical Lyapunov exponents (M2.8 verify)

Measured during the M2.8 verification of CPU-vs-GPU divergence
behaviour. Setup: 32×32 torus grid, `seed = 42`, `num_kernels = 10`,
default kernel-param sampling (`sample_random`). Two trajectories
diverged by an `ε = 1e-6` uniform perturbation of the initial
activation; tracked `max_abs` over 100 CPU-only steps.

| `C` | Empirical λ (per step) | Regime              |
|-----|------------------------|---------------------|
|  1  | **≈ 0.03**             | weakly chaotic      |
|  3  | **≈ 0.17**             | strongly chaotic    |

Test that produced these numbers:
`crates/flow-lenia-gpu/tests/diagnose_divergence.rs ::
estimate_lyapunov_exponent_c1_vs_c3` (run with
`cargo test --release ... -- --ignored --nocapture`).

Practical consequences for the project:
- **Field-comparison testing across implementations** (CPU↔GPU, JAX↔Rust,
  Rust↔Rust under different toolchains) is **only meaningful** for
  step counts much less than `1 / λ ≈ 6` steps at `C = 3`, or roughly
  30 steps at `C = 1`. Anything past that and the trajectories will
  *correctly* sit on different points of the same dynamics.
- **Mass conservation** is the right invariant for long-trajectory
  testing — it's a physical quantity, not a trajectory property, and
  stays at the f32 accumulation floor (`~5e-6` after 100 steps).
- **M5 evolutionary search horizon** should be on the order of
  `2/λ ≈ 12` steps for `C = 3` and `60` steps for `C = 1` — long enough
  for a creature's behaviour to differentiate, short enough that
  the f32 noise floor is still well below the dynamical-feature scale.
- **Lenia paper** (Chan 2019, 2020) and **Flow-Lenia paper** (Plantec
  2025) do not quote these numbers directly; this is repository-local
  measurement.
