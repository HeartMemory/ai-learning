"""L2 · 注意力 —— 关卡 4（块 2/3）：bigram 基线

一条根：判据 ② 要求「val CE **严格低于**同语料 bigram 基线」。
    ⇒ 先得有一个**可复现、可核对**的基线数字，否则这句话没有宾语。
    ⇒ 而基线必须是**同类里最好的**：故意做弱（比如用未平滑的 MLE 撞出 inf、
      或者只报 k 取坏的那一档）等于把判据做绿，不是把模型做对。

bigram 的假设最朴素：下一个字符只由**前一个**字符决定
    P(c_t | c_0 … c_{t-1}) ≈ P(c_t | c_{t-1}) = C(c_{t-1}, c_t) / C(c_{t-1})
    参数只有 V×V = 65×65 = 4225 个格子（对照：Transformer 是 10^6 量级）
    ⇒ 它的 val CE 就是「完全不看上下文、只记相邻字符对」能达到的水平

⚠️ 三个口径，改必须写理由：
  ① **单位一律 nats/char**（自然对数）。模型侧的 `cross_entropy` 天然是 nats；
     L1 的 bigram 是 log2（bits/char，6.78）—— 直接比等于拿尺子比秤。
     两个单位都打印（bits = nats / ln2），但**判据只认 nats**。
  ② **必须平滑**：train 上 C=0 的字符对，在 val 上会真的出现（实测下面会报比例）。
     不平滑 ⇒ log(0) = -inf ⇒ 基线变成 inf，判据 ② 自动通过 —— 这是**假绿**。
     用 add-k：p = (C + k) / (C_prev + k·V)，k 取三档 (1 / 0.1 / 0.01)，**取最好那档**当基线。
  ③ **val 的第一个字符不算损失**：它的前文是 train 的最后一个字符，按位置切它是
     合法输入；但只差 1 / 111540 个样本，为了让「val 的配对数 == len(val) - 1」
     这条可核对，统一从 val[1:] 开始算。

判据（本块自定 6 项，A–F）：
    A 计数对账：`sum(C_bi) == len(train) - 1` · `sum(C_uni) == len(train)`
      · 每一行的和 == 该字符作为「前一个字符」出现的次数（不是 C_uni！）
    B 归一：每行 `exp(logp)` 之和 == 1（含 C=0 的空行，add-k 下它是均匀分布）
    C 对账官方：自写 `cross_entropy_nats` 与 `F.cross_entropy(logp[prev], cur)` 逐元素差 < 1e-6
      —— 与关 3 的 D 项同一方法论：**用自己的实现对账独立参考实现**，别自己跟自己对
    D 平滑必要性（两路取证）：未平滑时 val 上「未见字符对」个数 > 0 且 CE = inf；
      加 k 后 CE 有限
    E 三档 k 的 val CE 都报，最优档 == min，且**低于 unigram 基线**（bigram 必须比不看前文强）
    F train CE < val CE（严格），差值报出来

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 B1 [实现] count_bigrams(ids, vocab_size)      -> (C_uni (V,), C_bi (V,V))
 B2 [实现] bigram_logprob(c_uni, c_bi, k)      -> logp (V,V)，add-k 平滑
 B3 [实现] cross_entropy_nats(logp, ids)       -> float，按 ids 的相邻对平均
 B4 [已判据] judge_baseline()                  A–F 六项（已写好）

写完一项跑一项（B1 单独验，命令别写成一条超长行）：
    uv run python -c "import corpus as c; print(len(c.load_corpus()))"
    uv run python baseline.py                   # 判据 A–F（未实现的项会当场报 NG）
════════════════════════════════════════════════════════════════════
"""

import math

import torch
import torch.nn.functional as F

from corpus import build_vocab, encode, load_corpus, split_ids

SMOOTH_KS: tuple[float, ...] = (1.0, 0.1, 0.01)
"""add-k 的三档。取最优档当基线（口径 ②）：基线越强，判据 ② 越可信。"""

LN2: float = math.log(2.0)
"""nats <-> bits 的换算常数：bits = nats / ln2。"""

CE_TOL: float = 1e-6
"""判据 C 的容差：自写 CE 与官方 `F.cross_entropy` 的差。float64 下实际在 1e-15 量级，
留 1e-6 是因为官方实现在 float64 上也走 log_softmax，末位有微小差异。"""

NORM_TOL: float = 1e-12
"""判据 B 的容差：每行 exp(logp) 之和与 1 的差（float64 精度下够用）。"""

PROBE_PAIRS: int = 50_000
"""判据 C 的对账样本数：`logp[train[:-1]]` 会物化 (N, 65) 的矩阵，
N 取全量 100 万 = 520 MB，没必要；前 5 万对足够验数值一致。"""


def count_bigrams(ids: torch.Tensor, vocab_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """B1 · 数一数：每个字符出现几次（C_uni），每个字符对出现几次（C_bi）。

    返回 `(c_uni (V,), c_bi (V, V))`，都是 int64。
    `c_bi[i, j]` = 「字符 i 后面紧跟字符 j」在 ids 里出现的次数。

    提示（三行）：
      ① `c_uni = torch.bincount(ids, minlength=vocab_size)`
         —— `bincount` 要求非负整数 1 维张量，返回长度 = max(值)+1，所以必须传 minlength
      ② `flat = ids[:-1] * vocab_size + ids[1:]`
         —— 把二维下标 (i, j) 压成一个整数 i*V + j，才能用 bincount 数
      ③ `c_bi = torch.bincount(flat, minlength=vocab_size * vocab_size)`
         再 `.reshape(vocab_size, vocab_size)` —— 压平数出来的结果按 i*V+j 排序，正好是二维表

    ⚠️ 别用两层 for 循环（65×65 表面上看不大，但每次都要扫 100 万字符 ⇒ 6500 万次 Python 循环）。
    ⚠️ `ids[:-1] * V` 里 V 是 Python int ⇒ int64 不溢出；但若哪天换成 float，
       这条式子会把下标压成塌缩的浮点数，bincount 直接报错。
    """
    c_uni = torch.bincount(ids, minlength=vocab_size)
    flat = ids[:-1] * vocab_size + ids[1:]
    c_bi = torch.bincount(flat, minlength=vocab_size * vocab_size)
    return c_uni, c_bi.reshape(vocab_size, vocab_size)


def bigram_logprob(c_uni: torch.Tensor, c_bi: torch.Tensor, k: float) -> torch.Tensor:
    """B2 · add-k 平滑后的 log 概率表：logp[i, j] = log((C_bi[i,j] + k) / (C_prev[i] + k·V))。

    注意分母用的是 **C_prev[i]** = 字符 i 作为「前一个字符」出现的次数，
    也就是 `c_bi[i].sum()` —— **不是** `c_uni[i]`。
    差在哪：`c_uni` 把 i 出现在**末尾**的那一次也算进去，而它没有「下一个字符」。
    全量 100 万字符下只差 1 次，但它会让判据 A 的行和对账对不上。

    提示（三行）：
      ① 先转 float64：`cb = c_bi.to(torch.float64)`
         —— 更正（2026-11-25 实测）：torch 的 `/` 在整数张量上**不截断**，会提升成
            float32（`tensor([1,3])/tensor([2,2])` = `[0.5, 1.5]`，截断的是 `//`）。
            要 float64 的真正理由是**精度**：float32 只有 7 位有效数字，实测行和偏差
            **1.19e-07**，撞不过判据 B 的 1e-12 容差。
      ② `p = (cb + k) / (cb.sum(dim=1, keepdim=True) + k * cb.shape[0])`
      ③ `return torch.log(p)` —— 别用 log(p) 之外的组合（log(a/b) 与 log a - log b 差末位）

    自检：`logp.exp().sum(dim=1)` 每行都是 1（连 C=0 的空行也是，均匀 1/V）。
    ⚠️ k=0 时这里会出现 log(0) = -inf，那是**故意的**（判据 D 要用它证明必须平滑）。
    """
    C_prev = torch.zeros_like(c_uni)
    for i in range(c_bi.shape[0]):
        C_prev[i] = c_bi[i].sum()
    
    cb = c_bi.to(torch.float64)
    p = (cb + k) / (C_prev[:, None] + k * cb.shape[0])
    return torch.log(p)

def cross_entropy_nats(logp: torch.Tensor, ids: torch.Tensor) -> float:
    """B3 · 在这段 ids 上算平均交叉熵（nats/字符）。

    定义：对每个相邻对 (ids[t], ids[t+1])，取 `-logp[ids[t], ids[t+1]]`，再对配对数取平均。
    配对数 = `len(ids) - 1`。

    提示（两行）：
      ① `picked = logp[ids[:-1], ids[1:]]` —— 高级索引一次取出所有对，返回 (N-1,)
      ② `return -picked.mean().item()` —— **负号别丢**（logp 是负的，取负才是正损失）

    ⚠️ 别写成 `logp[ids[:-1]][range(...), ids[1:]]`：那会先物化一个 (N-1, 65) 的矩阵，
       100 万行 = 520 MB。一次高级索引是 O(N)，不会展开整张表。

    自检：与 `F.cross_entropy(logp[ids[:-1]], ids[1:])` 差 < 1e-6（判据 C 会查）。
    """
    picked = logp[ids[:-1], ids[1:]]
    return -picked.mean().item()


def unigram_ce_nats(c_uni: torch.Tensor, ids: torch.Tensor, k: float = 1.0) -> float:
    """参照基线（AI 写）：只看单字符频率 + add-k 平滑，用来证明「bigram 比不看前文强」。

    这是判据 E 的对照物，不属于你的 TODO。
    """
    v = c_uni.shape[0]
    p = (c_uni.to(torch.float64) + k) / (c_uni.sum() + k * v)
    return float(-torch.log(p)[ids[1:]].mean().item())


def judge_baseline() -> bool:
    """B4 · 判据 A–F（全绿才 True）。"""
    checks: list[bool] = []

    text = load_corpus()
    itos, stoi = build_vocab(text)
    ids = encode(text, stoi)
    train, val = split_ids(ids)
    v = len(itos)

    try:
        c_uni, c_bi = count_bigrams(train, v)
    except NotImplementedError:
        print("A 计数：count_bigrams 未实现  NG")
        return False

    # A 计数对账：总量 + 行和（行和 = 该字符作为前一个字符的次数）
    prev = torch.bincount(train[:-1], minlength=v)
    ok_sum_bi = int(c_bi.sum()) == len(train) - 1
    ok_sum_uni = int(c_uni.sum()) == len(train)
    ok_row = bool((c_bi.sum(dim=1) == prev).all())
    checks += [ok_sum_bi, ok_sum_uni, ok_row]
    tag_a = "OK" if (ok_sum_bi and ok_sum_uni and ok_row) else "NG"
    print(
        f"A 计数 sum(C_bi) {int(c_bi.sum())} == {len(train) - 1} · "
        f"sum(C_uni) {int(c_uni.sum())} == {len(train)} · 行和对账 {ok_row}  {tag_a}"
    )

    try:
        logp = bigram_logprob(c_uni, c_bi, 0.1)
    except NotImplementedError:
        print("B 归一：bigram_logprob 未实现  NG")
        return False

    # B 归一：每行 exp(logp) 之和 == 1
    row_sums = logp.exp().sum(dim=1)
    dev = float((row_sums - 1.0).abs().max())
    ok_norm = dev < NORM_TOL
    checks.append(ok_norm)
    empty_rows = int((c_bi.sum(dim=1) == 0).sum())
    tag_b = "OK" if ok_norm else "NG"
    print(f"B 行归一 最大偏差 {dev:.3e}（空行 {empty_rows} 个，应为均匀 1/{v}）  {tag_b}")

    try:
        mine = cross_entropy_nats(logp, train[: PROBE_PAIRS + 1])
    except NotImplementedError:
        print("C 对账：cross_entropy_nats 未实现  NG")
        return False

    # C 对账官方
    sub = train[: PROBE_PAIRS + 1]
    ref = float(F.cross_entropy(logp[sub[:-1]], sub[1:]).item())
    d_ce = abs(mine - ref)
    ok_ce = d_ce < CE_TOL
    checks.append(ok_ce)
    tag_c = "OK" if ok_ce else "NG"
    print(f"C 自写 CE {mine:.12f} vs 官方 {ref:.12f} 差 {d_ce:.3e}  {tag_c}")

    # D 平滑必要性（两路取证）：未平滑时 val 上真实的未见对 + CE = inf
    unseen = int((c_bi[val[:-1], val[1:]] == 0).sum())
    ratio = 100.0 * unseen / (len(val) - 1)
    mle = bigram_logprob(c_uni, c_bi, 0.0)
    mle_ce = cross_entropy_nats(mle, val)
    ok_unseen = unseen > 0
    ok_inf = math.isinf(mle_ce) or math.isnan(mle_ce)
    checks += [ok_unseen, ok_inf]
    tag_d = "OK" if (ok_unseen and ok_inf) else "NG"
    print(
        f"D 未平滑 val 未见对 {unseen} / {len(val) - 1} = {ratio:.3f}%"
        f"，MLE CE = {mle_ce}  {tag_d}"
    )

    # E 三档 k 全报，取最优；且必须低于 unigram 基线
    val_ces: dict[float, float] = {}
    for k in SMOOTH_KS:
        val_ces[k] = cross_entropy_nats(bigram_logprob(c_uni, c_bi, k), val)
    best_k = min(val_ces, key=lambda kk: val_ces[kk])
    best = val_ces[best_k]
    uni_ce = unigram_ce_nats(c_uni, val)
    ok_min = all(best <= ce + NORM_TOL for ce in val_ces.values())
    ok_beat_uni = best < uni_ce
    checks += [ok_min, ok_beat_uni]
    tag_e = "OK" if (ok_min and ok_beat_uni) else "NG"
    print(
        "E val CE（nats/char）"
        + " · ".join(f"k={kk:g}: {ce:.4f}" for kk, ce in val_ces.items())
        + f" | 最优 k={best_k:g} → {best:.4f} | unigram {uni_ce:.4f}  {tag_e}"
    )

    # F train vs val
    train_ce = cross_entropy_nats(bigram_logprob(c_uni, c_bi, best_k), train)
    ok_gap = train_ce < best
    checks.append(ok_gap)
    tag_f = "OK" if ok_gap else "NG"
    print(
        f"F train {train_ce:.4f} < val {best:.4f}（差 {best - train_ce:.4f}）"
        f"  | bits/char: train {train_ce / LN2:.4f} · val {best / LN2:.4f}  {tag_f}"
    )

    print(f"   → 判据 ② 的基线数字取 **{best:.4f} nats/char**（k={best_k:g}）")
    return all(checks)


def main() -> None:
    """测试入口（纯胶水）。"""
    results = [judge_baseline()]
    print(f"判据 {sum(results)}/1 绿")


if __name__ == "__main__":
    main()
