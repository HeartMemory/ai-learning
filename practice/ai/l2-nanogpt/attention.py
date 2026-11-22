"""L2 · 注意力 —— 关卡 1：单头算术（手算 vs torch）

目的（一条根）：注意力只做一件事 —— 让每个 token 从**所有 token 里挑着听**。
    · 挑什么：Q（这个 token 想问什么）× K（每个 token 能提供什么）→ 分数
    · 听多少：分数过 softmax 变成**权重**，每行和 = 1（这是一份「注意力预算」）
    · 听回来：权重加权求和 V（每个 token 的「内容」）

判据 ① 只证明一句话：**这条链手算得出来，而且和 torch 一字不差**。
    4 token × d=2，Q/K/V 写死（见常量），同一件事用两种算法各算一遍：
      A 手算  ：纯 Python 双层循环（禁用 torch）—— 纸上算法逐字翻译成循环
      B torch ：张量一次算完
    两个 4×4 权重矩阵逐元素差 < 1e-12，输出（4×2）同理，且权重每行和 = 1（±1e-12）。

⚠️ dtype 必须 float64：float32 的机器 eps ≈ 1e-7 ⇒ 1e-12 的判据在 float32 上**恒红**，
   看着像算错，其实是精度不够。这是本关第一条「判据逼出来的工程约束」。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T1 关卡1 [实现] softmax_rows(scores)     torch 版：按**行**（dim=-1）归一
 T2 关卡1 [实现] attend_weights(q, k)     缩放点积 → 权重（torch 版，不乘 V）
 T3 关卡1 [实现] attention(q, k, v)       返回 (权重, 输出)，输出 = 权重 @ V
 T4 关卡1 [实现] hand_softmax(vals)       纯 Python 单行 softmax（手算那一半）
 T5 关卡1 [实现] hand_weights(q, k)       纯 Python 双层循环算 4×4 权重
 T6 关卡1 [实现] hand_output(w, v)        纯 Python 矩阵乘（权重 @ V）
 T7 关卡1 [判据] judge_hand_vs_torch()    判据 ①：逐元素差 < 1e-12 + 行和 1
════════════════════════════════════════════════════════════════════
 写完一项跑一项（别等七项全写完）：
   uv run python -c "from attention import judge_hand_vs_torch as f; print(f())"
 全绿后再跑总入口：uv run python attention.py

跑法：进 practice/ai/l2-nanogpt → uv sync → uv run python attention.py
文档依据（2026-11-22 查证，两条来源一致）：
  docs.pytorch.org · torch.nn.functional.scaled_dot_product_attention
      scale=None 时内部取 1/sqrt(E)（E = query.size(-1)）；公式 softmax(QK^T·scale)·V；
      softmax 作用在**最后一维**（key 序列维度）⇒ 每个 query 对全部 key 的权重和为 1
  PyTorch 仓库源码 wiki（pytorch/pytorch，math 后端）
      同上，且实现里调的是 _safe_softmax(attn, -1)，dim=-1 坐实
════════════════════════════════════════════════════════════════════
"""

import math

import torch

D_K: int = 2
"""每个 token 的维度。判据 ① 固定 d=2 —— 纸上能算完的最大尺寸。"""

Q_LIST: list[list[float]] = [
    [1.0, 0.0],
    [0.0, 1.0],
    [1.0, 1.0],
    [2.0, 0.0],
]
K_LIST: list[list[float]] = [
    [1.0, 0.0],
    [0.0, 2.0],
    [1.0, 1.0],
    [0.0, 1.0],
]
V_LIST: list[list[float]] = [
    [1.0, 0.0],
    [0.0, 1.0],
    [2.0, 2.0],
    [3.0, 0.0],
]
"""4 个 token 的 Q/K/V（写死）。手算那一半直接用这三份 list，别再抄一遍。"""

Q = torch.tensor(Q_LIST, dtype=torch.float64)
K = torch.tensor(K_LIST, dtype=torch.float64)
V = torch.tensor(V_LIST, dtype=torch.float64)


def softmax_rows(scores: torch.Tensor) -> torch.Tensor:
    """T1 · 按行 softmax：每一行**独立**归一，行和 = 1。

    镜子：`practice/ai/l1-bigram/softmax.py` 你写过「减最大值再 exp」。
    这里的差别只有一个 —— 按**行**（dim=-1）做，不是对整个向量做。
    提示：torch 里有现成函数一步做完，但先自己用 exp / sum 写一遍，
    再用它交叉验证（这跟本关「手算 vs torch」是同一个套路：两条路互相当判据）。
    """
    return scores.softmax(dim=-1)


def attend_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """T2 · 分数 → 权重（**不乘 V**）。

    scores = q @ k.T / sqrt(D_K)，形状 (4, 4)：第 i 行 = 第 i 个 token 对全部 4 个的分数。
    再对**每一行**做 softmax（可直接用 T1）。
    注意缩放除的是 sqrt(D_K)，别写死成 2 或别的东西。
    """
    scores = q @ k.T / D_K**0.5
    return softmax_rows(scores)


def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """T3 · 一次算完：返回 (权重, 输出)，权重 (4, 4)、输出 (4, 2)。

    返回权重是为了让判据能**分开**核对两段 —— 只比输出的话，
    权重算错 + V 也看错时可能凑巧对上，等于没验。
    """
    w = attend_weights(q, k)
    out = w @ v
    return w, out


def hand_softmax(vals: list[float]) -> list[float]:
    """T4 · 纯 Python 单行 softmax（禁用 torch）。

    输入一行的 4 个分数，返回 4 个权重（和 = 1）。
    要求：先减最大值再 exp（L1 写过理由：exp(大数) 会溢出）。

    四步，一步不能少：
      ① 找这一行的最大值 m
      ② 每个分数先减 m 再 exp  ——  减 m 不改变结果（分子分母同乘 e^-m），只为防溢出
      ③ 把 exp 结果求和
      ④ 每个都除以这个和  →  非负 + 和 = 1
    """
    m = max(vals)  # ①
    exps = [math.exp(v - m) for v in vals]  # ②
    total = sum(exps)  # ③
    return [e / total for e in exps]  # ④


def hand_weights(q: list[list[float]], k: list[list[float]]) -> list[list[float]]:
    """T5 · 手算 4×4 权重：对每个 (i, j) 求点积 → 缩放 → 整行 softmax。

    点积 = 对应位相乘再相加（d=2 就是两项相加）。
    先算出整行 4 个分数，再丢给 T4；不要一个个单独 softmax。

    两层循环的分工（与 torch 那半边一一对应）：
      外层 i = 第 i 个 query token  →  产出**一行**权重
      内层 j = 逐个 key token       →  算 q[i] · k[j] 这一个分数
      内层转完 → 这一行 4 个分数 → hand_softmax → 一行 4 个权重
    """
    rows: list[list[float]] = []
    for i in range(len(q)):
        scores: list[float] = []
        for j in range(len(k)):
            dot = sum(a * b for a, b in zip(q[i], k[j], strict=True))
            scores.append(dot / math.sqrt(len(q[i])))
        rows.append(hand_softmax(scores))
    return rows


def hand_output(w: list[list[float]], v: list[list[float]]) -> list[list[float]]:
    """T6 · 输出 = 权重 @ V：out[i][c] = Σ_j w[i][j] * v[j][c]。

    形状 (4, 2)：4 个 token，每个仍是 d=2 —— 注意力**不改变形状**，只改变内容。

    与 T5 的区别只有「内层在遍历谁」：
      T5 内层遍历 k，是为了**算分数**（点积求和）
      T6 内层遍历 j（全部 token），是为了**加权求和**（把 4 个 V 按比例掺起来）
      外层 i 不变：仍然是「一次算一个输出行」
    """
    out: list[list[float]] = []
    for i in range(len(w)):
        row: list[float] = []
        for c in range(len(v[0])):  # 输出的每一维（d=2 ⇒ c ∈ {0, 1}）
            row.append(sum(w[i][j] * v[j][c] for j in range(len(v))))
        out.append(row)
    return out


def judge_hand_vs_torch() -> bool:
    """T7 · 判据 ①（三条，全绿才返回 True）。

    ① 权重逐元素差：max|torch 权重 − 手算权重| < 1e-12
    ② 输出逐元素差：max|torch 输出 − 手算输出| < 1e-12
    ③ 行和：|权重每行和 − 1| < 1e-12（手算那份也要查 —— 两边都归一了才算对）
    失败时把两个矩阵和具体最大差值打出来（别只返回 False，否则不知道差在哪）。
    """
    # 阈值 1e-12 来自判据本身，不来自这里；本函数只做对账与打印（纯管道）
    w_torch, out_torch = attention(Q, K, V)

    w_hand_list = hand_weights(Q_LIST, K_LIST)
    out_hand_list = hand_output(w_hand_list, V_LIST)
    w_hand = torch.tensor(w_hand_list, dtype=torch.float64)
    out_hand = torch.tensor(out_hand_list, dtype=torch.float64)

    if w_hand.shape != w_torch.shape or out_hand.shape != out_torch.shape:
        print(
            f"形状不对：权重 {tuple(w_hand.shape)} vs {tuple(w_torch.shape)}"
            f"｜输出 {tuple(out_hand.shape)} vs {tuple(out_torch.shape)}"
        )
        return False

    d_w = (w_torch - w_hand).abs().max().item()
    d_out = (out_torch - out_hand).abs().max().item()
    d_row = (w_torch.sum(dim=-1) - 1.0).abs().max().item()
    d_row_hand = (w_hand.sum(dim=-1) - 1.0).abs().max().item()

    if max(d_w, d_out, d_row, d_row_hand) >= 1e-12:
        print("torch 权重：\n", w_torch)
        print("手算权重：\n", w_hand)
        print("torch 输出：\n", out_torch)
        print("手算输出：\n", out_hand)

    print(
        f"① 权重差 {d_w:.3e} ｜ ② 输出差 {d_out:.3e} "
        f"｜ ③ 行和偏差 torch {d_row:.3e} / 手算 {d_row_hand:.3e}"
    )
    return max(d_w, d_out, d_row, d_row_hand) < 1e-12


TOKENS: int = 64
"""判据 ② 的 token 数。64 ⇒ 均匀分布时最大权重 1/64 ≈ 0.0156、熵上界 log2(64) = 6 bit。"""

D_SET: tuple[int, ...] = (2, 8, 64, 512)
"""判据 ② 要求的四个 d。"""

SEEDS: int = 10
"""判据 ② ① 要取平均的 seed 数（单 seed 的比值噪声 ±15%，见 judge_scaling 的说明）。"""

RATIO_TOL: float = 0.05
"""判据 ② ① 的口径：实测比值与理论 √(d₂/d₁) 的相对误差上限。"""

MAXW_COLLAPSE: float = 0.9
"""判据 ② ② 的口径：未缩放时 max weight 达到多少算「坍缩」（逼近 1）。"""

H_COLLAPSE: float = 1.0
"""判据 ② ② 的口径：未缩放时平均熵低到多少算「坍缩」（单位 bit，上界 log2(TOKENS)=6）。"""

MAXW_REVIVED: float = 0.5
"""判据 ② ② 的口径：除以 √d 后 max weight 要掉回这个值以下（均匀时是 1/64≈0.016）。"""

H_REVIVED: float = 5.0
"""判据 ② ② 的口径：除以 √d 后平均熵要回升到这个值以上（上界 6 bit）。"""


def random_scores(d: int, tokens: int = TOKENS, seed: int = 0) -> torch.Tensor:
    """B1 · 造 (tokens, tokens) 的**未缩放**分数矩阵（脚手架，已写好）。

    q、k 各 (tokens, d) 从 N(0,1) 采样，固定 seed 保证可复现；返回 `q @ k.T`，**不除 √d**。
    为什么用随机数：判据 ② 要的是「d 变大时量级怎么涨」这个统计规律，随机数据才可复现且可算。
    """
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(tokens, d, generator=g, dtype=torch.float64)
    k = torch.randn(tokens, d, generator=g, dtype=torch.float64)
    return q @ k.T


def score_std(scores: torch.Tensor) -> float:
    """B2 · 整个分数矩阵的标准差（判据 ② 按 d 逐个报这个数）。

    理论预期：q、k 都是 N(0,1) 时，每个点积 = d 项相乘相加 ⇒ 方差 = d ⇒ 标准差 ≈ √d。
    所以 d 从 2 涨到 512，**未缩放**的 std 应涨 √(512/2) = 16 倍。
    """
    return scores.std().item()


def row_max_weight(w: torch.Tensor) -> float:
    """B3 · 全部行里**最大**的那个权重（坍缩指标）。

    解释：如果某行有一个数逼近 1，说明这个 token 几乎只盯一个 token，其余全被饿死。
    """
    return w.max().item()


def row_entropy_bits(w: torch.Tensor) -> float:
    """B4 · 所有行的**平均熵**，单位 bit：H = −Σ w·log2(w)。

    上界 = log2(token 数)（本关 64 ⇒ 6 bit，均匀分布时取到）；趋近 0 = 坍缩成 one-hot。
    注意：w 里可能出现 0（softmax 下溢）⇒ log2(0) = -inf，要先把 0 的那一项跳过。
    """
    nonzero_w = w[w > 0]
    return -(nonzero_w * nonzero_w.log2()).sum().item() / w.shape[0]


def judge_scaling() -> bool:
    """B5 · 判据 ②（两部分，两部分都要报数字）。

    ① **std 比**：d ∈ D_SET 的未缩放 std，两两比值 ≈ √(d₂/d₁)。
       口径：**先对 SEEDS 个 seed 取平均**，再要求 |实测比 / 理论比 − 1| < 5%。
       ⚠️ 为什么必须多 seed（2026-11-22 实测）：单 seed 的比值在 **14.8 ~ 18.4** 乱摆（理论 16），
          因为 (64,64) 里同一行的格子共享同一个 q 行向量，**不是独立样本**（有效样本远小于 4096）。
          10 个 seed 取平均后实测为 16.25 / 16.01 / 15.67 / 16.53 / 15.52（五组，全在 5% 内），
          30 seed 平均 15.98。⇒ 单 seed 判据会随机红，这一点和「单次测量不可判定」是同一个坑。
    ② **坍缩与回升**：d=512 时**不缩放**的 `max weight → 1`（口径 ≥ 0.9）、
       `熵 → 0`（口径 ≤ 1 bit）；
       **除以 √d 后** max weight 掉回来、熵**回升**（口径 ≥ 5 bit，上界 6）。
       两个数都要报：raw 和 scaled 各一份，别只报"回升了"。

    输出建议每个 d 一行，六个数字整齐排开：
        d=  2  std_raw=…  std_scaled=…  maxw_raw=…  maxw_scaled=…  H_raw=…  H_scaled=…
    任何一条不达标 ⇒ 返回 False（别静默凑绿）。
    """
    checks: list[bool] = []

    # ── ① 每个 d 的 std（raw 与 scaled 都报，各取 SEEDS 个 seed 的平均）──
    std_mean: dict[int, tuple[float, float]] = {}
    for d in D_SET:
        raws = [score_std(random_scores(d, seed=s)) for s in range(SEEDS)]
        scales = [score_std(random_scores(d, seed=s) / d**0.5) for s in range(SEEDS)]
        std_mean[d] = (sum(raws) / SEEDS, sum(scales) / SEEDS)
        raw_m, sc_m = std_mean[d]
        print(f"d={d:>3}  std_raw={raw_m:7.3f}（理论 √d={d**0.5:6.3f}）  std_scaled={sc_m:5.3f}")

    d0 = D_SET[0]
    for d in D_SET[1:]:
        measured = std_mean[d][0] / std_mean[d0][0]
        theory = (d / d0) ** 0.5
        err = abs(measured / theory - 1.0)
        passed = err < RATIO_TOL
        checks.append(passed)
        print(
            f"① {d:>3} / {d0}: 实测比 {measured:6.2f}  理论 {theory:6.2f}  "
            f"误差 {err:7.2%}  {'OK' if passed else 'NG'}"
        )

    # ── ② 最大 d 上的坍缩与回升（取 SEEDS 个 seed 的平均）──
    d_big = D_SET[-1]
    raw_maxw = raw_h = sc_maxw = sc_h = 0.0
    for s in range(SEEDS):
        scores = random_scores(d_big, seed=s)
        w_raw = softmax_rows(scores)
        w_scaled = softmax_rows(scores / d_big**0.5)
        raw_maxw += row_max_weight(w_raw) / SEEDS
        raw_h += row_entropy_bits(w_raw) / SEEDS
        sc_maxw += row_max_weight(w_scaled) / SEEDS
        sc_h += row_entropy_bits(w_scaled) / SEEDS

    ceiling = (TOKENS).bit_length() - 1  # log2(64) = 6
    pairs = [
        ("未缩放 maxw", raw_maxw, ">=", MAXW_COLLAPSE),
        ("未缩放 熵", raw_h, "<=", H_COLLAPSE),
        ("除√d 后 maxw", sc_maxw, "<=", MAXW_REVIVED),
        ("除√d 后 熵", sc_h, ">=", H_REVIVED),
    ]
    for name, value, op, limit in pairs:
        passed = value >= limit if op == ">=" else value <= limit
        checks.append(passed)
        print(f"② d={d_big} {name:>14}={value:8.4f}  ({op} {limit})  {'OK' if passed else 'NG'}")
    print(f"   （熵上界 log2({TOKENS}) = {ceiling} bit）")

    return all(checks)


def main() -> None:
    """测试入口（纯胶水）：分母 = 本关判据总数 2。"""
    results = [judge_hand_vs_torch(), judge_scaling()]
    print(f"判据 {sum(results)}/2 绿")


if __name__ == "__main__":
    main()
