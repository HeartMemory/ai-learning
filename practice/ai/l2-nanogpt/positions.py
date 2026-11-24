"""L2 · 注意力 —— 关卡 2：掩码与位置（块 2 = 位置编码）

一条根：注意力只看「谁和谁相关」，**天生不知道谁在前谁在后**。
    把 token 打乱重排，输出只是跟着一起重排 —— 这叫**置换等变**。
    ⇒ 「我打你」和「你打我」对纯注意力是同一件事，位置必须由外部塞进去。

两种塞法：
  A 正弦（加法，绝对）：x + pe(pos)
        pe[pos, 2i]   = sin(pos / 10000^(2i/d))
        pe[pos, 2i+1] = cos(pos / 10000^(2i/d))
        加完之后，同一个置换就**不再**等变了。
  B RoPE（乘法，旋转）：只对 q、k 施加随位置变化的二维旋转，v 不动
        q_m · k_n 精确只依赖 n − m（旋转矩阵正交 ⇒ R_m^T R_n = R_{n−m}）
        正交 ⇒ 不改变向量长度（这条可以当自检）

判据 ② 靠**置换测试**取证：拿一个固定置换 p，比较 out(x[p]) 与 out(x)[p]。
    无位置编码 ⇒ 差 **< 1e-12**（严格等变）
    加正弦 / RoPE ⇒ 差 **> 1e-3**（不再等变）

⚠️ 口径（2026-11-23 记）：判据 ② 用**不掩码**的 attention（关卡 1 的 `attention`）。
   因果掩码本身依赖位置（第 i 行只看前 i+1 个），重排后掩码语义已变 ⇒
   拿掩码版测「置换等变」必然大差，但那是掩码造成的，与位置编码无关。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T1 [实现] sinusoidal_pe(t, d)     正弦位置表 (t, d)：偶数维 sin、奇数维 cos
 T2 [实现] rope(x, base)           对 x 逐位置做二维旋转，返回同形状（v 不要用它）
 T3 [判据] judge_position()        判据 ②：无 PE 等变 / 正弦不等变 / RoPE 不等变（已写好）
════════════════════════════════════════════════════════════════════
 写完一项跑一项：
   uv run python -c "from positions import sinusoidal_pe; print(sinusoidal_pe(4, 4))"
 全绿后跑关卡入口（判据 ① + ②）：
   uv run python positions.py

文档依据（2026-11-23 查证）：
  arXiv:1706.03762（Attention Is All You Need）§3.5
      正余弦公式如上；作者选 sin/cos 的理由是「对任意固定偏移 k，PE(pos+k) 可写成 PE(pos)
      的线性（旋转）函数」，便于模型学相对位置；论文明确说这只是 hypothesis，
      与 learned embedding 实验打平（dev BLEU 25.7 vs 25.8），选正弦是为了可能的外推
  arXiv:2104.09864（RoFormer / RoPE）
      对 q、k 施加块对角旋转 R_m；由 R_m^T R_n = R_{n−m} 得 q_m·k_n 只依赖 n−m；
      无参数、不改变范数；v 不做位置变换
  docs.pytorch.org 2.15 · scaled_dot_product_attention
      缩放默认 1/√E（E = query.size(-1)）；等价实现 -inf 加在 softmax 之前
════════════════════════════════════════════════════════════════════
"""

import torch

from attention import attention
from masking import judge_causality, random_tokens

T: int = 8
"""置换测试的序列长度。"""

D: int = 4
"""每个 token 的维度。RoPE 要成对旋转 ⇒ d 必须是偶数。"""

PERM: tuple[int, ...] = (3, 0, 7, 2, 6, 1, 5, 4)
"""固定置换（下标 0..7 各一次）。写死而不是随机生成，判据才可复现。"""

EQ_TOL: float = 1e-12
"""判据 ② ① 的口径：无位置编码时必须等变，差要小于它。"""

PE_MIN: float = 1e-3
"""判据 ② ② 的口径：加了位置编码后，同一置换的差要大于它。"""

BASE: float = 10000.0
"""位置编码的基频。正弦编码与 RoPE 用同一个 10000（论文原值）。"""


def sinusoidal_pe(t: int = T, d: int = D) -> torch.Tensor:
    """T1 · 正弦位置表，形状 (t, d)，float64。

    公式（论文 §3.5，注意 2i 与 2i+1 的分工）：
        pe[pos, 2i]   = sin(pos / 10000^(2i/d))
        pe[pos, 2i+1] = cos(pos / 10000^(2i/d))

    拆成三块，一块一列，别一次写完：
      ① pos  ：形状 (t, 1) 的位置下标 —— 用 `torch.arange(t, dtype=torch.float64)` 再 unsqueeze
      ② 频率 ：形状 (d/2,) 的分母指数 —— `2i/d` 里的 i 取 0, 2, 4, ...（`torch.arange(0, d, 2)`）
               ⇒ `div = BASE ** (-i / d)`，注意是**负**指数（i 越大频率越低）
      ③ 填表 ：`torch.zeros(t, d, dtype=torch.float64)`（float64 别省，判据要 1e-12），
               偶数维填 sin(pos * div)、奇数维填 cos(pos * div)
               切片写法：`pe[:, 0::2]` 与 `pe[:, 1::2]`，形状都是 (t, d/2)

    自检三条（写完先不看判据，自己打一眼）：
      · 形状 (t, d)、值域 [-1, 1]；
      · 第 0 行（pos=0）是 [0, 1, 0, 1]（sin0=0，cos0=1）；
      · 低维（前面的列）随 pos 换得快，高维换得慢。
    """
    temp = list[int]()
    for i in range(t):
        temp.append(i)
    pos = torch.tensor(temp, dtype=torch.float64)
    pos = pos.unsqueeze(1)

    div = BASE ** (-torch.arange(0, d, 2, dtype=torch.float64) / d)

    pe = torch.zeros(t, d, dtype=torch.float64)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)

    return pe


def rope(x: torch.Tensor, base: float = BASE) -> torch.Tensor:
    """T2 · RoPE：对每个 token 做一次「按位置定角度」的二维旋转，返回同形状。

    只用于 q 和 k，**不要用在 v 上**（v 不做位置变换，这是 RoPE 与加性编码的分野）。

    做法（成对维度，相邻两维为一组，d 必须是偶数）：
      ① 角度表：`pos * div`，形状 (t, d/2)，div 与 T1 同一套 `base ** (-i/d)`（i = 0,2,4,...）
      ② 取成对切片：`x[..., 0::2]`（偶数维）、`x[..., 1::2]`（奇数维）
      ③ 二维旋转（θ 是该位置该组的角，cos/sin 用 `math.cos` 或 `torch.cos` 都行）：
             y_even = x_even * cos(θ) − x_odd * sin(θ)
             y_odd  = x_even * sin(θ) + x_odd * cos(θ)
      ④ 把两片写回同形状的新张量（别原地改 x，判据要拿原张量做对照）

    自检（这条能当场抓出「把 sin/cos 搞反」或「没做正交」）：
        torch.allclose(rope(x).norm(dim=-1), x.norm(dim=-1))
    旋转不改变长度 ⇒ 每一行的范数必须**逐位不变**（差 ~1e-16）。
    """
    div = base ** (-torch.arange(0, x.shape[-1], 2, dtype=torch.float64) / x.shape[-1])
    angle = torch.arange(0, x.shape[0], dtype=torch.float64).unsqueeze(1) * div
    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    y_even = x_even * torch.cos(angle) - x_odd * torch.sin(angle)
    y_odd = x_even * torch.sin(angle) + x_odd * torch.cos(angle)
    y = torch.zeros_like(x)
    y[..., 0::2] = y_even
    y[..., 1::2] = y_odd
    return y


def judge_position() -> bool:
    """T3 · 判据 ②（三项，全绿才 True）。

    置换 p 来自常量 PERM：`x[p]` = 按 p 重排后的输入。
      A 无位置编码 ⇒ out(x[p]) == out(x)[p]，差 < EQ_TOL（严格等变）
      B 加正弦     ⇒ 差 > PE_MIN（不再等变）
      C 加 RoPE    ⇒ 差 > PE_MIN（同上，且 v 不旋转）
    B/C 用「差够大」而不是精确值，是因为具体差多少取决于 seed 与置换。
    """
    x = random_tokens(T, D)
    p = torch.tensor(PERM)
    xp = x[p]
    checks: list[bool] = []

    # A 无位置编码：置换前后应当完全等变
    _, out = attention(x, x, x)
    _, out_p = attention(xp, xp, xp)
    d_a = (out_p - out[p]).abs().max().item()
    ok_a = d_a < EQ_TOL
    checks.append(ok_a)
    print(f"A 无位置编码 置换差 = {d_a:.3e}（要求 < 1e-12）  {'OK' if ok_a else 'NG'}")

    # B 正弦（加法）：同一位移后应不再等变
    pe = sinusoidal_pe(T, D)
    _, out_b = attention(x + pe, x + pe, x + pe)
    _, out_bp = attention(xp + pe, xp + pe, xp + pe)
    d_b = (out_bp - out_b[p]).abs().max().item()
    ok_b = d_b > PE_MIN
    checks.append(ok_b)
    print(f"B 正弦编码 置换差 = {d_b:.3e}（要求 > 1e-3）  {'OK' if ok_b else 'NG'}")

    # C RoPE（旋转，只动 q/k，v 保持原样）
    rx, rxp = rope(x), rope(xp)
    _, out_c = attention(rx, rx, x)
    _, out_cp = attention(rxp, rxp, xp)
    d_c = (out_cp - out_c[p]).abs().max().item()
    ok_c = d_c > PE_MIN
    checks.append(ok_c)
    print(f"C RoPE     置换差 = {d_c:.3e}（要求 > 1e-3）  {'OK' if ok_c else 'NG'}")

    # 附带自检：旋转不改变范数（与判据无关，但能一眼抓出实现错）
    dn = (rope(x).norm(dim=-1) - x.norm(dim=-1)).abs().max().item()
    print(f"  自检 RoPE 范数变化 = {dn:.3e}（应为 0 量级）")

    return all(checks)


def main() -> None:
    """关卡入口（纯胶水）：判据 ① 在 masking.py，判据 ② 在本文件，分母 = 2。"""
    results = [judge_causality(), judge_position()]
    print(f"判据 {sum(results)}/2 绿")


if __name__ == "__main__":
    main()
