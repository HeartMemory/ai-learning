"""L2 · 注意力 —— 关卡 2：掩码与位置（块 1 = 因果掩码）

一条根：自回归模型一次只预测下一个 token ⇒ 位置 i 只能看 0..i，更远的是「未来」。
    屏蔽的方式 = 给未来的分数加 -inf，**再** softmax。
        exp(-inf) = 0  ⇒ 那些位置的权重正好 0
        剩下的位置重新分掉这 100% 的预算 ⇒ 行和仍然 = 1
    ⚠️ 顺序不能反：先 softmax 再把未来置 0 ⇒ 行和 < 1，注意力预算漏了一块。

判据 ① 只证明一件事：**未来的门确实关上了，而且关的是它、不是扰动没生效**。
    取证靠扰动，不靠肉眼看矩阵：
      A 掩码版：改第 5、6 个 token → 前 4 个位置 Δ **严格 0**
      B 掩码版：只改第 6 个 token → 第 5 个位置 Δ **严格 0**
      C 对照（关卡 1 的不掩码 attention）：同样的 B 扰动能传到过去 ⇒ Δ > 0
        （没有 C，「Δ = 0」可能只是扰动根本没生效，判据空转）
      D 行统计：掩码后第 i 行非零个数 = i+1、行和 = 1（第 0 行只有 1 个合法位）

⚠️ 全行被屏蔽 ⇒ softmax(全 -inf) = 0/0 = NaN（数学事实，不是文档原话）。
   自注意力第 0 行恰好有 1 个合法位（自己），所以躲开了；做 padding 掩码时要小心。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T1 [实现] causal_mask(t)                 下三角掩码，True = 允许看
 T2 [实现] masked_scores(scores, mask)    屏蔽 = 加 -inf（不是乘 0、不是事后置 0）
 T3 [实现] causal_weights(q, k)           缩放点积 → 掩码 → 按行 softmax（复用关卡 1）
 T4 [实现] causal_attention(q, k, v)      返回 (权重, 输出)
 T5 [判据] judge_causality()              判据 ①：A / B / C / D 四项（已写好）
════════════════════════════════════════════════════════════════════
 写完一项跑一项（别等全写完）：
   uv run python -c "from masking import judge_causality as f; print(f())"
 全绿后跑总入口：uv run python masking.py

跑法：进 practice/ai/l2-nanogpt → uv sync → uv run python masking.py
文档依据（2026-11-23 查证，两条来源一致）：
  docs.pytorch.org 2.15 · torch.nn.functional.scaled_dot_product_attention
      is_causal 为真 ⇒ 掩码是下三角；等价实现 temp_mask = ones(L,S,bool).tril(diagonal=0)
      然后 attn_bias.masked_fill_(~temp_mask, float("-inf")) ⇒ -inf 在 softmax **之前**
      bool 掩码语义原文：True = 该位置**参与**注意力（与 MultiheadAttention 的
      key_padding_mask **相反**，迁过来要取反）
      同时设 attn_mask 与 is_causal ⇒ 抛错（An error is thrown if both ... are set）
  PyTorch 2.15 同页等价实现代码块：scale_factor = 1/sqrt(query.size(-1)) when scale is None
════════════════════════════════════════════════════════════════════
"""

import torch

from attention import attention

T: int = 8
"""序列长度。判据 ① 用 8 个 token —— 扰动项（改第 5、6 个）和观测量（前 4 个）都装得下。"""

D: int = 4
"""每个 token 的维度。本关不考手算，d 取小一点便于人眼核对权重矩阵。"""

DELTA: float = 0.7
"""扰动幅度。要足够大，让「.abs().max()」不可能被舍入噪声解释成 0。"""


def random_tokens(t: int = T, d: int = D, seed: int = 0) -> torch.Tensor:
    """脚手架 · 造 (t, d) 的 token 向量（已写好）。

    固定 seed ⇒ 判据可复现；float64 ⇒ 判据要「严格 0」，float32 的 eps 1e-7 会把
    「算得一样」和「差一点点」混在一起（关卡 1 那条 dtype 约束同一个理由）。
    """
    g = torch.Generator().manual_seed(seed)
    return torch.randn(t, d, generator=g, dtype=torch.float64)


def causal_mask(t: int) -> torch.Tensor:
    """T1 · 下三角掩码 (t, t)，**True = 允许看**（这是 SDPA 的约定，不是 True = 屏蔽）。

    就要文档等价实现里那一行，**原样**：
        torch.ones(t, t, dtype=torch.bool).tril(diagonal=0)
    diagonal=0 的意思是「对角线也在内」—— 第 i 个位置必须能看见自己。

    先想清楚形状：第 i 行 = 第 i 个 query 能看的 key 集合 ⇒
        第 0 行 [T, F, F, ...]  第 i 行前 i+1 个 T
    写完自己打一眼：print(causal_mask(4))。
    """
    return torch.ones(t, t, dtype=torch.bool).tril(diagonal=0)


def masked_scores(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """T2 · 屏蔽 = 给「不允许」的位置加 -inf。

    只有这一种做法在本关算对，另外两种都是错的（先想为什么，再看注释末尾）：
      ① 乘 0            → softmax 会把 0 当成「分数很低」重新归一，未来照样分到权重
      ② softmax 后置 0  → 行和 < 1，注意力预算漏掉一块
    提示：`masked_fill` 吃的是「要填哪些位置」的布尔张量，
    而 causal_mask 给的是「允许哪些位置」⇒ 差一个取反。
    ⚠️ -inf 要写成浮点负无穷（`float("-inf")`），别拿 -1e9 凑数：换个 dtype 就溢出/失真。
    """
    return scores.masked_fill(~mask, float("-inf"))


def causal_weights(q: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """T3 · 缩放点积 → 掩码 → 按行 softmax，返回 (t, t) 权重。

    镜子：关卡 1 的 `attend_weights` 只差一步 —— 中间插一道 T2。
      分数 = q @ k.T / sqrt(d)，d 取 q.shape[-1]（别写死）
      掩码 t 取自 q.shape[0]（query 数量 = 序列长度；不要用 k.shape[0]，块 2 会踩到）
      最后按最后一维归一：可直接 `torch.softmax(x, dim=-1)`，或复用关卡 1 的
      `softmax_rows`（那样需要在文件顶部自己加一行 `from attention import softmax_rows`）
    """
    scores = q @ k.T / q.shape[-1] ** 0.5
    mask = causal_mask(q.shape[0])
    return torch.softmax(masked_scores(scores, mask), dim=-1)


def causal_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """T4 · 返回 (权重, 输出) = (T3 的结果, 权重 @ v)。

    和关卡 1 一样，把权重一起返回 —— 判据要分开核对「门关没关」与「输出对不对」，
    只看输出的话，权重错 + V 也错可能凑巧对上。
    """
    w = causal_weights(q, k)
    return w, w @ v


def judge_causality() -> bool:
    """T5 · 判据 ①（四项，全绿才 True）。

    ⚠️ 口径说明（2026-11-23 记）：阶段表原文第三子句写作「只改第 6 个 → 第 5 个 Δ > 0」，
    但掩码正确时它**必须**是 0（否则前半句自相矛盾）。⇒ 这句按**对照**理解并落地：
    不掩码时同一扰动能传到过去（C 项 Δ > 0），掩码时被挡住（B 项 Δ = 0）。
    两项一起才说明「是掩码在挡，不是扰动没生效」。
    """
    x = random_tokens()
    w, out = causal_attention(x, x, x)
    checks: list[bool] = []

    # D 行统计：第 i 行非零个数 = i+1，行和 = 1
    nnz = (w != 0).sum(dim=-1)
    expect = torch.arange(1, T + 1)
    ok_nnz = torch.equal(nnz, expect)
    d_row = (w.sum(dim=-1) - 1.0).abs().max().item()
    ok_row = d_row < 1e-12
    checks += [ok_nnz, ok_row]
    print(f"D 行非零个数 {nnz.tolist()} vs 期望 {expect.tolist()}  {'OK' if ok_nnz else 'NG'}")
    print(f"D 行和偏差 {d_row:.3e}  {'OK' if ok_row else 'NG'}")

    # A 改第 5、6 个 token（0 基下标 4、5）→ 前 4 个位置（下标 0..3）必须一动不动
    x56 = x.clone()
    x56[[4, 5]] += DELTA
    _, out56 = causal_attention(x56, x56, x56)
    d_past = (out56[:4] - out[:4]).abs().max().item()
    ok_past = d_past == 0.0
    checks.append(ok_past)
    print(f"A 改 5,6 后前 4 个位置 Δ = {d_past:.3e}（要求严格 0）  {'OK' if ok_past else 'NG'}")

    # B 只改第 6 个 token（下标 5）→ 第 5 个位置（下标 4）必须一动不动
    x6 = x.clone()
    x6[5] += DELTA
    _, out6 = causal_attention(x6, x6, x6)
    d_prev = (out6[4] - out[4]).abs().max().item()
    ok_prev = d_prev == 0.0
    checks.append(ok_prev)
    print(f"B 改 6 后第 5 个位置 Δ = {d_prev:.3e}（要求严格 0）  {'OK' if ok_prev else 'NG'}")

    # C 对照：关卡 1 的不掩码 attention，同样的扰动会传到过去 ⇒ Δ > 0
    _, plain = attention(x, x, x)
    _, plain6 = attention(x6, x6, x6)
    d_leak = (plain6[4] - plain[4]).abs().max().item()
    ok_leak = d_leak > 0.0
    checks.append(ok_leak)
    print(f"C 不掩码对照组 第 5 个位置 Δ = {d_leak:.3e}（要求 > 0）  {'OK' if ok_leak else 'NG'}")

    if not all(checks):
        print("掩码：\n", w)
        print("A 两次输出（前 4 行）：\n", out[:4], "\n", out56[:4])
    return all(checks)


def main() -> None:
    """测试入口（纯胶水）：块 1 只覆盖判据 ①，分母先写 1；块 2 加上判据 ② 后改成 2。"""
    results = [judge_causality()]
    print(f"判据 {sum(results)}/1 绿")


if __name__ == "__main__":
    main()
