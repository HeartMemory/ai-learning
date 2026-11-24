"""L2 · 注意力 —— 关卡 3：最小 block（块 1 = 骨架 + 判据 ① · 块 2 = 梯度 + 判据 ②）

一条根：attention 只做一件事 —— 让 token 互相「听」，但它**仍然是线性的**
（权重之和为 1 的加权平均，叠两层还是加权平均）。
    ⇒ 一个 block 必须补三样，否则把 block 叠深等于什么都没叠：
      LayerNorm  每个 token 的 C 维向量拉回「均值 0、方差 1」⇒ 深浅都不会爆/塌
      残差 x + f(x)  原样留一条直通路给梯度（不然深了就学不动）
      FFN + GELU  逐 token 的非线性（注意力只在 token 之间混，不做逐点非线性）

顺序用 **pre-LN**（归一化放在子层入口、残差绕过整段），当前默认写法：
      x = x + attn(ln1(x))
      x = x + ff(ln2(x))
    ff(z) = gelu(z @ w1 + b1) @ w2 + b2   ← 中间把 C 维撑到 hidden 再压回来

判据 ① 问五件可测的事（**不碰梯度**，那是块 2 的判据 ②）：
    A 形状：(B, T, C) 进 → (B, T, C) 出，block 不改变形状
    B 参数量：sum(p.numel()) == 手算公式，两边都报数字
    C 有界：输入整体 ×1000，LayerNorm 输出仍 ≤ √(C−1)，且**没被压平成 0**
    D 口径：与官方 nn.LayerNorm 逐元素一致（随机 + 小方差两组）
      —— 2026-11-24 变异实测补的：M1「无偏方差」与 M2「EPS 放开方外」在 C 项下全绿
    E 残差：|block(x)|max / |x|max ∈ [0.5, 2]
      —— 同样因变异 M3「去掉 x +」形状不变、gradcheck 也过，C 项抓不到

⚠️ C 的上界是算出来的，不是拍的：归一化后 mean 0、有偏方差 1 ⇒ Σx² = C
   ⇒ 单个元素最大 √(C−1)（其余 C−1 维全相等时取到）。C=4 ⇒ 上界 1.7321。
   若只查「≤ 上界」不查「> 1.0」，一个恒返回 0 的假实现也能绿（空转）。

⚠️ 本关全程 **float64**：判据 ② 的 gradcheck 只吃 double；float32 的 eps 1e-7
   会让 1e-12 量级的判据恒红（关卡 1 那条 dtype 约束同一个理由）。

判据 ② 改问**梯度**（块 2），三件事：
    A gradcheck：把 16 个参数**和** x 都当输入，用有限差分算出的 Jacobian
      与 autograd 的解析梯度逐元素比，全在默认容差内 ⇒ True
    B CE 梯度：d(cross_entropy)/d(logits) = (softmax(logits) − onehot) / N
      与 autograd 实测差 < 1e-12（**N = 参与平均的样本数**，不是元素总数 N×V）
    C 交叉熵本身：手写的 `cross_entropy` 与官方 `F.cross_entropy` 数值一致
      （没有 C，B 验的是官方实现，T6 写错了也没人发现）

⚠️ gradcheck 只吃 double：float32 的有限差分噪声（相对 ~3e-4）与默认
   rtol 1e-3 / atol 1e-5 同量级 ⇒ 写对写错都可能随机红绿，测试就失去意义。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 块 1（已完成）：判据 ①
 T1 [实现] layer_norm(x, weight, bias)        (…, C) → (…, C)，对**最后一维**
 T2 [实现] gelu(x)                            0.5·x·(1 + erf(x/√2))
 T3 [实现] MiniBlock.__init__                 按下面「参数规格」建 16 个 nn.Parameter
 T4 [实现] MiniBlock.forward                  pre-LN 两段残差
 T5 [判据] judge_block_shape()                判据 ①：A / B / C（已写好）
 块 2（本块）：判据 ②
 T6 [实现] cross_entropy(logits, targets)     hand-written logsumexp + 取真类 + 平均
 T7 [判据] judge_backward()                   判据 ②：gradcheck / 梯度公式 / CE 对账（已写好）
════════════════════════════════════════════════════════════════════
 参数规格（T3 照这张表建，名字与形状一步都不能错 —— 判据 B 按这张表算数）
   属性          形状        初始化
   ln1_w        (C,)        全 1
   ln1_b        (C,)        全 0
   wq, wk, wv   (C, C)      randn * 0.02
   bq, bk, bv   (C,)        全 0
   wo           (C, C)      randn * 0.02
   bo           (C,)        全 0
   ln2_w        (C,)        全 1
   ln2_b        (C,)        全 0
   w1           (HIDDEN, C) randn * 0.02
   b1           (HIDDEN,)   全 0
   w2           (C, HIDDEN) randn * 0.02
   b2           (C,)        全 0
   ⇒ 共 16 个参数（4 个 ln + 3×2 qkv + 2 proj + 4 ff）。
   ⚠️ 必须用 `nn.Parameter(...)` 赋值成属性才会被 `parameters()` 收集；
      裸 `self.wq = tensor` 不会进 `parameters()`，判据 B 会当它不存在。
   ⚠️ ln 的 w/b 必须是全 1 / 全 0：判据 C 靠这一点才能用 √(C−1) 当硬上界。
   ⚠️ randn 一律带固定 seed（`torch.Generator().manual_seed(seed)`）⇒ 判据可复现。

 写完一项跑一项（别等四项全写完）：
   T1/T2 单测：`import block` 后自己造 (2,4) 的 float64 张量喂进去，打一眼形状与统计量
   判据：     uv run python -c "from block import judge_block_shape as f; print(f())"
 全绿后跑总入口：uv run python block.py

跑法：进 practice/ai/l2-nanogpt → uv sync → uv run python block.py

文档依据（2026-11-24 查证，官方原文 + 本机实测两条来源一致）：
  docs.pytorch.org 2.15 · torch.nn.LayerNorm
      公式 y = (x − E[x]) / sqrt(Var[x] + ε) * γ + β
      原文：方差用 **biased** 估计，等价 `torch.var(input, correction=0)`
      ε **加在方差上再开方**（不是 sqrt(var) + ε），默认 eps=1e-5；
      elementwise_affine 默认 True，γ 初始全 1、β 初始全 0；train/eval 同行为
      本机实测：与 correction=0 差 2.22e-16 / 与 correction=1 差 0.2146（N=4 时 var 比 4/3）
  docs.pytorch.org 2.15 · torch.nn.GELU
      GELU(x) = x * Φ(x)，approximate 默认 'none'（精确 erf 实现）
      本机实测：默认 GELU 与 0.5x(1+erf(x/√2)) 差 2.22e-16
  docs.pytorch.org 2.15 · Extending PyTorch（gradcheck 段，块 2 会用）
      gradcheck 入参必须是 tuple，张量必须 **dtype=double 且 requires_grad=True**；
      比较「解析梯度 vs 有限差分」Jacobian，全在容差内才返回 True
      默认 eps=1e-6 / atol=1e-5 / rtol=1e-3（float32 的差分噪声与门限同量级 ⇒ 不可用）
════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import gradcheck
from torch.func import functional_call

# T4 的 forward 里要调它（关卡 2 的因果注意力，只吃 2 维 (T, C)）
from masking import causal_attention

B: int = 2
"""batch 大小。判据只要 2 条样本就够 —— 主要是证明「batch 维被正确循环/广播」，不是压性能。"""

T_SEQ: int = 8
"""序列长度。与关卡 2 的 `T` 取同一个 8，便于把掩码/位置那两关的结果直接搬进来。"""

C: int = 4
"""每个 token 的维度（= 单头注意力的 d，也 = FFN 的输入输出维度）。取 4 是为了手算参数量。"""

HIDDEN: int = 16
"""FFN 中间层宽度（C → HIDDEN → C）。原论文是 4×d_model，这里 4×4 = 16 正好一致。"""

EPS: float = 1e-5
"""LayerNorm 的数值稳定项，加在**方差**上再开方。取官方默认值，别自己调。"""

SCALE: float = 1000.0
"""判据 C 的输入放大倍数：把输入整体乘它，看 LayerNorm 输出是否仍有界。"""

LN_MIN_OUT: float = 1.0
"""判据 C 的下界：归一化后必须有元素接近 √(C−1)，否则「有界」可能是被压成了 0。"""

BOUND: float = (C - 1) ** 0.5
"""判据 C 的硬上界 √(C−1)（C=4 ⇒ 1.7321）。推导见文件头：mean 0 + Σx² = C。"""

BOUND_TOL: float = 1e-12
"""上界容差。ε 只会让输出**变小**，所以实际值必然略低于 BOUND，容差只防浮点抖动。"""

VOCAB: int = 5
"""判据 ② 的类别数（= 交叉熵里 logits 的最后一维）。取 5 是因为判据要手算 onehot 时可读。"""

N_CLS: int = 6
"""判据 ② 的样本数（= 交叉熵 reduction='mean' 的**分母**，注意不是 VOCAB×N_CLS）。"""

GRAD_TOL: float = 1e-12
"""判据 ② B 的容差：autograd 的 d(CE)/d(logits) 与手算公式的差要小于它。"""

CE_TOL: float = 1e-12
"""判据 ② C 的容差：手写 cross_entropy 与官方 F.cross_entropy 的标量差要小于它。"""

REF_TOL: float = 1e-12
"""判据 ① D 的容差：layer_norm 与官方 nn.LayerNorm 的逐元素差要小于它。

为什么补这条（2026-11-24 变异实测）：把 `correction=0` 改成 `1`，判据 C 全绿
—— 输出从 1.7083 掉到 1.4795，仍在 [1.0, √3] 区间内。EPS 位置写错同理（常规数据差
~1e-5，看不出来）。⇒ 必须用**独立参考实现**逐元素对账，并额外喂一组**小方差输入**
（var ~1e-6）把 eps 位置的影响放大。
"""

RES_RATIO_MIN: float = 0.5
"""判据 ① E 的下界：残差保留下，`|block(x)|max / |x|max` 应接近 1。"""

RES_RATIO_MAX: float = 2.0
"""判据 ① E 的上界。为什么补这条：删掉 `x + out` 里的 `x` 后，形状不变、gradcheck
也照样通过（函数依然可微）—— 变异实测比值从 ~1.0 掉到 **0.0012**，只有这条能抓。"""


def layer_norm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """T1 · 对**最后一维**做 LayerNorm，返回同形状。

    公式（官方原文，注意 ε 的位置）：
        y = (x − mean) / sqrt(var + ε) * weight + bias
    ⚠️ 方差用**有偏**口径（分母 = C，等价 `correction=0`）——
       用 `x.var(...)` 不写参数拿到的是 `correction=1`（Bessel），C=4 时差 4/3。

    提示（三行就够，别写循环）：
      ① 统计量沿最后一维（都要 keepdim）：
           `x.mean(dim=-1, keepdim=True)` / `x.var(dim=-1, keepdim=True, correction=0)`
      ② 归一化后乘 weight、加 bias（两者形状都是 (C,)，靠广播对上最后一维）
      ③ keepdim 不能省：丢了它 (…, C) 的统计量会变成 (…,)，broadcast 直接错维度

    自检（写完先别看判据）：
      · 输出每行均值 ≈ 0、有偏方差 ≈ 1（ε 让方差略小于 1）
      · 全 1 的输入 ⇒ 方差 0 ⇒ 输出 ≈ 0 但**不是 nan**（ε 就是干这个的）
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, correction=0)
    normalized = (x - mean) / torch.sqrt(var + EPS)
    return normalized * weight + bias

def gelu(x: torch.Tensor) -> torch.Tensor:
    """T2 · GELU 精确版：0.5 · x · (1 + erf(x / sqrt(2)))，返回同形状。

    为什么不用 ReLU：ReLU 在 0 处硬拐、负半轴一律 0；
    GELU 是「按 x 有多大来放行」（x·Φ(x)），负半轴留一点负值，梯度更连续。
    本关只要求数值等于官方默认实现（判据外自检）。

    提示：`torch.erf(x / 2**0.5)`，常数别写 1.414。
    """
    return 0.5 * x * (1 + torch.erf(x / 2**0.5))

class MiniBlock(nn.Module):
    """T3 / T4 · 单头 pre-LN block：x → x + attn(ln1(x)) → x + ff(ln2(x))。

    参数字段一律 float64（构造时用 `dtype=torch.float64`），判据 ② 的 gradcheck 要用。
    """

    def __init__(self, c: int = C, hidden: int = HIDDEN, seed: int = 0) -> None:
        """T3 · 照文件头「参数规格」表建 16 个 nn.Parameter，顺序不限、名字必须一致。

        两个容易错的点：
          · 全 1 / 全 0 用 `torch.ones` / `torch.zeros`；随机权重用同一个 generator
            （`g = torch.Generator().manual_seed(seed)`），这样判据 C 的上界才可预期
          · `super().__init__()` 必须第一行 —— 不调它 nn.Module 不会初始化内部字典，
            后面赋值 `self.xxx = nn.Parameter(...)` 会直接报错
        """
        super().__init__()
        self.ln1_w = nn.Parameter(torch.ones(c, dtype=torch.float64))
        self.ln1_b = nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.wq = nn.Parameter(torch.randn(c, c, dtype=torch.float64) * 0.02)
        self.wk = nn.Parameter(torch.randn(c, c, dtype=torch.float64) * 0.02)
        self.wv = nn.Parameter(torch.randn(c, c, dtype=torch.float64) * 0.02)
        self.bq = nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.bk = nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.bv = nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.wo = nn.Parameter(torch.randn(c, c, dtype=torch.float64) * 0.02)
        self.bo = nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.ln2_w = nn.Parameter(torch.ones(c, dtype=torch.float64))
        self.ln2_b = nn.Parameter(torch.zeros(c, dtype=torch.float64))
        self.w1 = nn.Parameter(torch.randn(hidden, c, dtype=torch.float64) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(hidden, dtype=torch.float64))
        self.w2 = nn.Parameter(torch.randn(c, hidden, dtype=torch.float64) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(c, dtype=torch.float64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """T4 · x: (B, T, C) → (B, T, C)。形状链如下（自己翻译成代码）：

            z  = layer_norm(x, ln1_w, ln1_b)             (B, T, C)
            q  = z @ wq + bq                             (B, T, C)   ← 同理 k / v
            for b in range(B):                           关卡 2 的 causal_attention 只吃 (T, C)
                w, o = causal_attention(q[b], k[b], v[b])   ⇒ o 是 (T, C)
            拼回 (B, T, C) → out = that @ wo + bo
            x1 = x + out                                 第一段残差
            h  = gelu(layer_norm(x1, ln2_w, ln2_b) @ w1.T + b1)   (B, T, HIDDEN)
            x2 = x1 + h @ w2.T + b2                      第二段残差，返回 x2

        ⚠️ w1 是 (HIDDEN, C)、w2 是 (C, HIDDEN) —— 这是 `nn.Linear(out, in)` 的布局
           官方约定（weight 形状 = (输出维, 输入维)），所以手写矩阵乘必须 **.T**。
           写成 `z @ w1` 会当场报
           `mat1 and mat2 shapes cannot be multiplied (4x4 and 16x4)`。
           `.T` 只对 2 维合法（w1/w2 是 2 维，没问题）；高维张量要用 `.mT`。

        为什么先归一化再进子层（pre-LN）而不是反过来：post-LN 要把残差也归一化，
        深了训练不稳、还得靠 warmup 救；pre-LN 让残差那条路始终是恒等映射。
        """
        z = layer_norm(x, self.ln1_w, self.ln1_b)
        q = z @ self.wq + self.bq
        k = z @ self.wk + self.bk
        v = z @ self.wv + self.bv

        outs = []
        for b in range(q.shape[0]):
            _, o_b = causal_attention(q[b], k[b], v[b])
            outs.append(o_b)

        o = torch.stack(outs, dim=0)
        out = o @ self.wo + self.bo
        x1 = x + out
        h = gelu(layer_norm(x1, self.ln2_w, self.ln2_b) @ self.w1.T + self.b1)
        x2 = x1 + h @ self.w2.T + self.b2
        return x2

def judge_block_shape() -> bool:
    """T5 · 判据 ①（五项，全绿才 True）。

    A 形状：(B, T, C) 进 → (B, T, C) 出
    B 参数量：torch 数出来的 == 按规格表手算的（两个数都打印）
    C 有界：x*1000 过 LayerNorm 后 ≤ √(C−1)，且 > LN_MIN_OUT（没被压平）
    D 口径：与官方 nn.LayerNorm 逐元素一致（随机输入 + 小方差输入两组）
      —— 2026-11-24 补，理由：变异 M1（改成 correction=1）与 M2（EPS 放开方外）
         在 C 项下全绿，漏网
    E 残差：`|block(x)|max / |x|max` ∈ [RES_RATIO_MIN, RES_RATIO_MAX]
      —— 2026-11-24 补，理由：变异 M3（去掉 x +）形状不变、gradcheck 也过，漏网
    """
    g = torch.Generator().manual_seed(0)
    x = torch.randn(B, T_SEQ, C, generator=g, dtype=torch.float64)
    block = MiniBlock()
    y = block(x)
    checks: list[bool] = []

    # A 形状不变
    ok_shape = tuple(y.shape) == (B, T_SEQ, C)
    checks.append(ok_shape)
    print(f"A 形状 {tuple(x.shape)} -> {tuple(y.shape)}（要求不变）  {'OK' if ok_shape else 'NG'}")

    # 自检（不计入判据）：输出里不该有 nan / inf
    ok_finite = bool(torch.isfinite(y).all())
    print(f"  自检 输出全有限：{ok_finite}")

    # B 参数量：torch 侧 vs 规格表手算
    n_params = sum(p.numel() for p in block.parameters())
    ln = 2 * C
    qkv = 3 * (C * C + C)
    proj = C * C + C
    ff = (C * HIDDEN + HIDDEN) + (HIDDEN * C + C)
    expect = ln + qkv + proj + ln + ff
    ok_numel = n_params == expect
    checks.append(ok_numel)
    print(
        f"B 参数量 torch {n_params} vs 手算 {expect}"
        f"（ln {ln} + qkv {qkv} + proj {proj} + ln {ln} + ff {ff}）  {'OK' if ok_numel else 'NG'}"
    )

    # C 有界：×1000 后 LayerNorm 输出仍 ≤ √(C−1)，且没被压平成 0
    big = x * SCALE
    z = layer_norm(big, block.ln1_w, block.ln1_b)
    raw_max = big.abs().max().item()
    ln_max = z.abs().max().item()
    base_max = layer_norm(x, block.ln1_w, block.ln1_b).abs().max().item()
    ok_bound = ln_max <= BOUND + BOUND_TOL
    ok_alive = ln_max > LN_MIN_OUT
    checks += [ok_bound, ok_alive]
    tag = "OK" if ok_bound else "NG"
    print(f"C 输入 absmax {raw_max:.2f} -> LN 后 absmax {ln_max:.4f}（上界 {BOUND:.4f}）  {tag}")
    print(f"C 未放大时 LN 后 absmax {base_max:.4f}（对照：应与放大后几乎一样）")
    print(f"C 未被压平：{ln_max:.4f} > {LN_MIN_OUT}  {'OK' if ok_alive else 'NG'}")

    # D 口径：与官方 nn.LayerNorm 逐元素一致（随机 + 小方差两组）
    ref = nn.LayerNorm(C, dtype=torch.float64)
    unit_w = torch.ones(C, dtype=torch.float64)
    zero_b = torch.zeros(C, dtype=torch.float64)
    d_rand = (layer_norm(x, unit_w, zero_b) - ref(x)).abs().max().item()
    tiny = 5.0 + torch.arange(C, dtype=torch.float64) * 1e-3  # var ~1e-6 ⇒ 放大 eps 位置
    d_tiny = (layer_norm(tiny, unit_w, zero_b) - ref(tiny)).abs().max().item()
    ok_ref = max(d_rand, d_tiny) < REF_TOL
    checks.append(ok_ref)
    tag_d = "OK" if ok_ref else "NG"
    print(f"D 与官方 LayerNorm 差：随机 {d_rand:.3e} / 小方差 {d_tiny:.3e}  {tag_d}")

    # E 残差：输出量级必须与输入同阶（删掉 x + 后比值会塌到 0.001 量级）
    ratio = y.abs().max().item() / x.abs().max().item()
    ok_res = RES_RATIO_MIN <= ratio <= RES_RATIO_MAX
    checks.append(ok_res)
    tag_e = "OK" if ok_res else "NG"
    print(f"E 残差比 |out|/|in| = {ratio:.4f}（要求 {RES_RATIO_MIN}~{RES_RATIO_MAX}）  {tag_e}")

    if not all(checks):
        print("block 输出（首条样本）：\n", y[0])
    return all(checks)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """T6 · 交叉熵（reduction='mean' 口径），输入 (N, V) 与 (N,)，返回**标量**。

    官方等价式（F.cross_entropy 文档）：
        cross_entropy = nll_loss(log_softmax(logits, dim=1), targets)
    单个样本逐项写开（C = V 类）：
        loss_i = −log softmax(logits_i)[t_i]
               = logsumexp(logits_i) − logits_i[t_i]
    再对**样本**取平均 ⇒ 分母是 N（不是 N×V）。

    提示（三段，别写循环）：
      ① `torch.logsumexp(logits, dim=-1)` —— **别**手写 `log(exp(x).sum())`：
         大 logits 会先溢出成 inf（与关卡 1 的 naive softmax 同一个死法）
      ② 取真类下标：`logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)`
         （等价写法 `logits[torch.arange(logits.shape[0]), targets]`，选一个）
      ③ 平均：`(logsumexp − 取出的那列).mean()`

    自检：与 `F.cross_entropy(logits, targets)` 差 < 1e-12（判据 ② C 会查）。

    ⚠️ 为什么没有复用 L1 的 `ce_bigram`（2026-11-24 核对 `l1-bigram/entropy.py`）：
       L1 吃「字典概率表 + 字符串」，单位 log2（bits/char），纯 Python 不反传；
       这里吃「logits 张量 + 类别下标」，单位自然对数，且必须让 autograd 穿过去。
       能复用的只有它引用过的那条等式：cross_entropy = LogSoftmax + NLLLoss。
    """
    lse = torch.logsumexp(logits, dim=-1)  # ① (N,)
    picked = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # ② (N,)
    return (lse - picked).mean()  # ③ 核心：减完即 −log p(真类)；mean 对应 reduction='mean'
    

def judge_backward() -> bool:
    """T7 · 判据 ②（三项，全绿才 True）。

    A gradcheck：小尺寸 block（c=2, hidden=4 ⇒ 54 个参数元素 + x 的 6 个 = 60 个入参），
      用 `functional_call` 把参数当**输入**喂进去，这样每个参数都会被有限差分验一遍。
      ⚠️ 只把 x 当输入是不够的：那样只验了 d输出/d输入，16 个参数的梯度谁都没查。
    B CE 梯度：d(CE)/d(logits) 必须 = (softmax − onehot)/N。
    C CE 数值：手写 `cross_entropy` 必须与 `F.cross_entropy` 一致
      （没有 C，B 验的就只是官方实现，T6 写错也发现不了）。
    """
    checks: list[bool] = []

    # A gradcheck —— 参数与输入都当入参
    torch.manual_seed(0)
    small = MiniBlock(c=2, hidden=4)
    names = [n for n, _ in small.named_parameters()]
    vals = [p.detach().clone().requires_grad_(True) for p in small.parameters()]
    x = torch.randn(1, 3, 2, dtype=torch.float64, requires_grad=True)

    def f(x_in: torch.Tensor, *ps: torch.Tensor) -> torch.Tensor:
        return functional_call(small, dict(zip(names, ps, strict=True)), (x_in,))

    n_in = sum(v.numel() for v in vals) + x.numel()
    ok_gc = gradcheck(f, (x, *vals), raise_exception=False)
    checks.append(ok_gc)
    print(f"A gradcheck（{n_in} 个入参，float64） -> {ok_gc}  {'OK' if ok_gc else 'NG'}")

    # B / C —— 交叉熵与它的梯度
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(N_CLS, VOCAB, generator=g, dtype=torch.float64, requires_grad=True)
    targets = torch.randint(0, VOCAB, (N_CLS,), generator=g)
    loss = F.cross_entropy(logits, targets)
    loss.backward()

    soft = torch.softmax(logits.detach(), dim=-1)
    onehot = F.one_hot(targets, VOCAB).to(torch.float64)
    hand = (soft - onehot) / N_CLS
    d_grad = (logits.grad - hand).abs().max().item()
    ok_grad = d_grad < GRAD_TOL
    checks.append(ok_grad)
    tag_b = "OK" if ok_grad else "NG"
    print(f"B CE 梯度 vs 手算 (softmax - onehot)/N 最大差 {d_grad:.3e}  {tag_b}")

    mine = cross_entropy(logits.detach(), targets)
    d_ce = abs(mine.item() - loss.item())
    ok_ce = d_ce < CE_TOL
    checks.append(ok_ce)
    tag_c = "OK" if ok_ce else "NG"
    print(f"C 手写 CE {mine.item():.12f} vs 官方 {loss.item():.12f} 差 {d_ce:.3e}  {tag_c}")

    # 附带自检（不计入判据）：每行梯度之和应为 0（softmax 概率和为 1 ⇒ 平移不变）
    row_sum = logits.grad.sum(dim=-1).abs().max().item()
    print(f"  自检 每行梯度之和 |max| = {row_sum:.3e}（应为 0 量级）")
    return all(checks)


def main() -> None:
    """测试入口（纯胶水）：判据 ①（形状/参数/有界）+ 判据 ②（梯度），分母 = 2。"""
    results = [judge_block_shape(), judge_backward()]
    print(f"判据 {sum(results)}/2 绿")


if __name__ == "__main__":
    main()
