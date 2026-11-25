"""L2 · 注意力 —— 关卡 4（块 3a/3c）：char-level 最小 GPT

一条根：关 3 的 `MiniBlock` 只证明了「一个 block 形状对、梯度对」，它有两个
**故意留下**的局限，真训练必须换掉：
    ① dtype 固定 float64（为了 gradcheck）⇒ 训练要用 float32，否则显存翻倍、速度减半
    ② 逐 batch 调 2D 的 `causal_attention`（`for b in range(B)`）⇒ 真训练 B=64，
       一个 batch 就 64 次 Python 调用；注意力应当**一次算完整个 (B,T,·)**
所以块 3 的模型是「同一个结构、批量版、float32」。关 3 的版本留着当对照
（判据 ① 的 D 项、② 的 gradcheck 都在那边，不需要重跑）。

⚠️ 本块直接用官方 `F.scaled_dot_product_attention(is_causal=True)`，不再手写
   2D 版的原因（2026-11-25 查证 + 本机实测）：
   · 官方推荐的融合实现：一次 kernel 内完成 score / mask / softmax / 加权，
     不需要物化 (T,T) 权重矩阵，显存与速度都优于手写版
   · **等价性已在关 2 取证**：自写 `causal_attention` 与 `is_causal=True` 的输出差
     2.220e-16（见 `README.md` 关卡 2「独立复核」一行）
   · 本机实测：SDPA 接受 (B, T, C) 三维输入，`is_causal=True` 生效（改第 0 个位置，
     后面各行全部变化；第 0 行不受后面位置影响）
   ⇒ 「手写一遍」的教学目的关 2 / 关 3 已经达到，这里换成引擎件不损失内容。

⚠️ 结构上仍是**单头**（与关 2 / 关 3 同构）：多头只是把 C 切成 h 份并在前面插一个
   head 维，属于并行形状问题，不在本关判据里。留到后面要用时再拆。

形状链（贯穿全文件，写代码时对一遍）：
    idx      (B, T)          int64，0..V-1
    tok_emb  (B, T, C)
  + pos_emb  (T, C)   → 广播成 (B, T, C)
    block×4  (B, T, C)   ← 每层：x + attn(ln1(x))，再 x + ff(ln2(x))（pre-LN）
    ln_f     (B, T, C)
    head     (B, T, V)   ← logits，每个位置对 V 个字符打分
    loss     标量        ← cross_entropy(logits.reshape(-1,V), targets.reshape(-1))

⚠️ 参数量手算（C=128 · N_LAYER=4 · HIDDEN=512 · V=65 · BLOCK=128，判据 B 按它核）：
     每层：ln1 256 + qkv 3×16512 = 49536 + proj 16512 + ln2 256
           + ff1 512×128+512 = 66048 + ff2 128×512+128 = 65664  ⇒ 198272
     4 层 = 793088
     tok_emb 65×128 = 8320 · pos_emb 128×128 = 16384 · ln_f 256 · head 128×65 = 8320
    合计 **826368**（约 0.83 M）
   ⇒ 2000 步 × 64 × 128 = 16,384,000 token ⇒ **19.8 token/参数**（与常见的小模型预算同量级）

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个方法，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 M1 [实现] _Block.__init__     5 个 nn.Module 子层（ln1 / qkv 三件 / proj / ln2 / ff 两件）
 M2 [实现] _Block.forward      x → x + attn(ln1(x)) → x + ff(ln2(x))
 M3 [实现] CharGPT.__init__    tok_emb / pos_emb / blocks / ln_f / head
 M4 [实现] CharGPT.forward     idx → logits（给 targets 时同时返回 loss）
 M5 [已判据] judge_model()     A 形状 / B 参数量 / C CE 对账 / D 因果 / E 位置表（已写好）

写完一项跑一项（别等四个全写完）：
    uv run python model.py          # 判据 1/1 绿；未实现的项会当场报 NG
    （M4 单独验：`import model` 后跑 `model.CharGPT()(torch.zeros(2,16, dtype=torch.int64))`）
════════════════════════════════════════════════════════════════════
"""

import torch
import torch.nn.functional as F
from torch import nn

from corpus import BLOCK

V: int = 65
"""词表大小（char-level，`len(itos)`）。模型构造时按参数传，默认与语料一致。"""

C: int = 128
"""每个 token 的向量宽度（= d_model）。取 128 是为了 4 层下参数量落在 ~0.8 M。"""

HIDDEN: int = 4 * C
"""FFN 中间宽度（= 4×C，原论文比例）。C=128 ⇒ 512。"""

N_LAYER: int = 4
"""block 层数。层数越多「远关系」越深，但也越慢；2000 步预算下 4 层够。"""

JUDGE_B: int = 2
"""判据用的 batch 大小（只要证明形状与因果，不需要压性能）。"""

JUDGE_T: int = 16
"""判据用的序列长度。注意必须 ≤ BLOCK，否则 pos_emb 切片越界。"""

CE_TOL: float = 1e-6
"""判据 C 的容差：模型返回的 loss 与手动 `F.cross_entropy` 的差（float32 下 ~1e-7）。"""


class _Block(nn.Module):
    """单层 pre-LN block：x → x + attn(ln1(x)) → x + ff(ln2(x))。

    与关 3 的 `MiniBlock` **结构相同**，差别只有三点：
      · 用官方 `nn.Linear` / `nn.LayerNorm`（手写版已在关 3 验过，这里要的是速度与数值稳定性）
      · 注意力用 `F.scaled_dot_product_attention(is_causal=True)`（一次算完整个 batch）
      · float32 + 批量维，不再逐 batch 循环
    """

    def __init__(self, c: int = C, hidden: int = HIDDEN) -> None:
        """M1 · 按下面这张表建子层，名字必须一致（判据 B 只数参数量，但 3b 会按名字取）。

        属性     类型              形状（weight / bias）
        ln1      nn.LayerNorm(c)   (c,) / (c,)
        wq       nn.Linear(c, c)   (c, c) / (c,)     ← 注意 nn.Linear 的 weight 是 (out, in)
        wk       nn.Linear(c, c)   (c, c) / (c,)
        wv       nn.Linear(c, c)   (c, c) / (c,)
        proj     nn.Linear(c, c)   (c, c) / (c,)
        ln2      nn.LayerNorm(c)   (c,) / (c,)
        ff1      nn.Linear(c, hidden)   (hidden, c) / (hidden,)
        ff2      nn.Linear(hidden, c)   (c, hidden) / (c,)

        提示：`super().__init__()` 第一行；其余一行一个 `nn.Linear(...)`。
        ⚠️ 别写成 `self.wq = nn.Parameter(torch.randn(c, c))` —— 那样没有 bias、
           没有默认初始化，且与关 3 的参数规格不再是同一套。
        """
        super().__init__()
        self.ln1 = nn.LayerNorm(c)
        self.wq = nn.Linear(c, c)
        self.wk = nn.Linear(c, c)
        self.wv = nn.Linear(c, c)
        self.proj = nn.Linear(c, c)
        self.ln2 = nn.LayerNorm(c)
        self.ff1 = nn.Linear(c, hidden)
        self.ff2 = nn.Linear(hidden, c)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """M2 · x: (B, T, C) → (B, T, C)。

        形状链（自己翻译成代码）：
            z  = ln1(x)                       (B, T, C)
            q  = wq(z); k = wk(z); v = wv(z)  各 (B, T, C)
            o  = F.scaled_dot_product_attention(q, k, v, is_causal=True)   (B, T, C)
            x1 = x + proj(o)                  第一段残差
            h  = F.gelu(ff1(ln2(x1)))         (B, T, HIDDEN)
            return x1 + ff2(h)                第二段残差

        ⚠️ `is_causal=True` 要求 q 与 k 的**序列长度相同**（这里恒为 T），它会自己造
           下三角掩码，**不要**再手写 mask 传进去（传了反而被忽略或形状冲突）。
        ⚠️ SDPA 接受三维 (B, T, C) 输入：它把前两维当 batch、最后一维当每个头的通道。
           若哪天改多头，(B, T, C) 要先 reshape 成 (B, h, T, C/h)。
        ⚠️ 前向里**不**做 loss、**不**做 softmax —— 交给 M4 与 `F.cross_entropy`。
        """
        z = self.ln1(x)
        q = self.wq(z)
        k = self.wk(z)
        v = self.wv(z)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x1 = x + self.proj(o)
        h = F.gelu(self.ff1(self.ln2(x1)))
        return x1 + self.ff2(h)

class CharGPT(nn.Module):
    """char-level 最小 GPT：token 嵌入 + 位置嵌入 + N 层 block + 输出头。"""

    def __init__(self, vocab_size: int = V, c: int = C, n_layer: int = N_LAYER) -> None:
        """M3 · 建五个部件（判据 B 会数总参数量）。

        属性      类型                    形状 / 初始化
        tok_emb   nn.Embedding(vocab, c)  (vocab, c)   默认初始化即可
        pos_emb   nn.Parameter            (BLOCK, c)   randn * 0.02
        blocks    nn.ModuleList            n_layer 个 _Block(c)
        ln_f      nn.LayerNorm(c)         (c,) / (c,)
        head      nn.Linear(c, vocab, bias=False)   (vocab, c)

        提示：
          · `nn.ModuleList([...])` 而不是普通 list —— 普通 list 里的子模块**不会**被
            `parameters()` 收集（与关 3「裸 tensor 不进 parameters」同族，且不报错）
          · `pos_emb` 长度取 **BLOCK**（最大上下文），forward 里按实际 T 切片 `[:T]`
          · `head` 不要 bias：词表维的偏置等价于给每个字符一个固定先验，与
            `tok_emb` 的组合已有足够自由度；省 (vocab) 个参数，也是常见做法
          · 位置表初始化 `torch.randn(BLOCK, c) * 0.02`：小随机，不用全 0
            （全 0 时所有位置在第一步完全等价，靠梯度自己分开，慢）
        """
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, c)
        self.pos_emb = nn.Parameter(torch.randn(BLOCK, c) * 0.02)
        self.blocks = nn.ModuleList([_Block(c) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(c)
        self.head = nn.Linear(c, vocab_size, bias=False)
        self.vocab_size = vocab_size

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """M4 · idx (B, T) → (logits (B, T, V), loss 标量或 None)。

        形状链（自己翻译成代码）：
            T = idx.shape[1]                       必须 ≤ BLOCK
            x = tok_emb(idx) + pos_emb[:T]         (B, T, C) ← (T, C) 靠广播
            for blk in blocks: x = blk(x)          (B, T, C)
            x = ln_f(x)                            (B, T, C)
            logits = head(x)                       (B, T, V)
            loss = F.cross_entropy(logits.reshape(-1, vocab), targets.reshape(-1))
                   （targets 为 None 时 loss = None —— 生成时不需要标签）

        提示：
          · 用 `.reshape` 而不是 `.view`：`head` 的输出**通常**连续，但 `view` 一旦遇到
            非连续就报 `view size is not compatible`；`reshape` 会自动处理
          · loss 的分母是 **B×T**（所有位置），不是 B —— 与判据 ② 的 CE 单位
            (nats/字符) 直接对齐
          · 别在这里加 softmax：`F.cross_entropy` 内部就是 log_softmax + nll，
            多套一层会把 loss 变成垃圾（且不报错）

        自检：`logits.shape == (B, T, V)`、`loss.ndim == 0`、`torch.isfinite(loss)`。
        """
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb[:T]
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), targets.reshape(-1))
        return logits, loss

def judge_model() -> bool:
    """M5 · 判据 A–E（全绿才 True）。

    A 形状：logits (B,T,V)、loss 是 0 维有限标量
    B 参数量：torch 数出来 == 手算公式（两个数都打印）
    C CE 对账：模型返回的 loss == 手动 `F.cross_entropy(logits.reshape(-1,V), y.reshape(-1))`
    D 因果（两路取证）：改**最后一个** token ⇒ 前 T-1 个位置的 logits **严格不变**，
      且最后一个位置确实变了（缺后半句时「不变」可能只是扰动没生效 —— 关 2 的教训）
    E 位置表：`pos_emb.shape == (BLOCK, C)`，且 forward 能吃满 BLOCK 长度
    """
    torch.manual_seed(0)
    try:
        model = CharGPT(vocab_size=V, c=C, n_layer=N_LAYER)
    except NotImplementedError:
        print("A 形状：CharGPT.__init__ 未实现  NG")
        return False
    model.eval()
    checks: list[bool] = []

    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, V, (JUDGE_B, JUDGE_T), generator=g)
    y = torch.randint(0, V, (JUDGE_B, JUDGE_T), generator=g)

    try:
        logits, loss = model(x, y)
    except NotImplementedError:
        print("A 形状：模型 forward 未实现  NG")
        return False

    # A 形状与 loss
    ok_shape = tuple(logits.shape) == (JUDGE_B, JUDGE_T, V)
    ok_loss = loss is not None and loss.ndim == 0 and bool(torch.isfinite(loss))
    checks += [ok_shape, ok_loss]
    tag_a = "OK" if (ok_shape and ok_loss) else "NG"
    print(
        f"A 形状 logits {tuple(logits.shape)}（要求 {(JUDGE_B, JUDGE_T, V)}）"
        f"· loss 有限 {ok_loss}  {tag_a}"
    )

    # B 参数量
    n = sum(p.numel() for p in model.parameters())
    per_block = 2 * C + 3 * (C * C + C) + (C * C + C) + 2 * C
    per_block += (HIDDEN * C + HIDDEN) + (C * HIDDEN + C)
    expect = N_LAYER * per_block + V * C + BLOCK * C + 2 * C + V * C
    ok_numel = n == expect
    checks.append(ok_numel)
    tag_b = "OK" if ok_numel else "NG"
    print(
        f"B 参数量 torch {n} vs 手算 {expect}"
        f"（每层 {per_block} × {N_LAYER} + 嵌入/头部）  {tag_b}"
    )

    # C CE 对账（先确认 loss 不是 None —— 类型检查与运行时都需要）
    if loss is None:
        print("C CE 对账：forward 在给了 targets 时没返回 loss  NG")
        return False
    mine = float(loss.item())
    ref = float(F.cross_entropy(logits.reshape(-1, V), y.reshape(-1)).item())
    d_ce = abs(mine - ref)
    ok_ce = d_ce < CE_TOL
    checks.append(ok_ce)
    tag_c = "OK" if ok_ce else "NG"
    print(f"C CE 模型 {mine:.9f} vs 手动 {ref:.9f} 差 {d_ce:.3e}  {tag_c}")

    # D 因果（两路取证）
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % V
    with torch.no_grad():
        logits2, _ = model(x2)
        # 第一次 forward 带了 targets ⇒ logits 挂了计算图；直接 float() 会告警
        d_past = float((logits2[:, :-1] - logits[:, :-1]).abs().max())
        d_last = float((logits2[:, -1] - logits[:, -1]).abs().max())
    ok_past = d_past == 0.0
    ok_last = d_last > 0.0
    checks += [ok_past, ok_last]
    tag_d = "OK" if (ok_past and ok_last) else "NG"
    print(
        f"D 因果：改末位后 过去 {d_past:.3e}（要求严格 0）"
        f"· 末位 {d_last:.3e}（要求 > 0）  {tag_d}"
    )

    # E 位置表与满长前向
    ok_pos = tuple(model.pos_emb.shape) == (BLOCK, C)
    try:
        with torch.no_grad():
            big, _ = model(torch.zeros(1, BLOCK, dtype=torch.int64))
        ok_run = tuple(big.shape) == (1, BLOCK, V)
    except (RuntimeError, IndexError) as exc:
        ok_run = False
        print(f"   E 满长前向抛异常：{type(exc).__name__}: {exc}")
    checks += [ok_pos, ok_run]
    tag_e = "OK" if (ok_pos and ok_run) else "NG"
    print(
        f"E 位置表 {tuple(model.pos_emb.shape)}（要求 {(BLOCK, C)}）"
        f"· 满长 {BLOCK} 前向 {ok_run}  {tag_e}"
    )

    if not all(checks):
        print("logits（首条样本首位置，前 5 个）：", logits[0, 0, :5].tolist())
    return all(checks)


def main() -> None:
    """测试入口（纯胶水）。"""
    results = [judge_model()]
    print(f"判据 {sum(results)}/1 绿")


if __name__ == "__main__":
    main()
