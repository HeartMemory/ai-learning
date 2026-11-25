"""L2 · 注意力 —— 关卡 4（块 3c/3c）：生成与重复片段检测

一条根：判据 ② 的第二句是「生成 200 字内**最长重复片段 ≤ 20 字符**」。
    ⇒ 这句话测的不是 loss，而是**塌缩**：模型是否退化成一味循环几个字。
    ⇒ loss 低的模型也可能塌缩（尤其 char-level 小模型 + 贪心解码），
      所以要单独量它，不能拿 CE 代替。

⚠️ 三个口径，改必须写理由：
  ① **「最长重复片段」定义 = 最长重复子串（至少出现两次、允许重叠）**。
     不定义清楚这句话就是空的：是「连续两个相同子串」？「周期性循环」？
     这里取最常用的 LRS（longest repeated substring），O(n²) DP 实现，
     n = 200 时 4 万次比较，毫秒级。
  ② **解码参数提前定，不按结果回调**：temperature 0.8 + top_k 40。
     理由：贪心（argmax）在小模型上极易陷入死循环（"the the the…"），
     纯采样（无 top_k）会引入太多低概率噪声字符。0.8/40 是常见的折中。
     ⚠️ 若结果不达标，**改口径必须写理由**，不能"调到过为止"。
  ③ **同一个 seed 必须给同一个输出**：`torch.multinomial` 收 generator，
     否则两次跑生成不同文本，判据没法复现（与块 1 的 get_batch 同一条纪律）。

判据（本块自定 3 项，A–C）：
    A 形状：生成恰好 GEN_LEN 个字符，且全部来自词表
    B 塌缩：最长重复子串长度 ≤ REPEAT_LIMIT（20），并打印那段内容与位置
    C 可复现：同 seed 两次相同、换 seed 不同

════════════════════════════════════════════════════════════════════
 全部由 AI 写（用户 2026-11-25 明确要求代写；讲解见对话）
 跑法：uv run python sample.py     # 需要先跑过 train.py 生成 ckpt.pt
════════════════════════════════════════════════════════════════════
"""

from pathlib import Path

import torch
import torch.nn.functional as F

from corpus import BLOCK
from model import CharGPT

CKPT: Path = Path(__file__).with_name("ckpt.pt")
"""块 3b 训练产物（`.gitignore` 已排除 `*.pt`）。"""

GEN_LEN: int = 200
"""判据 ② 原文的「200 字内」—— 生成长度就是 200 个字符。"""

TEMPERATURE: float = 0.8
"""采样温度（口径 ②）。<1 让分布更尖，>1 更平；0.8 略偏保守。"""

TOP_K: int = 40
"""只在前 k 个候选里采样。0 表示不截断（纯采样）。"""

REPEAT_LIMIT: int = 20
"""判据 B 的上限：最长重复子串不得超过 20 字符。"""

PROMPT: str = "\n"
"""起始种子（一个换行）。之所以不用空串：模型的输入至少要 1 个 token；
用换行等于让它从「新起一段」开始写，最容易生成连贯文本。"""


def load_model() -> tuple[CharGPT, list[str]]:
    """读 checkpoint，返回 (model, itos)。"""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT, map_location=dev, weights_only=True)
    itos: list[str] = ckpt["itos"]
    model = CharGPT(vocab_size=len(itos)).to(dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, itos


@torch.no_grad()
def generate(
    model: CharGPT,
    itos: list[str],
    max_new: int = GEN_LEN,
    temperature: float = TEMPERATURE,
    top_k: int = TOP_K,
    seed: int | None = None,
) -> str:
    """自回归生成：每次只喂**最后 BLOCK 个 token**，采一个字符接在后面。

    为什么每次都要 `idx[:, -BLOCK:]`：位置表只有 BLOCK 行，长度超了就取不到位置
    向量（`pos_emb[:T]` 会静默截断成 BLOCK 行 ⇒ 形状对不上而报错）。
    这个「滑动窗口」就是上下文长度的真实含义：**模型只看得到最近 BLOCK 个字符**。

    ⚠️ `seed` 而不是传 `torch.Generator()`：generator 是**绑定设备**的 ——
       probs 在 cuda 上时传 CPU generator 会报
       `Expected a 'cuda' device type for generator but found 'cpu'`（实测踩过）。
       所以在函数内部按模型所在设备建 generator。
    """
    dev = next(model.parameters()).device
    stoi = {ch: i for i, ch in enumerate(itos)}
    idx = torch.tensor([[stoi[ch] for ch in PROMPT]], dtype=torch.int64, device=dev)
    gen = torch.Generator(device=dev)
    if seed is not None:
        gen.manual_seed(seed)

    for _ in range(max_new):
        logits, _ = model(idx[:, -BLOCK:])
        logits = logits[:, -1, :] / temperature  # 只关心最后一个位置的下一个字符
        if top_k > 0:
            kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
            logits = logits.masked_fill(logits < kth, float("-inf"))
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, num_samples=1, generator=gen)
        idx = torch.cat([idx, nxt], dim=1)

    out = idx[0, 1:].tolist()  # 去掉起始的那个换行
    return "".join(itos[i] for i in out)


def longest_repeat(s: str) -> tuple[int, str, int]:
    """最长重复子串（允许重叠）：返回 (长度, 片段, 起始位置)。

    DP：`dp[i][j]` = 从 s[i]、s[j] 出发的最长公共前缀长度（i ≠ j）。
        dp[i][j] = 1 + dp[i+1][j+1]  当 s[i] == s[j]，否则 0
    倒序填表即可（`i`、`j` 都从大到小）。n=200 ⇒ 200×200 表，毫秒级。

    ⚠️「允许重叠」意味着 "aaaa" 的最长重复是 "aaa"（长度 3），不是 "aa"。
       这个口径要写在判据里，否则同一个输出能算出不同的数。
    """
    n = len(s)
    dp = [[0] * (n + 1) for _ in range(n + 1)]
    best = 0
    best_i = 0
    for i in range(n - 1, -1, -1):
        row, nxt_row = dp[i], dp[i + 1]
        for j in range(n - 1, -1, -1):
            if i != j and s[i] == s[j]:
                row[j] = 1 + nxt_row[j + 1]
                if row[j] > best:
                    best = row[j]
                    best_i = i
    return best, s[best_i : best_i + best], best_i


def judge_sample() -> bool:
    """判据 A–C（全绿才 True）。"""
    if not CKPT.exists():
        print(f"A 生成：找不到 {CKPT.name}，先跑 `uv run python train.py`  NG")
        return False

    model, itos = load_model()
    checks: list[bool] = []

    # 生成两次（同 seed / 换 seed）
    text1 = generate(model, itos, seed=1337)
    text2 = generate(model, itos, seed=1337)
    text3 = generate(model, itos, seed=2026)

    # A 形状：长度 == GEN_LEN，且字符都在词表里
    ok_len = len(text1) == GEN_LEN
    ok_vocab = all(ch in itos for ch in text1)
    checks += [ok_len, ok_vocab]
    tag_a = "OK" if (ok_len and ok_vocab) else "NG"
    print(f"A 生成 {len(text1)} 字符（要求 {GEN_LEN}）· 全在词表 {ok_vocab}  {tag_a}")
    print(f"   样本：{text1!r}")

    # B 塌缩：最长重复子串
    size, frag, pos = longest_repeat(text1)
    ok_rep = size <= REPEAT_LIMIT
    checks.append(ok_rep)
    tag_b = "OK" if ok_rep else "NG"
    print(
        f"B 最长重复片段 {size} 字符（上限 {REPEAT_LIMIT}）"
        f"· 位置 {pos} · 内容 {frag!r}  {tag_b}"
    )

    # C 可复现：同 seed 相同、换 seed 不同
    ok_same = text1 == text2
    ok_diff = text1 != text3
    checks += [ok_same, ok_diff]
    tag_c = "OK" if (ok_same and ok_diff) else "NG"
    print(f"C 同 seed 相同 {ok_same} · 换 seed 不同 {ok_diff}  {tag_c}")

    if ok_same and ok_diff:
        print(f"   另一 seed 样本：{text3!r}")
    return all(checks)


def main() -> None:
    """测试入口（纯胶水）。"""
    results = [judge_sample()]
    print(f"判据 {sum(results)}/1 绿")


if __name__ == "__main__":
    main()
