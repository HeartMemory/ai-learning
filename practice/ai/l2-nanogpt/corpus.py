"""L2 · 注意力 —— 关卡 4（块 1/3）：char-level 语料与词表

一条根：关 3 的 block 只证明了「一个 block 形状对、梯度对」，
但它从来没**学过任何东西**。真训练要先有「可预测的东西」：
    char-level 语言模型 = 给定前 T 个字符，预测第 T+1 个字符
    ⇒ 预测目标就藏在原文里（下一个字符），不需要人工标注
    ⇒ 所以整条链路只有一件事要小心：**编码必须可逆**
       （encode 错一个字符，loss 还能降，但你永远生成不出原文）

本块只做数据侧（不碰模型、不碰 loss）：
    语料下载/缓存 → 词表 → encode/decode → train/val 切分 → 随机取 batch

⚠️ 三个口径，改必须写理由：
  ① **切分按位置切，不 shuffle**：语言模型的样本本来就是「连续窗口」，
     9:1 前后切分就是最朴素的 train/val；shuffle 后再切会让 val 里的
     上下文在 train 里出现过完整的下一句，等于泄题。
  ② **offset 上界 = len - block - 1**：要 block+1 个字符才凑得出 x（block 个）
     和 y（右移一位的 block 个）。少减 1 会在最后一个窗口越界。
  ③ **get_batch 必须收 generator**：判据要可复现（同 seed 同 batch），
     否则「loss 变了」永远分不清是模型变了还是数据变了。

判据（本块自定 7 项，A–G；**关卡判据的唯一家 = 根 `README.md` 阶段表**）：
    A 语料：字符数 == EXPECTED_CHARS，且二次调用不重复下载
    B 词表：`len(itos) == 65`（关卡判据原文的口径）+ `stoi`/`itos` 互逆 + 有序
    C 往返：10 段文本 `decode(encode(s)) == s` 逐字相等
    D 切分：`len(train) + len(val) == len(ids)`，`val[0] == ids[n_train]`（按位置切、无重叠）
    E batch 形状：`(B, T)`、int64、值域 `[0, 65)`，`y` 是 `x` 右移一位
    F 抽样覆盖：1000 次采样的窗口起点铺满 `[0, len-block-1]`，且窗口内连续
      —— 用 `torch.arange` 假数据探针（真语料看不出 offset 写死）
    G 可复现：同 seed 两次相同、不同 seed 至少一行不同

题外：为什么不用 nanoGPT 的 `prepare.py`（byte/char 都行）—— 它把
train.bin/val.bin 落盘成 numpy 数组，本关只需要「一个字符串 + 两个张量」，
多一层文件格式反而看不见 encode 到底做了什么。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 D1 [已实现] load_corpus()                     下载 / 缓存，返回全文（机械部分，AI 写）
 D2 [实现]   build_vocab(text)                 (itos, stoi)
 D3 [实现]   encode(text, stoi)                str -> int64 张量
 D4 [实现]   decode(ids, itos)                 int64 张量 -> str
 D5 [实现]   split_ids(ids, frac)              (train, val) 按位置切
 D6 [实现]   get_batch(data, ..., generator)   (x, y)，各 (B, T)
 D7 [已判据] judge_corpus()                    A–G 七项（已写好）

写完一项跑一项：
    uv run python -c "from corpus import load_corpus as f; t=f(); print(len(t))"
    uv run python corpus.py                     # 判据 A–G
════════════════════════════════════════════════════════════════════
"""

import urllib.request
from pathlib import Path

import torch

CORPUS_URL: str = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
"""tiny shakespeare 原文（nanoGPT / char-rnn 同一份：1 115 394 字符、65 个不同字符）。"""

CORPUS_PATH: Path = Path(__file__).with_name("data") / "input.txt"
"""缓存位置：`data/` 不入库（别人的 + 能重下），已在根 `.gitignore` 里。"""

EXPECTED_CHARS: int = 1_115_394
"""语料字符数（2026-11-25 本机实测 == 该值，bytes 也是这个数 —— 全 ASCII）。"""

EXPECTED_VOCAB: int = 65
"""关卡判据原文的「65 词表」。比它少 ⇒ 换行/空格被吃掉；比它多 ⇒ 编码没统一。"""

VAL_FRAC: float = 0.10
"""val 占比。char-level 1.1 M 字符，10% ≈ 11 万字符，足够把 CE 估到小数点后 3 位。"""

BATCH: int = 64
"""一个 batch 的窗口数（真训练也用这个，见 block 3）。"""

BLOCK: int = 128
"""一个窗口的字符数（= 模型能吃到的最大上下文）。"""

N_SNIPPETS: int = 10
"""判据 C 的往返样本数（首/尾/等距抽取 + 一个非连续抽样）。"""

COVER_TRIALS: int = 1000
"""判据 F 的采样次数：够把 1872 个起点铺满，又不是太慢。"""

COVER_MIN_RATIO: float = 0.99
"""判据 F 的覆盖门槛：最大起点要 ≥ 上界 × 它，最小起点要 ≤ 上界 × (1 - 它)。"""

PROBE_LEN: int = 2000
"""判据 F 用的假数据长度（`arange(2000)`），足够长到 offset 写死会被抓出来。"""


def load_corpus() -> str:
    """D1 · 读缓存，没有就下载（已实现，机械部分）。

    用 `Path.exists()` 判断，不做版本校验 —— 语料是死的，不会变。
    """
    if not CORPUS_PATH.exists():
        CORPUS_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(CORPUS_URL, CORPUS_PATH)
    return CORPUS_PATH.read_text(encoding="utf-8")


def build_vocab(text: str) -> tuple[list[str], dict[str, int]]:
    """D2 · 建词表：返回 `(itos, stoi)`，itos 是「下标 → 字符」的表，stoi 反过来。

    提示（两行）：
      ① `itos = sorted(set(text))` —— **必须排序**，不能按出现顺序：
         按出现顺序编的号换个语料就全变，判据 B 的有序性会红，且不可复现。
      ② `stoi = {ch: i for i, ch in enumerate(itos)}` —— 别用 dict(zip()) 反着写，
         反了就是「字符 → 下标」变成「下标 → 字符」的形状错误。

    自检：`len(itos) == 65`、`itos[0] == '\n'`（换行是 ASCII 10，最小）。
    """
    itos = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(itos)}
    return itos, stoi


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """D3 · 把字符串编成 int64 张量（1 维，长度 = 字符数）。

    提示：`torch.tensor([stoi[c] for c in text], dtype=torch.int64)`。
    ⚠️ dtype 必须显式写 int64：默认整数张量也是 int64，但 `torch.tensor` 遇到
       空列表会推成 float32，后面当索引用会报
       `Expected tensor for argument #1 'indices' to have ... Long`.
    """
    return torch.tensor([stoi[c] for c in text], dtype=torch.int64)


def decode(ids: torch.Tensor, itos: list[str]) -> str:
    """D4 · encode 的逆：int64 张量 → 字符串。

    提示：`"".join(itos[i] for i in ids.tolist())`。
    ⚠️ 必须先 `.tolist()`：张量元素是 0 维张量，直接当 list 下标会抛
       `TypeError: only integer tensors of a single element can be converted to an index`
       （或者更糟：被当成 float 索引静默报别的错）。判据 C 就是抓这一条。
    """
    return "".join(itos[i] for i in ids.tolist())


def split_ids(ids: torch.Tensor, frac: float = VAL_FRAC) -> tuple[torch.Tensor, torch.Tensor]:
    """D5 · 按位置切：前 (1-frac) 是 train，后面是 val（不 shuffle，理由见文件头口径 ①）。

    提示（三行）：
      ① `n = int(len(ids) * (1.0 - frac))` —— 用 `int()` 向下取，别用 `round()`
         （四舍五入会让 n 随语料长度抖，判据 D 的等式也就跟着抖）
      ② `return ids[:n], ids[n:]` —— 切片即可，别用 `torch.split`（它要的是长度）。

    自检：两段长度之和 == `len(ids)`，且 `train[-1]` 与 `val[0]` 在原文里是**相邻**字符。
    """
    n = int(len(ids) * (1.0 - frac))
    return ids[:n], ids[n:]


def get_batch(
    data: torch.Tensor,
    batch: int = BATCH,
    block: int = BLOCK,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """D6 · 随机取 `batch` 个长度为 `block` 的窗口，返回 `(x, y)`，各 `(batch, block)`。

    形状链（自己翻译成代码）：
        hi = len(data) - block - 1                      起点上界（口径 ②）
        offs = torch.randint(0, hi, (batch,), generator=generator)
        x = stack([data[o : o + block] for o in offs])    (batch, block)
        y = stack([data[o + 1 : o + block + 1] ...])      (batch, block)  ← 右移一位
        ⇒ 每个位置同时给出「当前字符」和「下一个字符」，这就是语言模型的 (输入, 标签)

    提示：
      · 合法 offset 是 `o ≤ len - block - 1`（这样 `y` 的右端 `o+block+1 ≤ len`）。
        `torch.randint(high)` 取不到 high ⇒ high 至少要 `len - block`；
      · 本文件写 `hi = len - block - 1`，比合法上界**少一位**（o 最大只到 len-block-2），
        属于保守写法：少用一个窗口，无害（2026-11-25 实测 len=10/block=3：o=6 合法，
        o=7 起 y 静默变短；原版本这里写成「len-block 会越界」，是错的，已更正）
      · 真正会炸的是 high 偏大（`len - block + 1` 以上）⇒ y 切片钳到末尾、静默变短，
        随后 `torch.stack` 报 `expects each tensor to be equal size`
      · `generator=None` 时走全局 RNG（判据 G 会显式传 seed 的 generator）。
      · `torch.stack` 而不是 `torch.tensor(list_of_tensors)`（后者会 copy 并告警慢）。

    自检：`x.shape == y.shape == (batch, block)`、`(y[:, :-1] == x[:, 1:]).all()`。
    """
    hi = len(data) - block - 1
    offs = torch.randint(0, hi, (batch,), generator=generator)
    x = torch.stack([data[o : o + block] for o in offs])
    y = torch.stack([data[o + 1 : o + block + 1] for o in offs])
    return x, y

def judge_corpus() -> bool:
    """D7 · 判据 A–G（全绿才 True）；任一函数未实现就当场打印 NG 并返回 False。

    为什么用 `torch.arange` 当假数据（F 项）：真语料里字符会重复，
    「offset 写死 0」和「随机采样」在真数据上看起来一样；换成 0,1,2,…
    每个位置唯一 ⇒ 窗口内部必须严格连续、起点必须铺满全区间，两种错都藏不住。
    """
    checks: list[bool] = []

    # A 语料
    text = load_corpus()
    ok_len = len(text) == EXPECTED_CHARS
    checks.append(ok_len)
    tag_a = "OK" if ok_len else "NG"
    print(f"A 语料 {len(text)} 字符（要求 {EXPECTED_CHARS}）  {tag_a}")

    try:
        itos, stoi = build_vocab(text)
    except NotImplementedError:
        print("B 词表：build_vocab 未实现  NG")
        return False

    # B 词表：大小 / 互逆 / 有序
    ok_size = len(itos) == EXPECTED_VOCAB
    ok_inv = all(stoi[ch] == i for i, ch in enumerate(itos))
    ok_sorted = itos == sorted(itos)
    checks += [ok_size, ok_inv, ok_sorted]
    tag_b = "OK" if (ok_size and ok_inv and ok_sorted) else "NG"
    print(
        f"B 词表 {len(itos)} 个（要求 {EXPECTED_VOCAB}）· 互逆 {ok_inv} · 有序 {ok_sorted}  {tag_b}"
    )

    try:
        ids = encode(text, stoi)
    except NotImplementedError:
        print("C 往返：encode 未实现  NG")
        return False

    # C 往返：首/尾/等距 9 段 + 1 段非连续抽样（text[::997] 抓「按下标算错」）
    n = len(text)
    starts = [n * k // (N_SNIPPETS - 1) for k in range(N_SNIPPETS - 1)]
    snippets = [text[s : s + 80] for s in starts] + [text[::997][:60]]
    try:
        round_trip = [decode(encode(s, stoi), itos) for s in snippets]
    except NotImplementedError:
        print("C 往返：decode 未实现  NG")
        return False
    bad = [i for i, (a, b) in enumerate(zip(snippets, round_trip, strict=True)) if a != b]
    ok_rt = not bad
    checks.append(ok_rt)
    tag_c = "OK" if ok_rt else "NG"
    detail_c = "全相等" if ok_rt else f"第 {bad} 段不等"
    print(f"C 往返 {len(snippets)} 段逐字相等：{detail_c}  {tag_c}")
    if not ok_rt:
        i = bad[0]
        print(f"   期望 {snippets[i]!r}")
        print(f"   实得 {round_trip[i]!r}")

    try:
        train, val = split_ids(ids)
    except NotImplementedError:
        print("D 切分：split_ids 未实现  NG")
        return False

    # D 切分：长度守恒 + 按位置切（val 起点紧跟 train 末尾）
    n_train = int(n * (1.0 - VAL_FRAC))
    ok_total = len(train) + len(val) == n
    ok_pos = int(val[0]) == int(ids[n_train]) and int(train[-1]) == int(ids[n_train - 1])
    checks += [ok_total, ok_pos]
    tag_d = "OK" if (ok_total and ok_pos) else "NG"
    print(
        f"D 切分 train {len(train)} + val {len(val)} = {len(train) + len(val)}（要求 {n}）"
        f" · val 起点对齐 {ok_pos}  {tag_d}"
    )

    try:
        x, y = get_batch(train, batch=4, block=16, generator=torch.Generator().manual_seed(0))
    except NotImplementedError:
        print("E batch：get_batch 未实现  NG")
        return False

    # E 形状 / dtype / 值域 / 右移一位
    ok_shape = tuple(x.shape) == (4, 16) and tuple(y.shape) == (4, 16)
    ok_dtype = x.dtype == torch.int64 and y.dtype == torch.int64
    ok_range = int(x.min()) >= 0 and int(x.max()) < len(itos) and int(y.max()) < len(itos)
    ok_shift = bool((y[:, :-1] == x[:, 1:]).all())
    checks += [ok_shape, ok_dtype, ok_range, ok_shift]
    tag_e = "OK" if (ok_shape and ok_dtype and ok_range and ok_shift) else "NG"
    print(
        f"E batch x{tuple(x.shape)} y{tuple(y.shape)} · int64 {ok_dtype} · 值域 {ok_range}"
        f" · y 是 x 右移一位 {ok_shift}  {tag_e}"
    )

    # F 覆盖 + 窗口连续（arange 探针）
    probe = torch.arange(PROBE_LEN, dtype=torch.int64)
    g = torch.Generator().manual_seed(1)
    firsts: list[int] = []
    ok_contig = True
    ok_cross = True
    hi = PROBE_LEN - BLOCK - 1
    for _ in range(COVER_TRIALS):
        px, py = get_batch(probe, generator=g)
        ok_contig = ok_contig and bool((px[:, 1:] == px[:, :-1] + 1).all())
        ok_cross = ok_cross and bool((py[:, -1] == px[:, -1] + 1).all())
        firsts.append(int(px[0, 0]))
    lo, hi_seen = min(firsts), max(firsts)
    ok_cover = lo <= hi * (1.0 - COVER_MIN_RATIO) and hi_seen >= hi * COVER_MIN_RATIO
    checks += [ok_contig, ok_cross, ok_cover]
    tag_f = "OK" if (ok_contig and ok_cross and ok_cover) else "NG"
    print(
        f"F 采样起点 [{lo}, {hi_seen}]（上界 {hi}）· 窗口连续 {ok_contig}"
        f" · 跨窗口右移 {ok_cross}  {tag_f}"
    )

    # G 可复现：同 seed 相同、不同 seed 至少一行不同
    a1 = get_batch(train, generator=torch.Generator().manual_seed(7))[0]
    a2 = get_batch(train, generator=torch.Generator().manual_seed(7))[0]
    a3 = get_batch(train, generator=torch.Generator().manual_seed(8))[0]
    ok_same = bool((a1 == a2).all())
    ok_diff = not bool((a1 == a3).all())
    checks += [ok_same, ok_diff]
    tag_g = "OK" if (ok_same and ok_diff) else "NG"
    print(f"G 同 seed 相同 {ok_same} · 换 seed 不同 {ok_diff}  {tag_g}")

    return all(checks)


def main() -> None:
    """测试入口（纯胶水）：本块只有 1 条判据组（A–G 全在里面）。"""
    results = [judge_corpus()]
    print(f"判据 {sum(results)}/1 绿")


if __name__ == "__main__":
    main()
