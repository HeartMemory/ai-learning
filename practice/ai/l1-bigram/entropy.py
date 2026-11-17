"""L1 · 词与概率 —— 关卡 2：交叉熵 / 困惑度（训练集严格序 + 留出集证据）

目的（一条根）：交叉熵 = 「模型给正确答案分配的概率，取负再取 log2」，再按字平均。
    · 单位 bits/char：平均每个字花掉几个 bit —— 越小说明模型越省
    · 均匀基线 log2(V)：完全不认识字，每一步在 V 个字里瞎猜
    · unigram    ：只知道「这个字本身多常见」，不看前一个字
    · bigram     ：知道前一个字（关卡 1 的条件概率表）
    三者必须严格递减：知道得越多，每个字越不惊讶。

生活模型：天气预报。你说下雨 90%，结果下了 → 惊讶度 -log2(0.9) ≈ 0.15 bit；
你说 1%，结果下了 → -log2(0.01) ≈ 6.64 bit。交叉熵 = 平均惊讶度。
困惑度 = 2^CE = 「模型每一步平均在几个字里犹豫」——bigram 的 CE 1.3 bit ⇒ 约 2.5 个字。

这一关不训任何东西，只把概率换算成一个可比的数。平滑在关卡 4。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T1 [实现] split_corpus(text, holdout)               前 80% 训练 / 后 20% 留出（按位置切）
 T2 [实现] unigram_probs(text)                       P(字) = 该字次数 / 总字数
 T3 [实现] ce_unigram(train, eval_text, skip_unseen) 一元模型在 eval_text 上的 CE（bits/char）
 T4 [实现] ce_bigram(counts, eval_text, skip_unseen) 二元模型在 eval_text 上的 CE（bits/char）
 T5 [判据] judge_train_order()                       判据 3：训练集严格序 + 两路互证
 T6 [判据] judge_holdout()                           判据 4：留出集上必为 inf + 只算见过的对会升高
════════════════════════════════════════════════════════════════════
 写完一项跑一项（别等六项全写完）：
   uv run python -c "from entropy import judge_train_order as f; print(f())"
   uv run python -c "from entropy import judge_holdout as f; print(f())"
 全绿后再跑总入口：uv run python entropy.py

跑法：进本目录 → uv sync（0 依赖）→ uv run python entropy.py
文档依据（2026-11-17 查证）：
  docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html
      交叉熵 = LogSoftmax + NLLLoss（reduction 默认 'mean'）—— 就是「取对数的概率、取负、再平均」
  scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html
      log loss 就是交叉熵；用**自然对数**（nats）；并把概率 clip 到 [eps, 1-eps] 以免 log(0)
      ⇒ 本关一律 log2（bits/char），且**不 clip**：0 概率就是 inf，那是真信息（关卡 4 治）
  docs.python.org/3/library/math.html
      log2(0) 抛 **ValueError**（不会返回 -inf）⇒ inf 那条分支必须自己 if 判
════════════════════════════════════════════════════════════════════
"""

import itertools
import math

from bigram import build_counts, row_totals
from corpus import CORPUS

TEXT: str = CORPUS
HOLDOUT: float = 0.2
"""后 20% 当留出集（考试卷）。改这个数，所有判据数字会跟着变。"""


def split_corpus(text: str, holdout: float = HOLDOUT) -> tuple[str, str]:
    """T1 · 切训练集 / 留出集 —— 返回 (train, held)。

    规则：**按位置切，不打乱**。切点 = `int(len(text) * (1 - holdout))`；
    train = 前面那截，held = 后面那截（连着的，因为语言模型吃的是连续文本）。

    判据：① `train + held == text`
         ② `len(held) == len(text) - len(train)`
         ③ `len(held) > 0`（切出空卷子的判据全是假绿）
    """
    spilt_index = int(len(text) * (1 - holdout))
    train = text[:spilt_index]
    held = text[spilt_index:]
    assert train + held == text, "train + held != text"
    assert len(held) == len(text) - len(train), "len(held) != len(text) - len(train)"
    assert len(held) > 0, "len(held) == 0, 留出集为空"
    return train, held


def unigram_probs(text: str) -> dict[str, float]:
    """T2 · 一元概率表：P(字) = 该字在 text 里出现次数 / len(text)。

    判据：① 每个值 > 0
         ② `abs(sum(值) - 1.0) < 1e-12`（归一）
         ③ 键集合 == set(text)（没出现的字不许凭空给概率）

    提示：`collections.Counter` 一行数完；分母是 `len(text)`，不是「不同字的个数」。
    """
    counts = {}
    for char in text:
        if char in counts:
            counts[char] += 1
        else:
            counts[char] = 1
    total_chars = len(text)
    probs = {char: count / total_chars for char, count in counts.items()}
    assert all(value > 0 for value in probs.values()), "存在非正概率"
    assert abs(sum(probs.values()) - 1.0) < 1e-12, "概率和不为 1"
    assert set(probs.keys()) == set(text), "键集合与 text 的字符集合不一致"
    return probs


def ce_unigram(train: str, eval_text: str, *, skip_unseen: bool = False) -> float:
    """T3 · 一元模型的交叉熵（bits/char）：`-(1/N) * Σ log2 P(字)`。

    概率表来自 **train**（训练集），被评估的是 **eval_text**（可以是 train 自己，也可以是留出集）。
    N = `len(eval_text)` —— 每个字都是一个预测位置。

    两种口径（由 `skip_unseen` 决定，**默认不许悄悄跳过**）：
      · `skip_unseen=False`：eval_text 里出现 train 没见过的字 → 返回 `math.inf`
        （不是抛异常、不是 clip、不是拿 1e-12 顶替 —— 那三种都是在掩盖症状）
      · `skip_unseen=True`：跳过这些位置，**分母 = 真正算到的位置数**（不是 len(eval_text)）；
        一个都没算到 → 返回 `math.nan`

    判据：① `ce_unigram(train, train)` 有限且 > 0
         ② 用 train 训练、在 train 上评估时，结果 == 分布自身的熵 `-Σ p log2 p`（T5 互证）
         ③ 留出集里只要有新字，默认口径必须返回 `math.inf`

    提示：`math.log2(0)` 会抛 `ValueError`，所以 inf 那条分支要**在算 log 之前**判掉。
    """
    probs = unigram_probs(train)
    if skip_unseen:
        for char in eval_text:
            if char not in probs:
                eval_text = eval_text.replace(char, "")
        if len(eval_text) == 0:
            return math.nan
    else:
        for char in eval_text:
            if char not in probs:
                return math.inf
    ce = -sum(math.log2(probs[char]) for char in eval_text)
    ce /= len(eval_text)
    return ce


def ce_bigram(
    counts: dict[tuple[str, str], int], eval_text: str, *, skip_unseen: bool = False
) -> float:
    """T4 · 二元模型的交叉熵（bits/char）：`-(1/M) * Σ log2 P(下一个字 | 前一个字)`。

    参数：
      · `counts`：**训练集**上的相邻字对计数（关卡 1 的 `build_counts`，别在这里重数）
      · `eval_text`：被评估的文本
      · M = `len(eval_text) - 1`（第一个字没有上下文，不参与打分）

    P 从 counts 现算：`counts[(prev, next)] / row_totals(counts)[prev]`。

    `skip_unseen` 的两种口径同 T3。这里「未见」有两种形态，**都算未见**：
      · 这对比从没出现过（对不在 counts 里）
      · 前一个字从没当过 context（prev 不在 row_totals 里 —— 语料最后一个字就是这样）

    判据：① `ce_bigram(build_counts(train), train)` 有限，且 **< `ce_unigram(train, train)`**
             （bigram 知道更多，必须更省 —— 反过来就是实现错了）
         ② 训练集上算、留出集上评，只要有未见的对，默认口径返回 `math.inf`
         ③ `skip_unseen=True` 时，留出集上得到有限数（分母 = 见得过的对的数量）

    提示：这是本关唯一会踩 `log2(0)` 的函数；先查 P 是否为 0，再取 log。
    提示：`row_totals` 在 `bigram.py` 里（关卡 1 写好的），顶部加一行
         `from bigram import row_totals` 就能用，别在这里重新数一遍。
    """
    totals = row_totals(counts)
    pairs = list(itertools.pairwise(eval_text))
    ce_sum = 0.0
    if len(pairs) == 0:
        return math.nan  # 没有相邻对，无法计算交叉熵
    if skip_unseen:
        valid_pairs = []
        for prev_char, next_char in pairs:
            if prev_char in totals and (prev_char, next_char) in counts:
                valid_pairs.append((prev_char, next_char))
        if not valid_pairs:
            return math.nan
        pairs = valid_pairs
    else:
        for prev_char, next_char in pairs:
            if prev_char not in totals or (prev_char, next_char) not in counts:
                return math.inf
    for prev_char, next_char in pairs:
        prob = counts[(prev_char, next_char)] / totals[prev_char]
        ce_sum += -math.log2(prob)
    ce = ce_sum / len(pairs)
    return ce


def judge_train_order() -> bool:
    """T5 · 判据 3：训练集上的严格序 `log2(V) > CE_unigram > CE_bigram`（bits/char）。

    ① 归一：`unigram_probs(train)` 的值和为 1.0（±1e-12）
    ② 两路互证同一个量（防「公式抄错但数字看着合理」）：
       · 逐字路：`-(1/N) Σ log2 P(c)`，c 遍历 train 的每个字
       · 频次路：`-Σ p log2 p`，p 遍历概率表的值
       两条路的差 < 1e-12，且都等于 `ce_unigram(train, train)`
    ③ 严格序：三个数都有限，且 `log2(V) > CE_uni > CE_bi`
       · V 取 **train 的**词表大小（不是整个语料的 180 —— 留出集里的新字训练时根本没见过）
       · 这条序是数学上必然的：均匀分布是「最不惊讶」的上界，bigram 是 unigram 的细分

    通过后 print 三行的 CE 与困惑度（`2 ** CE`），抄进本目录 README 的「实测数字」表。
    """
    train, _held = split_corpus(TEXT)
    vocab = sorted(set(train))
    baseline = math.log2(len(vocab))

    probs = unigram_probs(train)
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-12, f"unigram 概率和 {total} != 1.0"
    assert set(probs) == set(train), "unigram 键集合与 train 的字符集合不一致"
    assert min(probs.values()) > 0.0, f"unigram 出现非正概率 {min(probs.values())}"

    ce_uni = ce_unigram(train, train)
    ce_bi = ce_bigram(build_counts(train), train)

    by_chars = -sum(math.log2(probs[c]) for c in train) / len(train)
    by_freq = -sum(p * math.log2(p) for p in probs.values())
    assert abs(by_chars - by_freq) < 1e-12, f"两路互证失败：逐字 {by_chars} vs 频次 {by_freq}"
    assert abs(ce_uni - by_chars) < 1e-12, f"CE_unigram {ce_uni} != 逐字路 {by_chars}"

    # ② b（2026-11-17 补）bigram 也要两路互证 —— 否则「分母写成 len(eval_text)」这类错抓不到：
    #    · 逐位置路：`ce_bigram(counts, train)`
    #    · 按对路：  `-Σ (k/M) log2(k / row_totals[a])`，M = len(train) - 1
    #    两者是恒等式：前者把每个位置各算一次，后者按每种对的出现次数加权，再各自除以 M
    counts_tr = build_counts(train)
    totals_tr = row_totals(counts_tr)
    span = len(train) - 1
    by_pair_bi = -sum(
        (k / span) * math.log2(k / totals_tr[prev]) for (prev, _next), k in counts_tr.items()
    )
    assert abs(ce_bi - by_pair_bi) < 1e-12, (
        f"CE_bigram 两路互证失败：逐位置 {ce_bi} vs 按对加权 {by_pair_bi}"
        "（分母写成 len(eval_text) / 拿全局分布当条件概率 / 按 next 聚合，都会在这里炸）"
    )

    for name, value in (("log2(V)", baseline), ("CE_unigram", ce_uni), ("CE_bigram", ce_bi)):
        assert math.isfinite(value), f"{name} 不是有限数：{value}"
    assert ce_bi < ce_uni < baseline, (
        f"严格序不成立：log2(V)={baseline:.4f} / CE_uni={ce_uni:.4f} / CE_bi={ce_bi:.4f}"
    )

    print(
        f"V(train)={len(vocab)} | 基线 log2(V)={baseline:.4f} bit "
        f"(ppl {2**baseline:.2f}) | CE_unigram={ce_uni:.4f} bit (ppl {2**ce_uni:.2f}) | "
        f"CE_bigram={ce_bi:.4f} bit (ppl {2**ce_bi:.2f})"
    )
    return True


def judge_holdout() -> bool:
    """T6 · 判据 4：留出集 —— 先证明「MLE 在留出集上必死」，再报一个有限数当稀疏的证据。

    ① 独立数期望（不经过被测函数）：
       · `unseen_chars = set(held) - set(train)`，必须非空
       · `unseen_pairs = [p for p in 相邻对(held) if p not in counts_train]`，必须非空
       两个都空 ⇒ 这份语料/这个切点证明不了任何事，判据立不住（换切点或换语料）
    ② 默认口径（不跳过）在留出集上 **两个模型都返回 `math.inf`**：
       · `ce_unigram(train, held) == inf`（因为留出集里有训练没见过的新字）
       · `ce_bigram(counts_train, held) == inf`（因为留出集里有没见过的字对）
       ⇒ 这就是「不加平滑的模型在真实文本上困惑度直接报废」，关卡 4 的全部动机
    ③ 只算「训练时见过的对」时得到有限数，且 **高于训练集上的同一个量**：
       · `ce_bigram(counts_train, held, skip_unseen=True)` 有限
         且 > `ce_bigram(counts_train, train)`
       ⇒ 这是**稀疏 / 过拟合**的证据：模型在自己背过的卷子上更省
       （不同 unigram 的留出集数字：它也必然是 inf，同一件事，不用各报一遍）

    通过后 print：未见字数 / 未见对数与比例 / 两个 CE / 对应困惑度。
    """
    train, held = split_corpus(TEXT)
    counts = build_counts(train)

    unseen_chars = sorted(set(held) - set(train))
    held_pairs = list(itertools.pairwise(held))
    unseen_pairs = [pair for pair in held_pairs if pair not in counts]
    assert unseen_chars, "留出集没有训练时没见过的新字 —— 这条判据立不住"
    assert unseen_pairs, "留出集没有没见过的字对 —— 这条判据立不住"

    ce_uni_held = ce_unigram(train, held)
    ce_bi_held = ce_bigram(counts, held)
    assert ce_uni_held == math.inf, f"留出集有新字，CE_unigram 应为 inf，实际 {ce_uni_held}"
    assert ce_bi_held == math.inf, f"留出集有未见对，CE_bigram 应为 inf，实际 {ce_bi_held}"

    # ④ 边界（专门抓「删字再造对」这个错法）：
    #    tiny_train = "adbc" → counts 里**恰好有** (a, d)
    #    tiny_eval  = "axd"  → 相邻对只有 (a, x) 与 (x, d)：前者未见，后者连上下文都没有
    #    ⇒ 一对可用都没有，skip 口径必须返回 nan
    #    错法「先把 x 删掉再收相邻对」会得到 "ad" → 撞上真实存在的 (a, d) → 错误地算出有限值
    tiny_counts = build_counts("adbc")
    tiny_ce = ce_bigram(tiny_counts, "axd", skip_unseen=True)
    assert math.isnan(tiny_ce), (
        f'"axd" 里没有一对可用（a→x 未见、x→d 无上下文），应得 nan，实际 {tiny_ce!r}'
        "—— 把被跳过的字删掉再造相邻对，会凭空造出 (a, d) 这个原文里不存在的对"
    )
    assert ce_bigram(tiny_counts, "axd") == math.inf, "默认口径下 axd 有未见对，应返回 inf"

    # ⑤ 边界：**一对都没有**（空串 / 单字）→ 两种口径都必须 nan
    #    这里抓的是「除零」：分母是 len(pairs)，pairs 为空时 0/0 抛 ZeroDivisionError
    #    （`if pairs == 0:` 这种写法永远不成立 —— `[] == 0` 是 False，必须写 `if not pairs:`）
    for empty in ("", "a"):
        for skip in (False, True):
            got = ce_bigram(tiny_counts, empty, skip_unseen=skip)
            assert math.isnan(got), f"eval_text={empty!r} skip={skip} 应得 nan，实际 {got!r}"

    ce_bi_train = ce_bigram(counts, train)
    ce_bi_seen = ce_bigram(counts, held, skip_unseen=True)
    assert math.isfinite(ce_bi_seen), f"只算见过的对时应得到有限数，实际 {ce_bi_seen}"
    assert ce_bi_seen > ce_bi_train, (
        f"留出集（只算见过的对）{ce_bi_seen:.4f} 未高于训练集 {ce_bi_train:.4f} —— "
        "要么切点/分母写错了，要么这份语料证明不了稀疏"
    )

    print(
        f"留出集 {len(held)} 字 | 新字 {len(unseen_chars)} | "
        f"未见对 {len(unseen_pairs)}/{len(held_pairs)}"
        f"（{len(unseen_pairs) / len(held_pairs):.1%}）| 默认口径 CE_uni=inf CE_bi=inf | "
        f"只算见过的对：CE_bigram 留出 {ce_bi_seen:.4f} > 训练 {ce_bi_train:.4f} bit"
        f"（ppl {2**ce_bi_seen:.2f} > {2**ce_bi_train:.2f}）"
    )
    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的，不是硬编码字符串。"""
    results = [
        judge_train_order(),
        judge_holdout(),
    ]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
