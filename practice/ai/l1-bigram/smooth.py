"""L1 · 词与概率 —— 关卡 4：平滑（把「概率 0」变成「概率小但 > 0」）

目的（一条根）：MLE 说「没见过的对 = 概率 0」，而 0 的负 log 是无穷 ⇒ 留出集 CE 直接 `inf`。
    平滑只做一件事：给每个候选后继**垫一点**概率，再从已见事件上**按比例扣回来**。
    · 分子 `c(prev,next) + k`    —— 每个候选都加 k（未见过的从 0 变 k）
    · 分母 `c(prev) + k|V|`      —— 加了多少加量就还回多少，保证 Σ = 1
    · 等价写法（插值）`P = C/(C+k|V|) · P_MLE + k|V|/(C+k|V|) · 1/|V|`
      ⇒ add-k 就是把「我的经验」朝「完全无知（均匀）」拉一把，C 越小拉得越狠
    · k = 1 叫 Laplace / add-1；k 取小数（0.01 / 0.1）叫 add-k / Lidstone
    代价是**必然的**：已见事件被压低 ⇒ 训练集 CE 一定升高。平滑 = 拿训练集拟合度换「不崩溃」。

生活模型：分蛋糕。原来只按「谁平时抢得多」分，没抢过的人一口没有；
    现在给在场每个人先各垫一小块，垫掉的量从大家的份额里按同一比例抽回来 —— 总量还是一整个蛋糕。
    k 越大垫得越多：add-1 相当于给全城每个人都留一块，蛋糕立刻被摊薄（大词表下就是灾难）。

★ 本关三个坑（都是判据必须钉住的）：
    ① **`inf` 有两种病，平滑只治得了一半**（实测：84 个位置里）
       · 26 个位置是「行存在、缺列」（c(prev,next)=0）      → add-k 治
       · 41 个位置里有**生字**（不在 V 里，24 个在 next 侧）  → 垫过的 0 都不存在，必须改口径
       ⇒ 所以本关用**闭词表**：V = 训练集 158 个字 + `<UNK>` = **159**，评估前折叠生字。
    ② **V 必须是 159**（158 + UNK），不是 158，更不是全语料 180。V 直接进分母：
       158 ⇒ 生字位置概率仍是 0（还是 inf）；180 ⇒ 词表塞了训练时根本不存在的字（偷看字表）。
    ③ **C(prev) = 0 的 context 也会出现**（`<UNK>` 当「前一个字」时必然如此）：
       分母 = `0 + k|V|`，概率 = `1/|V|`（均匀）。add-k 顺手治好了「整行不存在」这个病；
       但也正因分母含 C(prev)，写成 `c(prev)` 直接除会 `ZeroDivisionError`。

★ 折叠必须返回 **token 列表**，不能返回字符串（实测教训）：
    `<UNK>` 是 5 个字符的**一个 token**。若用 `"".join(...)` 折叠，85 个 token 会变成
    **181 个字符**，位置数从 84 变成 180 —— 分母一变，所有 CE 都作废（正是老坑 M6 的变种）。
    ⇒ `fold_unk` 返回 `list[str]`；收相邻对用 `itertools.pairwise(列表)`。

★ 折叠的代价（可比性要写清，不许糊住）：
    折叠改变了 held 的 token 序列（22 个生字 → 同一个 `<UNK>`），所以**平滑后的 CE 不能与
    昨天未折叠的「只算见过的对 1.6935」直接并排**。真正可比的只有：同一份**折叠后**的 held、
    同一批 84 个位置、同一个 V=159、同一个 log2。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T0 [实现] fold_unk(text, vocab) -> list[str]     生字 → `<UNK>`（**token 数不变**）
 T1 [实现] smooth_row(counts, prev, vocab, k)     单个 context 的**完整**平滑分布（|V| 个键）
 T2 [实现] smooth_ce_bigram(counts, vocab, k, eval_text)
        「先折叠、再平滑」的交叉熵（bits/char），**不跳过任何位置**
 T3 [判据] judge_smoothing_math()       判据 1：行和为 1 / 未见 = k/(C+k|V|) / 均匀兜底 / k 单调
 T4 [判据] judge_comparable_numbers()   判据 2：折叠四条 + MLE 仍 inf + 平滑后有限 + 训练集必升
════════════════════════════════════════════════════════════════════
 写完一项跑一项（别等四项全写完）：
   uv run python -c "from smooth import judge_smoothing_math as f; print(f())"
   uv run python -c "from smooth import judge_comparable_numbers as f; print(f())"
 全绿后再跑总入口：uv run python smooth.py

跑法：进本目录 → uv sync（0 依赖）→ uv run python smooth.py
      ⚠️ 两个判据函数写出来之前，`uv run ruff check .` 会报 F401（unused import），
         实现一落地就消失，不是你的错。

必须能被判据抓住的错法（写完自己拿这 7 条对照，这也就是变异测试的种子）：
  M1 分母漏 `k|V|`                      → 行和 < 1，概率和 = C/(C+kV)
  M2 分母的 V 用全语料 180                → 偷看未来字表，数字变小、"更好看"
  M3 分子漏 k（只对未见对加 k）           → 行和 > 1，「垫」的量与分母不守恒
  M4 未见对仍返回 0                       → 留出集 CE 还是 inf，什么都没治
  M5 C(prev)=0 时不走均匀兜底（直接 c/C） → ZeroDivisionError
  M6 跳过位置后分母没跟着变               → CE 假低（关卡 2 老坑重演）
  M7 忘了折叠（直接拿原始 held 算）        → 24 个生字位置 P=0 ⇒ CE 还是 inf

════════════════════════════════════════════════════════════════════
文档依据（2026-11-19 查证，三条来源交叉）
════════════════════════════════════════════════════════════════════
  SLP3 Ch.3 *N-gram Language Models*（web.stanford.edu/~jurafsky/slp3/3.pdf）
      add-k 公式 = (c(w_{n-1}w_n) + k)/(c(w_{n-1}) + k|V|)；k=1 即 add-1；
      add-1 对 n-gram **过度平滑**，实用取小数 k。
      ⚠️ 诚实记录：该 PDF 抓取后是 FlateDecode 二进制流、正文没解码出来，
         公式与结论是按通行版本复述 + 下面两条可解析来源交叉确认的；页码需本地 PDF 核对。
  mbrenndoerfer.com《Language AI Handbook》§9.4 Add-k Smoothing（2025-03-22 更新）
      原文：分母加 k|V| 的唯一目的就是 Σ_w P = 1；
            「从已见事件移走的质量」= k(|V| − V_seen)/(C(w_{n-1}) + k|V|)，随 |V| 增大、
            随 C 减小；k ∈ (0, 1]，典型调参 0.001–0.5，用 development set 的 perplexity 调；
            批评原话：「same k for all n-grams, regardless of how often they appear」
            「treats the lower-order distribution as uniform」「even with optimally tuned k,
            add-k remains a relatively poor performer」⇒ 更强的是 Kneser-Ney（不同计数用不同折扣）。
  深度学习课程笔记（DeepLearning.AI《NLP with Probabilistic Models》W3 笔记，orenbochman.github.io）
      原文三步 recipe：「Create vocabulary V」→「Replace any word in corpus and not in V
            by <UNK>」→「Count the probabilities with <UNK> as with any other word」；
            「Perplexity - only compare LMs with the same V」；
            词表构建三标准：min word frequency / max |V| / **use <UNK> sparingly**。
            ⇒ 本关「可比」= 同一份折叠后 held + 同一 84 个位置 + 同一 V(159) + 同一底数 log2。
  ⚠️ 口径提醒：现代做法（子词 + 神经 LM）已不用 add-k；本关它是**教学最小实现**，
     目的是亲手看见「inf 被治好、UNK 为什么必须存在、训练集变差、V 决定可比性」四件事。
════════════════════════════════════════════════════════════════════
"""

import itertools
import math

from bigram import build_counts, row_totals, vocabulary
from corpus import CORPUS
from entropy import ce_bigram, split_corpus

TEXT: str = CORPUS
HOLDOUT: float = 0.2
"""与关卡 2 同一切点（后 20%），否则所有数字与昨天的不可比。"""

UNK: str = "<UNK>"
"""未登录字的占位 token。它不是汉字，是一个**模型认识的符号**（训练时计数为 0）。"""

KS: tuple[float, ...] = (1.0, 0.1, 0.001)
"""要各报一个数的 k：add-1 / add-0.1 / add-0.001。"""


def closed_vocab(train: str) -> list[str]:
    """已给（纯胶水）：闭词表 = 训练集字表 + `<UNK>`，长度 = |Vtrain| + 1 = 159。"""
    return [*vocabulary(train), UNK]


def fold_unk(text: str, vocab: list[str]) -> list[str]:
    """T0 · 把 `text` 里不在 `vocab` 的字**逐个**换成 `UNK`，返回 **token 列表**。

    这是「闭词表」的落地动作：模型只认识 vocab 里的符号，其余一律归到 `<UNK>`。
    返回列表（不是字符串）——`<UNK>` 是 5 个字符的**一个 token**，拼成字符串会改长度。

    判据：① `len(结果) == len(text)` —— **token 数不变**（位置集合不许变，否则 CE 不可比）
         ② `set(结果) <= set(vocab)` —— 折叠后全在词表里
         ③ 本来就认识的字**一个不动**（不是「全部重写」）
         ④ 幂等：对同一 vocab 再折一次，结果不变
         ⑤ `""` → `[]`；全生字 → 全是 `<UNK>`（长度不变）

    提示：`[c if c in vocab_set else UNK for c in text]`；先把 vocab 转 `set` 再查。
    提示：别用 `str.replace` 逐个替换（`<UNK>` 自带字母，会越换越乱）。
    """
    result = [c if c in set(vocab) else UNK for c in text]
    return result


def smooth_row(
    counts: dict[tuple[str, str], int], prev: str, vocab: list[str], k: float
) -> dict[str, float]:
    """T1 · 单个 context 的平滑分布：`P(next|prev) = (c(prev,next) + k) / (c(prev) + k|V|)`。

    返回**完整**分布：键集合 == `set(vocab)`（|V| 个），未见过 next 的概率 = `k/(C+k|V|)` > 0。

    参数：
      · `counts`：`build_counts(train)` 的数对计数（关卡 1 的产物，**不含** UNK 对）
      · `prev`：上下文（前一个 token）。可能是 `<UNK>`，也可能从没当过上下文 ⇒ C = 0
      · `vocab`：**闭词表** = `closed_vocab(train)`（159 项，含 UNK）；长度进分母
      · `k`：加量，必须 > 0（k ≤ 0 不是平滑，直接拒绝）

    判据：① `set(结果) == set(vocab)` 且长度 == len(vocab)
         ② `abs(sum(值) - 1.0) < 1e-12`，且每个值 > 0
         ③ 未见过的 next 概率**恰好** `k / (C + k*len(vocab))`（用恒等式核，别抄常数）
         ④ `C(prev) == 0`（含 `prev == UNK`）⇒ 整个分布是均匀的 `1/|V|`
         ⑤ k 越小，已见对的概率越接近 MLE `c/C`（k=1 的偏离 > k=0.001 的偏离）

    提示：`C(prev)` 用 `row_totals(counts).get(prev, 0)`，别对 `prev` 直接除。
    """
    if k <= 0:
        raise ValueError("平滑参数k必须 > 0")
    
    C_prev = row_totals(counts).get(prev, 0)
    V_size = len(vocab)
    denominator = C_prev + k * V_size
    
    smoothed_distribution = {}
    for next_token in vocab:
        count_prev_next = counts.get((prev, next_token), 0)
        smoothed_probability = (count_prev_next + k) / denominator
        smoothed_distribution[next_token] = smoothed_probability
    
    return smoothed_distribution


def smooth_ce_bigram(
    counts: dict[tuple[str, str], int], vocab: list[str], k: float, eval_text: str
) -> float:
    """T2 · 平滑后的交叉熵（bits/char）：**先 `fold_unk`，再** `-(1/M) * Σ log2 P_smooth`。

    M = 折叠后的相邻对数量 = `len(token 列表) - 1` —— **不跳过任何位置**（平滑的意义：
    每个位置现在都有定义）。折叠不改变 token 数 ⇒ M 与未折叠时相同（84）。

    判据：① train 上有限、held 上也有限（对比：未平滑的 `ce_bigram` 在 held 上是 `inf`）
         ② train 上必定 **高于** MLE 的 `ce_bigram(counts, train)`（平滑的代价）
         ③ k 越小，train 上越接近 MLE（但永远 ≥，因为分母多了 k|V|）
         ④ 空串 / 单字（M = 0）→ `math.nan`（问题不成立，与关卡 2 一致）
         ⑤ 传未折叠的 held 也能算（内部自己折），但**折叠必须发生在收相邻对之前**

    提示：顺序错一次就踩 M7 —— `tokens = fold_unk(eval_text, vocab)` 之后再
         `pairs = list(itertools.pairwise(tokens))`（`itertools` 已 import）。
    提示：概率从 `smooth_row` 的返回里取（`行[next]`），别自己再除一遍。
    """
    tokens = fold_unk(eval_text, vocab)
    M = len(tokens) - 1
    if M <= 0:
        return math.nan
    tokens = itertools.pairwise(tokens)
    log_sum = 0.0
    for prev, next_token in tokens:
        smoothed_probs = smooth_row(counts, prev, vocab, k)
        prob = smoothed_probs[next_token]
        log_sum += math.log2(prob)
    return -log_sum / M


def judge_smoothing_math() -> bool:
    """T3 · 判据 1：平滑公式本身（不依赖语料的巧合，能手算就手算）。

    ① 归一：对若干 `prev`（含 `<UNK>`）与 `k ∈ KS`：`Σ smooth_row(...) == 1.0`（±1e-12），
       键集合 == 闭词表（159 项）
    ② 未见列的 概率 == `k/(C+k|V|)`（恒等式核对，**不许抄常数**）；`<UNK>` 列同理
    ③ `C(prev)=0` 的上下文（`<UNK>` 天然是这种）⇒ 分布均匀 `1/|V|`
    ④ k 单调：同一 `prev` 上，`max|P_k − P_MLE|` 随 k 减小而减小（k=1 的偏离 > k=0.001 的偏离）
    ⑤ 非法 k（0 或负数）必须被拒（`ValueError` / `AssertionError` 都算，不能静默当成 MLE）
    ⑥ 小语料复核：`counts = build_counts("adbc")` = `{(a,d), (d,b), (b,c)}`、vocab 4 项、k=1
       ⇒ `P(d|a) = (1+1)/(1+4) = 0.4`（**a 的后继是 d，不是 b**）、`P(b|a) = P(c|a) = P(a|a) = 1/5`
       · `C('c') == 0`（c 从没当过前一个字）⇒ 整行均匀 `0.25`
       · ⚠️ 2026-11-19 更正：原先这里写成 `P(b|a)=0.4`、`P(*|d)=0.25`，是照 "ab..." 想当然写错的 ——
         "adbc" 的相邻对是 (a,d)(d,b)(b,c)，d **当过**上下文，c 才没当过。照错值写断言必然红。

    通过后 print：|V|、k=1 时某个 context 的「移走质量」实测值（应 == k|V|/(C+k|V|)）。
    """
    train, _held = split_corpus(TEXT)
    vocab = closed_vocab(train)
    vocab_set = set(vocab)
    counts = build_counts(train)
    totals = row_totals(counts)
    v_size = len(vocab)

    busiest = max(totals, key=lambda ch: totals[ch])
    rarest = min(totals, key=lambda ch: totals[ch])

    # ── ① 归一 + 键集合：真实语料三行（最大行和 / 最小行和 / UNK）× 三个 k ──
    for prev in (busiest, rarest, UNK):
        for k in KS:
            row = smooth_row(counts, prev, vocab, k)
            assert set(row) == vocab_set, f"键集合 ≠ 闭词表（prev={prev!r}, k={k}）"
            assert len(row) == v_size, f"键数 {len(row)} ≠ |V| {v_size}（prev={prev!r}）"
            assert all(p > 0 for p in row.values()), f"出现非正概率（prev={prev!r}, k={k}）"
            total = sum(row.values())
            assert abs(total - 1.0) < 1e-12, f"行和 {total!r} ≠ 1（prev={prev!r}, k={k}）"

    # ── ② 恒等式：(c + k)/(C + k|V|)；见过的列与未见列共用同一个分母 ──
    k0 = KS[0]
    for prev in (busiest, rarest, UNK):
        c_prev = totals.get(prev, 0)
        denominator = c_prev + k0 * v_size
        seen = {n for (p, n) in counts if p == prev}
        unseen = [n for n in vocab if n not in seen]
        assert unseen, f"{prev!r} 这一行没有未见列，证明不了恒等式"
        row = smooth_row(counts, prev, vocab, k0)
        expected_unseen = k0 / denominator
        for n in unseen[:2]:
            assert abs(row[n] - expected_unseen) < 1e-12, (
                f"未见列 {n!r} 概率 {row[n]!r} ≠ k/(C+k|V|) = {expected_unseen!r}（prev={prev!r}）"
            )
        for n in sorted(seen)[:2]:
            expected_seen = (counts[(prev, n)] + k0) / denominator
            assert abs(row[n] - expected_seen) < 1e-12, (
                f"见过的列 {n!r} 概率 {row[n]!r} ≠ (c+k)/(C+k|V|) = {expected_seen!r}"
            )

    # ── ③ C(prev) = 0 ⇒ 整行均匀 1/|V|（UNK 天然是这种行）──
    assert totals.get(UNK, 0) == 0, "UNK 不该在训练集里当过上下文"
    unk_row = smooth_row(counts, UNK, vocab, 1.0)
    unk_dev = max(abs(p - 1 / v_size) for p in unk_row.values())
    assert unk_dev < 1e-12, f"UNK 行不是均匀 1/{v_size}，最大偏差 {unk_dev:.3e}"

    # ── ④ k 单调：只在「见过的列」上比 P_MLE = c/C（未见列的 MLE 是 0，不是 1/|V|）──
    most_types = max(totals, key=lambda ch: len({n for (p, n) in counts if p == ch}))
    seen_best = {n for (p, n) in counts if p == most_types}
    c_best = totals[most_types]
    deviations = {}
    for k in KS:
        row = smooth_row(counts, most_types, vocab, k)
        deviations[k] = max(abs(row[n] - counts[(most_types, n)] / c_best) for n in seen_best)
    assert deviations[KS[0]] > deviations[KS[1]] > deviations[KS[2]] > 0.0, (
        f"k 越小应越接近 MLE，实测偏离 {deviations}"
    )

    # ── ⑤ 非法 k 必须被拒（不能静默当成 MLE，也不能崩在除零里）──
    for bad_k in (0.0, -1.0):
        try:
            smooth_row(counts, most_types, vocab, bad_k)
        except ValueError:
            continue
        raise AssertionError(f"k={bad_k} 应被拒绝，却正常返回了结果")

    # ── ⑥ 小语料手算复核（"adbc" 的相邻对是 (a,d)(d,b)(b,c)）──
    tiny_counts = build_counts("adbc")
    tiny_vocab = ["a", "b", "c", "d"]
    row_a = smooth_row(tiny_counts, "a", tiny_vocab, 1.0)
    assert abs(row_a["d"] - 0.4) < 1e-12, f"P(d|a) 应为 0.4，实测 {row_a['d']!r}"
    for n in ("a", "b", "c"):
        assert abs(row_a[n] - 0.2) < 1e-12, f"P({n}|a) 应为 0.2，实测 {row_a[n]!r}"
    assert row_totals(tiny_counts).get("c", 0) == 0, "c 从没当过上下文，C(c) 应为 0"
    row_c = smooth_row(tiny_counts, "c", tiny_vocab, 1.0)
    assert all(abs(p - 0.25) < 1e-12 for p in row_c.values()), "C('c')=0 应得均匀 0.25"

    c_rarest = totals[rarest]
    moved = 1.0 * v_size / (c_rarest + 1.0 * v_size)
    print(
        f"|V| = {v_size} | 最大行和 '{busiest}' = {totals[busiest]} | "
        f"k=1 时最惨的行 '{rarest}'（C={c_rarest}）移走质量 = {moved:.6f}"
        f" = 1×{v_size}/({c_rarest}+{v_size}) | UNK 行 = 均匀 1/{v_size}"
    )
    return True


def judge_comparable_numbers() -> bool:
    """T4 · 判据 2：折叠 + 证据 + 可比性（数字才是这一关的产出）。

    ① 折叠四条：token 数不变（85）· 折叠后 token 全 ∈ 闭词表 · 幂等 · 认识的字一个不动
       · 另报：不同 token 数 **61 → 40**（22 个生字并成 1 个 `<UNK>`）
    ② 动机仍在：**未折叠**的 `ce_bigram(counts, held)` == `math.inf`
    ③ V 钉死：`len(vocab) == len(set(train)) + 1 == 159`（多出来的那一个就是 UNK）
    ④ 可比前提：折叠后相邻对 M == `len(held) - 1` == 84（折叠没改位置数）
    ⑤ 平滑后有限：对每个 `k ∈ KS`，held 上 `math.isfinite(...)`
    ⑥ 代价可见：train 上 `smooth_ce(k=1) > smooth_ce(k=0.1) > MLE ce_bigram(counts, train)`
    ⑦ 各 k 报数：train / held 的 CE 与 ppl（held 的三个数是本关要交的证据）
    ⑧ 两路互证 held CE（k=1）：逐位置 vs 按对加权，差 < 1e-12
       ⇒ 专抓「跳过算不出的位置、分母却没跟着变」（2026-11-19 变异 M6 漏网后补的）

    通过后 print：生字数、折叠后不同 token 数、未见对比例、MLE 的 inf、三个 k 的 train/held
    CE 与 ppl、以及「k=1 从已见事件移走的质量」最大值（用 `k|V|/(C+k|V|)`，C 取最小行和）。
    ⚠️ 别把折叠后的 CE 与昨天未折叠的 1.6935 放同一句比较 —— 口径不同，要写清。
    """
    train, held = split_corpus(TEXT)
    vocab = closed_vocab(train)
    vocab_set = set(vocab)
    counts = build_counts(train)
    totals = row_totals(counts)
    v_size = len(vocab)
    known_chars = set(train)

    # ── ① 折叠四条 ──
    tokens = fold_unk(held, vocab)
    assert len(tokens) == len(held), f"折叠改了 token 数：{len(tokens)} ≠ {len(held)}"
    assert set(tokens) <= vocab_set, "折叠后仍有生字（不在闭词表里）"
    expected_kept = [c for c in held if c in known_chars]
    assert [t for t in tokens if t != UNK] == expected_kept, "本来就认识的字被改动了"
    new_chars = set(held) - known_chars
    oov_positions = sum(1 for c in held if c not in known_chars)
    assert sum(1 for t in tokens if t == UNK) == oov_positions, "UNK 个数 ≠ 生字出现的位置数"
    re_folded = [t if t in vocab_set else UNK for t in tokens]
    assert re_folded == tokens, "对已折叠的结果再折一次变了（不幂等）"
    joined = "".join(tokens)
    assert len(joined) != len(held), "拼回字符串居然没变长 —— UNK 不该是多字符 token？"

    # ── ② 动机仍在：未折叠的 MLE 在 held 上是 inf ──
    mle_held = ce_bigram(counts, held)
    assert mle_held == math.inf, f"未平滑的留出集 CE 应为 inf，实测 {mle_held!r}"

    # ── ③ V 钉死：158 + UNK = 159 ──
    assert v_size == len(set(train)) + 1, f"|V| {v_size} ≠ |Vtrain|+1 = {len(set(train)) + 1}"
    assert v_size == 159, f"闭词表应为 159（158 + UNK），实测 {v_size} —— 口径被改过？"

    # ── ④ 折叠没有改位置数 ──
    positions = len(tokens) - 1
    assert positions == len(held) - 1, f"位置数 {positions} ≠ 未折叠的 {len(held) - 1}"
    assert positions == 84, f"位置数应为 84，实测 {positions}"

    # ── ⑤ 平滑后有限（这一关的核心证据）──
    ce_train = {k: smooth_ce_bigram(counts, vocab, k, train) for k in KS}
    ce_held = {k: smooth_ce_bigram(counts, vocab, k, held) for k in KS}
    for k in KS:
        assert math.isfinite(ce_train[k]), f"train 上 k={k} 得到 {ce_train[k]!r}"
        assert math.isfinite(ce_held[k]), f"held 上 k={k} 得到 {ce_held[k]!r}"

    # ── ⑥ 代价可见：平滑让训练集变差，且 k 越大越差 ──
    mle_train = ce_bigram(counts, train)
    assert ce_train[KS[0]] > ce_train[KS[1]] > mle_train, (
        f"训练集应满足 k=1 > k=0.1 > MLE，实测 {ce_train[KS[0]]:.4f} / "
        f"{ce_train[KS[1]]:.4f} / {mle_train:.4f}"
    )
    assert ce_train[KS[2]] > mle_train, (
        f"k=0.001 的 train CE {ce_train[KS[2]]:.4f} 应仍高于 MLE {mle_train:.4f}"
    )

    # ── ⑧ 两路互证 held CE(k=1)：逐位置 vs 按对加权（专抓「跳位置不改分母」）──
    k0 = KS[0]
    pairs_all = list(itertools.pairwise(tokens))
    m_all = len(pairs_all)
    tallies: dict[tuple[str, str], int] = {}
    for pair in pairs_all:
        tallies[pair] = tallies.get(pair, 0) + 1
    by_position = smooth_ce_bigram(counts, vocab, k0, held)
    by_count = 0.0
    for (prev_tok, next_tok), pair_count in tallies.items():
        prob = smooth_row(counts, prev_tok, vocab, k0)[next_tok]
        by_count += -(pair_count / m_all) * math.log2(prob)
    gap = abs(by_position - by_count)
    assert gap < 1e-12, (
        f"两路互证失败：逐位置 {by_position:.6f} vs 按对加权 {by_count:.6f}（差 {gap:.3e}）"
        f" —— 典型原因是跳过了某些位置却没把分母改成实际算到的个数"
    )

    # ── ⑦ 报数：本关要交的证据表 ──
    small_prev = min(totals, key=lambda ch: totals[ch])
    c_small = totals[small_prev]
    moved = 1.0 * v_size / (c_small + 1.0 * v_size)
    held_pairs = len(held) - 1
    unseen_pairs = sum(1 for pair in itertools.pairwise(held) if pair not in counts)

    print(
        f"held {len(held)} token | 生字 {len(new_chars)} 种（出现 {oov_positions} 次）| "
        f"折叠后不同 token {len(set(tokens))}（未折叠 {len(set(held))}）| 位置 {positions}"
    )
    print(
        f"未见对 {unseen_pairs}/{held_pairs}（{unseen_pairs / held_pairs:.1%}）"
        f" | 未平滑 held = inf"
    )
    print(f"MLE train = {mle_train:.4f} bit（ppl {2**mle_train:.2f}）")
    for k in KS:
        print(
            f"k = {k:<5} train {ce_train[k]:.4f} bit（ppl {2**ce_train[k]:6.2f}）"
            f" | held {ce_held[k]:.4f} bit（ppl {2**ce_held[k]:6.2f}）"
        )
    print(
        f"k=1 移走质量最大（最惨行 '{small_prev}'，C={c_small}）= {moved:.6f}"
        f" = 1×{v_size}/({c_small}+{v_size})"
    )
    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的，不是硬编码字符串。"""
    results = [
        judge_smoothing_math(),
        judge_comparable_numbers(),
    ]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
