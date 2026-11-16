"""L1 · 词与概率 —— 关卡 1：char-level bigram 计数 → 条件概率表

目的（一条根）：语言模型只做一件事 —— 给「下一个字」打分。
    · char-level：词表 = 语料里出现过的**字符**，单位是「字」，不是「词」
    · bigram    ：打分时只看**前一个字**，把「历史」砍成长度 1

这一关不训任何东西，只做「数票 → 换算成比例」。交叉熵 / softmax / 平滑在后面三关。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T1 关卡1 [实现] build_counts(text)              相邻字符对 → 次数
 T2 关卡1 [实现] vocabulary(text)                出现过哪些字（排序，可复现）
 T3 关卡1 [实现] row_totals(counts)              每个「前一个字」的后继总数（行和）
 T4 关卡1 [实现] next_probs(counts, context)     P(下一个字 | 上一个字)
 T5 关卡1 [判据] judge_counts()                  判据 1：计数守恒（两条等式）
 T6 关卡1 [判据] judge_probs()                   判据 2：归一 + 真的是「条件」
════════════════════════════════════════════════════════════════════
 写完一项跑一项（别等六项全写完）：
   uv run python -c "from bigram import judge_counts as f; print(f())"
   uv run python -c "from bigram import judge_probs as f; print(f())"
 全绿后再跑总入口：uv run python bigram.py

跑法：进本目录 → uv sync（0 依赖，秒过）→ uv run python bigram.py
文档依据（2026-11-16 查证）：
  docs.python.org/3/library/math.html
      log(0) 抛 ValueError（返回 -inf 那种写法不存在）；exp 溢出抛 OverflowError；
      log2(x) 比 log(x, 2) 更准 —— 判据 2、3 用得上
  docs.pytorch.org/docs/2.15/generated/torch.nn.CrossEntropyLoss.html
      input 必须是**未归一化 logits**（不需为正、不需和为 1），等价于 LogSoftmax + NLLLoss；
      reduction 默认 'mean' —— 判据 3 的「先 log_softmax 再取负」就是照这个形状写的
  SLP3 Ch.3 N-gram Language Models
      交叉熵 = -(1/N)Σ log P；困惑度 = 2^H；概率连乘会下溢 ⇒ 一律 log 相加
════════════════════════════════════════════════════════════════════
"""

# 类型速查（不用改，读的时候当注释看）：
#   counts: dict[tuple[str, str], int] —— ("前一个字", "后一个字") -> 出现次数
# 本练习 0 依赖、0 次模型调用（0 token），不需要 .env

from corpus import CORPUS

TEXT: str = CORPUS
"""本练习统一读这个常量；换语料只改 corpus.py，bigram.py 一行不动。"""


def build_counts(text: str) -> dict[tuple[str, str], int]:
    """T1 · 相邻字符对计数：把 text 从头扫到尾，数每一对出现了几次。

    每个起始位置 i（0 <= i <= len(text) - 2）贡献一次 (text[i], text[i+1])；
    内容相同的对合并成同一个键，值 = 出现次数。

    提示：collections 里有一个类，天生就是「数票」用的，一行能做完。
    """
    dict_counts = {}
    for i in range(len(text) - 1):
        pair = (text[i], text[i + 1])
        if pair in dict_counts:
            dict_counts[pair] += 1
        else:
            dict_counts[pair] = 1
    return dict_counts


def vocabulary(text: str) -> list[str]:
    """T2 · 语料里出现过哪些字符（这就是 char-level 的**词表**）。

    判据：① 不重复 ② **排序后**返回（set 的迭代顺序不可依赖，排序才可复现）
         ③ len(结果) == len(set(text))

    顺便看一眼 len(结果)：这个数字就是判据 2 里「均匀基线」要用的 V。
    """
    return sorted(set(text))


def row_totals(counts: dict[tuple[str, str], int]) -> dict[str, int]:
    """T3 · 行和：每个「前一个字」作为条件时，它的后继总次数。

    判据：row_totals(counts)[c] == 所有以 c 开头的那些对的次数之和。

    ⚠️ 这里最容易抄错的地方：按**前一个字**聚合，不是按后一个字聚合。
    按后一个字聚合出来的表同样「看着很正常」，但下一关算概率时会全错。
    """
    totals = {}
    for (prev_char, _next_char), count in counts.items():
        if prev_char in totals:
            totals[prev_char] += count
        else:
            totals[prev_char] = count
    return totals


def next_probs(counts: dict[tuple[str, str], int], context: str) -> dict[str, float]:
    """T4 · 条件概率表：P(下一个字 | 上一个字 == context)。

    判据：① 每个值 = 该对的次数 / row_totals(counts)[context]
         ② 所有值加起来 == 1.0（浮点容差 1e-12）
         ③ context 从未出现过 → 返回 **空 dict**，不许抛 KeyError

    ③ 不是偷懒，是必须：真实语料里大量上下文组合从没出现过。返回空 dict 让调用方
    自己面对「概率和是 0」这件事 —— 关卡 4（平滑）就是从这里长出来的。
    """
    totals = row_totals(counts)
    if context not in totals:
        return {}

    total_count = totals[context]
    probs = {}
    for (prev_char, next_char), count in counts.items():
        if prev_char == context:
            probs[next_char] = count / total_count
    return probs


def judge_counts() -> bool:
    """T5 · 判据 1：计数守恒 —— 三条，缺一条都算假绿。

    ① 总对数：`sum(counts.values()) == len(CORPUS) - 1`
       每个起始位置恰好贡献一对；按内容合并只改「多少种」，不改「一共多少对」。
    ② 行和自证：对 set(CORPUS[:-1]) 里**每个**字符 c，
       `row_totals(counts)[c] == CORPUS[:-1].count(c)`
       意义：把「按 context 聚合」写成「按 next 聚合」时，① 照样过、② 立刻炸。
    ③ 键集合：`set(row_totals(counts)) == set(CORPUS[:-1])`
       **语料最后一个字不进键集合** —— 它后面没有字，没有条件可谈。

    通过后把这三个数 print 出来，抄进本目录 README 的「实测数字」表：
       总对数 / 不同 bigram 种数 / 行和最大者（哪个字、多少）
    """
    counts = build_counts(TEXT)
    totals = row_totals(counts)
    total_pairs = sum(counts.values())

    assert total_pairs == len(TEXT) - 1, f"总对数 {total_pairs} != {len(TEXT) - 1}"

    for c in totals:
        expect = TEXT[:-1].count(c)
        assert totals[c] == expect, f"行和不一致：{c!r} 给了 {totals[c]}，语料里 {expect} 次"

    assert set(totals) == set(TEXT[:-1]), "键集合不一致（末位那个字不该当 context）"

    top_ctx = max(totals, key=lambda k: totals[k])
    print(
        f"总对数={total_pairs} | 不同 bigram={len(counts)} | 行和最大 {top_ctx!r}={totals[top_ctx]}"
    )
    return True


def judge_probs() -> bool:
    """T6 · 判据 2：条件概率表 —— 归一 + 真的是「条件」。

    ① 归一（三条，缺一条都算假绿）：对每个出现过的 context c，
       · 概率和 == 1.0（±1e-12）
       · **键集合 == counts 里真实的后继集合**（防「凭空发明字符」）
       · **每个值 >= 0**（`{'包': 1.5, '面': -0.5}` 的和也是 1，只查和抓不到）
    ② 条件性（最难的一条，专门防「其实返回了全局分布」）—— 判据里**不写任何语料字面量**：
       · 从 totals 里按规则挑 context：`totals[c] >= 3` **且**后继只有 1 种
         （只出现 1 次的字不能挑：它必然是点质量，等于什么都没证明）
       · 断言它是**点质量**：`next_probs(counts, ctx) == {only: 1.0}`，`only` 由数据给
       · 断言**恒等式**（不是阈值）：点质量时 `max |P(·|ctx) − P_global| == 1 − P_global(only)`
         本语料实测：`1 − P(包) = 1 − 0.0213 = 0.9787`
       · ⚠️ 比较一律 `probs.get(c, 0.0)`：条件分布只有 1~9 个键，全局 180 个，
         写成 `probs[c]` 会 KeyError（实测第一个炸的是 '。'）
    ③ 边界：`next_probs(counts, "语料里没有的字")` → 返回 `{}`，不抛 KeyError
       空 dict 的和是 0.0，概率和不为 1 —— 这是「未见 context」的病，关卡 4 治。

    通过后 print：V（词表大小）/ 两个 context 各自最大的后继 / 最大差值。
    """
    counts = build_counts(TEXT)
    totals = row_totals(counts)
    vocab = vocabulary(TEXT)

    # 真实后继集合：期望值直接从 counts 推，不经过 T3 / T4
    successors: dict[str, set[str]] = {}
    for ctx, nxt in counts:
        successors.setdefault(ctx, set()).add(nxt)
    assert set(successors) == set(totals), "T3 的行和键集合与 counts 对不上"

    # ① 三条：归一 / 键集合 / 非负
    for ctx in successors:
        probs = next_probs(counts, ctx)
        got = sum(probs.values())
        assert abs(got - 1.0) < 1e-12, f"{ctx!r} 概率和为 {got}"
        extra = set(probs) - successors[ctx]
        missing = successors[ctx] - set(probs)
        assert not extra and not missing, (
            f"{ctx!r} 后继集合不对：多了 {len(extra)} 个 {sorted(extra)[:5]}，"
            f"少了 {len(missing)} 个 {sorted(missing)[:5]}"
        )
        assert min(probs.values()) >= 0.0, f"{ctx!r} 出现负概率 {min(probs.values())}"

    # ② 条件性：条件分布 ≠ 全局分布 —— 样本由规则挑，判据里不写字面量
    global_probs = {c: TEXT.count(c) / len(TEXT) for c in vocab}

    narrow_ctx = max(
        (c for c in totals if totals[c] >= 3 and len(successors[c]) == 1),
        key=lambda c: totals[c],
        default=None,
    )
    assert narrow_ctx is not None, "语料里没有「出现 ≥3 次且后继唯一」的 context，这条判据立不住"

    cond_narrow = next_probs(counts, narrow_ctx)
    only = next(iter(successors[narrow_ctx]))  # 期望值来自 counts，不来自被测函数
    assert cond_narrow == {only: 1.0}, (
        f"{narrow_ctx!r} 后继只有 1 种，应得点质量 {{{only!r}: 1.0}}，"
        f"实际 {len(cond_narrow)} 个键：{sorted(cond_narrow)[:5]}"
    )

    # 恒等式：点质量分布与全局分布的最大差 = 1 - P_global(唯一后继)
    max_diff = max(abs(cond_narrow.get(c, 0.0) - global_probs[c]) for c in vocab)
    expect_diff = 1.0 - global_probs[only]
    assert abs(max_diff - expect_diff) < 1e-12, (
        f"{narrow_ctx!r} 的最大差应等于 1 - P_global({only!r}) = {expect_diff}，实际 {max_diff}"
    )

    # ③ 边界：没出现过的 context 返回 {}，不许抛异常
    unseen_ctx = next(c for c in "abcxyz0123456789" if c not in totals)
    unseen = next_probs(counts, unseen_ctx)
    assert unseen == {}, f"未见 context {unseen_ctx!r} 应返回 {{}}，实际 {unseen}"
    assert sum(unseen.values()) == 0.0, f"空分布的和应为 0.0，实际 {sum(unseen.values())}"

    print(
        f"V={len(vocab)} | 自动挑 {narrow_ctx!r} → {only!r} 点质量 | "
        f"与全局最大差 {max_diff:.4f} == 1 - P({only!r}) {expect_diff:.4f}"
    )
    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的，不是硬编码字符串。"""
    results = [
        judge_counts(),
        judge_probs(),
    ]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
