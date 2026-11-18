"""L1 · 词与概率 —— 关卡 3：softmax（logits → 概率分布，且不许溢出）

目的（一条根）：softmax 只做两件事 —— ① 把「任意实数分数」变成「正数」② 让它们加起来是 1。
    · logits：模型的原始输出。可以是负数、可以很大、并不加起来等于 1 —— 它只是「分数」
    · softmax(z)_i = exp(z_i) / Σ exp(z_j)
    · 为什么用 exp 而不是「先减去最小、再除以和」：exp 恒正（概率必须 ≥ 0）、
      且保序（分数高的概率一定高）——「归一化 + 保序」这两条决定了它只能长这样

生活模型：食堂打饭。logits = 你给六个窗口的印象分（可以是 3、-2、0.5、0）；
softmax = 把印象分换算成「实际走进哪个窗口的概率」：印象最好的吃掉大部分概率，
但别的窗口也各有一点份 —— 分差越大，概率越集中。

这一关**不碰语料**（不 import corpus）：softmax 是纯函数，输入输出都是数，
所以判据里的期望值全部手算 / 由设计好的输入给出，不依赖语料的巧合。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T1 [实现] softmax_naive(logits)      直算 exp / Σexp —— **故意的错版本**，当反面对照
 T2 [实现] softmax(logits)            先减 max 再 exp —— 真正要用的版本
 T3 [判据] judge_shift_invariance()   判据 1：平移不变 + 精确值 + 归一/保序 + 边界 + 互证
 T4 [判据] judge_overflow()           判据 2：溢出 / 下溢对照（naive 必炸，稳定版必活）
════════════════════════════════════════════════════════════════════
 写完一项跑一项（别等四项全写完）：
   uv run python -c "from softmax import judge_shift_invariance as f; print(f())"
   uv run python -c "from softmax import judge_overflow as f; print(f())"
 全绿后再跑总入口：uv run python softmax.py

跑法：进本目录 → uv sync（0 依赖）→ uv run python softmax.py
      ⚠️ `math` 已经 import 好；T1/T2 写完之前 `uv run ruff check .` 会报 F401（unused import），
         实现一落地就消失，不是你的错。

文档依据（2026-11-18 查证）：
  pytorch/torch/nn/functional.py —— log_softmax 的 docstring 原文：
      "While mathematically equivalent to log(softmax(x)), doing these two operations
       separately is slower and numerically unstable. This function uses an alternative
       formulation to compute the output and gradient correctly."
      ⇒ 分开算不是「少个技巧」，是真的会坏。
  pytorch/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h
      CPU kernel 第一步就是 `sum_j exp(data[j] - max_input)`，且在注释里强调
      「运算顺序不能换」，否则大数 + 小差时出数值问题。
  pytorch/torch/_refs/__init__.py —— aten.logsumexp 的分解：
      减 max → exp → 求和 → 取 log → 把 max 加回，
      即 log Σ exp(x) = max + log Σ exp(x − max)。
  scipy/special/_logsumexp.py —— scipy.special.softmax 源码正文：
      `x_max = max(x); exp_x_shifted = exp(x - x_max); return exp_x_shifted / sum(exp_x_shifted)`
      ⇒ 减 max 是官方库的标准做法，不是偏方。
  docs.python.org/3/library/math.html + 本机实测（Python 3.14.8）：
      `math.exp(1000)` 抛 **OverflowError**（"math range error"）；
      `math.exp(-1000)` **不抛**，下溢返回 `0.0`；`0.0 / 0.0` 抛 **ZeroDivisionError**
      ⇒ 两个方向都要防，而且炸出来的是两种不同的异常。
════════════════════════════════════════════════════════════════════
"""

import math

# 类型速查：logits: list[float] —— 未归一化的分数；返回值 list[float] —— 概率，和 = 1


def softmax_naive(logits: list[float]) -> list[float]:
    """T1 · 公式直译版：`exp(z_i) / Σ exp(z_j)` —— **不许拿它算真实 logits**。

    造它出来只为一件事：判据 2 要拿**实测的异常**证明「不这么做真的会炸」，
    而不是靠「我听说会溢出」这种说法。

    判据（由 judge 函数检查，这里只列要求）：
      · 在安全区间（|z| ≤ 10）上与 T2 的结果一致，差 < 1e-15
      · `[1000.0, 0.0]` → **抛 OverflowError**（不是返回 `inf`、不是返回 `[nan, nan]`）
      · `[-1000.0, -1001.0]` → 抛 **ZeroDivisionError**（两个 exp 都下溢成 0.0，0/0）

    提示：先把每个 logit 取 exp 存进一个列表（别算两遍），再除以它们的和。
    """
    logits_exp = [math.exp(x) for x in logits]
    return [x / sum(logits_exp) for x in logits_exp]


def softmax(logits: list[float]) -> list[float]:
    """T2 · 稳定版：**先减 max，再 exp，再归一**。

    为什么减 max 不改变结果（这一句要能自己讲出来，才算懂）：
      `softmax(z + c)` 与 `softmax(z)` 完全相等 —— 分子分母同乘 `exp(c)` 约掉。
      取 `c = -max(z)`：最大那一项变成 `exp(0) = 1`，于是**分母里必然有一项是 1**，
      分母不可能下溢成 0；其余项都是 `exp(负数) ≤ 1`，不会向上溢出。
      ⇒ 数学上一模一样，数值上从「必炸」变成「几乎不可能炸」。

    判据（judge 函数检查）：
      · `softmax([])` → `[]`（别让 `max([])` 抛 ValueError）
      · `softmax([x])` → `[1.0]`（单元素，无论 x 多大）
      · 和 == 1（±1e-12）；每个值 > 0（下溢成 0.0 的情形只在极端输入下出现，见 T4）
      · 平移不变：`softmax(z)` 与 `softmax(z + c)` 差 < 1e-12（c 取 1 / -5 / 100）
      · `softmax([1000.0, 0.0])` → `[1.0, 0.0]`（那个 0.0 是**正常下溢**，不是 bug）

    提示：`m = max(logits)` 那行要在 exp 之前；空列表要在 max 之前判掉。
    """
    if not logits:
        return []
    m = max(logits)
    logits_shifted = [x - m for x in logits]
    logits_exp = [math.exp(x) for x in logits_shifted]
    return [x / sum(logits_exp) for x in logits_exp]


def judge_shift_invariance() -> bool:
    """T3 · 判据 1：平移不变 + 精确值 + 归一/保序 + 边界 + 互证。

    ⚠️ 判据里的期望值**不来自被测函数**：全用手算的数，或由设计好的输入给出。

    ① 平移不变（本关核心）：
       取 `z0 = [1.0, 2.0, 3.0]`，对 c ∈ {1.0, -5.0, 100.0}：
       `max | softmax([z + c]) - softmax(z0) | < 1e-12`
       理由：分子分母同乘 `exp(c)` 会约掉 —— 这是**恒等式**，不是近似。
       ⚠️ 只测这一条还不够：把「减 max」写成「减 mean」甚至「减首项」，
       整体平移时它们也一起平移 ⇒ 平移不变**照样绿**。所以必须有 ②。
    ② 精确值（手算，钉住「减的到底是 max」这件事）：
       · `softmax([0.0, 0.0])` → `[0.5, 0.5]`
       · `softmax([math.log(3), math.log(1)])` → `[0.75, 0.25]`（差 < 1e-15）
       · `softmax([math.log(1), math.log(7)])` → `[0.125, 0.875]`（差 < 1e-15）
       三组值的和都是 1，任何「分母算错」的写法都过不去。
    ③ 归一 + 正性 + 保序：
       · `abs(sum(p) - 1.0) < 1e-12`
       · 每个 `p > 0`（这里用温和的 z；`-1000` 那种留给 T4）
       · logits 升序 ⇒ 概率升序（exp 单调）。用 `z = [-3.0, 0.0, 2.5]` 检查两两顺序
    ④ 边界：
       · `softmax([]) == []`（不许抛 ValueError）
       · `softmax([7.5]) == [1.0]`
    ⑤ 互证（防「稳定版其实是另一个函数」）：
       在 `|z| ≤ 10` 上 `softmax(z)` 与 `softmax_naive(z)` 差 < 1e-15
       —— naive 是公式直译；两边在这个区间里对得上，才说明稳定版没偷偷改数学。

    通过后 print：平移误差最大值 / 两个精确值的实测 / 互证最大差。
    """
    z0 = [1.0, 2.0, 3.0]
    c = [1.0, -5.0, 100.0]
    worst = 0.0
    for ci in c:
        z = [x + ci for x in z0]
        diff = max(abs(a - b) for a, b in zip(softmax(z), softmax(z0), strict=True))
        assert diff < 1e-12, f"平移不变失败：c={ci}"
        worst = max(worst, diff)
    assert softmax([0.0, 0.0]) == [0.5, 0.5], "精确值失败：softmax([0, 0])"
    p1 = softmax([math.log(3), math.log(1)])[0]   # 0.75
    p2 = softmax([math.log(1), math.log(7)])[0]   # 0.125
    assert abs(p1 - 0.75) < 1e-15, f"精确值失败：softmax([log(3), log(1)]) = {p1}, 期望值 = 0.75"
    assert abs(p2 - 0.125) < 1e-15, f"精确值失败：softmax([log(1), log(7)]) = {p2}, 期望值 = 0.125"
    judge1 = abs(softmax([math.log(3), math.log(1)])[0] - 0.75) < 1e-15
    judge2 = abs(softmax([math.log(1), math.log(7)])[0] - 0.125) < 1e-15
    assert judge1 , "精确值失败，softmax([log(3), log(1)])"
    assert judge2 , "精确值失败，softmax([log(1), log(7)])"
    assert abs(sum(softmax([-3.0, 0.0, 2.5])) - 1.0) < 1e-12, "归一失败"
    assert all(p > 0 for p in softmax([-3.0, 0.0, 2.5])), "正性失败"
    probs = softmax([-3.0, 0.0, 2.5])
    assert probs[0] < probs[1] < probs[2], "保序失败"
    assert softmax([]) == [], "边界失败：空列表"
    assert softmax([7.5]) == [1.0], "边界失败：单元素"
    s1 = softmax([-10.0, 0.0, 10.0])
    s2 = softmax_naive([-10.0, 0.0, 10.0])
    judge3 = max(abs(a - b) for a, b in zip(s1, s2, strict=True))
    assert judge3 < 1e-15, "互证失败：稳定版与 naive 版结果差异过大"
    print(f"平移最大误差 {worst:.2e} | 精确值 {p1:.15f} / {p2:.15f} | 互证最大差 {judge3:.2e}")
    return True


def judge_overflow() -> bool:
    """T4 · 判据 2：溢出 / 下溢对照 —— **同一个输入**跑两个版本，差别才是证据。

    ⚠️ 结论不许是「我听说会溢出」；必须是 try/except 抓到的**实际异常类型**。

    ① 向上溢出（OverflowError）：
       `softmax_naive([1000.0, 0.0])` 必须抛 `OverflowError`
       （`math.exp(1000)` → "math range error"，本机 3.14.8 实测）
       抓法：`try: softmax_naive(...)  except OverflowError: 通过  else: raise AssertionError(...)`
       —— 没抛就是红，别用「返回了 inf」糊过去（`inf/inf` 只会得到 nan）。
    ② 同一输入，稳定版必须活：
       `softmax([1000.0, 0.0])` → `[1.0, 0.0]`，且和为 1.0。
       那个 `0.0` 是 `exp(-1000)` **下溢**的结果（不是异常）：logits 差 1000，
       第二个概率是 e⁻¹⁰⁰⁰ ≈ 0，double 存不下 —— 可以接受，但要**报出来**，不许装看不见。
    ③ 全下溢（比 ① 更阴的一条）：
       `softmax([-1000.0, -1001.0])` → `[0.7310585786300049, 0.2689414213699951]`
       （= 1/(1+e⁻¹) 与 e⁻¹/(1+e⁻¹)，期望值手算写死在判据里，不要调用被测函数算）
       同一输入喂 naive 版 → 抛 **ZeroDivisionError**（两个 exp 都成 0.0，0/0）
       ⇒ 「防溢出」不只防大数：全是大负数时 naive 版会以**另一种异常**死掉。
    ④ 极端量平移一致：
       `softmax([1000.0, 0.0])` 与 `softmax([0.0, -1000.0])` 差 < 1e-12
       —— 极端输入的平移不变，证明减 max 真的在起作用。

    通过后 print：三个异常/结果的实际值，抄进本目录 README 的「实测数字」表。
    """
    try:
        softmax_naive([1000.0, 0.0])
    except OverflowError:
        print("向上溢出：softmax_naive([1000.0, 0.0]) 抛 OverflowError")
    else:
        raise AssertionError("向上溢出失败：softmax_naive([1000.0, 0.0]) 没抛 OverflowError")
    s1 = softmax([1000.0, 0.0])
    assert abs(sum(s1) - 1.0) < 1e-12, "稳定版和不为 1"
    assert s1 == [1.0, 0.0], "稳定版第二个概率下溢成 0.0"
    try:
        softmax_naive([-1000.0, -1001.0])
    except ZeroDivisionError:
        print("全下溢：softmax_naive([-1000.0, -1001.0]) 抛 ZeroDivisionError")
    else:
        raise AssertionError("全下溢失败：softmax_naive([-1000.0, -1001.0]) 没抛 ZeroDivisionError")
    s2 = softmax([-1000.0, -1001.0])
    assert abs(s2[0] - 1 / (1 + math.exp(-1))) < 1e-15, "全下溢稳定版第一个概率不对"
    assert abs(s2[1] - math.exp(-1) / (1 + math.exp(-1))) < 1e-15, "全下溢稳定版第二个概率不对"
    big, small = softmax([1000.0, 0.0]), softmax([0.0, -1000.0])
    diff = max(abs(a - b) for a, b in zip(big, small, strict=True))
    assert diff < 1e-12, "极端量平移不变失败"
    rev = softmax([0.0, 1000.0])
    assert rev == [0.0, 1.0], f"[0,1000] 应得 [0.0, 1.0]（概率 1.0 在第二位），实际 {rev}"
    far = softmax([1e6, 0.0])
    assert far == [1.0, 0.0], f"[1e6,0] 应得 [1.0, 0.0]（差值 1e6，小项下溢），实际 {far}"
    print(f"病态形状：[0,1000]={rev} | [1e6,0]={far}")
    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的，不是硬编码字符串。"""
    results = [
        judge_shift_invariance(),
        judge_overflow(),
    ]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
