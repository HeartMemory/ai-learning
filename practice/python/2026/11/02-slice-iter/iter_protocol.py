"""S0 缺口 · 实验：可迭代 vs 迭代器 —— 一次性与可重复的分界

目的（一条根）：**可迭代（iterable）**是"能给你一个迭代器"的东西（有 `__iter__`）；
    **迭代器（iterator）**是"自己就是那个游标"的东西（`__iter__` 返回**自身** + `__next__`）。
    所以迭代器**一定是**可迭代的，反过来不成立；而迭代器是**一次性**的 —— 走完就空，
    第二次 `for` 一个都拿不到，且**不报错**（这就是静默 bug 的温床）。

判据（跑完把数字填进本目录 README 的实测列）：
  9. iter([1,2,3]) is not 原列表      → 序列的迭代器是**新对象**
 10. iter(it) is it                   → 迭代器的 `__iter__` 返回**自身**（幂等）
 11. list(it) 两次                     → 第一次 [1,2,3]，第二次 **[]**（一次性）；
                                         对照 list(nums) 两次都一样（列表可重复迭代）
 12. len(it) / it[0]                  → 都抛 **TypeError**（迭代器不是序列，没有长度与下标）
 13. count_to(3)                      → [1, 2, 3]（生成器函数，`yield` 自动实现协议）
 14. 生成器耗尽后                      → `iter(g) is g`；`next` 持续抛 **StopIteration**（官方"broken"条款）
 15. Countdown(3)                     → [3, 2, 1]（手写 `__iter__` + `__next__`）
 16. 生成器表达式 (x*x for x in ...)   → `sum()` 消费一次得 14，再 `list()` 得 **[]**

跑法：uv run --no-project python iter_protocol.py
文档依据：docs.python.org/3.14/library/stdtypes.html（Iterator Types：__iter__ 返回自身 /
         __next__ 耗尽抛 StopIteration 且**后续必须持续抛**）
         docs.python.org/3.14/tutorial/classes.html §9.8 Iterators · §9.9 Generators ·
         §9.10 Generator Expressions（"more memory friendly"，专为外层函数立即消费而设计）
"""

from collections.abc import Iterator
from typing import Any, Self, cast


def count_to(n: int) -> Iterator[int]:
    """判据 13-14：生成器函数 —— 数 1 到 n（含 n）。

    TODO(你写)：用 `yield` 写循环，**不要**自己维护 list。
    提示：函数体里出现 `yield` 之后，调用它**不执行函数体**，只返回一个生成器对象；
         函数的状态（循环变量跑到哪）由解释器自动保存 —— 这正是它比手写类省事的地方。
    """
    yield from range(1, n + 1)


class Countdown:
    """判据 15：手写一个迭代器 —— 把生成器自动做的事自己写一遍。

    `__init__` 已给好，你只写协议的两个方法。
    """

    def __init__(self, start: int) -> None:
        self.current = start

    def __iter__(self) -> Self:
        """返回**自身** —— 官方原话 "Return the iterator object itself"。

        `Self` 比 `"Countdown"` 更准：它是 3.11 起的 PEP 673，继承时会自动指向**子类**，
        而写死类名会把子类实例注解成父类。3.14 起注解默认延迟求值（PEP 649），
        本来也不再需要字符串前向引用了。
        """
        return self

    def __next__(self) -> int:
        """TODO(你写)：从 `current` 往下数到 1；数完必须抛 StopIteration。

        提示：先判"还有没有"，没有就 `raise StopIteration`（或 `raise StopIteration()`）；
             有的话先把 current 减一还是先返回，得想清楚 —— 判据要的是 3, 2, 1。
        """
        if self.current <= 0:
            raise StopIteration
        else:
            value = self.current
            self.current -= 1
            return value


def main() -> None:
    # ---- 判据 9：序列的迭代器是新对象 ----
    nums = [1, 2, 3]
    it = iter(nums)
    assert it is not nums, "iter(序列) 必须返回新的迭代器对象"

    # ---- 判据 10：迭代器幂等 ----
    assert iter(it) is it, "迭代器的 __iter__ 必须返回自身"

    # ---- 判据 11：迭代器一次性；列表可重复 ----
    assert list(it) == [1, 2, 3], list(it)
    assert list(it) == [], "同一个迭代器耗尽后再 list 必须为空（且不报错）"
    assert list(nums) == [1, 2, 3]
    assert list(nums) == [1, 2, 3], "列表每次 list() 都给新的迭代器，可重复"

    # ---- 判据 12：迭代器不是序列 ----
    # 注：cast(Any, ...) 是给 Pylance 看的 —— 这里**故意**要它抛 TypeError，
    #     静态分析若拦在前面就测不到运行期行为（同类问题见切片练习的 PLE0643）。
    for label, bad in (
        ("len(it)", lambda: len(cast(Any, it))),
        ("it[0]", lambda: cast(Any, it)[0]),
    ):
        try:
            bad()
        except TypeError:
            pass
        else:
            raise AssertionError(f"{label} 应当 TypeError（迭代器没有长度/下标）")

    # ---- 判据 13-14：生成器函数 ----
    assert list(count_to(3)) == [1, 2, 3], list(count_to(3))
    gen = count_to(2)
    assert iter(gen) is gen, "生成器对象本身就是迭代器"
    assert next(gen) == 1
    assert next(gen) == 2
    for attempt in range(2):
        try:
            next(gen)
        except StopIteration:
            pass
        else:
            raise AssertionError(f"第 {attempt + 1} 次越界 next 应当 StopIteration")

    # ---- 判据 15：自定义迭代器 ----
    assert list(Countdown(3)) == [3, 2, 1], list(Countdown(3))

    # ---- 判据 16：生成器表达式一次性 ----
    expr = (x * x for x in [1, 2, 3])
    assert sum(expr) == 14, "1 + 4 + 9"
    assert list(expr) == [], "生成器表达式被 sum 消费完就空了"

    print("判据 16/16 绿（本文件覆盖 9-16）")


if __name__ == "__main__":
    main()
