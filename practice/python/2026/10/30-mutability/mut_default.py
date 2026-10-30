"""可变性陷阱 · 实验一：默认参数

目的：证明 `def f(a, L=[])` 里的 `[]` 只在 `def` 执行时创建一次，之后被所有调用共享。

判据（三条都要看到）：
  1. f(1) / f(2) / f(3) 的返回值逐次累积（[1] → [1, 2] → [1, 2, 3]）
  2. `f.__defaults__[0]` 的 id 三次调用前后**相同**
  3. g 的三次调用**互不影响**，且 `g.__defaults__` 始终是 `(None,)`

跑法：uv run python mut_default.py（纯标准库，裸 python 也能跑）
"""


def f(a: int, L: list[int] = []) -> list[int]:
    """故意留的坏写法：先跑它、看现象，别急着改。"""
    L.append(a)
    return L


def g(a: int, L: list[int] | None = None) -> list[int]:
    """正确写法 —— 方法体留给你写。

    判据：
      1. g(1) 与 g(2) 返回的 list **不是同一个对象**（id 不同）
      2. 三次调用后 `g.__defaults__` 仍是 `(None,)`
      3. 调用方自己传列表时，改的必须是**他传进来的那个对象**（id 相同）
    """
    # TODO: 你的方法体（提示：默认值只是"调用方没给"的信号，不是容器本身）


if __name__ == "__main__":
    # ── 实验一：坏写法的现象 ─────────────────────────
    print(f(1))
    print(f(2))
    print(f(3))
    print("f.__defaults__ =", f.__defaults__)
    print("id(f.__defaults__[0]) =", id(f.__defaults__[0]))

    # ── TODO：换成 g 再跑一遍（写完方法体再取消注释）──
    # print(g(1))
    # print(g(2))
    # print(g(3))
    # print("g.__defaults__ =", g.__defaults__)
    # mine = [99]
    # print(g(7, mine), "mine is 同一个对象：", mine is g(0, mine))
