"""S0 · 函数与模块（D3）：装饰器 / 闭包 / wraps —— 9 项 TODO，一项 = 一个函数

目的（一条根）：`@timer` 就是 `f = timer(f)`，在 `def` 那一刻就执行完。所以装饰器 = 交出一个
    **替身**：接住任意参数 → 展开传给真身 → 把返回值交回去。闭包（替身记得住外层 fn）和
    functools.wraps（替身要挂着真身的工牌）都是围绕这一句话展开的。

════════════════════════════════════════════════════════════════════════
 TODO 清单：**一项 = 一个函数 = 一个整体**，把函数体里那句 raise 换成你的实现即可
════════════════════════════════════════════════════════════════════════
 T0  准备        加三行 import（functools / inspect / time）
 T1  关卡1 [实现] timer(fn)                计时装饰器：透参 / 透返回值 / 不吞异常
 T2  关卡1 [判据] judge_timer()            四条 assert
 T3  关卡2 [实现] timer_wrapped(fn)        T1 的逐字副本 + 一枚 @functools.wraps
 T4  关卡2 [判据] judge_wraps()            四项元数据对比
 T5  关卡3 [实现] make_multipliers(n)      晚绑定闭包（故意不钉 i）
 T6  关卡3 [实现] snapshot_by_default(n)   同一件事，用默认参数钉死当轮 i
 T7  关卡3 [判据] judge_closure()          [6,6,6] / [2,4,6]
 T8  关卡4 [实现] record(fn)               装饰时记一笔、调用时记一笔
 T9  关卡4 [判据] judge_definition_time()  证明"def 时就已经装饰完"
════════════════════════════════════════════════════════════════════════
 写完一项跑一项（别等九项全写完）：
   uv run python -c "from decorators import judge_timer as f; print(f())"
   uv run python -c "from decorators import judge_wraps as f; print(f())"
   uv run python -c "from decorators import judge_closure as f; print(f())"
   uv run python -c "from decorators import judge_definition_time as f; print(f())"
 全绿后再跑总入口：uv run python decorators.py

跑法：uv run python decorators.py
文档依据：docs.python.org/3/library/functools.html（update_wrapper / WRAPPER_ASSIGNMENTS /
        __wrapped__）
        docs.python.org/3/faq/programming.html（循环里 lambda 的晚绑定，`lambda n=x` 是解药）
        docs.python.org/3/tutorial/controlflow.html（默认参数在**定义时**求值）
        docs.python.org/3/reference/compound_stmts.html（装饰器等价于 `f = f1(f2(f))`）
════════════════════════════════════════════════════════════════════════
"""

import functools
import inspect
import time
from collections.abc import Callable
from typing import Any

EVENTS: list[str] = []
"""T8 / T9 的观测台：装饰记一笔、调用记一笔，两笔文字要能区分。"""


def add(a: int, b: int, *, scale: int = 1) -> int:
    """被装饰的真身，现成的，不用改。仅关键字参数 `scale` 专门用来测参数透传。"""
    return (a + b) * scale


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  关卡 1 · 计时装饰器                                                  ║
# ╚══════════════════════════════════════════════════════════════════════╝


def timer(fn: Callable[..., Any]) -> Callable[..., Any]:
    """T1 · 计时装饰器 —— 三条一起做到才算完成：

    ① 接住任意参数（位置的和关键字的都要）
    ② 把参数原样展开传给真身 fn，**存下**它的返回值
    ③ 打印这次调用的耗时（毫秒），然后把 ② 存的值 return 出去；异常不许拦

    框架已给：`def wrapper(...)` 与 `return wrapper` 两行不用改，你只写 wrapper 的**体内**。
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timestart = time.perf_counter()
        result = fn(*args, **kwargs)
        timeend = time.perf_counter()
        print(f"Call to {fn.__name__} took {(timeend - timestart) * 1000:.2f} ms")
        return result

    return wrapper


def judge_timer() -> bool:
    """T2 · 判据：T1 合格的四条（写成 assert，全过最后 return True）。

    ① `add(2, 3)` 经 timer 包装后仍是 5（返回值没被吞）
    ② `add(2, 3, scale=10)` 经包装后是 50（关键字参数传到了真身）
    ③ 自己写个 `boom()` 里面 `raise ValueError`，用 `try/except ValueError` 包住
       `timer(boom)()` 的调用 —— 能接到，说明异常没被装饰器吞掉
    ④ 计时那行肉眼可见（不必写成断言）

    提示：先在函数里 `calling = timer(add)`，后面用 `calling` 测 ①②。
    """
    calling = timer(add)
    assert calling(2, 3) == 5
    assert calling(2, 3, scale=10) == 50
    try:

        @timer
        def boom():
            raise ValueError("boom!")

        boom()
    except ValueError:
        pass
    else:
        raise AssertionError("timer 把 ValueError 吞掉了")
    return True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  关卡 2 · 工牌：@functools.wraps 到底补了什么                          ║
# ╚══════════════════════════════════════════════════════════════════════╝


def timer_wrapped(fn: Callable[..., Any]) -> Callable[..., Any]:
    """T3 · T1 的**逐字副本**，唯一差别 = 内层 wrapper 上多一枚 `@functools.wraps(fn)`。

    其余一个字都别改 —— T4 是一次"只差一枚工牌"的干净对照，改多了就归因不了。
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        timestart = time.perf_counter()
        result = fn(*args, **kwargs)
        timeend = time.perf_counter()
        print(f"Call to {fn.__name__} took {(timeend - timestart) * 1000:.2f} ms")
        return result

    return wrapper


def judge_wraps() -> bool:
    """T4 · 判据：两个装饰器只差一枚工牌，四项元数据必须拉开差距。

    先 `bare = timer(add)`、`kept = timer_wrapped(add)`，然后逐条断言：
    ① `bare.__name__ == "wrapper"` 且 `kept.__name__ == "add"`
    ② `bare.__doc__ is None` 且 `kept.__doc__ == add.__doc__`
    ③ `not hasattr(bare, "__wrapped__")` 且 `kept.__wrapped__ is add`
    ④ `inspect.signature` 两者不同：bare 那串里含 `kwargs`，kept 那串里含 `scale`
       （别断言整串完全相等 —— 注解写法受类型影响，太脆）

    顺手看一眼（不进判据）：`inspect.signature(bare, follow_wrapped=False)` 的差别。
    """
    bare = timer(add)
    kept = timer_wrapped(add)
    assert bare.__name__ == "wrapper" and kept.__name__ == "add"
    assert bare.__doc__ is None and kept.__doc__ == add.__doc__
    assert not hasattr(bare, "__wrapped__") and kept.__wrapped__ is add
    sig_bare = inspect.signature(bare)
    sig_kept = inspect.signature(kept)
    assert "kwargs" in str(sig_bare) and "scale" in str(sig_kept)
    return True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  关卡 3 · 闭包记住的是变量，不是值                                    ║
# ╚══════════════════════════════════════════════════════════════════════╝


def make_multipliers(count: int) -> list[Callable[[int], int]]:
    """T5 · 返回 count 个闭包，第 i 个返回 `x * i`（i 从 1 数到 count）。**故意不钉 i**。

    这一项要的就是"错"的结果 —— 三个闭包全拿到循环结束时的 i（晚绑定）。
    """
    result = []
    for i in range(1, count + 1):
        result.append(lambda x: x * i)  # noqa: B023 —— 刻意不绑 i，判据 3 要的就是晚绑定
    return result


def snapshot_by_default(count: int) -> list[Callable[[int], int]]:
    """T6 · 同一件事，但用**默认参数**把当轮的 i 钉死。

    与 T5 只差一个写法；默认参数在**定义时**求值 —— 第 3 章那个坑的解药就是它。
    """
    result = []
    for i in range(1, count + 1):
        result.append(lambda x, i=i: x * i)
    return result


def judge_closure() -> bool:
    """T7 · 判据：晚绑定 ⇒ `[6, 6, 6]`；默认参数钉死 ⇒ `[2, 4, 6]`。

    ① 三个闭包各传 2，`make_multipliers(3)` 的结果必须是 `[6, 6, 6]`
    ② 同样各传 2，`snapshot_by_default(3)` 的结果必须是 `[2, 4, 6]`
    """
    late = make_multipliers(3)
    assert [f(2) for f in late] == [6, 6, 6]
    early = snapshot_by_default(3)
    assert [f(2) for f in early] == [2, 4, 6]
    return True


# ╔══════════════════════════════════════════════════════════════════════╗
# ║  关卡 4 · 装饰发生在 def 时，不是调用时                               ║
# ╚══════════════════════════════════════════════════════════════════════╝


def record(fn: Callable[..., Any]) -> Callable[..., Any]:
    """T8 · 观测装饰器，一笔记录在**装饰时**、一笔记录在**调用时**，两笔文字要能区分。

    关键只有一处：往 EVENTS 里 append 的位置 —— 一句在内层 `def wrapper` **之前**，
    另一句在 wrapper **里面**。位置放对了，T9 才分得出"何时被装饰"。
    """
    EVENTS.append(f"decorated {fn.__name__}")

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        EVENTS.append(f"called {fn.__name__}")
        return fn(*args, **kwargs)

    return wrapper


def judge_definition_time() -> bool:
    """T9 · 判据：装饰在 `def` 时就发生了 —— 函数还没调用，EVENTS 里已经有事发生。

    ① 先 `EVENTS.clear()`；再用 `@record` 在本函数内定义一个空函数
    ② **紧接着**（还没调用它）断言：EVENTS 里已有"装饰"那一笔，且**没有**"调用"那一笔
    ③ 调用它之后再断言："调用"那一笔这时才出现
    """
    EVENTS.clear()

    @record
    def dummy():
        pass

    assert EVENTS == ["decorated dummy"]
    dummy()
    assert EVENTS == ["decorated dummy", "called dummy"]
    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的，不是硬编码字符串。"""
    results = [
        judge_timer(),
        judge_wraps(),
        judge_closure(),
        judge_definition_time(),
    ]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
