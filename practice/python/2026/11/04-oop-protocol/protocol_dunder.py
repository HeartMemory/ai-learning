"""S0 · 实验：Protocol（结构化子类型）与 __dunder__（把自己的类接进语言协议）

目的（一条根）：C# 的 interface 是**名义子类型** —— 必须写 `: IClosable` 登记入会，才算会员；
    Python 的 `Protocol` 是**结构化子类型**（PEP 544）—— **形状像就算**，不用继承、不用 import、
    被适配的类完全不知道自己被适配了。代价是：这层判定**默认只在静态检查器里生效**，
    运行时什么都不查 —— 所以 `@runtime_checkable` 只能降级成"成员存在性"的粗糙检查。

判据（跑完把数字填进本目录 README 的实测列）：
  1. plain_protocol_is_static_only() → 未加 @runtime_checkable 的协议用于 isinstance ⇒ TypeError
  2. runtime_checkable_presence()   → Duck 通过；**签名全错的 Pretender 也通过** ⇒ 只查成员存在
  3. data_member_presence()         → 缺 name 属性的 NoName 不通过 ⇒ 数据成员也按"存在"算
  4. dunder_len_iter()              → 自己实现 __len__ / __iter__ ⇒ len() / list() / in 全部生效
  5. dunder_context_manager()       → 自己实现 __enter__ / __exit__ ⇒ with 生效；__exit__ 返回
                                      True 吞异常、False 放行（与「返回 None 会怎样」一起验）

跑法：uv run python protocol_dunder.py
文档依据：docs.python.org/3/library/typing.html（Nominal vs structural subtyping /
        runtime_checkable 的五条限制 / 3.12 起 isinstance 改用 getattr_static、成员运行时冻结）
        PEP 544（Protocol 的原始提案：静态鸭子类型）
"""

from collections.abc import Iterator
from typing import Protocol, runtime_checkable


class Raw(Protocol):
    """判据 1 的对照组：**不加** @runtime_checkable —— 只在静态层面生效。"""

    def ping(self) -> str: ...


@runtime_checkable
class Closable(Protocol):
    """判据 2 / 3：加了装饰器 ⇒ 可以 isinstance，但只查"有没有这个成员"。

    注意：协议里既有方法（`close`）又有数据成员（`name`）。
    """

    name: str

    def close(self) -> None: ...


class Duck:
    """判据 2 的正例：**没有**继承 Closable，但形状对 —— 结构化子类型的意义就在这。"""

    name = "duck"

    def close(self) -> None:
        """把"关闭"这件事做成可观测（打印一行），判据 2 会真的调一次。"""
        print(f"{self.name} closed")


class Pretender:
    """判据 2 的反例：有 `close`，但**签名完全不对**（多一个必填参数、返回类型也不同）。

    如果 isinstance 仍然放行 ⇒ 证明 runtime_checkable 只看"名字在不在"。
    """

    name = "pretender"

    def close(self, force: bool) -> int:
        """故意写错签名：runtime_checkable 不会因此拦下。"""
        print(f"{self.name} closed (force: {force})")
        return 0


class NoName:
    """判据 3 的反例：有 `close` 但**没有** `name` ⇒ 数据成员缺失，应当不通过。"""

    def close(self) -> None:
        """签名对、成员缺一个。"""
        print(f"{self.name} closed")


class Counter:
    """判据 4：不继承任何东西，只实现两个 dunder —— `len()` / `list()` / `in` 就会认它。"""

    def __init__(self, n: int) -> None:
        """存下上界（例如 3 表示 1, 2, 3）。"""
        self.n = n

    def __len__(self) -> int:
        """返回元素个数。"""
        return self.n

    def __iter__(self) -> Iterator[int]:
        """产出 1..n —— 生成器或 return iter(...) 都行。"""
        yield from range(1, self.n + 1)


class Managed:
    """判据 5：上下文管理器协议 —— `with` 语句只认这两个 dunder，跟"是不是文件"无关。"""

    def __init__(self, swallow: bool = False) -> None:
        """`swallow=True` ⇒ __exit__ 返回真、吞异常；默认 False ⇒ 放行。"""
        self.swallow = swallow

    def __enter__(self) -> Managed:
        """进入 with 时被调，返回值绑定到 as 后面的名字（打印一行证明被调过）。"""
        print("Managed entered")
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: object
    ) -> bool:
        """退出时被调。**返回值决定 with 体内的异常吞不吞**（True 吞 / False 或 None 放行）。"""
        name = exc_type.__name__ if exc_type else None
        print(f"Managed exited (收到异常: {name})")
        return self.swallow


def plain_protocol_is_static_only() -> None:
    """判据 1：`isinstance(Duck(), Raw)` ⇒ 应当抛 `TypeError`。

    这就是"Protocol 默认不参与运行时"的硬证据（异常消息里会提到 @runtime_checkable）。
    用 try/except/else 三段式断言异常类型。
    """
    isinstance_raised = False
    try:
        isinstance(Duck(), Raw)
    except TypeError:
        isinstance_raised = True
    else:
        isinstance_raised = False
    assert isinstance_raised, "Raw 没有 @runtime_checkable，isinstance 应该抛出 TypeError"


def runtime_checkable_presence() -> None:
    """判据 2：
    ① `isinstance(Duck(), Closable)` ⇒ True，并调一次 `duck.close()` 证明它能用
    ② `isinstance(Pretender(), Closable)` ⇒ **也 True**（签名错到离谱照样放行）
    ⇒ 结论一句话写进注释：runtime_checkable 只做"成员存在性"检查，不看签名、不看类型。
    """
    duck = Duck()
    assert isinstance(duck, Closable), "Duck 应当是 Closable"
    duck.close()  # 调一次证明它能用

    pretender = Pretender()
    assert isinstance(pretender, Closable), "Pretender 也应当是 Closable"
    pretender.close(force=True)  # 调一次证明它能用


def data_member_presence() -> None:
    """判据 3：`isinstance(NoName(), Closable)` ⇒ False。

    顺带对照：`hasattr(NoName(), "close")` 为 True —— 说明不是"有没有方法"的问题，
    而是缺了协议声明的**数据成员** `name`。
    """
    noname = NoName()
    assert not isinstance(noname, Closable), "NoName 不应当是 Closable"
    has_close = hasattr(noname, "close")
    assert has_close, "NoName 应当有 close 方法"
    has_name = hasattr(noname, "name")
    assert not has_name, "NoName 不应当有 name 数据成员"


def dunder_len_iter() -> None:
    """判据 4：`c = Counter(3)` 之后断言
    ① `len(c) == 3`（__len__）
    ② `list(c) == [1, 2, 3]`（__iter__）
    ③ `2 in c` 为 True、`9 in c` 为 False（`in` 走 __iter__ 或 __contains__）
    ④ 想一句：`len(c)` 为什么要求 `__len__` 返回**非负整数**，返回 `-1` 会怎样？
    """
    c = Counter(3)
    assert len(c) == 3, "len(c) 应当返回 3"
    assert list(c) == [1, 2, 3], "list(c) 应当返回 [1, 2, 3]"
    assert 2 in c, "2 应当在 c 中"
    assert 9 not in c, "9 不应当在 c 中"


def dunder_context_manager() -> None:
    """判据 5：
    ① `with Managed() as m:` 体内代码正常执行，`m` 就是 `__enter__` 的返回值
    ② 体内 `1 / 0` → `__exit__` 返回 **True 时异常被吞**（with 之后照常往下走）
    ③ 体内 `1 / 0` → `__exit__` 返回 **False/None 时异常放行**（外面必须 try 住）
    ⇒ 用两个行为不同的子类（或改造成可配置参数）来验，别只验一条。
    """
    with Managed() as m:
        assert m is not None, "m 应当是 __enter__ 的返回值"
        print("Inside with block")

    # ② 吞：__exit__ 返回 True ⇒ 异常被吞，下面那行才会执行
    with Managed(swallow=True):
        raise ZeroDivisionError("判据 5 故意抛：返回 True 时应当被吞")
    print("② 走到了这里 —— 说明异常被吞掉了")

    # ③ 放行：__exit__ 返回 False ⇒ 异常抛到外面，必须被这里接住
    caught = False
    try:
        with Managed(swallow=False):
            raise ZeroDivisionError("判据 5 故意抛：返回 False 时应当放行")
    except ZeroDivisionError:
        caught = True
    else:
        raise AssertionError("__exit__ 返回 False 时异常应该放行")
    assert caught, "③ 异常应当被外层接住"


def main() -> None:
    plain_protocol_is_static_only()
    runtime_checkable_presence()
    data_member_presence()
    dunder_len_iter()
    dunder_context_manager()
    print("判据 5/5 绿")


if __name__ == "__main__":
    main()
