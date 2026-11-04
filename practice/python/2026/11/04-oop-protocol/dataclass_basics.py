"""S0 · 实验：dataclass —— 模板代填了什么 / 注解不是门禁 / __post_init__ 是唯一的校验钩子

目的（一条根）：`@dataclass` 是**代码生成器**，不是校验器 —— 它只根据字段注解替你生成
    `__init__` / `__repr__` / `__eq__`（可选 `__lt__` 家族 / `__hash__` / `__setattr__`），
    **除 ClassVar 与 InitVar 两处外完全不看注解里写的类型**。所以"带校验的数据模型"必须另找门 ——
    那个门就是 `__post_init__`（生成的 `__init__` 末尾会调它一次）。

⚠️ **两套门别混**：「不校验」说的是**运行时**（已用 CPython 3.14.8 实测：`P(name=123)` 构造成功）；
    **静态检查器**（Pylance / Pyright / mypy）会读同一份注解在编辑器里报红，那是**编译期**的门。
    判据 2 是**故意违规** ⇒ 那行红字是预期现象，用 `# pyright: ignore[reportArgumentType]` 静音
    （并在注释里写明"刻意违规"），别为了消红字把判据改掉。

判据（跑完把数字填进本目录 README 的实测列）：
  1. generated_methods()          → 同值两实例 ==、repr 含字段名、两实例 is not
  2. annotations_are_just_notes() → Plain(name=123) **不报错** —— 证伪"注解会校验类型"
  3. post_init_validates()        → 负数 → ValueError；total_tokens 构造后已算好
  4. default_factory_not_shared() → Batch() 两次的 items is not；写 `= []` 的类**装饰时** ValueError
  5. frozen_and_hashable()        → 赋值抛 FrozenInstanceError；frozen+eq ⇒ 可哈希，进 set 只剩 1 个

跑法：uv run python dataclass_basics.py
文档依据：docs.python.org/3/library/dataclasses.html（生成物清单 / __post_init__ 调用时机 /
        3.11 起「不可哈希即禁」的可变默认值检测 / frozen 注入 __setattr__ / __hash__ 生成规则）
"""

from dataclasses import FrozenInstanceError, dataclass, field


@dataclass
class Plain:
    """判据 2 的对照组：有注解、**没有** __post_init__ ——注解只是便签。"""

    name: str
    retries: int = 0


@dataclass
class TokenUsage:
    """一次模型调用的 usage —— 参照 10-29 实测的 prompt / completion / total 三个数。

    判据 3 要在这里落地两件事：
      ① `total_tokens` 由前两个字段**派生**（`field(init=False)`：不进 __init__ 参数表，
         所以调用方传不进来，只能由 __post_init__ 填）
      ② 任何字段为负 ⇒ 当场 `raise ValueError`（不许"先存进去、以后再说"）
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int = field(init=False)

    def __post_init__(self) -> None:
        """校验 + 派生。想清楚顺序：先校验还是先派生？哪一步失败该拦下整个构造？"""
        ...
        total = self.prompt_tokens + self.completion_tokens
        if self.prompt_tokens < 0 or self.completion_tokens < 0:
            raise ValueError("prompt_tokens 和 completion_tokens 都必须非负")
        self.total_tokens = total


@dataclass
class Batch:
    """判据 4 的正例：可变默认值必须走 default_factory。"""

    items: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Settings:
    """判据 5：frozen —— 字段列表就是"配置快照"的形状（model 名 + 温度）。"""

    model: str
    temperature: float = 0.0


def generated_methods() -> None:
    """判据 1：`Plain("a")` 造两个同值实例，断言
    ① 两实例 `==` 为 True（__eq__ 是生成的）
    ② `is` 为 False（不是一个对象）
    ③ `"name=" in repr(p)`（__repr__ 是生成的，且带字段名）
    """
    ...
    plain_a = Plain("a")
    plain_b = Plain("a")
    assert plain_a == plain_b, "同值实例应该相等"
    assert plain_a is not plain_b, "不同实例应该不是同一个对象"
    assert "name=" in repr(plain_a), "__repr__ 应该包含字段名"


def annotations_are_just_notes() -> None:
    """判据 2：故意违规构造 —— `Plain(name=123, retries="很多")`，断言**能构造成功**
    且 `p.name == 123`、`p.retries == "很多"`。
    ⇒ 结论一句话写进注释：注解不产生任何运行时约束或转换。
    ⇒ 构造那行 Pylance 会报 `reportArgumentType` 红字 = **预期**（静态门），
      在该行行尾加 `# pyright: ignore[reportArgumentType]` 静音，别改判据。
    """
    plain = Plain(name=123, retries="很多")  # pyright: ignore[reportArgumentType]
    assert plain.name == 123, "name 字段应该被正确赋值"
    assert plain.retries == "很多", "retries 字段应该被正确赋值"


def post_init_validates() -> None:
    """判据 3：
    ① `TokenUsage(38, 58)` → `total_tokens == 96`（派生成功）
    ② 构造 `TokenUsage(-1, 0)` 抛 `ValueError`（校验生效）—— 用 try/except 抓住并断言异常类型
    ③ 想一句：如果没写 `field(init=False)`，调用方传 `total_tokens=999` 会怎样？
    """
    ...
    token_usage = TokenUsage(38, 58)
    assert token_usage.total_tokens == 96, "total_tokens 应该被正确派生"
    try:
        TokenUsage(-1, 0)
    except ValueError as e:
        assert "非负" in str(e), "应该抛出 ValueError"
    else:
        raise AssertionError("构造负数 TokenUsage 应该抛出 ValueError")


def default_factory_not_shared() -> None:
    """判据 4：
    ① `Batch()` 造两个，断言 `a.items is not b.items`（各自独立）
    ② 在函数体里临时定义一个**故意错误**的 dataclass（字段 `list` 直接默认 `[]`），
       用 try/except 断言它抛 `ValueError` —— 记住报错时机是**装饰（类创建）时**，不是实例化时
    """
    ...
    batch_a = Batch()
    batch_b = Batch()
    assert batch_a.items is not batch_b.items, "items 列表应该是独立的"
    try:

        @dataclass
        class BadBatch:
            items: list[str] = []  # 直接写默认值，应该在装饰时抛 ValueError # noqa: RUF008
    except ValueError as e:
        assert "mutable default" in str(e), "应该抛出 ValueError，提示可变默认值"
    else:
        raise AssertionError("定义 BadBatch 应该抛出 ValueError，实际没抛")


def frozen_and_hashable() -> None:
    """判据 5：
    ① `s = Settings("deepseek-flash")`；`s.temperature = 0.7` 抛 `FrozenInstanceError`
       （它继承自 AttributeError，注意别被宽泛的 except 先抓走）
    ② 同值两实例放进 `set`，断言 `len(...) == 1`（frozen+eq ⇒ 自动生成 __hash__）
    ③ 顺带：`hash(a) == hash(b)`
    """
    ...
    s = Settings("deepseek-flash")
    try:
        s.temperature = 0.7
    except FrozenInstanceError:
        pass  # 预期抛出 FrozenInstanceError
    else:
        raise AssertionError("修改 frozen 实例的字段应该抛出 FrozenInstanceError")
    s1 = Settings("deepseek-flash")
    s2 = Settings("deepseek-flash")
    settings_set = {s1, s2}
    assert len(settings_set) == 1, "同值 frozen 实例放进 set 应该只剩一个"
    assert hash(s1) == hash(s2), "同值 frozen 实例的 hash 应该相等"


def main() -> None:
    generated_methods()
    annotations_are_just_notes()
    post_init_validates()
    default_factory_not_shared()
    frozen_and_hashable()
    print("判据 5/5 绿")


if __name__ == "__main__":
    main()
