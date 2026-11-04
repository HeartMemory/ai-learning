# 2026-11-04-oop-protocol —— dataclass（代码生成器）与 Protocol（结构化子类型）（S0 · D4）

> 目的：把「`@dataclass` 只生成方法、不校验类型」与「`Protocol` 默认只在静态层面生效」两条官方口径，
> 落到能实测的点上；再用 `__dunder__` 把类接进 `len` / `for` / `in` / `with` 四个语言协议。
> 本目录两个文件：`dataclass_basics.py`（判据 1-5）+ `protocol_dunder.py`（判据 1-5，含 dunder）。
> 跑法：`uv sync` → `uv run python dataclass_basics.py` / `uv run python protocol_dunder.py`

## 判据（跑通才算过）

### `dataclass_basics.py`

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `Plain("a")` 两实例 | `==` 为 True、`is` 为 False、`repr` 含字段名（证明方法是**生成**的） | ✅ 三条断言过 |
| 2 | `Plain(name=123, retries="很多")` | **能构造成功** —— 注解不产生运行时约束 | ✅ 构造成功（Pylance 报红 = 静态门，故意保留） |
| 3 | `TokenUsage(38, 58)` / `TokenUsage(-1, 0)` | `total_tokens == 96`（`field(init=False)` 派生）；负数 → `ValueError` | ✅ 96；`ValueError` 被 `except` 抓住 |
| 4 | `Batch()` 两次 vs 临时坏类 | `items is not` 各自独立；写 `= []` 的类**装饰时**抛 `ValueError` | ✅ 独立；`mutable default ... not allowed` |
| 5 | `frozen=True` 的 `Settings` | 赋值 → `FrozenInstanceError`；同值两实例进 `set` 只剩 1 个 | ✅ 两条断言过 |

### `protocol_dunder.py`

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `isinstance(Duck(), Raw)`（`Raw` **无** `@runtime_checkable`） | 抛 `TypeError` —— 协议默认不参与运行时 | ✅ 抛 |
| 2 | `isinstance(Duck(), Closable)` / `isinstance(Pretender(), Closable)` | 都 True；`Pretender.close` 签名完全不符也放行 | ✅ 两个都 True |
| 3 | `isinstance(NoName(), Closable)` | False；对照 `hasattr(NoName(), "close")` 为 True ⇒ 缺的是**数据成员** | ✅ False / True |
| 4 | `Counter(3)`（只实现 `__len__` / `__iter__`） | `len == 3`、`list == [1, 2, 3]`、`2 in` / `9 not in` | ✅ 四条断言过 |
| 5 | `Managed(swallow=?)` 的 `with` | 返回 True ⇒ 异常被吞（后续语句仍执行）；返回 False ⇒ 异常放行（外层 `try` 接住） | ✅ 吞/放各验一次 |

**实跑记录（2026-11-04）**：`uv run python dataclass_basics.py` → **`判据 5/5 绿`**；`uv run python protocol_dunder.py` → **`判据 5/5 绿`**（输出含 `Managed exited (收到异常: None)` 与 `(收到异常: ZeroDivisionError)` 两组）。

## 变异测试（判据能不能被"变坏"抓出来）

绿色本身不证明判据有效 —— 用**故意改坏**的方式反验，5 次全部变红：

| 变异 | 期望 | 实测 |
|---|---|---|
| 摘掉 `__post_init__` 里的 `raise ValueError` | 判据 3 红 | ✅ `AssertionError: 构造负数 TokenUsage 应该抛出 ValueError` |
| `@dataclass(frozen=True)` → `@dataclass` | 判据 5 红 | ✅ `AssertionError: 修改 frozen 实例的字段应该抛出 FrozenInstanceError` |
| 给 `Raw` 加 `@runtime_checkable` | 判据 1 红 | ✅ `AssertionError: Raw 没有 @runtime_checkable，isinstance 应该抛出 TypeError` |
| `__exit__` 恒返回 `True` | 判据 5③ 红 | ✅ `AssertionError: __exit__ 返回 False 时异常应该放行` |
| `__exit__` 恒返回 `False` | 判据 5② 红 | ✅ `ZeroDivisionError` 冒出去（没被吞） |

> 起因：`main()` 里的 `print("判据 5/5 绿")` 是**硬编码字符串**，函数缺断言时它照样印绿（当天撞了两次假绿）。

## 官方依据（2026-11-04 查证 · 两页 + PEP）

- **`dataclasses`**：装饰器返回**同一个类**（`slots=True` 除外），按字段顺序生成 `__init__` / `__repr__` / `__eq__`（`order` 才给比较家族、`frozen` 注入 `__setattr__`）；`__post_init__` 由**生成的** `__init__` 末尾调用（自己写 `__init__` 或 `init=False` 时不调）；**除 `ClassVar` / `InitVar` 两处外完全不检查注解类型**；可变默认值自 **3.11** 起按「不可哈希即禁」检测，在**装饰时**抛 `ValueError`；`frozen` 的 `__init__` 内部赋值须走 `object.__setattr__`。
- **`typing`**（Nominal vs structural subtyping）：PEP 544 引入结构化子类型；**未加 `@runtime_checkable` 的协议不能用于 `isinstance` / `issubclass`**；加了之后**只检查成员存在，不检查签名与类型**，且比普通类的 `isinstance` **明显更慢**；数据成员也按"存在"判定；**可检查性不被子协议继承**；3.12 起改用 `inspect.getattr_static` 且成员在运行时"冻结"；3.15 起对"未显式加装饰器的子协议"做 `isinstance` 已弃用（3.20 抛 `TypeError`）。
- **`datamodel`**：特殊方法由**类型**决定（`len(x)` → `type(x).__len__`）；`with` 的等价展开为 `if not exit_(manager, *sys.exc_info()): raise` ⇒ **`__exit__` 返回值真假决定吞不吞**。

## 怎么跑

```powershell
cd practice/python/2026/11/04-oop-protocol
uv sync
uv run python dataclass_basics.py    # 判据 1-5
uv run python protocol_dunder.py     # 判据 1-5
uvx ruff check . ; uvx ruff format --check .
```

## 踩过的坑

- **「注解不校验」说的是运行时，静态检查器照样报红**：`Plain(name=123)` 在 Pylance 里是 `reportArgumentType`，但 CPython **3.14.8 实测构造成功**（`P(name=123, n=0) int`）。两套门别混：注解 → 静态检查器 → 运行时，只有最后一层能拦，而那一层要你自己写。
- **`assert` 会被 `python -O` 整条删掉**：实测 `-O` 下 `assert False` 直接消失。⇒ 练习里的判据用 `assert` 没问题，**库里的输入校验必须 `raise`**（判据里的反例用 `try/except/else` + `raise AssertionError`，ruff **B011** 也禁止 `assert False`）。
- **`except X: pass` 后接 `else` 时 SIM105 不触发**：`contextlib.suppress` 覆盖不到"带 else 的三段式"，所以不用为它改写法。
- **可变默认值的报错时机是"装饰时"**：`try` 必须包住**整个 `class` 定义**（ruff **RUF008** 会先一步在静态层面拦，故意违规要写 `# noqa: RUF008`）。
- **`field(init=False)` 无默认值时属性"不存在"**：构造成功、`repr` 才崩（`AttributeError: ... has no attribute 'total'`）；补 `default=0` 才会在 `__post_init__` 之前先有值。
- **`default_factory` 与"输入类型"无关**：它只在**没传值**时被调一次（实测：显式传值时工厂调用次数不增），也没有任何类型强制（传 `tuple` / `None` 都行，报应发生在用 `list` 方法的那一刻）。
- **`...` 在协议里是对的，在实现类里是静默 bug**：`...` 是 `Ellipsis`，**真值** —— `__exit__` 返回 `...` 会被当成"吞掉异常"。另 `def f() -> int: ...` 实际返回 `None`，注解连自己都不保证。
- **`__exit__` 的 `tb` 不是 `object`**：真实类型是 `types.TracebackType`，`traceback.format_exception(exc_type, exc, tb)` 要用它。
- **给类加必填字段会打破已有调用点**：`Managed` 加 `swallow` 必须给 `= False` 默认值，否则原来无参的 `Managed()` 直接 `TypeError`。
- **裸 `1 / 0` 被 ruff 判无用表达式（B018）**：改 `raise ZeroDivisionError("...")`（意图更清楚）或 `_ = 1 / 0`。
- **3.14 起注解引号多余**：`-> "Managed"` 会被 ruff **UP037** 强制去掉（PEP 649 延迟求值）。
