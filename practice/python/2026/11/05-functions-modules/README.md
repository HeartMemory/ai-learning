# 2026-11-05-functions-modules —— 装饰器（def 时就执行）与闭包（记住变量不是值）（S0 · D3）

> 目的：把「`@deco` 只是 `f = deco(f)`、在 `def` 那一刻就执行完」与「闭包记住的是变量外壳，不是值的快照」
> 这两条口径，落到能实测的点上；再用手写 `@timer` 把 `*args / **kwargs` 透传、返回值透传、
> 异常不吞、`functools.wraps` 四件事一次验完。
> 本目录两个文件：`decorators.py`（判据 1-4，四关）+ `_shape_demo.py`（**镜子**：只示范形状，不含判据）。
> 跑法：`uv sync` → `uv run python decorators.py`

## 判据（跑通才算过）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `timer(add)` 后调 `calling(2, 3)` / `calling(2, 3, scale=10)` | 返回值 `5` / `50`；`boom()` 抛的 `ValueError` 能用外层 `try/except` 接到；计时行打印 | ✅ 三条都过（`50` 证明关键字参数穿过了替身） |
| 2 | `timer(add)` vs `timer_wrapped(add)` | `__name__` = `'wrapper'` / `'add'`；`__doc__` = `None` / 原文档；`__wrapped__` 无 / `is add`；`signature` 分别含 `kwargs` / `scale` | ✅ 四项全拉开 |
| 3 | `make_multipliers(3)` 三个闭包各传 2 | 晚绑定 ⇒ `[6, 6, 6]`（全拿到循环结束时的 `i`） | ✅ `[6, 6, 6]` |
| 4 | `snapshot_by_default(3)` 三个闭包各传 2 | 默认参数当场钉死 ⇒ `[2, 4, 6]` | ✅ `[2, 4, 6]` |
| 5 | `@record` 定义函数后**不调用** | `EVENTS` 里已有"装饰"那一笔、没有"调用"那一笔；调用之后才多出第二笔 | ✅ 两段断言都过 |

**实跑记录（2026-11-05）**：`uv run python decorators.py` → **`判据 4/4 绿`**（输出两行 `Call to add took 0.00 ms`）。

## 变异测试（判据能不能被"变坏"抓出来）

只改内存里的实现（不动源文件），6 次全部变红：

| 变异 | 期望 | 实测 |
|---|---|---|
| `timer` 的替身写成 `fn(*args)`（漏 `**kwargs`） | 判据 1 红 | ✅ 红（`50` → `5`） |
| `timer` 忘了 `return result` | 判据 1 红 | ✅ 红（拿到 `None`） |
| `timer` 用 `try/except Exception` 吞掉异常 | 判据 1 红 | ✅ 红：`timer 把 ValueError 吞掉了` |
| `timer_wrapped` 去掉 `@functools.wraps(fn)`（退化成 `timer`） | 判据 2 红 | ✅ 红 |
| `make_multipliers` 改用默认参数钉死（和 `snapshot_by_default` 同实现） | 判据 3 红 | ✅ 红（`[6,6,6]` → `[2,4,6]`） |
| `record` 把"装饰"那一笔也挪进 `wrapper` 体内 | 判据 5 红 | ✅ 红（`def` 之后 `EVENTS` 是空的） |

## 官方依据（2026-11-05 查证 · 四页）

- **`functools`**：`wraps` = `partial(update_wrapper, wrapped=...)`；`WRAPPER_ASSIGNMENTS` 直接赋值、`WRAPPER_UPDATES`（`__dict__`）用 update 合并；**自动添加 `__wrapped__`** 指向真身，供 `inspect.signature` 等"顺藤摸瓜"。
- **`faq/programming`**：循环里定义 lambda 是**晚绑定**（调用时才去读变量），官方解药 = 默认参数 `lambda n=x: n**2`。
- **`tutorial/controlflow`**：默认参数在**定义时**求值 —— `i = 5; def f(arg=i)` 之后把 `i` 改成 6，`f()` 仍打印 5。
- **`reference/compound_stmts`**：函数定义处的装饰器表达式**在 def 时求值**，`@f1 @f2` 等价于 `f = f1(f2(f))`（自下而上套）。
- ⚠️ **本机与文档的一处不一致**：3.14.8 的 `Lib/functools.py` 里 `WRAPPER_ASSIGNMENTS` 是 `('__module__', '__name__', '__qualname__', '__doc__', '__annotate__', '__type_params__')` —— 是 **`__annotate__`**，不是文档里常写的 `__annotations__`（PEP 649 起注解惰性求值）。**以库源码 + 实测为准。**

## 怎么跑

```powershell
cd practice/python/2026/11/05-functions-modules
uv sync
uv run python decorators.py     # 判据 1-5，末行应为「判据 4/4 绿」
uv run python _shape_demo.py    # 镜子：装饰器形状的示范（无判据）
uvx ruff check . ; uvx ruff format --check .
```

## 踩过的坑

- **`fn(args)` 少一个星号**：把整个 tuple 当成**第 1 个位置参**传进去 ⇒ `TypeError`。定义处 `*args` 是"收"，调用处 `*args` 是"拆"，同一个星号方向相反。
- **替身只收 `*args`、不收 `**kwargs`**：`add(2, 3, scale=10)` 直接 `TypeError: unexpected keyword argument 'scale'`。这种炸得响，反而好办。
- **收下 `**kwargs` 却忘了转发**：`add(*args)` 漏掉 `**kwargs`，`scale` 悄悄回到默认值 1 ⇒ **静默返回 5**，不报错。判据 1 用 `5` vs `50` 把它逼出来。
- **异常路径不会打印耗时**：`result = fn(...)` 先抛，后面的 `print` 永远执行不到 ⇒ 输出里只有两行 `Call to add`，没有 `Call to boom`。想让失败也留一条耗时可改用 `try/finally`（注意 `finally` 里别再吞异常）。
- **`@functools.wraps` 漏写 `(fn)`**：`wrapper` 会被替换成一个 `functools.partial` 对象，**定义时不报错**，调用时才炸。
- **`wraps` 并没有改签名**：`inspect.signature` 是顺着 `__wrapped__` 找到真身再读的（`follow_wrapped=True` 是默认）。所以判据 2 的第 4 项成立靠的是 `__wrapped__`，不是"签名被复制"。
- **`except` 后面空着 = `IndentationError`**：`try/except` 的 `except` 分支至少要有一句（`pass` 也算），否则整个模块都 import 不了。
- **`try/except: pass` 会造出"假绿"**：若装饰器真把异常吞了，`try` 块顺利走完，`except` 分支根本不进 ⇒ `pass` 什么都没干、判据照样绿。写法必须是 `try / except / else`，`else` 里 `raise AssertionError(...)`（ruff **B011** 禁 `assert False`；带 `else` 时 **SIM105** 也不会劝你改 `contextlib.suppress`）。
- **函数体只剩 docstring 是合法语法**：不报错、返回 `None`（当初 `make_multipliers` 就是这个状态）。"能跑"不等于"做对"。
- **ruff 自己就能抓晚绑定**：`lambda x: x * i` 在循环里定义会被 **B023** 报"未绑定循环变量"。本练习是**刻意**保留这个 bug 做对照，所以写 `# noqa: B023` 并注明意图；`snapshot_by_default` 那个版本不该有这条。
- **列表里必须装"函数"，不是"调用结果"**：`append(inner(i))` 或 `append((lambda x: x * i)())` 会立刻求值，列表里全是数字 ⇒ `'int' object is not callable`。
