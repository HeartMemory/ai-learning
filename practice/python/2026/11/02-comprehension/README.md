# 2026-11-02-comprehension —— 推导式到底是什么（S0 缺口：容器整形）

> 目的：把「推导式 = 换了视角描述同一件事」这条根，落到「list / dict / set 三种筐 + 独立作用域」四个可测的点上。
> 对应笔记：`Notes/Python-基础笔记.md`（推导式 / 切片 一节）
> 跑法：`uv run --no-project python comprehension.py`（纯标准库，不用 `uv sync`）

## 判据（跑通才算过，实测数字填右侧）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `even_squares([1, 2, 3, 4])` | 返回 `[4, 16]`：`if` 做**过滤**（安检门），只留偶数再平方，顺序不变 | ✅ `[4, 16]` |
| 2 | `even_squares([])` | 空输入返回空列表，**不报错**（`for` 零次迭代即零次产出） | ✅ `[]`（未写任何 guard） |
| 3 | `even_squares([2])` | 单元素仍返回列表（不是标量）—— 推导式**永远返回容器** | ✅ `[4]` |
| 4 | `index_by_id(sample)` | 键 = `item["id"]`，值 = `item`；结果与期望 dict **内容相等** | ✅ 内容相等 |
| 5 | 同上，`got["a"] is sample[0]` | 值必须是**原对象**（`is` 为 `True`）—— 推导式不复制元素，只搬运引用 | ✅ `is` → `True` |
| 6 | `unique_domains([...3 条 URL...])` | 返回 `{"a.com", "b.com"}`：`set` 自动去重（3 条 URL → 2 个域名） | ✅ `3 → 2`（`{"a.com", "b.com"}`） |
| 7 | `unique_domains([])` | 空输入返回空 set；注意**不能写 `{}`**（`{}` 是空 dict） | ✅ 空 **set**（不是 `{}`） |
| 8 | `n = 0` 后跑 `[n for n in range(3)]` | 结束后 `n` 仍是 `0` —— 推导式**不泄漏**迭代变量 | ✅ `n == 0` |
| 9 | `for n in range(3)` 后看 `n` | `n` 变成 `2` —— **`for` 语句会泄漏**，这是第 8 条的反面 | ✅ `n == 2` |

**第二块：从「一个 for」升级到「多个 for」+ `if` 的两种位置**

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 10 | `flatten([[1, 2], [3], []])` | 多重 `for` → `[1, 2, 3]`；书写顺序 = 等价双层循环顺序（**外层先写**） | ✅ `[1, 2, 3]` |
| 11 | `label_parity([1, 2, 3])` | **前置三元 = 分流**：`[("odd", 1), ("even", 2), ("odd", 3)]` | ✅ 完全一致 |
| 12 | 与 `even_squares([1, 2, 3])` 比长度 | 分流 **3** ≠ 过滤 **1**：`if` 只差位置，语义差一整类 | ✅ **3 vs 1** |
| 13 | `transpose([[1, 2, 3], [4, 5, 6]])` | 嵌套推导式 → `[[1, 4], [2, 5], [3, 6]]` | ✅ `[[1, 4], [2, 5], [3, 6]]` |
| 14 | 同上 vs `[list(t) for t in zip(*m)]` | 两者结果**完全一致**（官方推荐用 `zip`，练嵌套是为了看懂它） | ✅ 一致 |
| 15 | `transpose([])` | 返回 `[]` —— **此处 guard 是必要的**（要取 `matrix[0]` 算列数，空会 `IndexError`） | ✅ `[]`（加 guard 前是 `IndexError`） |
| 16 | `compile("[x, x**2 for x in range(3)]")` | 抛 **`SyntaxError`**：元组表达式**必须加括号** | ✅ `SyntaxError` |
| 17 | `[(x, x**2) for x in range(3)]` | 加括号即合法 → `[(0, 0), (1, 1), (2, 4)]` | ✅ `[(0, 0), (1, 1), (2, 4)]` |

**实跑记录（2026-11-02）**：`uv run --no-project python comprehension.py` → 末行 **`判据 17/17 绿`**（第一块先跑到 9/9，第二块补完到 17/17）；`uvx ruff check .` → `All checks passed!`；`uvx ruff format .` → 补了 2 处顶层函数间空行（原文件只有 1 行，`ruff check` 默认规则**查不出来**，只有 `ruff format` 管）。

## 官方依据（2026-11-02 查证 · 三条来源，两条独立互证）

- **教程 5.1.3 / 5.1.4**（`docs.python.org/3.14/tutorial/datastructures.html`）：语法 = 方括号 + 一个表达式 + **至少一个 `for`** + 零或多个 `for`/`if`；原文明确「推导式中 `for` / `if` 的书写顺序与等价嵌套循环**完全一致**」（自左向右由外向内）；元组表达式**必须加括号**（`[x, x**2 for ...]` 直接 `SyntaxError`）；嵌套推导式的内层在**紧随其后那个 `for` 的上下文**中求值。转置优先 `list(zip(*matrix))` 而非嵌套推导式 —— 官方建议。
- **语言参考 §6.2.5 Displays**（`docs.python.org/3.14/reference/expressions.html`）：除**最左 `for` 的可迭代对象**外，整个推导式在**独立隐式嵌套作用域**中执行 ⇒ 迭代变量不泄漏到外层（判据 8）；最左可迭代对象反而在外层求值。dict 推导式**重复键静默覆盖、不报错**（后者胜）；**3.8 起 key 先于 value 求值**（之前未定义，CPython 实为先 value 后 key）。
- **PEP 709 Inlined comprehensions（3.12 起，Final）**：list / dict / set 推导式**已内联**，**不再创建独立函数帧与一次性函数对象**；隔离语义**未变**（靠 `LOAD_FAST_AND_CLEAR` 保存/恢复外层同名变量）。微基准 ≈ **1.96x**，pyperformance ≈ **11%**。⇒ 「推导式比 `for` + `append` 快」仍成立，但**「因为编译成独立 `<listcomp>` 帧」是 3.12 前的老说法**，别再这么写。

## 怎么跑

```powershell
cd practice/python/2026/11/02-comprehension
uv run --no-project python comprehension.py
uvx ruff check .          # 注意是 uvx，不是 uv run（见下方坑 1）
```

九条判据都在 `main()` 里，跑一次全出；顺序执行 ⇒ 第一个红就停，改完再跑。

## 踩过的坑

- **`uv run --no-project ruff check .` 报 `Failed to spawn: ruff`（program not found）**：`--no-project` 表示「不加载任何工程环境」，自然也没有 ruff 可执行文件。纯标准库练习没建 `pyproject.toml`，所以静态检查要用 **`uvx ruff check .`**（临时环境跑一次，不落盘）。
- **`{}` 是空 dict，不是空 set**：空集只能 `set()`。但注意 **空推导式 `{x for x in []}` 返回的仍是 `set`** —— 字面量才是 dict，推导式看花括号里有没有冒号。判据 7 就是卡这条：`{} == set()` 为 `False`（类型都不同）。
- **手切 URL 能过判据，过不了真实数据**：`split("//")[1].split("/")[0]` 在 5 类边界上会失真（2026-11-02 实测对照标准库）——
  | 输入 | 手切结果 | `urlsplit(url).hostname` |
  |---|---|---|
  | `HTTP://A.COM/x` | `"A.COM"`（大小写不归一） | `"a.com"` |
  | `http://user@a.com:8080/x` | `"user@a.com:8080"`（混入用户信息+端口） | `"a.com"` |
  | `http://` | `""`（**静默**产出空域名） | `None`（可区分"不存在"） |
  ⇒ 练习里保留手切（练的是 `split` 链与索引），**工程里一律 `urlsplit().hostname`**。
- **dict 推导式的重复键静默覆盖，不报错**：两个 `item["id"]` 相同时后者顶掉前者，数据悄悄变少。想守住必须在判据里**数条数**（`len(got) == len(items)`），别指望异常提示。
- **键别写成 `item.get("id")`**：缺键时 `.get` 返回 `None` 当键，多个缺失会**互相覆盖**，把「脏数据」变成「正常输出」；`item["id"]` 的 `KeyError` 才是该有的显式失败。
- **多重 `for` 写反是 `NameError`，不是 `TypeError`**：`[n for n in row for row in matrix]` 实测报 `NameError: name 'row' is not defined` —— 因为**最左 `for` 的可迭代对象在外层作用域求值**，那一刻 `row` 还没绑定。不是"数据错"，是"名字还不存在"，所以别去查数据类型。
- **guard 该不该写，判据是"空输入会在哪个具体操作上炸"**：`even_squares` 里 `if not nums: return []` **多余**（`for` 零次迭代天然安全，写了还会让判据 2 失去区分度）；`transpose` 里 `if not matrix: return []` **必需**（要 `matrix[0]` 取列数 ⇒ 空输入 `IndexError`）。同一招两处相反结论，靠的是具体操作，不是习惯。
- **`ruff check` 全绿 ≠ 格式合规**：顶层函数之间少一个空行（`E302`）**不在 ruff 默认规则集里**，`check` 查不出来 —— 必须再跑 `uvx ruff format .`。本文件第一次提交前就被 `format` 补了 2 处空行。
