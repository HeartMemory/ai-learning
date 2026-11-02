# 2026-11-02-slice-iter —— 切片：新对象 ≠ 深拷贝（S0 缺口：容器整形）

> 目的：把「切片**永远产生新对象**，但元素仍是原对象的**引用**」这条根，落到「原列表不动 / 浅拷贝内层共享 / 切片赋值原地可变长 / 越界规则」四个可测的点上；后半程接「可迭代 vs 迭代器」——**一次性 vs 可重复**的分界。
> 本目录两个文件：`slice_basics.py`（判据 1-8）+ `iter_protocol.py`（判据 9-16）。
> 对应笔记：`Notes/Python-基础笔记.md`（切片 / 迭代器 各一节）
> 跑法：`uv run --no-project python slice_basics.py`（纯标准库，不用 `uv sync`）

## 判据（跑通才算过，实测数字填右侧）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `first_three([1, 2, 3, 4])` | 返回 `[1, 2, 3]`，且 `is not` 原列表、**原列表不变**（切片是读，不是就地切） | ✅ `[1, 2, 3]`；`is not` 成立；原列表仍 `[1, 2, 3, 4]` |
| 2 | `reversed_copy([1, 2, 3, 4])` | 返回 `[4, 3, 2, 1]`，同样是**新对象**；对照 `nums.reverse()`（原地改 + 返回 `None`） | ✅ `[4, 3, 2, 1]`，`is not` 成立 |
| 3 | `every_other([0, 1, 2, 3, 4])` | 返回 `[0, 2, 4]`（步长 2） | ✅ `[0, 2, 4]` |
| 4 | `shallow_copy([[1, 2], [3]])` | **外层 `is not`、内层 `is` 相同**；改内层后原列表跟着变 —— 浅拷贝的铁证 | ✅ 外层新 / 内层同；追加 `99` 后原列表变 `[[1, 2, 99], [3]]` |
| 5 | `replace_middle([1, 2, 3, 4, 5])` | `nums[1:3] = [9]`：**长度 5 → 4**，且返回的 `is` 原列表（**原地改**） | ✅ `[1, 9, 4, 5]`；`is` 原列表成立 |
| 6 | `drop_middle([1, 2, 3, 4])` | `del nums[1:3]` 后 `[1, 4]`，函数返回 **`None`**（原地删除） | ✅ 返回 `None`；列表变 `[1, 4]` |
| 7 | `[1, 2, 3][1:100]` / `[1, 2, 3][100]` | 切片**静默截断**为 `[2, 3]` 不报错；索引 `[100]` 才抛 `IndexError` | ✅ `[2, 3]` / `[1, 2, 3]`；索引越界抛 `IndexError` |
| 8 | `s[::2] = [7, 8]` / `s[::2] = [1, 2, 3]` | 步长 ≠ 1 时必须**等长**：等长合法，不等长抛 **`ValueError`** | ✅ `[7, 0, 8, 0]`；不等长抛 `ValueError` |

**实跑记录（2026-11-02）**：`uv run --no-project python slice_basics.py` → 末行 **`判据 8/8 绿`**；`uvx ruff check .` → `All checks passed!`；`uvx ruff format .` → **又补了 1 处顶层函数间空行**（同 `11/02-comprehension` 那个坑，第二次踩）。

### 第三块：可迭代 vs 迭代器（`iter_protocol.py`）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 9 | `iter([1, 2, 3])` | 返回**新的**迭代器对象（`is not` 原列表） | ✅ `is not` 成立 |
| 10 | `iter(it) is it` | 迭代器的 `__iter__` 返回**自身**（幂等）—— 所以迭代器**一定**是可迭代，反之不成立 | ✅ `is` 成立 |
| 11 | `list(it)` 连跑两次 | 第一次 `[1, 2, 3]`、第二次 **`[]`**（一次性，且**不报错**）；对照 `list(nums)` 两次都一样 | ✅ `[1, 2, 3]` → `[]`；`list(nums)` 两次均 `[1, 2, 3]` |
| 12 | `len(it)` / `it[0]` | 都抛 **`TypeError`**：迭代器不是序列，没有长度、不支持下标 | ✅ 两条均 `TypeError` |
| 13 | `count_to(3)` | 生成器函数 → `[1, 2, 3]`（`yield` 自动提供 `__iter__`/`__next__`/`StopIteration`） | ✅ `[1, 2, 3]`（最终写法 `yield from range(1, n + 1)`） |
| 14 | 生成器耗尽后 | `iter(g) is g`；`next` **持续**抛 `StopIteration`（官方"broken"条款） | ✅ `iter(g) is g`；越界两次均抛 `StopIteration` |
| 15 | `Countdown(3)` | 手写 `__iter__` + `__next__` → `[3, 2, 1]` | ✅ `[3, 2, 1]`（先得到 `[3, 2]`，修掉 `<= 1` 的 off-by-one 后才绿） |
| 16 | `(x * x for x in [1, 2, 3])` | `sum()` 消费后得 **14**，再 `list()` 得 **`[]`** | ✅ `14` → `[]` |

**实跑记录（2026-11-02）**：`uv run --no-project python iter_protocol.py` → 末行 **`判据 16/16 绿（本文件覆盖 9-16）`**；`uvx ruff check .` → `All checks passed!`；`uvx ruff format .` → 补 1 处空行（第三次踩同一个 `E302`）。Pylance/Ruff 诊断最终 **0 条**。

## 官方依据（2026-11-02 查证 · 两条独立页面）

- **`library/stdtypes.html` · Common Sequence Operations（注 3/4/5/8）**：`s[i:j:k]` 的索引是 `x = i + n*k`（`0 <= n < (j-i)/k`），**永不包含 `j`**；`i`/`j` 省略时取"端点值"，**取哪一端取决于 `k` 的符号**；负索引按 `len(s) + i` 折算；**越界一律截断、切片不抛 `IndexError`**（`k>0` 截到 `len(s)`，`k<0` 截到 `len(s)-1`），而单元素访问 `s[i]` 越界**会**抛 `IndexError`；**`k` 不能为 0**（`ValueError: slice step cannot be zero`）。
- **`library/stdtypes.html` · Mutable Sequence Types（注 1）**：`s[i:j] = t`（即 `k == 1`）**长度可变**、右侧可任意可迭代；**`k != 1` 时 `t` 必须与被替换切片长度相同**（CPython 抛 `ValueError: attempt to assign sequence of size X to extended slice of size Y`）；`del s[i:j:k]` 按步长删除、会变短。
- **`library/copy.html`**：**浅拷贝** = 新建复合对象后「插入原对象内部元素的**引用**」；**深拷贝** = 递归插入**副本**。原文明确列表浅拷贝的两种写法是 **`a.copy()` 与 `a[:]`**（整段切片）—— 注意子类实例用这两种会退化成基类实例，`copy.copy()` 才保类型。
- **`library/stdtypes.html` · Iterator Types**：迭代器协议 = `__iter__`（**返回迭代器自身**，原文 "Return the iterator object itself"）+ `__next__`；耗尽后**必须**抛 `StopIteration`，且原文明确「**一旦抛过 `StopIteration`，后续每次调用都必须继续抛**，否则该实现被视为 broken」。
- **`tutorial/classes.html` §9.8 · §9.9 · §9.10**：`for` 幕后 = `iter()` 取迭代器 → 反复 `__next__()` → 捕获 `StopIteration` 终止（**不是错误，是协议信号**）；生成器「**自动创建** `__iter__`/`__next__`、**自动保存**局部变量与执行状态、结束时**自动抛** `StopIteration`」；生成器表达式用**圆括号**、专为"被外层函数**立即消费**"设计、比等价列表推导式 **more memory friendly**（原文用词，不是 "lazy"）。

## 怎么跑

```powershell
cd practice/python/2026/11/02-slice-iter
uv run --no-project python slice_basics.py    # 判据 1-8
uv run --no-project python iter_protocol.py   # 判据 9-16
uvx ruff check . && uvx ruff format .
```

判据全在各自文件的 `main()` 里，跑一次全出；顺序执行 ⇒ 第一个红就停。

## 踩过的坑

- **`uvx ruff check .` 会静态抓「常量下标的越界」**：判据 7 原本直接写 `[1, 2, 3][100]`（就是要让它抛 `IndexError`），被 ruff 报 **`PLE0643 Expression is likely to raise IndexError`** —— 规则本身是对的（那行确实必炸），但**在判据代码里我们就是要它炸**。解法不是 `# noqa`，而是改成用变量算下标（`seq[len(seq)]`），静态算不出来就不告警，运行期照样抛。
- **切片赋值本身不是表达式，拿不到返回值**：`nums[1:3] = [9]` 只能当语句写，写成 `return nums[1:3] = [9]` 是 `SyntaxError`；所以判据 5 必须「赋值语句 + `return nums`」两行。跟 `list.append()` 返回 `None` 是同一类"原地改的操作不给结果"。
- **`ruff check` 绿了不代表格式合规（本练习第二次踩）**：缺顶层两空行（`E302`）**不在 ruff 默认规则集**里，`check` 查不出来，只有 `format` 会改。⇒ 跑法里的两条命令是**必须都跑**，不是二选一。
- **`for` 循环体只有一句 `yield` 会被 Ruff 报 `UP028`**（`Replace 'yield' over 'for' loop with 'yield from'`）：这不是错误，是 pyupgrade 的"更地道写法"建议。`yield from` 语义相同且**在 C 层委托**（少一层 Python 循环），还能透传 `send()`/`throw()`/`StopIteration.value` —— 本练习最终采用它。
- **`__next__` 的停止条件写错会"少交一个值"，而且一声不响**：最初写成 `if self.current <= 1: raise StopIteration` ⇒ `Countdown(3)` 产出 `[3, 2]`，**没有异常、没有警告**，只在判据里红。根因是把"最后一个合法值"当成了"结束标志"；正确判据是「**低于下界**」（`< 1`）而不是「等于最后一个值」。⇒ 判据必须写**精确期望序列**，只验"非空"抓不到它。
- **Pylance 会静态拦住判据故意制造的 `TypeError`**：判据 12 要验 `len(迭代器)` / `迭代器[0]` 必抛 `TypeError`，但 `Iterator` 类型上确实没有 `__len__`/`__getitem__`，于是先报 `reportArgumentType` / `reportIndexIssue`。解法 = `cast(Any, it)` 把类型信息擦掉，**运行期行为不变**（同 `PLE0643` 那个坑的思路：静态分析别挡在判据前面）。
