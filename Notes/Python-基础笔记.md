# Python 基础笔记（2026-10-29 起 · AI 主线的语言层）

> **定位**：AI 主线的**语言层**（Python 本身）专册，与 `AI-Agent-笔记.md`（LLM 与 Agent 层）平行。
> **分家判据**：**Python 语言本身**（语法 / 容器 / 可变性 / 装饰器 / async / httpx / pydantic）进本册；**LLM 与 Agent 的东西**（messages / 流式 / token / 结构化输出 / 工具调用 / RAG / 评测）进 `AI-Agent-笔记.md`。**一个概念只留一个家**。
> 代码实例都在 `practice/` 下的对应目录里，可运行验证。
> 原则：**只记真实学到的内容与真实踩过的坑**；每个知识点尽量配一条**可实测的判据**。

**覆盖进度**：**第 1 章 语法总览 ✅**（2026-10-29）｜ **第 2 章 list 容器 API ✅**（2026-10-29）｜ **第 3 章 可变性陷阱 ✅**（2026-10-30）｜ **第 4 章 dict·set·tuple（可哈希）✅**（2026-10-30）｜ **第 5 章 async / await 与事件循环 ✅**（2026-10-31）｜ **第 6 章 推导式 / 切片 / 迭代器 ✅**（2026-11-02）｜ 函数进阶（`*args` / 装饰器）/ 模块与包 / httpx / pydantic **未开始**

## 回访清单（已迁出）

> 题库单独成文件 → [Notes/回访清单.md](回访清单.md)（活跃 · 周日盲写用）｜ [_archive/回访-已销.md](_archive/回访-已销.md)（已销）
> **本册只留指针**：清单混在正文里，按需查任何一章都会把它整份读进来，白烧上下文。入库门槛与销账规则见清单文件头部。

---

## 1. 语法总览：C# → Python 对照（2026-10-29 · S0）

### 1.1 变量不是盒子，是名字（本册所有坑的总根）

- C# 的 `int x = 5;` 模型是**盒子里装 5**；Python 里 **`5` 是堆上一个对象，`x` 只是贴上去的标签**
- `x = 6` 不是"改盒子"，是**把标签撕下来贴到 6 上**；原来的 5 没人贴 ⇒ 引用计数归零 ⇒ 回收
- ⭐ **推论**：`a = b` 是**再贴一张标签**，不是复制。列表可变 ⇒ 两张标签指向同一个列表，改一个另一个跟着变
- **判据（实测）**：`b = [1,2]; c = b; c.append(3)` ⇒ `b` 也变 `[1,2,3]`，且 `c is b` 为 `True`

### 1.2 类型注解：现代写法（已查证官方文档 · 2026-10）

```python
nums: list[int] = []          # 3.9+ 内置泛型，别写 typing.List[int]
name: str | None = None       # 3.10+ 联合类型，别写 Optional[str]
def search(nums: list[int], target: int) -> int: ...
```

| 事实 | 版本 | 说明 |
|---|---|---|
| `list[int]` 而非 `typing.List[int]` | 3.9+ | PEP 585，内置类型直接可下标 |
| `X \| None` 而非 `Optional[X]` | 3.10+ | PEP 604 |
| 注解**默认立即求值** | 3.0 – 3.13 | PEP 3107 / 526，写在源码里就被算一次 |
| `from __future__ import annotations` = **存成字符串** | 3.7+ | PEP 563；好处是前向引用不用加引号 |
| 注解**默认延迟求值** | **3.14+** | PEP 649 / 749；官方明确说那行 `__future__` 已 **deprecated** |
| 那行 `__future__` **会被移除** | 不早于 **3.13 EOL（2029）** | PEP 749；在 3.14 里它的行为**不变** |

- ⚠️ **注解运行期一律不检查** —— 它只是给 `Pylance`（类型检查）和读者看的
- ⭐ **到底该不该加（2026-10 定）**：判据不是"我自己跑哪个版本"，而是 **`requires-python` 下限 + 要不要发布给别人**

  | 场景 | 现状做法 | 判据 |
  |---|---|---|
  | **应用 / 脚本**，锁 3.10+ | 多数**不加** | 3.10+ 运行期就认 `X \| None` |
  | **库 / 包**，还要支持 3.9 | **普遍加** | 3.9 里 `X \| None` 在运行期 `TypeError` |
  | 注解要被 `get_type_hints()` 读 | 谨慎加 | 官方原话：加了之后**更难在运行期解析** |

- ⭐ **"大家都升到最新版了所以不用加"是错的**：本机实测 19 个包、**4180 个 `.py` 文件里有 1659 个（≈40%）** 带这行（`anyio` 93% / `httpx` 83% / `pydantic` 52% / `openai` 44% / `rich` 9%）⇒ 生态里**没有统一做法**，差别来自各库自己的最低版本策略
- ⭐ **本项目（2026-10-29 起跑 Python 3.14.8）** ⇒ 注解**默认延迟求值** ⇒ **全仓库已删除这行**（`ask.py` / `check_llm.py` / `solution.py`）；⚠️ 当初留它还有一个用途 —— 前向引用不用加引号，而 3.14 的延迟求值同样满足这一点

### 1.3 条件

```python
if n == 0:
    return -1
elif n == 1:            # 不是 else if
    return 0
```

| 事项 | C# | Python |
|---|---|---|
| 块的边界 | `{ }` | **缩进 4 空格**，行尾**冒号 `:`** |
| 逻辑运算 | `&&` `\|\|` `!` | **`and` `or` `not`** |
| 链式比较 | 写不出 | `0 <= i < n`（等价 `0 <= i and i < n`，但 `i` **只算一次**） |
| 三元 | `c ? a : b` | `a if c else b` |
| 多分支 | `switch/case` | `match/case`（3.10+） |
| 真值 | 必须 `bool` | **任何对象都能当条件**：`0` `""` `[]` `{}` `None` 全假；`"0"`、`[0]` 是真 |
| 相等 | `==` 值 / `ReferenceEquals` | **`==` 比值**（调 `__eq__`）；**`is` 比身份**（是不是同一个对象）；判空一律 **`is None`** |

### 1.4 循环

```python
for x in [4, 5, 6]:            # = C# 的 foreach
    print(x)

for i in range(len(nums)):     # 要下标就自己造
    print(i, nums[i])

for i, v in enumerate(nums):   # 更 Python：下标 + 值一起
    print(i, v)
```

| 事项 | C# | Python |
|---|---|---|
| 计数循环 | `for (int i = 0; i < n; i++)` | `for i in range(n)` —— **右端开区间** |
| 三段式 | `(i = a; i < b; i += s)` | `range(a, b, s)`，同样右端开；`range(10, 0, -1)` = 10…1 |
| `range` 是什么 | — | **惰性序列对象**，不是列表（`list(range(5))` 才是） |
| 自增 | `i++` | ❌ 没有，写 `i += 1` |
| `do-while` | 有 | ❌ 没有，用 `while True: … ; break` |
| 作用域 | 块级 | ⚠️ **函数级** —— `for` 的变量出了循环**还在** |
| 分号 | 必须 | **可选且合法**（`a = 1; b = 2` 能跑），但 PEP 8 不推荐 |

### 1.5 声明 / 赋值 / 常量

| # | 事项 | C# | Python |
|---|---|---|---|
| 1 | 声明 + 赋值 | `int x = 5;` | `x = 5`（无关键字、无分号） |
| 2 | 只声明不赋值 | `int x;` 合法 | ❌ 没有"声明"这回事，直接用 `x` → `NameError` |
| 3 | 类型 | 编译期定死 | **变量无类型，对象有类型**；同一名字可先绑 `int` 再绑 `str` |
| 4 | 类型注解 | 必须 | **可选**，且运行期不检查 |
| 5 | 常量 | `const` | 约定**全大写** `MAX_N = 100`，语言不强制 |
| 6 | 一次赋多个 | 分开写 | `a, b = 1, 2`；交换 `a, b = b, a`（不需要临时变量） |
| 7 | 空值 | `null` | `None`，判空用 `is None` |
| 8 | 布尔字面量 | `true` / `false` | **`True` / `False`**（大写，是关键字） |
| 9 | 除法 | `7 / 2 == 3`（int） | `7 / 2 == 3.5`（**永远是真除**）；整除必须 **`//`** |

### 1.6 踩过的坑：把赋值语句写进了括号里

- **现象**：在 `raise NotImplementedError(` 的括号中间写 `min = 5;`，一次冒出三条诊断
- **三条诊断各说一件事**：

| 诊断 | 出处 | 含义 |
|---|---|---|
| `Expected ',', found ';'` | Ruff（语法） | 我在**参数列表**里，下一个该是逗号 ⇒ `;` 非法 |
| `没有名为"min"的参数` | Pylance | `min = 5` 被当成**关键字参数** `min=5` 传给 `NotImplementedError` |
| `"(" 未关闭` | Pylance | 括号里塞了非法内容 ⇒ `(` 找不到配对的 `)` |

- ⭐ **判据（读语法报错的第一把钥匙）**：看它**期待什么** —— 期待逗号 ⇒ 你在**参数列表**里；期待缩进/新行 ⇒ 你在**语句区**
- ⚠️ **两个易混模型**：① "不要声明"免掉的是**类型声明**，没免掉"语句必须站在语句位上" ② **分号在 Python 是合法的**，只是参数列表里非法 + PEP 8 不推荐
- ⚠️ `min` / `max` / `sum` / `list` / `dict` 都是**内置函数名**，拿来当变量名会**遮蔽**它

---

## 2. list 容器 API（2026-10-29）

### 2.1 它是什么：动态数组，不是链表

- CPython 的 `list` = **一段连续的对象指针数组** + 一个"已用长度" + 一个"已分配容量"
- 元素本身（int / str / 对象）都**不在数组里**，数组里只放**指向它们的指针** ⇒ 所以 `list` 能装任何东西（无类型约束）
- **超额分配**：快满时按比例扩容（CPython 约 ×1.125 + 常数），所以 `append` **均摊 O(1)**
- **判据（实测）**：

| 观测 | 数值 | 说明 |
|---|---|---|
| `sys.getsizeof([])` | **56 B** | 空表本体（不含元素） |
| 前 4 次 `append` 后 | **88 B × 4** | 第 1 次 append 就**预分配了 4 个槽** |
| 第 5–8 次 | **120 B × 4** | 容量 8（56 + 8×8 = 120） |
| 第 9–12 次 | **184 B × 4** | 容量 16 |

⇒ **容量阶梯 4 → 8 → 16**，且**连续 4 次 append 尺寸不变** = 典型的超额分配指纹。

### 2.2 创建

| 写法 | 结果 | 备注 |
|---|---|---|
| `[]` / `list()` | 空表 | — |
| `[1, 2, 3]` | 字面量 | — |
| `list(range(5))` | `[0,1,2,3,4]` | `range` 惰性，必须转 |
| `[0] * 5` | `[0,0,0,0,0]` | 元素是**不可变**的才安全（见 2.8） |
| `[x * x for x in range(5)]` | `[0,1,4,9,16]` | **列表推导**，比 C# 的 `Select().ToList()` 快也短 |
| `[x for x in a if x > 0]` | 过滤 | 推导式带条件 |

### 2.3 增

| API | 复杂度 | 说明 |
|---|---|---|
| `a.append(x)` | **O(1) 均摊** | 尾部加**一个**（哪怕 `x` 本身是列表，也是加一个元素） |
| `a.extend(it)` | O(k) | 把可迭代对象**摊开**逐个加（对比 `append` 会加成一个嵌套元素） |
| `a.insert(i, x)` | **O(n)** | 插入点之后整体后移 |
| `a += [x, y]` | O(k) | 等价 `extend`，**原地**改 |
| `a + [x]` | O(n) | ⚠️ **返回新列表**，原表不变（别在大循环里用） |
| `a * 3` | O(n) | 重复拼接，返回新表（⚠️ 嵌套时是同一引用，见 2.8） |

### 2.4 删

| API | 复杂度 | 语义（⚠️ 与 C# 不同） |
|---|---|---|
| `a.pop()` | **O(1)** | 弹出**尾部**并返回 |
| `a.pop(i)` | **O(n)** | 弹出第 `i` 个并返回（后面整体前移）；`pop(0)` 是最贵的删法 |
| `a.remove(x)` | O(n) | 删**第一个等于 `x`** 的；⚠️ **找不到抛 `ValueError`**（C# `Remove` 返 `bool`） |
| `del a[i]` | O(n) | 按位置删，不返回 |
| `del a[i:j]` | O(n) | 删一段 |
| `a.clear()` | O(n) | 清空（容量可能保留） |

### 2.5 查与改

| API | 复杂度 | 说明 |
|---|---|---|
| `a[i]` | **O(1)** | 下标读写（连续数组，直接算地址） |
| `a[-1]` | O(1) | **负下标从尾部数**：`-1` 末位、`-2` 倒数第二 |
| `a[i] = x` | O(1) | 赋值 |
| `a.index(x)` | O(n) | 返回首个下标；⚠️ **找不到抛 `ValueError`**（C# `IndexOf` 返 `-1`） |
| `a.index(x, start, stop)` | O(n) | 限定区间；右端开 |
| `a.count(x)` | O(n) | 出现次数 |
| `x in a` | O(n) | 成员判断（**不是哈希**，`set` 才是 O(1)） |
| `len(a)` | **O(1)** | 长度存在对象头里，不用数 |
| `min(a)` / `max(a)` / `sum(a)` | O(n) | 内置函数，不是方法 |

### 2.6 排序与反转（本册最贵的一组坑）

| API | 原地？ | 返回 | 说明 |
|---|---|---|---|
| `a.sort()` | ✅ 原地 | **`None`** | ⚠️⚠️ `a = a.sort()` ⇒ `a` 变 `None` |
| `forward = sorted(a)` | ❌ 新建 | 新列表 | 原表不动 |
| `a.sort(reverse=True)` | ✅ | `None` | 降序 |
| `a.sort(key=len)` | ✅ | `None` | **按 key 函数算出的值排**（不要写 C# 那种 `Comparison<T>` 返回负数） |
| `a.sort(key=lambda p: (-p.score, p.name))` | ✅ | `None` | 多级排序：**元组逐项比**，负数实现降序 |
| `a.reverse()` | ✅ 原地 | `None` | 反转 |
| `reversed(a)` | ❌ | **迭代器** | 要列表得 `list(reversed(a))` |
| `a[::-1]` | ❌ | **新列表** | 切片反转（最常用） |

- 排序**稳定**（相等元素保持原相对顺序，同 C# `List<T>.Sort` 的稳定版语义）
- 时间复杂度 **O(n log n)**（Timsort，已部分有序时更快）

### 2.7 切片：右端开 + 负步长

```python
a = [0, 1, 2, 3, 4, 5]
a[1:3]      # [1, 2]        ← 右端开！不含 a[3]
a[:3]       # [0, 1, 2]
a[3:]       # [3, 4, 5]
a[::2]      # [0, 2, 4]     ← 步长 2
a[::-1]     # [5, 4, 3, 2, 1, 0]   ← 负步长 = 反转
a[5:1:-1]   # [5, 4, 3, 2] ← 负步长时 start 默认末尾、stop 默认开头（**仍是右端开**）
```

- ⚠️ **切片赋值可以改长度**（这是 C# 完全没有的能力）：
  ```python
  a = [0, 1, 2, 3]
  a[1:3] = []           # 删掉 1,2  →  [0, 3]
  a[1:1] = [8, 9]       # 空切片处插入 → [0, 8, 9, 3]
  ```
- 切片**产生新列表**（不是视图）⇒ `b = a[:]` 就是**浅拷贝**

### 2.8 拷贝三兄弟 + 嵌套乘法坑

| 写法 | 是什么 | 判据（实测） |
|---|---|---|
| `b = a` | **别名**（同一个对象） | `b is a` → **True**；`b.append(3)` ⇒ `a` 也变 |
| `b = a[:]` / `a.copy()` / `list(a)` | **浅拷贝**（新列表，元素仍是同一批对象） | `b is a` → **False**；但元素若可变，改元素**两边都变** |
| `copy.deepcopy(a)` | **深拷贝**（递归复制元素） | 需要 `import copy` |

⚠️ **`[[0] * 2] * 2` 的坑（实测）**：

```python
m = [[0] * 2] * 2      # 看起来是 2×2 零矩阵
m[0][0] = 9
print(m)               # [[9, 0], [9, 0]]  ← 两行是同一个列表！
```

- **原因**：`*` 复制的只是**指向同一个内层列表的指针**，不是内层列表本身
- **正解**：`m = [[0] * 2 for _ in range(2)]`（推导式每轮**新建**一层）
- **判据**：`len({id(row) for row in m})` —— 乘法版是 **1**，推导式版是 **2**

### 2.9 复杂度总表

| 操作 | 复杂度 | 直觉 |
|---|---|---|
| `a[i]` 读 / 写 | **O(1)** | 连续数组，算地址 |
| `len(a)` | **O(1)** | 长度存在对象头 |
| `append` / `pop()` | **O(1) 均摊** | 尾部 + 超额分配 |
| `insert(i, x)` / `pop(i)` / `del a[i]` | **O(n)** | 插入点之后整体挪 |
| `x in a` / `index` / `remove` / `count` | **O(n)** | 线性扫 |
| `sort` | **O(n log n)** | Timsort |

⭐ **实测（10 万次操作，表长 30 万）**：

| 操作 | 耗时 | 比值 |
|---|---|---|
| `pop()` | **3.7 ms** | 1× |
| `pop(0)` | **20250.9 ms** | **≈ 5473×** |

⇒ **"尾部当栈用"是免费，"头部当队列用"是灾难**（要队列用 `collections.deque`）。

### 2.10 C# `List<T>` → Python `list` 对照

| 意图 | C# | Python |
|---|---|---|
| 加一个 | `Add(x)` | `a.append(x)` |
| 加一批 | `AddRange(en)` | `a.extend(en)` |
| 插入 | `Insert(i, x)` | `a.insert(i, x)` |
| 删尾部 | `RemoveAt(Count - 1)` | `a.pop()` |
| 删指定位置 | `RemoveAt(i)` | `a.pop(i)` |
| 删值 | `Remove(x)` → **bool** | `a.remove(x)` → **None / 抛 ValueError** |
| 批量删 | `RemoveAll(pred)` | `a[:] = [x for x in a if not pred(x)]` |
| 查下标 | `IndexOf(x)` → **-1** | `a.index(x)` → **抛 ValueError** |
| 包含 | `Contains(x)` | `x in a` |
| 数量 | `Count` | `len(a)` |
| 排序 | `Sort()` | `a.sort()`（**返回 None**） |
| 自定义排序 | `Sort(Comparison<T>)` | `a.sort(key=...)` |
| 反转 | `Reverse()` | `a.reverse()` / `a[::-1]` |
| 清空 | `Clear()` | `a.clear()` |
| 索引器 | `a[i]` | `a[i]`（另有**负下标**） |
| 类型 | 强类型 `List<int>` | **什么都能装**（元素是指针） |

### 2.11 写二分题要用的子集（够用就行）

| 要做的事 | 写法 |
|---|---|
| 取中点 | `mid = (lo + hi) // 2` ← **`//` 才是整除** |
| 取长度 | `len(nums)`（不是 `.Length`） |
| 下标访问 | `nums[i]`；`nums[-1]` = 末位 |
| 闭区间循环 | `while lo <= hi:` |
| 早返回 / 找不到 | `return mid` / `return -1`（**不是 `None`**） |

### 2.12 坑清单（写完代码过一遍）

1. `a.sort()` **返回 `None`** ⇒ 别写 `a = a.sort()`
2. `b = a` 是**别名** ⇒ 要拷贝写 `a[:]` / `a.copy()`
3. `[[0] * n] * m` 的 **m 行是同一行** ⇒ 用推导式
4. 遍历中 `remove` / `append` 同一个列表 ⇒ **会跳过元素**（正解：倒序删、或推导式重建）
5. `remove` / `index` **找不到抛 `ValueError`** ⇒ C# 迁移必栽；不确定就先 `if x in a`
6. `in` / `index` / `remove` 都是 **O(n)** ⇒ 反复查同一个表要换 `set` / `dict`

---

## 3. 可变性陷阱：四张面孔（2026-10-30 · S0）

> 四张面孔共用**一条判据**：看 `id`、看调用方手里那份数据 —— 不看代码长什么样。
> 官方依据：Tutorial 4.9.1（默认值 *evaluated at the point of function definition*）+ Programming FAQ（*created exactly once*）+ Reference 7.2.1（`+=` 是 *when possible* 原地）。

### 3.1 第一张面孔：默认参数只在 `def` 时求值一次

```python
def f(a, L=[]):        # ← 这个 [] 在 def 执行那一刻造一次，之后所有调用共用
    L.append(a)
    return L

f(1); f(2); f(3)       # [1] → [1, 2] → [1, 2, 3]
```

| 事实 | 说明 |
|---|---|
| 求值时机 | **`def` 语句执行时**，在定义所在作用域内求值 |
| 求值次数 | **仅一次**，结果存在**函数对象**上（不在代码对象里） |
| 物证 | `f.__defaults__` 越变越长 → 那个列表被调用**污染**了 |
| 正确写法 | `L: list[int] \| None = None` + 函数体里 `if L is None: L = []` |
| 为什么不写 `L: list[int] = None` | 类型检查器报"默认值类型不匹配"（`None` 不是 `list[int]`） |
| 正当用途 | 记忆化缓存（`def f(x, *, _cache={})`）—— **故意**共享，且用 `*` 把缓存藏起来 |

### 3.2 第二张面孔：改对象 vs 重新绑定（分水岭）

| 函数里做了什么 | 调用方看得见 | 例子 |
|---|---|---|
| **改对象** | ✅ | `append` / `[i]=` / `sort` / `clear` / `update` / `+=` |
| **重新绑定名字** | ❌ | `x = [1, 2]` / `x = x + [item]` |
| 返回新对象 | 只有接了才有 | `return x + [item]` |

⚠️ 两个常见误解：

1. **"`nums` 是副本"** —— 错。`id(nums) == id(调用方的 list)`，它是**第二个名字**，不是第二份数据
2. **"没有 `return` 所以没改到"** —— 错。`append` 不 return 也照样改得到；反过来，`return x + [item]` 也改不到原对象

⭐ 判据：**共享的是「对象」这个身份，不是「名字」。** 想让调用方看见，只能动对象本身。

### 3.3 第三张面孔：`+=` 的真相是「when possible」

官方原话：*when possible, the actual operation is performed in-place* —— **由类型决定，不由符号决定**。

| 类型 | `x += y` | 判据 |
|---|---|---|
| `list` | ✅ 原地（内部 `__iadd__` ≈ `extend`） | `id` 前后**相同** |
| `set` | ✅ 原地（`\|=` 同理） | `id` 相同 |
| `int` / `str` / `tuple` | ❌ 新建再绑 | `id` 前后**不同** |
| `dict` | ❌ 不支持 `+=`（`TypeError`） | — |

⚠️ 检查器在 `rebind()` 里报的 `Local variable 'nums' is assigned to but never used` **不是说代码难看，是在说这次赋值没有意义** —— 语言语义在报警。

### 3.4 第四张面孔：`__defaults__` 证明默认值活在函数对象上

| 项 | 值 |
|---|---|
| 类型 | `tuple`；**没有默认值时是 `None`**（不是 `()`） |
| 内容 | 只装**带默认值的位置参数**，按顺序、**靠右对齐** |
| 关键字专用参数 | 走另一个属性 `__kwdefaults__`（`dict`，键是参数名） |
| 可写 | ✅ 属于 *Special writable attributes*，运行时可整体替换 |

⭐ 官方在 Code objects 那节写死了原因：

> *the default argument values are stored in the **function object**, not in the code object (because they represent values calculated at run-time)*

⇒ 一次调用结束后栈帧销毁、局部变量全没了，**但那个列表活着** —— 因为它挂在函数对象上，而函数对象是模块级全局名。

硬判据（一行）：

```python
f(1)
assert f(1) is f.__defaults__[0]     # True ⇒ 返回值就是挂在函数身上那个对象，不是副本
```

### 3.5 坑清单（写完含可变对象的代码过一遍）

1. 参数里别写 `= []` / `= {}` / `= set()` ⇒ 用 `None` + 函数体内新建
2. 函数"顺便"改了传进来的 list ⇒ 调用方数据被污染（多轮对话的 `messages` 是重灾区）
3. 要"试一下再决定" ⇒ 先 `copy()` 一份，或改成返回新对象
4. 想知道改没改到 ⇒ **看 `id`，别用 `==` 猜**（两个内容相同的新列表 `==` 也是 True）

---

## 4. dict / set / tuple：可哈希是通行证（2026-10-30 · S0）

> 练习：`practice/python/2026/10/30-containers/`（`dict_schema.py` + `set_tuple.py`，**7 条判据全绿**）
> 官方依据：Python **3.14.10** Tutorial §5、Glossary · `hashable`、OpenAI API 参考（工具 `name` 约束）

### 4.1 dict = 哈希表 + 记住插入顺序

| 事实 | 版本 / 出处 | 说明 |
|---|---|---|
| 插入顺序是**语言保证** | 3.7+（3.6 只是实现细节） | `list(d)` 按插入序；要"有序值"得显式 `sorted(d)`，别把插入序当排序 |
| 键必须**不可变 / 可哈希** | Tutorial §5.5 | `list` 不行，官方理由：*can be modified in place* |
| `\|` / `\|=` 合并 | 3.9+（PEP 584） | `\|` 新建（第三个对象）、`\|=` 原地；两者都是**浅**合并 |
| `keys()` / `values()` / `items()` | — | **动态视图**不是快照；keys/items 支持集合运算，values 不支持 |
| `frozendict` | **3.15**（PEP 814） | 本机 3.14.8 **用不到**，别照 3.15 文档写 |

取值三连（与第 3 章的默认参数陷阱**互为镜像**）：

| 写法 | 缺失键时 | 默认值什么时候求值 |
|---|---|---|
| `d[k]` | `KeyError` | — |
| `d.get(k)` | `None` | — |
| `d.get(k, [])` | `[]` | **每次调用**都现场新建 ⇒ 两次拿到的不是同一对象 |

⚠️ 正因为是"每次新建"，它反而**安全**；而 `def f(x=[])` 是 `def` 时求值一次 ⇒ 危险。同一把尺子，两个方向。

### 4.2 tool schema 就是 dict 的嵌套（一个类都不用建）

```text
{"type": "function", "function": {"name": ..., "description": ...,
  "parameters": {"type": "object", "properties": {...}, "required": [...]}}}
```

| 键 | 谁读 | 要点 |
|---|---|---|
| `type: "function"` | SDK | 工具**种类**（协议扩展位） |
| `function` | SDK | 本体容器（名字/描述住在里面） |
| `name` | 模型 | 回传标识；**`a-z A-Z 0-9 _ -`，≤64 字符**（超了/带中文会被拒） |
| `description` | 模型 | ⭐ **决定"要不要调"的 prompt**，不是注释；写清"什么时候不用"最能减少误调 |
| `parameters` | 模型 | 一份 **JSON Schema**（说明书本身也是 dict） |
| `type: "object"` | 模型 | 参数整体是 **JSON 对象**（不是 Python 的 `object` 类型）；与外层 `type` 同名不同义 |
| `properties` | 模型 | 每个参数名 → 它自己的子 schema（可递归嵌套） |
| `required` | 模型 | 只放参数名；不列进去的 = 可选；**塞进可选项会让模型编值** |

- 返回的 dict 里 `properties` / `required` 与入参**是同一个对象**（`is` 为 True，没拷贝）
- 模型回来的 `arguments` 是**一段 JSON 字符串**，不是 dict（S2 第一个坑）

### 4.3 合并语义：三条规则 + 一条纪律

| # | 规则 | 判据 |
|---|---|---|
| 1 | 同名键：**右边赢** | `{"k":"a"} \| {"k":"b"}` → `"b"` |
| 2 | 不同名键：都留下 | 并集 |
| 3 | ⭐ **浅合并**：嵌套 dict **整块替换**，不递归 | `{"p":{"city":1}} \| {"p":{"unit":2}}` → `{"p":{"unit":2}}`，`city` 消失 |
| — | `\|` 换 `id`、`\|=` 保 `id` | 与第 3 章 `+` / `+=` 同一把尺子 |

**纪律：别人传给你的 dict 一律用 `|`（复印），绝不 `|=`（划改）** —— 否则就是"函数有隐藏副作用"，bug 会出现在别人的代码里。

### 4.4 可哈希：一条根管三处（set 元素 / dict 键 / tuple 当键）

官方定义（Glossary）：

> hashable = **hash 值终身不变**（`__hash__`）+ **可比较**（`__eq__`）；且契约：`a == b` ⟹ `hash(a) == hash(b)`。
> 可哈希性使对象能当 **dict 键**与 **set 成员**，因为二者内部都靠哈希值。

| 类别 | 可哈希 | 例 |
|---|---|---|
| 不可变内置（`int`/`str`/`bytes`） | ✅ | `hash("abc")` |
| 可变容器（`list`/`dict`/`set`） | ❌ | `{[1,2]}` → `TypeError` |
| 不可变容器（`tuple`/`frozenset`） | ⚠️ **只有当元素都可哈希** | `(1,2)` ✅ / **`(1,[2])` ❌** |
| 自定义类实例 | ✅ 默认（hash 由 `id()` 派生） | 但**只写 `__eq__` 不写 `__hash__`** ⇒ `__hash__` 变 `None` ⇒ 不可哈希 |

为什么必须"终身不变"：哈希表靠 hash 定位桶。**包进柜子后自己改了名字 ⇒ 再算柜号对不上 ⇒ 东西永远取不回来。**

两个附带结论：

- `frozenset([[1,2]])` **构造时**就 `TypeError` ⇒ **集合里不可能有不可哈希元素** ⇒ 对 `set`/`frozenset` 做递归转换是"防御性冗余"，不是必需
- `{1, True, 1.0}` 只剩 **1** 个元素（`1 == True == 1.0` 且 hash 相同）

### 4.5 `id()` 是内存地址，不是编号

- 对象销毁后地址**会被复用** ⇒ 内联比较 `id(表达式)` 不可信：

```python
id(d.get("x", [])) == id(d.get("x", []))   # True（前一个已回收，地址被复用）
a = d.get("x", []); b = d.get("x", [])
a is b                                      # False（两个对象同时存活才比得准）
```

- **判据纪律**：比"是不是同一个对象"用 `is`；要 `id` 就必须**先各绑变量**。

### 4.6 相等语义决定"键的形状"

| 原类型 | 它的 `==` | 摊平后必须用 | 反例后果 |
|---|---|---|---|
| `list` / `tuple` | 顺序敏感 | `tuple` | — |
| `set` / `frozenset` | 无序 | `frozenset` | — |
| `dict` | **忽略顺序** | **`frozenset(items)`** | `{'a':1,'b':2}` 与 `{'b':2,'a':1}` `==` 为 True，但 `tuple(items)` 得到**两个不同的键** ⇒ 缓存不命中 |

口诀：**外壳的有序/无序，必须跟原类型 `==` 的有序/无序对齐。**

### 4.7 坑清单

- `{}` 是空 **dict**，空集合必须 `set()`
- `{"a"}` 是集合、`{"a": 1}` 是字典 —— 有没有冒号决定
- `if isinstance(obj, (list, tuple))` 优于 `type(obj) is list`（子类、`OrderedDict` 才算 dict）
- 用 `try/except TypeError` 判断可哈希，别用类型白名单
- 注释别以 `noqa` 开头（ruff 会当成指令报 RUF100）；故意违规用行尾 `# noqa: CODE`
- `to_key` 的终止条件必须在容器分支**之后**，否则递归进不去

### 4.8 判据（可复测）

1. tool schema 可 `json.dumps(ensure_ascii=False)`，且 `required is` 入参 = True
2. `d["x"]` 抛 `KeyError` / `d.get("x")` 为 `None` / 两次默认 `[]` 的 `is` 为 False
3. `\|` 换 id、`\|=` 保 id；浅合并让嵌套键整块被替换
4. 去重 **6→3** 且与手工去重一致
5. `list`/`dict`/`set` 不可哈希；`(1, [2])` 不可哈希
6. `{to_key([1,2]): "v"}` 能当键；`{1, True, 1.0}` 元素数 = 1
7. 嵌套 `[1, [2, {"a": 3}, (4, [5])]]` → 可哈希、同内容同键、不同内容不撞

---

## 5. async / await：让出点决定一切（2026-10-31 · S0）

> 实例：`practice/python/2026/10/31-async/`（`bench.py` 四条判据一次跑完）
> 一条根：**`async def` 只是贴标签，`await` 才是执行点；而"快"来自 `await` 让出控制权给事件循环。**

### 5.1 `async def` = 贴标签：调用不执行

```python
async def nap(name):
    await asyncio.sleep(1)
    return f"{name} done"

c = nap("a")        # ← 只是造了个「待办单」
type(c)             # <class 'coroutine'>，函数体一行都没跑
```

- 调用 `async def` 函数 → 返回 **coroutine 对象**，**不执行**。
- 漏了 `await` → 不报错、不执行，退出时只嘟囔 `RuntimeWarning: coroutine 'nap' was never awaited`，耗时 **0.00 s**。**看到 0.00 s 就先查 `await`。**
- `await` 只能写在 `async def` 里：普通函数里写待 `SyntaxError: 'await' outside function`（语法层就拦住，比 5.1 那个 Warning 好抓）。

### 5.2 三个对象：coroutine / Task / Future

| 东西 | 是什么 | 谁创建 |
|---|---|---|
| **coroutine 对象** | 一张待办单，**没有执行权** | 调用 `async def` 函数 |
| **Task** | 协程 + 被事件循环**调度** + 能取结果 / 取消 | `create_task()` / `gather` / `TaskGroup` 内部 |
| **Future** | "未来会有值"的底层占位，几乎不直接用 | 事件循环 / 库 |

`gather` / `TaskGroup` 干的事就是**把 N 个 coroutine 变成 N 个 Task 排队**。

### 5.3 让出点：唯一能切换的地方

**只有三种情况控制权回到事件循环**：

| 路径 | 任务还活着吗 | 之后 |
|---|---|---|
| `await` 一个**未完成**的 awaitable | ✅ 挂起，活着 | 事件就绪后**被切回来接着跑** |
| 函数 `return` / 抛异常 | ❌ 结束（Task 完成） | 没有下文，结果只能 `.result()` 取 |
| 被 `cancel()` | ❌ 结束 | 抛 `CancelledError` |

口诀：**让出 = 中途离场还会回来；结束 = 散场。**

**两种"等"要分清**：

| 状态 | 在哪里等 | 何时继续 |
|---|---|---|
| 任务已创建、**还没跑** | **就绪队列** | 当前任务让出 → 循环立刻轮到它 |
| 任务已跑、**卡在 IO** | **等待队列** | IO 就绪 / 定时器到点 → 标回就绪 |

> **"有空" 不是 CPU 空闲，而是「有人让出」。** 同步阻塞代码里永远没人让出 ⇒ 队列全饿死。

### 5.4 有 `await` ≠ 会让出（今天最贵的一条）

`await` 一个**内部没有挂起点**的协程（或已完成的 Future），控制权**不会**交出去，直接穿过去：

| 情况 | 实测输出 | 结论 |
|---|---|---|
| `await instant()`（内部无 await） | `X前 1.06 / X后 1.08 / Y前 1.08 / Y后 1.09` | **零交错** ⇒ 没让出 |
| `await asyncio.sleep(0)` | `X前 1.13 / Y前 1.13 / X后 1.14 / Y后 1.14` | **严格交错** ⇒ 真让出 |

判据：**看「被 await 的东西需不需要等外部」**（网络 / 定时器 / 磁盘 / 另一个任务）。要强制让一轮就写 `await asyncio.sleep(0)`。

### 5.5 并发 ≠ 并行

- 事件循环**只有一个线程**（实测 `threading.active_count()` = **1**），所有协程挤在它上面轮流跑。
- 同一时刻**只有一个任务**在 CPU 上执行；"同时"指**同时在等外部**。
- 推论：**纯 CPU 活（自算 1 秒）用 `async` 不会变快**，只会多一层调度开销。async 只对"等"有效。

### 5.6 `gather`：批量递单 + 按传入顺序收结果

```python
results = await asyncio.gather(nap("a"), nap("b", 0.2))   # ['a done', 'b done']
```

| 事实 | 判据 |
|---|---|
| 返回 **list**，顺序 = **传入顺序** | `b` 只睡 0.2 s 仍排在第二位 |
| 任务**提交即建好**，但**开跑要等让出** | 提交完 `done=False`（0.92 ms），0.99 ms 才见 `[a] 开始` |
| 首个异常**立刻**抛给等待者，**不取消**其余 | 0.05 s 抛 `RuntimeError`，另一个任务 1.00 s **正常跑完** |
| `return_exceptions=True` | 异常当结果收进 list，不中断 |
| `gather` 自己**没有抢占权** | 里面放 `time.sleep` 照样串行（见 5.7） |

> ⚠️ **类型存根坑**：运行时返回 `list`，但 Pylance 的存根说 `tuple` ⇒ `-> list[str]` 会报 `reportReturnType`。解法 `list(await asyncio.gather(...))`，**别把注解改成 `tuple`**（注解要描述运行时）。

### 5.7 同步阻塞是毒药（`gather` 救不回来）

| 写法 | 实测 | 原因 |
|---|---|---|
| `gather(blocking("a"), blocking("b"))`，`blocking` 里 `time.sleep(1)` | **2.00 s**，`a` 出来才见 `b` 进入 | 没有 await ⇒ 唯一的线程被睡死 ⇒ 循环停摆 |
| `gather(to_thread(time.sleep, 1), ...)` | **1.00 s**，交错 | `to_thread` 把阻塞扔进**另一个线程**，`await` 它是真挂起 ⇒ 循环空出来 |

**线程 = 抢占式**（OS 强制切，睡一个线程不影响别人）；**协程 = 协作式**（代码不让出，谁也插不进来）。

**识别清单**：`async def` 里出现 `time.sleep` / `requests.get` / 同步 DB 驱动 / 大文件 `read()` / 同步 SDK ⇒ 全是毒。

**解药优先级**：`asyncio.sleep`、`httpx.AsyncClient`、`asyncpg`、官方 `Async*` SDK **>**（实在没有异步版才）`await asyncio.to_thread(...)`。

`to_thread` 两个边界：① 线程池默认 `min(32, cpu+4)`，超了要排队；② CPU 密集受 GIL 限制，**不会变快**，只是不卡循环。

> **实战形态**：装了 `openai` 同步版却在 `async def` 里调 → 20 个并发请求变成严格串行，代码看着"全是 async"却查不出原因。判据：**并发 20 个，总耗时 ≈ 单个 × 20 ⇒ 中毒。**

### 5.8 `TaskGroup`：结构化的 `gather`（3.11+，新代码首选）

```python
async with asyncio.TaskGroup() as tg:
    task_a = tg.create_task(nap("a"))       # 提交，不 await
    task_b = tg.create_task(nap("b", 0.2))  # 必须接住返回值（否则可能被回收）
# ← 退出 with 的这一刻才「等」
return [task_a.result(), task_b.result()]
```

| 事实 | 判据 |
|---|---|
| 等待发生在**退出 `async with`** | 出块前 `done=False`；块内取 `.result()` → `InvalidStateError` |
| **不返回 list** | 结果只能从各 Task `.result()` 取 |
| 一个失败 → **取消其余** | 另一个任务打印"被取消" |
| 多个异常合并成 **`ExceptionGroup`** | `except*` 或 `except ExceptionGroup`；`CancelledError` 不计入 |
| `create_task` 必须在块内 | 组非活跃时传协程 → 关闭协程并抛 `RuntimeError` |
| 总耗时 = **最慢那个** | `max(1.0, 0.2)` = **1.00 s** |

**`gather` vs `TaskGroup`**：要"容忍部分失败、按传入顺序拿 list" → `gather`；要"一批同生共死" → `TaskGroup`。

### 5.9 常用 API 面

| Task 自身 | 作用 / 坑 |
|---|---|
| `.result()` | 取结果；未完成 → `InvalidStateError`；有异常 → **原样重抛** |
| `.exception()` | 取异常**对象**不抛；没异常为 `None` |
| `.done()` / `.cancelled()` | 结束了没 / 是否被取消结束 |
| `.cancel()` | **请求**取消，下一次让出点才生效；同步阻塞里取消不掉 |
| `.set_name()` / `.get_name()` | 起名，日志排查必用（默认 `Task-5`） |
| `.add_done_callback()` | 完成回调（里面不能 await） |

| 模块级 | 用途 |
|---|---|
| `gather` | 一批，按传入顺序收 list |
| `TaskGroup` | 一批，同生共死 |
| `asyncio.wait(tasks, timeout=)` | 拿 `(done, pending)`，**超时不抛异常**，pending 要自己 `cancel()` |
| `asyncio.as_completed(tasks)` | 按**完成顺序**产出（3.13 起可 `async for`）—— "谁先回来先处理" |
| `asyncio.timeout(d)` | 给一段代码限时（3.11+，推荐），抛内置 `TimeoutError` |
| `asyncio.wait_for(aw, d)` | 给单个 awaitable 限时 |
| `asyncio.shield(aw)` | 保护不被取消（扣费 / 写文件） |
| `asyncio.to_thread(fn, *a)` | 同步阻塞的兜底 |

### 5.10 坑清单

- 漏 `await` ⇒ 不执行、不报错、耗时 0.00 s（只看 Warning 不看耗时最容易漏）
- 有 `await` 不等于让出（5.4）；要强制让一轮写 `await asyncio.sleep(0)`
- `return` 是**结束**不是让出；结束之后循环才收回控制权
- `time.sleep` / `requests` 写进 `async def` = 整套并发作废（5.7）
- `create_task` 不接返回值 ⇒ 任务可能被回收；不 await 也不取结果 ⇒ 异常凭空消失（`Task exception was never retrieved`）
- `gather` 抛首个异常后其余任务**不被取消**，但 `main` 一退，`asyncio.run` 收尾会把它们 cancel ⇒ 要观察"跑完"必须在 `except` 后继续 await
- `asyncio.wait()` **不能**直接传协程（3.11 起禁止），先 `create_task`
- 3.10 起所有 asyncio API 的 `loop` 参数已移除；3.14 起 `get_event_loop()` 不再隐式建循环 ⇒ 入口一律 `asyncio.run()`

### 5.11 判据（可复测）

1. 顺序 `await` 两次 `sleep 1` = **2.02 s**；`gather` = **1.00 s**；`TaskGroup` = **1.00 s**
2. `gather` 结果顺序 = 传入顺序（`b` 只睡 0.2 s 仍排第二）
3. `time.sleep` + `gather` = **2.00 s**；`await asyncio.to_thread(time.sleep, 1)` = **1.00 s**
4. 一个任务抛异常：`gather` 下另一个**照跑完**；`TaskGroup` 下**被取消**并抛 `ExceptionGroup`
5. `await` 内部无挂起的协程 ⇒ 输出零交错；`await asyncio.sleep(0)` ⇒ 严格交错

---

## 6. 推导式 / 切片 / 迭代器：写规格、造新壳、走一次（2026-11-02 · S0 缺口）

> 这三样合在一章，因为它们是**同一件事的三个面**：**"怎么描述一整批数据"**。
> 推导式管**造**，切片管**取**，迭代器管**走一遍**。
> 代码：`practice/python/2026/11/02-comprehension/`（判据 17 条）｜ `practice/python/2026/11/02-slice-iter/`（判据 16 条）

### 6.1 推导式：写的是规格，不是步骤

`for` + `append` 描述**过程**（建空表 → 遍历 → 判断 → 追加）；推导式描述**规格**（从哪进料 / 哪些放行 / 放行后加工成什么）。

- **产出物 = 最前面那个表达式**，`for` 之后全是数据源。
- **`if` 的位置决定语义**（今天最值钱的一条）：
  - `[... for x in xs if cond]` —— 后置 `if` 是**安检门**（过滤，会丢元素）
  - `[... if cond else ... for x in xs]` —— 前置三元是**分流**（进来的都要，长度不变）
  - 同一批数据、只差 `if` 位置：**分流长度 3 / 过滤长度 1**（实测）
  - 前置三元**必须带 `else`**，否则 `SyntaxError`（它不再是过滤）
- **多重 `for` ≠ 嵌套推导式**：`[n for row in m for n in row]` 压平成一层；`[[n for n in row] for row in m]` 保持两层。区别在**产出物本身是不是推导式**。书写顺序 = 等价双层循环顺序（外层先写），写反是 **`NameError`** 而不是 `TypeError`（最左 `for` 的可迭代对象在**外层作用域**求值）。
- **元组表达式必须加括号**：`[x, x**2 for x in range(3)]` 直接 `SyntaxError`；`[(x, x**2) ...]` 才合法。
- **三元只求值被选中的那一支** —— 没选中的分支里若有函数调用，不会执行。
- **可读性**：三元两支只有**一部分**不同时，把三元缩到那部分上（`("even" if c else "odd", x)` 优于整个元组写两遍）。
- **作用域**：推导式有**独立隐式作用域** ⇒ 迭代变量不泄漏（`n` 跑完仍是原值）；但**最左 `for` 的可迭代对象在外层求值**。
- **3.12 起（PEP 709）list/dict/set 推导式已内联**，不再创建独立函数帧；隔离语义未变（靠 `LOAD_FAST_AND_CLEAR` 保存/恢复外层同名变量）。微基准 ≈ **1.96x**、pyperformance ≈ **11%**。⚠️ "推导式编译成独立 `<listcomp>` 帧"是 **3.12 前**的老说法。
- **dict 推导式**：**3.8 起 key 先于 value 求值**（之前未定义，CPython 实为先 value 后 key）；**重复键静默覆盖**，不报错 ⇒ 想守住只能数条数（`len(got) == len(items)`）。值写变量名 = 存**引用**（`is` 为 True），写 `dict(item)` 才是副本。
- **空容器字面量**：`{}` 是空 **dict**；空 set 只能 `set()`。但**空推导式 `{x for x in []}` 返回的是 set** —— 字面量才歧义。
- **不需要为"空输入"写 guard**：`for` 零次迭代即零次产出，天然返回空容器；写了反而让"空输入判据"失去区分度。

### 6.2 切片：新对象 ≠ 深拷贝

两件不同的事，必须分开记：

1. **`s[i:j]` 永远返回新容器** —— 切片是读操作，原容器一动不动。
2. **新容器里装的是原来那些元素的引用** —— 所以整段切片 `a[:]` 是**浅拷贝**：外层 `is not`、内层**同一个对象**（改内层，原列表跟着变）。

**半开区间 `[start, stop)`**：终点**取不到**。四个好处：长度 = `stop - start`（无 `±1`）、相邻切片**无缝拼接**（`s[:k] + s[k:] == s`）、**能表示空区间**（`s[i:i] == []`）、与 `range` 一致。

- **边界在"缝"上不在元素上**：`n` 有 5 个元素就有 **6 条缝**；`stop` 缺省值 = `len(s)`（末尾之外一条缝）⇒ **省略 stop 反而包含最后一个元素**，想排除才要显式写 `[:-1]`。
- **越界一律截断、不报错**（`[1,2,3][1:100]` → `[2,3]`；`[-100:]` → 全部）；**索引越界才抛 `IndexError`**。
- **负步长**：`k<0` 时缺省端点方向反转（起点 = 末尾、终点 = 开头之前）⇒ `n[::-1]` 完整反转。**唯一会报错的输入是 `k == 0`**（`ValueError: slice step cannot be zero`）。
- ⚠️ **`n[3:-1:-1]` 是 `[]`**，不是 `[3,2,1]` —— 终点位置的 `-1` 照样按 `len-1` **折算**，方向就反了；想到头必须写 `n[3::-1]` 或 `n[3:None:-1]`。
- **切片赋值是另一个操作**：`s[i:j] = t`（`k=1`）**原地改 + 长度可变**（`[1,2,3,4,5]` 的 `[1:3] = [9]` → `[1,9,4,5]`）；`k != 1` 时 **`t` 必须等长**，否则 `ValueError`。切片赋值**不是表达式**，拿不到返回值（必须"赋值语句 + `return s`"两行）。
- **`del s[i:j]`** 原地删除、返回 `None`。
- 三种"反转"三种语义：`s[::-1]`（新 list）/ `reversed(s)`（**迭代器**）/ `s.reverse()`（原地改、返回 **`None`**）。

### 6.3 可迭代 vs 迭代器：一次性是分界线

- **可迭代（iterable）** = 能给你一个游标（有 `__iter__`）；**迭代器（iterator）** = 自己就是游标（`__iter__` **返回自身** + `__next__`）。
- ⇒ **迭代器一定是可迭代，反之不成立**；`iter(it) is it` 为 True（幂等）。
- **`for` 背后三步**：`iter(容器)` → 反复 `__next__()` → 捕获 `StopIteration` 终止。`StopIteration` **不是错误，是"读完了"的协议信号**。
- **一次性**：同一个迭代器 `list()` 两次 → 第二次 **`[]`**，**且不报错**。列表没这问题（每次 `iter()` 都造新游标）⇒ **函数收到迭代器当"容器"遍历两遍，第二遍静默空**，修法是先 `list()` 物化。
- **迭代器不是序列**：没有 `len()`、不支持下标，两者都抛 `TypeError`。
- **`generator` 三自动**：函数体出现 `yield` 就把函数变成生成器函数（调用**不执行**函数体，只返回生成器对象）；`__iter__`/`__next__`/`StopIteration` 全自动，局部变量与执行位置自动保存。
- **`yield` = 暂停 + 交出**（保留状态），`return` = 结束（丢状态）。生成器里 `return` 可提前结束（值进 `StopIteration.value`）。
- **`yield from`**：`for` 里只有一句 `yield` 时 Ruff 报 **`UP028`** 建议改成它 —— 语义相同但**在 C 层委托**（少一层 Python 循环），且能透传 `send()`/`throw()`/`StopIteration.value`（PEP 380）。
- **`__next__` 的停止条件是"低于下界"**，不是"等于最后一个合法值"：写 `<= 1` ⇒ `Countdown(3)` 只产出 `[3, 2]`，**无异常无警告**。
- **官方健全性要求**：一旦抛过 `StopIteration`，后续每次调用**都必须继续抛**，否则视为 broken。
- `__iter__` 返回 `self` ⇒ 该对象**只能遍历一次**；要可重复就让 `__iter__` 返回**新的**游标（生成器天然如此）。
- 注解：`-> Self`（PEP 673，3.11+）比写类名更准（继承时指向子类）；3.14 起 PEP 649 让注解**默认延迟求值**，字符串前向引用不再必要。

### 6.4 判据（可复测 · 汇总）

1. 后置 `if` vs 前置三元：同输入 `[1,2,3]` ⇒ 长度 **1 vs 3**
2. `even_squares([])` → `[]`；`{x for x in []}` 类型是 **`set`**；`{}` 是 **dict**
3. `index_by_id(sample)["a"] is sample[0]` → **True**（引用，不是副本）
4. 多层 for 写反 → **`NameError`**；`[x, x**2 for x in ...]` → **`SyntaxError`**
5. `rows[:]`：外层 `is not`、内层 `is` 相同；改内层 → 原列表跟着变
6. `[1,2,3][1:100]` → `[2,3]`；`[1,2,3][100]` → **`IndexError`**；`[::0]` → **`ValueError`**
7. `n[3:-1:-1]` → **`[]`**；`n[3::-1]` → `[3,2,1,0]`
8. `s[1:3] = [9]` 后长度 5→4 且 `is` 原列表；`s[::2] = [1,2,3]` → **`ValueError`**
9. `it = iter(nums); list(it)` 两次 → `[1,2,3]` / **`[]`**；`iter(it) is it` → True
10. `len(it)` / `it[0]` → **`TypeError`**
11. `Countdown(3)` → `[3,2,1]`；`sum(x*x for x in [1,2,3])` → **14**，再 `list()` → `[]`

---

## 7. httpx 与一次 HTTPS 请求的完整账单（2026-11-03 · S0 提前）

> 本章把「发一个请求」拆成**两张账单**：握手费（连接层）与等待重叠（并发层）—— 两个实验各测一行。
> 代码：`practice/python/2026/11/03-httpx/`（`fetch_basics.py` 判据 4 条 + `fetch_concurrent.py` 判据 4 条）
> 官方依据：python-httpx.org 的 Quickstart / Advanced·Clients / Async 三页互证

### 7.1 请求/响应模型：GET = 按门牌号寄信，不是去某处"查找"

- **URL 解剖**：`https://httpbin.org/delay/1` = scheme + 主机 + 路径。**主机走连接层**（DNS → TCP/TLS），**路径进请求行**（`GET /delay/1 HTTP/1.1` + `Host:` 头）原文写给服务器。
- **服务器是路由匹配的程序，不是文件柜**：`/delay/1`、`/status/404` 后面没有存好的文件，是服务器代码现场生成的响应；都不匹配 → 404。
- **`params` 把拼接交给 httpx**：手工拼查询串会漏 percent-encoding（`好` → `%E5%A5%BD`）；`params={"q": "httpx", "page": "1"}` 值全是**文本**（写 `"1"` 不写 `1`），值可以是 list（同名键重复）。
- **`r.url` 是 URL 对象不是字符串**：`"q=httpx" in r.url` 直接 `TypeError`，要 `str(r.url)`；更严的验法是 `r.url.params["q"]`（结构化取值，不依赖子串顺序）。
- **`raise_for_status()` = 「状态码」与「异常」之间的翻译官**：httpx 默认**不把状态码当异常**（404 是服务器好好回的话）；`raise_for_status()` 对非 2xx 抛 `HTTPStatusError`，对 2xx **不抛且返回 Response**（可链式：`httpx.get(url).raise_for_status().json()`）。3xx **不算**错误（httpx 默认不跟随重定向，要 `follow_redirects=True`）。
- **异常两支是兄弟不是父子**：`HTTPError` = 基类；`RequestError`（请求没走完，带 `.request`）∥ `HTTPStatusError`（走完了但非 2xx，带 `.request` + `.response`）。**互不为子类 ⇒ `except` 顺序无所谓**；坏域名在 `get()` **内部**就抛（连接没建立 ⇒ 根本没有响应和状态码，`raise_for_status()` 无从谈起）。
- **默认超时 5 秒**（网络无活动）；`timeout=None` 禁用。

### 7.2 Client：连接池 = 把付过的握手费留住

- **一次 HTTPS 首请求的账单**：DNS（1 次，可缓存）→ TCP 三次握手（1 RTT，"双方都确认对方收得到"恰好三次）→ TLS 握手（1-2 RTT：证书防冒充 + 非对称换对称密钥）→ HTTP 往返（才开始传正文）。
- **顶层 API 每个请求各付全套；`Client` 用连接池复用 keep-alive 连接**，握手只付一次 —— 这是「复用更快」的全部根。**判据 4 两组请求数完全相同（都是 5 个），差的只是握手次数**，别理解成"Client 能一次抓多个"。
- **实测（5 × `httpbin.org/get`）**：每次新建 **3.431 s**（0.686 s/请求）vs 复用 **1.573 s**（0.315 s/请求）⇒ 握手费真实单价 **≈ 0.37 s/次**，复用快 **2.2×**。
- **单批两版账单几乎一样，差别在第二批起**：并发批共享 client 也照样开多条新连接（握手互相重叠）；省下的钱在**第二批**（keep-alive 连接还在池里，零握手）—— 官方"别在热循环里实例化 client"的精确含义是**损失在复用，不在单批**。
- **`AsyncClient` = 协程版**：方法要 `await`；`await` 只能写在 `async def` 里（普通函数里是 `SyntaxError`）。`async with ... as client` = 自动建池 + 离开时自动 `aclose()`（防连接泄漏）。

### 7.3 并发抓取：省的全是"等待"

- **实测（5 × `httpbin.org/delay/1`，服务器故意睡 1 s）**：串行 **7.036 s** ≈ 5 × (1 s 等 + 0.4 s 握手)；`gather` 并发 **1.723 s** ≈ 单个请求的完整成本 ⇒ **加速比 4.1×**。与第 5 章 `serial 2.02 s` vs `gather 1.00 s` 同一条定律：**等待不占 CPU，所以能重叠**。
- **协程 ≠ 执行**：`tasks = [fetch_one(...) for ...]` 此刻**一个请求都没发**，只是 5 张工单；`asyncio.gather(*tasks)` 拆包一次性提交，结果**按传入顺序**返回。漏 `await` ⇒ 错误**后移**到"用结果的地方"（`RuntimeWarning: never awaited` + 断言 `TypeError`），结果列表里装的是协程对象。
- **`*` 的对仗**：定义 `def f(*args)` 是**打包**，调用 `f(*lst)` 是**拆包**；不拆包 = 把整个列表当**一个**参数 ⇒ `gather` 报 `TypeError`（要求每个参数都是 awaitable）。
- **短板效应**：`gather` 要等**全部** ⇒ 抓 5 个**不同主机**，总耗时 ≈ **最慢单个**（不是单个）；同主机同延迟才近似单个。给并发批做超时按最慢者算，配 `return_exceptions` 防一个拖垮全场。
- **计时纪律**：单样本是随机变量（一个请求抽风就翻车）+ 先跑的批付冷启动费 ⇒ **耗时对比只记数字、复跑取稳态，不写死 assert**。

### 7.4 判据（可复测 · 汇总）

1. `params={"q": "httpx", "page": "1"}` → `str(r.url)` 里同时有 `q=httpx` 与 `page=1`
2. `/status/404` → `HTTPStatusError`（`exc.response.status_code` == 404）；`no-such-host.invalid` → `RequestError`
3. 5 × `/get`：新建 **3.431 s** vs 复用 **1.573 s**（差额 ≈ 每请求 0.37 s 握手费）
4. 5 × `/delay/1`：串行 **7.036 s** vs `gather` **1.723 s**（**4.1×**）
5. 两条结果条数 == 5 且全 200；JSON 落盘 `json.load` 读回 == 内存对象
6. `gather` 不拆包 → **`TypeError`**；漏 `await` → 结果是协程对象 + `never awaited`

---

## 第 8 章 · OOP 与协议：dataclass / Protocol / `__dunder__`（2026-11-04）

### 8.1 `@dataclass` = 代码生成器，不是校验器

- **它只干一件事：按字段顺序生成方法**。返回的是**同一个类**（`slots=True` 除外）。生成物：`__init__` / `__repr__` / `__eq__` 默认开；`__lt__` 家族要 `order=True`；`__hash__` 看条件；`__setattr__` 由 `frozen=True` 注入。
- **C# 对照**：`@dataclass` ≈ `record`（干掉 `Equals` / `GetHashCode` / `ToString` 那一坨）；C# 编译器做，Python 运行时做。
- ⭐ **除 `ClassVar` / `InitVar` 两处例外，完全不检查注解里写的类型**。`Plain(name=123)` 在 CPython **3.14.8 实测构造成功**，`repr` 出来是 `name=123`。
- **两套门要分清**：① 注解本身（没人管）→ ② **静态检查器**（Pylance / Pyright / mypy 读注解**报红**，但只是提示）→ ③ **运行时**（解释器不管，**只有你自己写的校验能拦**）。编辑器红字与"能不能跑"是两层，互不决定。
- `dataclass` 唯一的"静态错误"是**默认值顺序**：无默认值字段不能跟在有默认值字段后面（`TypeError`）。

### 8.2 `__post_init__` 是唯一的校验钩子

- 生成的 `__init__` **末尾**会调 `self.__post_init__()`；**自己写了 `__init__` 或 `init=False` 时不会调**。`replace()` 内部走 `__init__` ⇒ 也会执行。
- **顺序：先校验、后派生**。反了就会拿非法输入算出"看着挺正常"的脏值（`-1 + 58 = 57`），属于静默 bug。
- **派生字段用 `field(init=False)`**：不进 `__init__` 参数表 ⇒ 调用方**传不进来**（`TypeError: takes 2 positional arguments but 3 were given`），签名层面堵死"`total` 与两个分量自相矛盾"这种对象。代价：**没默认值 + 没在 `__post_init__` 赋值 ⇒ 属性不存在**，构造成功、第一次读（`repr`）才 `AttributeError`。
- ⭐ **库里的校验必须 `raise`，不能 `assert`**：`python -O` 会把**所有** `assert` 语句整条删掉（实测 `-O` 下 `assert False` 静默消失）⇒ 用 assert 写的校验在优化模式里等于没有。练习里的**判据**用 assert 没问题，那是开发期自检。
- **可变默认值必须 `field(default_factory=list)`**：默认值存在**类属性**上，生成的 `__init__` 等价于 `def __init__(self, xs=xs)` ⇒ 所有实例共享同一个对象（与第 3 章默认参数陷阱同根）。**3.11 起判据改成「不可哈希即禁」**（近似"可变"），在**装饰时**抛 `ValueError`，不是实例化时。`default` 与 `default_factory` **不能同时给**。
- **`default_factory` = 缺省值工厂，与"输入类型"无关**：只在**没传值**时调一次（实测显式传值时计数不增）；没有任何类型强制（传 `tuple` / `None` 都行，报应发生在用 `list` 方法那一刻）。
- **`frozen=True` 是伪不可变**：注入 `__setattr__` / `__delattr__` ⇒ `FrozenInstanceError`（`AttributeError` 的**子类**，宽 `except` 会先吞掉）。三条边界实测：① **浅冻结**（里面的 `list` 照样 `append`）② `object.__setattr__` **能绕过** ③ `frozen` 只**允许**生成 `__hash__`，字段里塞 `list` 则 `hash()` 照样 `TypeError`。
- **`frozen` + `eq` ⇒ 自动 `__hash__`**；**可变** dataclass 的 `__hash__` 被置 `None`（能被改的键放进 dict 就永久找不回来）—— **不可变才配当字典键**。
- **`frozen` 的 `__post_init__` 里赋值要走 `object.__setattr__`**：注入的 `__setattr__` 连自己的 `__init__` 都拦。
- **给类加必填字段会打破已有调用点**：加 `swallow` 参数必须给默认值，否则原来的无参调用直接 `TypeError`（与 dataclass 加无默认值字段同一个坑）。

### 8.3 `Protocol` = 结构化子类型（形状像就算）

- **C# 的 `interface` 是名义子类型**：必须 `: IClosable` 登记入会。**PEP 544 的 `Protocol` 是结构化子类型**：不用继承、不用 import，被适配的类甚至不知道自己被适配。第三方库的 client 改不了继承链时，这套是唯一出路。
- ⭐ **默认只在静态检查器里生效**：未加 `@runtime_checkable` 的协议用于 `isinstance` / `issubclass` → **`TypeError`**。
- **`@runtime_checkable` 只查"成员名在不在"**：`Pretender.close(self, force: bool) -> int` 与协议完全不符，`isinstance` **照样 True**。⇒ 它**不是 validator**。
- 数据成员（`name: str`）也按"存在"算，**不验类型**；缺了就 False。
- 其余限制：运行时可检查性**不被子协议继承**；`isinstance` 对协议**明显更慢**（3.12 起改用 `inspect.getattr_static`）；成员在类创建后**运行时"冻结"**（monkey-patch 不影响判定）；3.15 起"继承自可检查协议但自己没加装饰器"的用法**已弃用**（3.20 抛 `TypeError`）。
- **优先用标准库现成协议**：`collections.abc` 的 `Sized` / `Iterable` / `Iterator` / `Callable` 就是预置协议；`typing.Callable` 等旧别名已 deprecated。

### 8.4 `__dunder__` = 把类接进语言协议

- **特殊方法由类型决定**：`len(x)` 找 `type(x).__len__`，不是"调用 `x.len()`"。实现对应 dunder，`len` / `for` / `in` / `with` 直接认它，**不需要继承任何东西**（这与 `Protocol` 是同一套机制的"语言内建版"）。
- **`__len__` 必须返回非负整数**：返回 `-1` → `ValueError: __len__() should return >= 0`。
- **`__iter__` 用 `yield` 或 `return iter(...)` 都行**；`yield` 让函数变**生成器函数**（调用时不执行函数体，第一次 `next` 才开始 —— 错误也随之**后移**）。
- **可迭代 vs 迭代器在类里的分工**：`__iter__` 每被调用一次就**新建**生成器 ⇒ 实例可重复迭代；若把迭代器缓存到 `self._it`，第二次就是 `[]`（第 6 章那条"迭代器一次性"的实战后果）。
- ⭐ **`__exit__` 的返回值决定吞不吞**：`with` 的等价展开里是 `except: if not exit_(manager, *sys.exc_info()): raise` ⇒ **真值吞、假值放行**。正常退出时三个参数全是 `None`（`__exit__` 仍会被调用）。
- **异常三联**：`exc_type` 是**类**（不是实例）、`exc` 是**实例**、`tb` 是 `types.TracebackType`（`traceback.format_exception(exc_type, exc, tb)` 要用它）。声明用 `BaseException` 而非 `Exception` —— `KeyboardInterrupt` / `SystemExit` 也是 `BaseException` 的子类，`with` 里按 Ctrl+C 一样会传进来。
- **`...`（`Ellipsis`）是真值**：`__exit__` 返回 `...` ⇒ **吞掉所有异常**。协议里写 `...` 是"只有形状"的约定；**实现类里留 `...` 就是静默 bug**。`def f() -> int: ...` 实际返回 `None`（注解连自己都不保证）。
- **3.14 起注解不需要引号**：PEP 649 延迟求值 ⇒ `-> "Self"` 这类前向引用写法被 ruff **UP037** 强制去掉。

### 8.5 判据（可复测 · 汇总）

1. `Plain(name=123)` / `P(name=123, n=0)` → **构造成功**（注解不校验）
2. `TokenUsage(38, 58).total_tokens == 96`；`TokenUsage(-1, 0)` → `ValueError`
3. `@dataclass class Bad: xs: list = []` → **装饰时** `ValueError: mutable default ... not allowed`
4. `frozen` 实例赋值 → `FrozenInstanceError`；同值两实例进 `set` 只剩 **1** 个
5. 无 `@runtime_checkable` 的协议 `isinstance` → `TypeError`；有装饰器时**签名错的类也通过**
6. `Counter(3)`：`len == 3`、`list == [1, 2, 3]`、`2 in` 真 / `9 in` 假
7. `__exit__` 返回 True ⇒ `with` 体内异常被吞（后续语句照跑）；返回 False ⇒ 异常放行
8. `python -O -c 'assert False; print("照跑")'` → **print 执行**（assert 被剥掉）

---

## 附录 · 待学章节（计划编号 · 学到时展开成正文）

> 只列主题，不提前写内容 —— 避免"抄来的笔记"占位。

| 阶段 | 章节 | 主题 |
|---|---|---|
| S0 | 4 | ✅ 已展开为 **第 4 章**（2026-10-30）；`Counter` / `defaultdict` **用到时再补** |
| S0 | 缺口 | ✅ 已展开为 **第 6 章**（2026-11-02）：推导式 / 切片 / 可迭代与迭代器 —— 原计划未单列，属 10-31 复盘补的缺口 |
| S0 | — | ✅ 已展开为 **第 7 章**（2026-11-03）：httpx / 一次 HTTPS 请求的账单 / 并发抓取 —— 原排期 11-05，随交付物地基提前 2 天落地 |
| S0 | D4 | ✅ 已展开为 **第 8 章**（2026-11-04）：OOP 与协议 —— dataclass 生成物 / `__post_init__` 校验 / Protocol 结构化子类型 / `__dunder__` 接进语言协议 |
| S0 | 5 | 函数进阶：`*args / **kwargs`、闭包、装饰器（对照 C# 委托 / Attribute） |
| S0 | 6 | 模块与包：`import` 机制、`__name__`、虚拟环境 |
| S0 | 7 | ✅ 已展开为 **第 5 章**（2026-10-31）：`async/await` 与 C# `Task` 的异同、事件循环。⚠️ 正文章号按**实际学习顺序**编（第 5 章 ↔ 本表编号 7），两者不再一一对应 |
| S0 | 8 | httpx 并发抓取 + 存 JSON（S0 交付物） |
| S0 | 9 | pydantic 数据校验（S1 结构化输出的前置） |
