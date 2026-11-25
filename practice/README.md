# practice —— 练习区（两个桶 · 年/月分层）

> 为验证某个 API / 写法而建的**一次性小工程**：不追求完整、不占交付物名额，学完留在原地当参照。
> 完整交付物放 `../projects/`。

## 分桶判据（新建练习前先问一句）

| 桶 | 判据 | 例子 |
|---|---|---|
| **`python/`** | 只跑标准库 / 只教 **Python 语言本身**（语法、容器、可变性、装饰器、async、httpx、pydantic） | 语法对照、list 容器、可变性陷阱 |
| **`ai/`** | 涉及 **模型 / token / prompt / 工具调用 / RAG / 评测**（要 key、要计费、要联网） | `chat.completions`、流式、结构化输出 |

**边界情形**：只装了 `openai` 但一次模型都没调的脚手架（如环境基线）→ 归 `python/`。

> ⚠️ 将来某个桶**同时**满足「出现工程级独立文档」+「单个练习超 1 万个文件」两条，才值得升为顶级目录；到不了就留在这里。

## 路径与命名（**两条轨道**，2026-11-17 起）

```
practice/<桶>/<年>/<月>/<日期-主题>/     ← 轨道 A：单次验证型（一天写完）
practice/<桶>/<关卡编号>-<主题>/         ← 轨道 B：关卡型（跨天推进）
```

| 轨道 | 什么时候用 | 例 |
|---|---|---|
| **A 单次验证** | 一天内验完一个写法 / 一个 API，验完留原地当参照 | `python/2026/10/30-mutability/` |
| **B 关卡** | 一个关卡要分几步写、后一步 `import` 前一步（L1–L8 主线） | `ai/l1-bigram/`、`ai/l2-attention/`、`ai/l5-agent/` |

| 规则 | 说明 |
|---|---|
| **为什么 A 分 `年/月`** | 这是一年的仓库 —— 按天平铺一年会堆出几百个目录；分到 `年/月` 后，一个月最多 30 个左右 |
| **A 的叶目录带完整日期** | `30-mutability` 而非 `mutability`：脱离父目录也能看出是哪天，排序天然是按日 |
| **B 不带日期，日期写在目录 README 头部** | 关卡是「一个概念的连续几步」（如 L1 = 计数 → 交叉熵 → softmax → 平滑），拆成四个日期目录会**切断 import 链**、复制语料、把一张判据表拆成四份。日子在 `README.md` 打卡行和每日复盘里各记了一份，练习目录不必再背 |
| **B 内部按「步」分文件** | `l1-bigram/` 里 `bigram.py` → `entropy.py` → `softmax.py` → `smooth.py`，一份 `README.md` 管全关的判据表与实测数字 |
| **长期项（非关卡）** | 跨月推进的独立项目（如 nanoGPT）同样不带日期：`practice/python/nanogpt/` |

## 内容约定

- 每个练习至少一个能跑的 `.py`，文件头一行 `# 目的 / 判据`
- 各自带 `README.md`：**判据表 + 实测数字 + 踩过的坑**
- **不逐条登记详情**；本文件下方两张表是唯一索引，加一行判据即可

## 跑 AI 调用的规矩（2026-10-30 起）

| 规则 | 说明 |
|---|---|
| **调模型的脚本由用户亲手跑** | 涉及**计费**与 **key**；AI 不代跑任何会调用模型的脚本 |
| AI 被要求代跑时 | **必须把完整输出原样贴回**，不许只说"跑通了" |
| 非模型脚本（纯标准库 / 环境检查 / ruff） | AI 可代跑，同样贴输出 |

## 已有练习

### `python/`（语言层）

| 目录 | 主题 | 关键判据（实测） |
|---|---|---|
| `2026/10/28-python-env/` | uv 工程骨架 + 环境自检（S0 前置） | `uv run python check_env.py` 9 个包无 MISSING；`uv run ruff check .` 全绿；`sys.prefix` 指向本工程 `.venv` |
| `2026/10/30-mutability/` | 可变性陷阱（默认参数 / 函数传参 / `+=`） | `mut_default.py`：`f(1/2/3)` → `[1]` / `[1, 2]` / `[1, 2, 3]`，`f.__defaults__` 变 `([1, 2, 3],)`、`f(1) is f.__defaults__[0]` 为 `True`；`g.__defaults__` 恒 `(None,)`。`mut_arg.py`：`append` 调用方可见（`id` 不变）、`nums = nums + [x]` 不可见、返回新对象 `id` 不同、函数内 `+=` 原地改；`ruff check` 全绿 |
| `2026/10/30-containers/` | **dict / set / tuple**（S0 第 4 章）：tool schema 形状、合并的 id 判据、可哈希 | `dict_schema.py`：schema 可 `json.dumps` 且 `required is` 入参 = `True`；`KeyError` / `None` / 默认值 `[]` 两次 `is` = `False`；`\|` 换 id、`\|=` 保 id。`set_tuple.py`：去重 **6→3** 且与手工一致；`list` / `dict` / `set` 不可哈希、`(1, [2])` 不可哈希；`{1, True, 1.0}` 元素数 **1**；嵌套 `[1, [2, {"a": 3}, (4, [5])]]` 可哈希（判据 **7/7 绿**，10-30 补测） |
| `2026/10/31-async/` | **async / await 与事件循环**（S0 第 5 章）：让出点、`gather` vs `TaskGroup`、同步阻塞 | `bench.py`：`serial` **2.02 s** / `gather` **1.00 s** / `TaskGroup` **1.00 s**；直接 `time.sleep` + `gather` **2.00 s** → `to_thread` **1.00 s**；`gather` 抛异常后其余任务**不取消**（`safe` 照跑完）、`TaskGroup` **取消**并抛 `ExceptionGroup`；`ruff check` 全绿 |
| `2026/11/02-comprehension/` | **推导式**（S0 缺口）：list / dict / set 三种筐、多重 for、`if` 位置的两种语义、独立作用域 | `comprehension.py`：判据 **17/17 绿**；后置 `if`（过滤）长度 **1** vs 前置三元（分流）长度 **3**；`got["a"] is sample[0]` → **True**；多层 for 写反 → **`NameError`**；`[x, x**2 for ...]` → **`SyntaxError`**；嵌套转置 == `[list(t) for t in zip(*m)]`；`{x for x in []}` 是 **set**、`{}` 是 **dict**；`ruff check`/`format` 全绿 |
| `2026/11/02-slice-iter/` | **切片 + 可迭代 vs 迭代器**（S0 缺口）：新对象/浅拷贝/切片赋值/越界、迭代协议与一次性 | `slice_basics.py` 判据 **8/8 绿**：`rows[:]` 外层 `is not` 但内层 `is` 同（浅拷贝）；`[1:3] = [9]` 长度 **5→4** 且 `is` 原列表；`[1:100]` → `[2,3]` 不报错、`[100]` → `IndexError`；`[3:-1:-1]` → **`[]`**。`iter_protocol.py` 判据 **16/16 绿**：`iter(it) is it` → True；`list(it)` 两次 → `[1,2,3]` / **`[]`**；`len(it)`/`it[0]` → `TypeError`；`Countdown(3)` → `[3,2,1]`；`sum(genexpr)` **14** 后 `list()` → `[]` |
| `2026/11/03-httpx/` | **httpx 入门**（S0 提前）：Client 复用 vs 每次新建（握手账单）、串行 vs `gather`、JSON 落盘读回 | `fetch_basics.py` 判据 **4/4 绿**：404 → `HTTPStatusError`、坏域名 → `RequestError`；新建 **3.431 s** vs 复用 **1.573 s**（5 × `/get`，差 ≈ 0.37 s/请求握手费）。`fetch_concurrent.py` 判据 **8/8 绿**：串行 **7.036 s** vs `gather` **1.723 s**（**4.1×**，5 × `/delay/1`）；JSON 读回 == 内存对象；`uvx ruff check` 全绿 |
| `2026/11/04-oop-protocol/` | **OOP 与协议**（S0 D4）：dataclass 生成物 / `__post_init__` 校验 / `default_factory` / `frozen`；Protocol 三件事；`__len__` / `__iter__` / `__enter__` / `__exit__` | `dataclass_basics.py` 判据 **5/5 绿**：`Plain(name=123)` **构造成功**（注解不校验）、`field(init=False)` 派生 `total` == **96**、负数 → `ValueError`、可变默认值**装饰时** `ValueError`、frozen 赋值抛 `FrozenInstanceError` 且同值进 set 只剩 **1** 个。`protocol_dunder.py` 判据 **5/5 绿**：未加 `runtime_checkable` 的协议 `isinstance` 抛 `TypeError`、**签名全错的 `Pretender` 也通过**、缺 `name` 的 `NoName` 不通过；`Counter(3)` → `len 3` / `[1, 2, 3]`；`__exit__` **True 吞 / False 放行**各验一次。**变异测试 5 次全部变红**（假绿两次的教训） |
| `2026/11/05-functions-modules/` | **装饰器与闭包**（S0 D3）：`@timer` 在 def 时执行、`*args/**kwargs` 透传、返回值透传、异常不吞、`functools.wraps` 工牌（四项元数据）、闭包晚绑定 vs 默认参数钉死、`record` 观测"装饰时 vs 调用时" | `decorators.py` 判据 **4/4 绿**：`calling(2,3)`→`5` / `calling(2,3,scale=10)`→`50`（关键字参数穿过替身）、`ValueError` 经 `try/except/else` 接到、`bare.__name__`=`'wrapper'` vs `kept.__name__`=`'add'` 且 `kept.__wrapped__ is add`、`make_multipliers(3)`→`[6,6,6]` vs `snapshot_by_default(3)`→`[2,4,6]`、`def` 之后 `EVENTS` 已有"装饰"那笔。**变异测试 6 次全红**（丢 kwargs / 丢 return / 吞异常 / 去 wraps / 闭包也钉死 / 记录挪进 wrapper） |
| `2026/11/05-exceptions-pathlib/` | **异常链 + pathlib**（S0 D5）：`raise ... from` 三态（`__cause__` / `__context__` / `__suppress_context__`）、`from None` 只切显示、`Path` 的 `/` 拼接与 `read_text`/`write_text` | `exceptions.py` 判据 **1/1 绿**（内含 7 条断言）：显式 `from exc` ⇒ `__cause__` 是 `ValueError` 且 `suppress_context` True；裸 `raise` ⇒ `cause` None、`context` 非空、`suppress` False；`from None` ⇒ `suppress` True 但 `context` 仍在。`paths.py` 判据 **1/1 绿**（5 条）：中文往返一致、`path == root/"notes"/"day.txt"`、`read_bytes() == text.encode("utf-8")`、`root / "/abs"` 绝对路径重置。**变异 7 次全红**（漏 from / 多 from / from None→from exc / 写入截断 / 路径谎报 / 不给 encoding / gbk 写） |

### `ai/`（LLM 层）

| 目录 | 主题 | 关键判据（实测） |
|---|---|---|
| `2026/10/29-llm-hello/` | **模型 API 打通**（过渡期闸门 · 第一次 `chat.completions`） | 正文非空 + 打印 `usage` 三个 token 数 + 端到端耗时；**换模型只改 `.env` 三个变量、`ask.py` 一行不动**。✅ 已跑通（10-29）：`deepseek-flash` / **38·58·96 tokens** / **1254.49 ms**（复跑 1303.34 ms） |
| `2026/10/30-multi-turn/` | **多轮对话**：`messages` 历史维护（可变性知识第一次实战） | 每轮 `len(messages)` **+2**、消息数 **2→4→6**；`prompt_tokens` **46→62→96**（增量 +16 / +34，与上一轮回答长度成正比 = 每次都重发完整历史）；**第 2 轮完整复述第 1 轮问题**；首轮端到端 **1185.61 ms**（含建连） |
| `l2-nanogpt/` | **L2 关卡型目录（起 11-19）**：六关 0–5，判据全文 = 根 `README.md` 阶段表 L2 行；**有 GPU 依赖**（torch cu128，2–3 GB） | **关卡 0 环境** 判据 2/2 绿：`torch 2.11.0+cu128`（编译期 CUDA **12.8**）· RTX 5060 Laptop · capability **(12, 0)** · 显存 **7.93 GiB**；matmul(1024³) 首次 **388.8 ms** → 稳态 **0.65 ms**（**3.29 TFLOPS**）；attention(B8/T1024/d64) 首次 148.7 ms → 稳态 **1.50 ms**；峰值显存 **126 MiB**；softmax 行和偏差 **2.38e-07**；**关卡 4 真训练** 判据 2/2 绿：**2000 步** · 显存峰值 **0.322 GiB** · **45.7 s**（22.9 ms/步）· val CE **1.7983 < bigram 基线 2.4819**（−27.5%）· 生成 200 字最长重复 **5 字符** · 参数 **826,368 == 手算**；`ruff` 全绿 |
| `l1-bigram/` | **L1 全关（关卡型目录，起 11-16）**：关卡 1 char-level bigram → 条件概率表 ✅ ｜ 关卡 2 交叉熵 / 困惑度 ✅ ｜ 关卡 3 softmax ✅ ｜ 关卡 4 平滑 ✅（闭词表 V=159 + `<UNK>`；0 依赖 · 0 token） | **关卡 1** 判据 2/2 绿：总对数 **421**（= `len-1`）、不同 bigram **350** 种、V **180**、行和最大 `'，'`=**28**；点质量 `'面'→'包'` + 恒等式 `最大差 == 1 − P_global('包')`（**0.9787**），变异 5/5 全红。**关卡 2** 判据 2/2 绿：基线 `log2(158)`=**7.3038** > uni **6.7141** > bi **1.2943**（ppl 158→105→**2.45**），两路互证差 **0.0**；留出集 85 字 / 新字 22 / 未见对 **67/84 = 79.8%** ⇒ 默认口径 **`inf`**，只算见过的对 **1.6935**（ppl 3.23）> 训练 1.2943；边界 `"axd"`→`nan`、`""`/单字→`nan`；**变异 6/6 全红**（分母 / 全局分布 / 删守卫 / 删字造假对 / log2→log / 按 next 聚合）。**关卡 3** 判据 2/2 绿：平移不变误差 **0.00e+00**（c ∈ {1,−5,100}）、精确值 **0.75 / 0.125**、稳定版↔naive 互证 **6.78e-21**；`exp(710)` → **`OverflowError`**、`exp(-746)` → **`0.0`**；`[1000,0]`→`[1.0,0.0]`、`[-1000,-1001]`→**0.73106/0.26894**、病态形状 `[0,1000]`→`[0.0,1.0]`、`[1e6,0]`→`[1.0,0.0]`；**变异 6/6 全红**（分母=max / 减 mean / 减首项 / 不减 / 排序 / 删空守卫）。**关卡 4** 判据 2/2 绿：闭词表 V **159**（= 158 + `<UNK>`）、折叠 token **85→85** / 不同 token **61→40** / 生字 **22 种（出现 24 次）**；MLE held **`inf`** → add-k k=1 **7.0590**（ppl 133.34）· **k=0.1 6.7812（109.99，最优）** · k=0.001 **8.4070**（339.44）；train 1.2943 → **6.1048 / 3.8545 / 1.3944**；移走质量（k=1，`'清'` C=1）= **0.99375**；**变异 7/7 全红**（M6 先漏网，补两路互证才红） |
