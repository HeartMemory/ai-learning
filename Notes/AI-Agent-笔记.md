# AI Agent / 大模型应用笔记（2026-10-28 起）

> AI 主线知识点沉淀，按主题组织，随进度更新（与 `Python-基础笔记.md` 同规格，一本只留一个家）。
> 代码实例都在 `practice/` 下的对应目录里，可运行验证。
> 原则：**只记真实学到的内容与真实踩过的坑**；每个知识点尽量配一条**可实测的判据**。

**覆盖进度**：**第 1–3 章 ✅**（环境基线 / 模型 API 打通 / 多轮对话，2026-10-28 ~ 10-30）｜ **第 4 章 ✅**（并发与异步，11-01）｜ **第 5–8 章 ✅**（L1 四关：计数 → 交叉熵 → softmax → 平滑，11-16 ~ 11-19）｜ **第 9 章 ✅**（GPU 环境与轮子匹配，11-19）｜ **第 10 章 ✅**（L4 关卡 1 流式，11-20）｜ **第 11 章 ✅**（L4 关卡 2 结构化输出，11-21 白天）｜ **第 12–13 章 ✅**（L4 关卡 3 多轮与上下文记账 + 崩溃安全，11-21 夜间）｜ **第 14–15 章 ✅**（L4 关卡 4 成本记账 + prompt 长度与缓存门槛，11-22）｜ **下一关 = L5 工具与循环（手写 agent，禁框架）**｜ ⚠️ **2026-10-29 起 Python 语言层单独成册** → 语法 / 容器 / 可变性 / 装饰器 / async / httpx / **JSON 落盘** / pydantic 一律记进 `Python-基础笔记.md`，**本册只管 LLM 与 Agent 层**

## 回访清单（已迁出）

> 题库单独成文件 → [Notes/回访清单.md](回访清单.md)（活跃 · 周日盲写用）｜ [_archive/回访-已销.md](_archive/回访-已销.md)（已销）
> **本册只留指针**：清单混在正文里，按需查任何一章都会把它整份读进来，白烧上下文。入库门槛与销账规则见清单文件头部。

---

## 1. 环境基线：uv / venv / 解释器隔离（2026-10-28 · S0 前置）

### 1.1 uv 是什么 —— 不是 Python 扩展

- **定位**：Astral（`ruff` 同厂）用 **Rust** 写的**独立可执行程序**，管「解释器 / 项目环境 / 运行」三件事，替代 `pip` + `venv` + `pyenv` + `pipx` + `poetry`
- **判据（实测）**：`(Get-Command uv).Source` → `D:\python\Scripts\uv.exe`，**单文件 40 MB、同级无配套 dll**（静态链接）；在 `C:\` 根目录、无任何工程、未激活环境下 `uv --version` 正常输出版本 ⇒ **它不依赖 Python 就能跑**
- ⚠️ **误区**：`python -c "import uv"` 能成功，是因为用 `pip install uv` 时顺带装了 PyPI 上**同名的包装包**（空壳，`site-packages\uv\__init__.py`），**不是 uv 本体**
- **为什么必须用 Rust 写**：一个"帮你装 Python"的工具**不能依赖 Python 才能跑**（鸡生蛋）；`pip` 正是因此永远装不了 Python 本身

### 1.2 venv 是什么 —— 一个文件夹 + 一张身份证

```
.venv\
├── Scripts\python.exe      ← 启动器（不是复制整个解释器）
├── Lib\site-packages\      ← 本项目独占的包目录
└── pyvenv.cfg              ← 身份证
```

- **机制**：Python 启动时先看自己 exe 旁边有没有 `pyvenv.cfg`；有 → 把 `sys.prefix` 改成该目录，`site-packages` 跟着指过去
- ⭐ **最硬的判据（实测两组输出）**：

| 位置 | `sys.prefix` | `sys.base_prefix` |
|---|---|---|
| 工程内（`uv run`） | `...\2026-10-28-python-env\.venv` | `D:\python` |
| 全局（`cd C:\` 后） | `D:\python` | `D:\python` |

⇒ **两个 prefix 不相等就是在 venv 里**，相等就是在全局。

- `pyvenv.cfg` 两项关键：`home = 真实解释器路径`、`include-system-site-packages = false`（**隔离开关**）
- **为什么 `.venv` 不进 git**：里面写死本机绝对路径（`home = D:\python`）+ 可随时由 `uv sync` 重建

### 1.3 解析 ≠ 安装（今天卡了十几分钟的地方）

- **现象**：`uv sync` 停在 "Downloading…" 不动
- **判据**：`uv.lock` 已生成（346 行 / 50 KB）而 `.venv\Lib\site-packages` 里 **0 个包** ⇒ **卡的是 wheel 下载，不是依赖解析**
- **处置**：换国内镜像（实测握手延迟：**阿里云 582 ms < 清华 976 ms < PyPI 1043 ms**，中科大 HEAD 403）
- **官方写法**（本机 uv 0.12.23 的 `--help` 证据）：用户级 `%APPDATA%\uv\uv.toml` 里写 `[[index]]` + `url` + `default = true`
  - ⚠️ `--index-url` / `UV_INDEX_URL` **已标注 Deprecated** ⇒ 用 `--default-index` / `UV_DEFAULT_INDEX`
- **效果**：同一份 lock，**5.4 秒**装完 19 个包

### 1.4 ruff ≠ 类型检查（最容易混的一对）

| 工具 | 管什么 | 不管什么 |
|---|---|---|
| `ruff` | 风格 / lint（`E F I UP B SIM RUF`）/ 格式化 | **类型检查**（写错 `list[int]` 它不管） |
| `Pylance` / `pyright` | **类型检查**（注解、参数不匹配、返回值） | 风格 / 格式化 |

- **中文项目必配**：`[tool.ruff.lint] ignore = ["RUF001","RUF002","RUF003"]` —— 否则 docstring 里的全角冒号 / 括号被判成"易混淆字符"（实测一次报 9 处）

### 1.5 扩展的「引擎墙」

- 每个扩展的 `package.json` 里有 `engines.vscode` 字段，**客户端内核低于它 ⇒ 装不上**
- **实测**：CodeBuddy（内核 **1.106.1**）装不了要求 `^1.107.0` 的 Pylance / Error Lens / Python Environments；VS Code **1.136.2** 可以
- **判据命令**：`(Get-Content <扩展目录>\package.json -Raw | ConvertFrom-Json).engines.vscode`
- 另一个事实：**不同编辑器产品的扩展目录彼此独立**（`~\.codebuddycn\extensions` vs `~\.vscode\extensions`），用 A 的 CLI 装不到 B

### 1.6 记忆的三层（别把它们当成一个东西）

| 层 | 落点 | 谁写 | 手改有效？ |
|---|---|---|---|
| **画像** | `~\.codebuddy\memery\<UID>_memery.md` | 产品（「本地记忆」开关触发） | ✅ **能落盘**（10-20 受控实验 T0/T1/T2：+100 B / mtime 更新 / **重启会话后标记仍在**）—— ⚠️ 「产品会整份重写」这个旧结论**已被推翻** |
| **记忆条目** | 编辑器面板「保存的记忆」 | AI（每次会话写入） | ✅ |
| **事实源** | 仓库根 `README.md` 打卡表 + `Notes/YYYY/MM/DD.md` | 人与 AI | ✅ **唯一权威** |

- ⚠️ **教训（10-20 修正）**：手改画像**能落盘**，真正的问题是**不可验收**（仓库外 + 无版本控制）⇒ 改前备份 `.bak-YYYYMMDD` → **只写不变量** → **改完立刻读回**（体积 + mtime + 头部版本号）→ 同步 `user-memery-state.json` 的 `current_version`。**不验收 = 没做**
- 记忆**只存不变量**（偏好 / 风格 / 红线 / 入口指针）；**进度、题数、成品数、阶段日期一律读文件**——数字进记忆必然腐烂

---

## 2. 模型 API 打通：一次最小闭环（2026-10-29 · 过渡期闸门）

> 代码在 `practice/ai/2026/10/29-llm-hello/`（`ask.py` = 调用；`check_llm.py` = 接入探针）。

### 2.1 三件配置各管一件事

| 变量 | 管什么 | 生活类比 | 判据 |
|---|---|---|---|
| `LLM_API_KEY` | **你是谁 + 谁付钱**（身份 + 计费凭证二合一） | 银行卡 + 密码 | 丢了去控制台**吊销重发**（= 换锁） |
| `LLM_BASE_URL` | **打给谁** —— 哪家厂商的哪台服务器 | 银行网点地址 | 换厂商只改它 |
| `LLM_MODEL` | **用哪个脑子**（同一个厂商里有贵/便宜/强/快） | 点哪道菜 | 换模型只改它 |

⭐ **推论（这就是 `.env` 存在的理由）**：key 决定谁付账、base_url 决定找哪家、model 决定用哪个档位 ⇒ **三者都与业务逻辑无关**，所以放配置不放代码。

### 2.2 `base_url` 取代了旧的 `api_base`

- 旧教程写 `openai.api_base = "..."` —— **已过时**
- 当前写法：`OpenAI(api_key=..., base_url=...)`；`base_url` 也认环境变量 `OPENAI_BASE_URL`
- **"OpenAI 兼容"只意味着接口形状像**：第三家厂商照着 OpenAI 做，代码才能只改 `BASE_URL` 就换过去；不照着做的（如 Anthropic 那套 `/v1/messages`、`system` 是顶层参数、`max_tokens` 必填）**光换地址跑不通**
- ⚠️ **同一家厂商可能给多个入口**，比如同时提供「OpenAI 兼容」与「Anthropic 兼容」两条路径 —— **用哪个 SDK 就选对应入口**，跟"新旧"无关

### 2.3 `messages` 是消息数组，不是字符串

```python
messages=[{"role": "user", "content": "..."}]   # ✅
messages="你好"                                  # ❌ 过不了
```

- 一条消息 = `{"role": ..., "content": ...}` 的字典；`role` 常见三种：`system`（行为设定）/ `user` / `assistant`
- ⭐ `system` 不是必需的，但它决定"模型扮演谁" —— S1 讲 Prompt / Context Engineering 的起点

### 2.4 `usage` 是 token 的唯一来源（且藏着一个坑）

| 字段 | 实测值（`deepseek-flash`，同一 prompt 两次） |
|---|---|
| `prompt_tokens` | **38 / 38**（输入是确定的，稳定） |
| `completion_tokens` | **58 / 72** |
| `total_tokens` | **96 / 110** |
| **`completion_tokens_details.reasoning_tokens`** | **38 / 47** |

⭐⭐ **两个必须记住的判据**：

1. **`reasoning_tokens` 算在 completion 里面** —— 上表第一次跑，58 个 completion token 里有 **38 个是"想"掉的（65%）** ⇒ 该模型**默认走思考模式**，"便宜 = 快 = 不用想"这个假设**不成立**；算成本要按 `total_tokens`，不能按"输出多少字"估
2. **同一 prompt 两次 `total` 差 14** ⇒ **输出非确定性** ⇒ 任何"通过率 / 准确率"都必须**跑 N 次取分布**，单次结果不算数（这就是 eval 集存在的理由）

### 2.5 计时的三个函数别混

| 函数 | 读数是什么 | 能当日历 | 能测时长 |
|---|---|---|---|
| `time.time()` | Unix 时间戳（1970 起秒数） | ✅ | ⚠️ 会被校时/NTP 改动，可能算出负数 |
| **`time.perf_counter()`** | **任意起点**起经过的秒数，**单调只增** | ❌ | ✅ **标准做法** |
| `time.process_time()` | 只算 CPU 时间 | ❌ | ❌ 网络等待不算 CPU ⇒ 结果≈0 |

- ⭐ 模型：`perf_counter()` 是**一块从不归零、从不停摆的秒表**，你只是"看表读数"；**测时长 = 看两次、相减**
- ⭐ 判据：**只打印 `perf_counter()` 而不相减，得到的数字什么都不表示**（实测两次读数 `10443.024` → `10443.524`，差 0.5 秒才是时长）
- 计时点选在哪决定数字的含义：只夹 `create(...)` = 网络往返 + 服务端推理；把建 client 也夹进去 = 还含本地初始化（语义就混了）

### 2.6 接入探针：先验地址再写业务

不写业务代码，先让服务器回答"地址 / key / 模型名"三件事对不对（`check_llm.py`：`GET {BASE_URL}/models`）：

| 返回 | 说明 | 下一步 |
|---|---|---|
| **200 + 模型列表** | 域名对、`/v1` 对、key 对 | 再核对 `MODEL` 在不在列表里 |
| **401** | **地址对，key 错**（粘不全 / 带空格 / `.env` 没读到） | 查 `.env`，别改地址 |
| **404** | **地址错**（域名错，或 `/v1` 缺/多） | 换候选地址 |
| **403** | 地址与 key 都对，但**没开通**（实名 / 充值 / 模型未申请） | 去控制台 |
| **429** | 都对，触发限流 | 等一会 |

⚠️ 探针 200 **只证明地址和 key 对**，不证明对话接口开通了 —— **真判据仍是跑一次 `chat.completions` 拿到正文**。

### 2.7 本日数字（后续对比基准）

| 项 | 值 |
|---|---|
| 模型 | `deepseek-flash` |
| 端到端耗时 | **1254.49 ms**（复跑 1303.34 ms） |
| token | **38 / 58 / 96**（复跑 38 / 72 / 110） |

---

## 3. 多轮对话：上下文的真相（2026-10-30 · S0 应用）

### 3.1 模型是**无状态**的 —— 上下文在你的代码里

| 你以为的 | 实际发生的 |
|---|---|
| 模型记得上次说过什么 | **你的代码**在记，每次把全部历史重发 |
| 对话有"会话" | API 没有 session 概念，请求之间毫无关联 |
| 模型在"成长" | 服务端每次都当第一次见你 |

⭐ 一句话：**上下文 = 你手里那个 `messages` 列表本身**，不是它的索引。

### 3.2 多轮的唯一秘密：把整个历史递进去

```python
messages=[{"role": "user", "content": q}]   # 单轮：每次现造一个新列表
messages=messages                            # 多轮：把整个历史原样递过去
```

⚠️ 因此 `messages` 必须**原地追加**（`append`）。一旦写成 `messages = messages + [...]`，历史每轮都是空的 —— **而且不会报错**，模型只会开始瞎编。这正是 Python 笔记 3.2「改对象 vs 重新绑定」的重灾区。

### 3.3 代价是 O(n²)

每轮 prompt ≈ 历史全文。设每轮新增 ≈ 100 token：

| 轮次 | 本次 prompt | n 轮累计 |
|---|---|---|
| 1 | 100 | 100 |
| 10 | 1000 | **5500** |
| 50 | 5000 | **127,500** |

公式：累计 ≈ `50·n·(n+1)`。长对话突然变贵变慢不是模型变笨，是**你每次都在重发整叠记录**。

### 3.4 三个工程手段（S1/S3 展开）

| 手段 | 做什么 | 解决什么 |
|---|---|---|
| **截断** | 只保留最近 N 轮 | 成本线性化 |
| **摘要** | 早期对话压成一段 system 消息 | 保语义、丢细节 |
| **RAG** | 需要时再查外部知识，不全塞 | 上下文窗口装不下 |

### 3.5 响应的三层结构（两个 `None` 都不是臆想的）

```
ChatCompletion
├── choices: list[Choice]           ← ① 数组：n 参数可生成多个候选（默认 1）
│   └── [0].message.content         ← ② str | None：走 tool_calls 时没有正文
└── usage: CompletionUsage | None   ← ③ 流式（stream=True）默认不返回
    ├── prompt_tokens / completion_tokens / total_tokens
```

### 3.6 实测数字（2026-10-30 · `deepseek-flash` · 3 轮同一进程）

| 轮次 | 消息数 | prompt | completion | total | 耗时 |
|---|---|---|---|---|---|
| 1 | 2 | **46** | 32 | 78 | 1185.61 ms |
| 2 | 4 | **62** | 78 | 140 | 821.39 ms |
| 3 | 6 | **96** | 36 | 132 | 749.14 ms |

⭐ 两处可复算（这就是"每次重发"的证据）：

- 第 2 轮 prompt 增量 **16** ≈ 上轮回答 `"2"`（1）+ 本轮问题（15）
- 第 3 轮 prompt 增量 **34** ≈ 上轮"复述整句"（≈20）+ 本轮问题（≈14）

⇒ **增量与上一轮回答的长度成正比**。

⚠️ 首轮耗时最长（1185 ms）含建连 / TLS 握手；**单次耗时不算数**，要比性能必须跑 N 次看分布。

---

## 4. 并发与异步的底层：GIL、线程池与「该并发什么」（2026-11-01 · 计基 30min · 复盘日整理）

### 4.1 一句话根

**并发只在"等"上赚钱**，而"能不能等"由**阻塞发生在哪一层**决定 —— 不是由 `async` 关键字决定。
第 5 章讲的是**语言层怎么写**（让出点）；本章讲**操作系统层为什么这样**，并且回答一个更实用的问题：**哪些活该并发、用什么并发**。

### 4.2 抢占 vs 协作：两种"切"

| | 线程（抢占式） | 协程（协作式） |
|---|---|---|
| 谁决定切 | **OS 调度器**，时间片到就切，代码无法拒绝 | **代码自己** `await` 让出，不让就没人能切 |
| "睡一个"的代价 | `time.sleep` 只睡**这个线程**，别的线程照跑 | `await sleep` 挂起的是**这个任务**；但**同步** `time.sleep` 睡死整个事件循环线程 |
| 切换成本 | 上下文切换 + 每线程独立栈（MB 级） | 单次切换 **≈10 µs**（10-31 实测：8 行交替输出 `0.78→0.83 ms`） |
| 实测对照 | — | 串行 **2.02 s** → `gather` **1.00 s**（10-31） |

⇒ 协程便宜在**切换成本**，强在**单线程扛住上万并发等待**；弱在**一个任务不让出，全体饿死**。

### 4.3 GIL：为什么"多开线程"救不了 CPU 密集

- CPython **默认构建**带 GIL：同一瞬间**只有一个线程在执行 Python 字节码**。多线程只是在 GIL 上**轮流**跑，**不并行**。
- ⇒ `ThreadPoolExecutor(max_workers=4)` 跑 4 个纯 Python 死循环，总耗时 ≈ 串行，**还多付切换开销**（加速比 ≈ 1，不是 4）。
- **唯一例外**：C 扩展在计算时**主动释放 GIL**（NumPy、`zlib`、`hashlib` 等），那一段才真能并行。

### 4.4 `to_thread` 的官方边界（最容易写错的一条）

`asyncio` 文档 · *Coroutines and tasks → Running in threads* 的 Note 原文：

> Due to the **GIL**, `asyncio.to_thread()` can **typically only be used to make IO-bound functions non-blocking**. However, for extension modules that **release the GIL** or alternative Python implementations that don't have one, `asyncio.to_thread()` can also be used for CPU-bound functions.

⇒ **`to_thread` 是"把阻塞挪到别的线程"，不是"把计算变快"**。
10-31 的 `1.00 s` 之所以成立，是因为 `time.sleep` 属于 **IO 型等待**；换成纯 Python 的 CPU 忙等，`to_thread` 只会更慢（还得加线程开销）。
**记住这句话的用途**：它同时解释了 10-31 的"毒药"与"解药"——毒是毒在**占了事件循环唯一的线程**，解是解在**把等待挪走**，与算得快不快无关。

### 4.5 线程池开多大：默认值有公式，且 3.13 改过

| 版本 | `ThreadPoolExecutor` 默认 `max_workers` |
|---|---|
| 3.5 | CPU 数 × 5 |
| 3.8 – 3.12 | `min(32, os.cpu_count() + 4)` |
| **3.13 / 3.14（本环境 3.14.8）** | **`min(32, (os.process_cpu_count() or 1) + 4)`** |

- 3.8 的设计理由：**至少留 5 个 worker 给 IO**，上限 32 给"会释放 GIL 的 CPU 任务"。
- 3.13 把 `os.cpu_count()` 换成 **`os.process_cpu_count()`**：按**本进程可用 CPU** 算 —— 容器 / cgroup 限核时不会"看到宿主机 64 核就开 32 线程"。
- **3.14 未再改该公式**（3.14 在本类的变更是新增 `ctxkwargs` / `prepare_context()` / `InterpreterPoolExecutor`）。
- 文档明确**不推荐**线程池跑长任务：所有入队线程会在解释器退出前被 join，且**早于** `atexit` 处理器。

### 4.6 3.14 的四个真变化（会静默踩，全部来自官方 What's New）

| 变化 | 影响 | 正确写法 |
|---|---|---|
| `asyncio.get_event_loop()` 无当前循环 ⇒ **抛 `RuntimeError`**，不再隐式创建 | 老教程的 `loop = asyncio.get_event_loop()` 直接炸 | 一般用 `asyncio.run()`；分阶段/复用用 `asyncio.Runner`；async 内部用 `get_running_loop()` |
| **policy 系统整体弃用**（`get/set_event_loop_policy`、`WindowsProactorEventLoopPolicy` 等，**计划 3.16 移除**） | 按平台选循环的老写法要改 | `asyncio.run(main(), loop_factory=asyncio.SelectorEventLoop)` |
| **`asyncio.iscoroutinefunction()` 弃用**（3.16 移除） | 用了会告警，将来报错 | `inspect.iscoroutinefunction()` |
| `create_task()` 系列（含 `TaskGroup.create_task()`）接受**任意关键字参数**，`name` / `context` 不再特殊处理 | 自定义任务工厂时名称要自己按工厂的 `name` 参数处理 | 交给工厂 |

### 4.7 free-threading：3.14「正式支持」≠「GIL 没了」

- **PEP 779：free-threaded Python 在 3.14 是 officially supported**（3.13 引入 PEP 703 时还是 experimental）。
- 但它仍是**独立构建**（编译期 `Py_GIL_DISABLED`）；**默认构建照样带 GIL**。
- 代价：free-threaded 构建单线程性能损失 **5–10%**。
- 附带福利：asyncio 对 free-threading 有一等支持 —— **不同线程里的多个事件循环可以并行**，随线程数线性扩展。
- ⇒ 对本环境的结论：**3.14.8 默认构建 = 有 GIL**，4.3 / 4.4 的结论**照旧成立**。**别把"3.14 正式支持 free-threading"读成"GIL 已经没了"** —— 这是今年最容易传错的一条。

### 4.8 这张卡的正事：AI 项目里「该并发什么」

| 动作 | 类型 | 有效手段 | 判据 |
|---|---|---|---|
| **调模型 API**（等网络） | **IO 密集** | `gather` + 异步客户端（`httpx.AsyncClient` / SDK 异步接口） | 3 路并发总耗时 **≈ 最慢那路**，不是三者之和 |
| 并发抓网页 / 读文件 | IO 密集 | 同上 | 同上 |
| 本地算 token / 跑纯 Python 的预处理 | **CPU 密集** | **进程池**（`ProcessPoolExecutor`），不是线程 | 线程池加速比 **≈1** ⇒ 说明被 GIL 卡住 |
| NumPy 大矩阵 / 压缩 / 哈希 | CPU 密集但**释放 GIL** | 线程、`to_thread` **有效** | 加速比 ≈ 核数 |

⭐ **一句话判据：等外部 → 协程；算自己 → 进程（除非库自己放 GIL）。**
⭐ 对 S1/S3 的直接含义：**模型调用、检索、批量 embedding 的"等"全在 IO 层** ⇒ 并发收益大；**本地分词 / 拼 prompt / 解析 JSON** 是 CPU ⇒ 并发无收益。S1 的流式与并发批处理就靠这条判断。

### 4.9 自测卡（遮住上面，口述；答不出就回扫对应小节）

1. `to_thread` 能让 4 个纯 Python 死循环快 4 倍吗？为什么？
2. 为什么同步 `time.sleep` 在 `async def` 里会毁掉并发，而 `await asyncio.sleep` 不会？
3. 3.14 线程池默认开几个线程（写出公式）？为什么把 `cpu_count` 换成 `process_cpu_count`？
4. 「free-threading 在 3.14 正式支持」—— 这对我们的 3.14.8 默认构建有没有影响？
5. 3.14 里判断"是不是协程函数"该用哪个？`asyncio.iscoroutinefunction()` 怎么了？

### 4.10 查证来源（2026-11-01 实查 · 官方文档优先）

| 结论 | 来源 |
|---|---|
| `to_thread` 的 GIL 限制（只能救 IO 密集） | `asyncio` 文档 · Coroutines and tasks → Running in threads（官方 Note 原文） |
| 线程池默认值公式与 3.8 / 3.13 变更 | `concurrent.futures` 文档 · `ThreadPoolExecutor.max_workers` |
| PEP 779 free-threading officially supported / policy 弃用 / `iscoroutinefunction` 弃用 / `create_task` 参数变化 | `What's New in Python 3.14` |

> 交叉方式：**"`to_thread` 受 GIL 限制"** 与 **"线程池默认值只为 IO 设计（≥5 worker）+ 上限 32 留给释放 GIL 的任务"** 是两份独立页面的说法，互证"多线程不是万能的"；`to_thread` 页面本身**没有**写线程池大小，所以 4.5 的数字必须去 `concurrent.futures` 查 —— **一处来源不足就补第二处**。

---

## 5. 语言模型的第一层：从计数到条件概率（char-level bigram · 2026-11-16 · L1 关卡 1）

> **一句话根**：语言模型只做一件事 —— 给「下一个 token」打分。`char-level` 把 token 定成一个**字符**；`bigram` 把「历史」砍到**长度 1**（只看前一个字）。这一层没有神经网络、没有梯度，只有数票和除法。

### 5.1 三个量，按依赖顺序

| 量 | 定义 | 本语料实测 |
|---|---|---|
| `counts[(前, 后)]` | 相邻字符对出现几次（每个起始位置贡献一次，内容相同的合并） | 总对数 **421** = `len - 1`，不同 bigram **350** 种 |
| `row_totals[前]` | 所有以「前」开头的对的次数之和（**按前一个字聚合**） | 最大者 `'，'` = **28** |
| `P(后 \| 前)` | `counts[(前,后)] / row_totals[前]` | `P(包\|面) = 1.0` |

条件概率的**分母是行和**，不是语料总对数。这是全部差别的来源：

- `P(包|面) = 1.0`（面出现 9 次，9 次后面全是「包」）
- 全局 `P(包) = 9/422 = 0.0213`
- 同一个事件，两种分母差 **47 倍**

用总对数当分母的后果是**静默的**：每个 context 的概率和只剩 `1/90` 左右，代码不报错，模型只是变笨。

### 5.2 为什么 char-level：V=180 的取舍

| | char-level | 词级（word-level） |
|---|---|---|
| 词表 V | **180**（本语料） | 几万，还要分词器 |
| 未见组合 | 少见得多 | 极多（词形变化、新词） |
| 代价 | 表达同样信息需要**长得多**的上下文 | 一个 token 就带很多信息 |

字符级最反直觉的一点：**标点也是 token**。模型不知道 `。` 表示句子结束，它只是 180 个符号里的一个，只能从统计里发现「`。` 后面通常跟一句开头」（本语料 14 个句号 → 13 个跨句 bigram，后继是 8 种不同字符）。

### 5.3 分母的两种错法

| 错法 | 后果 | 哪条判据抓得住 |
|---|---|---|
| 除以**总对数**（421） | 每个 context 概率和 ≈ 1/90 | 归一判据（`Σ P == 1.0`） |
| 按 **next** 聚合（行列颠倒） | 概率和仍是 1，但分箱全错 | **只能**靠「逐字符行和自证」`row_totals[c] == text[:-1].count(c)` |

⭐ 第二种错在「总对数守恒」这条判据上**完全隐形**（总数没变），所以判据至少要有一条拿**独立来源**对照（直接数 `text[:-1]`）。

### 5.4 「条件 ≠ 边际」怎么证明

条件分布的键只有 1~9 个，全局分布在 180 个字符上都有值，**键集合不同 ⇒ 相减必须 `probs.get(c, 0.0)`**（实测直接索引抛 `KeyError: '。'`）。

阈值不能拍脑袋：取「出现最多的两个字」时最大差只有 **0.1087**（勉强压线，换语料就假红）。改选「**后继种类最少**」的 context：`'面'` 9 票全给 `'包'` ⇒ 差 **0.9787**，这个选法与语料规模无关。

### 5.5 边界：未见 context 返回空 dict

`next_probs(counts, 没出现过的字)` 必须返回 `{}`，不许抛 `KeyError`：

- 空分布的和是 `0.0`，**不是 1** —— 调用方必须自己面对这件事；
- 一旦抛异常，真实代码里就是「随机某天崩掉」；实测删掉守卫立刻 `KeyError: 'a'`。

这就是**平滑（add-k）**的动机：421 对里有 350 种，留出集上必然出现没见过的 bigram，未平滑的模型会给出概率 0 ⇒ 交叉熵变 `inf`。

### 5.6 与后续关卡的接口

| 关卡 | 接口 |
|---|---|
| L1 关卡 2（交叉熵） | `CE = -(1/N) Σ ln P(w_i \| context_i)`（nats；换 `log2` 得 bits/char）；困惑度 `PP = 2^H` |
| L1 关卡 3（softmax） | `logits → 减 max → exp → 除以和`；`CrossEntropyLoss` 的输入必须是**未归一化 logits**，等价于 `LogSoftmax` + `NLLLoss` |
| L1 关卡 4（平滑） | 未见 bigram 让 CE 变 `inf`，add-k 后变有限 |

### 5.7 官方依据（2026-11-16 查证）

- `docs.python.org/3/library/math.html`：`log(0)` 抛 **`ValueError`**（不会返回 `-inf`）；`exp` 溢出抛 **`OverflowError`**；`log2(x)` 比 `log(x, 2)` 更准
- `docs.pytorch.org/docs/2.15/.../CrossEntropyLoss.html`：input 必须是**未归一化 logits**；等价 `LogSoftmax` + `NLLLoss`；`reduction` 默认 `'mean'`；`ignore_index` 默认 `-100`
- 版本陷阱：`math` 文档站当前展示 **3.15.2**，本机 **3.14.8** ⇒ `fmax` / `fmin` / `signbit` / `isnormal` 用不了

---

## 6. 交叉熵与困惑度：把概率换算成一个可比的数（2026-11-17 · L1 关卡 2）

> 一句话：**交叉熵 = 模型给正确答案分配的概率的平均惊讶度**。`-(1/N) Σ log2 P`，单位 bits/char；
> 困惑度 `ppl = 2^CE`，含义是「每一步平均在几个候选里犹豫」。天预报类比：报 90% 下雨、真下了 ⇒
> `-log2(0.9) ≈ 0.15 bit`；只报 1% ⇒ `-log2(0.01) ≈ 6.64 bit`。

### 6.1 三个数必须严格递减（训练集）

| 模型 | CE（bits/char） | 困惑度 | 含义 |
|---|---|---|---|
| 均匀基线 | **7.3038** = `log2(158)` | 158.00 | 完全不认识字，在 V 个字里瞎猜 |
| unigram | **6.7141** | 104.99 | 只知道「这个字本身多常见」 |
| bigram | **1.2943** | **2.45** | 还知道前一个字 |

- 这条序**数学上必然成立**（均匀分布是最不惊讶的上界；bigram 是 unigram 的细分）⇒ 反了就是实现错。
- ⚠️ **V 必须取训练集的词表大小**（158），不是全语料（180）：22 个字只出现在留出集，训练时根本没见过，拿它们进基线等于作弊。
- ⚠️ 关卡 4 会把 V 变成 **159**（+`<UNK>`）—— 那时它进的是**平滑分母**，不是这里的基线。两处口径不同、不可互换，见 §8.4。
- 两个数在**训练集上**是同一个量：`ce_unigram(train, train)` = 分布自身的熵 `-Σ p log2 p`。

### 6.2 两路互证：同一个量的两种算法

这是 L1 关卡 2 最有用的一招 —— **不引入阈值，只用恒等式**：

| 量 | 逐位置路 | 加权路 |
|---|---|---|
| unigram | `-(1/N) Σ_c log2 P(c)`，c 遍历每个**位置** | `-Σ_char p(c) log2 p(c)` |
| bigram | `ce_bigram(counts, train)` | `-Σ_pair (k/M) log2(k / row_totals[a])`，M = `len-1` |

两条路差 **< 1e-12**。为什么必须补 bigram 那条：**分母写成 `len(eval_text)`（差 1）时，「严格序」和「留出集升高」两条判据全都绿** —— 只差 0.4%，方向不变。补了互证之后立刻红。

### 6.3 留出集：`inf` 与 `nan` 是两个不同的结论

| 返回值 | 含义 | 什么时候出现 |
|---|---|---|
| `inf` | **模型说这不可能**（有信息量） | 评估文本里有训练没见过的字 / 字对；`log2(0)` 抛 `ValueError` ⇒ 必须在取 log **之前**判 0 |
| `nan` | **压根没测到**（评测失效） | `skip_unseen=True` 且一个位置都没算到；`len(eval_text) < 2`（没有相邻对） |

本语料实测：留出集 85 字里 **22 个新字**、**67/84 = 79.8%** 的相邻对训练时没见过 ⇒ 不加平滑的模型困惑度**直接报废**（这就是 L1 关卡 4 的全部动机）。

### 6.4 跳过未见位置会让分数「变好看」

`skip_unseen=True` 实测：unigram 留出集 **6.5798 < 训练集 6.7141** —— 不是变强，是把最难的题划掉了。

⇒ 纪律：**默认口径不许跳过**（要 `inf` 就老实地出 `inf`），skip 只能作为「有限可比的辅助数字」，并且必须同时报「漏了多少」（`67/84`）。另外**分母必须跟着口径走**：跳过 k 个位置后分母是「真正算到的位置数」，写成 `len-1` 会让留出集 bigram CE 从 **1.6935** 掉到 **0.3427**。

### 6.5 平滑 ≠ clip

- **clip**（sklearn `log_loss` 的做法：概率裁到 `[eps, 1-eps]`）：只是让 `log(0)` 不炸，**代价是编造了一个概率**，数字好看但病因还在。
- **add-k 平滑**（`(C+k)/(row+kV)`）：承认「没见过的也可能出现」，给未见事件分配真实的质量，CE 变成**有限且可比**。
- 二者在数值上都能让 CE 从 `inf` 变成有限数 —— 区别在**是否改变了模型的信念**。L1 关卡 4 要的是后者。

### 6.6 与后续关卡的接口

| 关卡 | 接口 |
|---|---|
| L1 关卡 3（softmax） | 交叉熵的输入是 logits，`softmax` 负责把 logits 变成概率；数值稳定 = **先减 max** |
| L1 关卡 4（平滑） | 今天算出的 `inf` 与 79.8% 未见对，就是平滑要解决的问题 |
| L2（注意力） | 同样的 `log2` 平均惊讶度会以「训练 loss」的身份回来，只是换成 nats |

### 6.7 官方依据（2026-11-17 查证）

- `docs.pytorch.org/docs/2.15/.../CrossEntropyLoss.html`：input 是**未归一化 logits**（不必为正、不必和 1）；等价 `LogSoftmax` + `NLLLoss`；`reduction` 默认 `'mean'`
- `scikit-learn.org/stable/.../sklearn.metrics.log_loss.html`：log loss **就是**交叉熵；用**自然对数**（nats）；把概率 clip 到 `[eps, 1-eps]` 以避免 `log(0)` ⇒ **clip 是掩盖、add-k 才是治**
- `docs.python.org/3/library/itertools.html#itertools.pairwise`：3.10+；产出 n−1 个相邻对，**返回迭代器**（要 `list()` 才能 `len` / 多次遍历）

---

## 7. softmax 与数值稳定：数学上等价，浮点上生死（2026-11-18 · L1 关卡 3）

> 一句话：**softmax 把「任意实数分数」变成「正数且和为 1」的概率分布**，`exp(z_i)/Σexp(z_j)`；
> 而 `exp` 只有约 `[−745, 709]` 的可用区间，所以真实实现一律**先减 max** ——
> 不是优化技巧，是「不这么做就会抛异常或塌成 0」。
> 生活模型：食堂打饭。logits = 印象分（可负、可很大），softmax = 换算成「走进哪个窗口的概率」，
> 分差越大越集中；减 max = 把印象最好的窗口定为 0 基准，**只保留分差**。

### 7.1 定义与三条性质

`softmax(z)_i = exp(z_i) / Σ_j exp(z_j)`

| 性质 | 意义 |
|---|---|
| **恒正** | `exp > 0` ⇒ 概率 ≥ 0（分布的定义） |
| **严格单调** | `a > b ⟺ exp(a) > exp(b)` ⇒ 保序：分高的概率一定高 |
| **平移不变** | `softmax(z + c) ≡ softmax(z)`，因为分子分母同乘 `exp(c)` 约掉 |

代价：**任意一个位置的概率都依赖全体**（分母是全局和），所以位置间不能独立算 —— 这正是后面 attention 里要按行归一化的原因。

### 7.2 为什么减 max 是**唯一**的稳妥选择

取 `m = max(z)`、`u_i = z_i − m`：

1. `u_i ≤ 0` **恒成立** ⇒ `exp(u_i) ∈ (0, 1]` ⇒ 向上溢出**结构上不可能**；
2. `exp(0) = 1` 必在分母里 ⇒ 分母 ≥ 1 ⇒ **不会 0/0**；
3. `u_i` 恰好是「与最强者之差」：`P(i)/P(winner) = exp(u_i)` —— softmax 只在乎**差值**，绝对高度无用。

为什么不能是别的「合法平移」：

| 减什么 | 反例 | 后果 |
|---|---|---|
| `mean` | `[1e6, 0]`（mean = 5e5） | 位移后仍剩 5e5 ⇒ `exp(5e5)` **溢出** |
| `logits[0]` | `[0, 1000]`（首项是最小值） | 位移后 `exp(1000)` **溢出** |
| `max + 100` | 任何输入 | 不会溢出，但全体一起缩小 ⇒ 小项提前下溢，**白丢精度** |

⇒ 「`m ≥ 所有元素`」是硬约束，满足它的**最小值**就是 max。

### 7.3 `exp` 的可用区间（本机 3.14.8 实测）

| 表达式 | 结果 |
|---|---|
| `exp(709)` | `8.218407461554972e+307`（double 最大 1.8e308） |
| `exp(710)` | **`OverflowError: math range error`** |
| `exp(-745)` | `5e-324`（最小非零 double，勉强） |
| `exp(-746)` | **`0.0`（静默下溢，不抛异常）** |

两种失效模式**完全不同**：向上是异常（能 catch），向下是**静默的数据丢失**（catch 不到，只能靠判据发现）。

### 7.4 下溢不是「减 max 的副产品」，也救不回来

| 输入 | naive | 稳定版 | 结论 |
|---|---|---|---|
| `[1000, 0]` | `OverflowError` | `[1.0, 0.0]` | 差 1000 ⇒ 第二项 `e⁻¹⁰⁰⁰` 存不下 |
| `[0, -800]` | `[1.0, 0.0]` | `[1.0, 0.0]` | **naive 没溢出也照样丢** ⇒ 下溢与减 max 无关 |
| `[-1000, -1001]` | `ZeroDivisionError` | `[0.73106, 0.26894]` | 差 1 ⇒ 精确（`1/(1+e⁻¹)`） |

要救下溢只能**换坐标**：`log_softmax = z_i − m − log Σ exp(z_j − m)`，原值从不进 `exp`，`-1000` 原样保留。
这也是 PyTorch 提供 `log_softmax` 而不是让人写 `log(softmax(x))` 的原因。

### 7.5 同序 vs 保序（判据设计的两个不同靶子）

| 词 | 含义 | 来源 | 怎么验 |
|---|---|---|---|
| **同序** | 输出第 `i` 项属于输入第 `i` 项 | 逐元素映射（`[f(x) for x in z]`）+ 除同一标量；`list` 带位置 | **升序输入**的精确值（`[log1, log7]` → `0.125` 而非 `0.875`） |
| **保序** | `z_i < z_j ⟹ p_i < p_j` | `exp` 单调 + 正的标量缩放 | 链式比较 `p[0] < p[1] < p[2]` |

两者不等价：排序实现**保序仍成立**，但同序被破坏 ⇒ 只有精确值 + 升序输入能抓。

### 7.6 判据（可复测 · 汇总）

| 判据 | 期望 | 实测 |
|---|---|---|
| 平移不变 | 加 1 / −5 / 100 后最大差 < 1e-12 | **0.00e+00**（逐位相同） |
| 精确值 | `[log3, log1]` → 0.75；`[log1, log7]` → 0.125 | 差 < 1e-15 |
| 归一 / 正性 / 保序 | 和 = 1、值 > 0、`p` 递增 | 全绿 |
| 边界 | `[]` → `[]`；`[7.5]` → `[1.0]` | 全绿 |
| 互证 | 与公式直译版在 `\|z\| ≤ 10` 上一致 | **6.78e-21** |
| 溢出 | naive + `[1000, 0]` → `OverflowError` | 红=抓到 |
| 下溢 | naive + `[-1000, -1001]` → `ZeroDivisionError` | 红=抓到 |
| 病态形状 | `[0, 1000]` → `[0.0, 1.0]`；`[1e6, 0]` → `[1.0, 0.0]` | 抓「减 mean / 减首项」的唯一用例 |

⚠️ **教训**：前 6 条全绿时，M2/M3（减 mean / 减首项）**照样绿** —— 因为它们也是合法平移，
只是**不保证不溢出**，而原有极端输入的形状（首项恰是 max、位移后仍在安全区）恰好避开了这一点。
**判据强度取决于输入形状，而不是断言条数。**

### 7.7 官方依据（2026-11-18 查证）

- `pytorch/torch/nn/functional.py` — `log_softmax` 文档原文：*"While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically unstable."*
- `pytorch/aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h` — CPU kernel 第一步即 `sum_j exp(data[j] - max_input)`，注释强调**运算顺序不可换**（大数 + 小差会出数值问题）
- `pytorch/torch/_refs/__init__.py` — `aten.logsumexp` 分解：减 max → exp → 求和 → 取 log → 把 max 加回
- `scipy/special/_logsumexp.py` — `scipy.special.softmax` 源码：`x_max = max(x)` → `exp(x - x_max)` → 除以和；`logsumexp` 用 `log1p(s) + a_max`
- `docs.python.org/3/library/math.html` + 本机实测 — `exp` 上溢抛 `OverflowError`、下溢返回 `0.0`

---

## 8. 平滑：把「概率 0」变成「概率小但 > 0」（2026-11-19 · L1 关卡 4）

> 一句话：MLE 对没见过的对说「概率 0」，而 0 的负 log 是无穷 ⇒ 整篇 CE 报废。
> **add-k 就一件事：给每个候选垫 k，再从已见事件按比例扣回来**（分母 `C + k|V|`）。
> 生活模型：分蛋糕。给在场每个人先各垫一小块，垫掉的量按同一比例从大家份额里抽回来 —— 总量仍是一个蛋糕；
> k 越大垫得越多，add-1 等于给全城每人留一块，蛋糕立刻被摊薄（V=159 时 `'清'` 行被移走 **99.375%**）。

### 8.1 两种写法是同一件事（插值）

$$\frac{c+k}{C+k|V|} \;=\; \underbrace{\frac{C}{C+k|V|}}_{\text{信自己的经验}}\cdot\underbrace{\frac{c}{C}}_{P_{\text{MLE}}} \;+\; \underbrace{\frac{k|V|}{C+k|V|}}_{\text{承认自己无知}}\cdot\frac{1}{|V|}$$

- 右边是**凸组合**：权重和为 1 ⇒ 概率和必然还是 1（归一不用另证）；
- 第二个权重就是「移走的质量」，随 **|V| 变大**、随 **C 变小**而变大；
- `C = 0`（含 `<UNK>` 当上下文）⇒ 权重 100% 落在均匀那侧 ⇒ 整行 `1/|V|`。这是 add-k 顺手治好的第二病。

### 8.2 分母为什么必须加 `k|V|`

分子给 **V 里每一个**候选都加了 k，总加量是 `k|V|`；分母不加回去，行和就变成 `C/(C+k)` ≠ 1（实测漏写后行和 **7.8696**）。
⚠️ 只给「见过的列」加 k（分母写 `C + k·已见列数`）也是错的：行和仍能凑到 1，但 V 一变数字就飘，跨模型不可比。

### 8.3 这个 `inf` 其实是两种病（84 个位置的实测分解）

| 病 | 位置数 | 病因 | 谁能治 |
|---|---|---|---|
| 行存在、缺列 | **26** | `c(prev,next) = 0` | add-k 的 `k/(C+k|V|)` |
| 位置里有生字 | **41**（next 侧 24、prev 侧 24，有重叠） | 字符**不在候选集**里，连「垫过的 0」都不存在 | 必须**改词表口径**（闭词表 + `<UNK>`） |

⇒ 只加平滑、不动词表，held 的 CE **仍然是 `inf`**（实测：V=158 时 24 个位置概率为 0）。

### 8.4 闭词表与 `<UNK>`：V 从 158 变 159

V 在这条线上有三种口径，混用就是静默错：

| V | 是谁 | 用在 | 用错的后果 |
|---|---|---|---|
| **180** | 全语料字表 | 关卡 1 计数 / 关卡 2 的直觉对照 | 用在平滑分母 ⇒ **偷看留出集字表**，数字偏小、不可比 |
| **158** | 训练集字表 | 关卡 2 的均匀基线 `log2(158)=7.3038` | 用在平滑 ⇒ 生字位置概率仍是 0 ⇒ **CE 还是 inf** |
| **159** | 158 + `<UNK>` | 关卡 4 唯一正确口径 | 用在基线 ⇒ `log2(159)=7.3129`（差 0.0091 bit，但口径要标） |

折叠的落地细节（都是实测教训）：

- **`<UNK>` 是一个 token、五个字符** ⇒ 折叠必须返回 **token 列表**，不能 `"".join(...)`：85 个 token 拼成 **181** 个字符，位置数 84→180，分母一换全盘作废；
- 折叠四条判据：`len` 不变（85）· 结果全 ∈ 闭词表 · 认识的字一个不动 · 幂等；
- 折叠后不同 token **61 → 40**（22 种生字合并成 1 个 UNK；生字**出现 24 次**）；
- `<UNK>` 在训练集里计数为 0 ⇒ 它的概率**全部来自垫的 k**，不是数据 —— 这就是「用一个符号吸收所有未知」的粗糙处。

### 8.5 k 的选择：不是越小越好（本机实测）

| k | train CE（ppl） | held CE（ppl） |
|---|---|---|
| MLE | **1.2943**（2.45） | **`inf`** |
| 1.0（Laplace） | 6.1048（68.82） | 7.0590（133.34） |
| **0.1** | 3.8545（14.47） | **6.7812（109.99）** ← 最优 |
| 0.001 | 1.3944（2.63） | 8.4070（339.44） ← 最差 |

- 训练集单调：k 越小越接近 MLE，但**永远 ≥**（分母多了 `k|V|`）；
- 留出集不是单调：k=0.001 时 24 个 UNK 位置的概率被压到 \(10^{-5}\) 量级 ⇒ 每位置约 15 bit 的惩罚；
- ⇒ **k 用 dev 的 ppl 挑**（来源给的经验区间 0.001–0.5），不是「越小越接近 MLE 就越好」。

### 8.6 判据（可复测 · 汇总）与两次教训

| 判据 | 期望 | 实测 |
|---|---|---|
| 归一 / 键集合 | 三行 × 三个 k：键集 == 闭词表、行和 = 1（±1e-12）、全正 | 全绿（|V| = 159） |
| 恒等式 | 未见列 == `k/(C+k|V|)`，见过列 == `(c+k)/(C+k|V|)` | 逐位一致 |
| `C=0` 兜底 | `<UNK>` 行整行 `1/159`（0.006289） | 全等 |
| k 单调 | 见过的列上 `max|P_k − c/C|` 随 k 递减 | 成立 |
| 非法 k | k ≤ 0 必须抛 | `ValueError` |
| 折叠 | token 数不变、全 ∈ 词表、幂等 | 85 / 40 / 通过 |
| 有限性 | held 上三个 k 都 `isfinite` | 7.059 / 6.781 / 8.407 |
| 代价 | train：k=1 > k=0.1 > MLE | 6.10 > 3.85 > 1.29 |
| 两路互证 | 逐位置 vs 按对加权，差 < 1e-12 | 通过 |

⚠️ **教训 1（漏网变异 M6）**：把「跳过算不出的位置、分母仍用 84」注进去，上面**前 8 条全绿** ——
它只让 held 数字悄悄变低（1.2038 vs 真值 7.0590），行和判据管的是概率表、管不到「平均口径」。
补的**两路互证**（逐位置 vs 按对加权）一上就红（差 5.855）。⇒ 变异 7/7 全红，其中这条是补出来的。
⚠️ **教训 2（我自己写错的手算）**：`build_counts("adbc")` 的相邻对是 `(a,d)(d,b)(b,c)` ——
`P(d|a)=0.4`、`P(b|a)=0.2`、`P(*|c)=0.25`（c 才没当过上下文）。照「a 后面是 b」想当然写断言，必红。

### 8.7 官方依据（2026-11-19 查证 · 三条来源）

- **SLP3 Ch.3 *N-gram Language Models*** — add-k 公式 `(c+k)/(C+k|V|)`；k=1 即 add-1；add-1 对 n-gram **过度平滑**，实用取小数 k。⚠️ 该章 PDF 抓取后是 FlateDecode 二进制流、正文未解码，公式按通行版本 + 下面两条可解析来源交叉确认，**页码需本地 PDF 核对**。
- **mbrenndoerfer.com《Language AI Handbook》§9.4 Add-k Smoothing**（2025-03-22 更新）— 分母加 `k|V|` 的唯一目的就是 Σ=1；移走质量 `k(|V|−V_seen)/(C+k|V|)` 随 |V| 增大、随 C 减小；k ∈ (0,1]、典型 0.001–0.5、用 dev 的 ppl 调；批评原话：「same k for all n-grams, regardless of how often they appear」「treats the lower-order distribution as uniform」「even with optimally tuned k, add-k remains a relatively poor performer」⇒ 更强的是 Kneser-Ney（按计数用不同折扣）。
- **深度学习课程笔记（NLP with Probabilistic Models W3）** — 闭词表三步：建 V → 语料里不在 V 的替换成 `<UNK>` → `<UNK>` 像普通词一样计数；**「Perplexity — only compare LMs with the same V」**；「use `<UNK>` sparingly」。
- **本机实测（Python 3.14.8）** — `math.log2(0)` 抛 `ValueError`（不会返回 `-inf`）⇒ `inf` 必须**自己判、`return math.inf`**；`bit → nat` 是 `×ln2`（**÷1.4427**，不是 ÷2.303）；`ppl = b^{CE_b}` 与底数**配对抵消**（1.2943 bit 与 0.8971 nat 的 ppl 都是 2.45）。

---

## 9. GPU 环境与轮子匹配：cp314 / cu128 / sm_120（2026-11-19 · L2 关卡 0）

> 一句话：GPU 版 torch 是**按「平台 + Python 版本 + CUDA 版本」预编译好的二进制**。
> 装错不会「慢一点」，而是第一次 GPU 运算就报
> `no kernel image is available for execution on the device` —— **看着像代码错，其实是轮子错**。
> 生活模型：买轮胎。轮毂尺寸（Python ABI）、螺栓孔距（CUDA 版本）、车架型号（算力 sm_XX）三者都得对上，
> 差一个不是「开得慢」，是根本装不上。

### 9.1 一个轮子文件名的三段话

```
torch-2.11.0+cu128-cp314-cp314-win_amd64.whl
      └─版本──┘└CUDA┘└Python ABI┘       └平台┘
```

| 标签 | 本机要的值 | 不对会怎样 |
|---|---|---|
| 平台 | `win_amd64` | 装不上（直接报错） |
| Python ABI | **`cp314`**（Python 3.14.8） | 找不到可用轮子（解析阶段就失败） |
| CUDA 渠道 | **`cu128`** | 装得上但**跑不了**：`no kernel image is available` |

### 9.2 本机的三个硬约束（叠起来才是「torch ≥ 2.9」）

| 约束 | 来自 | 结论 |
|---|---|---|
| Blackwell **sm_120** | RTX 5060 Laptop（capability (12,0)） | 轮子里必须有 sm_120 内核 ⇒ **cu128 及以后**、torch **≥ 2.7** |
| **cp314** | 系统 Python 3.14.8 | cu128 索引里 cp314 **从 2.9.0 起**才有（2.7.x / 2.8.0 没有） |
| 渠道最新 | `download.pytorch.org/whl/cu128/torch/` | 最高 **2.11.0** ⇒ 实际装 `2.11.0+cu128` |

⇒ 「torch ≥ 2.7」是**架构侧**的门槛，「torch ≥ 2.9」是**Python 版本侧**的门槛，**两者取严**。

### 9.3 用 uv 钉住渠道（不钉就会拿到默认渠道的轮子）

```toml
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = { index = "pytorch-cu128" }
```

- `explicit = true` = 这个索引**只服务于**下面显式点名的包，不会污染其它依赖的解析；
- 不写 `[tool.uv.sources]` ⇒ torch 仍从默认索引拿（不是 cu128 那份）；
- 装完必须打印 `torch.__version__` 与 `torch.version.cuda` 自证：实测 `2.11.0+cu128` / `12.8`。

### 9.4 首次 ≠ 稳态：探针不预热就会误判显卡

| 项目 | 首次 | 稳态（预热 3 次、测 10 次均值） | 倍数 |
|---|---|---|---|
| matmul 1024³ | **388.8 ms** | **0.65 ms**（3.29 TFLOPS） | **≈600×** |
| attention(B=8,T=1024,d=64) 前向 | 148.7 ms（首轮曾 **5443.7 ms**） | **1.50 ms** | ≈100× |

首轮要付 **CUDA context 初始化 + cuBLAS/cuDNN 惰性加载**的账；把首次当性能，会把一块 8 GiB 的 5060 说成玩具。
⇒ **报耗时必须写清「首次 / 稳态」，并说明预热次数与重复次数。**

### 9.5 判据（可复测 · 汇总）

| 判据 | 期望 | 实测 |
|---|---|---|
| CUDA 可用 | `torch.cuda.is_available()` | `True` |
| 算力匹配 | `get_device_capability() == (12, 0)` | `(12, 0)` |
| 版本自证 | 报 `torch.__version__` / `torch.version.cuda` | `2.11.0+cu128` / `12.8` |
| 冒烟 | 1024³ matmul + attention 前向跑通，报耗时/显存 | 稳态 0.65 ms / 1.50 ms · 峰值 **126 MiB** |
| 数值正确 | softmax 行和 = 1 | 偏差 **2.38e-07** |

### 9.6 官方依据（2026-11-19 查证）

- `download.pytorch.org/whl/cu128/torch/` — 索引里 cu128 覆盖 2.7.0–2.11.0；**cp314 Windows 轮子从 2.9.0 起**（2.7.x / 2.8.0 无）。
- PyTorch 2.7 发布说明 — 2.7 起支持 **Blackwell**，并提供 **CUDA 12.8** 预编译轮子（Linux x86/arm64；Windows 同样有 cu128 轮子）。
- Clore.ai 兼容表 + 社区复现 gist — Blackwell / sm_120 需要 **CUDA 12.8+ 与 torch ≥ 2.7**；不匹配的症状就是 `no kernel image is available`。
- 本机实测（Python 3.14.8 / uv 0.12.23 / 驱动 621.14）— `torch 2.11.0+cu128`、`capability (12, 0)`、首次 vs 稳态差 ≈600×；`torch` 不自动带 `numpy`（导入告警 `Failed to initialize NumPy`）。

---

## 10. 流式：把「等生成完」换成「边生成边交」（2026-11-20 · L4 关卡 1）

### 10.1 流式改的是「交付方式」，不是「生成速度」

模型是**一个 token 一个 token 接龙**的。非流式 = 服务端接完整段再打包发回；流式 = 每接出几个 token 就往前推一小包。

| 项 | 流式影响 | 原因 |
|---|---|---|
| **首字延迟（TTFT）** | 大幅下降 ← **唯一的核心收益** | 收到第一包就能开始显示 |
| 总时长 | **基本不变**（实测差 3%） | 生成工作量没变，逐帧解析还多一点开销 |
| token / 费用 | **完全不变** | 计费看 token 数，不看分几包发 |
| 可中断性 | 从「不能」变成「能」 | 看到跑偏可以立刻停，省掉后续输出 token |

⚠️ 反直觉：中断**省不多**。prompt 的钱在请求那一刻就花了，中断只能砍掉「还没生成的输出」。

### 10.2 传输层：SSE，一条长连接 + 一行行 `data:`

```
data: {"choices":[{"index":0,"delta":{"role":"assistant"}}]}      ← 开场帧：只有角色，没正文
data: {"choices":[{"index":0,"delta":{"content":"二"}}]}          ← 正文增量
data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]} ← 终止帧：只有原因，没正文
data: {"choices":[],"usage":{"prompt_tokens":31,...}}             ← 账单帧：choices 是空列表！
data: [DONE]
```

SDK 把这些吞成 `Stream[ChatCompletionChunk]`（可迭代对象），`create(stream=True)` 就拿到它。

### 10.3 `delta` 是增量，`usage` 是总计 —— 两种东西在同一个流里

| | 正文 | 账单 |
|---|---|---|
| 字段 | `choices[0].delta.content` | 顶层 `chunk.usage` |
| 语义 | **增量**（这一口新增） | **总计**（整次请求） |
| 频率 | 几乎每帧 | 只有最后一帧，其它帧是 `null` |
| 谁负责拼 | **客户端**（`"".join(parts)`） | 不用拼 |

为什么不每帧发全量：输出 300 字切 100 帧，全量方案要传 ≈15000 字，增量方案 ≈300 字 —— 多出来的全是重复。工具调用参数（`delta.tool_calls[i].function.arguments`）同样是碎片，**必须拼完再 `json.loads`**。

`None` 的语义是「这一帧没说这个字段」，不是「内容为空」：开场帧只有 `role`，终止帧只有 `finish_reason`，思考帧只有 `reasoning_content`。

### 10.4 循环为什么「不会提前结束」：阻塞藏在 `__next__` 里

```python
for chunk in stream:   # = while True: chunk = next(stream)
```

`for` 每转一圈都要问迭代器要东西；`Stream.__next__` → `_iter_events()` → `self.response.iter_bytes()` —— **真正阻塞在 socket 读**，没数据就挂起线程（GIL 释放，CPU 占用 0）。

⇒ 消费端代码看不出快慢差别：`iter([1,2,3])` 的 `__next__` 微秒返回，`iter(stream)` 的 `__next__` 可能挂几秒。**判别标准不是「容器 vs 生成器」，而是「`__next__` 里有没有等外部世界」**（`for line in open(...)` 早就在等了，只是磁盘太快没感觉）。

### 10.5 三种「循环结束」，只有一种算成功

| # | 触发 | 表现 | 算成功吗 |
|---|---|---|---|
| A | 收到 `data: [DONE]` | SDK 主动 `break` → `StopIteration` | ✅ |
| B | 对端**关连接但不发 `[DONE]`** | 迭代器自然耗尽 → **平静退出、无异常** | ❌ 半截，但长得跟成功一样 |
| C | 读超时 / 连接错误 | `APITimeoutError` / `APIConnectionError`（原异常在 `__cause__`） | ❌ |
| D | 自己 `break` | 正常退出（SDK 的 `finally` 会 `response.close()`） | ❌ 半截 |

⇒ **「循环自然结束 ≠ 成功」**。唯一判据 = 收到过非 `None` 的 `finish_reason`。另：**流消费不自动重试**（普通请求默认重试 2 次），因为重放会把已交付的输出再吐一遍。

### 10.6 `print` 的三层缓冲（「明明在流式，屏幕却不刷新」的元凶）

| 层 | 会不会加换行 |
|---|---|
| 字符串变量 | 不会（正文里的 `\n` 属于内容） |
| `print()` | 会，默认 `end="\n"` |
| stdout 缓冲 | 终端 = **行缓冲**（遇 `\n` 吐）；管道 / 文件 = **块缓冲**（本机 **128 KiB**） |

关键：`end=""` 去掉了 `\n`，**行缓冲就失去触发器** ⇒ 边打边输必须 `flush=True`。另：Windows 把 `\n` 写成 `\r\n` 只发生在落盘 / 重定向，内存字符串不受影响 ⇒ **比对要在内存里做，别看屏幕、别复制终端**。

### 10.7 思考模式：采样参数失效与 `reasoning_content`

| 事实 | 影响 |
|---|---|
| 思考模式**默认开启**（effort 默认 `high`） | 不写参数就是思考模式 |
| 思考模式**不支持** `temperature` / `presence_penalty` / `frequency_penalty` | **收下不报错、也不生效** ⇒ 「我设了 temp=0」不等于「输出确定」 |
| `top_p` 因模型而异 | 该模型思考模式 0.95–1.0（低于 0.95 按 0.95），**非思考模式恒 1.0** |
| `thinking` 非 SDK 原生字段 | 走 `extra_body={"thinking": {"type": "disabled"}}`；`reasoning_effort` 走普通参数 |
| 思考内容在 `delta.reasoning_content` | 与 `content` **同级**、**不同时非空**；非流式在 `message.reasoning_content` |
| 带 `tools` 的会话 | **必须回传**历史 `reasoning_content`，漏了直接 **400** |
| 隐藏成本 | 同一 prompt：思考模式输入 **45** token → 关思考 **19** token ⇒ **≈26 个看不见的输入 token** |

⇒ 首 token 的口径必须写死：**正文**首个非空 `delta.content` 的到达时刻（不含 role / 终止 / 账单 / 纯思考帧），否则跨模型不可比。

### 10.8 判据：两个数怎么取，以及「不可判定」怎么写

- **首 token / 总时长取中位数**：5 次实测中位 **567.15 / 2219.08 ms**，而平均值 **648.54 / 2236.72 ms** —— 第 3 次 1059 ms 的异常值把平均值抬了 **14.3%**。中位数抗极端值，这就是判据写「中位数」的理由。
- **「流式文本 == 非流式文本」同时测了两件事**：拼接无损 + 两次生成相同。后者在思考模型下不可控（温度不生效）⇒ 判据变成**不可判定**，不许改判据凑绿。
- 分开两件事的办法：**低熵任务**（「从 1 数到 30，每行一个数字」）—— 两次生成几乎必然一致，此时 `差 = 0` 才真正证明拼接无损（实测 `True ｜ -1 ｜ 0`，token 22/60 两边一致）。
- 判「不是拼接 bug」的三条证据：分歧形态（措辞不同 vs 丢字/重复）、长度差、两边 token 数。拼接只能造出丢字 / 重复 / 乱序，**造不出整段改写**。

### 10.9 官方依据（2026-11-20 查证 · 四条来源）

- SDK 源码 `types/chat/chat_completion_stream_options_param.py` — `include_usage` 在 `data: [DONE]` 前多推一条 `choices` 为空的帧；原文 *"If the stream is interrupted, you may not receive the final usage chunk"*。
- SDK 源码 `_streaming.py` — `__stream__` 里 `if sse.data.startswith("[DONE]"): break`；`finally: response.close()`；超时映射 `APITimeoutError` / `APIConnectionError`。
- SDK README — 流消费**不会自动重试**（重放会重复输出）；「不要假设循环走完就是成功」。
- 供应商「思考模式」官方文档 — `thinking.type` 走 `extra_body`、默认 enabled（effort `high`）、思考模式不支持 `temperature` / 两个 penalty、`top_p` 分模型、`reasoning_content` 位置、带 `tools` 必须回传（否则 400）。

---

## 11. 结构化输出：把「请输出 JSON」换成可校验的契约（2026-11-21 · L4 关卡 2）

### 11.1 两种 JSON 模式不是一回事

| | `json_object` | `json_schema` + `strict` |
|---|---|---|
| 保证 | **能 parse** | **严格遵循 schema** |
| 不保证 | 键名、字段类型、字段数量（供应商原话：「不同提示词或不同次调用的返回结果可能存在差异」） | 业务口径（空字符串照样合法） |
| 提示词要求 | `messages` 里**必须**出现 "json" 字样，否则 **400** | 建议说明，无硬性关键词 |
| 实现层 | 只约束「输出是 JSON」 | 约束交给**解码器**：不合法的 token 在生成时概率为 0 |

### 11.2 strict 的两条硬条件（都能用代码判定）

1. **`required` 必须覆盖全部 `properties`**。想表达「可缺」不能用「不写进 required」，而要用**可空联合类型** `["string","null"]`（并仍留在 `required` 里）。
2. **每个 object 节点都要 `additionalProperties: false`** —— 包括 `$defs` 里的嵌套模型定义。

两家的口径**不一样**，写 schema 要按交集：

| 项 | OpenAI / Azure | 阿里云百炼 |
|---|---|---|
| 可选字段 | **必须**进 `required`（可空表达） | 可以**不进** `required`（不进 = 真可选） |
| 关键字子集 | 窄：不收 `minLength` / `pattern` / `format` / `minimum` / `minItems` / `maxItems` | 宽：上述都收；但 `multipleOf` / `uniqueItems` / `patternProperties` / `propertyNames` 不收，**直接 400** |
| `$ref` | `$defs` 本地引用可用 | 同；**外部 URL 的 `$ref` 不生效且不报错** |
| 多模态输入 | — | 带图/视频/音频时 `json_schema` **静默降级**成 `json_object`，约束失效且**不报错** |

⇒ 结论：schema 里只用**交集关键字**（`type` / `enum` / `items` / `anyOf` / `required` / `additionalProperties`），业务口径（姓名非空、列表至少一项）放到**校验层**。

### 11.3 pydantic → schema：三个会静默犯的错

| 现象 | 真因 | 后果 |
|---|---|---|
| schema 里没有 `additionalProperties` | pydantic 2.14 的 `model_json_schema()` 只给 `type: "object"` | strict 要求每个 object 都有 ⇒ 自己递归补；漏了就是"看起来合规" |
| 字段突然不在 `required` 里 | 写了 `= None`：pydantic 把「有默认值」当「非必填」 | ① OpenAI strict **400** ② 本地校验放水（缺字段用默认值补，照样"通过"）⇒ **20 次通过率虚高** |
| `dict[str, str]` 字段报错 / 需要改写 | 它的 `additionalProperties` 是**一个 schema**（`{"type":"string"}`）而不是布尔 `false` | 覆盖成 `False` = 「只接受空对象」，语义反转且不报错；strict 下只能改写成 `list[KV]` 小模型 |

判据写法：**用「有没有 `properties`」判断 object 节点**，不要用「`type == "object"`」（后者在 `dict[str, X]` 上也成立，还会漏 `type` 为列表的联合类型）。

### 11.4 「解码器在挡」和「校验层在挡」是两件事（本关最大教训）

- **约束解码**：服务端把 schema 编译成语法约束，生成时把不合法 token 的概率压成 0。所以「坏输出」根本没被生成，而不是生成后被拦。
- 观测证据（**推断，不是实证**）：20 次全合规；输出里保留了 `\n  ,` 这种冗余空白（后处理会规整掉，逐 token 生成才会留）⇒ 与约束解码一致。**服务端具体实现未公开**，不能断定是 logit mask 还是"后校验 + 重跑"。
- 于是判据 ② 的正确取证方式是**两路**：
  - 真实破坏性输入（引号 / 换行 / 注入句各 1 次）⇒ 证明**上游**在挡；
  - 负例注入（把坏 JSON 直接喂校验层，**0 次调用**）⇒ 证明**校验层会红**。
  - 「三次都通过」单独看什么都没证明 —— **门没被撞过，证明不了门结实**。
- 校验层仍然必须存在，因为 strict 有**三处静默失效**：多模态降级、`max_tokens` 截断（文档明写别设）、拒绝/空正文。

### 11.5 `model_validate_json`：把两种失败合成一个异常源

```python
data = Enrollment.model_validate_json(text)   # parse + 校验，一步
```

| 失败形态 | `err["type"]` | `loc` | `msg` |
|---|---|---|---|
| 整段不是 JSON（含空串 / 代码块 / 被注入带偏） | `json_invalid` | `()` 空元组 | 自带 `at line X column Y` |
| 字段缺失 / 类型不符 | `missing` / `int_parsing` … | `('age',)`，嵌套为 `('courses', 0, 'title')` | 英文说明 |

- 所以一个 `except ValidationError` 就够，不需要手动 `json.loads` 再 `model_validate`；
- `loc` 是**元组**，元素可能是 `int`（列表下标）⇒ 拼路径前必须 `str()`；
- 默认**宽松模式**：`"25"` 会被转成 `25` 并接受。想判死要 `strict=True` / `StrictInt`，但本地口径就比发出去的 schema 更严 ⇒ **失败归因会变模糊**（是模型错还是本地多拦了一道）。

### 11.6 pydantic 只查类型，业务口径一律不管

实测 `{"name":"", "age":-3, "courses":[]}` **全部通过** —— 因为类上只有 `str` / `int` / `list[str]`，没有 `min_length` / `ge`（这两个会变成 `minLength` / `minimum`，OpenAI strict 不收，所以故意不写进 schema）。

⇒ 「结构对但内容不合规」只能在校验层补，错误串建议带前缀分桶：`json:`（正文不是 JSON）/ `schema:`（结构与类型）/ `rule:`（业务口径），统计脚本按前缀分组。

### 11.7 实测数字（2026-11-21 · 24 次调用）

| 指标 | 值 |
|---|---|
| 通过率（20 次同 prompt） | **20/20**，`finish_reason` 全 `stop` |
| 耗时中位 | **9982.8 ms**（6909.9 ~ 15181.5） |
| 输入 token | **102/次（恒定）** ⇒ schema 走 `response_format`，**不占输入 token** |
| 输出 token | **381 ~ 2032/次（5.3×）** ⇒ reasoning 计在 `completion_tokens` 里，成本要等拆 `reasoning_tokens` |
| 负例注入 | **3/3 判死**，均带 `at line X column Y` |

### 11.8 官方依据（2026-11-21 查证 · 两条来源）

- 阿里云百炼《结构化输出》— `deepseek-v4.1-flash` / `-1017` 在**北京地域的 OpenAI 兼容 Chat Completions** 上支持 JSON Schema，**思考与非思考模式均支持**；给出关键字白名单与不支持项（不支持的直接 400）；可选字段两种写法；**多模态静默降级**；生产建议「不要设置 `max_tokens`，会导致 JSON 被截断」。
- Microsoft Learn《Structured outputs》（Azure OpenAI，口径同 OpenAI）— 专节 `ALL FIELDS MUST BE REQUIRED` 与 `ALWAYS SET ADDITIONALPROPERTIES: FALSE`；根节点不能是 `anyOf`；支持 `$defs` / `$ref` 与递归；`refusal` 字段表达拒绝。

---

## 第 12 章 · 多轮与上下文记账：input 到底在数什么（2026-11-21 · L4 关卡 3）

### 12.1 API 是无状态的

官方口径（阿里云百炼《多轮对话》）：**API 不保存对话历史**，所谓"多轮"完全由客户端模拟 —— 每轮把 `user` 与 `assistant` 追加进 `messages` 数组后**整体重发**。openai-python README 同样没有任何 session / 记忆接口。

⇒ 三个直接后果：

1. 每轮 `usage.prompt_tokens` = **整段历史** + 本次提问的 token 数，不是"这一句"的
2. 同一段历史会被**反复计费**（除非供应商有上下文缓存，那是另一回事）
3. `messages` 越长，每轮越贵 —— 这是"多轮对话为什么成本非线性上升"的全部原因

### 12.2 三个「累计」口径，别混成一个

| 名字 | 公式 | 回答什么问题 |
|---|---|---|
| 本轮 input | `usage.prompt_tokens` | 这一轮实际读入多少（**含全部历史**） |
| 本轮 output | `usage.completion_tokens` | 这一轮生成多少（**含思考 token**） |
| **累计上下文** | **Σ 各轮 input** | 这段对话一共"读入"了多少 token（判据 ② 的口径） |
| 会话当前占用 | 最后一轮 input + 最后一轮 output | 下次请求要发的 `messages` 有多长、还能不能加一轮 |
| 总消耗 | Σ(每轮 input + output) | 账单口径 |

常见错法：把「累计上下文」写成"最后一轮的值"（那是"当前占用"的前半截），或把本轮 `completion_tokens` 也加进去（那是 output，不是上下文）。

### 12.3 思考 token 不随历史重发（本关最重要的一条）

实测（`deepseek-v4.1-flash-1017`，思考模式默认开）：

```
轮1: input 41, output 28, 累计 41
轮2: input 55, output 20, 累计 96   (=41+55)
轮3: input 72, output 21, 累计 168  (=41+55+72)
```

轮2 的 input 只比轮1 **多 14**，而轮1 的 output 是 **28** —— 增量**小于**上一轮输出。原因：

- `completion_tokens` 里混着 **reasoning（思考）token**
- 重发的历史只带 `content`（正文），思考过程**不拼回去**

⇒ 两条推论：

1. **拼思考过程进历史是白烧钱**：供应商口径只要求 `content`；拼了会让 input 涨一大截而模型未必用得上
2. **判据不能拿 output 当"历史增量"的上界**：`input3 − input1 ≥ output1 + output2` 这种写法会把我今天本来绿的判据判成红。正确写法 = **input 单调递增** 或 `input3 > input1`

### 12.4 判据（可复测）

| 判据 | 期望 |
|---|---|
| 无状态证据 | 连续 3 轮，`input` 严格递增（今天 41 → 55 → 72） |
| 累计口径 | 每轮打印的累计 == 逐轮 `prompt_tokens` 求和（今天 96 / 168 / 258，差 **0**） |
| 重启读回 | `exit` 后重启打印「已读回 N 轮」；读回后首轮 input **明显大于**上一轮（带着历史） |
| 思考不入历史 | 每轮 input 增量 **<** 上一轮 output（今天 14 < 28） |
| 不压缩 | 撞上限时报错原文被打印、历史条数不变（今天 `[1, 1048576]` / 240 条未动） |

---

## 第 13 章 · 崩溃安全：保存粒度、原子替换与异常分层（2026-11-21 · L4 关卡 3）

### 13.1 保存粒度决定「最多丢多少」

| 保存时机 | Ctrl+C | 关窗口 / 断电 |
|---|---|---|
| 只在 `exit` 时 | 全丢 | 全丢 |
| 出错分支也存 | 全丢（除非真的走到那个分支） | 全丢 |
| **每轮（append 之后）** | 最多丢正在跑的那一轮 | 最多丢正在跑的那一轮 |
| 追加日志（JSONL/SQLite WAL） | 最多丢最后半条 | 最多丢最后半条 |

⚠️ **保存必须写在数据"进内存之后"**：先 `save` 再 `append` ⇒ 文件永远落后一轮，而 `exit` 时补存一次会把问题藏住 —— **只用 exit 测不出来，必须用 Ctrl+C 测**。

### 13.2 覆盖写不是原子的，改名才是

`open(path, "w")` 一打开就把原文件清零；崩在中间 ⇒ 新旧两头空。正确姿势：

```python
tmp.write_text(json.dumps(...))   # 先写完
tmp.replace(path)                 # 再改名（同盘 rename）
```

`Path.replace()` 是 **rename**，不是复制：实测 20 MB 文件 **0.823 ms**，`st_ino` 与 `st_mtime` 都不变（同一个文件对象换了名字），旧文件的数据块在引用解除后由文件系统回收。

⚠️ Windows 特例：**目标文件被别的程序占用时 rename 会失败**（`PermissionError [WinError 5]`），POSIX 下开着句柄也能改名。

### 13.3 快照 / 追加 / 事务三档

| 档 | 形态 | 崩了丢什么 | 代价 |
|---|---|---|---|
| A 快照 | 整份 JSON 重写（tmp + replace） | 正在跑的那一轮 | 每轮 O(n) 重写，几 KB 量级可忽略 |
| B 追加 | JSONL 每行一条，`append` 写 | 最后半条（读时跳过坏行） | 永不重写历史，天然抗截断 |
| C 事务 | SQLite + WAL | 几乎不丢，还能并发 | 引入依赖、schema 要设计 |

工业界的 agent 多在 B/C：**追加写从不触碰已写好的字节**，所以崩溃只会伤到"尾巴"。

### 13.4 异常分层：只接预期失败，bug 要炸出来

| 异常来源 | 例子 | 策略 |
|---|---|---|
| 预期失败 | 文件被占用、无权限、磁盘满、网络断 | 捕获 → 降级/重试 → **打印人话** |
| 程序 bug | `asdict` 用错、字段拼错、类型不符 | **不捕获**，fail fast（traceback 当场出现） |

官方口径交叉印证：

- **PEP 8**：*"When catching exceptions, mention specific exceptions whenever possible"*；裸 `except:` 会吞 `KeyboardInterrupt` 且 *"can disguise other problems"*；`try` 的范围要最小以免掩盖 bug
- **The Rust Book**：可恢复的**预期**失败 → `Result`；**契约违反 / 调用方 bug** → `panic!`（契约违反永远是调用方的 bug，只能改代码）

⇒ 落盘代码里的正确写法是 `except OSError`。今天实测反例：把 `save_turns` 改成抛 `TypeError`，`except Exception` 会把它报成「历史保存失败」并返回退出码 1 —— 故障现场消失，下次排查只会在磁盘上打转。

### 13.5 坏文件：隔离、特判、TOCTOU

| 做法 | 理由 |
|---|---|
| 坏文件**改名** `*.corrupt-<时间戳>.json`，不删 | 保留证据（是写盘被打断，还是手改坏？）；`raise` 会让一次截断永久封死启动 |
| 先判 `st_size == 0` | 空文件与截断文件**共用** `JSONDecodeError: Expecting value`，不特判会把"本来就没内容"当损坏去隔离 |
| 用一次 `path.stat()` 代替 `exists()` + `stat()` | 两次系统调用之间有窗口（TOCTOU），中间被删会让 `stat()` 抛到容错逻辑之外 |
| `OSError` **不接** | 没权限 / 传了目录是环境问题，把它当"文件坏了"去改名 = 把好文件也毁掉 |
| 结构校验下沉到每条记录 | dataclass 运行时零校验：`{"prompt_tokens": "abc"}` 能一路构造成功，滚到 `sum()` 才炸（且 `bool` 是 `int` 子类，`True` 会冒充 1） |

### 13.6 判据（可复测）

| 判据 | 期望 |
|---|---|
| 崩不丢 | 聊 2 轮 → Ctrl+C → 文件 **2** 条；无 `.tmp` 残留 |
| 原子性 | 写盘中途被杀时，磁盘上仍是"上一份完整文件"（先写 tmp 再 replace） |
| 改名不是复制 | 20 MB `replace` < 5 ms，`st_ino` / `st_mtime` 不变 |
| 坏文件隔离 | 截断 → 出现 `*.corrupt-*.json`、程序继续；空文件 → 打印「空历史」且**不**隔离 |
| 结构校验 | 缺字段 / 多字段 / `"abc"` / `bool` → 报「缺少 [...]」而不是 `missing N positional arguments` |
| 异常分层 | 落盘抛 `TypeError` → **traceback**（不是「保存失败」） |

---

## 第 14 章 · 成本记账：单价是三档，不是一档（2026-11-22 · L4 关卡 4）

### 结论先摆

一次调用的钱 = **四段**，不是"总 token × 一个价"：

```
cost(元) = (cached × p_缓存 + (prompt − cached) × p_入 + completion × p_出) / 1e6
```

单价（元 / 百万 tokens，华北2·北京，官方两条来源交叉）：

| 计费项 | 闲时 22:00–次日 08:00 | 忙时 |
|---|---|---|
| 输入（缓存未命中） | **1** | **2** |
| 输入（缓存命中） | **0.02** | **0.04** |
| 输出 | **4** | **8** |

- 峰谷差 **2×**，缓存命中差 **50×** —— 后者才是真正拉开差距的那一档；
- 互证：DeepSeek 官方页（USD）非高峰 `$0.15 / $0.60` per 1M ≈ ¥1.07 / ¥4.26，与百炼闲时价同量级。

### 三个容易写错的地方

1. **`prompt_tokens` 已包含命中部分**。所以未命中段必须是 `prompt − cached`；写成 `prompt × p_入 + cached × p_缓存` 会把命中部分算两遍。
2. **`reasoning_tokens` 不单独计价**。官方价格表里没有"思考"这一栏，思考按**输出价**计，并且**已经包含在 `completion_tokens` 里**。拆它只为了解释波动（同 prompt 输出 381→2032），不是为了少算钱。
3. **`cached_tokens` 拿不到 ≠ 没命中**。它是 Optional；`None` 只能"不知道"，算钱时按全部未命中（保守高估），但账本要保留 `None` —— 填 0 等于宣称"全按新价"。

### 上游只给 token，不给钱

- 原生 API 的 `usage` **没有金额字段**；聚合网关（OpenRouter 一类）回传的 `cost` 是网关扩展，**带加价**，不能当官方价反推单价；
- 钱在平台账单侧另算：推理账单**分钟级出账**，实测"调用次数 10 分钟可见、token/金额约 40 分钟"；按量付费是"**预占 + 月结**"，出账 ≠ 扣款；
- 平台统计**按请求名分组**：别名与快照名（`deepseek-v4.1-flash` / `-1017`）是**两栏**，对账要按实际请求的那个名字筛。

### 对账的三级降级

| 级 | 取证 | 强度 |
|---|---|---|
| **Ⅰ** | 单次 token 逐条对账（平台 input/output == 程序打印） | 最强 |
| **Ⅱ** | 调用次数对账（次数一致） | 中 |
| **Ⅲ** | 金额对账（差值 0 或写清差异侧：缓存命中 / 峰谷 / 代金券） | 弱 —— 界面精度常只到 0.01 元，0.002 元一律显示 `0.00` |

本关实测取到 **Ⅰ 级（聚合口径）**：平台 `4 次 / 391 tokens`，本地 4 次 `total` 合计 `106+109+79+97 = 391`。
⚠️ 平台只给"入+出合计"、**不给拆分** ⇒ 这一级能证明总账对，证明不了"输入/输出各自对"。

### 代码落点

`llm_cli/billing.py`：`Price` / `PRICES`（白名单）→ `is_off_peak()`（**aware 时间先 `astimezone`**）→ `bill()`（四段 + 三条守卫）→ `format_bill()`（6 位小数）。
`bill()` **不接收** `model` 参数，只认 `answer.model`（上游回传）—— 让"手传错模型"这件事**无法被表达**。

---

## 第 15 章 · prompt 长度值不值：三版实测与缓存门槛（2026-11-22 · L4 关卡 4）

### 实测（同一任务 10 条输入，只有"指令细不细"一维不同）

| 版本 | 前缀 | 输入 | 命中 | 输出 | 成本 | 每分成本 |
|---|---|---|---|---|---|---|
| short | 88 字符 | 706 | **0** | 689 | 0.0069 | 0.000239 |
| mid | 207 字符 | 1326 | **0** | 531 | 0.0069 | **0.000238** |
| long | 568 字符 | 3186 | **1024（32%）** | 691 | 0.0099 | 0.000341 |

**结论**：三版效果**打平**（29/30），长 prompt **只加成本（+43%）**，不换来可测量提升。

### 两条被数据纠正的直觉

1. **"长 prompt 一定更贵"不完全对**：long 的输入 token 是 short 的 **4.5 倍**，成本只高 35% —— 因为 64%/32% 的输入躺在缓存里（命中价 0.02 vs 1，50 倍）。
2. **"长 prompt 效果更好"也不成立**：只要判据里有一项写错（`completion ≤ 30` 对思考模型恒假），"长版多 2 分"就会是假象。

### 缓存门槛：前缀长度决定一切

- 官方口径：缓存以 **64 token** 为存储单元，不足的不缓存；前缀必须**从第 0 个字符**开始完全一致；
- 实测门槛：本平台 **≈ 256 token**（short 88 字符 / mid 207 字符 两轮 **恒 0**；long 568 字符才命中）；
- 命中率**不稳定**：同一前缀两轮分别 **64% / 32%**（官方"尽力而为"），且第二次跑并没有更热 ⇒ 有 TTL / 淘汰；
- ⇒ 想让缓存真生效：**稳定内容放前缀**（system / 规则 / 长文档），**变化内容放后缀**，前缀别插变量。

### 超长输入：截断是错解，压缩才是正解

- 超限行为：`400 Range of input length should be [1, 1048576]`，**服务端不裁历史**（官方 FAQ：「遇到上下文超限相关的报错时建议新建会话」）；
- **分段总结不省 token**：5 万 token 文档 —— 全量一次 `0.1064` 元 vs 分 5 段 `0.1164` 元（输入总量不变，还多一层输出）；
- 分段还**不可逆地毁掉全局信息**（每段不知道全文）；
- ⇒ 单次场景**不该截断**；**压缩上下文**（摘要替原文）只对"同一段内容反复重发"划算 = **多轮历史**（关卡 3 实测 input 41→55→72→90，每轮重发全史），落点在 L5 / coach。

### 一个判据设计教训

判"输出啰嗦"必须用**可见输出** `completion_tokens − reasoning_tokens`。思考 token 混在 completion 里，用 `completion ≤ 30` 等于要求模型不许思考 —— 本关 30 条只过 1 条，整轮结论作废。

---

## 第 16 章 · torch 张量 API：L2 关卡 1–2 用得到的那一套（2026-11-23）

> 只收「本关真用过、能把判据写成数字」的 API。没用过的（`register_buffer` / `gradcheck` / 多头 reshape / KV cache）**不提前占位** —— 那属于关卡 3–4 / L5。

### 16.1 速查表

**建张量**

| API | 干什么 | 坑 |
|---|---|---|
| `torch.tensor(list, dtype=...)` | Python list → 张量 | 不写 dtype 就是全局默认 `torch.float32` |
| `torch.randn(shape, generator=g, dtype=)` | 从 **N(0,1)** 采样（不是任意值） | 多数落在 ±1、95% 落在 ±2 |
| `torch.Generator().manual_seed(seed)` | 独立随机源 | **不显式传 `generator=g` 就不生效**，退回全局源 |
| `torch.arange(1, T+1)` | 等差数列，做「期望值」 | 终点要 +1 |
| `torch.ones(shape, dtype=torch.bool)` | 全 True | 默认 dtype 是 **float32**，不写 `dtype` 得到的是 1.0 |

**形状**

| API | 干什么 | 坑 |
|---|---|---|
| `x.shape` | 形状元组 | 前面的数是**数量**（B / T / heads），`shape[-1]` 才是**特征维 d** |
| `x.T` | 2D = 转置 | **ndim≠2 已 deprecated**（本机 torch 2.11 实测告警）⇒ 高维用 `x.mT` 或 `x.transpose(-2,-1)` |
| `x.shape[-1]` / `x.size(-1)` | 取特征维 | 缩放因子 `1/√E` 的 E 就是它 |

**相乘**

| API | 干什么 | 坑 |
|---|---|---|
| `a @ b`（= `torch.matmul`） | 批量矩阵乘 | 在最后两维上做 |
| `a * b` | **逐元素**乘 | 不是矩阵乘；配 `.sum()` 才是手写点积 |

**归一与统计**

| API | 干什么 | 坑 |
|---|---|---|
| `x.softmax(dim=-1)` | 沿最后一维归一，行和 = 1 | `dim` 选错就是沿错方向归一 |
| `x.log2()` | 以 2 为底，熵用（bit） | 0 会得 `-inf` ⇒ 先 `x[x>0]` |
| `x.std()` | 标准差 | **默认 `correction=1`（Bessel，除以 N−1）**；N=64 时比有偏版高 **0.79%**，N=4096 时只差 **0.012%**（本机实测） |

**掩码（本关主角）**

| API | 干什么 | 坑 |
|---|---|---|
| `torch.ones(t,t,dtype=torch.bool).tril(diagonal=0)` | 下三角，True = 保留 | `diagonal` **默认就是 0**；`-1` → 第 0 行全 False → softmax 全 -inf → **NaN**；`+1` → 每行多留一个「未来」位（泄漏） |
| `~mask` / `mask.logical_not()` | 布尔取反（实测两者逐元素相同） | **只对 bool/int 有效**；float 抛 `TypeError: ~ (operator.invert) is only implemented on integer and Boolean-type tensors`；int 上是**位运算**（`~[0,1,2] = [-1,-2,-3]`） |
| `Tensor.masked_fill(mask, value)` | mask 为 **True** 处填 value | 语义是「填」不是「留」⇒ 传 `causal_mask` 时要**取反** |
| `float("-inf")` | 加进分数 ⇒ `exp(-inf) = 0`，权重精确 0 | 别用 `-1e9` 凑数（换 dtype 会溢出/失真） |
| `torch.finfo(dtype).min` | 有限最小值（fp16 **-65504**、fp32 **-3.40e38**） | 做 padding 掩码时替代 `-inf`，避免「整行屏蔽 ⇒ `softmax(全 -inf) = nan`」（本机实测确实是 `nan`） |

**判据写法**

| API | 干什么 | 坑 |
|---|---|---|
| `(a-b).abs().max().item()` | 最大逐元素差 —— 判据的标准量 | 忘了 `.item()` 拿到的是 0 维张量 |
| `(w != 0).sum(dim=-1)` | 每行非零个数 | 配 `torch.equal(nnz, arange(1,T+1))` 查结构 |
| `torch.equal(a, b)` | 形状 + 元素**完全**相等 | `nan != nan` ⇒ 含 NaN 恒 False；形状不同直接 False |
| `x.clone()` | 深拷贝 | **扰动前必须 clone**，否则 `x[[4,5]] += d` 改的是基准那一份 |
| `x.tolist()` | → 嵌套 list | 打印/交给纯 Python 用 |

**融合实现（生产写法，本关只做对照）**

| API | 干什么 | 坑 |
|---|---|---|
| `F.scaled_dot_product_attention(q,k,v, is_causal=True)` | 一步算完的融合核 | 与 `attn_mask` **互斥**（同时给抛错） |
| bool `attn_mask` | 掩码 | **`True` = 参与注意力**，与 `MultiheadAttention` 的 `key_padding_mask` **相反** |
| `scale=` | 缩放因子 | 默认 `None` ⇒ `1/√E`，E = `query.size(-1)` |

### 16.2 四处反直觉（考的就是这四条）

1. **`True` 表示「参与」**，不是「屏蔽」—— 从 MHA 迁过来必须取反。
2. **`shape[-1]` 是内容，前面的数才是数量** —— 缩放用小 `-1`（d），掩码边长用 `-2` 或 `0`（T）。
3. **`std()` 默认无偏** —— 1/64 这种量级的比较会被 0.79% 的偏置吃掉一半阈值。
4. **`masked_fill` 填的是 True 处** —— 所以 `masked_fill(~mask, -inf)` 才对。

### 16.3 本关实测数字（2026-11-23）

| 项 | 实测 |
|---|---|
| 判据 ① D 行和偏差 | **2.220e-16** |
| 判据 ① A（改 5、6 → 前 4 个位置） | **0.000e+00**（严格 0） |
| 判据 ① B（只改 6 → 第 5 个位置） | **0.000e+00**（严格 0） |
| 判据 ① C（不掩码对照组） | **2.632e-02**（> 0，证明扰动真会传到过去） |
| 自写 vs 官方 `is_causal=True` | 输出差 **2.220e-16**（float64 eps 量级） |

### 16.4 官方依据（2026-11-23 查证）

- `docs.pytorch.org` 2.15 · `scaled_dot_product_attention`：`is_causal` 为真 = 下三角（等价实现 `ones(L,S,bool).tril(diagonal=0)` → `masked_fill_(~temp_mask, float("-inf"))`，**-inf 在 softmax 之前**）；bool 掩码 `True` = 参与；`attn_mask` 与 `is_causal` 同时给**抛错**；`scale=None ⇒ 1/√E`。
- `docs.pytorch.org` 2.15 · `torch.tril`：`diagonal` 默认 **0**；正值多包含上方对角线，负值排除下方对角线；**非原地**，返回新张量。
- `docs.pytorch.org` 2.15 · `torch.std`：默认 `correction=1`（Bessel）；**2.0 起** `unbiased` 改名 `correction`，且 `correction` 只能关键字传参。
- 本机实测（torch 2.11.0+cu128）：高维 `.T` 触发 deprecation 告警；`~float_tensor` 抛 `TypeError`；`softmax(全 -inf) = nan`；`exp(-inf) = 0.0`。

### 16.5 自测卡（遮住上面口述，答不出回扫）

1. `q` 形状 `(8,4)` 时，`q.shape[-1]` 和 `q.shape[-2]` 各是什么？缩放该用哪个，为什么？
2. 为什么要 `masked_fill(~mask, -inf)`，少一个 `~` 会怎样？
3. `tril(diagonal=-1)` 之后第 0 行是什么？跑判据会红在哪一项，报什么值？
4. `torch.equal` 遇到 `nan` 返回什么？这会不会让判据假绿？
5. `torch.randn` 不传 `generator` 与传了固定 seed，跑两次的区别是什么？

---

## 第 17 章 · 最小 block：三件缺一不可，与「判据有牙」的取证法（2026-11-24 · L2 关卡 3）

### 一个 block 为什么必须补三样

注意力本身只是「权重和为 1 的加权平均」，**仍是线性的**；叠多少层都等价于一个矩阵。所以一个 block 必须补：

| 补给 | 解决什么 | 实测证据 |
|---|---|---|
| **LayerNorm** | 每个 token 的 C 维拉回「均值 0、有偏方差 1」⇒ 深浅都不爆/塌 | 输出行范数 = **1.99999…**（√4 = 2）⇒ 落在半径 √C 的球面 |
| **残差** `x + f(x)` | 保尺度 + 给梯度留恒等通路 | 支路 absmax **0.002** vs 主项 **2.31**（0.09%）⇒ 初始 ≈ 恒等；输入 ×1000 输出也跟着 ×1000 |
| **FFN + GELU** | 唯一的逐点非线性 | 去掉 GELU 后 `C→4C→C` == 一层矩阵（差 **8.88e-16**，128 参数等于 16 参数） |

pre-LN（归一化放在子层入口、残差绕过整段）是当前默认写法：`x = x + attn(ln1(x))` → `x = x + ff(ln2(x))`。post-LN 要把残差也归一化，深了要靠 warmup 救。

### LayerNorm 的四条口径（错一条就和官方对不上）

| 口径 | 正确值 | 错法症状 |
|---|---|---|
| 方差分母 | **有偏**，等价 `correction=0` | 用默认（`correction=1`）在 C=4 时差 **4/3**，输出差 15% |
| ε 位置 | 加在**方差里**再开方 | 写成 `sqrt(var) + ε`：数值仍有限，但 var=0 处**梯度仍是 inf** |
| 输出方差 | `var/(var+ε)`，**不是** 1 | `[0.1,0.2]` → ±**0.998**（差 0.2%），所以判据不能卡 `|var−1|<1e-12` |
| 单元素上界 | **√(C−1)** | `[0,0,0,5]` 归一化 = `−0.577×3` 与 **1.732 = √3**，正是取到上界的形状 |

几何说法：归一化后每个 token 落在半径 √C 的球面、且垂直于全 1 方向；γ/β 再把长度与偏移还回去（初始 γ=1、β=0 ⇒ 纯标准化）。

### 尺度不变性只属于 LN 支路，block 整体是线性传递

输入 ×1000 时：LN 输出仍 **1.7083**（在 √3 = 1.7321 内），注意力权重几乎不变（差 6.37e-08），但 **block 输出 absmax 跟着 ×1000** —— 因为残差把主项原样传下去。⇒ 判据只能查「LayerNorm 输出有界」，查 block 输出有界就是错的判据。

### 「判据有牙」的取证：变异测试 + 两个反例

第一版判据（形状 / 参数量 / ×1000 有界）**三个变异全绿漏网**：

| 变异 | 为什么漏 |
|---|---|
| `correction=0 → 1` | 输出只是 1.7083 → 1.4795，仍在 `[1.0, √3]` 内 |
| `sqrt(var+ε) → sqrt(var)+ε` | 随机数据上只差 **7.58e-05**，看不见 |
| 删掉第一段残差 | 形状不变，**gradcheck 也过**（函数仍处处可微） |

补法与两条方法论：

1. **「有界」类判据抓不到「口径错」** ⇒ 必须与**独立参考实现**逐元素对账（`nn.LayerNorm`，阈值 1e-12）。补后 M1 差 **2.289e-01**、M2 差 **8.825e-01**。
2. **边界用例要按「哪个量会暴露这个 bug」设计**：M2 必须喂 `var≈1e-6` 的输入才能把 ε 位置的影响放大。
3. **结构性问题要单独取证**：残差用 `|block(x)|max / |x|max ∈ [0.5, 2]`（正常 **1.0004**，删残差塌到 **0.0006**）。

### gradcheck 的两条前提

| 前提 | 原因 |
|---|---|
| 张量必须 **float64** | float32 的有限差分噪声相对 ~3e-4，与默认 `rtol=1e-3 / atol=1e-5` 同量级 ⇒ 写对写错都可能随机红绿 |
| **参数也要当输入** | `gradcheck(f, (x,))` 只验 `d输出/d输入`；要验 16 个参数的梯度，用 `functional_call(module, dict(zip(names, ps)), (x,))` 一起喂（本关 60 个入参） |

### 交叉熵作为损失：梯度是反传的第一环

$$\frac{\partial\,CE}{\partial\,\text{logits}}=\frac{\text{softmax}-\text{onehot}}{N}$$

- 三个口径：**onehot 硬标签 + 自然对数（nats）+ 按样本平均**（分母 N，不是 N×V）。
- 每行梯度之和恒为 **0**（实测 3.166e-17）—— softmax 概率和为 1 ⇒ 只有相对值有意义。
- 实现用 `logsumexp`（内部 max-shift）：朴素 `log(exp(1000).sum())` 得 `inf`，稳定版得 **1000.6931**。
- `gather` 取真类那格有三条硬约束：`index.ndim == input.ndim`、dtype int64、值在 `[0, size(dim))` ⇒ `targets (N,)` 先 `unsqueeze(-1)`、取完 `squeeze(-1)`。

### 顺带查清：bias 在单头注意力里谁有用

| 参数 | 实测 | 结论 |
|---|---|---|
| `bk` | 权重差 **5.55e-17** | 无效 —— softmax 平移不变，只给每行加同一常数 |
| `bv` | 与 `bo=bv@wo` 差 **4.44e-16** | 冗余 —— 行和为 1 ⇒ 退化成常数 |
| `bq` | 权重差 **0.077** | 有效 —— 等于给每个 key 一个固定加分 |

**注意力的「参与」由 softmax 权重 + 掩码决定**（掩码才是 0/1 硬开关）；bias 只改偏移。真正像「门槛」的是 FFN 里 GELU 之前的 `b1`（平移激活切入点）。

### 思考题

1. 为什么 `nn.Linear` 的 weight 是 `(out, in)`？手写 `z @ w1` 报的 `(4x4 and 16x4)` 里，两个 4 分别是哪个维度？
2. 判据 D 只喂随机输入能不能抓 M2？把 `var` 从 1e-6 调到 1e-2，差值会变成多少量级？
3. 去掉残差后 gradcheck 为什么还能过？那它到底验的是什么？
4. 如果 `cross_entropy` 用 `.sum()`，判据 ② 的哪一项会红，值差多少倍？
5. 输入 ×1000 时，为什么「注意力权重几乎不变」而「block 输出跟着 ×1000」两件事可以同时成立？

---

## 附录 · 待学章节（计划编号 · 学到时展开成正文）

> 只列主题，不提前写内容——避免"抄来的笔记"占位。编号沿用计划表，与正文章号不强制对应。

| 阶段 | 章节 | 主题 |
|---|---|---|
| S0 | 1-9 | **Python 语言层整体迁出** → 见 `Python-基础笔记.md`（语法总览 / list / 可变性陷阱 / dict·set / 装饰器 / 模块 / async / httpx / pydantic）—— 本册不再记 Python 语言本身 |
| S1 | 5 | messages 结构 / role / system vs user（**基础已在第 2.3 节**；剩 `system` 与 `user` 的取舍、多轮对话与上下文拼接） |
| S1 | 6 | 流式（SSE）与首 token 延迟 → **已在第 10 章**（2026-11-20 落章） |
| S1 | 7 | 采样参数：temperature / top_p / max_tokens → **失效条件**见第 10.7 节（思考模式下温度不生效）；其余待学 |
| S1 | 8 | token 计量与成本估算 |
| S1 | 9 | 结构化输出：JSON Schema + 校验 + 重试 |
| S1 | 10 | Context Engineering：上下文怎么组织 |
| S2 | 11 | 工具调用协议与工具描述（ACI） |
| S2 | 12 | ReAct 循环与终止条件 |
| S2 | 13 | 上下文裁剪 / 压缩 / 记忆 |
| S3 | 14 | 切分策略与 embedding |
| S3 | 15 | 向量检索与混合检索 / rerank |
| S3 | 16 | 评测：eval 集 / LLM-as-judge / 幻觉归因 |
| S4 | 17 | LangGraph：State / Node / Edge / checkpoint |
| S4 | 18 | MCP：Tools / Resources / Prompts 与 server 编写 |
| S4 | 19 | 可观测：trace / 成本与延迟报表 |
| S4 | 20 | 部署：FastAPI + Docker |
| 原理 | 21 | nanoGPT：token → embedding → attention → 训练循环 |
