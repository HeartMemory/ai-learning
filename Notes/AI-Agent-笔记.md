# AI Agent / 大模型应用笔记（2026-10-28 起）

> AI 主线知识点沉淀，按主题组织，随进度更新（与 `Python-基础笔记.md` 同规格，一本只留一个家）。
> 代码实例都在 `practice/` 下的对应目录里，可运行验证。
> 原则：**只记真实学到的内容与真实踩过的坑**；每个知识点尽量配一条**可实测的判据**。

**覆盖进度**：**S0 环境基线 ✅**（2026-10-28 · 第 1 章）｜ **模型 API 打通 ✅**（2026-10-29 · 第 2 章 · 过渡期闸门提前达成）｜ **多轮对话与上下文 ✅**（2026-10-30 · 第 3 章）｜ S1 的流式 / 结构化输出 / 成本 ⬜ ｜ ⚠️ **2026-10-29 起 Python 语言层单独成册** → 语法 / 容器 / 可变性 / 装饰器 / async / httpx / pydantic 一律记进 `Python-基础笔记.md`，**本册只管 LLM 与 Agent 层**

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

## 附录 · 待学章节（计划编号 · 学到时展开成正文）

> 只列主题，不提前写内容——避免"抄来的笔记"占位。编号沿用计划表，与正文章号不强制对应。

| 阶段 | 章节 | 主题 |
|---|---|---|
| S0 | 1-9 | **Python 语言层整体迁出** → 见 `Python-基础笔记.md`（语法总览 / list / 可变性陷阱 / dict·set / 装饰器 / 模块 / async / httpx / pydantic）—— 本册不再记 Python 语言本身 |
| S1 | 5 | messages 结构 / role / system vs user（**基础已在第 2.3 节**；剩 `system` 与 `user` 的取舍、多轮对话与上下文拼接） |
| S1 | 6 | 流式（SSE）与首 token 延迟 |
| S1 | 7 | 采样参数：temperature / top_p / max_tokens |
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
