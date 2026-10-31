# AI 主线学习打卡（智能体 / 大模型应用）

> **2026-10-30 起，本仓库是 AI 主线的唯一学习记录**：代码 + 打卡 + 复盘 + 笔记全在这里。
> 目标：**2027 暑假实习**，双方向投递 —— A 版简历（AI 应用 / Agent 岗）+ B 版简历（游戏公司 AI 岗）。
> 学习方法沿用：每日打卡 → 复盘 → 笔记 → 按主题提交；**判据要有实测数字**。

## 目录结构

```
ai-learning/
├── practice/          # 小练习（两个桶：python/ 语言层 · ai/ LLM 层）
├── projects/          # 交付物（能跑、有 README、有实测数字）
├── Notes/             # 每日复盘（YYYY/MM/DD.md）+ Python / AI-Agent 两册笔记
└── README.md          # 本文件（打卡表 + 阶段 + 纪律）
```

## 练习 vs 交付物

> **练习**（`practice/`）为学某个特性而建，**不追求完整、不占交付物名额**，学完留在原地当参照。
> **交付物**（`projects/`）要**能跑一局 + 有 README + 有实测数字**，是简历与面试的取景地，编号从 1 起。
>
> 已验证的写法再搬进交付物；**旧交付物一律不回改**。

练习区内部再分两个桶，判据见 `practice/README.md`：

| 桶 | 判据 |
|---|---|
| `practice/python/` | 只跑标准库 / 只教 **Python 语言本身** |
| `practice/ai/` | 涉及 **模型 / token / prompt / 工具调用 / RAG / 评测** |

## 阶段与交付物

| 阶段 | 内容 | 交付物 | 状态 |
|---|---|---|---|
| **S0** | Python 主线（语法差异 / 类型注解 / async / httpx / pydantic） | 并发抓取 + 存 JSON 的小工具 | 🔄 进行中 |
| **S1** | LLM API：流式 / 结构化输出 / token 与成本 / prompt | `projects/llm-cli/` | ⬜ |
| **S2** | **手写 Agent（禁用框架）**：ReAct / trace / 工具 schema / **工具鲁棒性** | `projects/mini-agent/` | ⬜ |
| **S3** | RAG + 评测：切分 / embedding / 检索 / **失败归因三分法** | `projects/rag-notes/` | ⬜ |
| **S3.5** | **Agent 安全与红队**：提示注入 / 间接注入 / 工具越权 / 沙箱 | `projects/redteam-report/` | ⬜ |
| **S4** | LangGraph / MCP server / 观测 / FastAPI + Docker | `projects/mcp-server/` | ⬜ |
| **S5** | 旗舰项目（游戏场景 Agent，双简历通用） | `projects/game-agent/` | ⬜ |
| 并行 | 底层原理：手敲一遍 nanoGPT（每周 1-2h） | `practice/python/nanogpt/`（长期，不按日归档） | ⬜ |

## 环境要求与本地运行

| 部分 | 需要什么 | 怎么跑 |
|---|---|---|
| `practice/python/` | **Python 3.14.8** + **uv 0.12.23**（纯标准库的练习不需要 venv） | 进到练习目录：`uv sync` → `uv run python <脚本>.py` |
| `practice/ai/` | 同上 + **模型 key 放本项目 `.env`**（不入库，模板为 `.env.example`） | `uv sync` → `uv run python ask.py`；静态检查 `uv run ruff check .` |
| `projects/` | 按各项目自己的 `README.md` | 同上 |

- **别用裸 `python`** 跑装了依赖的脚本（看不到工程 `.venv` 里的包）
- 国内下载慢要配用户级 `%APPDATA%\uv\uv.toml` 镜像

## 打卡规则

1. **提交信息**：`MM-DD: [AI] 描述`（算法用 `[算法]`、笔记用 `[笔记]`、打卡用 `[打卡]`）
2. **每日收尾必做**：更新本文件打卡行 → 写 `Notes/YYYY/MM/DD.md` → 维护两册笔记 → 提交
3. **判据要有数字**：token / 耗时 / 通过率 / 成本，四者至少填一个；没有就写「今日无」
4. **算法（2026-10-30 起）**：**暂停刷题**，改为「**随用随学**」—— AI 项目用到哪个算法就当天补哪个，记进笔记；**2027-02 投递前恢复集中刷题约 4 周**
5. **硬纪律**：`S2 之前禁止引入 langchain / langgraph`（先会手写）；**key 永不入库**
6. **交接件** `HANDOVER.md` 含个人信息，已 gitignore，**不上传**

## 打卡进度

| 日期 | 学习内容 | 实测数字 | 状态 |
|---|---|---|---|
| 10-28 周三 | AI 主线立项；env 工程 `practice/python/2026/10/28-python-env` 建成（uv + ruff + 依赖基线） | Python 3.14.8 ｜ uv 0.12.23 ｜ ruff 全绿 | ✅ |
| 10-29 周四 | 模型 API 打通：`practice/ai/2026/10/29-llm-hello`（`ask.py` / `check_llm.py`）；Python 升到 3.14.8 | **96 tokens**（38+58）｜ **1254.49 ms** | ✅ |
| 10-30 周五 | S0 第 3 章**可变性陷阱**（默认参数 / 改对象 vs 重新绑定 / `+=` 看类型）+ 第 4 章 **dict / set / tuple**（tool schema 形状、`\|` 与 `\|=` 的 id 判据、**可哈希**与多层递归 `to_key`）；多轮对话实战 `practice/ai/2026/10/30-multi-turn`；练习区按 `python/` / `ai/` 分桶并按年/月分层 | **prompt 46→62→96** ｜ 消息数 **2→4→6** ｜ 首轮 **1185.61 ms** ｜ `uv sync` **59 包 / 10517 ms** ｜ 第 4 章判据 **7/7 绿**（`{1, True, 1.0}` 只留 **1** 个；嵌套键可哈希） | ✅ |
| 10-31 周六 | S0 第 5 章 **async / await 与事件循环**：顺序 await vs `gather` vs `TaskGroup` 三条实测 + `time.sleep` 同步阻塞反例与 `to_thread` 解药 + 异常对照（`gather` 不取消 / `TaskGroup` 取消并抛 `ExceptionGroup`）；练习 `practice/python/2026/10/31-async` | `serial` **2.02 s** ｜ `gather` **1.00 s** ｜ `TaskGroup` **1.00 s** ｜ 同步阻塞 **2.00 s** → `to_thread` **1.00 s** ｜ 判据 **5/5 绿** ｜ **0 次模型调用（0 token）** | ✅ |

（每天收工后更新）

## 沉淀笔记

- [Python 基础笔记](Notes/Python-基础笔记.md)：**语言层**专册（语法总览 C#→Python 对照 / list 容器 API / 可变性陷阱 …），2026-10-29 建
- [AI Agent 笔记](Notes/AI-Agent-笔记.md)：**LLM 与 Agent 层**专册（环境基线 / 模型 API 打通 / usage 与 reasoning_tokens …），2026-10-28 建
- 每日复盘：`Notes/YYYY/MM/DD.md`（自 2026-10-28 起连续）
