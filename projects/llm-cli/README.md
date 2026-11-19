# llm-cli

**一句话**：把「调一次模型」做成能演示的命令行 —— 一条命令进去，正文、**输入 token / 输出 token / 耗时**一起出来。

> L4（调用与可控）交付物。四关依次往上加：**关卡 0 打通与记账** → 关卡 1 流式 → 关卡 2 结构化输出 → 关卡 3 成本与 prompt。

## 怎么跑

```powershell
cd projects/llm-cli

# 1) 配置（.env 不入库，模板入库）
Copy-Item .env.example .env     # 填 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL

# 2) 装依赖（首次）
uv sync

# 3) 跑
uv run python -m llm_cli "用一句话解释：什么是二分查找？"
uv run python -m llm_cli "换个模型再问一次" --model <模型名>
```

只用标准库 + `openai` + `python-dotenv`；`requires-python >= 3.14`。

## 判据（L4 关卡 0 · 两条全绿才算过）

- [x] **① 打通与记账**：CLI 跑通一次**真实调用**，报三个数 —— **输入 token / 输出 token / 耗时(ms)**，三个数**用平台返回的用量字段**（`response.usage`），不自己估 —— ✅ 2026-11-19 实测（见下「实测数字」）
- [x] **② key 只在 `.env`**：下面三处自查全过 —— ✅ 2026-11-19 三处全过

```powershell
git check-ignore -v projects/llm-cli/.env          # 期望命中 .gitignore 里的 `.env`
git grep -I -n "sk-"                               # 期望无输出
Select-String -Path .\llm_cli\*.py -Pattern 'api_key\s*=\s*"'   # 期望无输出（只允许从 settings 取）
```

## 代码地图（哪些要自己写）

| 文件 | 谁写 | 内容 |
|---|---|---|
| `llm_cli/config.py` | **AI（胶水）** | 读 `.env` → `Settings`；缺项 `SystemExit(2)`；`api_key` 的 `repr=False` 防日志泄漏 |
| `llm_cli/llm.py` → `ask()` | **你** | 计时 → 发请求 → 从 `usage` 取三个数 → 返回 `Answer` |
| `llm_cli/cli.py` → `main()` | **你** | 建 client → 调 `ask()` → 打印三个数 → 返回退出码 |
| `llm_cli/cli.py` → `build_parser()` | AI（机械） | `prompt` 位置参数 + `--model` 覆盖 |
| `llm_cli/__main__.py` | AI（胶水） | `python -m llm_cli` 入口 |

退出码：`0` 成功 ｜ `1` 其它异常（含 `usage` 缺失、请求失败）｜ `2` 参数 / 配置缺失（argparse 与 `config` 抛的）｜ `3` 跑通了但**正文为空**（已记账）。

关卡 0 完成状态（2026-11-19）：`ask()` 与 `main()` 均由本人写完，`uv run ruff check .` 全绿、Pylance **0 诊断**。

## 实测数字（2026-11-19 跑通）

| 指标 | 值 | 怎么测的 |
|---|---|---|
| 模型 / BASE_URL 形态 | `LLM_MODEL` + `LLM_BASE_URL` 均来自 `.env`（OpenAI 兼容入口） | `.env` |
| 输入 token（`usage.prompt_tokens`） | **31** | 终端输出 |
| 输出 token（`usage.completion_tokens`） | **295** | 终端输出 |
| 端到端耗时（ms） | **2873.43** | `perf_counter` 差值 |
| `finish_reason` | **stop** | 终端输出 |
| 判据 ② 三处自查 | `check-ignore` → 命中 `.gitignore:12:.env` ｜ `git grep -I 'sk-'` → 无输出 ｜ `cli.py`/`llm.py` 无 `os.environ`、无字面量 key | 上面三条命令 |
| 输出 / 输入 token 比 | **295 / 31 ≈ 9.5×** | 成本大头在**输出**，不在提问 |

## 设计取舍（判据要的是「做过选择」，这两条必须能自己讲清）

| 决策 | 选了什么 | 理由 |
|---|---|---|
| 正文为 `None` | `message.content or ""` 兜底 | `finish_reason` 可能是 `length` / `content_filter` / `tool_calls` —— 这几种**token 照扣**，不能因为没正文就丢掉账单 |
| 正文为空怎么算 | 打印完整账单后 `return 3` | **记账与成败判定分离**：已扣费要可查，成败交给退出码 |
| `usage` 拿不到 | 抛 `RuntimeError` | 判据 ① 要平台数字；拿不到不许编一个 0 |
| 请求异常（401 / 429 / 超时） | **当前不接**，让其冒泡 | 退出码仍是 `1`，traceback 里能直接看到 `openai` 的错误码；接不接属于「能不能说一句比栈更有用的话」 |
| `--model ""` 空串 | 静默回落配置（`or` 的语义） | 已知行为，未额外校验 |

## 踩坑

| 现象 | 真因 |
|---|---|
| `401` | key 粘错 / 带空格 / `.env` 没被读到（`config.py` 用绝对路径读项目根的 `.env`） |
| `404 Not Found` | `LLM_BASE_URL` 少了或多了 `/v1` |
| 报「缺配置」但 `.env` 明明填了 | 变量名必须是 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` |
| 限长参数不生效 | `max_tokens` 已被官方标 deprecated ⇒ 用 `max_completion_tokens`（o 系模型只认后者） |
| 正文是 `None` | 模型可能只回 reasoning 或空内容，`choices[0].message.content` 类型上是 `str \| None` |
| 判空 `usage` 后 Pylance 仍报 `不是 "None" 的已知属性` | **类型收窄只作用于被判断的那个表达式**：`if usage is None: raise` 之后再写 `response.usage.xxx` 等于重新访问属性，类型又变回 `Optional` ⇒ 判空后必须用 `usage.xxx` |
| 写 `message.finish_reason` 报「属性未知」 | `finish_reason` 挂在 **`choices[i]`** 上，`message` 里没有；`message` 里只有 `content` / `refusal` / `role` / `tool_calls` |
| `or ""` 让三种失败长得一样 | 拒答（`refusal`）/ 只回工具调用 / 网关不给 —— 在终端上都只是「一行空白」；`finish_reason` 才能区分 |
| `completion_tokens_details` 取不到 | 它是 **Optional**（`prompt_tokens_details` 同样），别把 `reasoning_tokens` 当必填 |
