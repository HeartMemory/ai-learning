# 2026-10-28 · Python 环境基线（S0 前置）

> 目的：把 AI 主线的 Python 环境一次配好，并留下**可核验的判据**。
> 本目录同时是 S0 的脚手架样板 —— 后续练习照抄 `pyproject.toml` + 跑 `uv sync` 即可。

## 判据（全部实测，非"应该能跑"）

| # | 判据 | 命令 | 实测结果 |
|---|---|---|---|
| 1 | uv 能解析并安装依赖 | `uv sync` | 解析 **20 个包 / 840 ms**；装好 **19 个包**，配镜像后总耗时 **5.4 s** |
| 2 | 解释器指向**本工程**的 `.venv`（依赖隔离生效） | `uv run python check_env.py` | `exe` = `...\2026-10-28-python-env\.venv\Scripts\python.exe` |
| 3 | 静态检查全绿 | `uv run ruff check .` | `All checks passed!`（退出码 0） |
| 4 | 关键包真的装上 | `uv run python check_env.py` | httpx **0.28.1** / pydantic **2.14.1** / python-dotenv **1.2.4** / pytest **9.1.1** / ruff **0.17.3** |
| 5 | S0/S1 阶段依赖全装齐 | `uv pip list` | **openai 3.40.0** / **tiktoken 0.14.0** / **rich 15.0.0** / **tenacity 9.2.1** / ipykernel 7.4.0；`site-packages` 共 **117 个目录** |

## 环境

| 项 | 值 |
|---|---|
| Python | **3.14.8**（**2026-10-29 由 3.13.7 升级**；`pyproject.toml` 要求 `>=3.14`；`.python-version` 锁 3.14） |
| uv | **0.12.23** |
| OS | Windows 11 |
| 下载源 | 用户级 `%APPDATA%\uv\uv.toml`（阿里云镜像；**在仓库外，不进 git**） |
| 依赖（运行） | httpx · pydantic · python-dotenv · **openai** · **tiktoken** · **rich** · **tenacity** |
| 依赖（开发） | pytest · ruff · ipykernel |

## 怎么跑

```powershell
uv sync                      # 建 .venv + 按 uv.lock 装依赖
uv run python check_env.py   # 环境自检：打印解释器与各包版本
uv run ruff check .          # 静态检查
```

## 3.14 升级记录（2026-10-29）

| 项 | 变更前 | 变更后 |
|---|---|---|
| 解释器 | 3.13.7 | **3.14.8**（`uv python list` 里最新的**正式版**；3.15 只有 `rc3`，不上） |
| `.python-version` | `3.13` | `3.14` |
| `requires-python` | `>=3.12` | `>=3.14` |
| ruff `target-version` | `py312` | `py314`（ruff 0.17.3 已支持） |
| `from __future__ import annotations` | 有 | **删**（3.14 起注解默认延迟求值，该写法已 deprecated） |

**验证（实测）**：`uv run python -V` = **3.14.8**；`check_env.py` 9 个包**零 MISSING**；`ruff check` **全绿**（`py314` 目标可用）；`check_llm.py` 接入探针 **HTTP 200**。

⚠️ **本机两套解释器并存，别混**：

- **uv 自管**（`%APPDATA%\uv\python\`，本工程用这套）：`uv run python` 走这里
- **系统 `python`**：2026-10-29 当日晚**已另外补装 3.14.8** 到 `%LOCALAPPDATA%\Programs\Python\Python314`（PATH 最前，带成对的 `pip`）

⇒ **工程内一律 `uv run python`**；裸 `python` 现在也是 3.14，但它**看不到本工程 `.venv` 里装的包**，别用它跑本工程的脚本。

## 编辑器（CodeBuddy / VS Code）

**扩展**（用户级装一次，换机器重装）：

| 扩展 | 作用 |
|---|---|
| `ms-python.python` + `ms-python.vscode-pylance` + `ms-python.debugpy` | 语言支持 / 类型检查 / 调试 |
| `ms-python.vscode-python-envs` | 环境面板（直接选 `.venv`） |
| **`charliermarsh.ruff`** | **用工程里的 ruff 实时告警 + 保存时格式化**（`importStrategy = fromEnvironment` ⇒ 版本与 `.venv` 一致） |
| `tamasfe.even-better-toml` | `pyproject.toml` / `uv.toml` 的 schema 校验 |
| `usernamehw.errorlens` | 告警直接显示在行尾 |
| `ms-toolsai.jupyter` | 跑 notebook（学底层原理时用） |

**关键 settings（用户级，改前先备份）**：

```jsonc
"[python]": {
  "editor.defaultFormatter": "charliermarsh.ruff",
  "editor.formatOnSave": true,
  "editor.codeActionsOnSave": {
    "source.fixAll.ruff": "explicit",
    "source.organizeImports.ruff": "explicit"
  }
},
"ruff.importStrategy": "fromEnvironment",
"python.analysis.typeCheckingMode": "basic"
```

⚠️ **打开工程后必须手动选一次解释器**（`Ctrl+Shift+P` → `Python: Select Interpreter` → 选本目录的 `.venv\Scripts\python.exe`），否则编辑器不知道包在哪，会满屏黄波浪线。

⚠️ **不要装** `flake8` / `pylint` / `black` / `isort`：与 ruff 规则重叠，会两边打架。

## 踩过的坑（本次真实遇到）

1. **`uv sync` 卡在下载进度** —— 不是解析问题：`uv.lock` 早已生成（346 行 / 50 KB），而 `.venv\Lib\site-packages` 里 **0 个包** ⇒ 卡在**下载 wheel**。配国内镜像后同一份 lock **5.4 秒装完**。
   - **判据**：`uv.lock` 有内容 + `site-packages` 为空 = 卡下载，去治镜像而不是治依赖。
2. **`--index-url` 已废弃** ⇒ 用 `--default-index`（环境变量 `UV_DEFAULT_INDEX`）；配置文件里用 `[[index]]` + `default = true`。
3. **`RUF002` 误伤中文 docstring** —— 全角冒号 / 括号被判成"易混淆字符"，中文项目需在 `[tool.ruff.lint]` 里 ignore `RUF001`/`RUF002`/`RUF003`（本项目已配）。
4. **hardlink 警告**（缓存与工程不在同一文件系统）→ uv 自动退回 copy，只影响速度，不影响结果。

## `.env`

S1 起才需要：`Copy-Item .env.example .env` 再填值。`.env` 已被 `.gitignore` 排除，**永不入库**；仓库只留 `.env.example`（只有变量名，没有值）。
