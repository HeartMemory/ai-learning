# 2026-10-29-llm-hello —— 模型 API 打通（过渡期闸门）

> 练习工程，不占交付物名额。**唯一目的：让第一次模型调用真的返回东西，并且看得见 token 与耗时。**

## 判据（三条全绿才算通）

| # | 判据 | 怎么看 |
|---|---|---|
| 1 | 正文**非空** | 终端打印出模型回答，不是报错、不是空串 |
| 2 | token 数**来自 `response.usage`** | 打印 `prompt_tokens` / `completion_tokens` / `total_tokens` 三个数 |
| 3 | **端到端耗时** | `time.perf_counter()` 前后差值，打印出来 |

⭐ **附加判据（S1 的地基）**：**换供应商 / 换模型只改 `.env` 三个变量，`ask.py` 一行不动** —— 做不到说明 provider 耦合进来了。

## 怎么跑

```powershell
cd practice/ai/2026/10/29-llm-hello

# 1) 复制配置模板，再把真值填进去（.env 不入库）
Copy-Item .env.example .env

# 2) 装依赖（首次）
uv sync

# 3) 跑
uv run python ask.py
```

`.env` 三个变量：

| 变量 | 说明 |
|---|---|
| `LLM_API_KEY` | 供应商控制台里拿的 key |
| `LLM_BASE_URL` | **OpenAI 兼容入口**，形如 `https://api.xxx.com/v1`（**带不带 `/v1` 按供应商文档**） |
| `LLM_MODEL` | 模型名，如供应商文档里的 `deepseek-chat` 之类 |

## 供应商参考（都走 OpenAI 兼容协议）

| 供应商 | BASE_URL 形态 | 备注 |
|---|---|---|
| **DeepSeek** | `https://api.deepseek.com/v1` | 便宜，国内直连，学生预算友好 |
| 通义 / 智谱 / Kimi / 豆包 | 各自控制台的 OpenAI 兼容入口 | 换用只改 BASE_URL |

⚠️ 海外（OpenAI / Anthropic）要代理，且国内支付不便 —— 起步阶段不必碰。

## 已知坑（写进笔记的那种）

| 现象 | 真因 |
|---|---|
| 报 401 / 认证失败 | key 粘错、带空格、或 `.env` 没被读到（本脚本用 `load_dotenv()` 读同目录） |
| 报 404 / `Not Found` | `BASE_URL` 少了或多了 `/v1` |
| 报连接超时但网络正常 | `BASE_URL` 指到了被墙的域名 |
| 改了 `.env` 不生效 | 变量名拼错（**必须**是 `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL`） |

## 实测数字（2026-10-29 跑通）

| 项 | 值 |
|---|---|
| 模型 | **`deepseek-flash`**（BASE_URL = `https://api.deepseek.com`） |
| prompt_tokens / completion_tokens / total_tokens | **38 / 58 / 96** |
| 端到端耗时 | **1254.49 ms** |
| 判据 1（正文非空） | ✅ |
| 判据 2（token 三数来自 `usage`） | ✅ |
| 判据 3（端到端耗时） | ✅ |

⚠️ **一个值得记的发现**：`usage.completion_tokens_details.reasoning_tokens = 38` ⇒ **该模型默认走思考模式**，58 个 completion token 里有 38 个是"想"掉的。这意味着「便宜 = 快 = 不用思考」的假设不成立，后面算成本要把 reasoning tokens 一起算。

## 待做（可选，不算闸门）

- [ ] **换模型一行验证**：把 `.env` 的 `LLM_MODEL` 改成 `deepseek-v4-pro`，跑一次，确认**代码一行不改**也能通 —— 这是 S1「provider 抽象层」的预热
- [ ] 把 `print("token 用量：", response.usage)` 改成打印三个独立字段（现在是一整串 `CompletionUsage(...)`，数字在里面但不好读）
