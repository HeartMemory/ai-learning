# 2026-11-03-httpx —— httpx 入门：Client 复用 vs 每次新建 + 串行 vs gather 并发抓取（S0 交付物地基）

> 目的：把「顶层 API 每请求新建连接 / Client 用连接池复用」这条官方口径，落到可测的点上；再接 async 那章 —— 串行 vs `gather` 抓 N 个延迟页面，写出两个实测耗时数字；结果落 JSON 并读回对账（S0 交付物「并发抓取 + 存 JSON」的预演）。
> 本目录三个文件：`fetch_basics.py`（判据 1-4）+ `fetch_concurrent.py`（判据 5-8）+ `pyproject.toml`（只装 httpx）。
> 跑法：`uv sync` → `uv run python fetch_basics.py` / `uv run python fetch_concurrent.py`
> 测试靶：`httpbin.org`（官方示例同款；`/delay/1` 服务器故意等 1 秒，专供耗时对比）

## 判据（跑通才算过，实测数字填右侧）

### `fetch_basics.py`

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `httpx.get(GET_URL)` | `status_code == 200` 且正文非空 | ✅ `200`；正文 = httpbin 的请求回声 JSON |
| 2 | `params={"q": "httpx", "page": "1"}` | `str(r.url)` 里同时看到 `q=httpx` 与 `page=1` | ✅ 两条 assert 均过 |
| 3 | `/status/404` + 坏域名 | 404 → `httpx.HTTPStatusError`（有 `.response`）；坏域名 → `httpx.RequestError`（有 `.request`） | ✅ `HTTPStatusError: 404`；`RequestError: https://no-such-host.invalid/` |
| 4 | 5 个请求：每次 `httpx.get` vs 复用 `Client` | 两个总耗时实测数字；复用 ≤ 新建（网络抖动可复跑取稳态） | ✅ 新建 **3.431 s**（0.686/请求）vs 复用 **1.573 s**（0.315/请求）—— 差额 ≈ **0.37 s/请求握手费**，复用快 **2.2×** |

### `fetch_concurrent.py`（覆盖 5-8）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 5 | `fetch_one` | 返回 dict 四键：url / status_code / elapsed_s / body_len | ✅ 四键 dict |
| 6 | 串行 vs `gather` 抓 5 个 `/delay/1` | 串行 ≈ **N × 单个**、并发 ≈ **单个**（两个实测数字） | ✅ 串行 **7.036 s** vs 并发 **1.723 s**（**4.1×**）；串行 = 5 × (1 s 等 + 0.4 s 握手)，并发 = 等待与握手全部重叠 |
| 7 | 两条路的结果 | 条数都 == 5，每条 `status_code == 200` | ✅ 两条 assert 均过 |
| 8 | JSON 落盘读回 | `json.load` 读回 == 内存对象 | ✅ `assert loaded == results` 过 |

**实跑记录（2026-11-03）**：`uv run python fetch_basics.py` → 末行 **`判据 4/4 绿`**；`uv run python fetch_concurrent.py` → `serial: 7.036 s` / `concurrent: 1.723 s` / **`判据 8/8 绿`**。`uv sync` → **7 包**（httpx 0.28.1，124 ms）。

## 官方依据（2026-11-03 查证 · 三页互证）

- **QuickStart**：`httpx.get` / `params` 自动拼查询串 / `r.status_code`、`r.text`、`r.content`、`r.json()` / `raise_for_status()` 对非 2xx 抛 `HTTPStatusError`、对 2xx **不抛且返回 Response 可链式** / 异常两支 `RequestError`（带 `.request`）与 `HTTPStatusError`（带 `.request` + `.response`）、共同基类 `HTTPError` / **默认超时 5 秒**（网络无活动），`timeout=None` 禁用。
- **Advanced › Clients**：顶层 API **每个请求都建立新连接**（连接不复用）；`Client` 用 **HTTP 连接池**复用底层 TCP 连接 —— 降请求间延迟、降 CPU 与网络往返、降拥塞；额外功能 = cookie 持久化、统一配置、代理、HTTP/2；对应 `requests.Session()`；推荐上下文管理器（`with httpx.Client() as client:`）。
- **Async Support**：`async with httpx.AsyncClient() as client` + `await client.get(...)`；同步/异步 API 对应表（`close` → `aclose`、`iter_bytes` → `aiter_bytes` 等）；官方明确警告：**为最大程度获得连接池收益，别在热循环里反复实例化 client**，应传递单个作用域内的 client 实例；并发请求**共享同一个 client**。

## 怎么跑

```powershell
cd practice/python/2026/11/03-httpx
uv sync
uv run python fetch_basics.py      # 判据 1-4
uv run python fetch_concurrent.py  # 判据 5-8
uvx ruff check . && uvx ruff format .
```

## 踩过的坑

- **`open` 的 `mode="r"` 是默认值 ⇒ Pylance 报灰色 hint「Unnecessary mode argument」**：不是错误（实参 = 默认值 = 冗余），读侧删掉 `"r"` 即清零；写侧 `"w"` 非默认，不提示。顺带钉了检查器三级：红 error 必修 / 黄 warning 该修 / 灰 hint 观点性建议。
- **`await` 的通行证是 `async def`**：普通 `def` 的函数体里写 `await` 直接 `SyntaxError`（跑都跑不起来）；签名与注释自相矛盾时先怀疑签名。
- **计时对比别写死 assert**：单样本是随机变量（一个请求抽风就把正确程序判红），且先跑的批付 DNS 冷启动费（量级几十 ms，在秒级账单里是零头）⇒ 只记数字、复跑取稳态（回访 #42）。
- **读数要拆账别只看加速比**：串行 7.036 = 5 × (1 s 服务器等待 + 0.4 s 握手)；并发 1.723 ≈ 单个请求的完整成本 —— 两个实验测的是账单上**不同的行**（判据 4 省握手费 / 判据 6 省等待重叠）。
- **`uvx ruff format` 又补了顶层函数间空行**（`E302` 不在 `check` 默认规则集，`check` 绿 ≠ 格式合规 —— 第 4 次踩同一坑，两条命令必须都跑）。
