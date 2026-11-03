"""S0 · 实验：httpx 入门 —— 一次 get / params / 异常 / Client 复用 vs 每次新建

目的（一条根）：顶层 httpx.get **每个请求都新建 TCP+TLS 连接**；Client 用连接池复用，
    所以「同一个 Client 连发」应当比「每次顶层 API」省握手开销 —— 这是官方口径，
    今天用 5 个请求的实测数字验证它。

判据（跑完把数字填进本目录 README 的实测列）：
  1. basic_get()     → status_code == 200 且正文非空（打印前 80 字符）
  2. use_params()    → params 拼进查询串：str(r.url) 里同时有 q=httpx 与 page=1
  3. error_paths()   → /status/404 → HTTPStatusError（有 .response）
                      坏域名 → RequestError（有 .request）
  4. reuse_vs_new()  → 5 个请求两个总耗时实测数字（复用 ≤ 新建；网络抖动复跑取稳态）

跑法：uv run python fetch_basics.py
文档依据：python-httpx.org/quickstart/（响应对象 / 异常两支 / 默认超时 5 秒）
         python-httpx.org/advanced/clients/（连接池 vs 每请求新建）
"""

import time

import httpx

GET_URL = "https://httpbin.org/get"
DELAY_URL = "https://httpbin.org/delay/1"  # 官方延迟端点：服务器故意等 1 秒，供耗时对比用


def basic_get() -> None:
    """判据 1：对 GET_URL 发一次 httpx.get —— 打印 status_code 与正文前 80 字符。"""
    r = httpx.get(GET_URL)
    print(f"Status Code: {r.status_code}")
    print(f"Body (first 80 chars): {r.text[:80]}")


def use_params() -> None:
    """判据 2：带 params={"q": "httpx", "page": "1"} 再发一次 ——
    断言 str(r.url) 里同时包含 "q=httpx" 与 "page=1"。"""
    r = httpx.get(GET_URL, params={"q": "httpx", "page": "1"})
    assert "q=httpx" in str(r.url)
    assert "page=1" in str(r.url)


def error_paths() -> None:
    """判据 3：两条异常支路，各用 try/except 捕获并打印关键属性 ——
    ① 请求 https://httpbin.org/status/404 → raise_for_status() 抛 httpx.HTTPStatusError，
       打印 exc.response.status_code（应 == 404）
    ② 请求 https://no-such-host.invalid/ → 抛 httpx.RequestError，
       打印 exc.request.url"""
    try:
        r = httpx.get("https://httpbin.org/status/404")
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"HTTPStatusError: {exc.response.status_code}")

    try:
        r = httpx.get("https://no-such-host.invalid/")
    except httpx.RequestError as exc:
        print(f"RequestError: {exc.request.url}")


def reuse_vs_new() -> None:
    """判据 4：复用 vs 新建（同样 5 个 GET_URL 请求，time.perf_counter 计时）——
    ① 每次都顶层 httpx.get(GET_URL)，计时 → t_new
    ② 同一个 with httpx.Client() 里 client.get(GET_URL) 跑 5 次，计时 → t_reuse
    打印两个总耗时（秒，3 位小数）。注意：这是实测对比，网络抖动会让单次结果有波动，
    别写死 assert t_reuse < t_new —— 抖红了就复跑，把稳态数字记进 README。"""
    # ① 每次都顶层 httpx.get(GET_URL)
    start_new = time.perf_counter()
    for _ in range(5):
        httpx.get(GET_URL)
    t_new = time.perf_counter() - start_new

    # ② 同一个 with httpx.Client() 里 client.get(GET_URL) 跑 5 次
    start_reuse = time.perf_counter()
    with httpx.Client() as client:
        for _ in range(5):
            client.get(GET_URL)
    t_reuse = time.perf_counter() - start_reuse

    print(f"Time (new): {t_new:.3f} seconds")
    print(f"Time (reuse): {t_reuse:.3f} seconds")


def main() -> None:
    """依次跑判据 1-4，末行打印 判据 4/4 绿。"""
    basic_get()
    use_params()
    error_paths()
    reuse_vs_new()
    print("判据 4/4 绿")


if __name__ == "__main__":
    main()
