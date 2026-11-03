"""S0 · 实验：并发抓取 —— 串行 vs asyncio.gather（S0 交付物预演）

目的（一条根）：await 等待时**不占 CPU**，所以 5 个「服务器故意等 1 秒」的请求
    串行 ≈ 5 × 1 s、gather 并发 ≈ 1 s；并发时所有请求**共享同一个 AsyncClient**
    （官方：别在热循环里反复实例化 client，会破坏连接池收益）。

判据（跑完把数字填进本目录 README 的实测列；接上文件，本文件覆盖 5-8）：
  5. fetch_one()       → dict 四键：url / status_code / elapsed_s / body_len
  6. serial vs gather  → 串行 ≈ N × 单个、并发 ≈ 单个（两个总耗时实测数字）
  7. 两条路的结果       → 条数 == 5 且每条 status_code == 200
  8. JSON 落盘读回      → json.load 读回 == 内存对象

跑法：uv run python fetch_concurrent.py
文档依据：python-httpx.org/async/（AsyncClient / 共享实例警告）
         docs.python.org/3/library/asyncio-task.html（gather，10-31 已实测）
"""

import asyncio
import json
import time

import httpx

DELAY_URL = "https://httpbin.org/delay/1"
URLS = [DELAY_URL] * 5
JSON_PATH = "fetch_results.json"


async def fetch_one(client: httpx.AsyncClient, url: str) -> dict:
    """判据 5：await client.get(url) —— 返回 {"url", "status_code", "elapsed_s", "body_len"}。
    elapsed_s 用 time.perf_counter 包住这一次请求；body_len 取 len(r.content)。"""
    start = time.perf_counter()
    r = await client.get(url)
    elapsed_s = time.perf_counter() - start
    return {
        "url": url,
        "status_code": r.status_code,
        "elapsed_s": elapsed_s,
        "body_len": len(r.content),
    }


async def serial_fetch(client: httpx.AsyncClient, urls: list[str]) -> list[dict]:
    """判据 6 串行：for 循环逐个 await fetch_one —— 总耗时 ≈ N × 单个。"""
    results = []
    for url in urls:
        result = await fetch_one(client, url)
        results.append(result)
    return results


async def concurrent_fetch(client: httpx.AsyncClient, urls: list[str]) -> list[dict]:
    """判据 6 并发：asyncio.gather(*(...)) 把 5 个协程一次性交给事件循环 —— 总耗时 ≈ 单个。
    注意 client 是同一个实例（官方：别在热循环里反复实例化 client）。"""
    tasks = [fetch_one(client, url) for url in urls]
    return await asyncio.gather(*tasks)


async def main() -> None:
    """依次跑判据 5-8，末行打印 判据 8/8 绿。"""
    async with httpx.AsyncClient() as client:
        start = time.perf_counter()
        serial_results = await serial_fetch(client, URLS)
        t_serial = time.perf_counter() - start

        start = time.perf_counter()
        concurrent_results = await concurrent_fetch(client, URLS)
        t_concurrent = time.perf_counter() - start

    print(f"serial:     {t_serial:.3f} s")
    print(f"concurrent: {t_concurrent:.3f} s")

    # TODO(你写) 判据 7：断言两条结果 len == len(URLS)，且每条 status_code == 200
    assert len(serial_results) == len(URLS)
    assert len(concurrent_results) == len(URLS)
    assert all(result["status_code"] == 200 for result in serial_results)
    assert all(result["status_code"] == 200 for result in concurrent_results)
    # TODO(你写) 判据 8：并发结果写入 JSON_PATH（json.dump），再 json.load 读回，assert == 并发结果
    with open(JSON_PATH, "w") as f:
        json.dump(concurrent_results, f)
    with open(JSON_PATH) as f:
        loaded_results = json.load(f)
    assert loaded_results == concurrent_results
    print("判据 8/8 绿")


if __name__ == "__main__":
    asyncio.run(main())
