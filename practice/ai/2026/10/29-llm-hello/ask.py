"""模型 API 打通 —— 过渡期闸门（2026-10-29）

目的：走通一次最小的模型调用闭环
      读 .env → 建 client → 发一次 chat.completions → 打印正文 / token 用量 / 耗时

判据（三条全绿才算"通了"）：
  1. 正文非空（不是报错、不是空串）
  2. 打印的 token 数来自 **response.usage**，不是自己估的（prompt / completion / total 三个都要）
  3. 打印端到端耗时（time.perf_counter 的差值）

铁律：换供应商 / 换模型时 **只改 .env 三个变量，本文件一行不动** ——
      这就是后续 S1 封 provider 层的地基。

运行：uv run python ask.py
"""

import io
import os
import sys
import time

# 先读同目录的 .env，再把变量取出来
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("LLM_API_KEY", "").strip()
BASE_URL = os.getenv("LLM_BASE_URL", "").strip()
MODEL = os.getenv("LLM_MODEL", "").strip()


def ask(prompt: str) -> None:
    """发一次请求，打印正文 + token 用量 + 耗时。"""
    # TODO ① 建客户端：用 API_KEY 与 BASE_URL 造一个 OpenAI 实例
    #        提示：from openai import OpenAI（**不是**老教程里的 openai.api_base 那套写法）
    from openai import OpenAI
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    # TODO ② 发一次 chat.completions：model 用 MODEL，messages 至少一条 role="user"
    #        提示：返回对象的正文在 response.choices[0].message.content
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    # TODO ③ 打印三样东西：
    #        · 正文
    #        · response.usage 里的 prompt_tokens / completion_tokens / total_tokens
    #        · 端到端耗时（time.perf_counter 前后差值，自己选秒还是毫秒）
    print("正文：", response.choices[0].message.content)
    print("token 用量：", response.usage)
    print(f"耗时：{elapsed_ms:.2f} ms")


if __name__ == "__main__":
    # Windows 控制台默认 GBK，中文输出经管道会乱码 ⇒ 强制 UTF-8
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")

    # 缺配置就别往下跑，否则报错长得像"网络问题"，其实是 .env 没填
    cfg = {"LLM_API_KEY": API_KEY, "LLM_BASE_URL": BASE_URL, "LLM_MODEL": MODEL}
    missing = [k for k, v in cfg.items() if not v]
    if missing:
        raise SystemExit(
            f"缺配置：{', '.join(missing)} —— 把 .env.example 复制成 .env 填好再跑"
            "（Copy-Item .env.example .env）"
        )

    ask("用一句话解释：什么是二分查找？")
