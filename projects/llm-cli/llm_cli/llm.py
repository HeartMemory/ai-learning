"""调用层：一次 `chat.completions` + 把用量与耗时记下来。

⭐ **今天的练习就在 `ask()` 里** —— 方法体自己写。

判据（L4 关卡 0 ①，三条同时成立才算通）：
  1. 正文**非空**（不是报错、不是空串、不是 None）
  2. 三个 token 数**取自 `response.usage`**，不是自己数字数
  3. 耗时 = `time.perf_counter()` 的前后差值，单位毫秒

镜子：`practice/ai/2026/10/29-llm-hello/ask.py`（同族写法，但那边是一把梭；这边要拆成
「调用 → 记账 → 返回结构」，因为后面关卡 1（流式）/ 关卡 3（成本）都要复用这笔账）。
"""

import time
from dataclasses import dataclass

from openai import OpenAI


@dataclass(frozen=True, slots=True)
class Answer:
    """一次调用的产出：正文 + 三个 token 数 + 端到端耗时（ms）。"""

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_ms: float
    finish_reason: str


def ask(prompt: str, *, model: str, client: OpenAI) -> Answer:
    """发一次请求，返回 `Answer`。

    实现要点（2026-11-19 定稿）：
      · 计时包住**整次请求**（起点在 create 之前，终点在它返回之后）
      · 正文可能为 `None`（拒答 / 只回工具调用 / 网关不给）⇒ `or ""` 兜底，**不抛错**：
        这些情况下 token 照扣，抛错会把账单一起扔掉
      · 三个 token 数取自 `usage`（`usage is None` 才抛 —— 那时真无法记账）
      · `finish_reason` 在 `choices[i]` 上，**不在 `message` 上**
    """
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not response.choices:
        raise RuntimeError("供应商返回了空 choices")
    message = response.choices[0].message
    usage = response.usage
    if usage is None:
        raise RuntimeError("供应商没返回 usage，无法记账（判据 ① 要平台数字）")
    text = message.content or ""

    return Answer(
        text=text,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        elapsed_ms=elapsed_ms,
        finish_reason=response.choices[0].finish_reason,
    )
