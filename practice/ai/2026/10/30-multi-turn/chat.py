"""练习：多轮对话 —— 用今天学的可变性知识维护 messages

目的：把 S0 第 3 章（改对象 vs 重新绑定）扔进一个真实会炸的地方：聊天历史。

判据（跑通才算过）：
  1. 每轮结束后 `len(messages)` 恰好 +2（一问一答）
  2. 历史**连续增长**，不是每轮被重置 —— 也就是 ask() 没有把调用方的 list 换掉
  3. `prompt_tokens` **逐轮变大**（说明历史真的被带上了，模型不是每轮从零开始）
  4. 第 2 轮问「我刚才问的是什么」，模型能**复述第 1 轮的问题** ← 多轮生效的硬证据
  5. 3 轮结束后共 6 条消息，user 条数 = 3（历史没串、没重复）

跑法：
  Copy-Item .env.example .env   # 填三个值（可从 llm-hello 那份直接复制）
  uv sync
  uv run python chat.py
"""

import os
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI
from openai.types import CompletionUsage

# ── messages 的类型：暂时这样写（2026-10-30 定） ─────────────────────────────
# openai SDK 用 TypedDict 描述 messages，最严格的写法是
#     list[ChatCompletionMessageParam] + 在构造单条消息处 cast(...)
# 但 Pyright 对「联合 TypedDict 的字面量」推断经常失败，要补好几次 cast 才能过。
#
# 练习阶段用 Any（不是"掩盖错误"，而是"这一层不在今天练"）：
#   · 我们自己代码里的真错照样会被检查出来（Any 只影响这一层容器）
#   · 不在第三方 SDK 的类型细节上耗时间
# → S0-9 学完 pydantic 后改成 `class Message(BaseModel)` + `model_dump()`，
#   那时类型全程受控，连 cast 也不需要。
# ────────────────────────────────────────────────────────────────────────────

load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ["LLM_BASE_URL"],
)
MODEL = os.environ["LLM_MODEL"]


def add_message(messages: list[Any], role: str, content: str) -> None:
    """把一条消息追加进历史。

    判据：调用后 len +1，且 id(messages) **不变**（原地追加，不是重新绑定）。
    """
    messages.append({"role": role, "content": content})


def ask(messages: list[Any], question: str) -> CompletionUsage:
    """一轮完整对话：加 user → 调模型 → 加 assistant → 返回 usage。

    判据：
      1. 返回后 len(messages) 恰好 +2
      2. 最后两条的 role 依次是 "user"、"assistant"
      3. 返回的 usage.total_tokens 逐轮变大
    """
    add_message(messages, "user", question)

    # ⭐ 关键：把整个历史递进去（单轮是现造一个 list，多轮是把这个 list 递过去）
    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
    )

    content = response.choices[0].message.content
    add_message(messages, "assistant", content or "")

    usage = response.usage
    assert usage is not None
    return usage


if __name__ == "__main__":
    messages: list[Any] = []
    questions = [
        "1+1 等于几？只回一个数字，不要任何解释。",
        "我刚才问的是什么？原样复述我的问题。",
        "那 2+2 呢？同样只回一个数字。",
    ]

    for i, q in enumerate(questions, 1):
        t0 = time.perf_counter()
        usage = ask(messages, q)
        ms = (time.perf_counter() - t0) * 1000
        print(
            f"[{i}] 消息数={len(messages):2d}  prompt={usage.prompt_tokens:4d}  "
            f"completion={usage.completion_tokens:3d}  "
            f"total={usage.total_tokens:4d}  {ms:8.2f} ms"
        )
        print("     答：", (messages[-1]["content"] or "").strip()[:60])

    print("\n最终消息数：", len(messages), "（期望 6）")
    print("user 条数：", sum(1 for m in messages if m["role"] == "user"), "（期望 3）")
