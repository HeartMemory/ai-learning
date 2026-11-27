"""L5 关 0 前置探针：确认当前模型是否支持工具调用（tools）。

AI 代写（探针 / 判据脚本，不是练习）。一次调用，成本 < ¥0.001。

为什么先测：官方兼容文档只声明 `tools` 支持，但**没写明 DeepSeek 系列是否给
`tool_calls`**；若模型不支持，后面整套 mini-agent 无从谈起。

判据（看输出即可）：
  绿 = finish_reason == 'tool_calls' 且 tool_calls 里有 'get_time'
  红 = 打印服务端错误原文（据此决定换模型 / 换供应商，而不是改代码）
"""

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionToolParam

# 用 __file__ 定位，不依赖当前工作目录（从仓库根或子目录跑都能读到）
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# 显式标注类型：裸字面量会被推导成 list[dict[str, Unknown]]，Pylance 与 SDK 的
# ChatCompletionToolUnionParam 对不上就报 reportArgumentType（只是类型告警，运行没问题）
TOOLS: list[ChatCompletionToolParam] = [
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "查询指定城市的当前时间",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "城市名"}},
                "required": ["city"],
            },
        },
    }
]


def main() -> int:
    """发一次带 tools 的请求，把关键字段原样打印。"""
    client = OpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
    )

    try:
        resp = client.chat.completions.create(
            model=os.environ["LLM_MODEL"],
            messages=[{"role": "user", "content": "现在北京几点？用工具查一下。"}],
            tools=TOOLS,
        )
    except Exception as exc:  # 探针的意义就是把服务端报错原文打出来
        print(f"FAIL: {type(exc).__name__}")
        print(exc)
        return 1

    choice = resp.choices[0]
    message = choice.message
    print(f"model         = {resp.model}")
    print(f"finish_reason = {choice.finish_reason!r}")
    print(f"content       = {message.content!r}")
    calls = [tc.model_dump() for tc in (message.tool_calls or [])]
    print(f"tool_calls    = {json.dumps(calls, ensure_ascii=False)}")
    if resp.usage:
        print(
            f"usage         = prompt {resp.usage.prompt_tokens}"
            f" / completion {resp.usage.completion_tokens}"
            f" / total {resp.usage.total_tokens}"
        )
    return 0 if message.tool_calls else 2


if __name__ == "__main__":
    sys.exit(main())
