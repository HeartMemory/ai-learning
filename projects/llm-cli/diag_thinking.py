"""诊断：**关掉思考模式**后再比一次「流式拼接 vs 非流式」，只花 2 次调用。

跑法：`uv run python diag_thinking.py`

为什么需要它（判据 ② 的卡点）：
  测试所用模型默认开思考模式，而供应商文档明示思考模式**不支持**
  `temperature` / `presence_penalty` / `frequency_penalty`（收下不报错、也不生效）。
  ⇒ 两次请求都是不可控采样，判据 ② 的「逐字符相等」变成「不可判定」。
  ⇒ 把思考关掉，温度才真的生效，这条判据才有判定条件。

依据（2026-11-20 查证 · 官方文档）：
  · 开关：`{"thinking": {"type": "enabled" | "disabled"}}`，默认 **enabled**（effort 默认 high）
  · `thinking` 不是 OpenAI SDK 的原生字段 ⇒ 必须走 `extra_body`
  · 非思考模式下 `top_p` 恒为 1.0（该模型），温度正常生效
"""

import io
import sys
from contextlib import redirect_stdout

from openai import OpenAI

from bench import PROMPT, compare_text, pin_request_defaults
from llm_cli.config import load_settings
from llm_cli.llm import ask, ask_stream


def main() -> int:
    """纯胶水：关思考 + 钉温度 → 1 次流式、1 次非流式 → 打数 + 比对。

    prompt 可用命令行覆盖（带空格请用引号）：`uv run python diag_thinking.py "从 1 数到 30"`
    覆盖的用途 = 低熵任务复核：高熵回答在 temp=0 下仍会被供应商采成不同文本，
    低熵任务才能把「拼接无损」与「生成不确定」这两件事分开。
    """
    prompt = sys.argv[1] if len(sys.argv) > 1 else PROMPT
    settings = load_settings()
    client = pin_request_defaults(
        OpenAI(api_key=settings.api_key, base_url=settings.base_url),
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )

    sink = io.StringIO()
    with redirect_stdout(sink):
        streamed = ask_stream(prompt, model=settings.model, client=client)
        plain = ask(prompt, model=settings.model, client=client)

    print(f"model : {settings.model}（thinking=disabled, temperature=0.0）")
    print(f"prompt: {prompt}")
    print(
        f"流式  : 首 token {streamed.first_token_ms:.2f} ms ｜ 耗时 {streamed.elapsed_ms:.2f} ms"
        f" ｜ 块数 {streamed.chunks}"
        f" ｜ token {streamed.prompt_tokens}/{streamed.completion_tokens}"
        f" ｜ {streamed.finish_reason}"
    )
    print(
        f"非流式: 耗时 {plain.elapsed_ms:.2f} ms"
        f" ｜ token {plain.prompt_tokens}/{plain.completion_tokens}"
        f" ｜ {plain.finish_reason}"
    )

    equal, index, delta = compare_text(streamed.text, plain.text)
    print(f"逐字符相等: {equal} ｜ 首次分歧下标: {index} ｜ 长度差: {delta}")
    print(f"流式前 40 字  : {streamed.text[:40]!r}")
    print(f"非流式前 40 字: {plain.text[:40]!r}")
    if not equal:
        print(f"  流式[{index}:{index + 20}]  : {streamed.text[index:index + 20]!r}")
        print(f"  非流式[{index}:{index + 20}]: {plain.text[index:index + 20]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
