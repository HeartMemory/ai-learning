"""命令行入口：参数解析（机械部分已给好）+ `main()`（要你自己写）。

退出码约定：`0` 成功 ｜ `2` 配置缺失（`config.load_settings` 抛的）｜ `1` 其它异常（自己接）
           `3` 输出正文为空（模型拒绝回答 / 只返回空串 / 只返回 None）
"""

import argparse

from openai import OpenAI

from .config import load_settings
from .llm import ask, ask_stream


def build_parser() -> argparse.ArgumentParser:
    """`prompt` 是位置参数；`--model` 用来临时覆盖 `.env` 里的模型名。"""
    parser = argparse.ArgumentParser(
        prog="llm_cli",
        description="最小 LLM 命令行（L4 关卡 0：打通与记账 · 关卡 1：流式）",
    )
    parser.add_argument("prompt", help="要问模型的一句话（带空格请用引号包住）")
    parser.add_argument("--model", default=None, help="临时覆盖 .env 里的 LLM_MODEL")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="走流式：报「首 token(ms)」与「耗时(ms)」，正文边到边打印",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """跑一次真实调用，并把三个数打在终端上。

    判据（L4 关卡 0 ①）：终端同时出现 **输入 token / 输出 token / 耗时(ms)**，
    且 token 数来自平台返回的 `usage`，不是自己估的。

    流程（2026-11-19 定稿）：建 client → 调 `ask()` → **先打账单（三个数 + 结束原因）
    再判成败** —— 正文为空只代表这条命令失败，账已经花出去了，所以先打印后判定。
    退出码：`0` 有正文 ｜ `3` 正文为空（已记账）｜ `2` 参数/配置缺失（上游抛）｜ `1` 其它异常。
    """
    args = build_parser().parse_args(argv)
    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    if args.stream:
        answer = ask_stream(prompt=args.prompt, model=args.model or settings.model, client=client)
        print()  # 收尾换行：边到边打印的最后一块不带 \n，不补的话账单会贴在正文末尾

        print(f"首 token(ms): {answer.first_token_ms}")
        print(f"耗时(ms): {answer.elapsed_ms}")
        print(f"块数: {answer.chunks}")
        print(f"输入 token: {answer.prompt_tokens}")
        print(f"输出 token: {answer.completion_tokens}")
        print(f"结束原因: {answer.finish_reason}")
        print(f"completed: {answer.completed}")

        if answer.completed and answer.text:
            return 0
        return 3

    answer = ask(prompt=args.prompt, model=args.model or settings.model, client=client)

    print(f"正文: {answer.text}")
    print(f"输入 token: {answer.prompt_tokens}")
    print(f"输出 token: {answer.completion_tokens}")
    print(f"耗时(ms): {answer.elapsed_ms}")
    print(f"结束原因: {answer.finish_reason}")

    if not answer.text:
        return 3
    return 0
