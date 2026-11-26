"""命令行入口：参数解析（机械部分已给好）+ `main()`（要你自己写）。

退出码约定：`0` 成功 ｜ `2` 配置缺失（`config.load_settings` 抛的）｜ `1` 其它异常（自己接）
           `3` 输出正文为空（模型拒绝回答 / 只返回空串 / 只返回 None）
"""
import argparse
import sys
from pathlib import Path
from typing import get_args

from openai import OpenAI

from .billing import bill, format_bill
from .chat import run_chat
from .config import PROJECT_ROOT, load_settings
from .llm import ThinkLevel, ask, ask_stream


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
    parser.add_argument(
        "--think",
        choices=list(get_args(ThinkLevel)),
        default=None,
        help="思考强度：off=关思考；low/high/max=开思考并指定强度；不传=厂商默认（本模型默认开、effort=high）",
    )

    return parser


def build_chat_parser() -> argparse.ArgumentParser:
    """`chat` 子命令的参数（机械部分，AI 代写）。"""
    parser = argparse.ArgumentParser(
        prog="llm_cli chat",
        description=(
            "多轮对话 + 上下文记账（L4 关卡 3）："
            "每轮报本轮 input / 本轮 output / 累计上下文"
        ),
    )
    parser.add_argument("--model", default=None, help="临时覆盖 .env 里的 LLM_MODEL")
    parser.add_argument(
        "--history",
        default=None,
        help="历史 JSON 路径（默认 <项目根>/chat_history.json，已 gitignore）",
    )
    return parser


def chat_main(argv: list[str]) -> int:
    """`chat` 子命令的胶水：读参数 → 建 client → 交给 `chat.run_chat()`（AI 代写）。"""
    args = build_chat_parser().parse_args(argv)
    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    history_path = Path(args.history) if args.history else PROJECT_ROOT / "chat_history.json"
    return run_chat(
        model=args.model or settings.model,
        client=client,
        history_path=history_path,
    )


def main(argv: list[str] | None = None) -> int:
    """跑一次真实调用，并把三个数打在终端上。

    判据（L4 关卡 0 ①）：终端同时出现 **输入 token / 输出 token / 耗时(ms)**，
    且 token 数来自平台返回的 `usage`，不是自己估的。

    流程（2026-11-19 定稿）：建 client → 调 `ask()` → **先打账单（三个数 + 结束原因）
    再判成败** —— 正文为空只代表这条命令失败，账已经花出去了，所以先打印后判定。
    退出码：`0` 有正文 ｜ `3` 正文为空（已记账）｜ `2` 参数/配置缺失（上游抛）｜ `1` 其它异常。
    """
    argv_list = list(sys.argv[1:] if argv is None else argv)
    if argv_list[:1] == ["chat"]:
        return chat_main(argv_list[1:])

    args = build_parser().parse_args(argv_list)
    settings = load_settings()
    print(f"思考: {args.think or '(未指定 = 厂商默认)'}")
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)

    if args.stream:
        answer = ask_stream(
            prompt=args.prompt,
            model=args.model or settings.model,
            client=client,
            think=args.think,
        )
        print()  # 收尾换行：边到边打印的最后一块不带 \n，不补的话账单会贴在正文末尾

        print(f"首 token(ms): {answer.first_token_ms}")
        print(f"耗时(ms): {answer.elapsed_ms}")
        print(f"块数: {answer.chunks}")
        print(f"输入 token: {answer.prompt_tokens}")
        print(f"输出 token: {answer.completion_tokens}")
        print(f"结束原因: {answer.finish_reason}")
        print(f"completed: {answer.completed}")

        # 流被打断 ⇒ 拿不到 usage 汇总帧，这笔钱**算不出来**（不许编 0 冒充当次免费）
        if answer.model is None or answer.prompt_tokens is None or answer.completion_tokens is None:
            print("费用: 无账单（流未走完 / 供应商没给 usage 或模型名）")
        else:
            print(f"其中思考 token: {answer.reasoning_tokens}")
            print(format_bill(bill(answer)))

        if answer.completed and answer.text:
            return 0
        return 3

    answer = ask(
        prompt=args.prompt,
        model=args.model or settings.model,
        client=client,
        think=args.think,
    )

    print(f"正文: {answer.text}")
    print(f"输入 token: {answer.prompt_tokens}")
    print(f"输出 token: {answer.completion_tokens}")
    print(f"其中思考 token: {answer.reasoning_tokens}")  # None = 供应商没给，不是 0
    print(f"耗时(ms): {answer.elapsed_ms}")
    print(f"结束原因: {answer.finish_reason}")
    print(f"模型: {answer.model}")
    print(format_bill(bill(answer)))  # 缓存命中数也在这一行里

    if not answer.text:
        return 3
    return 0
