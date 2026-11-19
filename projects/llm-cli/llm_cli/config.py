"""读配置：`.env` → `Settings`（纯胶水，AI 代写，不是今天的练习）。

为什么单开一个文件：换供应商 / 换模型只改 `.env` 三个变量，代码一行不动。
两条硬性约束：
  1. key 只从环境读，**代码里不出现字面量**（判据 ② 的第三处自查就查这个）
  2. `Settings` 的 repr 不打印 key —— 否则随便一个报错栈就把 key 抖出去
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# 项目根（本文件在 <根>/llm_cli/config.py）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

REQUIRED = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")


@dataclass(frozen=True, slots=True)
class Settings:
    """一次调用需要的三样东西。`api_key` 用 `repr=False` 挡住日志泄漏。"""

    api_key: str = field(repr=False)
    base_url: str
    model: str


def load_settings(env_file: Path = ENV_FILE) -> Settings:
    """读 `.env` 取出三变量；缺哪项就报哪项（退出码 2 = 配置错误，与 argparse 用法错误一致）。"""
    load_dotenv(env_file)
    values = {key: os.environ.get(key, "").strip() for key in REQUIRED}

    missing = [key for key, value in values.items() if not value]
    if missing:
        print(
            f"缺配置：{', '.join(missing)} —— "
            f"把 .env.example 复制成 .env 填好再跑（Copy-Item .env.example .env）",
            file=sys.stderr,
        )
        raise SystemExit(2)

    return Settings(
        api_key=values["LLM_API_KEY"],
        base_url=values["LLM_BASE_URL"].rstrip("/"),
        model=values["LLM_MODEL"],
    )
