"""环境自检：打印解释器与实际安装的依赖版本，确认 uv 工程可用。

目的 / 判据：
    1. `uv run python check_env.py` 能打印出 python 路径指向**本工程**的 `.venv`
       （不是全局解释器）—— 这是"依赖隔离生效"的硬证据。
    2. 依赖列表无 `MISSING`，说明 `uv sync` 把 pyproject 里声明的包都装上了。
"""

import platform
import sys
from importlib.metadata import PackageNotFoundError, version

DISTRIBUTIONS = (
    "httpx",
    "openai",
    "pydantic",
    "python-dotenv",
    "rich",
    "tenacity",
    "tiktoken",
    "pytest",
    "ruff",
)


def dist_version(dist_name: str) -> str:
    """返回已安装发行包的版本；未安装返回 MISSING（不抛异常）。"""
    try:
        return version(dist_name)
    except PackageNotFoundError:
        return "MISSING"


def main() -> None:
    print(f"python  : {sys.version.split()[0]}")
    print(f"exe     : {sys.executable}")
    print(f"prefix  : {sys.prefix}")
    print(f"platform: {platform.system()} {platform.release()}")
    print()
    for dist_name in DISTRIBUTIONS:
        print(f"{dist_name:<12}: {dist_version(dist_name)}")


if __name__ == "__main__":
    main()
