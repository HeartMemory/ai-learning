"""`uv run python -m llm_cli` 的入口（纯胶水，AI 代写）。"""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
