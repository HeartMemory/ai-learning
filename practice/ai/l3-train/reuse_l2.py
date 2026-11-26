"""L3 · 训练工程 —— 复用 L2 关 4 的数据与模型（不复制代码）

为什么复用而不是复制：
  L3 的「单变量原则」要求**语料 / 词表 / 模型规模 / 步数 / tokenizer 与 L2 关 4 完全一致**。
  复制一份实现就等于给「不一致」留了口子（改一处忘一处 = 单变量失效，且从数字上看不出来）。
  ⇒ 这里用 sys.path 把 `../l2-nanogpt` 挂进来，import 的就是 L2 跑出 1.7983 的那份代码。

为什么 append 而不是 insert(0)：
  `sys.path[0]` 是脚本所在目录（本目录）。用 append 让**本目录同名文件优先**，
  以后 L3 自己长出 `model.py` 时不会被 L2 的盖掉。

为什么不用装包 / `__init__.py`：
  两个练习文件、没有包结构，一次 sys.path 比改 import 路径更轻。
  下面的 import 必然出现在 sys.path 修改**之后**（模块级 import 也只能这么写）。
"""

import sys
from pathlib import Path

import torch

_L2_DIR: Path = Path(__file__).resolve().parents[1] / "l2-nanogpt"
"""L2 关 4 的目录（`corpus.py` / `model.py` 所在）。"""

if str(_L2_DIR) not in sys.path:
    sys.path.append(str(_L2_DIR))

from corpus import (
    BATCH,
    BLOCK,
    build_vocab,
    encode,
    get_batch,
    load_corpus,
    split_ids,
)
from model import CharGPT

__all__ = [
    "BATCH",
    "BLOCK",
    "CharGPT",
    "build_vocab",
    "encode",
    "get_batch",
    "load_corpus",
    "load_data",
    "split_ids",
]
"""本模块 = 「L2 关 4 的数据与模型」的转发层；`__all__` 同时告诉 ruff 这些 import 是**再用**，不是多余。"""


def load_data() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """读语料 → 编码 → 切分（胶水，与 L2 `train.py::load_data` 同一份逻辑）。

    返回 `(train_ids, val_ids, itos)`；词表 65、train 1,003,854 / val 111,540（2026-11-25 实测）。
    """
    text = load_corpus()
    itos, stoi = build_vocab(text)
    ids = encode(text, stoi)
    train, val = split_ids(ids)
    return train, val, itos
