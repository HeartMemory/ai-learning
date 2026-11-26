"""L3 · 训练工程 —— 关卡 0（块 2）：开确定性开关，把「不可复现」压掉

块 1 的实测结论（`curves/same-1.csv` vs `same-2.csv`）：
    第 37 步首次分叉（差约 1e-6）· 1400/2000 步超 1e-9 · 最大差 2.59e-02（第 1475 步）
    · 终值 val CE 1.800093 vs 1.802309（差 2.216e-03）
⇒ 默认设置下同 seed 重跑**不复现**。本块问两件事：
    ① 打开确定性开关以后，逐步差能否压到 < 1e-9、终值差能否 == 0
    ② 压到之后多花了多少秒（块 1 基线：47.0 s / 30.5 s，峰值 0.322 GiB）

本文件只做「全局开关」，判据直接复用 `train_base.py::judge_repeat`（一字不改）。
这样「变量只有一个」：跑的是同一份 `train_once`，只有环境与全局 flag 不同。

三个开关的作用（2026-11-26 查证 PyTorch Reproducibility 笔记）：
  1. `CUBLAS_WORKSPACE_CONFIG=:4096:8` —— CUDA >= 10.2 且用 cuBLAS 时**必须**设，
     否则第 2 条一开就报错。必须写在 `import torch` **之前**（它是 cuBLAS 启动参数）。
  2. `torch.use_deterministic_algorithms(True)` —— 让原子累加类算子改走确定实现；
     **遇到没有确定实现的算子直接抛 RuntimeError**（这正是我们要抓的信号）。
  3. `cudnn.benchmark = False` + `cudnn.deterministic = True` —— 关掉 autotune，
     避免不同 run 选到不同算法。（本模型没有卷积，第二条实际不参与，但口径要写全。）

⚠️ 预期里有个岔路：SDPA 的 flash 反向用原子操作累加 dQ（PyTorch 2.13 博客），
   若它没有确定实现，`loss.backward()` 会抛 RuntimeError。那时**不改判据**，
   先记录原文，再换 SDPA 后端（关 flash）重跑 —— 那一档的耗时就是「确定性的真实价格」。

跑法：
    cd practice/ai/l3-train
    uv run --project ../l2-nanogpt python train_det.py     # 两次 run，判据 A–B
"""

import math
import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from train_base import judge_repeat, run_and_save

NOISE_SEEDS: tuple[int, ...] = (1337, 2024, 7)
"""块 3 的三个 seed。跑前写死（含 1337 —— 它会和块 2 的 det-same-1 撞同一个值，
顺带当一次「跨进程复现」的交叉验证）。"""


def judge_noise_band() -> bool:
    """关 0 第 ②：换 3 个 seed 各跑一次，报三个 val CE 与**极差**。

    这条是**取值题**，不是是非题：极差就是后面四关所有「变好 / 变坏」的门限。
    绿的判据 = 三个数都拿到（有限）且极差 > 0 —— 极差为 0 说明训练根本不吃 seed，
    那要先查数据流（训练流没接 seed）。

    ⚠️ 必须在确定性模式下跑：默认设置下同一个 seed 两次就能差 6.4e-3（块 1 实测），
    那条带子里混着「不可复现」的成分，会把 real 的 seed 噪声量高。
    """
    runs = [run_and_save(s, f"det-seed-{s}") for s in NOISE_SEEDS]
    vals = [r.val_ce for r in runs]
    band = max(vals) - min(vals)
    ok = all(math.isfinite(v) for v in vals) and band > 0.0
    print(f"C 3 seed val CE 极差 {band:.6f}（= 噪声带门限）  {'OK' if ok else 'NG'}")
    return ok


def main() -> None:
    """入口（纯胶水）：`python train_det.py [all|repeat|noise]`，默认 all。

    `noise` 只跑块 3（3 个 seed，约 105 s）；`repeat` 只跑块 2（两次同 seed，约 70 s）。
    """
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    print(
        f"确定性开关：use_deterministic_algorithms "
        f"{torch.are_deterministic_algorithms_enabled()} · "
        f"CUBLAS_WORKSPACE_CONFIG {os.environ.get('CUBLAS_WORKSPACE_CONFIG')} · "
        f"cudnn.benchmark {torch.backends.cudnn.benchmark} · 任务 {which}"
    )
    results: list[bool] = []
    try:
        if which in ("all", "repeat"):
            results.append(judge_repeat("det-"))
        if which in ("all", "noise"):
            results.append(judge_noise_band())
    except RuntimeError as exc:
        print(f"RuntimeError（确定性模式下有算子没有确定实现）：{exc}")
        return
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
