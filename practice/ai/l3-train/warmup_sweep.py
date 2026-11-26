"""L3 · 训练工程 —— 关卡 2：warmup（0 / 100 / 500 步）

判据全文 = 根 `README.md` 阶段表 L3 关 2 行。要点：
  ① 取关卡 1 的**稳值**，0 / 100 / 500 步 warmup 三档，报每档**前 200 步 CE 峰值**与**最终 val CE**
  ② 结论必须有数据：三档终值差 ≤ 噪声带 ⇒ 记「本模型不需要 warmup」；
     需要则报峰值被压低的具体数值，并说清 warmup 解决的是「早期大 lr 把权重推坏」，不是「后期收敛慢」

本文件两组实验（**分开记**）：
  base 组：lr = **1e-3**（关 1 的最优稳值），0 / 100 / 500 —— 判据 ① 的原文要求
  hot  组：lr = **1e-1**（关 1 里前 100 步峰值 12.69 的那档），同样三档
          —— 判据 ② 要求「说清 warmup 解决早期推坏」。稳值档本来就不炸，warmup 无处发力；
          要验证机制，必须在**会早炸的那一档**上做对照，否则「没变化」说明不了任何事。

单变量：只改 warmup 步数，其余（seed 1337、语料、词表、模型、2000 步、确定性开关）不动。
warmup 的实现（在 `train_base.train_once` 里）：前 `warmup` 步 `lr_scale = (step+1)/warmup`，
之后恒 1 —— 线性升温，不用 `LambdaLR`（本关只有一个参数组，直接改 `param_groups` 更好核对）。

跑法：
    cd practice/ai/l3-train
    uv run --project ../l2-nanogpt python warmup_sweep.py         # base 组（约 2 分钟）
    uv run --project ../l2-nanogpt python warmup_sweep.py hot     # hot 组（约 2 分钟）
    uv run --project ../l2-nanogpt python warmup_sweep.py read    # 只读 CSV 复算（不用 GPU）
"""

import csv
import math
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from train_base import CURVE_DIR, SEED, STEPS, RunResult, train_once, write_curve

WARMUPS: tuple[int, ...] = (0, 100, 500)
"""判据原文的三档，跑前写死。"""

LR_BASE: float = 1e-3
"""关 1 的最优稳值（val CE 1.611282）—— 判据 ① 说的「取关卡 1 的稳值」。"""

LR_HOT: float = 1e-1
"""关 1 里「前 100 步峰值 12.69、终值 3.3488」的那档：会早炸，用来验证 warmup 的机制。"""

NOISE_BAND: float = 0.019560
"""关 0 的 3 seed 极差，作为「三档终值差算不算有差别」的门限。"""

PEAK_WINDOW: int = 200
"""判据原文的「前 200 步 CE 峰值」窗口。"""


def run_warmup(lr: float, warmup: int) -> RunResult:
    """跑一档 + 落曲线（胶水）。"""
    res = train_once(SEED, lr, warmup)
    res.tag = f"wu-{lr:g}-w{warmup}"
    path = write_curve(res)
    print(
        f"  [{res.tag}] val CE {res.val_ce:.6f} · {res.seconds:.1f}s"
        f" · 峰值 {res.peak_gib:.3f} GiB · {path.name}"
    )
    return res


def peak_of(res: RunResult, window: int = PEAK_WINDOW) -> float:
    """前 `window` 步里有限 CE 的最大值（爆炸时也可能出现非有限值，先过滤）。"""
    finite = [v for v in res.step_ce[:window] if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def report(runs: list[tuple[int, RunResult]], lr: float, label: str) -> bool:
    """打表 + 判据 ②（三档终值极差与噪声带比较），全绿才 True。"""
    print(f"\n== {label}：lr={lr:g} · warmup {WARMUPS} ==")
    print("\n   warmup   终值 valCE    前200步峰值   峰值出现步   耗时")
    for warmup, res in runs:
        peak = peak_of(res)
        idx = res.step_ce[:PEAK_WINDOW].index(peak) + 1 if math.isfinite(peak) else -1
        print(
            f"   {warmup:<8d} {res.val_ce:<13.6f} {peak:<13.4f} {idx:<12d} {res.seconds:.1f}s"
        )

    finals = [res.val_ce for _, res in runs]
    spread = max(finals) - min(finals)
    peaks = [peak_of(res) for _, res in runs]
    peak_spread = max(peaks) - min(peaks)
    ok = spread <= NOISE_BAND
    # ⚠️ 只用 ASCII 标记：⇒（U+21D2）在中文 Windows 控制台会 UnicodeEncodeError（GBK）
    print(
        f"   三档终值极差 {spread:.6f}（噪声带 {NOISE_BAND:.6f}）"
        f" -> {'不需要 warmup' if ok else '终值确有差别'}  {'OK' if ok else 'NG'}"
    )
    print(f"   三档峰值极差 {peak_spread:.6f}（峰值 {[round(p, 4) for p in peaks]}）")
    return ok


def load_result(path: str) -> RunResult:
    """从 CSV 还原 RunResult（只填判据用得到的字段）。"""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    step_ce = [float(r["train_step_ce"]) for r in rows if r["train_step_ce"]]
    evals = [
        (int(r["step"]), float(r["train_eval_ce"]), float(r["val_eval_ce"]))
        for r in rows
        if r["val_eval_ce"]
    ]
    return RunResult(
        seed=SEED,
        tag=Path(path).stem,
        step_ce=step_ce,
        evals=evals,
        train_ce=evals[-1][1],
        val_ce=evals[-1][2],
    )


def report_from_curves(lr: float, label: str) -> bool:
    """只读 `curves/wu-<lr>-w*.csv` 复算（免 GPU）。"""
    key = f"wu-{lr:g}-w"
    paths = {w: CURVE_DIR / f"{key}{w}.csv" for w in WARMUPS}
    missing = [w for w, p in paths.items() if not p.exists()]
    if missing:
        print(f"{label}：缺 warmup 档 {missing} 的 CSV  NG")
        return False
    return report([(w, load_result(str(p))) for w, p in paths.items()], lr, label)


def main() -> None:
    """入口（纯胶水）：`python warmup_sweep.py [base|hot|read]`，默认 base。"""
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    print(
        f"确定性 {torch.are_deterministic_algorithms_enabled()}"
        f" · {STEPS} 步 · seed {SEED} · 噪声带 {NOISE_BAND} · 任务 {which}"
    )
    if which == "read":
        ok_base = report_from_curves(LR_BASE, "判据 ① base 组")
        ok_hot = report_from_curves(LR_HOT, "机制对照 hot 组")
        print(f"组 {int(ok_base) + int(ok_hot)}/2 绿")
        return
    lr = LR_HOT if which == "hot" else LR_BASE
    label = "机制对照 hot 组" if which == "hot" else "判据 ① base 组"
    ok = report([(w, run_warmup(lr, w)) for w in WARMUPS], lr, label)
    print(f"判据 {1 if ok else 0}/1 绿")


if __name__ == "__main__":
    main()
