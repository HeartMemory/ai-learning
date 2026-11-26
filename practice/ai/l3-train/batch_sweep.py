"""L3 · 训练工程 —— 关卡 4：batch（与 lr 的耦合）

判据全文 = 根 `README.md` 阶段表 L3 关 4 行。要点：
  ① batch ∈ {8, 32, 128} 三档、**lr 按 batch 线性缩放**，各 2000 步，报每步平均耗时、显存峰值、终值 val CE
  ② 再做一次「同 lr 只换 batch」（lr 固定、batch 8 vs 128），报两档 val CE 与每步耗时，
     说明 batch 与 lr 的耦合（大 batch 不缩 lr 会怎样）

lr 的锚点口径（**跑前写死**）：
  判据原文写「lr 按 batch/32 线性缩放」，但「batch 32 对应多少 lr」原文没给。本目录取：
      **锚点 = 关 1 的实测最优组合：batch 64 + lr 1e-3** ⇒ `lr(b) = 1e-3 * b / 64`
      结果：batch 8 → 1.25e-4 · 32 → 5e-4 · 128 → 2e-3
  理由：锚点必须落在一次**已实测**的最优上；若强行令 batch 32 ↦ 1e-3，则 batch 64 会跑 2e-3
  （关 1 从没测过这个组合），对照就同时动了两个量。缩放规则本身是线性的，换锚点只是整体乘一个常数，
  「缩 vs 不缩」的相对结论不变 —— 但绝对数值会变，所以要写清是哪个锚点。

单变量：只改 batch（①）或只改 batch 而 lr 固定（②），其余（语料 1.1 MB、seed 1337、2000 步、
确定性开关）不动。**评估口径固定**：`estimate_ce` 永远用 20 个 64 x 128 的 batch，不随训练 batch 变。

跑法：
    cd practice/ai/l3-train
    uv run --project ../l2-nanogpt python batch_sweep.py          # ① 三档缩放的 batch
    uv run --project ../l2-nanogpt python batch_sweep.py fixed    # ② 同 lr 只换 batch
    uv run --project ../l2-nanogpt python batch_sweep.py read     # 只读 CSV 复算
"""

import csv
import os
import sys
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from harness import Cfg, Run, train, write_csv
from train_base import CURVE_DIR, STEPS, build_data

BATCHES: tuple[int, ...] = (8, 32, 128)
"""判据原文的三档。"""

LR_REF: float = 1e-3
BATCH_REF: int = 64
"""锚点 = 关 1 的实测最优（batch 64 + lr 1e-3）。"""

LR_FIXED: float = 1e-3
"""判据 ② 的「同 lr」：固定 1e-3，batch 8 与 128 各跑一次。"""


def lr_for(batch: int) -> float:
    """线性缩放：lr = LR_REF * batch / BATCH_REF。"""
    return LR_REF * batch / BATCH_REF


def run_one(tag: str, batch: int, lr: float) -> Run:
    """跑一次 + 落曲线 + 打摘要（每步平均耗时、峰值显存、终值）。"""
    train_data, val_data, itos = build_data()
    cfg = Cfg(lr=lr, batch=batch, weight_decay=0.01, steps=STEPS)
    print(f"\n== {tag}：batch {batch} · lr {lr:g} · {STEPS} 步 ==")
    run = train(cfg, train_data, val_data, itos)
    path = write_csv(run, CURVE_DIR / f"{tag}.csv")
    print(
        f"   [{tag}] val CE {run.val_ce:.6f} · train CE {run.train_ce:.4f}"
        f" · {run.seconds:.1f}s（{run.seconds / max(run.steps_done, 1) * 1000:.1f} ms/步）"
        f" · 峰值 {run.peak_gib:.3f} GiB · {path.name}"
    )
    return run


def summary(run: Run) -> dict[str, float]:
    """一档的四个数：终值 val CE、train CE、每步耗时(ms)、峰值显存(GiB)。"""
    return {
        "val_ce": run.val_ce,
        "train_ce": run.train_ce,
        "ms_per_step": run.seconds / max(run.steps_done, 1) * 1000,
        "peak_gib": run.peak_gib,
    }


def report_scaled() -> bool:
    """判据 ①：三档缩放后的 batch，报三个终值 + 耗时 + 显存。"""
    runs = [(b, run_one(f"bs-{b}", b, lr_for(b))) for b in BATCHES]
    print("\n   batch   lr        终值 valCE    trainCE    ms/步    峰值 GiB")
    for batch, run in runs:
        s = summary(run)
        print(
            f"   {batch:<7d} {lr_for(batch):<10g} {s['val_ce']:<13.6f}"
            f" {s['train_ce']:<10.4f} {s['ms_per_step']:<8.1f} {s['peak_gib']:.3f}"
        )
    # 判据只要求「报」，不设阈值；给一条辅助观察：终值随 batch 的走向
    vals = [run.val_ce for _, run in runs]
    best = min(runs, key=lambda pair: pair[1].val_ce)
    print(
        f"   终值最好的一档：batch {best[0]}（val {best[1].val_ce:.6f}）· 三档极差 {max(vals) - min(vals):.6f}"
    )
    return all(abs(v - v) == 0 for v in vals)  # 判据 ① 是取值题：跑到就绿


def report_fixed() -> bool:
    """判据 ②：lr 固定 1e-3，只换 batch（8 vs 128）。"""
    runs = [(b, run_one(f"bs-fixed-{b}", b, LR_FIXED)) for b in (8, 128)]
    print("\n   batch   lr        终值 valCE    trainCE    ms/步    峰值 GiB")
    for batch, run in runs:
        s = summary(run)
        print(
            f"   {batch:<7d} {LR_FIXED:<10g} {s['val_ce']:<13.6f}"
            f" {s['train_ce']:<10.4f} {s['ms_per_step']:<8.1f} {s['peak_gib']:.3f}"
        )
    small, large = runs[0][1], runs[1][1]
    print(
        f"   同 lr 下：batch 8 → {small.val_ce:.6f} · batch 128 → {large.val_ce:.6f}"
        f" · 差 {large.val_ce - small.val_ce:+.6f}"
    )
    print(
        f"   缩放该给的值：batch 8 应 {lr_for(8):g}（实跑 {LR_FIXED:g}，{LR_FIXED / lr_for(8):.0f} 倍）"
        f" · batch 128 应 {lr_for(128):g}（实跑 {LR_FIXED:g}，{lr_for(128) / LR_FIXED:.1f} 分之一）"
    )
    return True


def load_run(path: Path) -> tuple[float, float, float, float]:
    """从 CSV 复算 (val_ce, train_ce, 峰值占位 0, ms 占位 0) —— 只看终值的场合够用。"""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    evals = [
        (float(r["train_eval_ce"]), float(r["val_eval_ce"]))
        for r in rows
        if r["val_eval_ce"]
    ]
    return evals[-1][1], evals[-1][0], 0.0, 0.0


def read_all() -> None:
    """只读 CSV 复算两组终值（不用 GPU；耗时/显存那两列 CSV 里没有）。"""
    print("\n   [read] ①缩放组")
    for b in BATCHES:
        path = CURVE_DIR / f"bs-{b}.csv"
        if not path.exists():
            print(f"   batch {b}: 缺 CSV")
            continue
        val, tr, _, _ = load_run(path)
        print(f"   batch {b:<5d} lr {lr_for(b):<10g} val {val:.6f} train {tr:.4f}")
    print("\n   [read] ②同 lr 组")
    for b in (8, 128):
        path = CURVE_DIR / f"bs-fixed-{b}.csv"
        if not path.exists():
            print(f"   batch {b}: 缺 CSV")
            continue
        val, tr, _, _ = load_run(path)
        print(f"   batch {b:<5d} lr {LR_FIXED:<10g} val {val:.6f} train {tr:.4f}")


def main() -> None:
    """入口（纯胶水）：`python batch_sweep.py [scaled|fixed|read]`，默认 scaled。"""
    which = sys.argv[1] if len(sys.argv) > 1 else "scaled"
    print(
        f"确定性 {torch.are_deterministic_algorithms_enabled()} · 任务 {which} · 锚点 batch {BATCH_REF} / lr {LR_REF}"
    )
    if which == "read":
        read_all()
        return
    ok = report_fixed() if which == "fixed" else report_scaled()
    print(f"判据 {1 if ok else 0}/1 绿")


if __name__ == "__main__":
    main()
