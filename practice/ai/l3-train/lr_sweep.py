"""L3 · 训练工程 —— 关卡 1：lr（五档单变量）

判据全文 = 根 `README.md` 阶段表 L3 关 1 行。要点：
  ① lr ∈ {1e-5, 1e-4, 3e-4, 1e-3, 1e-2} 各 2000 步：报**终值 val CE** / **前 100 步 CE 峰值** /
     有无 **inf/NaN**，并保证「不动 / 稳降 / 发散」三种形态**各至少一档**
  ② 给出**稳值区间与上下界**，并用数字说明**上界之外为什么炸**

单变量：只改 lr，其余（seed 1337、语料、词表、模型、2000 步、tokenizer、确定性开关）
与关 0 完全一致。**确定性开关必须有** —— 关 0 实测默认设置同 seed 重跑就差 6.4e-3，
而本关 1e-4 与 3e-4 两档的终值差只有 1e-3 量级，不压噪声就分不清谁是谁。

三种形态的口径（**跑前写死**，改必须写理由）：
    起点 val CE = `START_CE` 4.3175 —— 关 0 第 0 步实测；所有档共用同一份初始权重，所以起点相同
    噪声带     = `NOISE_BAND` 0.019560 —— 关 0 的 3 seed 极差
    发散：`step_ce` 出现非有限值（inf / NaN），或「末 200 步均值 − 首 200 步均值 > 噪声带」（整体往上走）
    不动：终值 val CE ≥ START_CE − 噪声带，且不属发散
    稳降：其余（终值低于「起点 − 噪声带」，且不属发散）
  ⚠️ 「不动」拿**起点**判、不拿 bigram 基线 2.4819 判：1e-5 可能学得动一点点但远够不到基线，
     用基线判会把它误归「稳降」，三种形态就凑不齐了。

跑法（约 3 分钟 = 5 x 35 s）：
    cd practice/ai/l3-train
    uv run --project ../l2-nanogpt python lr_sweep.py
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

LRS: tuple[float, ...] = (1e-5, 1e-4, 3e-4, 1e-3, 1e-2)
"""判据原文的五档，跑前写死（不许按结果挑）。"""

EXT_LRS: tuple[float, ...] = (1e-7, 1e-6, 3e-2, 1e-1, 1.0)
"""扩展档（2026-11-26 加，理由写在下面）：
原文五档实测**全部是「稳降」**（1e-5 终值 2.5432、1e-2 终值 1.6843，都无 NaN）
⇒ ① 的「三种形态各至少一档」在原文档位内凑不齐；② 又要求给出**上下界**。
所以向两侧各推两档：下界找「不动」（1e-7 / 1e-6），上界找「发散」（3e-2 / 1e-1 / 1.0）。
⚠️ 扩展档只用来回答 ②「界在哪」，**不替换** ① 的原文档位；两种结果在文档里分开记。"""

TINY_LRS: tuple[float, ...] = (1e-8,)
"""第二批扩展（2026-11-26 加）：扩展档里 **1e-7 仍在降**（4.3175 → 4.1874，−0.130 远超噪声带），
说明「不动」不是靠 1e-7 能凑出来的。这一档只回答一个问题：**这个形态到底存不存在**。
若 1e-8 仍在降，结论就是「本模型 + 2000 步 + AdamW 下不存在『不动』」——
因为 Adam 的每步位移约等于 lr（梯度被 v 归一化），2000 x 1e-8 也够把 CE 推下来一点点。"""

START_CE: float = 4.3175
"""第 0 步（未训练）的 val CE —— 关 0 实测；五档共用同一份初始权重，所以起点相同。"""

NOISE_BAND: float = 0.019560
"""关 0 的 3 seed 极差，作为「有没有真的变化」的门限。"""

BIGRAM_CE: float = 2.4819
"""L2 关 4 的 bigram 基线（val，nats/char）。只用来说「有效区间」—— 低于它才算真学到东西。"""

PEAK_WINDOW: int = 100
"""判据原文的「前 100 步 CE 峰值」窗口。"""

TREND_WINDOW: int = 200
"""判「整体上升」用的窗口：末 200 步均值 vs 首 200 步均值。"""


def run_lr(lr: float) -> RunResult:
    """跑一档 + 落曲线（胶水）。"""
    res = train_once(SEED, lr)
    res.tag = f"lr-{lr:g}"
    path = write_curve(res)
    print(
        f"  [{res.tag}] val CE {res.val_ce:.6f} · {res.seconds:.1f}s"
        f" · 峰值 {res.peak_gib:.3f} GiB · {path.name}"
    )
    return res


def first_nonfinite(res: RunResult) -> int | None:
    """第一个 inf/NaN 出现在第几步（步号从 1 开始）；全有限返回 None。"""
    for i, v in enumerate(res.step_ce, start=1):
        if not math.isfinite(v):
            return i
    return None


def first_above(res: RunResult, threshold: float) -> int | None:
    """第一个 CE > threshold 出现在第几步（描述爆炸的起点）。"""
    for i, v in enumerate(res.step_ce, start=1):
        if math.isfinite(v) and v > threshold:
            return i
    return None


def finite_peak(values: list[float]) -> float:
    """只在前 100 步里取有限值的最大值（含 NaN 时 max() 结果不可靠）。"""
    finite = [v for v in values if math.isfinite(v)]
    return max(finite) if finite else float("nan")


def classify(res: RunResult) -> str:
    """返回 '发散' / '不动' / '稳降'（口径见文件头）。

    ⚠️ 发散的第一条是「终值比不训练还差」—— 2026-11-26 实测补的：
       lr=1.0 第 2 步就炸到 CE>10，之后一路在 100~250 之间晃、末段还往下掉，
       若只按「末段在上升」判，它会被误判成「不动」（91.25 > 起点）。爆炸必须先判。
    """
    if first_nonfinite(res) is not None or not math.isfinite(res.val_ce):
        return "发散"
    if res.val_ce > START_CE:
        return "发散"
    head = res.step_ce[:TREND_WINDOW]
    tail = res.step_ce[-TREND_WINDOW:]
    if sum(tail) / len(tail) - sum(head) / len(head) > NOISE_BAND:
        return "发散"
    if res.val_ce >= START_CE - NOISE_BAND:
        return "不动"
    return "稳降"


def report(runs: list[tuple[float, RunResult]], label: str) -> bool:
    """打表 + 判据：① 三种形态是否齐 / ② 稳值区间与上界外的数字。"""
    lrs = tuple(lr for lr, _ in runs)
    kinds = {lr: classify(res) for lr, res in runs}
    print(f"\n== {label}：{lrs} ==")
    print("\n   lr        终值 valCE    前100步峰值   首个非有限步   形态")
    for lr, res in runs:
        peak = finite_peak(res.step_ce[:PEAK_WINDOW])
        peak_s = "nan" if not math.isfinite(peak) else f"{peak:.4f}"
        nonfinite = first_nonfinite(res)
        print(
            f"   {lr:<9g} {res.val_ce:<13.6f} {peak_s:<13} "
            f"{str(nonfinite) if nonfinite else '-':<14} {kinds[lr]}"
        )

    checks: list[bool] = []
    for kind in ("不动", "稳降", "发散"):
        got = [lr for lr in lrs if kinds[lr] == kind]
        ok = bool(got)
        checks.append(ok)
        print(f"  形态「{kind}」：{got if got else '缺'}  {'OK' if ok else 'NG'}")

    stable = [lr for lr in lrs if kinds[lr] == "稳降"]
    if stable:
        best_lr, best_ce = min(
            ((lr, res.val_ce) for lr, res in runs if kinds[lr] == "稳降"),
            key=lambda pair: pair[1],
        )
        print(
            f"  稳值区间 [{min(stable):g}, {max(stable):g}]"
            f" · 最优 lr {best_lr:g}（val CE {best_ce:.6f}）"
        )
        useful = [lr for lr in stable if dict(runs)[lr].val_ce < BIGRAM_CE]
        if useful:
            print(
                f"  有效区间（val CE < bigram 基线 {BIGRAM_CE}）："
                f"[{min(useful):g}, {max(useful):g}]"
            )
    for lr, res in runs:
        if kinds[lr] == "发散":
            print(
                f"  上界外 lr={lr:g}：首个非有限步 {first_nonfinite(res)}"
                f" · 首个 CE>10 步 {first_above(res, 10.0)}"
                f" · 末200步均值 {sum(res.step_ce[-TREND_WINDOW:]) / TREND_WINDOW:.4f}"
            )
    return all(checks)


def judge_lr(lrs: tuple[float, ...], label: str) -> bool:
    """跑一组 lr 再出表（跑 = GPU；表见 `report`）。"""
    return report([(lr, run_lr(lr)) for lr in lrs], label)


def load_result(path: str) -> RunResult:
    """从落盘的 CSV 还原一个 RunResult（只填判据用得到的字段），用于免 GPU 复算。"""
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


def report_from_curves() -> bool:
    """只读 `curves/lr-*.csv` 出总表 —— 不训练、不用 GPU，结果与现场跑一致（确定性模式）。"""
    keys = sorted(
        ((float(p.stem[3:]), p) for p in CURVE_DIR.glob("lr-*.csv")),
        key=lambda pair: pair[0],
    )
    runs = [(lr, load_result(str(p))) for lr, p in keys]
    return report(runs, "全部已落盘档位（从 CSV 复算）")


def main() -> None:
    """入口（纯胶水）：`python lr_sweep.py [base|ext|tiny|all|read]`，默认 base。"""
    which = sys.argv[1] if len(sys.argv) > 1 else "base"
    print(
        f"确定性 {torch.are_deterministic_algorithms_enabled()}"
        f" · {STEPS} 步 · seed {SEED} · 起点 val CE {START_CE}"
        f" · 噪声带 {NOISE_BAND} · 任务 {which}"
    )
    if which == "read":
        ok = report_from_curves()
        print(f"判据 {1 if ok else 0}/1 绿")
        return
    results: list[bool] = []
    if which in ("base", "all"):
        results.append(judge_lr(LRS, "判据 ① 原文五档"))
    if which in ("ext", "all"):
        results.append(judge_lr(EXT_LRS, "扩展档（回答 ② 的上下界）"))
    if which in ("tiny", "all"):
        results.append(judge_lr(TINY_LRS, "第二批扩展（查「不动」是否存在）"))
    print(f"组 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
