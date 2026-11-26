"""L3 · 训练工程 —— 关卡 3：过拟合（100 KB 小语料 · 训到 train CE < 0.5）

判据全文 = 根 `README.md` 阶段表 L3 关 3 行。要点：
  ① 100 KB 小语料训到**训练集 CE < 0.5**，报留出集 CE 从第几步开始**连续上升**（拐点），
     并报拐点与最后一步的留出集 CE（证明真升）
  ② 一次干预（这里选 **weight decay 0.01 -> 0.1**，二选一），报「训练集到 0.5 所需步数」与拐点位置变化；
     无效也要报两个数并记一行「无效」

为什么用小语料：1.1 MB + 2000 步那组里，过拟合已经在发生（L2 关 4 末段 val 1.7974 -> 1.7983），
但幅度 0.0009 远小于噪声带 0.019560 ⇒ 看不见。100 KB 把「每参数分到的字符」从 1.21 压到 0.11，
让「记忆」这条捷径的收益大于「学规律」，现象才从噪声级升到可测量级。

口径（**跑前写死**，改必须写理由）：
  · 语料切法 = `text[:100_000]`（**从头切**，不用等距抽样）。理由：等距抽样等于同时动两个变量
    （语料量 + 分布），而且 val 会失去「同一段剧本」的同分布性质。代价：只覆盖 tinyshakespeare 前半
    （历史剧），val 也只考这一种文风 —— 如实记，不假装它代表全部语料。
  · 语料 100 KB 后按 9:1 切 ⇒ train ~90,000 / val ~10,000 字符。**评估口径变了**：
    `estimate_ce` 仍是 20 x 64 x 128 = 163,840 token，而 val 只有 1 万字符、窗口起点只有约 9,871 个
    ⇒ 等于把整段 val 重复采样约 16 遍，从「抽样估计」变成「近似全量」。误差更小，但**不能跨关比 CE**。
  · lr 固定 1e-3（关 1 的最优）；batch 64、block 128、2000 步上限不设，改为「训到指标 + 再跑一段」。
  · 停止条件 = 训练集**评估** CE 首次 < 0.5（评估每 100 步一次，所以实际停在第一个跌破的评估点）。
  · 到达后再跑 `EXTRA_STEPS` 步，用来观察 val 是否从拐点继续爬（判据 ① 要「连续上升」）。
  · 拐点定义 = val 序列的**最小值点**；「真升」= 从拐点起出现 ≥3 次上升，且末值 − 拐点值 > 噪声带。

跑法：
    cd practice/ai/l3-train
    uv run --project ../l2-nanogpt python overfit.py probe   # 2000 步探速，看 train CE 掉到哪
    uv run --project ../l2-nanogpt python overfit.py         # 判据 ①② 全跑（基线 + 干预）
"""

import os
import sys

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

from harness import Cfg, Run, train, write_csv
from reuse_l2 import build_vocab, encode, load_corpus
from train_base import CURVE_DIR

SMALL_CHARS: int = 100_000
"""小语料长度（判据原文的 100 KB）。"""

VAL_FRAC: float = 0.10
LR: float = 1e-3
BATCH: int = 64
STOP_TRAIN_CE: float = 0.5
EXTRA_STEPS: int = 2000
EVAL_EVERY: int = 100
NOISE_BAND: float = 0.019560
WD_BASE: float = 0.01
WD_INTERVENE: float = 0.1

_SMALL: tuple[torch.Tensor, torch.Tensor, list[str]] | None = None


def build_small() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """取语料前 100 KB → 建词表 → 编码 → 按位置 9:1 切（胶水，缓存一次）。"""
    global _SMALL
    if _SMALL is None:
        text = load_corpus()[:SMALL_CHARS]
        itos, stoi = build_vocab(text)
        ids = encode(text, stoi)
        n = int(len(ids) * (1.0 - VAL_FRAC))
        _SMALL = (ids[:n], ids[n:], itos)
    return _SMALL


def run_one(tag: str, weight_decay: float, steps: int, probe: bool = False) -> Run:
    """跑一次 + 落曲线 + 打摘要。"""
    train_data, val_data, itos = build_small()
    cfg = Cfg(
        lr=LR,
        batch=BATCH,
        weight_decay=weight_decay,
        steps=steps,
        eval_every=200 if probe else EVAL_EVERY,
        stop_train_ce=None if probe else STOP_TRAIN_CE,
        extra_steps=0 if probe else EXTRA_STEPS,
    )
    print(f"\n== {tag}：lr {LR} batch {BATCH} wd {weight_decay} 预算 {steps} 步 ==")
    print(
        f"   语料 {len(train_data) + len(val_data)} 字符 · train {len(train_data)} / val {len(val_data)} · 词表 {len(itos)}"
    )
    run = train(cfg, train_data, val_data, itos)
    path = write_csv(run, CURVE_DIR / f"{tag}.csv")
    print(
        f"   [{tag}] 走了 {run.steps_done} 步 · 停止点 {run.stop_step}"
        f" · train {run.train_ce:.4f} / val {run.val_ce:.4f}"
        f" · {run.seconds:.1f}s · 峰值 {run.peak_gib:.3f} GiB · {path.name}"
    )
    return run


def turning_point(run: Run) -> tuple[int, float, float]:
    """返回 (拐点步, 拐点 val, 末步 val)：拐点 = val 序列最小值所在步。"""
    best_step, best_val = min(
        ((step, va) for step, _, va in run.evals), key=lambda pair: pair[1]
    )
    return best_step, best_val, run.val_ce


def rising_count(run: Run, after_step: int) -> int:
    """拐点之后 val 一共上升过几次（相邻两点比较）。"""
    vals = [(step, va) for step, _, va in run.evals if step >= after_step]
    return sum(1 for i in range(1, len(vals)) if vals[i][1] > vals[i - 1][1])


def describe(run: Run, label: str) -> dict[str, object]:
    """算出一份摘要：到 0.5 的步数、拐点、末值、上升次数、是否真升。"""
    tp_step, tp_val, last_val = turning_point(run)
    rises = rising_count(run, tp_step)
    real_rise = rises >= 3 and (last_val - tp_val) > NOISE_BAND
    print(
        f"   {label}：到 {STOP_TRAIN_CE} 用了 **{run.stop_step}** 步"
        f" · 拐点 step {tp_step}（val {tp_val:.4f}）"
        f" · 末步 step {run.steps_done}（val {last_val:.4f}）"
        f" · 拐点后上升 {rises} 次 · 升幅 {last_val - tp_val:+.4f}"
        f" · 真升 {real_rise}"
    )
    return {
        "stop_step": run.stop_step,
        "tp_step": tp_step,
        "tp_val": tp_val,
        "last_val": last_val,
        "rises": rises,
        "real_rise": real_rise,
        "train_ce": run.train_ce,
        "steps_done": run.steps_done,
    }


def judge_overfit() -> bool:
    """判据 ①②：基线跑一次、干预跑一次，全绿才 True。"""
    base = run_one("of-base", WD_BASE, steps=20000)
    info_base = describe(base, "基线 wd=0.01")
    checks = [
        base.stop_step is not None,
        bool(info_base["real_rise"]),
    ]
    print(
        f"   ① 训到 train CE < {STOP_TRAIN_CE} 且 val 真升  {'OK' if all(checks) else 'NG'}"
    )

    inter = run_one("of-wd0.1", WD_INTERVENE, steps=20000)
    info_inter = describe(inter, "干预 wd=0.1")
    print(
        f"   ② 干预效果：到 0.5 的步数 {info_base['stop_step']} -> {info_inter['stop_step']}"
        f" · 拐点 {info_base['tp_step']} -> {info_inter['tp_step']}"
        f" · 拐点 val {info_base['tp_val']:.4f} -> {info_inter['tp_val']:.4f}"
    )
    changed = (
        info_base["stop_step"] != info_inter["stop_step"]
        or info_base["tp_step"] != info_inter["tp_step"]
    )
    if not changed:
        print("   ② 记一行「无效」：两个数都没变（判据要求无效也报数）")
    return all(checks)


def main() -> None:
    """入口（纯胶水）：`python overfit.py [probe|full]`，默认 full。"""
    which = sys.argv[1] if len(sys.argv) > 1 else "full"
    print(f"确定性 {torch.are_deterministic_algorithms_enabled()} · 任务 {which}")
    if which == "probe":
        run = run_one("of-probe", WD_BASE, steps=2000, probe=True)
        step, val, _ = turning_point(run)
        print(
            f"   探速：2000 步 train {run.train_ce:.4f} · 拐点 step {step}（val {val:.4f}）· {run.seconds:.1f}s"
        )
        return
    ok = judge_overfit()
    print(f"判据 {1 if ok else 0}/1 绿")


if __name__ == "__main__":
    main()
