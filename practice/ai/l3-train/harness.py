"""L3 · 训练工程 —— 通用训练器（关 3 / 关 4 共用）

为什么另起一份，而不把参数塞进 `train_base.train_once`：
  `train_once` 是关 0/1/2 的基线（2000 步固定、batch 64、语料固定），它的判据靠「同 seed 逐位一致」，
  一个参数都不许多；关 3 要「训到训练集 CE < 0.5 再继续跑看 val 上升」，关 4 要把 batch 当变量，
  两者都需要额外入口。塞进去会让关 0 的口径表越滚越长。
  ⇒ 另建带 `Cfg` 的实现，**共用** `reuse_l2` 的语料/模型与 `train_base.estimate_ce` 的评估口径。
  代价 = 训练主循环重复约 25 行；收益 = 每关的入口参数一目了然。

评估口径**不随训练 batch 变**：`estimate_ce` 固定用 20 个 64 x 128 的 batch（163,840 token），
这样关 4 里 batch 8 / 32 / 128 的终值才可比。
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from reuse_l2 import CharGPT, get_batch
from train_base import (
    BLOCK,
    EVAL_EVERY,
    EVAL_ITERS,
    GRAD_CLIP,
    LR,
    STEPS,
    estimate_ce,
    get_device,
)


@dataclass
class Cfg:
    """一次 run 的全部入口参数。默认值 = 关 0/1/2 的口径。"""

    seed: int = 1337
    lr: float = LR
    batch: int = 64
    block: int = BLOCK
    weight_decay: float = 0.01
    steps: int = STEPS
    eval_every: int = EVAL_EVERY
    eval_iters: int = EVAL_ITERS
    stop_train_ce: float | None = None
    """训练集**评估** CE 低于它就停（关 3 的「训到 train CE < 0.5」）。None = 跑满 steps。"""
    extra_steps: int = 0
    """到达 stop 之后再跑多少步（关 3 要看 val 从拐点继续爬）。"""


@dataclass
class Run:
    """一次 run 的产出。"""

    cfg: Cfg
    step_ce: list[float] = field(default_factory=list)
    evals: list[tuple[int, float, float]] = field(default_factory=list)
    steps_done: int = 0
    stop_step: int | None = None
    """训练集评估 CE 首次跌破 `stop_train_ce` 的步号（没跌破 = None）。"""
    train_ce: float = float("nan")
    val_ce: float = float("nan")
    seconds: float = 0.0
    peak_gib: float = 0.0


def train(
    cfg: Cfg, train_data: torch.Tensor, val_data: torch.Tensor, itos: list[str]
) -> Run:
    """按 `Cfg` 跑一次。返回 `Run`（曲线、评估、停止点、耗时、峰值显存）。"""
    dev = get_device()

    torch.manual_seed(cfg.seed)
    gen_train = torch.Generator().manual_seed(cfg.seed)
    gen_eval = torch.Generator().manual_seed(cfg.seed + 1)

    model = CharGPT(vocab_size=len(itos)).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    model.train()

    run = Run(cfg=cfg)
    budget = cfg.steps + cfg.extra_steps
    t0 = time.perf_counter()

    for step in range(budget + 1):
        evaluate = step % cfg.eval_every == 0 or step == budget
        if evaluate:
            tr = estimate_ce(model, train_data, gen_eval, cfg.eval_iters)
            va = estimate_ce(model, val_data, gen_eval, cfg.eval_iters)
            run.evals.append((step, tr, va))
            print(f"  step {step:6d}  train {tr:.4f}  val {va:.4f}")
            if (
                cfg.stop_train_ce is not None
                and run.stop_step is None
                and tr < cfg.stop_train_ce
            ):
                run.stop_step = step
                # 到了目标就只再多跑 extra_steps 步
                budget = min(budget, step + cfg.extra_steps)
        if step == budget:
            break

        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_data, cfg.batch, cfg.block, gen_train)
        _, loss = model(x.to(dev), y.to(dev))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        run.step_ce.append(float(loss.item()))
        run.steps_done = step + 1

    run.seconds = time.perf_counter() - t0
    run.train_ce = run.evals[-1][1]
    run.val_ce = run.evals[-1][2]
    run.peak_gib = (
        torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
    )
    return run


def write_csv(run: Run, path: Path) -> Path:
    """落曲线（关 3 / 关 4 的步数不固定，所以不能用 `train_base.write_curve` 的固定 STEPS 版本）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    evals = {step: (tr, va) for step, tr, va in run.evals}
    lines = ["step,train_step_ce,train_eval_ce,val_eval_ce"]
    for step in range(run.steps_done + 1):
        loss = f"{run.step_ce[step - 1]:.6f}" if 1 <= step <= len(run.step_ce) else ""
        ev = evals.get(step)
        tr_s = "" if ev is None else f"{ev[0]:.6f}"
        va_s = "" if ev is None else f"{ev[1]:.6f}"
        lines.append(f"{step},{loss},{tr_s},{va_s}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
