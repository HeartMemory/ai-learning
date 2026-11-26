"""L3 · 训练工程 —— 关卡 0：基线可复现（块 1：同 config 连跑两次）

一条根：L2 关 4 跑出了「val CE 1.7983 < 基线 2.4819」，但那是**一次**结果。
L3 后面四关全是「只改一个超参，看 CE 怎么变」—— 若同一份 config 跑两次都得不到同一个数，
所有对照都无从谈起。所以本关不学新东西，先把**尺子**校准：
    本块 ①：同 config（seed 固定）连跑两次 ⇒ 逐步 CE 差多少、最终 val CE 差多少
    下一块 ②：换 3 个 seed 各跑一次 ⇒ val CE 极差 = 后续「变好 / 变坏」的噪声带门限

⚠️ 三条口径，改必须写理由：
  ① **seed 进入一次 run 的唯一入口是 `train_once(seed)`**：模型初始化、训练采样、评估采样
     全部由它派生 ⇒ 同 seed 两次必须走到同一份初值、同一批数据。
  ② **评估采样器也必须固定**：`estimate_ce` 用 `torch.Generator().manual_seed(seed + 1)`。
     同一份权重 + 同一个评估流 ⇒ val CE 才可能逐位相同；否则「差 0」测的只是采样噪声。
  ③ **逐步 CE 取「该步那个 batch 的 loss」**：它是唯一能逐步对账的量。8192 token 的
     单 batch 噪声大（±0.05 量级），但同 seed 两次的噪声是**共模**的 ⇒ 差应该压在 0 附近。

⚠️ 先测，别先开确定性开关（本关要学的就在这里）：
   官方 Reproducibility 笔记原话大意：**seed 只固定随机数序列，不固定浮点累加顺序**；
   而 `scaled_dot_product_attention` 的 flash 后端反向用**原子操作累加 dQ** ⇒ 天然非确定
   （2026-11-26 查证两条来源：PyTorch 论坛 SDPA 复现讨论 + PyTorch 2.13 发布博客的
   deterministic backward 一节）。⇒ 本块先按**默认设置**测，把差如实记下来；
   若差 > 1e-9，下一块再开 `torch.use_deterministic_algorithms(True)` +
   `CUBLAS_WORKSPACE_CONFIG=:4096:8` 重测，并报出耗时代价。

判据（本关自定，序号对齐根 `README.md` 阶段表 L3 关 0）：
    A 逐步一致：两次 run 的 `step_ce` 逐元素最大绝对差 < 1e-9
    B 终值一致：两次 run 的最终 val CE 差 == 0
  附加（不计入判据）：报两次的 val CE、耗时、峰值显存

════════════════════════════════════════════════════════════════════
 TODO 清单
════════════════════════════════════════════════════════════════════
 T1 [实现] train_once(seed)        一次完整训练（镜子 = `l2-nanogpt/train.py::train_model`）
 T2 [已写] estimate_ce / write_curve / judge_repeat（测量与判据，AI 写）

跑法（复用 L2 的 uv 工程，不重装 3 GB 的 torch）：
    cd practice/ai/l3-train
    uv run --project ../l2-nanogpt python train_base.py     # 两次 run 约 100 s
════════════════════════════════════════════════════════════════════
"""

import time
from dataclasses import dataclass, field
from pathlib import Path

import torch

from reuse_l2 import BATCH, BLOCK, CharGPT, get_batch, load_data

SEED: int = 1337
"""本关的主 seed。判据 A / B 就是拿它连跑两次。"""

STEPS: int = 2000
"""固定 2000 步（与 L2 关 4 一致，单变量原则：这一关不能动步数）。"""

LR: float = 3e-4
"""学习率。与 L2 关 4 同值 —— 本关不调它，调它是关 1 的事。"""

WEIGHT_DECAY: float = 0.01
"""AdamW 解耦权重衰减，与 L2 关 4 同值。"""

GRAD_CLIP: float = 1.0
"""梯度全局范数裁剪阈值，与 L2 关 4 同值。"""

EVAL_EVERY: int = 100
"""每多少步评估一次（含第 0 步与最后一步）。"""

EVAL_ITERS: int = 20
"""每次评估用多少个 batch 估 CE（20 x 8192 ≈ 16 万 token）。"""

STEP_TOL: float = 1e-9
"""判据 A 的阈值：两次 run 逐步 CE 的最大绝对差必须低于它。
选 1e-9 而不是「严格 0」：float32 的 loss 是 1e-7 量级的数，1e-9 已经比末位还紧，
过了它就等价于「同一条计算路径」；卡严格 0 会把「同一路径的末位 rounding」也判红。"""

CURVE_DIR: Path = Path(__file__).with_name("curves")
"""曲线落盘目录（**不叫 `runs/`**：根 `.gitignore` 忽略 `runs/`，而曲线是要入库的判据证据）。"""


@dataclass
class RunResult:
    """一次完整训练的产出。**字段名固定** —— 判据按这些名字取数，T1 必须原样填。"""

    seed: int
    tag: str = ""
    step_ce: list[float] = field(default_factory=list)
    """len == STEPS：`step_ce[k]` = 第 k+1 步那个 batch 的训练 CE（nats/char）。"""
    evals: list[tuple[int, float, float]] = field(default_factory=list)
    """(step, train CE, val CE)：第 0 步 / 每 EVAL_EVERY 步 / 最后一步。"""
    train_ce: float = float("nan")
    """最后一次评估得到的**训练集** CE（`estimate_ce` 的估计值，不是单 batch loss）。"""
    val_ce: float = float("nan")
    """最后一次评估得到的**留出集** CE —— 判据 B 比的就是它。"""
    seconds: float = 0.0
    """本次 run 的总耗时（秒）。"""
    peak_gib: float = 0.0
    """本次 run 的显存峰值（GiB，`max_memory_allocated`；CPU 上记 0）。"""


_DATA: tuple[torch.Tensor, torch.Tensor, list[str]] | None = None
"""语料缓存：编码 1.1 M 字符约 1.5 s，一次进程跑 5 个 run 不必重复做。"""


def build_data() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """读语料 → 编码 → 切分（胶水，首调约 1.5 s，之后走缓存）。"""
    global _DATA
    if _DATA is None:
        _DATA = load_data()
    return _DATA


def get_device() -> torch.device:
    """有 cuda 就走 GPU（胶水）。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_ce(
    model: CharGPT,
    data: torch.Tensor,
    generator: torch.Generator,
    iters: int = EVAL_ITERS,
) -> float:
    """在 data 上估平均 CE（nats/char）。**测量代码，已实现**（不属于你的 TODO）。

    两点口径：
      · 用**固定批数**而不是全量：val 11 万字符 = 872 个 batch，每 100 步全量跑一遍
        等于训练时间翻 20 倍。20 个 batch 的蒙特卡洛误差约 ±0.01，够判「低于 2.4819」。
      · `torch.no_grad()` 不能省：否则 20 个 batch 的反向图全挂在显存里。
    ⚠️ 内部会切 `model.eval()` / `model.train()`；调用方不必自己切。
    """
    dev = next(model.parameters()).device
    model.eval()
    total = 0.0
    with torch.no_grad():
        for _ in range(iters):
            x, y = get_batch(data, BATCH, BLOCK, generator)
            _, loss = model(x.to(dev), y.to(dev))
            total += float(loss.item())
    model.train()
    return total / iters


def write_curve(res: RunResult) -> Path:
    """把一次 run 的曲线写成 CSV（胶水）：`curves/<tag>.csv`，每行一个 step。

    四列：`step` / 该步单 batch 的 `train_step_ce`（step 0 为空）/ 每 100 步的
    `train_eval_ce` / `val_eval_ce`。稀疏列留空而不是补 0 —— 补 0 会被误读成「CE 掉到 0」。
    """
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    path = CURVE_DIR / f"{res.tag or f'seed-{res.seed}'}.csv"
    evals = {step: (tr, va) for step, tr, va in res.evals}
    lines = ["step,train_step_ce,train_eval_ce,val_eval_ce"]
    for step in range(STEPS + 1):
        loss = f"{res.step_ce[step - 1]:.6f}" if 1 <= step <= len(res.step_ce) else ""
        tr_va = evals.get(step)
        tr_s = "" if tr_va is None else f"{tr_va[0]:.6f}"
        va_s = "" if tr_va is None else f"{tr_va[1]:.6f}"
        lines.append(f"{step},{loss},{tr_s},{va_s}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def train_once(seed: int, lr: float = LR, warmup: int = 0) -> RunResult:
    """T1 · 一次完整训练。**同 seed 必须给出逐位相同的结果**（判据 A / B 全靠这条）。

    `lr` 默认取模块常量 `LR`（= 3e-4，关 0 的口径）；关 1 扫 lr 时传别的值 —— 只改这一个量。
    `warmup` 是线性升温的步数（关 2）：前 `warmup` 步把 lr 从 `lr/warmup` 线性升到 `lr`，
    之后恒定；`0` = 不升温（关 0 / 关 1 的口径）。

    必须做到（顺序与字段名都别改，判据按 `RunResult` 的字段取数）：
      1. 随机性只从 seed 来（关 0 的唯一入口）：
             torch.manual_seed(seed)                                # 模型初始化
             gen_train = torch.Generator().manual_seed(seed)        # 训练采样
             gen_eval = torch.Generator().manual_seed(seed + 1)     # 评估采样（口径 ②）
         三个流必须分开 —— 评估流插进训练流，同 seed 就走不到同一批数据。
      2. 建模型 + 上设备；cuda 下先 `torch.cuda.reset_peak_memory_stats()`
         （不重置就是从进程启动算起，会把上一个 run 的峰值算进来）。
      3. 优化器 `torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)`。
      4. 跑 **STEPS** 步，每步六件事（顺序不能换）：
             opt.zero_grad(set_to_none=True)
             x, y = get_batch(train, BATCH, BLOCK, gen_train)
             _, loss = model(x.to(dev), y.to(dev))
             loss.backward()
             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
             opt.step()
         每步把 `float(loss.item())` 追加进 `step_ce` —— **必须逐步**，判据 A 比的就是这张表。
      5. 每 **EVAL_EVERY** 步 + 第 0 步 + 最后一步：
             tr = estimate_ce(model, train, gen_eval)
             va = estimate_ce(model, val, gen_eval)
         记进 `evals` 并打印一行（**只用 ASCII 标记**：GBK 控制台打不出 ✓ / − / ³）。
      6. 返回填满的 `RunResult`：`seconds` 用 `time.perf_counter()`，`peak_gib` 用
         `torch.cuda.max_memory_allocated() / 2**30`（CPU 上记 0.0）。

    提示：
      · 镜子在 `l2-nanogpt/train.py::train_model`（那次只留了每 100 步的评估值，
        本关多一件事：**每步的 loss 也要留**）
      · 第 0 步只评估、不更新 —— L2 的写法是 `for step in range(STEPS + 1)` + 末尾 `break`
      · `clip_grad_norm_` 放在 `backward()` 之后、`step()` 之前（顺序错了阈值没意义）
      · 别每步都调 `estimate_ce`（2000 x 40 次前向会让训练时间翻倍）

    自检：`len(step_ce) == STEPS`、val CE ≈1.80（L2 关 4 是 1.7983）、耗时 ≈46 s。
    """
    train, val, itos = build_data()
    dev = get_device()

    torch.manual_seed(seed)
    gen_train = torch.Generator().manual_seed(seed)
    gen_eval = torch.Generator().manual_seed(seed + 1)

    model = CharGPT(vocab_size=len(itos)).to(dev)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    model.train()

    res = RunResult(seed=seed)
    t0 = time.perf_counter()

    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0 or step == STEPS:
            tr = estimate_ce(model, train, gen_eval)
            va = estimate_ce(model, val, gen_eval)
            res.evals.append((step, tr, va))
            print(f"  step {step:5d}  train {tr:.4f}  val {va:.4f}")
        if step == STEPS:
            break

        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train, BATCH, BLOCK, gen_train)
        _, loss = model(x.to(dev), y.to(dev))
        assert loss is not None
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        if warmup > 0:
            scale = min(1.0, (step + 1) / warmup)
            for group in opt.param_groups:
                group["lr"] = lr * scale
        opt.step()
        res.step_ce.append(float(loss.item()))

    res.seconds = time.perf_counter() - t0
    res.train_ce = res.evals[-1][1]
    res.val_ce = res.evals[-1][2]
    res.peak_gib = (
        torch.cuda.max_memory_allocated() / 2**30 if dev.type == "cuda" else 0.0
    )
    return res


def run_and_save(seed: int, tag: str) -> RunResult:
    """跑一次 + 落曲线 + 打摘要（胶水，已实现）。

    外部再掐一次表（`time.perf_counter()`）与 `res.seconds` 对照：
    CUDA 是异步的，内部计时若漏了 `synchronize` 会偏小 —— 两个数都打出来才看得出。
    """
    t0 = time.perf_counter()
    res = train_once(seed)
    wall = time.perf_counter() - t0
    res.tag = tag
    path = write_curve(res)
    print(
        f"  [{tag}] seed {seed} · val CE {res.val_ce:.6f} · train CE {res.train_ce:.6f}"
        f" · 内部 {res.seconds:.1f}s / 外部 {wall:.1f}s · 峰值 {res.peak_gib:.3f} GiB"
        f" · 曲线 {path.name}"
    )
    return res


def judge_repeat(tag_prefix: str = "") -> bool:
    """判据 A / B：同 config（seed 固定）连跑两次，全绿才 True。

    A 逐步一致：两次 run 的 `step_ce` 逐元素最大绝对差 < STEP_TOL
    B 终值一致：两次 run 的最终 val CE 差 **== 0**

    `tag_prefix` 给曲线文件名加前缀（默认设置用 `raw-`、确定性模式用 `det-`）——
    两块的 CSV 必须分开存，否则后跑的会把先跑的证据覆盖掉（2026-11-26 踩过）。
    """
    checks: list[bool] = []
    train, val, itos = build_data()
    device = get_device()
    print(
        f"设备 {device} · 语料 train {len(train)} / val {len(val)} · 词表 {len(itos)}"
        f" · {STEPS} 步 · lr {LR}"
    )

    try:
        first = run_and_save(SEED, f"{tag_prefix}same-1")
        second = run_and_save(SEED, f"{tag_prefix}same-2")
    except NotImplementedError:
        print("A 逐步一致：train_once 未实现  NG")
        return False

    # A 逐步一致
    if len(first.step_ce) != STEPS or len(second.step_ce) != STEPS:
        print(
            f"A 逐步一致：step_ce 长度 {len(first.step_ce)} / {len(second.step_ce)}"
            f"（要求 {STEPS}）  NG"
        )
        return False
    diffs = [abs(a - b) for a, b in zip(first.step_ce, second.step_ce, strict=True)]
    d_max = max(diffs)
    ok_a = d_max < STEP_TOL
    checks.append(ok_a)
    tag_a = "OK" if ok_a else "NG"
    print(f"A 逐步 CE 最大差 {d_max:.3e}（要求 < {STEP_TOL:.0e}）  {tag_a}")
    if not ok_a:
        bad = diffs.index(d_max) + 1
        print(
            f"   最差分叉在第 {bad} 步：{first.step_ce[bad - 1]:.9f} vs {second.step_ce[bad - 1]:.9f}"
        )

    # B 终值一致
    d_val = abs(first.val_ce - second.val_ce)
    ok_b = d_val == 0.0
    checks.append(ok_b)
    tag_b = "OK" if ok_b else "NG"
    print(
        f"B 最终 val CE {first.val_ce:.9f} / {second.val_ce:.9f}"
        f" 差 {d_val:.3e}（要求严格 0）  {tag_b}"
    )
    print(
        f"   耗时 {first.seconds:.1f}s / {second.seconds:.1f}s"
        f" · 峰值 {first.peak_gib:.3f} / {second.peak_gib:.3f} GiB"
    )
    print(f"   train CE {first.train_ce:.6f} / {second.train_ce:.6f}")

    return all(checks)


def main() -> None:
    """测试入口（纯胶水）。关 0 的第 ② 项（3 个 seed 的噪声带）在下一块补进来。"""
    results = [judge_repeat("raw-")]
    print(f"判据 {sum(results)}/1 绿")


if __name__ == "__main__":
    main()
