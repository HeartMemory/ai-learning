"""L2 · 注意力 —— 关卡 4（块 3b/3c）：真训练（2000 步）

一条根：前面三块都在造「零件」，这一块才把它们接上电源：
    数据（块 1）→ 基线（块 2）→ 模型（块 3a）→ **梯度下降**（本块）
与前几关最大的不同：这里第一次出现「**模型自己在变**」。
    ⇒ 判据不能只看形状对不对，要看**数字有没有往对的方向走**：
       val CE 必须严格低于 bigram 基线的 2.4819 nats
    ⇒ 同时必须证明「**它没有偷偷超预算**」：显存峰值 < 6 GiB、不 OOM

⚠️ 三个口径，改必须写理由：
  ① **固定 2000 步**，不按 epoch、不按时间。步数固定 ⇒ 每步用的 token 数固定
     （64×128=8192）⇒ 总预算 16,384,000 token ⇒ 与 0.83 M 参数的比例 ≈ 19.8:1。
     换成「训到收敛」的话，两次跑的步数不同，判据不可比。
  ② **训练与评估用两个独立的 generator**。同一个 generator 会让评估的采样
     插进训练流，下次同样的 seed 走不到同样的数据 —— 可复现性就没了。
  ③ **显存取 `max_memory_allocated`，训练前 `reset_peak_memory_stats`**。
     官方文档：该值统计**tensor 实际占用**（`allocated ≤ reserved ≤ nvidia-smi`），
     不重置就是从进程启动算起，会把 3a 判据那次前向也算进来。

判据（本块自定 3 项，A–C）：
    A 显存：`max_memory_allocated` < 6 GiB，且全程不 OOM
    B 步数：恰好 STEPS（2000）步
    C 效果：val CE < BASELINE_CE（= 块 2 实测的 bigram 基线 2.4819 nats）
  附加（不计入）：报出 train CE、耗时、每 EVAL_EVERY 步的曲线

════════════════════════════════════════════════════════════════════
 TODO 清单
════════════════════════════════════════════════════════════════════
 T1 [实现] train_model(...)     训练主循环（核心练习：zero_grad → forward → backward → step）
 T2 [已写] load_data / get_device / estimate_ce / save_ckpt（胶水与测量，AI 写）
 T3 [已判据] judge_train()      A–C（已写好）

写完直接跑：
    uv run python train.py       # 判据 1/1 绿（约 1–3 分钟，日志每 100 步一行）
════════════════════════════════════════════════════════════════════
"""

import time
from pathlib import Path

import torch

from corpus import BATCH, BLOCK, build_vocab, encode, get_batch, load_corpus, split_ids
from model import CharGPT

SEED: int = 1337
"""全局种子。模型初始化 + 训练采样都由它派生 ⇒ 同一次提交可复现。"""

STEPS: int = 2000
"""判据 ① 的「固定 2000 步」（口径 ①）。"""

LR: float = 3e-4
"""学习率。小模型 + 无 warmup 时的常用稳值；太大（1e-3↑）loss 会抖，太小（1e-5）学不动。"""

WEIGHT_DECAY: float = 0.01
"""AdamW 的解耦权重衰减。0.01 是常见默认；本关不调它。"""

GRAD_CLIP: float = 1.0
"""梯度全局范数裁剪阈值。fp32 下通常用不到，但成本极低，留着防一次性大梯度。"""

EVAL_EVERY: int = 100
"""每隔多少步评估一次（含第 0 步与最后一步）。"""

EVAL_ITERS: int = 20
"""每次评估用多少个 batch 估计 CE（20 × 8192 ≈ 16 万 token，误差约 ±0.01）。"""

MEM_LIMIT_GIB: float = 6.0
"""判据 A 的显存上限（单位 GiB）。本机 8151 MiB，留桌面余量。"""

BASELINE_CE: float = 2.4819
"""块 2 实测的 bigram 基线（nats/char，k=1，val）。判据 C 要求严格低于它。"""

CKPT: Path = Path(__file__).with_name("ckpt.pt")
"""训练产物（`.pt` 已在根 `.gitignore` 里）。块 3c 生成时从这里读。"""


def load_data() -> tuple[torch.Tensor, torch.Tensor, list[str]]:
    """读语料 → 编码 → 切分（胶水，已实现）。"""
    text = load_corpus()
    itos, stoi = build_vocab(text)
    ids = encode(text, stoi)
    train, val = split_ids(ids)
    return train, val, itos


def get_device() -> torch.device:
    """选设备（胶水，已实现）。有 cuda 就走 GPU。"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def estimate_ce(
    model: CharGPT, data: torch.Tensor, generator: torch.Generator, iters: int = EVAL_ITERS
) -> float:
    """在 data 上估平均 CE（nats/char）；测量代码，已实现（不属于你的 TODO）。

    为什么用固定 batch 数而不是全量：val 有 11 万个字符，全量一次 872 个 batch，
    每 100 步都跑一遍等于训练时间翻 20 倍。20 个 batch 的蒙特卡洛误差约 ±0.01，
    对「必须低于 2.4819」这个判断足够。
    ⚠️ `torch.no_grad()` 不能省：否则 20 个 batch 的反向图全挂在显存里。
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


def train_model(
    model: CharGPT,
    train_data: torch.Tensor,
    val_data: torch.Tensor,
    gen_train: torch.Generator,
    gen_eval: torch.Generator,
) -> dict[str, object]:
    """T1 · 训练主循环。返回 dict（键名固定，判据按这些键取数）。

    必须做到（判据 A–C 全依赖它）：
      1. 优化器用 `torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)`
      2. 跑 **STEPS** 步；每步四件事（顺序不能换）：
             opt.zero_grad(set_to_none=True)
             x, y = get_batch(train_data, BATCH, BLOCK, gen_train)
             _, loss = model(x, y)
             loss.backward()
             torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
             opt.step()
      3. 每 **EVAL_EVERY** 步（以及第 0 步、最后一步）评估一次：
             tr = estimate_ce(model, train_data, gen_eval)
             va = estimate_ce(model, val_data, gen_eval)
         把 (step, tr, va) 记进 log 并打印一行（`OK`/数字，别用 ✓ 之类的非 ASCII 标记）
      4. 返回：
             {"steps": 实际步数, "train_ce": 最后一次 train CE,
              "val_ce": 最后一次 val CE, "log": [(step, tr, va), ...],
              "seconds": 总耗时（float）}

    提示：
      · 训练前 `model.train()`、评估前 `model.eval()`（本模型无 dropout，但这是习惯；
        更重要的是 `estimate_ce` 内部已经切好了这两个状态）
      · 每步 `float(loss.item())` 会强制 GPU 同步，2000 次影响不大；想更省就每 10 步取一次
      · `clip_grad_norm_` 放在 `backward()` 之后、`step()` 之前（顺序错了阈值就没意义）
      · 别在每步都调 `estimate_ce` —— 2000 × 40 次前向，训练时间直接翻倍

    自检：跑完打印的 val CE 应该一路下降；若第 0 步 ≈ 4.3、1000 步后仍 > 2.5，
    说明 lr 或数据流有问题（而不是「模型容量不够」）。
    """
    dev = next(model.parameters()).device
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    model.train()
    log: list[tuple[int, float, float]] = []
    train_ce = val_ce = float("nan")
    steps = 0
    t0 = time.perf_counter()

    for step in range(STEPS + 1):
        if step % EVAL_EVERY == 0 or step == STEPS:
            train_ce = estimate_ce(model, train_data, gen_eval)
            val_ce = estimate_ce(model, val_data, gen_eval)
            log.append((step, train_ce, val_ce))
            print(f"  step {step:5d}  train {train_ce:.4f}  val {val_ce:.4f}")
        if step == STEPS:
            break  # 最后一步只评估，不再更新
        opt.zero_grad(set_to_none=True)
        x, y = get_batch(train_data, BATCH, BLOCK, gen_train)
        _, loss = model(x.to(dev), y.to(dev))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        opt.step()
        steps += 1

    return {
        "steps": steps,
        "train_ce": train_ce,
        "val_ce": val_ce,
        "log": log,
        "seconds": time.perf_counter() - t0,
    }


def save_ckpt(model: CharGPT, itos: list[str]) -> None:
    """存一份权重 + 词表（胶水，已实现；块 3c 生成时用）。"""
    torch.save({"model": model.state_dict(), "itos": itos}, CKPT)


def judge_train() -> bool:
    """T3 · 判据 A–C（全绿才 True）。

    A 显存：训练前 `reset_peak_memory_stats`，训练后读 `max_memory_allocated` < MEM_LIMIT_GIB
    B 步数：恰好 STEPS
    C 效果：val CE < BASELINE_CE（2.4819 nats）
    """
    checks: list[bool] = []
    train_data, val_data, itos = load_data()
    device = get_device()
    print(f"设备 {device} · 语料 train {len(train_data)} / val {len(val_data)} · 词表 {len(itos)}")

    torch.manual_seed(SEED)
    model = CharGPT(vocab_size=len(itos)).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()  # 口径 ③

    gen_train = torch.Generator().manual_seed(SEED)
    gen_eval = torch.Generator().manual_seed(SEED + 1)  # 口径 ②：两个流分开

    oom = False
    t0 = time.perf_counter()
    try:
        info = train_model(model, train_data, val_data, gen_train, gen_eval)
    except NotImplementedError:
        print("A 显存：train_model 未实现  NG")
        return False
    except torch.OutOfMemoryError:
        oom = True
        info = {"steps": -1, "train_ce": float("nan"), "val_ce": float("nan"), "log": []}
        print("  训练中途 OOM")
    elapsed = time.perf_counter() - t0
    info.setdefault("seconds", elapsed)

    # 曲线由 train_model 边跑边打印，这里不再重复

    # A 显存
    peak_gib = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
    ok_mem = (not oom) and peak_gib < MEM_LIMIT_GIB
    checks.append(ok_mem)
    tag_a = "OK" if ok_mem else "NG"
    print(f"A 显存峰值 {peak_gib:.3f} GiB（上限 {MEM_LIMIT_GIB}）· 未 OOM {not oom}  {tag_a}")

    # B 步数
    ok_steps = info["steps"] == STEPS
    checks.append(ok_steps)
    tag_b = "OK" if ok_steps else "NG"
    print(f"B 步数 {info['steps']}（要求 {STEPS}）  {tag_b}")

    # C 效果
    val_ce = float(info["val_ce"])
    ok_ce = val_ce < BASELINE_CE
    checks.append(ok_ce)
    tag_c = "OK" if ok_ce else "NG"
    print(
        f"C val CE {val_ce:.4f} < 基线 {BASELINE_CE:.4f}（差 {BASELINE_CE - val_ce:+.4f}）"
        f" · train {float(info['train_ce']):.4f} · 耗时 {float(info['seconds']):.1f}s  {tag_c}"
    )

    if all(checks):
        save_ckpt(model, itos)
        print(f"   权重已存 {CKPT.name}（块 3c 生成用）")
    return all(checks)


def main() -> None:
    """测试入口（纯胶水）。"""
    results = [judge_train()]
    print(f"判据 {sum(results)}/1 绿")


if __name__ == "__main__":
    main()
