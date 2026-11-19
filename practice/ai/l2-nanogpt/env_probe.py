"""L2 关卡 0 · 环境探针（判据 ① 环境 / ② 冒烟）

判据全文的唯一家 = 根目录 `README.md` 阶段表 L2 行（**不要在这个文件里改判据**）：
  ① `torch.cuda.is_available()` 且 `get_device_capability() == (12, 0)`
  ② 1024³ matmul + 一次 attention 前向在 GPU 上跑通，报耗时与显存

为什么必须是 cu128：本机 **RTX 5060 Laptop = Blackwell sm_120**，而 cu126 及更早的轮子
里根本没有 sm_120 的内核 —— 症状是第一次 GPU 运算报
`no kernel image is available for execution on the device`（**不是代码写错**）。
又因为系统 Python 是 **3.14.8**，轮子还得是 cp314，cu128 渠道里从 **torch 2.9.0** 起才有 cp314。

本文件是**环境探针（机械胶水）**，由 AI 代写；从关卡 1 起回到「文件 + 框架、方法体自己写」。
跑法：进本目录 → `uv sync`（首次会拉 cu128 轮子，约 2–3 GB）→ `uv run python env_probe.py`
"""

import math
import time

import torch


def judge_env() -> bool:
    """判据 ①：CUDA 可用，且算力正是本机的 (12, 0)。"""
    available = torch.cuda.is_available()
    assert available, f"cuda.is_available() = {available} —— 轮子不是 cu128，或驱动/CUDA 不匹配"

    capability = torch.cuda.get_device_capability()
    name = torch.cuda.get_device_name()
    total_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    assert capability == (12, 0), (
        f"get_device_capability() = {capability}，本机应为 (12, 0) —— "
        "sm_120 需要 cu128 轮子 + torch ≥ 2.9（cp314）"
    )

    print(
        f"torch {torch.__version__} | 编译期 CUDA {torch.version.cuda} | "
        f"{name} | capability {capability} | 显存 {total_gib:.2f} GiB"
    )
    return True


def _sync() -> None:
    torch.cuda.synchronize()


def judge_smoke() -> bool:
    """判据 ②：1024³ matmul + 一次 attention 前向真在 GPU 上跑通，并报耗时 / 显存。

    ⚠️ 必须分「首次」和「稳态」两个数：第一次调用要付 CUDA context + cuBLAS/softmax
       kernel 的惰性加载账单（本机实测首次 matmul **311.9 ms**、attention 前向 **5443.7 ms**，
       而稳态只有 **0.51 ms / 1.94 ms**）—— 只报首次会把 5060 说成一块玩具卡。
    """
    device = "cuda"
    dtype = torch.float32
    torch.cuda.reset_peak_memory_stats()

    a = torch.randn(1024, 1024, device=device, dtype=dtype)
    b = torch.randn(1024, 1024, device=device, dtype=dtype)
    _sync()
    t0 = time.perf_counter()
    c = a @ b
    _sync()
    matmul_cold_ms = (time.perf_counter() - t0) * 1000
    assert torch.isfinite(c).all(), "matmul(1024^3) 出现 inf/nan"

    # 一次「最小 attention 前向」：B=8, T=1024, d=64
    batch, seq, dim = 8, 1024, 64
    q = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, dim, device=device, dtype=dtype)

    def attention_forward() -> tuple[torch.Tensor, torch.Tensor]:
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
        weights = torch.softmax(scores, dim=-1)
        return weights, weights @ v

    _sync()
    t1 = time.perf_counter()
    attn, out = attention_forward()
    _sync()
    attn_cold_ms = (time.perf_counter() - t1) * 1000
    assert out.shape == (batch, seq, dim), f"attention 输出形状 {tuple(out.shape)} 不对"
    assert torch.isfinite(out).all(), "attention 前向出现 inf/nan"

    # 预热 3 次再测 10 次的平均（稳态）
    warmup, repeats = 3, 10
    for _ in range(warmup):
        c = a @ b
        attention_forward()
    _sync()

    t2 = time.perf_counter()
    for _ in range(repeats):
        c = a @ b
    _sync()
    matmul_ms = (time.perf_counter() - t2) / repeats * 1000

    t3 = time.perf_counter()
    for _ in range(repeats):
        attn, out = attention_forward()
    _sync()
    attn_ms = (time.perf_counter() - t3) / repeats * 1000

    row_sums = attn.sum(dim=-1)
    assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-6), "softmax 行和不为 1"

    flops = 2 * 1024**3
    peak_mib = torch.cuda.max_memory_allocated() / 1024**2
    print(
        f"matmul(1024^3) 首次 {matmul_cold_ms:.1f} ms → 稳态 {matmul_ms:.2f} ms"
        f"（{flops / (matmul_ms / 1000) / 1e12:.2f} TFLOPS）| "
        f"attention(B=8,T=1024,d=64) 首次 {attn_cold_ms:.1f} ms → 稳态 {attn_ms:.2f} ms | "
        f"峰值显存 {peak_mib:.0f} MiB | softmax 行和偏差 "
        f"{(row_sums - 1).abs().max().item():.2e}"
    )
    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的，不是硬编码字符串。"""
    results = [
        judge_env(),
        judge_smoke(),
    ]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
