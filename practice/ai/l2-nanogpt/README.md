# l2-nanogpt —— L2 注意力（单头 attention → 最小 Transformer block）

> **起 2026-11-19（关卡 0）** —— 关卡型目录：跨天推进，不按天新开子目录。
> **判据的唯一家 = 根目录 `README.md` 阶段表 L2 行**；本文件只记**状态**与**实测数字**，不抄判据全文。
> 与 L1 的 `l1-bigram`（0 依赖、纯标准库）不同：这里**有 GPU 依赖**（torch cu128 轮子，约 2–3 GB）。

## 五关与状态

| # | 关卡 | 状态 |
|---|---|---|
| **0** | 环境（cu128 轮子 / capability / 冒烟） | ✅ 绿（判据 2/2） |
| 1 | 单头算术（手算 vs torch、缩放要有数字） | ⬜ |
| 2 | 掩码与位置（因果掩码、置换等变 vs 位置编码） | ⬜ |
| 3 | 最小 block（形状 / 参数量 / LayerNorm / gradcheck） | ⬜ |
| 4 | 真训练（tinyshakespeare 2000 步、显存 < 6 GiB、val CE < bigram） | ⬜ |

## 实测数字（关卡 0 · 2026-11-19）

| 项 | 实测 |
|---|---|
| torch | **2.11.0+cu128**（编译期 CUDA **12.8**） |
| GPU | NVIDIA GeForce RTX 5060 Laptop · capability **(12, 0)** · 显存 **7.93 GiB** |
| 驱动 | 621.14 ｜ 运行时 SM 时钟 2557 MHz（max 3090） |
| matmul(1024³) | 首次 **388.8 ms** → 稳态 **0.65 ms**（**3.29 TFLOPS**） |
| attention(B=8,T=1024,d=64) 前向 | 首次 **148.7 ms** → 稳态 **1.50 ms** |
| 峰值显存 | **126 MiB** |
| softmax 行和偏差 | **2.38e-07** |
| 安装 | 11 个包 **1m00s**（torch 2.11.0+cu128）+ numpy 1 包 5.15s |

## 怎么跑

```powershell
cd practice/ai/l2-nanogpt
uv sync                      # 首次拉 cu128 轮子，约 2–3 GB
uv run python env_probe.py   # 关卡 0：判据 2/2 绿
uv run ruff check .
```

## 文档依据（2026-11-19 查证）

| 来源 | 确认了什么 |
|---|---|
| `download.pytorch.org/whl/cu128/torch/` 索引 | cu128 覆盖 **2.7.0–2.11.0**；**cp314（Python 3.14）Windows 轮子从 2.9.0 起**才有（2.7.x / 2.8.0 没有）⇒ 本机 Python 3.14.8 的实际下限是 **torch ≥ 2.9** |
| PyTorch 2.7 发布说明（pytorch.org） | 2.7 起支持 **Blackwell**，并提供 **CUDA 12.8 预编译轮子** |
| Clore.ai 兼容表 + 社区复现记录（gist） | Blackwell / sm_120 需要 **CUDA 12.8+ 且 torch ≥ 2.7**；不匹配时症状 = `no kernel image is available for execution on the device` |
| 本机实测 | `get_device_capability() == (12, 0)`；首次 vs 稳态耗时差 **≈600×** |

## 踩过的坑

| 坑 | 现象 / 怎么抓 |
|---|---|
| **首次调用耗时不是显卡性能** | 第一个 matmul 要付 CUDA context + cuBLAS 惰性加载的账：**388.8 ms** vs 稳态 **0.65 ms**（差 600×），attention 更夸张（首轮曾测到 **5443.7 ms** vs 1.50 ms）。⇒ 探针必须**预热 3 次、测 10 次平均**，首次与稳态两个数都报 |
| **PyPI 默认渠道的 torch ≠ cu128** | 不写 `[[tool.uv.index]]` + `[tool.uv.sources]` 就会装默认轮子；在 sm_120 上第一次 GPU 运算报 `no kernel image is available` —— **看着像代码错，其实是轮子错** |
| **`1024³` 在中文 Windows 控制台直接崩** | `UnicodeEncodeError: 'gbk' codec can't encode character '\xb3'`（`³` 编不出来）⇒ 打印文本只用 ASCII 上标 `1024^3` |
| **torch 不自动装 numpy** | 导入即告警 `Failed to initialize NumPy: No module named 'numpy'`；`uv add numpy` 后消失（关卡 4 的 char-level 数据也要用） |
| **hardlink 降级警告** | 缓存与 venv 不在同一分区 ⇒ `Failed to hardlink files; falling back to full copy`（只影响安装耗时，不影响正确性；要消掉设 `UV_LINK_MODE=copy`） |
