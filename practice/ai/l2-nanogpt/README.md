# l2-nanogpt —— L2 注意力（单头 attention → 最小 Transformer block）

> **起 2026-11-19（关卡 0）** —— 关卡型目录：跨天推进，不按天新开子目录。
> **判据的唯一家 = 根目录 `README.md` 阶段表 L2 行**；本文件只记**状态**与**实测数字**，不抄判据全文。
> 与 L1 的 `l1-bigram`（0 依赖、纯标准库）不同：这里**有 GPU 依赖**（torch cu128 轮子，约 2–3 GB）。

## 五关与状态

| # | 关卡 | 状态 |
|---|---|---|
| **0** | 环境（cu128 轮子 / capability / 冒烟） | ✅ 绿（判据 2/2） |
| 1 | 单头算术（手算 vs torch、缩放要有数字） | ✅ 绿（判据 2/2 · 2026-11-22） |
| 2 | 掩码与位置（因果掩码、置换等变 vs 位置编码） | ✅ 绿（判据 2/2 · 2026-11-23） |
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
uv run python attention.py   # 关卡 1：判据 2/2 绿（①手算对账 ②缩放与熵）
uv run python masking.py     # 关卡 2 判据 ①（A/B/C/D 四项）
uv run python positions.py   # 关卡 2 判据 ②＋关卡入口：判据 2/2 绿
uv run ruff check .
```

## 实测数字（关卡 1 · 2026-11-22）

| 判据 | 实测 |
|---|---|
| ① 权重逐元素差（手算 vs torch） | **5.551e-17**（阈值 1e-12） |
| ① 输出逐元素差 | **2.220e-16** |
| ① 行和偏差 | torch **2.220e-16** ／ 手算 **0** |
| ② std（d=2/8/64/512，10 seed 平均） | **1.391 / 2.841 / 7.938 / 22.649**（理论 √d = 1.414 / 2.828 / 8.000 / 22.627） |
| ② 除以 √d 后的 std | **0.984 / 1.004 / 0.992 / 1.001**（与 d 无关） |
| ② 比值误差 | **2.09% / 0.86% / 1.75%**（阈值 5%） |
| ② d=512 未缩放 | maxw **1.0000**（≥0.9）· 熵 **0.2454 bit**（≤1） |
| ② d=512 除 √d 后 | maxw **0.2747**（≤0.5）· 熵 **5.3109 bit**（≥5，上界 log2(64)=6） |

### 判据 ② 的口径（写死成常量，改必须写理由）

| 常量 | 值 | 含义 |
|---|---|---|
| `TOKENS` | 64 | 熵上界 = log2(64) = 6 bit |
| `SEEDS` | 10 | ① 取平均的 seed 数（单 seed 噪声 ±15%，见下） |
| `RATIO_TOL` | 0.05 | 实测比与理论 √(d₂/d₁) 的相对误差上限 |
| `MAXW_COLLAPSE` | 0.9 | 未缩放时「坍缩」的 max weight 门槛 |
| `H_COLLAPSE` | 1.0 bit | 未缩放时「坍缩」的熵门槛 |
| `MAXW_REVIVED` | 0.5 | 除 √d 后 max weight 要掉到其下 |
| `H_REVIVED` | 5.0 bit | 除 √d 后熵要回升到其上 |

**为什么必须多 seed**：单 seed 的 std 比值实测在 **14.8 ~ 18.4** 之间乱摆（理论 16），`(64,64)` 里同一行的格子共享同一个 `q` 行向量、**不是独立样本**。10 seed 平均后落进 5%；30 seed 平均 15.98。

## 实测数字（关卡 2 · 2026-11-23）

文件：`masking.py`（判据 ①：`causal_mask` / `masked_scores` / `causal_weights` / `causal_attention`）＋ `positions.py`（判据 ②：`sinusoidal_pe` / `rope`，关卡入口也在这个文件）。

| 判据 | 实测 |
|---|---|
| ① D 行非零个数 | `[1,2,3,4,5,6,7,8]` == 期望 `i+1` |
| ① D 行和偏差 | **2.220e-16** |
| ① A 改第 5、6 个 token → 前 4 个位置 | Δ **0.000e+00**（严格 0） |
| ① B 只改第 6 个 → 第 5 个位置 | Δ **0.000e+00**（严格 0） |
| ① C 不掩码对照组（同一扰动） | Δ **2.632e-02**（> 0，证明扰动真会传到过去） |
| ② A 无位置编码的置换差 | **4.441e-16**（< 1e-12，等变） |
| ② B 加正余弦后的置换差 | **2.298**（> 1e-3） |
| ② C 加 RoPE 后的置换差 | **1.037**（> 1e-3） |
| ② 自检 RoPE 范数变化 | **2.220e-16**（旋转正交，保长度） |
| 独立复核：自写 vs 官方 `F.scaled_dot_product_attention(is_causal=True)` | 输出差 **2.220e-16** |
| 独立复核：`rope` vs 显式旋转矩阵 `R_m x_m` | 差 **0.0**；`R^T R − I` 偏差 **1.11e-16** |
| 独立复核：`sinusoidal_pe` vs 纯 Python `math` 公式 | 差 **0.0** |

### 判据 ① 的口径（改必须写理由）

阶段表原文第三子句写作「只改第 6 个 → 第 5 个 Δ > 0」，但掩码正确时它**必须**是 0，否则与前半句自相矛盾。⇒ 落地为**两路取证**：「掩码版 = 0」+「同一扰动在不掩码版的 Δ > 0」。缺后一条时，`Δ = 0` 可能是扰动没生效 / 权重恒等 / 算错成 0，判据会空转。

### 判据 ② 的口径（改必须写理由）

必须用**不掩码**的 attention（关卡 1 的 `attention`）。因果掩码本身依赖位置（第 i 行只看前 i+1 个），重排后掩码语义已变 ⇒ 拿掩码版测「置换等变」，差必然很大，但那是掩码造成的，与位置编码无关。

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
| **判据脚本里打 `✓`/`✗` 会崩** | 中文 Windows 控制台是 GBK，`✓`（U+2713）/`✗`（U+2717）触发 `UnicodeEncodeError: 'gbk' codec can't encode character` ⇒ 判据输出只用 ASCII 标记（`OK` / `NG`）。与本目录 `1024³` 那条是同一个坑 |
| **单 seed 的统计量会被采样噪声主导** | `std` 比值同一份代码在 **14.8~18.4** 间跳（理论 16）—— `(64,64)` 同行共享同一 `q` 行，有效样本远小于 4096。⇒ 统计类判据一律**多 seed 取平均**再定阈值，先测噪声再定口径 |
| **`w.numel()` 不是行数** | 熵要按「行」平均（`w.shape[0]`）；误除 `numel()`（4096）会把 6 bit 缩成 0.094，跨 token 数不可比 |
| **hardlink 降级警告** | 缓存与 venv 不在同一分区 ⇒ `Failed to hardlink files; falling back to full copy`（只影响安装耗时，不影响正确性；要消掉设 `UV_LINK_MODE=copy`） |
| **掩码的 `True` 是「参与」，`masked_fill` 填的却是 True 处** | SDPA 的 `attn_mask`：**True = 参与注意力**；`MultiheadAttention` 的 `key_padding_mask`：**True = 屏蔽**（官方 Note 明说相反）。`masked_fill(mask, v)` 是「mask 为 True 处填 v」⇒ 传「允许」语义的掩码必须 `~mask`。⚠️ `~` 只对 bool/int 有效：float 抛 `TypeError: ~ (operator.invert) is only implemented on integer and Boolean-type tensors`；int 上是**位运算**（`~[0,1,2] = [-1,-2,-3]`） |
| **`tril(diagonal=-1)` 会把第 0 行整行屏蔽成 `nan`** | `diagonal` 默认 0（自己看得见自己）；`-1` ⇒ 第 0 行全 False ⇒ `softmax(全 -inf) = nan`（实测），反向传播会污染整个 batch；`+1` ⇒ 每行多留一个未来位（泄漏） |
| **高维 `.T` 已 deprecated** | torch 2.11 实测告警：`The use of x.T on tensors of dimension other than 2 to reverse their shape is deprecated` ⇒ 2D 用 `.T`，高维用 `.mT` 或 `x.transpose(-2,-1)`。另：1 维 `.T` 是**空操作**（`(4,)` → `(4,)`），变竖排要 `unsqueeze(1)` |
| **`arange` 的 dtype 是推断的，不是你要的** | `arange(0,4,2)` 是 `int64`，除以 `d` 后落 **float32** ⇒ `10000 ** (-arange(0,4,2)/d)` 得 `0.009999999776482582`（与 float64 的 0.01 相对差 **2.2e-08**）。要 1e-12 量级的判据必须显式 `dtype=torch.float64` |
| **`std()` 默认无偏，`LayerNorm` 内部却相反** | `torch.std` 官方 2.15：`correction` 默认 **1**（Bessel，2.0 起由 `unbiased` 改名）；N=64 时比 `correction=0` 高 **0.79%**、N=4096 只差 **0.012%**。而 `nn.LayerNorm` 用的是**有偏**方差（等价 `correction=0`）⇒ 两处口径相反，别混用 |
