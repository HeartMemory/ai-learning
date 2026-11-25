# l2-nanogpt —— L2 注意力（单头 attention → 最小 Transformer block）

> **起 2026-11-19（关卡 0）** —— 关卡型目录：跨天推进，不按天新开子目录。
> **判据的唯一家 = 根目录 `README.md` 阶段表 L2 行**；本文件只记**状态**与**实测数字**，不抄判据全文。
> 与 L1 的 `l1-bigram`（0 依赖、纯标准库）不同：这里**有 GPU 依赖**（torch cu128 轮子，约 2–3 GB）。

## 六关与状态

| # | 关卡 | 状态 |
|---|---|---|
| **0** | 环境（cu128 轮子 / capability / 冒烟） | ✅ 绿（判据 2/2） |
| 1 | 单头算术（手算 vs torch、缩放要有数字） | ✅ 绿（判据 2/2 · 2026-11-22） |
| 2 | 掩码与位置（因果掩码、置换等变 vs 位置编码） | ✅ 绿（判据 2/2 · 2026-11-23） |
| 3 | 最小 block（形状 / 参数量 / LayerNorm / gradcheck） | ✅ 绿（判据 2/2 · 变异 4/4 · 2026-11-24） |
| 4 | 真训练（tinyshakespeare 2000 步、显存 < 6 GiB、val CE < bigram） | ✅ 绿（判据 2/2 · 2026-11-25） |
| 5 | Tokenization（手写 BPE ≥200 merge、encode/decode 往返、与 API token 数对账） | ✅ 绿（判据 3/3 · 2026-11-25 · 对账达标档 **1000 merge**, 相对差 10.81%） |

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
uv run python block.py       # 关卡 3 入口：判据 2/2 绿（① A–E · ② A–C）
uv run python corpus.py      # 关卡 4 块 1：语料/词表/切分/采样，判据 A–G
uv run python baseline.py    # 关卡 4 块 2：bigram 基线，判据 A–F（取出 2.4819）
uv run python model.py       # 关卡 4 块 3a：模型形状/参数/CE/因果，判据 A–E
uv run python train.py       # 关卡 4 块 3b：2000 步训练，判据 A–C（约 46 s，存 ckpt.pt）
uv run python sample.py      # 关卡 4 块 3c：生成 200 字 + 重复片段，判据 A–C
uv run python tokenizer.py   # 关卡 5：BPE 判据 C（底层）B（官方值）A（十句往返），3/3 绿
uv run python judge_api.py   # 关卡 5：判据 ② 与 API 对账（**会真实计费**，只手动跑）
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

## 实测数字（关卡 3 · 2026-11-24）

文件：`block.py`（`layer_norm` / `gelu` / `MiniBlock.__init__` / `MiniBlock.forward` 本人写；判据 `judge_block_shape` / `judge_backward` 由 AI 写；`cross_entropy` 无可复用代码 —— L1 的 `ce_bigram` 吃「字典 + 字符串」、单位 `log2`、不参与反传 —— 由 AI 代写后核对数值）。

| 判据 | 实测 |
|---|---|
| ① A 形状 | `(2,8,4)` → `(2,8,4)`，输出全有限 |
| ① B 参数量 | torch **244** == 手算 **244**（ln 8 + qkv 60 + proj 20 + ln 8 + ff 148） |
| ① C 有界 | 输入 absmax **3145.30** → LN 后 **1.7083**（上界 √3 = 1.7321）；未放大同值 1.7083 |
| ① D 口径（补） | 与 `nn.LayerNorm` 差：随机 **4.441e-16** / 小方差 **8.826e-15** |
| ① E 残差（补） | `\|out\|/\|in\|` = **1.0004** |
| ② A gradcheck | **True**（60 个入参 = 54 个参数元素 + 6 个输入，float64，约 0.95 s） |
| ② B CE 梯度 | 与 `(softmax − onehot)/N` 差 **2.776e-17**；每行梯度和 **3.166e-17** |
| ② C 手写 CE | 2.411830664494 vs 官方 2.411830664494，差 **4.441e-16** |

### 变异种子（4/4 全红）

| 变异 | 改坏什么 | 抓它的项 | 实测 |
|---|---|---|---|
| M1 | `correction=0` → `1`（无偏方差） | **① D** | 与官方差 随机 **2.289e-01** / 小方差 **8.059e-03** |
| M2 | `sqrt(var + EPS)` → `sqrt(var) + EPS` | **① D** | 小方差 **8.825e-01**（随机输入只差 7.58e-05，靠 `var≈1e-6` 才放大） |
| M3 | 去掉第一段残差（`x1 = out`） | **① E** | `\|out\|/\|in\|` = **0.0006**（正常 1.0004） |
| M4 | `cross_entropy` 的 `.mean()` → `.sum()` | **② C** | 手写 **14.47** vs 官方 **2.41** |

⚠️ **M1 / M2 / M3 在第一版判据下全部绿（漏网）**，因此补了两项，改判据必写理由：

- **D 口径对账**：`layer_norm` 与官方 `nn.LayerNorm` 逐元素差 < 1e-12，喂两组输入（随机 + `var≈1e-6`）。随机输入单独不够 —— M2 在随机数据上只差 **7.58e-05**，必须用小方差把 eps 位置的影响放大。
- **E 残差保真**：`|block(x)|max / |x|max ∈ [0.5, 2]`。这条 **gradcheck 也抓不到** —— 去掉残差后函数依然处处可微，只是塌成 0.001 量级。

## 实测数字（关卡 4 · 2026-11-25）

三块五文件：`corpus.py`（数据）· `baseline.py`（基线）· `model.py`（模型）· `train.py`（训练）· `sample.py`（生成）。
代码归属：**D2–D6 / B1–B3 / M1–M4 本人写**；训练主循环 `train_model` 与 `sample.py` 由 AI 代写（本人明确要求，以逐行讲解补，判据未降级）。

### 块 1 数据（判据 A–G，1/1 绿）

| 判据 | 实测 |
|---|---|
| A 语料 | **1,115,394** 字符（bytes == chars，纯 ASCII） |
| B 词表 | **65** · 互逆 True · 有序 True（`itos[0] == '\n'`） |
| C 往返 | 10 段逐字相等（含 `text[::997]` 非连续抽样） |
| D 切分 | train **1,003,854** + val **111,540** = 1,115,394 · val 起点对齐 |
| E batch | `(4,16)` int64 · 值域 [0,65) · `y` 是 `x` 右移一位 |
| F 采样覆盖 | `arange(2000)` 探针：起点 **[0, 1869]**（上界 1871）· 窗口连续 · 跨窗口右移 |
| G 可复现 | 同 seed 相同 · 换 seed 不同 |

### 块 2 bigram 基线（判据 A–F，1/1 绿）

| 判据 | 实测 |
|---|---|
| A 计数 | `sum(C_bi)` **1,003,853** == len−1 · `sum(C_uni)` **1,003,854** == len · 行和对账 True |
| B 行归一 | 最大偏差 **3.331e-16**（float64；float32 是 1.19e-07，撞不过 1e-12） |
| C 对账官方 | 自写 CE 2.463054321128 vs `F.cross_entropy` 2.463054321128，差 **0.000e+00** |
| D 平滑必要性 | val 未见对 **187 / 111,539 = 0.168%**；MLE（k=0）CE = **inf** |
| E 三档 k（val CE） | k=1 **2.4819** · k=0.1 2.4838 · k=0.01 2.4875 → 最优 **k=1**；unigram 3.3473 |
| F train/val | train **2.4546** < val **2.4819**（差 0.0273）｜bits/char 3.5412 / **3.5806** |

**判据 ② 的基线 = 2.4819 nats/char（k=1）**。扩档验证（不入判据）：k=2 2.4829 · k=5 2.4877 · k=10 2.4968 · k=30 2.5323 ⇒ 最优点就在 k≈1，网格够用。

### 块 3a 模型（判据 A–E，1/1 绿）

| 判据 | 实测 |
|---|---|
| A 形状 | logits `(2,16,65)` · loss 0 维有限 |
| B 参数量 | torch **826,368** == 手算 826,368（每层 198,272 × 4 + 8,320 + 16,384 + 256 + 8,320） |
| C CE 对账 | 模型 4.312124252 vs 手动 4.312124252，差 **0.000e+00** |
| D 因果（两路） | 改末位后 过去 **0.000e+00**（严格 0）· 末位 **1.750e+00**（> 0） |
| E 位置表 | `pos_emb (128, 128)` · 满长 128 前向通过 |

结构：C=128 · 4 层 · FFN 4× · **单头** · 注意力用官方 `F.scaled_dot_product_attention(is_causal=True)`（与关 2 自写版差 2.220e-16，等价性已在关 2 取证）。

### 块 3b 训练（判据 A–C，1/1 绿）

| 判据 | 实测 |
|---|---|
| ① 显存 | 峰值 **0.322 GiB**（上限 6.0）· 未 OOM |
| ① 步数 | **2000**（固定，16,384,000 token ≈ 19.8 token/参数） |
| ② val CE | **1.7983 < 2.4819**（低 0.6836，即 **−27.5%**） |

曲线（nats/char）：0 步 **4.3175** → 100 步 2.5367 → 200 步 2.4549（**约 150 步穿过基线**）→ 1000 步 2.0456 → 1900 步 **1.7974** → 2000 步 1.7983（末段 val 微升 = 过拟合开始）。train 末值 **1.5957**，gap 0.20。耗时 **45.7 s**（**22.9 ms/步**）。bits/char：6.23 → **2.594**。

### 块 3c 生成（判据 A–C，1/1 绿）

| 判据 | 实测 |
|---|---|
| A 形状 | 200 字符 · 全在词表 |
| **B 塌缩** | 最长重复子串 **5 字符**（上限 20）· 位置 130 · 内容 `' the '` |
| C 可复现 | 同 seed 相同 · 换 seed 不同 |

样本（seed 1337）：`\n\nCOMINIUS:\nMy will, is and the a see bober of and rud my call'd\nMy arm that us art to bar dings aners.\n\nHENRY BOTHAK:\n...` —— 学到了「人名行 + 空行 + 标点」的排版结构，词汇是坏的（1.8 nats 水平该有的样子）。

### 口径（改必须写理由）

| 项 | 口径 |
|---|---|
| CE 单位 | **一律 nats/char**；bits 仅作展示（÷ln2）。不跨语料比较 |
| 基线取哪档 | add-k 三档取**最优**（k=1）；故意取坏档 = 把判据做绿 |
| 「最长重复片段」 | **最长重复子串（至少出现两次、允许重叠）**，`O(n²)` DP；`"aaaa"` 的最长重复是 `"aaa"` |
| 解码参数 | **temperature 0.8 + top_k 40**，先定后用，不按结果回调 |
| 显存口径 | `max_memory_allocated`（tensor 实际占用），训练前 `reset_peak_memory_stats` |

## 实测数字（关卡 5 · 2026-11-25）

两个文件：`tokenizer.py`（BPE 本体，**T1–T8 方法体本人写**；判据 A/B/C 与 `_show`/`_expand` 由 AI 写）+ `judge_api.py`（判据 ② 对账脚本，AI 写，**本人手动跑**）。
口径与理由见 `tokenizer.py` 文件头 ①–⑧；下面是数字。

### 判据 C 底层手算（块 1 · 3/3）

| 项 | 实测 |
|---|---|
| C1 字节层 | 你好 **6 B** · 😉 **4 B** · 空串 **0 B** · 值域 `[0,255]` · 往返 True |
| C2 相邻对 | `[1,1,2,1,2,2,1]` → `{(1,1):1,(1,2):2,(2,1):2,(2,2):1}` == 手算 · 累加语义（传入 `{(9,9):5}` 不被清空）· 空/单元素 → `{}` |
| C3 merge | `[1,1,2,1,2]`+(1,2)→9 = `[1,9,9]` · **不重叠**（`[1,1,1]`→`[9,1]`）· **不就地改** · 元素守恒 |

### 判据 B 官方参考值（块 2 · 1/1）

| 项 | 实测 |
|---|---|
| 参考例 | `"aaabdaaabac"` + **3** merge ⇒ merges **3** 条 · 词表 **259** · 引用合法（`a,b < 256+k`） |
| encode | **`[258, 100, 258, 97, 99]`** == minbpe README 官值（逐元素） |
| decode | 往返 True（11 字符 → 5 token） |

### 判据 A 十句往返（块 2 · 10/10）

训练切片 **100,000** 字符（语料前 10 万，`TRAIN_CHARS`）· **200** merge · 词表 **456**。

| # | 句子类型 | 字符 → token | 字符/token |
|---|---|---|---|
| 0 | 莎士比亚英文 | 42 → 18 | **2.33** |
| 1 | 英文（含撇号） | 39 → 20 | 1.95 |
| 2 | 全大写 + 数字 + 感叹 | 27 → 27 | 1.00 |
| 3 | 空串 | 0 → 0 | — |
| 4 | 前后空格 | 31 → 18 | 1.72 |
| 5 | 中文 | 13 → 39 | **0.33** |
| 6 | 中英混排 | 19 → 28 | 0.68 |
| 7 | emoji | 10 → 17 | 0.59 |
| 8 | 全角标点 | 17 → 51 | **0.33** |
| 9 | 制表符 / 换行 | 12 → 32 | 0.38 |
| — | **合计** | 210 → 250 | 0.84 |

**10/10 逐字相等**。第 5 句的 0.33 = **3 token/汉字**，正是「语料里没有中文 ⇒ 退回 UTF-8 原始字节」的直接证据。

### 判据 ② 与 API 对账（块 3 · 通过 · 1000 merge 档）

对账文本 = 语料第 100,000 字符之后的 **2000** 字符（held-out，与训练切片不重叠）；差分消模板（base = 1 个字符 `"x"`）。

| 项 | 实测 |
|---|---|
| API 侧 | base **31** · sample **623** ⇒ 内容 token **592**（≈ 3.38 字符/token） |
| 200 merge（判据下界） | 我 **1013** token（1.97 字符/token）· 相对差 **71.11%** ❌ |
| 500 merge | 我 **774**（2.58 字符/token）· 相对差 **30.74%** ❌（只差 0.74 个点） |
| **1000 merge** | 我 **656**（3.05 字符/token）· 相对差 **10.81%** ✅ |
| 结论 | 判据 ② 取**能进 30% 的最小档 = 1000 merge**（相对差 10.81%） |
| 成本 | **2 次调用** · 输入 654 token · ≈ **¥0.002** |

⚠️ 三条如实记：

1. **200 merge 档是红的（71.11%）** —— 判据原文「≥ 200」是**下界**而不是等号；档位（200/500/1000）在跑之前就写死在 `judge_api.py` 的 `MERGE_SCAN` 里，不是事后挑出来的。
2. **500 档 30.74%，只差 0.74 个点** —— 30% 门槛附近对档位极敏感；若允许「跑完再挑档位」，这条判据等于可以随便做绿 ⇒ 口径必须**先写档位再跑**。
3. **中文不参与判 OK/NG** —— 本词表 0.33 字符/token vs 官方约 1.7 汉字/token，差 4 倍量级；根因是语料没覆盖，不是算法错。中文只作判据 ③ 的证据。

### 判据 ③ 为什么 token 数 ≠ 字数（三句话 + 数字）

1. **token 是字节/子词单位，不是字**：英文 **3.05 字符/token**（1000 merge 档）、中文 **0.33**、全角标点 **0.33**、空串 **0 个 token**。
2. **差多少只取决于词表覆盖**：同一段 2000 字符英文，我的 **200 / 500 / 1000** merge 分别切 **1013 / 774 / 656** 个 token，服务商切 **592** —— 算法是同一套 byte-level BPE，差的只是「合并多少次 + 语料覆盖什么」。
3. **所以不能按字数折算**：官方「1 英文字符 ≈ 0.3 token、1 中文字符 ≈ 0.6 token」只是平均值；本关实测中英文相差 **9 倍**（3.05 vs 0.33 字符/token）⇒ 混排文本按字数估算会差一个数量级。精确值只能看响应里的 `usage`。

## 文档依据（2026-11-19 查证）

| 来源 | 确认了什么 |
|---|---|
| `download.pytorch.org/whl/cu128/torch/` 索引 | cu128 覆盖 **2.7.0–2.11.0**；**cp314（Python 3.14）Windows 轮子从 2.9.0 起**才有（2.7.x / 2.8.0 没有）⇒ 本机 Python 3.14.8 的实际下限是 **torch ≥ 2.9** |
| PyTorch 2.7 发布说明（pytorch.org） | 2.7 起支持 **Blackwell**，并提供 **CUDA 12.8 预编译轮子** |
| Clore.ai 兼容表 + 社区复现记录（gist） | Blackwell / sm_120 需要 **CUDA 12.8+ 且 torch ≥ 2.7**；不匹配时症状 = `no kernel image is available for execution on the device` |
| 本机实测 | `get_device_capability() == (12, 0)`；首次 vs 稳态耗时差 **≈600×** |

### 关卡 5（2026-11-25 查证：手写 BPE / token 口径）

| 来源 | 确认了什么 |
|---|---|
| `karpathy/minbpe` README（BPE 参考实现） | 起点固定 **256 个单字节 token**；`词表 = 256 + merge 次数`；训练 = 反复「统计相邻对 → 合并最高频」；**encode 按学到的 merge 顺序（rank）逐条应用**，不重新统计；byte-level ⇒ 任何文本可编码、`decode(encode(x)) == x`；解释 token ≠ 字的三个原因（一个 token 可覆盖多字符 / 汉字 UTF-8 占 **3 字节** / emoji **4 字节**） |
| `openai/tiktoken` README | 同一套 byte-level BPE；`enc.decode(enc.encode(x)) == x`；平均约 **4 字节/token**；encode 实现 = **每轮取全局最小 rank 的那一对合并**（与「按 merges 顺序逐条应用」等价）；`mergeable_ranks` 是 `dict[bytes, int]`（⇒ 词表按字节串存，与本关口径 ② 一致） |
| DeepSeek 官方《Token 用量计算》 | **1 个英文字符 ≈ 0.3 token、1 个中文字符 ≈ 0.6 token**；「不同模型分词不同」；精确值以 `usage` 为准（⇒ 判据 ② 必须用差分对账，不能按字数折算） |
| ⚠️ 交叉核对发现的一处矛盾 | minbpe README 文字写「第 2 次合并 `(a,b)`」，其代码在平票时实际给 `(256,97)`；**两条路径的 encode 结果相同**（`[258,100,258,97,99]`）⇒ 判据只卡 encode 结果，不卡 merges 逐项；平票口径必须自己写死（`tokenizer.py` 口径 ③） |

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
| **3D 张量喂给只吃 2D 的注意力函数** | `causal_attention(q,k,v)` 收到 `(B,T,C)` ⇒ `k.T` 反转**所有**维变 `(4,8,2)`，`q(2,8,4) @ (4,8,2)` 报 `size of tensor a (2) must match the size of tensor b (4) at non-singleton dimension 0`。更阴的是 `causal_mask(q.shape[0])` 会拿 **B** 当序列长度（掩码 `(2,2)` 而非 `(8,8)`）⇒ **静默算错**。修法 = 逐 batch 调 2D 版再 `torch.stack(dim=0)` |
| **`nn.Linear` 的 weight 是 `(out, in)`** | 手写矩阵乘必须 `.T`：写成 `z @ w1`（`(1,4,4)` 对 `(16,4)`）报 `mat1 and mat2 shapes cannot be multiplied (4x4 and 16x4)`。2D 的 `.T` 合法，高维要用 `.mT` |
| **编辑器红线抓不到形状错** | Pylance 只验类型：`(2,8,4)` 与 `(8,4)` 同为 `torch.Tensor` ⇒ 一行都不标红，运行时才炸。补法 = docstring 写形状 + 入口 `assert x.ndim == 3` + 最小输入冒烟（`(1,2,4)`） |
| **判据脚本里打 `−`（U+2212）也崩** | 与 `✓/✗`、`1024³` 同一族：`UnicodeEncodeError: 'gbk' codec can't encode character '\u2212'`。`print` 里一律用 ASCII 减号 |
| **单头注意力里 `bk` 无效、`bv` 冗余、`bq` 有效** | softmax 平移不变 ⇒ `bk` 只给每行加同一常数，权重差 **5.55e-17**（完全无效）；`bv` 因权重行和为 1 退化成常数 `bv@wo`，与 `bo=bv@wo` 的输出差 **4.44e-16**（可被 `bo` 顶替）；只有 `bq` 有效（权重差 **0.077**，等于给每个 key 一个固定加分） |
| **`nn.Parameter` 不包裹 = 静默丢参数** | `self.w = torch.zeros(2)` 不会进 `parameters()`（**不报错**，优化器永远不更新它）；而漏调 `super().__init__()` 报 `AttributeError: cannot assign parameters before Module.__init__() call` |
| **`gather` 的三个硬约束** | `index.ndim` 必须等于 `input.ndim`（否则 `Index tensor must have the same number of dimensions as input tensor`）、dtype 必须 int64、值必须在 `[0, size(dim))`。所以 `targets (N,)` 要先 `unsqueeze(-1)`、取完再 `squeeze(-1)` |
| **`super().__init__()` 写在类末尾 = 赋值子模块时崩** | `self.ln1 = nn.LayerNorm(c)` 走 `nn.Module.__setattr__`，它要先查 `self._modules`（在 `super().__init__()` 里创建）⇒ 报 `AttributeError: cannot assign module before Module.__init__() call`。变体更阴：`self.blocks = [blk1, blk2]`（普通 list）被当普通属性塞进 `__dict__`，**不报错、不进 `parameters()`**、优化器永不更新（关卡 4 实测：用 `ModuleList` 才是 826,368，用 list 会少 793,088） |
| **`__init__` 的参数在 `forward` 里不存在** | `vocab_size` 是 `__init__` 的局部变量，`forward` 看不到 ⇒ `NameError: name 'vocab_size' is not defined`。改法：`logits.shape[-1]`（最稳）或 `self.head.out_features`。**这条 ruff 能提前抓**（`F821 Undefined name`）⇒ 写完方法体先 `ruff check .`，别等跑判据读 traceback |
| **`model.to(dev)` 不搬数据** | 参数上了 GPU、`get_batch` 的 `x/y` 还在 CPU ⇒ `RuntimeError: Expected all tensors to be on the same device, but got index is on cpu ... cuda:0`。每步 `x.to(dev), y.to(dev)`（2×64 KB）即可；把整份数据搬上去仍需 CPU 的 offset，省不掉 |
| **`torch.Generator` 绑定设备** | probs 在 cuda 上却传 CPU generator ⇒ `RuntimeError: Expected a 'cuda' device type for generator but found 'cpu'`。生成函数收 `seed: int`，内部 `torch.Generator(device=dev)` 再 `manual_seed` |
| **`randint` 的上界 ≠ 合法 offset 上界** | 合法 offset 最大 = `len - block - 1`；`randint(high)` 取不到 high ⇒ high 至少要 `len - block`。本目录用 `len-block-1` 是**保守一位**（无害）。**真正会炸的是 high 偏大**：`len-block+1` 起 y 右端钳到末尾、静默变短 → `stack expects each tensor to be equal size`（实测 len=10/block=3：o=6 合法、o=7 起短一位） |
| **控制台把 UTF-8 当 GBK 解码 = 一堆 `�`** | 跑出来 `A ���� 200 �ַ�` 不是代码错，是终端代码页问题。判别：看 `OK` / `判据 1/1 绿` 是否正常。解决：`$env:PYTHONIOENCODING='utf-8'`（本会话）或 `chcp 65001`。与 `1024³` / `✓` / `−` 同族：**控制台编码 ≠ 文件编码** |
| **GBK 打不出 emoji，判据一打印数据原文就崩** | `UnicodeEncodeError: 'gbk' codec can't encode character '\U0001f609'`；emoji（U+1F609）就是编不出来，与 `✓/✗`、`−`、`1024³` 同一族。⇒ 判据里凡要打印**数据原文**（含 emoji 的句子）都先过一层转义（`tokenizer.py::_show`，`backslashreplace`），标记本身仍只用 ASCII |
| **`max(stats, key=stats.get)` 被 Pylance 判错，运行却正常** | `dict.get` 的类型是 `(k) -> V \| None`，而 `max` 的 `key` 要求返回值可比较 ⇒ `reportCallIssue` / `reportArgumentType`。运行不炸是因为 `max` 遍历的 key 必然命中。改法：`key=lambda p: stats[p]`（类型是 `V`）。⚠️ 别改成 `key=lambda kv: (kv[1], kv[0])` —— 元组比较在**平票时会退回按 pair 字典序**，等于偷偷改掉口径 ③（`aaabdaaabac` 的第 1 条 merge 就从 `(97,97)` 变 `(97,98)`） |
| **默认参数 `None` 不兜底 = 第一个元素就崩** | `counts: dict \| None = None` 之后直接 `counts[pair] = ...` ⇒ `AttributeError: 'NoneType' object has no attribute 'get'`；Pylance 同时报 `reportOptionalMemberAccess` / `reportReturnType`。**类型注解就是提示**：写了 `\| None` 就必须在函数体第一行把它变成真对象 |
| **判据的「下界」被当成等号 = 可以把判据做绿** | 判据 ② 实测 200 merge 差 **71.11%**、500 merge 差 **30.74%**（只差 0.74 个点）、1000 merge 差 **10.81%**。若允许「跑完再挑档位」，30% 这条几乎必然能过 ⇒ 档位必须在跑之前写死（`judge_api.py::MERGE_SCAN`） |
