# l1-bigram —— L1 词与概率（char-level bigram → 交叉熵 → softmax → 平滑）

> **起 2026-11-16（关卡 1）** —— 关卡型目录：跨天推进，不按天新开子目录，
> 四关按「步」分文件（`bigram.py` → `entropy.py` → `softmax.py` → `smooth.py`）。
> 目的：把「语言模型 = 给下一个字打分」这句话，落到能跑、能报数字的最小实现上。
> 不是交付物（`projects/` 才是），是 L1 的练习目录。
> 0 依赖、**0 次模型调用（0 token）** —— 全部标准库，不需要 `.env`。

## 四关与判据（四关全绿 = L1 过）

| # | 关卡 | 判据 | 状态 |
|---|---|---|---|
| 1 | 计数 → 条件概率表 | **计数守恒**：`sum(counts.values()) == len(CORPUS) - 1` **且** 每个 context 的行和 == 该字在 `CORPUS[:-1]` 里出现的次数（按 next 聚合会过第一条、炸第二条）｜**归一 + 条件性**：每个 context 满足三条 —— `Σ P == 1.0`（±1e-12）、**后继键集合 == counts 里的真实后继**、**每个值 ≥ 0**（`{'包':1.5,'面':-0.5}` 的和也是 1）；且条件分布 ≠ 全局字符分布 —— **自动挑「出现 ≥3 次且后继唯一」的 context**，断言点质量 `{唯一后继: 1.0}`，并用恒等式 `最大差 == 1 − P_global(唯一后继)`（实测 `0.9787`）。判据里**没有语料字面量、没有阈值** | ✅ 绿（判据 2/2 · 变异 5/5） |
| 2 | 交叉熵 / 困惑度 | **严格序**：`log2(V) > CE_unigram > CE_bigram`（bits/char，三个数都要报）；留出集（后 20%）上 bigram 的 CE **必须升高** —— 这是稀疏/过拟合的证据 | ⬜ 待写 |
| 3 | softmax | **平移不变**：logits 全体加常数 c，输出概率差 < 1e-12；**溢出对照**：`[1000.0, 0.0]` 直接 `math.exp` 抛 `OverflowError`，先减 max 再算则正常返回 | ⬜ 待写 |
| 4 | 平滑 | 未见 bigram 在留出集上让 CE 变成 `inf`；add-k（k=1）之后变**有限**，且比 `inf` 前无法比较的状态给出可比的数字（add-1 / add-0.1 各报一个） | ⬜ 待写 |

## 实测数字

| 指标 | 实测 | 备注 |
|---|---|---|
| 语料长度 | **422** | `len(CORPUS)` |
| 总对数 | **421** | `len(CORPUS) - 1` |
| 不同 bigram 种数 | **350** | 421 对合并成 350 种 |
| 词表 V | **180** | `len(set(CORPUS))` |
| 行和最大者 | | 哪个字、多少 |
| 条件-全局差值 | **0.9787**（'面'→'包'，后继仅 1 种）｜ **0.1087**（'猫'→'又'，9 种后继） | 判据 1②；选「后继最少」的 context 最稳 |
| CE 均匀基线（bits/char） | | `log2(V)` |
| CE unigram（训练集） | | |
| CE bigram（训练集） | | |
| CE bigram（留出集） | | |
| softmax 平移误差 | | 期望 < 1e-12 |
| add-1 / add-0.1 后 CE | | 判据 4 |

## 怎么跑

```powershell
cd practice/ai/l1-bigram
uv sync
uv run python bigram.py          # 全绿后：判据 2/2 绿
uv run ruff check .
```

单项验证（写完一个函数就验一个）：

```powershell
uv run python -c "from bigram import judge_counts as f; print(f())"
uv run python -c "from bigram import judge_probs as f; print(f())"
```

## 文档依据（2026-11-16 查证）

| 来源 | 拿它确认了什么 |
|---|---|
| `docs.python.org/3/library/math.html` | `log(0)` 抛 **ValueError**（不会返回 `-inf`）；`exp` 溢出抛 **OverflowError**；`log2(x)` 比 `log(x, 2)` 更准 |
| `docs.pytorch.org/docs/2.15/.../CrossEntropyLoss.html` | input 必须是**未归一化 logits**；等价于 `LogSoftmax` + `NLLLoss`；`reduction` 默认 `'mean'`；`ignore_index` 默认 `-100` |
| SLP3 Ch.3 *N-gram Language Models* | 交叉熵 = `-(1/N)Σ log P`；困惑度 = `2^H`；概率连乘会下溢 ⇒ 一律 log 相加 |

⚠️ 版本陷阱：`math` 文档站当前展示的是 **3.15.2**，本机跑 **3.14.8** —— 3.15 新增的
`math.fmax` / `math.fmin` / `math.signbit` / `math.isnormal` **用不了**，别照抄。

## 踩过的坑

（学到时补）
