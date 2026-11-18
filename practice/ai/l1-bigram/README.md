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
| 2 | 交叉熵 / 困惑度 | **严格序**：`log2(V) > CE_unigram > CE_bigram`（bits/char，三个数都要报）+ **两路互证**（逐位置 vs 按对/按字频加权，差 < 1e-12）；留出集（后 20%）**默认口径必须 `inf`**，只算见过的对时 **必须升高** —— 稀疏/过拟合的证据；边界：`"axd"`（无可用对）→ `nan`、`""`/单字 → `nan` | ✅ 绿（判据 2/2 · 变异 6/6） |
| 3 | softmax | **平移不变**：logits 全体加常数 c（`1 / −5 / 100`），输出概率差 < 1e-12；**精确值**（`[log3, log1]` → 0.75、`[log1, log7]` → 0.125，同时钉住**同序**）；**归一 + 正性 + 保序**；**边界**（`[]` → `[]`、`[7.5]` → `[1.0]`）；**溢出现象**：`[1000, 0]` 直算抛 `OverflowError`、`[-1000, -1001]` 直算抛 `ZeroDivisionError`，稳定版两者都正常；**病态形状**（`[0, 1000]` → `[0.0, 1.0]`、`[1e6, 0]` → `[1.0, 0.0]`） | ✅ 绿（判据 2/2 · 变异 6/6） |
| 4 | 平滑 | 未见 bigram 在留出集上让 CE 变成 `inf`；add-k（k=1）之后变**有限**，且比 `inf` 前无法比较的状态给出可比的数字（add-1 / add-0.1 各报一个） | ⬜ 待写 |

## 实测数字

| 指标 | 实测 | 备注 |
|---|---|---|
| 语料长度 | **422** | `len(CORPUS)` |
| 总对数 | **421** | `len(CORPUS) - 1` |
| 不同 bigram 种数 | **350** | 421 对合并成 350 种 |
| 词表 V | **180** | `len(set(CORPUS))` |
| 行和最大者 | **28**（`'，'`） | 全语料 |
| 条件-全局差值 | **0.9787**（'面'→'包'，后继仅 1 种）｜ **0.1087**（'猫'→'又'，9 种后继） | 判据 1②；选「后继最少」的 context 最稳 |
| 切分（后 20%） | train **337** / held **85** | `int(422 * 0.8)` |
| V（train） | **158**（全语料 180） | 22 个字只出现在留出集 ⇒ 基线用 158 |
| CE 均匀基线（bits/char） | **7.3038**（ppl **158.00**） | `log2(158)` |
| CE unigram（训练集） | **6.7141**（ppl **104.99**） | 两路互证差 **0.0** |
| CE bigram（训练集） | **1.2943**（ppl **2.45**） | 两路互证差 **0.0** |
| CE bigram（留出集） | **`inf`**（默认口径）｜ **1.6935**（ppl **3.23**，只算见过的对） | 新字 **22** 个、未见对 **67/84 = 79.8%** |
| softmax 平移误差 | **0.00e+00**（c ∈ {1, −5, 100}） | 整数平移后减 max 得到**同一批数**，逐位相同；不是「没测」 |
| softmax 精确值 | **0.750000000000000**（`[log3, log1]`）｜ **0.125000000000000**（`[log1, log7]`） | 手算期望，差 < 1e-15；第二组是**升序输入**，专抓「实现偷偷排序」 |
| 稳定版 ↔ naive 互证 | **6.78e-21** | `[-10, 0, 10]` 上两版一致 ⇒ 稳定版没改数学 |
| `exp` 可用区间 | `exp(709)`=**8.218e+307** ｜ `exp(710)` → **`OverflowError`** ｜ `exp(-745)`=**5e-324** ｜ `exp(-746)`=**`0.0`** | 本机 3.14.8 实测；上溢抛异常、下溢**静默变 0** |
| naive 两种死法 | `[1000, 0]` → **`OverflowError`** ｜ `[-1000, -1001]` → **`ZeroDivisionError`** | 同一台机器、两种异常，防溢出不只防大数 |
| 稳定版极端值 | `[1000, 0]` → **`[1.0, 0.0]`** ｜ `[-1000, -1001]` → **`0.73106 / 0.26894`** | 差 1000 时小项下溢为 0.0；差 1 时精确（= `1/(1+e⁻¹)`） |
| 病态形状 | `[0, 1000]` → **`[0.0, 1.0]`** ｜ `[1e6, 0]` → **`[1.0, 0.0]`** | 抓「减首项」「减 mean」的唯一输入（见踩坑） |
| add-1 / add-0.1 后 CE | | 判据 4 |

## 怎么跑

```powershell
cd practice/ai/l1-bigram
uv sync
uv run python bigram.py          # 关卡 1：判据 2/2 绿
uv run python entropy.py         # 关卡 2：判据 2/2 绿
uv run python softmax.py         # 关卡 3：判据 2/2 绿
uv run ruff check .
```

单项验证（写完一个函数就验一个）：

```powershell
uv run python -c "from bigram import judge_counts as f; print(f())"
uv run python -c "from bigram import judge_probs as f; print(f())"
uv run python -c "from entropy import judge_train_order as f; print(f())"
uv run python -c "from entropy import judge_holdout as f; print(f())"
uv run python -c "from softmax import judge_shift_invariance as f; print(f())"
uv run python -c "from softmax import judge_overflow as f; print(f())"
```

## 文档依据（2026-11-16 / 11-17 查证）

| 来源 | 拿它确认了什么 |
|---|---|
| `docs.python.org/3/library/math.html` | `log(0)` 抛 **ValueError**（不会返回 `-inf`）；`exp` 溢出抛 **OverflowError**；`log2(x)` 比 `log(x, 2)` 更准 |
| `docs.pytorch.org/docs/2.15/.../CrossEntropyLoss.html` | input 必须是**未归一化 logits**；等价于 `LogSoftmax` + `NLLLoss`；`reduction` 默认 `'mean'`；`ignore_index` 默认 `-100` |
| `scikit-learn.org/stable/.../sklearn.metrics.log_loss.html` | log loss **就是**交叉熵；用**自然对数**（nats）；把概率 clip 到 `[eps, 1-eps]` 以避开 `log(0)` ⇒ **clip 是掩盖、add-k 才是治** |
| SLP3 Ch.3 *N-gram Language Models* | 交叉熵 = `-(1/N)Σ log P`；困惑度 = `2^H`（底数必须配对）；零概率 ⇒ 整篇困惑度报废 |
| `docs.python.org/3/library/itertools.html#itertools.pairwise` | 3.10+ 才有；产出 n−1 个相邻对，**返回迭代器**（要 `list()` 才能 len / 多次遍历） |
| `github.com/pytorch/pytorch` → `torch/nn/functional.py` | `log_softmax` 文档原文：*"While mathematically equivalent to log(softmax(x)), doing these two operations separately is slower and numerically unstable."* ⇒ 分开算不是少技巧，是真会坏 |
| 同上 → `aten/src/ATen/native/cpu/LogSoftmaxKernelImpl.h` | CPU kernel 第一步就是 `sum_j exp(data[j] - max_input)`；注释强调**运算顺序不可换**（输入数值大、差值小时会出数值问题） |
| `github.com/scipy/scipy` → `scipy/special/_logsumexp.py` | `scipy.special.softmax` 源码：`x_max = max(x)` → `exp(x - x_max)` → 除以和；`logsumexp` 用 `log1p(s) + a_max` 把 max 加回 ⇒ 官方库同样是「先减 max」 |
| 本机实测（Python 3.14.8） | `exp(710)` 抛 `OverflowError`、`exp(-746)` 返回 `0.0`（不抛）、`0.0/0.0` 抛 `ZeroDivisionError` |

⚠️ 版本陷阱：`math` 文档站当前展示的是 **3.15.2**，本机跑 **3.14.8** —— 3.15 新增的
`math.fmax` / `math.fmin` / `math.signbit` / `math.isnormal` **用不了**，别照抄。

## 踩过的坑

| 坑 | 现象 / 判据怎么抓 |
|---|---|
| **分母用「不同字数」而不是总字数** | `unigram_probs` 分母写成 `len(counts)` ⇒ 每个值放大 337/158 倍、概率和 = **2.13**；判据 ①「和为 1」当场炸 |
| **`log2(0)` 抛异常、不是返回 `-inf`** | `ce_unigram(train, held)` 撞上新字时 `math.log2(0)` **抛 `ValueError`** ⇒ 必须在取 log **之前**判 0 并 `return math.inf`；换成 clip / 1e-12 顶替 = 掩盖病因，判据看不出「模型说不可能」 |
| **跳过未见位置后分母没跟着变** | `skip_unseen=True` 时跳了位置却仍除以 `len(eval_text)` ⇒ 留出集 CE 从 **1.6935** 掉到 **0.3427**（「比训练集还好」）→ 判据「必须升高」红 |
| **把未见字删掉再收相邻对** | 删字会造出原文里不存在的对（`a b c d e` 删 `c` ⇒ 凭空出现 `(b,d)`）。真实语料里恰好没撞上 ⇒ **变异一度是绿的**；改用设计好的小语料 `tiny_train="adbc"` / `eval="axd"`（正确答 `nan`，删字法答 `0.0`）才抓得住 |
| **`if pairs == 0:` 判空列表** | `[] == 0` 是 **`False`**，守卫形同虚设 ⇒ 空串/单字走到 `0.0 / 0` → `ZeroDivisionError`。要写 `if not pairs:` |
| **`math.log2` 误写成 `math.log`** | 单位从 bit 变 nat，所有 CE 一起除以 **2.303** ⇒ 严格序仍成立（1.29→0.56）、留出集「升高」也仍成立，**只有两路互证能抓**（0.8972 vs 1.2943） |
| **守卫拿原文判「见过」** | `char not in train` 现在等价于 `char not in probs`；一旦关卡 4 加平滑/`<UNK>`，模型知道训练集没有的字，用原文判会提前 `return inf`，把平滑白做 |
| **`itertools.pairwise` 忘了 `list()`** | 返回迭代器，`len()` 抛 `TypeError`，遍历第二次是空的（一次性） |
| **保序判据写成 `all(a < b for a, b in [zip(x, x[1:])])`（多套一层方括号）= 假绿** | `for a, b in diff` 只从「只含 1 个元素的列表」里取到那个 zip，于是 `a` / `b` 变成**两个元组**，比较的是**元组字典序**：非单调的 `[0.1, 0.5, 0.4]` 也返回 `True`。正确写法是链式比较 `probs[0] < probs[1] < probs[2]` |
| **变异 M2（减 mean）/ M3（减首项）一开始漏网** | 原有极端输入 `[1000, 0]` 的首项**恰好是 max**、`mean` 位移只到 500（`exp(500)≈1.4e217` 仍在安全区）⇒ 两条变异全绿。补 `[0, 1000]`（首项是最小值）与 `[1e6, 0]`（差值 1e6 ⇒ mean 位移仍剩 5e5）才红。**判据强度看输入形状，不看断言条数** |
| **`zip` 静默截断（ruff B905）** | `zip([1,2,3], [4,5])` → 2 对，**不报错**；长度不等会被洗成「绿」。`strict=True`（3.10+）才抛 `ValueError`；`pyproject.toml` 开了 `B`，漏写 `strict=` 直接 lint 红 |
| **`abs(实测 − 期望)` 里把期望也算成 bool** | 写成 `judge1 = abs(...) < 1e-15` 就会丢掉实测值，README 的数字再也拿不回来 —— 断言用 bool，**打印要留原值** |
