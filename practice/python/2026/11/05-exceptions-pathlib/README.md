# 2026-11-05-exceptions-pathlib —— 异常链（`raise ... from` 三态）与 pathlib（路径是对象）（S0 · D5）

> 目的：把「异常包装 = 对外换一种说法、对内保住原始线索」落到三个属性上（`__cause__` / `__context__` /
> `__suppress_context__`），并用同一段 json 解析代码的**三个版本**做干净对照；再把手拼路径换成 `Path`：
> `/` 拼接、`mkdir`、`read_text` / `write_text`。
> 本目录两个文件：`exceptions.py`（关卡 1，T1-T4）+ `paths.py`（关卡 2，T5-T6）。
> 跑法：`uv sync` → `uv run python exceptions.py` / `uv run python paths.py`

## 判据（跑通才算过）

### `exceptions.py` · judge_chain（7 条断言）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `load_config(BAD_TEXT)`（`raise ... from exc`） | 抛 `ConfigError`；`__cause__` 是原始 `ValueError`（`JSONDecodeError`）；`__suppress_context__ is True` | ✅ 三条都过 |
| 2 | `load_config_implicit(BAD_TEXT)`（裸 `raise`） | `__cause__ is None`；`__context__` 非空；`__suppress_context__ is False` | ✅ 三条都过 |
| 3 | `load_config_cut(BAD_TEXT)`（`from None`） | `__cause__ is None`；`__suppress_context__ is True`；**`__context__` 仍然非空** | ✅ 三条都过 |
| 4 | 三个成功路径 | `CONFIG_TEXT` 解析出的 dict 完全一致 | ✅ |

> 第 3 组最反直觉：**`from None` 切的是"显示"，不是"记录"** —— 原异常照样躺在 `__context__` 里。

### `paths.py` · judge_pathlib（5 条断言）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `roundtrip(root, "中文测试")` | 读回内容严格等于写入内容 | ✅ |
| 2 | 返回的路径 | `path == root / "notes" / "day.txt"` | ✅ |
| 3 | 目录层 | `path.parent.name == "notes"` 且 `path.exists() is True` | ✅ |
| 4 | `/` 拼接语义 | `(root / "a" / "b").parent.name == "a"`；`(root / "/abs").name == "abs"`（绝对路径重置） | ✅ |
| 5 | **字节层** | `path.read_bytes() == text.encode("utf-8")` | ✅ 这条才真正守住 encoding |

**实跑记录（2026-11-05）**：`exceptions.py` → **`判据 1/1 绿`**；`paths.py` → **`判据 1/1 绿`**；`ruff check` / `format --check` 双绿。

## 变异测试（判据能不能被"变坏"抓出来）

| 变异 | 期望 | 实测 |
|---|---|---|
| T1 漏 `from exc`（退化成隐式链） | 判据 1 红 | ✅ 红 |
| T2 多写 `from exc` | 判据 2 红 | ✅ 红 |
| T3 的 `from None` 换成 `from exc` | 判据 3 红 | ✅ 红 |
| `roundtrip` 写入时砍掉最后一个字符 | 判据 1 红 | ✅ 红 |
| `roundtrip` 真写到 `day2.txt` 却谎报 `day.txt` | 判据 2 红 | ✅ 红 |
| `write_text` **不给 `encoding`** | 判据 5 红 | ✅ 红（下沉到字节层之后才抓得住） |
| 显式用 `gbk` 写（模拟"别人机器默认 GBK"） | 判据 5 红 | ✅ 红 |

### 两条变异测试自己的教训

1. **无效变异**：第一次拿 `text.upper()` 当变异，结果**没变红** —— 因为 `"中文测试".upper()` 还是 `"中文测试"`，中文没有大小写。变异测试的前提是**这次改动真的改变了行为**，"我改过了"不算。
2. **行为对了但判据看不到**：`write_text` 不给 `encoding` 时判据照样绿 —— 同机往返用的是同一个默认编码。要抓它必须**下沉到字节层**（`read_bytes()` vs `text.encode("utf-8")`），这也是判据 5 的由来。

## 官方依据（2026-11-05 查证 · 四页）

- **`reference/simple_stmts`**：`raise ... from` 的 `from` 子句设置 `__cause__`；未捕获时**两段 traceback 一起打印**，中间那句是 "The above exception was the direct cause of the following exception"。
- **`library/exceptions`**：`from` 后面必须是**异常或 `None`**；设置 `__cause__` 会**隐式把 `__suppress_context__` 置 True**；`raise ... from None` 在**显示层面**替换旧异常，但旧异常仍留在 `__context__` 供调试。
- **`tutorial/errors`**：`raise RuntimeError from None` 的输出对照（只打一段）。
- **`library/pathlib`**：`/` 等价于 `joinpath`；**参数是绝对路径时前面整段被忽略**（Windows 上盘符不重置）；`read_text` / `write_text` 的用法；`exists()` 对"不存在/不可访问"都返回 False。

## 怎么跑

```powershell
cd practice/python/2026/11/05-exceptions-pathlib
uv sync
uv run python exceptions.py     # 关卡 1：判据 1/1
uv run python paths.py          # 关卡 2：判据 1/1
uvx ruff check . ; uvx ruff format --check .
```

## 踩过的坑

- **`except` 里忘写 `from exc` 会静默降级成隐式链**：异常照样抛，但 `__cause__` 变成 `None`，traceback 中间那句从 "direct cause" 变成 "During handling"。ruff **B904** 专治这个；本练习的 T2 是**故意**不写，所以行尾 `# noqa: B904` 并注明。
- **`noqa` 必须写在同一行**：写在上一行不生效（诊断挂在 `raise` 那一行）。
- **`from None` 不等于"原异常消失了"**：`__context__` 里还在，`traceback` 里看不到而已。
- **`write_text` 不会建目录**：目录不存在直接 `FileNotFoundError`，得先 `mkdir(parents=True)`。`exist_ok=True` 让它幂等（第二次跑判据不炸）。
- **`exist_ok=True` 只放过"已经是目录"**：同名路径是个**文件**时，照样 `FileExistsError`。
- **`write_text` 返回的是字符数不是字节数**：`"中文abc"` 返回 **5**，而 UTF-8 下实际写入了 9 字节。
- **`encoding` 这条判据在同机往返下抓不到**：默认编码在写和读两边一致，中文照样往返成功。必须**下沉到字节层**（`read_bytes() == text.encode("utf-8")`）才能证明"真的是 UTF-8"。
- **`root / "/abs"` 会丢掉前面整段**：实测结果是 `<盘符>:\abs` —— 绝对路径重置，但 Windows 上**盘符保留**。
- **判据自己造临时目录**：用 `tempfile.TemporaryDirectory()`，用完自动删。写到练习目录里会污染 `git status`，还会让第二次跑判据读到上一次的残留。
- **变异测试要先确认"改动真的改变了行为"**：`upper()` 对中文无效，那种"变异"红不了是正常的，别把它记成"判据不行"。
