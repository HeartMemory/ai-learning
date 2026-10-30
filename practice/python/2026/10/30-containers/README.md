# 2026-10-30-containers —— dict / set / tuple（S0 第 4 章）

> 目的：把「**可哈希 / 不可变**」这条根，落到 dict 键、set 元素、tuple 当键三个场景；顺带产出第一份能直接喂给模型的 tool schema。
> 官方依据（Python **3.14.10** Tutorial §5 + Built-in Types）：键可以是**任何不可变类型**；`list` 不能当键（可原地改）；`tuple` 只有递归地只含 `str` / 数字 / `tuple` 时才能当键；`list(d)` 按**插入顺序**返回键；`{}` 是空 **dict**，空集合必须写 `set()`。
> 版本注意：`frozendict` 是 **3.15** 的新增（PEP 814），本机 **3.14.8** 用不到，别照 3.15 文档写。

## 判据（跑通才算过，实测数字填右侧）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `dict_schema.py` · `build_tool_schema` | 返回的 dict 能被 `json.dumps(ensure_ascii=False)` 序列化；`返回["function"]["parameters"]["required"] is required` 为 `True` | ✅ 序列化成功（`add` 工具，中文不转义）；`required is` 入参 list = **True**；`tool["type"]` = `function` |
| 2 | `dict_schema.py` · 缺失键三连 | `d["x"]` 抛 `KeyError`；`d.get("x")` → `None`；`d.get("x", [])` 每次新建 ⇒ 两次**不是同一对象** | ✅ `KeyError` / `None` / `e1 is e2` = **False**（id `…261952` ≠ `…263552`）⚠️ 必须**先各绑变量再比**，内联 `id(...) == id(...)` 会因地址复用误判成 True |
| 3 | `dict_schema.py` · 合并两连 | `a \| b` 生成新对象（id 变、`a` 内容不变）；`a \|= b` 原地（id 不变） | ✅ `merge_new`：新 id `…437248` ≠ `a` 的 `…435520`，`a` 仍为 `{'k1': 1}`；`merge_inplace`：id 前后同为 `…435520`，内容变 `{'k1': 1, 'k2': 2}` |
| 4 | `set_tuple.py` · `dedup` | 去重后 `len` 变小，`sorted` 结果与手工去重一致 | ✅ **6 个 → 3 个**；`sorted` = `[1, 2, 3]` 与手工去重**一致**（手工用循环 + `not in`，不拿 set 验自己） |
| 5 | `set_tuple.py` · `hash_probe` | `list` / `dict` / `set` → 不可哈希（`TypeError: unhashable type`）；`str` / `int` / 纯不可变 `tuple` → 可哈希；`(1, [2])` → **不可哈希** | ✅ `list`/`dict`/`set` → **False**；`str`/`int`/`(1, 2)` → **True**（`hash(42)` = 42）；**`(1, [2])` → False**（外壳不可变，里面装了 list） |
| 6 | `set_tuple.py` · `to_key` | `[1, 2]` 当 dict 键抛 `TypeError`，转 `tuple` 后可当键；`{1, True}` 只剩 **1** 个元素 | ✅ `{[1,2]: ...}` → `TypeError: cannot use 'list' as a dict key`；`{to_key([1,2]): 'v'}` → `{(1, 2): 'v'}`；`{1, True, 1.0}` 元素数 = **1** |
| 7 | `set_tuple.py` · 多层递归（进阶） | 嵌套 list/dict/tuple 转换后仍可哈希；同内容同键、不同内容不撞 | ✅ `[1, [2, {"a": 3}, (4, [5])]]` → `(1, (2, frozenset({('a', 3)}), (4, (5,))))`；可哈希 **True**；`list` 与 `tuple` 同键 **True**；不同内容不撞 **True** |

跑法：`uv run python dict_schema.py`（纯标准库；若提示没有工程，可 `uv run --no-project python dict_schema.py`）

## 踩过的坑

| 坑 | 现象 | 正解 |
|---|---|---|
| **`id()` 是内存地址，会被复用** | 写 `print(id(d.get("x", [])), id(d.get("x", [])))` → 两个 id **相同**，误判成"同一个对象" | 比 id 的前提是**两个对象同时存活**：先 `a = ...; b = ...` 再 `a is b` / `id(a) == id(b)` |
| **Windows 控制台中文乱码** | 脚本里 `"加法运算"` 打印成乱码 | 设 `$env:PYTHONUTF8='1'` 再跑；这是终端编码，不是代码 bug |
| **本目录没装 ruff** | `uv run ruff` 报 `program not found` | 本练习是纯标准库、无工程环境，用 `uvx ruff check .` |
| **故意写重复元素被 lint 报** | `{1, True, 1.0}` 触发 B033（重复元素）—— 而这正是要演示的现象 | 行尾 `# noqa: B033` 豁免；**lint 不懂意图** |
| **注释以 `noqa` 开头也会被当指令** | 一行说明写成 `# noqa 说明：…` → 触发 **RUF100**（unused noqa） | 注释换个开头（`# 关于 B033：…`），`# noqa:` 只留给真要豁免的那一行 |
