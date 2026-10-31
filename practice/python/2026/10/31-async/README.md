# 2026-10-31-async —— async / await 到底快在哪（S0 第 5 章）

> 目的：把「`async def` 调用不执行，`await` 才是执行点 + **让出点**」这条根，落到「顺序 await 慢、gather 快、同步阻塞救不回来」三个可测的场景上。
> 对应笔记：`Notes/Python-基础笔记.md` 第 5 章
> 跑法：`uv run --no-project python bench.py`（纯标准库，不用 `uv sync`）

## 判据（跑通才算过，实测数字填右侧）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | `serial()` 顺序 await 两次 `sleep 1` | 总耗时 ≈ **2.0 s**（基准：看起来 async，实际零并发） | **2.02 s** ✅ |
| 2 | `gather_run()` 用 `asyncio.gather` 一起提交 | 总耗时 ≈ **1.0 s**；返回 list 的顺序 = **传入顺序**（不是完成顺序） | **1.00 s** ✅，`b` 只睡 0.2 s 仍排在 `b` 位 —— 顺序 = `['a done', 'b done']` |
| 3 | `taskgroup_run()` 用 `asyncio.TaskGroup` | 总耗时 ≈ **1.0 s**；等待发生在**退出 `async with`** 时，不是 `create_task` 那一刻 | **1.00 s** ✅ |
| 4 | `blocking()` 同步阻塞 **反例 + 解药** | 直接 `time.sleep` + `gather` → ≈ **2.0 s**；改 `await asyncio.to_thread(time.sleep, 1)` → ≈ **1.0 s** | 反例 **2.00 s**（`a` 出来才轮到 `b` 进入）；`to_thread` 版 **1.00 s** ✅ |
| 5 | 一个子任务抛 `RuntimeError`，另一个需 1 s | `gather` **不取消**其余任务（它照跑完）；`TaskGroup` **取消**其余并抛 `ExceptionGroup` | `gather`：**0.05 s** 冒出异常，`safe` 仍在 **1.00 s** 正常跑完；`TaskGroup`：`safe` 被取消（打印"被取消"），抛 `ExceptionGroup`（内含 1 个 `RuntimeError`）✅ |

## 官方依据（2026-10-31 查证 · 三条来源）

- `docs.python.org/3/library/asyncio-task.html`：新版代码推荐 **`TaskGroup`（3.11+）**，`gather` 适用于"按传入顺序收集全部结果"。`gather` 的 Note 原文要点：子任务（含子任务派生的子任务）抛异常时 **TaskGroup 会取消其余任务，gather 不会**。
- **Python 3.14 What's New**：`gather` **无变更、未弃用**；`get_event_loop()` 不再隐式创建事件循环 ⇒ 入口用 `asyncio.run()`；`create_task()` 返回的 Task 被事件循环**弱引用**，不保存变量任务可能中途被回收。
- 结构化并发（Trio 教程，独立来源）：`async with` 作用域退出时**必定等待全部子任务**，异常**绝不静默丢弃** —— 这就是 TaskGroup 比 `gather` 更安全的根。
- 版本红线：**3.10 起所有 asyncio API 的 `loop` 参数已移除**，别再写 `asyncio.sleep(1, loop=loop)`。

## 怎么跑

```powershell
cd practice/python/2026/10/31-async
uv run --no-project python bench.py
```

四条判据都在 `main()` 里，跑一次全出。

## 踩过的坑

- **`...` 占位被 ruff 判 `PIE790`**，而且它会让函数**静默返回 `None`**（跑出来 `0.00 s`，容易骗自己"通过了"）⇒ 占位改 `raise NotImplementedError(...)`。
- **`asyncio.gather` 运行时返回 `list`，但类型存根说 `tuple`** ⇒ `-> list[str]` 会被 Pylance 报 `reportReturnType`。物证：`isinstance(r, list)` 为 `True`、`isinstance(r, tuple)` 为 `False`。解法 = `list(await asyncio.gather(...))`（注解保持诚实，别改成 `tuple` —— 注解要描述运行时）。
- **`gather` 冒出第一个异常后其余任务不取消**，但 `main` 一退出，`asyncio.run` 收尾会把它们 `cancel()` ⇒ 想看到"它照跑完"，必须在 `except` 之后**继续 `await` 一会儿**，否则看到的是"被取消"，结论会反过来。
- **`TaskGroup` 里 `.result()` 在块内取会 `InvalidStateError`**（任务还没完成）；`create_task` 也必须在 `with` 块内（组非活跃时传协程会被关闭并抛 `RuntimeError`）。
- **有 `await` ≠ 会让出**：`await` 一个内部无挂起的协程，控制权**不会**交出去（实测 `X前 → X后 → Y前 → Y后`，没交错）；要强制让一轮得写 `await asyncio.sleep(0)`。
- `ruff format` 会把 `nap("b",0.2)` 改成 `nap("b", 0.2)`、把多行 `gather` 收成 `list(await asyncio.gather(...))` —— 提交前跑一遍，别带着未格式化文件入库。
