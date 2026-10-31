"""S0 第 5 章 · 实验：async / await —— 并发到底快在哪

目的（一条根）：`async def` 调用后什么都不做，只返回协程对象；`await` 才是执行点。
    所以「快」不是因为协程，而是因为 await 把控制权还给事件循环，循环趁机跑别的任务。

判据（跑完把数字填进本目录 README 的实测列）：
  1. serial()      顺序 await 两次 sleep 1          → ≈ 2.0 s
  2. gather_run()  asyncio.gather 一起提交两次       → ≈ 1.0 s，返回顺序 = 传入顺序
  3. taskgroup_run() asyncio.TaskGroup 一起提交      → ≈ 1.0 s（第二轮再写）
  4. blocking()    用 time.sleep 代替 asyncio.sleep   → ≈ 2.0 s（反例，第二轮再写）

跑法：uv run --no-project python bench.py
文档依据：docs.python.org/3/library/asyncio-task.html
"""

from __future__ import annotations

import asyncio
import time


async def nap(name: str, seconds: float = 1.0) -> str:
    """一个「耗时 IO」的替身：睡一下，然后回报名字。

    已经写好，它不是练习点 —— 但要看清一件事：
    `asyncio.sleep` 是 async 的，**await 它才会让出控制权**给事件循环。
    """
    await asyncio.sleep(seconds)
    return f"{name} done"


async def serial() -> list[str]:
    """判据 1：顺序 await 两次 → 总耗时 ≈ 2 s。

    TODO(你写)：依次 await nap("a") 和 nap("b")，两个结果收进 list 返回。
    提示：这是「看起来 async、其实一点没并发」的写法 —— 先跑出来当基准，
         不然等下 gather 快了 1 秒，你不知道那 1 秒是从哪来的。
    """
    return [await nap("a"), await nap("b")]


async def gather_run() -> list[str]:
    """判据 2：`asyncio.gather(nap("a"), nap("b"))` → 总耗时 ≈ 1 s。

    TODO(你写)：用 asyncio.gather 一次提交两个协程，返回结果列表。
    要点：gather 的返回顺序 = **传入顺序**，不是「谁先完成谁在前」。
         想验证这点，可以把 b 改成 sleep 0.2 —— 结果里 a 仍然排第一。
    """
    return list(await asyncio.gather(nap("a"), nap("b", 0.2)))


async def taskgroup_run() -> list[str]:
    """判据 3（第二轮）：用 asyncio.TaskGroup 一起提交 → 总耗时 ≈ 1 s。

    TODO(你写)：`async with asyncio.TaskGroup() as tg:` → `tg.create_task(...)` 两次，
    退出 with 块后再逐个 `.result()` 取值。
    要点：TaskGroup **不返回 list** —— 结果只能从各自的 Task 上取；
         而且「等待」发生在 with 块退出时，不是 create_task 那一刻。
    """
    async with asyncio.TaskGroup() as tg:
        task_a = tg.create_task(nap("a"))
        task_b = tg.create_task(nap("b", 0.2))
    return [task_a.result(), task_b.result()]


async def blocking() -> list[str]:
    """判据 4（第二轮 · 反例）：把 asyncio.sleep 换成 time.sleep，再 gather → 仍 ≈ 2 s。

    TODO(你写)：
      ① 在函数内定义一个同步函数 `def block(name): time.sleep(1); return name`
      ② 用 gather 同时提交 `block("a")` 和 `block("b")`
      ③ 观察：耗时回到 2 s，gather 一点忙都没帮上
    思考题（答案写进当天笔记，不要写在代码里）：
      为什么 gather 救不回来？如果把 time.sleep 换成 asyncio.to_thread(time.sleep, 1) 呢？
    """

    def block(name: str) -> str:
        time.sleep(1)
        return name

    return list(
        await asyncio.gather(
            asyncio.to_thread(block, "a"),
            asyncio.to_thread(block, "b"),
        )
    )


async def timed(label: str, coro) -> None:
    """计时外壳（已写好，机械活）：跑一个协程并打印耗时与返回值。"""
    t0 = time.perf_counter()
    got = await coro
    print(f"{label:<11} {time.perf_counter() - t0:5.2f} s  → {got}")


async def main() -> None:
    """按顺序打四条判据。第二轮再把下面两行取消注释。"""
    await timed("serial", serial())
    await timed("gather", gather_run())
    await timed("taskgroup", taskgroup_run())
    await timed("blocking", blocking())


if __name__ == "__main__":
    asyncio.run(main())
