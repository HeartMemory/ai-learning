# 2026-10-30-mutability —— 可变性陷阱（S0 第 3 章）

> 目的：把「变量是名字，不是盒子」这条根，落到三个真实会炸的场景上。
> 对应笔记：`Notes/Python-基础笔记.md` 第 3 章（学到时展开成正文）
> 官方依据：Python Tutorial 4.9.1 与 Programming FAQ —— 默认值 `evaluated at the point of function definition` / `created exactly once`

## 判据（跑通才算过，实测数字填右侧）

| # | 实验 | 判据 | 实测 |
|---|---|---|---|
| 1 | 坏写法 `def f(a, L=[])` | 三次调用返回值累积 `[1]` → `[1, 2]` → `[1, 2, 3]`；`f.__defaults__[0]` 的 id 三次相同 | 待填 |
| 2 | 正确写法 `def g(a, L=None)` | `g(1)` / `g(2)` 的 id **不同**；`g.__defaults__ == (None,)`；传自己的列表时 `is` 为 `True` | 待填 |
| 3 | 函数传给函数（别名） | 被调函数 `append` 后，调用方的 list **也变了**（同一个对象） | 待填 |
| 4 | `+=` vs `+` | `lst += [x]` 会**原地改**（id 不变）；`lst = lst + [x]` 生成新对象（id 变） | 待填 |

## 怎么跑

```powershell
cd practice/python/2026/10/30-mutability
uv run python mut_default.py
```

纯标准库，不需要 `uv sync`，裸 `python mut_default.py` 也能跑。

## 踩过的坑

（学到时补）
