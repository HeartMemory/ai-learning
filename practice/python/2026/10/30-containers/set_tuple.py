"""S0 第 4 章 · 实验二：set 与 tuple —— 「可哈希」是通行证

目的：一条根解释三个现象
  - set 的元素、dict 的键，为什么都要求「可哈希」
  - tuple 为什么能当键，list 为什么不能
  - 为什么 `{1, True}` 里只剩一个元素

判据（四条都要看到，跑完把数字填进本节 README 的实测列）：
  1. dedup([...]) 去掉重复后元素个数变少，且 sorted() 结果与手工去重一致
  2. hash_probe 探针：list / dict / set 不可哈希（TypeError: unhashable type）；
     str / int / tuple(只含不可变) 可哈希
  3. 递归陷阱：(1, 2) 可哈希，但 (1, [2]) **不可哈希** —— tuple 只是"外壳不可变"
  4. 把 list 转成 tuple 之后就能当 dict 的键；`{1, True}` 只剩 1 个元素

跑法：uv run --no-project python set_tuple.py
"""

from __future__ import annotations

from typing import Any


def dedup(items: list[Any]) -> set[Any]:
    """把列表里的重复元素去掉，返回集合。

    判据：len(返回值) < len(items)；且 sorted(返回值) == 手工去重后的排序结果。
    """
    return set(items)


def hash_probe(obj: Any) -> tuple[bool, str]:
    """探针：这个对象能不能进 set / 当 dict 键？

    返回 (能与否, 说明)：
      - 能   → (True,  "hash=...")
      - 不能 → (False, "TypeError: unhashable type: 'xxx'")

    判据：list / dict / set → False；str / int / tuple(纯不可变) → True；
          (1, [2]) → False（tuple 里装了可变对象）。
    提示：用 try/except TypeError 抓，别用 type() 判断（第三方对象也能可哈希）。
    """
    try:
        h = hash(obj)
        return True, f"hash={h}"
    except TypeError as e:
        return False, f"TypeError: {e}"


def to_key(obj: Any) -> Any:
    """把可能嵌套的可变结构转成同构的不可变结构（可哈希）。

    映射规则：
      list / tuple     → tuple（逐个元素递归）
      dict             → frozenset{(k, v), ...}（键值对递归）
      set / frozenset  → frozenset（逐个元素递归）
      其他（str/int/None…）→ 原样返回（叶子）

    为什么 dict 用 frozenset：dict 本身不可哈希，必须"摊平成键值对集合"；
    frozenset 不可变、可哈希。代价 = 丢失顺序（顺序敏感就改用
    tuple(sorted(items))，但 sorted 要求键可比较，异构键会抛 TypeError）。

    已知边界：
      - 自引用结构（a = [1]; a.append(a)）→ RecursionError，逻辑上无解，不处理
      - list 与 tuple 会转成同一个键（都走 tuple 分支）—— 若要区分得另加标记
      - 返回类型是 Any：叶子原样返回，容器返回 tuple / frozenset
    """
    # ① 三个容器分支先分流
    if isinstance(obj, dict):
        # 生成器直接喂给 frozenset，不用先建中间 list
        return frozenset((to_key(k), to_key(v)) for k, v in obj.items())
    if isinstance(obj, (list, tuple)):
        # tuple 也要递归：tuple 里可能塞着 list（(1, [2]) 就不可哈希）
        return tuple(to_key(x) for x in obj)
    if isinstance(obj, (set, frozenset)):
        return frozenset(to_key(x) for x in obj)
    # ② 叶子：str / int / float / bool / None / 本身可哈希的自定义对象
    return obj


def main() -> None:
    """把四条判据跑出来并打印，数字抄进 README 的实测列。"""
    # ── 判据 1：去重（手写一份"手工去重"来对照，别再用 set 自己验自己）
    raw = [3, 1, 2, 3, 1, 1]
    got = dedup(raw)
    manual: list[Any] = []
    for x in raw:
        if x not in manual:
            manual.append(x)
    print(
        f"[1] dedup: {len(raw)} 个 → {len(got)} 个 | "
        f"sorted={sorted(got)} | 手工={sorted(manual)} | 一致={sorted(got) == sorted(manual)}"
    )

    # ── 判据 2：探针（哪些能进 set / 当 dict 键）
    print("[2] hash_probe:")
    for obj in ([1, 2], {"a": 1}, {1, 2}, "abc", 42, (1, 2)):
        ok, info = hash_probe(obj)
        print(f"    {type(obj).__name__:>9} → 可哈希={ok}  {info}")

    # ── 判据 3：tuple 的递归陷阱（外壳不可变 ≠ 里面不可变）
    ok_plain, _ = hash_probe((1, 2))
    ok_nested, info_nested = hash_probe((1, [2]))
    print(f"[3] (1, 2) 可哈希={ok_plain} | (1, [2]) 可哈希={ok_nested}  {info_nested}")

    # ── 判据 4：list 直接当键会炸；转 tuple 后能当键；{1, True} 只剩 1 个
    try:
        _bad = {[1, 2]: "v"}
    except TypeError as err:
        print(f"[4] list 直接当键 → TypeError: {err}")
    dict_by_tuple = {to_key([1, 2]): "v"}
    print(f"    to_key([1, 2]) 当键 → {dict_by_tuple}")

    # 关于 B033：ruff 会报"集合里有重复元素"，但这里**故意**写重复值 ——
    # 就是要看 1 == True == 1.0 且 hash 相同 ⇒ 三个只留一个。lint 不懂意图，故按需豁免。
    mixed = {1, True, 1.0}  # noqa: B033
    print(f"    集合 {{1, True, 1.0}} 元素数 = {len(mixed)} → {mixed}")

    # ── 附加：多层嵌套也能当键（进阶判据）
    nested = [1, [2, {"a": 3}, (4, [5])]]
    key = to_key(nested)
    print(f"[5] 多层嵌套键 = {key}")
    print(
        f"    可哈希={hash(key) is not None} | "
        f"list 与 tuple 同键={to_key([1, 2]) == to_key((1, 2))} | "
        f"不同内容不撞={to_key([1, 2]) != to_key([1, 3])}"
    )


if __name__ == "__main__":
    main()
