"""S0 缺口 · 实验：推导式 —— 三种筐 + 独立作用域

目的（一条根）：推导式写的是「规格」（从哪进料 / 哪些放行 / 放行后加工成什么），
    for 循环写的是「步骤」（建空表 → 遍历 → 判断 → 追加）。所以 `if` 的位置决定语义：
    写在 for 后面 = 过滤（安检门）；写进最前面的三元 = 分流（进来的都要，分两种加工）。

判据（跑完把数字填进本目录 README 的实测列）：
  1. even_squares([1, 2, 3, 4])            → [4, 16]
  2. even_squares([])                      → []（零次迭代即零次产出，不报错）
  3. even_squares([2])                     → [4]（推导式永远返回容器，不是标量）
  4. index_by_id(sample)                   → 键 = item["id"]，值 = item，内容相等
  5. got["a"] is sample[0]                 → True（搬运引用，不复制元素）
  6. unique_domains(3 条 URL)               → {"a.com", "b.com"}（集合自动去重）
  7. unique_domains([])                    → set()（注意 `{}` 是空 dict，不是空 set）
  8. n = 0 后跑 [n for n in range(3)]       → n 仍是 0（推导式不泄漏迭代变量）
  9. for n in range(3) 之后                 → n 变成 2（for 语句会泄漏，第 8 条的反面）

  ── 第二块：从「一个 for」升级到「多个 for」+ if 的两种位置 ──
 10. flatten([[1, 2], [3], []])             → [1, 2, 3]（多重 for，自左向右由外向内）
 11. label_parity([1, 2, 3])                → [("odd",1), ("even",2), ("odd",3)]（前置三元 = 分流）
 12. len 对比                                → 分流长度不变(3)，后置 if 的过滤长度会变(1)
 13. transpose([[1,2,3],[4,5,6]])           → [[1,4],[2,5],[3,6]]（嵌套推导式）
 14. 同上与 list(zip(*m)) 对照               → 结果一致（官方推荐的写法）
 15. transpose([])                          → []（**这里 guard 是必要的**，因为要取 matrix[0]）
 16. compile("[x, x**2 for x in range(3)]")  → SyntaxError（元组表达式必须加括号）
 17. [(x, x**2) for x in range(3)]           → [(0,0),(1,1),(2,4)]（加括号就合法）

跑法：uv run --no-project python comprehension.py
文档依据：docs.python.org/3.14/tutorial/datastructures.html §5.1.3·5.1.4
         docs.python.org/3.14/reference/expressions.html §6.2.5（独立隐式作用域）
         peps.python.org/pep-0709/（3.12 起内联，不再有独立函数帧）
"""


def even_squares(nums: list[int]) -> list[int]:
    """判据 1-3：只留偶数，平方后按原顺序；空输入返回空列表。

    TODO(你写)：用 list 推导式一行写完，不许 for + append。
    提示：`if` 写在 `for` 后面 = 安检门（不合格的根本不进筐）。
    """
    return [x**2 for x in nums if x % 2 == 0]


def index_by_id(items: list[dict]) -> dict[str, dict]:
    """判据 4-5：键 = item["id"]，值 = item 本身（不是副本）。

    TODO(你写)：用 dict 推导式，形如 `{键表达式: 值表达式 for ...}`。
    提示：值那一侧直接写变量名，就是原对象；写成 `dict(item)` 之类才是副本。
    """
    return {item["id"]: item for item in items}


def unique_domains(urls: list[str]) -> set[str]:
    """判据 6-7：域名 = 去掉协议后的第一段，重复的自动合并。

    TODO(你写)：用 set 推导式，形如 `{表达式 for ...}`。
    提示：`"http://a.com/x"` 按 `//` 切一刀再按 `/` 切一刀，取第一段。
    """
    return {url.split("//")[1].split("/")[0] for url in urls}


def flatten(matrix: list[list[int]]) -> list[int]:
    """判据 10：把两层列表压成一层。

    TODO(你写)：用**两个 `for`**（这叫多重 for，不是嵌套推导式）。
    提示：书写顺序 = 等价双层循环的顺序，外层在前：`for row in matrix for n in row`。
         写反（`for n in row for row in matrix`）是 **NameError: name 'row' is not defined**，
         不是 TypeError —— 因为**最左 `for` 的可迭代对象在外层作用域求值**，那时 row 还没绑定。
    """
    return [n for row in matrix for n in row]


def label_parity(nums: list[int]) -> list[tuple[str, int]]:
    """判据 11-12：**分流** —— 元素一个都不丢，只分两类。

    TODO(你写)：把三元表达式放在最前面（`... if ... else ...`），**不是**后置 if。
    提示：后置 `if` 是安检门（会丢元素）；前置三元是分流（长度不变）。
         这条判据就是拿它和 even_squares 的长度对比 —— 写法只差位置，语义完全不同。
    """
    return [("even" if x % 2 == 0 else "odd", x) for x in nums]


def transpose(matrix: list[list[int]]) -> list[list[int]]:
    """判据 13-15：转置 —— 嵌套推导式，内层在**外层 for 的上下文**中求值。

    TODO(你写)：外层按列号、内层按行，取出 `row[i]` 构成一行；空矩阵返回 []。
    提示：这里是**唯一一个 guard 该写的地方** —— 因为要 `matrix[0]` 取列数，
         空列表会直接 IndexError。跟 even_squares 那个多余的 `if not nums` 对照着看。
    追问：官方建议转置用 `list(zip(*matrix))`，那为什么还让你写这一版？
    """
    if not matrix:
        return []
    return [[row[i] for row in matrix] for i in range(len(matrix[0]))]


def main() -> None:
    # ---- 判据 1-3：list 推导式 ----
    assert even_squares([1, 2, 3, 4]) == [4, 16], even_squares([1, 2, 3, 4])
    assert even_squares([]) == []
    assert even_squares([2]) == [4]

    # ---- 判据 4-5：dict 推导式（第 5 条验「值不是副本」）----
    sample = [{"id": "a", "v": 1}, {"id": "b", "v": 2}]
    got = index_by_id(sample)
    assert got == {"a": {"id": "a", "v": 1}, "b": {"id": "b", "v": 2}}, got
    assert got["a"] is sample[0]

    # ---- 判据 6-7：set 推导式 ----
    assert unique_domains(["http://a.com/x", "https://b.com", "http://a.com/y"]) == {
        "a.com",
        "b.com",
    }
    assert unique_domains([]) == set()

    # ---- 判据 8-9：作用域（观察项，不用写代码，跑通即证明）----
    n = 0
    _ = [n for n in range(3)]
    assert n == 0, f"推导式不该泄漏循环变量，实际 n={n}"
    for n in range(3):
        pass
    assert n == 2, f"for 循环会泄漏循环变量，实际 n={n}"

    # ---- 判据 10：多重 for（不是嵌套推导式）----
    assert flatten([[1, 2], [3], []]) == [1, 2, 3], flatten([[1, 2], [3], []])

    # ---- 判据 11-12：前置三元 = 分流（长度不变）；对照后置 if = 过滤（长度会变）----
    labeled = label_parity([1, 2, 3])
    assert labeled == [("odd", 1), ("even", 2), ("odd", 3)], labeled
    assert len(labeled) == 3, f"分流不丢元素，实际 {labeled}"
    assert len(even_squares([1, 2, 3])) == 1, "后置 if 是过滤，长度会变"

    # ---- 判据 13-15：嵌套推导式，并与官方推荐的 zip 对照 ----
    m = [[1, 2, 3], [4, 5, 6]]
    assert transpose(m) == [[1, 4], [2, 5], [3, 6]], transpose(m)
    assert transpose(m) == [list(t) for t in zip(*m)]
    assert transpose([]) == []

    # ---- 判据 16-17：语法红线（不用写代码，跑通即证明）----
    try:
        compile("[x, x**2 for x in range(3)]", "<tuple-without-parens>", "eval")
    except SyntaxError:
        pass
    else:
        raise AssertionError("元组表达式不加括号应当 SyntaxError")
    assert [(x, x**2) for x in range(3)] == [(0, 0), (1, 1), (2, 4)]

    print("判据 17/17 绿")


if __name__ == "__main__":
    main()
