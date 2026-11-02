"""S0 缺口 · 实验：切片 —— 新对象 / 浅拷贝 / 可变长赋值

目的（一条根）：切片**永远产生新对象**（`s[i:j]` 读出来的是新容器），
    但「新容器」不等于「深拷贝」—— 里面的元素还是原来那些对象的**引用**。
    切片赋值是另一回事：它**原地改**原对象，而且长度可变（`k=1` 时）。

判据（跑完把数字填进本目录 README 的实测列）：
  1. first_three([1,2,3,4])        → [1, 2, 3]，且 `is not` 原列表、原列表不变
  2. reversed_copy([1,2,3,4])      → [4, 3, 2, 1]（`[::-1]`，新对象，不是原地反转）
  3. every_other([0,1,2,3,4])      → [0, 2, 4]（`[::2]`）
  4. shallow_copy(rows)            → 外层 `is not`，但**内层元素 `is` 相同**（浅拷贝铁证）
  5. replace_middle(nums)          → `nums[1:3] = [9]` 后长度 5 → 4，且 `is` 原列表（原地）
  6. drop_middle(nums)             → `del nums[1:3]`，返回 None，原列表变短
  7. [1,2,3][1:100]                → [2, 3]（切片越界**静默截断**）；[1,2,3][100] 才 IndexError
  8. s[::2] = [7, 8]               → 合法；`s[::2] = [1,2,3]` → **ValueError**（步长≠1 必须等长）

跑法：uv run --no-project python slice_basics.py
文档依据：docs.python.org/3.14/library/stdtypes.html（Common Sequence Operations 注 3·4·5·8
         ／ Mutable Sequence Types 切片赋值注 1）
         docs.python.org/3.14/library/copy.html（浅拷贝 = 插入原对象的引用）
"""


def first_three(nums: list[int]) -> list[int]:
    """判据 1：取前三个元素。

    TODO(你写)：用切片，一行。不要写 `if len(nums) < 3` —— 想清楚为什么不需要。
    """
    return nums[:3]


def reversed_copy(nums: list[int]) -> list[int]:
    """判据 2：返回**倒序的新列表**，原列表必须保持原样。

    TODO(你写)：用切片，一行。不许 `nums.reverse()`（那是原地改，会破坏判据）。
    追问：`nums.reverse()` 与 `nums[::-1]` 的返回值分别是啥？
    """
    return nums[::-1]


def every_other(nums: list[int]) -> list[int]:
    """判据 3：隔一个取一个（下标 0, 2, 4…）。

    TODO(你写)：用切片，一行。
    """
    return nums[::2]


def shallow_copy(rows: list[list[int]]) -> list[list[int]]:
    """判据 4：浅拷贝 —— 外层是新的，内层还是原来那些列表。

    TODO(你写)：用整段切片 `rows[:]` 返回（一行）。
    提示：判据会先验 `什么 is not 什么`，再验 `什么 is 什么` —— 两层都要对上。
         然后 main() 会改内层，看原列表是否跟着变。
    """
    return rows[:]


def replace_middle(nums: list[int]) -> list[int]:
    """判据 5：用切片赋值把 `nums[1:3]` 整体换成一个元素 9，返回 nums 本身。

    TODO(你写)：切片赋值一行 + return。
    提示：这条要验「长度会变」和「原地改」两件事。
    """
    nums[1:3] = [9]
    return nums


def drop_middle(nums: list[int]) -> None:
    """判据 6：删掉 `nums[1:3]`，原地生效，不返回新列表。

    TODO(你写)：用 `del` + 切片，函数返回 None。
    """
    del nums[1:3]


def main() -> None:
    # ---- 判据 1：切片产生新对象，原列表不动 ----
    src = [1, 2, 3, 4]
    got = first_three(src)
    assert got == [1, 2, 3], got
    assert got is not src, "切片必须返回新对象"
    assert src == [1, 2, 3, 4], f"原列表被改了：{src}"

    # ---- 判据 2：负步长也是新对象 ----
    rev = reversed_copy(src)
    assert rev == [4, 3, 2, 1], rev
    assert rev is not src, "倒序切片也必须返回新对象"

    # ---- 判据 3：步长 2 ----
    assert every_other([0, 1, 2, 3, 4]) == [0, 2, 4]

    # ---- 判据 4：浅拷贝（两层都要验）----
    rows = [[1, 2], [3]]
    copied = shallow_copy(rows)
    assert copied == rows and copied is not rows, "外层必须是新列表"
    assert copied[0] is rows[0], "内层必须是同一个对象（浅拷贝）"
    copied[0].append(99)
    assert rows[0] == [1, 2, 99], f"改内层原列表应跟着变，实际 {rows}"

    # ---- 判据 5：切片赋值可变长 + 原地 ----
    nums = [1, 2, 3, 4, 5]
    same = replace_middle(nums)
    assert same is nums, "切片赋值是原地改，返回的应是同一个对象"
    assert nums == [1, 9, 4, 5], f"长度 5 → 4，实际 {nums}"

    # ---- 判据 6：del 切片 ----
    nums2 = [1, 2, 3, 4]
    assert drop_middle(nums2) is None, "原地删除不该返回新列表"
    assert nums2 == [1, 4], f"实际 {nums2}"

    # ---- 判据 7：切片越界不炸，索引越界才炸 ----
    assert [1, 2, 3][1:100] == [2, 3]
    assert [1, 2, 3][-100:] == [1, 2, 3]
    seq = [1, 2, 3]
    out_of_range = len(seq)  # 用变量算，避免 ruff 的 PLE0643 常量下标静态告警
    try:
        seq[out_of_range]
    except IndexError:
        pass
    else:
        raise AssertionError("索引越界应当 IndexError")

    # ---- 判据 8：步长 ≠ 1 的赋值必须等长 ----
    s = [0, 0, 0, 0]
    s[::2] = [7, 8]
    assert s == [7, 0, 8, 0], s
    try:
        s[::2] = [1, 2, 3]
    except ValueError:
        pass
    else:
        raise AssertionError("扩展切片赋值长度不等应当 ValueError")

    print("判据 8/8 绿")


if __name__ == "__main__":
    main()
