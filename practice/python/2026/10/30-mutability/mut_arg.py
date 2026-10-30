"""可变性陷阱 · 实验二：函数传参（别名跨函数）

目的：分清「改对象」与「重新绑定名字」—— 前者调用方看得见，后者看不见。

判据（四条都要看到）：
  1. add_bad(nums, 5) 之后：调用方 len +1，且 `nums` 与调用前是同一个对象
  2. rebind(nums, 5) 之后：调用方 list **完全没变**
  3. add_good(nums, 5) 返回**新对象**（id 不同），调用方原 list 不变
  4. inplace_iadd(nums, 5) 之后：调用方**变了** ← 这条最反直觉，先猜再跑

跑法：uv run python mut_arg.py
"""


def add_bad(nums: list[int], item: int) -> None:
    """改对象 —— 调用方会看到。方法体留给你。

    判据：调用后 len(nums) 加 1，且 id(nums) 不变。
    """
    # TODO
    nums.append(item)


def rebind(nums: list[int], item: int) -> None:
    """重新绑定名字 —— 调用方看不到。方法体留给你。

    判据：调用后调用方的 list 一个元素都没多。
    """
    # TODO
    nums = nums + [item]

def add_good(nums: list[int], item: int) -> list[int]:
    """不碰输入，返回新列表。方法体留给你。

    判据：返回值的 id 与原 list 不同；原 list 不变。
    """
    # TODO
    return nums + [item]


def inplace_iadd(nums: list[int], item: int) -> None:
    """陷阱版：用 `nums += [item]` 实现。先猜调用方会不会变，再跑。"""
    # TODO
    nums += [item]


if __name__ == "__main__":
    base = [1, 2, 3]

    n = base.copy()
    add_bad(n, 5)
    print("1) 改对象：", n, "| 原对象未变：", base == [1, 2, 3])

    n = base.copy()
    before_id = id(n)
    rebind(n, 5)
    print("2) 重新绑定：", n, "| id 未变：", id(n) == before_id)

    n = base.copy()
    new = add_good(n, 5)
    print("3) 返回新对象：", new, "| 输入未变：", n == [1, 2, 3], "| id 不同：", id(new) != id(n))

    n = base.copy()
    inplace_iadd(n, 5)
    print("4) 函数里 `+=`：", n, "  ← 猜对了？")
    
