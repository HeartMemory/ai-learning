"""镜子（示范用）—— 看清"装饰器"到底长什么样。**这里没有判据**，判据在 decorators.py。

跑法：uv run python _shape_demo.py
"""


# ① 最朴素：deco 就是一个"收函数、返回函数"的普通函数
def shout(fn):
    def wrapper(name):
        result = fn(name)
        return result.upper()

    return wrapper


def hello(name):
    return f"hello, {name}"


loud_hello = shout(hello)  # 手动套（没有任何语法糖）
print("① 手动套：", loud_hello("kai"))


# ② @ 是同一件事的简写
@shout
def bye(name):
    return f"bye, {name}"


print("② @ 语法：", bye("kai"))
print("② __name__ =", repr(bye.__name__), "← 工牌丢了")


# ③ 通用形状：*args / **kwargs 透传 + return 透传
def trace(fn):
    def wrapper(*args, **kwargs):
        print(f"[trace] 调用 {fn.__name__}  args={args}  kwargs={kwargs}")
        value = fn(*args, **kwargs)
        print(f"[trace] 返回 {value!r}")
        return value

    return wrapper


@trace
def mul(a, b, times=1):
    return (a * b) * times


print("③ 结果：", mul(2, 3))
print("③ 结果：", mul(2, 3, times=10))


# ④ *args / **kwargs 到底"收"了什么（定义处的星号 = 打包）
def collect(*args, **kwargs):
    print("  args   =", args, "→", type(args).__name__)
    print("  kwargs =", kwargs, "→", type(kwargs).__name__)


print("④ collect(2, 3, scale=10)：")
collect(2, 3, scale=10)

print("④ collect(1, 2, 3, 4)（多个位置参都进 args）：")
collect(1, 2, 3, 4)


# ⑤ 没有星号时塞 3 个值 → 直接 TypeError
def strict(a, b):
    return a + b


print("⑤ strict(1, 2, 3) 会怎样：")
try:
    strict(1, 2, 3)
except TypeError as exc:
    print("  TypeError:", exc)
