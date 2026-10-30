"""S0 第 4 章 · 实验一：用 dict 造出第一份 tool schema

目的：证明「结构化输出」在 Python 里就是 dict + list 的嵌套 ——
      它和模型 API 的 tools 字段是**同一个形状**，不是两种东西。

判据（三条都要看到，跑完把数字填进本节 README 的实测列）：
  1. build_tool_schema(...) 的返回值能被 json.dumps(ensure_ascii=False) 序列化；
     最外层两个键是 "type" 与 "function"；tool["type"] == "function"；
     tool["function"]["parameters"]["required"] 与入参 required 是**同一对象**（is 为 True）
  2. 缺失键三连：d["x"] 抛 KeyError；d.get("x") 得到 None；
     d.get("x", []) 的默认值 `[]` 是**每次调用现场新建**的 ⇒ 两次结果不是同一对象。
     ⚠️ 必须**各自先绑变量再比**：`a = d.get("x", []); b = d.get("x", [])` → `a is b` 为 False。
        写成 `id(d.get("x", [])) == id(d.get("x", []))` 会因**地址复用**误判成相同
        （前一个 [] 已销毁，内存地址被后一个复用 —— 测 id 的前提是两个对象同时存活）
  3. 合并两连：a | b 生成**新** dict（新对象 id 与 a 不同、a 内容不变）；
     a |= b 是**原地**（a 的 id 不变、内容变成合并后）

跑法：uv run python dict_schema.py   （纯标准库，裸 python 也能跑）
"""

from __future__ import annotations

import json
from typing import Any


def build_tool_schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> dict[str, Any]:
    """造出 OpenAI 兼容的单个 tool 描述（就是一份 dict）。

    目标形状（自己照着写，别抄别处）：
        {
          "type": "function",
          "function": {
            "name": name,
            "description": description,
            "parameters": {
              "type": "object",
              "properties": properties,
              "required": required,
            },
          },
        }

    判据：json.dumps(返回值, ensure_ascii=False) 不抛异常；
          返回值["function"]["parameters"]["required"] is required 为真。
    """
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        },
    }


def merge_new(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """用 `|` 合并两个 dict，**不改** a。

    判据：返回值的 id 与 a 不同；a 的内容与调用前一致。
    """
    return a | b


def merge_inplace(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    """用 `|=` 合并，**就地改** a 并返回它。

    判据：返回的 id 与传入 a 相同；同名键取 b 的值。
    """
    a |= b
    return a


def main() -> None:
    """把三条判据跑出来并打印，数字抄进 README 的实测列。"""
    # 1. 判据一：能序列化 + 结构对 + 参数是同一个对象
    #    - json.dumps(ensure_ascii=False) 不抛异常，中文原样输出
    #    - tool["type"] == "function"；tool 里有 "type" 与 "function" 两个键
    #    - tool["function"]["parameters"]["required"] is 入参 required → True（没拷贝）
    tool = build_tool_schema(
        name="add",
        description="加法运算",
        properties={"x": {"type": "number"}, "y": {"type": "number"}},
        required=["x", "y"],
    )
    print(json.dumps(tool, ensure_ascii=False))
    print(tool["type"])
    print(tool["function"]["parameters"]["required"])

    # 2. 判据二：缺失键三连
    #    - d["x"]            → 抛 KeyError（下标取值，缺就报错）
    #    - d.get("x")        → None（不报错）
    #    - d.get("x", [])    → 默认值 [] 是**每次调用现场新建**的，两次不是同一对象
    #    ⚠️ 必须**先各自绑定变量再比**（e1 / e2）；写成 id(d.get(...)) == id(d.get(...))
    #       会因**地址复用**误判成 True —— 测 id 的前提是两个对象同时存活
    d = {}
    try:
        d["x"]
    except KeyError:
        print("KeyError")
    print(d.get("x"))
    e1 = d.get("x", [])
    e2 = d.get("x", [])
    print("两个默认值 list 是同一对象吗:", e1 is e2, "| id:", id(e1), id(e2))

    # 3. 判据三：合并两连（同一把尺子：看 id）
    #    - merge_new 用 `|`  → 造出第三个对象：新 id、a 内容不变
    #    - merge_inplace 用 `|=` → 原地改 a：id 不变、同名键取 b 的值
    #    注意 `|` 是**浅合并**：嵌套 dict 的值会被整块替换，不会递归合并
    a = {"k1": 1}
    b = {"k2": 2}
    c = merge_new(a, b)
    print(id(a), id(c), a)
    d = merge_inplace(a, b)
    print(id(a), id(d), a)


if __name__ == "__main__":
    main()
