"""S0 · 关卡 1：异常链 —— `raise ... from` 到底改了哪几个属性

目的（一条根）：`raise New from old` 不是"顺便提一句 old"，它做的是把 `old` 钉进
    `__cause__`、**并把 `__suppress_context__` 置 True**。三态（`from exc` / 裸 `raise` /
    `from None`）只差在这一对属性上；traceback 打印成什么样，全是这两句的后果。

════════════════════════════════════════════════════════════════════════
 TODO 清单：**一项 = 一个函数 = 一个整体**，把函数体里那句 raise 换成你的实现
════════════════════════════════════════════════════════════════════════
 T1  [实现] load_config(raw)            json 解析失败 → `raise ConfigError(...) from exc`
 T2  [实现] load_config_implicit(raw)   同样包装，但不带 `from`（隐式链）
 T3  [实现] load_config_cut(raw)        `raise ConfigError(...) from None`（切断）
 T4  [判据] judge_chain()               验三态的 __cause__ / __context__ / __suppress_context__

跑法：uv run python exceptions.py
文档依据：docs.python.org/3/reference/simple_stmts.html（raise 的 from 子句 = 设置 __cause__，
        未捕获时两段 traceback 一起打印）
        docs.python.org/3/library/exceptions.html（from 后面必须是异常或 None；设置 __cause__
        会隐式把 __suppress_context__ 置 True；被切掉的原异常仍留在 __context__ 里）
        docs.python.org/3/tutorial/errors.html（`raise RuntimeError from None` 的输出对照）
════════════════════════════════════════════════════════════════════════
"""

import json
from typing import Any

CONFIG_TEXT = '{"model": "demo-model", "temperature": 0.2}'
"""一个合法配置（T1-T3 的成功路径用它）。"""

BAD_TEXT = '{"model": "demo-model",'
"""一个坏配置（少了右括号，json 会抛错）。"""


class ConfigError(Exception):
    """本项目自己的异常类型 —— 对外只暴露它，不把 json 的解析错误直接漏给调用方。"""


def load_config(raw: str) -> dict[str, Any]:
    """T1 · `json.loads(raw)`；解析失败时抛 **ConfigError**，并把原始的 json 错误**链上去**。

    要求：`try/except` 里接住 `ValueError`（`json.JSONDecodeError` 是它的子类），
    然后 `raise ConfigError(...) from exc` —— `from` 后面就是接住的那个异常。
    成功路径返回解析出来的 dict。
    """
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"配置解析失败：{exc}") from exc


def load_config_implicit(raw: str) -> dict[str, Any]:
    """T2 · 与 T1 逐字相同，**唯一差别**：`raise ConfigError(...)` 后面不写 `from`。

    这一版要的正是"隐式链"：Python 会自己在 `__context__` 里记下原异常，但 `__cause__` 留空、
    `__suppress_context__` 为 False ⇒ traceback 会多打一段 "During handling of the above..."。
    """
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"配置解析失败：{exc}")  # noqa: B904 刻意不写 from（T2 要隐式链）


def load_config_cut(raw: str) -> dict[str, Any]:
    """T3 · 与 T1 逐字相同，**唯一差别**：写成 `raise ConfigError(...) from None`。

    作用 = 显示层面切断（调用方只看到 ConfigError），但原异常**并没有消失** ——
    它还在 `__context__` 里，供调试时翻。
    """
    try:
        return json.loads(raw)
    except ValueError as exc:
        raise ConfigError(f"配置解析失败：{exc}") from None


def judge_chain() -> bool:
    """T4 · 判据：三态各验一遍（写成 assert，全过 return True）。

    T1（显式 from）：
      ① 抛的确实是 ConfigError
      ② `exc.__cause__` 是原始 json 错误（`isinstance(exc.__cause__, ValueError)`）
      ③ `exc.__suppress_context__ is True`
    T2（隐式）：
      ④ `exc.__cause__ is None` 且 `exc.__context__` 不是 None
      ⑤ `exc.__suppress_context__ is False`
    T3（from None）：
      ⑥ `exc.__cause__ is None` 但 `exc.__suppress_context__ is True`
      ⑦ `exc.__context__` 仍然不是 None（切的是显示，不是记录）

    提示：每条都用 `try: ... except ConfigError as exc:` 拿到异常对象再看属性；
    `BAD_TEXT` 是现成的坏输入。另外可以顺手打印一次 `traceback.format_exception(exc)`
    对比 T1 与 T2 的段数（仅供观察，不进判据）。
    """
    assert load_config(CONFIG_TEXT) == {"model": "demo-model", "temperature": 0.2}
    assert load_config_implicit(CONFIG_TEXT) == {"model": "demo-model", "temperature": 0.2}
    assert load_config_cut(CONFIG_TEXT) == {"model": "demo-model", "temperature": 0.2}

    try:
        load_config(BAD_TEXT)
    except ConfigError as exc:
        assert isinstance(exc.__cause__, ValueError)
        assert exc.__suppress_context__ is True
    else:
        raise AssertionError("T1 应该抛 ConfigError")

    try:
        load_config_implicit(BAD_TEXT)
    except ConfigError as exc:
        assert exc.__cause__ is None
        assert exc.__context__ is not None
        assert exc.__suppress_context__ is False
    else:
        raise AssertionError("T2 应该抛 ConfigError")

    try:
        load_config_cut(BAD_TEXT)
    except ConfigError as exc:
        assert exc.__cause__ is None
        assert exc.__suppress_context__ is True
        assert exc.__context__ is not None
    else:
        raise AssertionError("T3 应该抛 ConfigError")

    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的。"""
    results = [judge_chain()]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
