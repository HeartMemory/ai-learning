"""多轮对话 + 上下文记账（`python -m llm_cli chat`）。

判据（L4 关卡 3，四条全绿才算过）：
  ① 第 3 轮的 input **明显大于**第 1 轮 —— 三轮的数都要报（证明 API 无状态、全历史重发）
  ② **累计上下文 = 各轮 input 之和**，不是最后一轮的值（程序打印值 == 求和值）
  ③ 输入 `exit` 后历史存成 JSON，重启能读回继续（报文件名 + 读回后第 1 轮的 input 数）
  ④ **不做任何压缩**；撞上下文上限时把**报错原文**打印出来，不静默截断

原理（2026-11-21 查证 · 两条来源）：
  · 阿里云百炼《多轮对话》：API **无状态**，服务端不保存历史；每轮把 user 与 assistant
    追加进 messages 数组后**整体重发**。思考模型返回的 `reasoning_content` **不该**拼回历史
    （只拼 `content`），否则下一轮输入 token 白涨。
  · openai-python README：SDK 没有任何记忆 / session 续接机制，请求之间互不相关。
  ⇒ 每轮 `usage.prompt_tokens` = **整段历史的 token 数**，不是「这一句话」的 token 数；
    所以「累计上下文」只能逐轮求和 —— 这正是判据 ② 存在的理由（最后一轮的值 ≠ 累计值）。
  · 撞上限的报错形态（阿里云《错误码》）：HTTP 400 / `InternalError.Algo.InvalidParameter`，
    原文形如 `Range of input length should be [1, XXX]` ⇒ 打印**原文**，别自己编一句
    「上下文太长了」。

分工：`Turn` / 函数签名 / 判据注释都是 AI 建好的框架，**四个 TODO 的方法体自己写**。
"""

import json
from dataclasses import asdict, dataclass, fields
from datetime import datetime
from pathlib import Path

from openai import OpenAI

from .llm import ask_messages


@dataclass(frozen=True, slots=True)
class Turn:
    """一轮问答：用户说了什么、模型回了什么、**这一轮自己**花了多少。"""

    user: str
    reply: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_ms: float


# 一条历史记录应有的字段名（缺一个 / 多一个都算"文件坏了"）
TURN_FIELDS = frozenset(f.name for f in fields(Turn))


def _turn_from(item: object) -> Turn:
    """把一条 JSON 记录还原成 `Turn`；结构不对就抛 `TypeError` / `ValueError`。

    为什么查得这么细（2026-11-21 实测）：
      · dataclass **运行时零校验**：`{"prompt_tokens": "abc"}` 能一路装进 `Turn`，
        等到 `sum()` 才炸 —— 报错现场离真因几百行远
      · 顶层若不是数组（例如 `{...}`），`for item in data` 拿到的是**键**（`str`），
        `Turn(**item)` 抛的是 `argument after ** must be a mapping, not str`
      · 缺字段/多字段抛的是 `missing N required positional arguments` / `unexpected keyword`
        —— 信息里不说**缺哪个**，所以字段集合自己比一遍
    """
    if not isinstance(item, dict):
        raise TypeError(f"这条记录不是对象，而是 {type(item).__name__}")

    missing = TURN_FIELDS - item.keys()
    extra = item.keys() - TURN_FIELDS
    if missing:
        raise ValueError(f"字段不符：缺少 {sorted(missing)}")
    if extra:
            raise ValueError(f"字段不符：，多出 {sorted(extra)}")

    for name in ("prompt_tokens", "completion_tokens"):
        value = item[name]
        # bool 是 int 的子类 ⇒ True/False 会冒充数字，必须单独排掉
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} 应为整数，实为 {type(value).__name__}：{value!r}")
    elapsed = item["elapsed_ms"]
    if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)):
        raise ValueError(f"elapsed_ms 应为数字，实为 {type(elapsed).__name__}：{elapsed!r}")

    return Turn(**item)


def _quarantine(path: Path, exc: Exception) -> list[Turn]:
    """把坏文件改名留证，然后从空对话开始（**不删、不静默**）。"""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = path.with_name(f"{path.stem}.corrupt-{stamp}{path.suffix}")
    path.replace(target)
    print(f"历史读取失败：{type(exc).__name__}: {exc}")
    print(f"  原文件：{path}")
    print(f"  已隔离：{target}（保留证据，可人工修回）")
    print("  从空对话开始")
    return []


def load_turns(path: Path) -> list[Turn]:
    """读回历史。三种情况都不许让程序崩在启动：

    1. 文件不存在 ⇒ `[]`（第一次跑）
    2. 文件存在但**空**（0 字节）⇒ `[]` —— 要单独判，否则它和"截断"共用同一个
       `JSONDecodeError`，会被误当成损坏、白隔离一个本来就没内容的文件
    3. 文件坏了（截断 / 编码不对 / 结构不符）⇒ **改名隔离** + 打印原因 + `[]`

    只接这几类（`JSONDecodeError` 与 `UnicodeDecodeError` 都是 `ValueError` 子类）：
    `ValueError` / `TypeError`。**不接 `OSError`** —— 没权限、路径是目录属于环境问题，
    把它当"文件坏了"去改名，等于把好文件也毁掉，必须让它冒泡。
    """
    # 一次 stat 同时拿到「在不在」和「多大」：exists() + stat() 是两次系统调用，
    # 中间有窗口（TOCTOU）—— 文件恰好在两次之间被删掉，stat() 会抛 FileNotFoundError 逃出去
    try:
        info = path.stat()
    except FileNotFoundError:
        return []
    if info.st_size == 0:
        print(f"历史文件是空的：{path}（按新对话处理）")
        return []

    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return [_turn_from(item) for item in data]
    except (ValueError, TypeError) as exc:
        return _quarantine(path, exc)


def save_turns(path: Path, turns: list[Turn]) -> None:
    """把历史写回 JSON（**每轮成功后**、`exit` 时、出错那一路都调）。

    两处口径：
      · **原子替换**：先写 `*.tmp` 再 `replace()`。`open(path, "w")` 会先把原文件清空，
        崩在写盘中途就什么都不剩；改名是原子的，磁盘上永远是"上一份完整文件"
      · `ensure_ascii=False` —— 否则中文全变 `\\uXXXX`，人看不懂（判据 ③ 要能核对内容）
    """
    # TODO(你写)
    temp = path.with_suffix(".tmp")
    with open(temp, "w", encoding="utf-8") as f:
        json.dump([asdict(turn) for turn in turns], f, ensure_ascii=False, indent=4)
    temp.replace(path)


def _save_or_report(path: Path, turns: list[Turn]) -> bool:
    """保存并如实报告，返回是否成功。只接 OSError（磁盘 / 占用 / 权限）。"""
    try:
        save_turns(path, turns)
    except OSError as exc:
        print(f"历史保存失败: {exc!r}")
        return False
    print(f"历史已保存到 {path}")
    return True


def build_messages(turns: list[Turn], user: str) -> list[dict[str, str]]:
    """把历史 + 本次提问铺成 `messages`（判据 ① 就靠这里）。

    口径：每轮都**从头铺**（`user` / `assistant` 交替，最后一条是本次的 `user`）。
    只发「本次这一句」的话，第 3 轮 input 不会变大 —— 判据 ① 会立刻红。
    """
    # TODO(你写)
    messages = []
    for turn in turns:
        messages.append({"role": "user", "content": turn.user})
        messages.append({"role": "assistant", "content": turn.reply})
    messages.append({"role": "user", "content": user})
    return messages


def run_chat(*, model: str, client: OpenAI, history_path: Path) -> int:
    """主循环：读一行 → 调模型 → 打印三个数 → 追加历史。返回退出码。

    每轮必须打印这三个数（判据 ① ② 全靠它）：
        本轮 input / 本轮 output / 累计上下文

    退出码：`0` 正常 `exit` 且历史已落盘 ｜ `1` 请求失败或落盘失败 ｜ `2` 配置缺失（上游抛）

    流程（2026-11-21 定稿，**请求与落盘分开处理**）：
      1. 起手 `load_turns()`；非空报「已读回 N 轮」+ 文件名（判据 ③ 要这个），空报「新对话」
      2. `input()` 收一行：`strip()` 后 `exit` 就存盘退出；空行跳过
      3. **请求段** `try: ask_messages(...)`，失败只打印 `请求失败: repr(exc)`
         —— 判据 ④ 要原文里的 `Range of input length should be [1, XXX]`
      4. **先打印三个数，再落盘**：账单已经花出去了，落盘成功与否都不能挡住这三个数
      5. 落盘统一走 `_save_or_report()`：成功才打印「历史已保存」；`OSError` 打印
         「历史保存失败」并返回 `1`（**别的异常不接** —— 那是代码 bug，要炸出来）
      6. **不压缩、不截断历史**：撞上限就让这一轮失败，历史原样留着，下次还能接着问
    """
    turns = load_turns(history_path)
    if turns:
        print(f"已读回 {len(turns)} 轮历史，文件：{history_path}")
    else:
        print("新对话")

    while True:
        line = input("你：").strip()
        if line == "exit":
            if not _save_or_report(history_path, turns):
                return 1
            return 0
        if not line:
            continue

        try:
            answer = ask_messages(build_messages(turns, line), model=model, client=client)
            total_prompt = sum(t.prompt_tokens for t in turns) + answer.prompt_tokens
            turns.append(Turn(user=line, reply=answer.text, prompt_tokens=answer.prompt_tokens,
                          completion_tokens=answer.completion_tokens, elapsed_ms=answer.elapsed_ms))
            print(f"input: {answer.prompt_tokens}")
            print(f"output: {answer.completion_tokens}")
            print(f"累计上下文: {total_prompt}")
            print(f"本轮耗时: {answer.elapsed_ms:.2f} ms")
        except Exception as exc:
            print(f"请求失败: {exc!r}")
            _save_or_report(history_path, turns)
            return 1

        if not _save_or_report(history_path, turns):
            return 1
    
        
