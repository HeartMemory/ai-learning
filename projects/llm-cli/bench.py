"""L4 关卡 1 判据的测量脚本（证据工具，不属于运行时路径）。

跑法：`uv run python bench.py` —— **会产生 10 次真实调用**（约 2~3 千 token）。

口径（判据全文的唯一家 = 仓库根 `README.md` 阶段表 L4 行）：
  ① 同一 prompt，流式与非流式各跑 5 次，报「首 token 到达时间」与「总时长」的**中位数**
     （两个数都要，且首 token 明显早于总时长）
  ② 流式增量拼接出的文本与非流式返回的文本**逐字符相等**（差 = 0）

⚠️ 判据 ② 的已知边界（写进 README 时要连着这条一起写）：
   这是**两次独立请求**。温度压到 0 只是把随机性降到最低，**不保证**服务端逐字节一致
   （浮点累加顺序 / MoE 路由 / 服务端 batch 拼装都可能让两次输出差 1~2 个字符）。
   ⇒ 差不为 0 时，报「首次分歧下标 + 长度差」，并注明这是供应商属性，**不是拼接 bug**。
   真正能自证「拼接没丢字」的是：`text` 只由 `delta.content` 按顺序累加而成，且 `chunks`
   与正文增长次数一致 —— 那条不依赖第二次请求。

🔧 温度只在**本文件**固定（`pin_temperature`），`llm_cli/llm.py` 一个字都不改：
   生产走供应商默认值（厂商调好的值通常比随手填的更好），测试才需要控变量。

⚠️ **实测结论（2026-11-20，测试所用模型 `deepseek-v4.1-flash`）：思考模式下 temperature 无效。**
   供应商文档明示：思考模式不支持 `temperature` / `presence_penalty` / `frequency_penalty`，
   传了不报错、也不生效（`top_p` 在非思考模式恒为 1.0）。
   ⇒ 对本模型来说 `pin_temperature` 是**无效动作**（保留它，是为了换模型 / 关思考之后仍然可控）。
   ⇒ 判据 ② 的实测结果因此为「不可判定」而不是「红」：两次请求都是不可控采样，
      文本差异与被测目标（拼接无损）无法分离。实测分歧出现在**下标 0**、长度差 3，
      两段开头措辞完全不同 —— 拼接 bug 只能造成丢字 / 重复 / 乱序，造不出「整段改写」。
"""

import io
from contextlib import redirect_stdout
from typing import Any, cast

from openai import OpenAI

from llm_cli.config import load_settings
from llm_cli.llm import Answer, StreamAnswer, ask, ask_stream

RUNS = 5
PROMPT = "用 200 字解释注意力机制里的 Q/K/V 分别在干什么"
PINNED_TEMPERATURE = 0.0


def pin_request_defaults(client: OpenAI, **defaults: Any) -> OpenAI:
    """在**测试层**给这个 client 补默认请求参数，生产代码一个字不改。

    做法：把 `client.chat.completions.create` 换成一层薄包装，用 `setdefault` 补齐缺省参数
    再转给原方法（调用方显式传了的不会被覆盖）。
    为什么不干脆给 `ask_stream` 加参数：那是生产接口，不该为了测试方便留一个平时用不到的口子。

    副作用范围：只影响本进程里**这一个 client 实例**（`client.chat.completions` 是缓存的同一实例），
    不写全局、不改 SDK、不碰磁盘。

    ⚠️ 对本仓库测试所用的 `deepseek-v4.1-flash`：**思考模式下 `temperature` 不生效**（供应商文档明示
    该模式不支持 `temperature` / `presence_penalty` / `frequency_penalty`，收下不报错也不生效）。
    ⇒ 要用温度，就得同时关思考：`pin_request_defaults(client, temperature=0.0, extra_body=
    {"thinking": {"type": "disabled"}})`（`thinking` 不是 SDK 原生字段，只能走 `extra_body`）。
    """
    completions = client.chat.completions
    original = completions.create

    def create(**kwargs: Any) -> Any:
        for key, value in defaults.items():
            kwargs.setdefault(key, value)
        return original(**kwargs)

    cast("Any", completions).create = create
    return client


def median(values: list[float]) -> float:
    """中位数：奇数个取正中，偶数个取中间两个的均值。

    判据注释：
      · 判据 ① 要的是**中位数**不是平均值 —— 第 1 次调用含冷启动（DNS / TLS / 预填充），
        平均值会被它拖歪，中位数抗极端值
      · 不许用 `statistics.mean` 冒充
      · 空列表的行为自己定（抛 `ValueError` 或返回别的），但要能说清为什么
    """
    if not values:
        raise ValueError("median 需要至少一个数（空列表没有中位数）")
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def measure_stream(prompt: str, *, model: str, client: OpenAI) -> list[StreamAnswer]:
    """跑 `RUNS` 次流式，按时间顺序返回每一次的 `StreamAnswer`。

    判据注释：
      · 每次都**新起一次请求**：`stream` 对象是迭代器，一次性，复用拿不到第二遍
      · 5 次里 prompt / model 必须完全一致（判据 ① 的前提是「同一 prompt」）
      · 任何一次出现 `completed is False` 或 token 为 `None`，都要保留下来暴露给人看，
        **不许静默跳过** —— 跳过等于把要报的失败抹掉
    """
    answers: list[StreamAnswer] = []
    for _ in range(RUNS):
        answers.append(ask_stream(prompt, model=model, client=client))
    return answers


def measure_plain(prompt: str, *, model: str, client: OpenAI) -> list[Answer]:
    """跑 `RUNS` 次非流式，按时间顺序返回每一次的 `Answer`（其余口径同 `measure_stream`）。"""
    answers: list[Answer] = []
    for _ in range(RUNS):
        answers.append(ask(prompt, model=model, client=client))
    return answers


def compare_text(stream_text: str, plain_text: str) -> tuple[bool, int, int]:
    """逐字符比对两段文本，返回 `(是否完全相等, 首次分歧下标, 长度差)`。

    完全一致时返回 `(True, -1, 0)`。

    判据注释：
      · **不许**先 `strip()` / 归一化空白 / 忽略标点 —— 换行和空格也是正文的一部分，
        先洗再比等于自己把判据放宽
      · 首次分歧下标要能直接 `stream_text[i:i+20]` 打出来看，所以是**字符**下标不是字节
      · 长度差 = `len(stream_text) - len(plain_text)`，带符号，能看出谁多谁少
      · 长度不等时，分歧下标可能是 `min(len_a, len_b)`（短的那个到头了）
    """
    min_len = min(len(stream_text), len(plain_text))
    for i in range(min_len):
        if stream_text[i] != plain_text[i]:
            return False, i, len(stream_text) - len(plain_text)
    if len(stream_text) != len(plain_text):
        return False, min_len, len(stream_text) - len(plain_text)
    return True, -1, 0


def main() -> int:
    """纯胶水：建 client → 各跑 5 次 → 打表。判据绿不绿由人看数字下结论。"""
    settings = load_settings()
    # 温度只在测试里钉死：生产代码调 ask() / ask_stream() 时仍是供应商默认
    client = pin_request_defaults(
        OpenAI(api_key=settings.api_key, base_url=settings.base_url),
        temperature=PINNED_TEMPERATURE,
    )

    # ask_stream 内部会边到边打印正文（显示逻辑在库函数里，见 README 设计取舍），
    # 这里把它吞掉，否则 5 次 × 200 字的正文会把表格冲烂。
    sink = io.StringIO()
    with redirect_stdout(sink):
        streams = measure_stream(PROMPT, model=settings.model, client=client)
        plains = measure_plain(PROMPT, model=settings.model, client=client)

    print(f"model : {settings.model}")
    print(f"prompt: {PROMPT}")
    print(f"runs  : {RUNS} + {RUNS}")
    print()
    header = f"{'#':>2}  {'流式首token(ms)':>16}  {'流式耗时(ms)':>13}  {'块数':>4}"
    print(f"{header}  {'非流式耗时(ms)':>15}")
    for i, (s, p) in enumerate(zip(streams, plains, strict=True), start=1):
        first = "None" if s.first_token_ms is None else f"{s.first_token_ms:.2f}"
        print(f"{i:>2}  {first:>16}  {s.elapsed_ms:>13.2f}  {s.chunks:>4}  {p.elapsed_ms:>15.2f}")

    firsts = [s.first_token_ms for s in streams if s.first_token_ms is not None]
    print()
    if len(firsts) == len(streams):
        print(f"首 token 中位数(ms) : {median(firsts):.2f}")
    else:
        print(f"首 token 中位数(ms) : 算不了 —— {len(streams) - len(firsts)} 次没收到正文")
    print(f"流式耗时中位数(ms) : {median([s.elapsed_ms for s in streams]):.2f}")
    print(f"非流式中位数(ms)   : {median([p.elapsed_ms for p in plains]):.2f}")

    equal, index, delta = compare_text(streams[0].text, plains[0].text)
    print()
    print(f"逐字符相等: {equal} ｜ 首次分歧下标: {index} ｜ 长度差(流式-非流式): {delta}")
    if not equal:
        head = f"  流式[{index}:{index + 20}]   : {streams[0].text[index:index + 20]!r}"
        tail = f"  非流式[{index}:{index + 20}] : {plains[0].text[index:index + 20]!r}"
        print(head)
        print(tail)

    missing = [s for s in streams if s.prompt_tokens is None or s.completion_tokens is None]
    print()
    print(f"流式账单缺失: {len(missing)}/{len(streams)} 次（>0 说明供应商不给 usage 汇总帧）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
