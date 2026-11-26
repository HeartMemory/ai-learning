"""调用层：一次 `chat.completions` + 把用量与耗时记下来。

⭐ **今天的练习就在 `ask()` 里** —— 方法体自己写。

判据（L4 关卡 0 ①，三条同时成立才算通）：
  1. 正文**非空**（不是报错、不是空串、不是 None）
  2. 三个 token 数**取自 `response.usage`**，不是自己数字数
  3. 耗时 = `time.perf_counter()` 的前后差值，单位毫秒

镜子：`practice/ai/2026/10/29-llm-hello/ask.py`（同族写法，但那边是一把梭；这边要拆成
「调用 → 记账 → 返回结构」，因为后面关卡 1（流式）/ 关卡 3（成本）都要复用这笔账）。
"""

import time
from dataclasses import dataclass
from typing import Any, Literal, cast

from openai import OpenAI

# 思考强度档：`None`（不传 = 厂商默认）+ 四档。
# `off` = 关思考；`low` / `high` / `max` = 开思考并指定强度（官方有效档位就这三档）。
ThinkLevel = Literal["off", "low", "high", "max"]


@dataclass(frozen=True, slots=True)
class Answer:
    """一次调用的产出：正文 + 三个 token 数 + 端到端耗时（ms）+ 成本要的两个细分。

    最后三个字段是 L4 关卡 4 加的（2026-11-22）：
      · `model`：**上游回传的实际模型名**（`response.model`），账单只认它 ——
        我们请求时传的可能是别名，网关可能改写成快照名，回传的才是它真正服务的那个
    后两个默认 `None`，表示**供应商没给**：
      · `reasoning_tokens`：思考 token，**已包含在** `completion_tokens` 里（官方口径），
        拆出来只用于解释波动，**不参与计价**
      · `cached_tokens`：输入里命中上下文缓存的部分，单价与未命中差 **50 倍**
      · ⚠️ `None` ≠ 0：填 0 等于宣称"思考免费 / 全按新价"，账单会静默偏低
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    elapsed_ms: float
    finish_reason: str
    model: str
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None


@dataclass(frozen=True, slots=True)
class StreamAnswer:
    """流式一次调用的产出：拼接正文 + 两个时间 + 账单 + 有没有真的走完。

    ⭐ 与 `Answer` 的两处差别（都是踩出来的，不是设计洁癖）：
      · token 三数是 `int | None` —— 流被打断时**官方不给 usage 汇总帧**，那时没有账单可记
      · 多个 `completed` —— 「循环自然结束」不等于「成功」，只有收到 `finish_reason` 才算走完

    后三个字段是 L4 关卡 4 加的（三条路径都要算账，字段与 `Answer` 对齐）：
      · `model`：取**任意一帧**的 `chunk.model`（上游回传），账单只认它
      · `reasoning_tokens` / `cached_tokens`：来自汇总帧，拿不到就是 `None`
    """

    text: str
    chunks: int
    first_token_ms: float | None
    elapsed_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
    finish_reason: str | None
    completed: bool
    model: str | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None


def _thinking_kwargs(think: ThinkLevel | None) -> dict[str, Any]:
    """把 `--think off|low|high|max` 翻成 **`create()` 的附加关键字参数**。

    ⭐ **今天的练习就在这个函数体**（`None` 那条已给，剩下两条）—— 别改签名。

    为什么返回「kwargs」而不是「一个 extra_body dict」：官方把这两个参数放在**不同通道**上
    （见下第 3 条），塞进同一个 dict 就成了自己发明的一套走法。

    官方原文（2026-11-26 查证 · 阿里云百炼《DeepSeek》页；与 DeepSeek 直连页冲突时以百炼为准）：
      1. 默认值：「`deepseek-v4.1-flash`、`deepseek-v4-pro`、`deepseek-v4-flash-0731` 和
         `deepseek-v4-pro-0813` **默认开启思考模式**。通过 `reasoning_effort` 参数可以调整
         推理强度，可选值为 `low`（仅 `deepseek-v4.1-flash`、`deepseek-v4-flash-0731` 和
         `deepseek-v4-pro-0813` 支持）、`high` 和 `max`，**默认为 `high`**。」
         ⇒ `None`（不传）**当天**等价于 `"high"`；显式传的意义是把默认权拿回来
      2. 计费：「思考模式下，**思维链按照输出 Token 计费**」，`reasoning_tokens` 含在
         `completion_tokens` 里 ⇒ 判据 ②（算 input 还是 output）有原文可引
      3. 通道：「`enable_thinking` **非 OpenAI 标准参数**，OpenAI Python SDK 通过 `extra_body`
         传入……`reasoning_effort` **是 OpenAI 标准参数**，可直接作为顶层参数传入。」
         ⇒ 开关走 `extra_body={"enable_thinking": ...}`，强度走顶层 `reasoning_effort=...`
      4. 文档空白（如实记）：百炼 DeepSeek 页**通篇没写**「思考模式下 `temperature` 不生效」，
         DeepSeek 直连页则明写不支持 ⇒ 判据 ③ 只能**实测**，不许引百炼页当证据
      5. 文档矛盾（如实记）：`low` 的支持清单含 `deepseek-v4.1-flash`，但同页推理强度示例
         用的是 `deepseek-v4-pro`（**不在** low 清单）⇒ `low` 是否真能降下来，实测为准

    三条口径：
      · `None` ⇒ 返回 `{}`：一个参数都不加 ⇒ 请求体与关卡 0–4 **完全一致**（可复现性靠这条）
      · `"off"` ⇒ **只**关思考，不顺手传强度：官方没写「关思考时 effort 该不该传」
        （DeepSeek 页未说明、百炼页未列）⇒ 别赌
      · 三个强度档 ⇒ **显式开 + 指定强度**：开要显式，因为「默认开启」是厂商承诺、会变；
        写死才是我们的契约
      ⚠️ 官方没有一处承诺「字段名写错会报错」⇒ 生效与否只能靠 `reasoning_tokens` +
        耗时 + 文本一起反证，不能靠「我传了」。
    """
    if think is None:
        return {}
    if think == "off":
        return {"extra_body": {"enable_thinking": False}}
    return {"extra_body": {"enable_thinking": True}, "reasoning_effort": think}



def ask(
    prompt: str,
    *,
    model: str,
    client: OpenAI,
    think: ThinkLevel | None = None,
) -> Answer:
    """发一次请求，返回 `Answer`。`think` 的档位语义见 `_thinking_kwargs()`。

    ⚠️ 2026-11-22 改为**一行转调 `ask_messages()`** —— 记账段只在那边维护一份。
    原来这里另有一份一字不差的实现，两边一改就漂移（同一段类型收窄，写法差一个词，
    一边绿一边红：`usage.xxx` 收窄生效，`response.usage.xxx` 不生效）。

    记账要点仍在 `ask_messages()` 里：
      · 计时包住**整次请求**（起点在 create 之前，终点在它返回之后）
      · 正文可能为 `None`（拒答 / 只回工具调用 / 网关不给）⇒ `or ""` 兜底，**不抛错**：
        这些情况下 token 照扣，抛错会把账单一起扔掉
      · 三个 token 数取自 `usage`（`usage is None` 才抛 —— 那时真无法记账）
      · `finish_reason` 在 `choices[i]` 上，**不在 `message` 上**
      · `model` 取**上游回传**的 `response.model`，不是入参
    """
    return ask_messages(
        [{"role": "user", "content": prompt}],
        model=model,
        client=client,
        think=think,
    )


def ask_messages(
    messages: list[dict[str, str]],
    *,
    model: str,
    client: OpenAI,
    think: ThinkLevel | None = None,
) -> Answer:
    """发一次**多轮**请求：`messages` = 完整历史 + 本次提问，返回 `Answer`。

    判据（L4 关卡 3）：每轮都要能从 `usage.prompt_tokens` 读到**整段历史**的 token 数 ——
    所以 `messages` 必须原样交给 SDK，**不许**在这里偷偷只发最后一条。

    实现要点（2026-11-22 起**本函数是唯一实现**，`ask()` 一行转调到这里）：
      · `messages=` 收整张消息表（不是 `[{"role": "user", "content": prompt}]` 一条）
      · 计时包住整次请求 → 空 `choices` 就抛 → `usage is None` 就抛 →
        `content or ""` 兜底 → `finish_reason` 从 `choices[0]` 取
      · `model` 与两个 token 细分都取**上游回传**的，不取入参 / 不自算
      · `think`（关卡 5 加）：经 `_thinking_kwargs()` 翻成 create() 的额外关键字参数；
        `None` ⇒ `{}` ⇒ **一个参数都不加**，请求体与关卡 0–4 一模一样
    """
    started = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        # `messages` 是对外的 `list[dict[str, str]]`（`chat.py` 造的就是这个形状），
        # SDK 要的是 `Iterable[ChatCompletionMessageParam]`（Union[TypedDict...]）。
        # 形状 runtime 上完全对（只有 role/content 两个 str 字段），但静态收窄不过去 ⇒
        # 在这里断言一次，而不是把 SDK 的类型泄漏到对外签名里（那会逼 `chat.py` 跟着改）
        messages=cast("Any", messages),
        **_thinking_kwargs(think),
    )
    elapsed_ms = (time.perf_counter() - started) * 1000
    if not response.choices:
        raise RuntimeError("供应商返回了空 choices")
    message = response.choices[0].message
    usage = response.usage
    if usage is None:
        raise RuntimeError("供应商没返回 usage，无法记账（判据 ① 要平台数字）")
    text = message.content or ""

    # 两个细分**先落成局部变量**再判 None：
    #   `usage` 上面已判空，但 `usage.xxx` 每写一次都是重新访问属性，类型收窄不会带过去；
    #   写成 `response.usage.xxx` 更是连"判空过的那个变量"都不是 ⇒ Pylance 必红（README 踩坑表同条）
    c_details = usage.completion_tokens_details
    p_details = usage.prompt_tokens_details

    return Answer(
        text=text,
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        total_tokens=usage.total_tokens,
        elapsed_ms=elapsed_ms,
        finish_reason=response.choices[0].finish_reason,
        model=response.model,
        reasoning_tokens=c_details.reasoning_tokens if c_details else None,
        cached_tokens=p_details.cached_tokens if p_details else None,
    )


def ask_stream(
    prompt: str,
    *,
    model: str,
    client: OpenAI,
    think: ThinkLevel | None = None,
) -> StreamAnswer:
    """流式发一次请求：边到边收增量，同时记「首 token 何时到」与「总共多久」。

    判据（L4 关卡 1 ①，两个数都要报，且首 token 明显早于总时长）：
      · `first_token_ms` = 第一个**非空正文增量**到达的时刻 − 计时起点
      · `elapsed_ms`     = 整个循环结束 − 计时起点

    实现要点（按踩坑顺序排，**不是**方法体）：
      1. `create(..., stream=True, stream_options={"include_usage": True})` —— 不传
         `stream_options` 就没有 usage 汇总帧，流式调用将**完全无法记账**
      2. `chunk.choices` 可能为空列表（就是那条只带 usage 的汇总帧）⇒ 取 `choices[0]` 前
         必须先判空，否则 `IndexError`（与关卡 0「空 choices 守卫前移」同源）
      3. `delta.content` 可能为 `None`：role 帧、纯 reasoning 帧、终止帧都有 content=None
         ⇒ `None` **不算首 token**，也不拼接（拼进去会让判据 ② 逐字符相等永远差几个字符）
      4. `finish_reason` 只在最后一条**有 choices** 的帧上非 None ⇒ 拿它当「走完了」的唯一判据；
         **循环自然结束 ≠ 成功**（官方原话：不要假设流走完就是成功）
      5. usage 只在汇总帧上，且**流被打断时拿不到** ⇒ 本函数拿不到就把三个 token 数填
         `None`、`completed=False`，**不抛错** —— 与关卡 0「拿不到 usage 就 raise」相反：
         那边是「跑完了没账单」，这边是「本来就没跑完，半截账单不存在」

    官方出处（2026-11-20 查证 · 两条来源）：
      · ① SDK 源码 `types/chat/chat_completion_stream_options_param.py`：`include_usage` 会在
        `data: [DONE]` 前多推一条 `choices` 为空、只带 `usage` 的帧；原文
        "If the stream is interrupted, you may not receive the final usage chunk"
      · ① SDK 源码 `_streaming.py`：`__stream__` 的 `finally` 里调 `response.close()`
        ⇒ 循环里 `break` 会自动关连接，不需要自己 try/finally
      · ② 官方 README：流消费**不会自动重试**（重放会重复输出）；读超时抛
        `APITimeoutError`，其它连接失败抛 `APIConnectionError`，原始异常在 `__cause__`

    `think`（关卡 5 加，档位见 `_thinking_kwargs()`）：流式下思考增量走
    `delta.reasoning_content`（与 `content` 并列）⇒ 上面第 3 条记的「纯 reasoning 帧
    content=None」就是它。本函数**只记账不看内容**，思考 token 数仍从汇总帧取。
    """
    started = time.perf_counter()
    text = ""
    first_token_ms: float | None = None
    chunks = 0
    finish_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    # 局部变量别叫 model —— 会盖住入参 `model`（那是请求名，账单要的是回传名）
    resp_model: str | None = None
    reasoning_tokens: int | None = None
    cached_tokens: int | None = None

    stream = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        stream_options={"include_usage": True},
        **_thinking_kwargs(think),
    )
    for chunk in stream:
        chunks += 1
        resp_model = chunk.model or resp_model
        if not chunk.choices:
            usage = chunk.usage
            if usage is not None:
                prompt_tokens = usage.prompt_tokens
                completion_tokens = usage.completion_tokens
                total_tokens = usage.total_tokens
                c_details = usage.completion_tokens_details
                p_details = usage.prompt_tokens_details
                reasoning_tokens = c_details.reasoning_tokens if c_details else None
                cached_tokens = p_details.cached_tokens if p_details else None
            continue
        delta = chunk.choices[0].delta
        if delta.content is not None:
            if first_token_ms is None:
                first_token_ms = (time.perf_counter() - started) * 1000
            text += delta.content
            print(delta.content or '', end="", flush=True)
        if chunk.choices[0].finish_reason is not None:
            finish_reason = chunk.choices[0].finish_reason

    return StreamAnswer(
        text=text,
        chunks=chunks,
        first_token_ms=first_token_ms,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        finish_reason=finish_reason,
        completed=finish_reason is not None,
        model=resp_model,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
    )