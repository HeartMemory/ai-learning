"""账单层：把一次 `Answer` 的 token 数折算成钱（L4 关卡 4 的地基）。

为什么单独一层：
  `completion_tokens` 里**混着思考 token**，而供应商价格表里**没有「思考」这一栏**
  （只有 输入 / 输出 / 输入缓存命中 三档）⇒ 拆 `reasoning_tokens` **不是为了少算钱**，
  而是为了解释波动（关卡 2 同 prompt 输出 381→2032，差 5.3×，差值几乎全在思考上）。
  真正拉开价差的是**缓存命中**：命中价 0.02 vs 未命中 1，**50 倍**。

官方口径（2026-11-22 查证 · 两条来源交叉 + usage 字段一份规范）：
  · 阿里云百炼《deepseek-v4.1-flash》华北2（北京），元/百万 tokens：
      输入 闲 **1** / 忙 **2** ｜ 输出 闲 **4** / 忙 **8**
      输入(缓存命中) 闲 **0.02** / 忙 **0.04**
      闲时 = 北京时间 **22:00–次日 08:00**，其余为忙时
  · DeepSeek 官方 Models & Pricing：deepseek-flash 非高峰 输入 **$0.15** / 输出 **$0.60** per 1M
      （按 ~7.1 汇率 ≈ ¥1.07 / ¥4.26，与百炼闲时价同一量级 ⇒ 互证）
  · OpenAI API Reference：`completion_tokens_details.reasoning_tokens`、
      `prompt_tokens_details.cached_tokens` **均为 optional**；
      `reasoning_tokens` **已包含在** `completion_tokens` 内
  · 两家都**不针对思考模式单独计价** ⇒ 思考按输出价计，一分不少

判据（本块全绿才进块 2）：
  ① 一次真实调用打印出 输入 / 输出 / 其中思考 / 输入缓存命中 四个数 + 本次费用（元，2 位小数）
  ② 手算互证 cost == (cached*p_cache + (prompt-cached)*p_in + completion*p_out) / 1e6，
     与程序值差 < 1e-9 元
  ③ reasoning ≤ completion 恒成立；变异（把 reasoning 从 completion 里减掉再乘出价）
     必须让 ② 变红 —— 否则 ② 只是抄数，不是算数

镜子（同族写法，别另起一套）：
  · `llm.py` 的 `Answer` / `StreamAnswer`：frozen + slots 的 dataclass，字段带口径注释
  · `structured.py` 的 `schema_for()`：不合规**当场 raise**，不静默回落
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from .llm import Answer, StreamAnswer

PER_MILLION = 1_000_000.0
# 北京时间 = UTC+8 固定偏移（中国自 1991 年起无夏令时）。
# ⚠️ 2026-11-22 踩过：`ZoneInfo("Asia/Shanghai")` 在本机（Windows）直接
#    `ZoneInfoNotFoundError` —— Windows 不自带 IANA 时区库，要用 `zoneinfo` 就得额外装
#    `tzdata`。这里没有跨时区需求，固定偏移更省事，也就不引新依赖。
BEIJING = timezone(timedelta(hours=8))


@dataclass(frozen=True, slots=True)
class Price:
    """元 / 百万 tokens（华北2·北京）。闲时 = 北京时间 22:00–次日 08:00。"""

    input_idle: float
    input_busy: float
    output_idle: float
    output_busy: float
    cached_idle: float
    cached_busy: float


# 官方数字照抄，别手改：改了就等于自己造了一个不存在的价目表。
_TABLE = Price(
    input_idle=1.0,
    input_busy=2.0,
    output_idle=4.0,
    output_busy=8.0,
    cached_idle=0.02,
    cached_busy=0.04,
)

PRICES: dict[str, Price] = {
    "deepseek-v4.1-flash-1017": _TABLE,
    "deepseek-v4.1-flash": _TABLE,
}


def price_for(model: str) -> Price:
    """按模型名取价目表。

    实现要点：
      · 价目表是**白名单**，不是模糊匹配：模型名在 `.env`（或 `--model` 覆盖）里就已经定死，
        没有"猜"的余地 ⇒ 一次查表，查不到**当场 raise**
      · 不许回落到某个默认价 —— 静默回落会让"换了个贵模型"这件事在账单上看不出来
      · 想加模型就加键，不要在代码里写剥后缀 / 前缀匹配的兜底（那会把价目表变成一间黑箱）
    """
    if model in PRICES:
        return PRICES[model]
    raise KeyError(f"模型 {model!r} 没有对应的价目表（PRICES）")


def is_off_peak(now: datetime | None = None) -> bool:
    """现在是不是闲时（北京时间 22:00–次日 08:00）。

    实现要点：
      · `now is None` 时取当前时间；传进来的时间若是 naive，要按北京时间解释
      · **带别的时区**的 aware 时间必须先 `astimezone(BEIJING)` —— 直接取 `.hour` 拿到的是
        那个时区的钟点（实测：UTC 15:00 是北京 23:00 的**闲时**，却被判成忙时）
      · **跨零点区间**：`hour >= 22 or hour < 8` —— 写成 `22 <= hour < 8` 永远为 False
      · 判据要能复现：拿固定时刻跑（`is_off_peak(datetime(2026, 11, 22, 23, 0, tzinfo=BEIJING))`
        与 `...7, 59)` / `...8, 0)` 各来一次），不靠"现在是几点"撞运气
    """
    if now is None:
        now = datetime.now(tz=BEIJING)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=BEIJING)  # naive ⇒ 当成北京时间
    else:
        now = now.astimezone(BEIJING)  # aware 但是在别的时区 ⇒ 先换算
    hour = now.hour
    return hour >= 22 or hour < 8


@dataclass(frozen=True, slots=True)
class Bill:
    """一次调用的账单：四个 token 数 + 三段费用 + 合计（元）+ 用的哪档价。

    `reasoning_tokens` / `cached_tokens` 为 `None` 时说明**供应商没给**，
    这里如实保留，别在计算前先填 0 —— 填 0 等于宣称"思考免费 / 全按新价"。
    """

    prompt_tokens: int
    cached_tokens: int | None
    completion_tokens: int
    reasoning_tokens: int | None
    cached_yuan: float
    input_yuan: float
    output_yuan: float
    total_yuan: float
    off_peak: bool


def bill(answer: Answer | StreamAnswer, *, now: datetime | None = None) -> Bill:
    """把一次调用的用量折算成 `Bill`（四段公式，别退化成「总 token × 一个价」）。

    收两种入参：`Answer`（非流式 / 多轮）与 `StreamAnswer`（流式）——
    三条路径都要算账，但两条路的账单字段是同一套，所以不写两份。

    模型名从哪来（2026-11-22 定）：**只认 `answer.model`**（= 上游 `response.model`），
    本函数**不接收** `model` 参数 —— 手传一个模型名就能让账单静默按错价算
    （`--model` 覆盖时把 `settings.model` 传进来，数字看着完全正常）。
    删掉这个参数，等于让这个错误**无法被表达**。

    实现要点：
      · 流式那条路可能**根本没钱可算**（流被打断 ⇒ 拿不到 usage，`prompt_tokens is None`）
        ⇒ 三个核心字段只要有一个 None 就 raise，**不许编 0** 冒充当次免费
      · 几个 Optional 字段先落局部变量再判空 —— 直接写 `answer.xxx` 判完又访问，
        类型收窄不生效（README 踩坑表同条）
      · `prompt_tokens` **已包含**缓存命中的部分 ⇒ 未命中段是 `prompt - cached`，
        不是 `prompt`（写成 `prompt * 入价 + cached * 缓存价` 会把命中部分算两遍）
      · `cached_tokens is None` ⇒ **算钱**时按全部未命中（保守，宁可高估不低估），
        但 `Bill` 里**如实保留 None** —— 账本要能回答"当时到底知不知道"
      · 两条数据自洽性守卫（不自洽就 raise，算下去只会得到一张看着正常的错账）：
        `cached > prompt` ⇒ 负成本；`reasoning > completion` ⇒ 取错字段
      · `reasoning_tokens` **不参与**计价（思考已含在 completion 里），只随账单透传
      · 三段分别保留原值，合计再相加 —— 分开记才能回答"钱花在哪一段"
    """
    model = answer.model
    prompt_tokens = answer.prompt_tokens
    completion_tokens = answer.completion_tokens
    reasoning_tokens = answer.reasoning_tokens
    cached_tokens = answer.cached_tokens

    if model is None or prompt_tokens is None or completion_tokens is None:
        raise ValueError(
            "这份账单不完整（流式被打断 / 供应商没给 usage 或模型名），不能算钱"
        )

    price = price_for(model)
    off_peak = is_off_peak(now)

    # 供应商没给命中数 ⇒ 算钱按全部未命中（保守高估），但账单里保留 None
    cached_for_cost = 0 if cached_tokens is None else cached_tokens

    if cached_for_cost > prompt_tokens:
        raise ValueError(
            f"缓存命中 {cached_for_cost} > 输入总数 {prompt_tokens}：usage 自相矛盾，"
            "算下去会得到负成本"
        )
    if reasoning_tokens is not None and reasoning_tokens > completion_tokens:
        raise ValueError(
            f"思考 token {reasoning_tokens} > 输出总数 {completion_tokens}："
            "reasoning 本来就含在 completion 里（官方口径），超了说明取错字段"
        )

    # 先选档、再乘：三个公式各只写一次（改公式 / 改档位都只改一处）
    if off_peak:
        p_in, p_out, p_cached = price.input_idle, price.output_idle, price.cached_idle
    else:
        p_in, p_out, p_cached = price.input_busy, price.output_busy, price.cached_busy

    cached_yuan = cached_for_cost * p_cached / PER_MILLION
    input_yuan = (prompt_tokens - cached_for_cost) * p_in / PER_MILLION
    output_yuan = completion_tokens * p_out / PER_MILLION
    total_yuan = cached_yuan + input_yuan + output_yuan

    return Bill(
        prompt_tokens=prompt_tokens,
        cached_tokens=cached_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_yuan=cached_yuan,
        input_yuan=input_yuan,
        output_yuan=output_yuan,
        total_yuan=total_yuan,
        off_peak=off_peak,
    )


def format_bill(bill: Bill) -> str:
    """把一份 `Bill` 打成**一行**文本（三条路径共用，别各写一份）。

    为什么要 6 位小数：单次调用只有几厘钱（实测问一句「1」= ¥0.003038），
    按 2 位小数打出来就是 `¥0.00`，连峰谷差一倍都看不出来。
    **判据里的「总成本 2 位小数」是 10 条 × 三版的合计**，不是这里这一行。

    返回字符串、不直接 `print`：这样才能被测试、被 `chat` 攒成表、写进文件。

    ⚠️ 只用 GBK 能编码的字符（实测本机 `uv run python` 被管道接住时 stdout 是 **cp936**，
    `¥`（U+00A5）和全角竖线都会 `UnicodeEncodeError` 崩在打印那一步 —— 钱已经花掉了）。
    ⇒ 单位写 `元`，分隔符用 ASCII `|`。
    """
    cached = "未知" if bill.cached_tokens is None else str(bill.cached_tokens)
    peak = "闲时" if bill.off_peak else "忙时"
    return (
        f"费用: {bill.total_yuan:.6f} 元 ({peak}"
        f" | 入 {bill.input_yuan:.6f} + 缓存 {bill.cached_yuan:.6f}"
        f" + 出 {bill.output_yuan:.6f})"
        f" | 缓存命中 {cached}"
    )