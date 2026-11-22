"""测量 L4 关卡 4-①：三版 prompt（短/中/长）在同一批输入上的成本与效果。

跑法（key 在 `.env`；**每一次都是计费调用，命令由你自己敲**）：

    uv run python bench_prompt.py --smoke          # 3 次（每版 1 条），先验通路
    uv run python bench_prompt.py --variant short  # 10 次，只跑一版
    uv run python bench_prompt.py                  # 30 次，三版全跑

判据对应（README 阶段表 L4 关卡 4 ①）：
  同一任务三版 prompt（短/中/长）各跑同样 10 条输入，报：
    · 输入 token 总数 / 输出 token 总数
    · 按单价折算的总成本（元，2 位小数）
    · 「每分效果的成本」= 总成本 ÷ 效果总分（效果用 3 条自定标准打分）
  ⚠️ 成本是**折算值**（原生 API 不回金额）；外部取证三级降级见 README。

任务为什么选「分类」：效果要能被**自动打分**。主观题（改写、摘要）只能人工看，
10 条 × 3 版 = 30 条判断，判据就变成了"感觉"，不可复现。

三条打分标准（每分 = 每条用例通过一项，满分 30 = 10 条 × 3 项）：
  ① **类别正确**：输出与 `CASES` 里的人工标注完全一致
  ② **格式合规**：正文能 `json.loads`，且**只有** `category` 一个键、值是 4 个合法值之一
     （容忍 ```json 围栏；但多出键 / 多出解释文字一律算不合规）
  ③ **不长篇**：**可见输出** token（`completion_tokens - reasoning_tokens`）`<= 30` ——
     分类题的答案就几个字，正文写一大堆说明就是没抓住任务
     ⚠️ **2026-11-22 修正（原口径是错的，如实记着）**：原来写的是 `completion_tokens <= 30`，
     首轮 30 条只过 **1** 条 —— 因为思考 token 混在 completion 里（每条 59~77 个思考，
     正文才几个字），该口径等于要求模型**不许思考**，与思考模式直接冲突。
     这与关卡 2/3 记过的「`completion_tokens` 混着 reasoning」是同一个坑，第三次栽在同一处。

口径（不写清就没法对比）：
  · 模型 / 单价一律走 `.env` 与 `billing.PRICES`，脚本里不出现数字价目
  · 三版 prompt 只在「指令详细程度」上不同：**前缀都会带同一句任务说明**，
    否则差异会混进"换了个任务"
  · 同一条用例三版各跑 **1 次**（30 次调用 ≈ ¥0.1 量级）——
    多次取平均会翻倍花钱，而本关看的是"量级差异"，不是小数后三位
  · 失败（异常 / 空正文）也计入分母：中途退出或跳过会让通过率虚高
"""

import argparse
import json

from openai import OpenAI

from llm_cli.billing import bill
from llm_cli.config import load_settings
from llm_cli.llm import Answer, ask

# ⚠️ 写 `score_one()` 时还要这一个（现在注着，否则 ruff 报 F401 unused）：
#   import json

# ---------------------------------------------------------------------------
# 10 条用例：(原文, 人工标注)。标注是**判据 ①**的答案，改之前先想清楚为什么改
# ---------------------------------------------------------------------------
CASES: list[tuple[str, str]] = [
    ("我这个月账单比上个月多了 80 块，能帮我看看是哪一项涨的吗？", "billing"),
    ("发票抬头写错了，已经开出来的能重开吗？", "billing"),
    ("登录时一直提示密码错误，但我确定没输错", "account"),
    ("想换绑手机号，原来的号码已经不用了", "account"),
    ("导出的 CSV 打开是乱码", "tech"),
    ("批量上传图片时一直卡在 99%", "tech"),
    ("欠费停机以后还能不能登录后台看数据？", "billing"),  # 边界：登录问题，但成因是钱
    ("同事的账号能看到报表，我的看不到", "account"),  # 边界：像权限，也像功能
    ("你们的服务支持开发票吗", "billing"),
    ("想问下明年的价格会不会涨", "other"),  # 边界：像报价，但问的是未来政策
]

VALID = ("billing", "account", "tech", "other")

# ---------------------------------------------------------------------------
# 三版 prompt：只有「指令有多细」这一维在变
# ---------------------------------------------------------------------------
TASK = "把下面的用户问题分类，类别只能是 billing / account / tech / other 之一。"

PROMPTS: dict[str, str] = {
    "short": f"""{TASK}

只输出 JSON：{{"category": "..."}}

问题：""",
    "mid": f"""{TASK}

类别定义：
· billing —— 账单、费用、发票、退款、欠费
· account —— 登录、密码、账号、权限、绑定的手机号邮箱
· tech —— 接口报错、文件格式、上传下载、性能
· other —— 以上都不是

只输出 JSON，不要解释：{{"category": "..."}}

问题：""",
    "long": f"""{TASK}

类别定义：
· billing —— 账单、费用、发票、退款、欠费、价格政策
· account —— 登录、密码、账号、权限、绑定的手机号邮箱
· tech —— 接口报错、文件格式、上传下载、性能
· other —— 客服能力范围之外，或以上都不是

判定规则（按顺序命中即停）：
1. 问题里出现「钱 / 账单 / 发票 / 欠费」且**直接原因是费用** ⇒ billing
2. 问题是「登不上 / 看不到 / 改不了」且**直接原因是身份或权限** ⇒ account
3. 问题是「报错 / 乱码 / 卡住」且**不涉及钱与身份** ⇒ tech
4. 其余一律 other

示例：
输入：这个月扣了两次款 → 输出：{{"category": "billing"}}
输入：我忘记密码了 → 输出：{{"category": "account"}}
输入：页面一直转圈打不开 → 输出：{{"category": "tech"}}
输入：你们公司有多少人 → 输出：{{"category": "other"}}

只输出 JSON，不要解释，不要 Markdown 围栏：{{"category": "..."}}

问题：""",
}


class CaseResult:
    """一条用例的结果（判据 ①②③ 的原料都在这里）。"""

    def __init__(self, label: str, answer: Answer | None, ok_a: bool, ok_b: bool) -> None:
        self.label = label
        self.answer = answer
        self.ok_a = ok_a
        self.ok_b = ok_b
        self.ok_c = answer is not None and self.visible_tokens <= 30

    @property
    def visible_tokens(self) -> int:
        """**可见输出** token = `completion_tokens - reasoning_tokens`。

        `reasoning_tokens` 为 None（供应商没给）时按 0 算 ⇒ 等于用 completion，
        是**保守**方向（宁可判它超长，也不白送分）。
        """
        if self.answer is None:
            return 0
        return self.answer.completion_tokens - (self.answer.reasoning_tokens or 0)


class VariantResult:
    """一版 prompt 的汇总（判据 ① 要报的那几个数）。"""

    def __init__(self, name: str) -> None:
        self.name = name
        self.cases: list[CaseResult] = []
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.cached_tokens = 0  # 输入里命中缓存的部分（长版前缀够长，可能 >0）
        self.reasoning_tokens = 0  # 思考 token（含在 completion 里，只用于解释成本）
        self.cost_yuan = 0.0

    @property
    def score(self) -> int:
        """效果总分：10 条 × 3 项 = 30 分满分。

        `ok_a/ok_b/ok_c` 是 `bool`，`True + True + False == 2`（bool 是 int 子类），
        所以三项直接相加就是这一条的得分。
        """
        return sum(c.ok_a + c.ok_b + c.ok_c for c in self.cases)

    @property
    def cost_per_score(self) -> float:
        """每分效果的成本（判据 ① 的最后一个数）：总成本 ÷ 效果总分。

        0 分时返回 `inf` 而不是 0：一分没得还花了钱，成本是"无穷大"；
        返回 0 会被读成"免费"，方向完全相反。
        """
        score = self.score
        if score == 0:
            return float("inf")
        return self.cost_yuan / score


def score_one(answer: Answer | None, expected: str) -> tuple[bool, bool]:
    """判据 ①② 两项，返回 `(类别正确, 格式合规)`。

    实现要点：
      · `answer is None`（调用抛异常）⇒ `(False, False)`，别让异常吞掉分母
      · **A 口径（2026-11-22 定稿）**：正文 `strip()` 之后必须**直接**是合法 JSON ——
        围栏、解释文字、尾部杂字一律判**不合规**（prompt 里写了"不要围栏"，那就是没遵守指令；
        只削空白，**不剥符号、不抠括号**）
      · 格式合规要求：顶层是 dict、键集合**恰好** == {"category"}、值 ∈ VALID
      · 类别正确 = 合规 **且** `parsed["category"] == expected`
        （不合规就没有"类别"可言，别让 parse 失败的顺便算对）
    """
    answer_text = answer.text if answer is not None else ""
    stripped = answer_text.strip()
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return False, False

    if not isinstance(parsed, dict):
        return False, False

    if set(parsed.keys()) != {"category"}:
        return False, False

    if parsed["category"] not in VALID:
        return False, False

    return parsed["category"] == expected, True


def run_variant(
    name: str, client: OpenAI, model: str, cases: list[tuple[str, str]]
) -> VariantResult:
    """跑一版 prompt 的全部用例，累加 token 与成本。

    实现要点：
      · 每条：`ask(prompt=PROMPTS[name] + text, model=model, client=client)`
      · 每次拿到 `Answer` 就 `bill(answer)` 累加 `total_yuan`（**别自己乘单价**）
      · **同时累加四个原始数**：`prompt_tokens` / `completion_tokens` /
        `cached_tokens`（None 记 0）/ `reasoning_tokens`（None 记 0）
      · 异常（请求失败）⇒ `answer=None` 记一条 `(False, False)`，**继续跑下一条**
        —— 中途退出会让通过率失去意义
      · 每条只 print 一行进度：请求失败是**环境问题**，要看得见；
        但账单明细留给最后的汇总表，30 行会把终端冲烂
    """
    result = VariantResult(name)
    total = len(cases)
    for index, (text, expected) in enumerate(cases, 1):
        answer: Answer | None = None
        try:
            answer = ask(prompt=PROMPTS[name] + text, model=model, client=client)
        # 单条失败不该打断整批（要继续跑完），否则通过率失去意义
        except Exception as exc:
            print(f"  [{index}/{total}] 请求失败：{type(exc).__name__}: {exc}")

        if answer is not None:
            result.prompt_tokens += answer.prompt_tokens
            result.completion_tokens += answer.completion_tokens
            result.cached_tokens += answer.cached_tokens or 0
            result.reasoning_tokens += answer.reasoning_tokens or 0
            result.cost_yuan += bill(answer).total_yuan

        ok_a, ok_b = score_one(answer, expected)
        result.cases.append(CaseResult(expected, answer, ok_a, ok_b))
        print(f"  [{index}/{total}] {expected:8s} 通过 {ok_a}/{ok_b}")

    return result


def print_summary(results: list[VariantResult]) -> None:
    """判据 ① 要的那张表（AI 代写：纯排版）。"""

    print()
    print(f"{'版本':6s} {'调用':>4s} {'输入tok':>9s} {'命中tok':>9s} {'输出tok':>9s} "
          f"{'其中思考':>9s} {'可见输出':>9s} {'成本(元)':>10s} {'得分':>5s} {'每分成本':>10s}")
    for r in results:
        visible = r.completion_tokens - r.reasoning_tokens
        print(f"{r.name:6s} {len(r.cases):4d} {r.prompt_tokens:9d} {r.cached_tokens:9d} "
              f"{r.completion_tokens:9d} {r.reasoning_tokens:9d} {visible:9d} "
              f"{r.cost_yuan:10.4f} {r.score:5d} {r.cost_per_score:10.6f}")


def main(argv: list[str] | None = None) -> int:
    """入口（AI 代写胶水：参数 → 建 client → 交给 `run_variant`）。"""
    parser = argparse.ArgumentParser(prog="bench_prompt")
    parser.add_argument("--smoke", action="store_true", help="每版只跑 1 条，验通路")
    parser.add_argument("--variant", choices=sorted(PROMPTS), default=None, help="只跑某一版")
    args = parser.parse_args(argv)

    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url)
    model = settings.model

    names = [args.variant] if args.variant else list(PROMPTS)
    todo = CASES[:1] if args.smoke else CASES

    results = [run_variant(name, client, model, todo) for name in names]
    print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
