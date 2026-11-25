"""L2 · 关卡 5 块 3：判据 ② —— 我的手写 BPE vs API 报的 prompt_tokens

一条根：判据 ①（tokenizer.py 的 A 组）只证明「无损」，证明不了「压得像不像」。
同样 2000 字符英文，我切 850 个 token、服务商切 600 个 —— 两边都能用，
但账单差 40%。判据 ② 就是把这个差距量成数字，要求 < 30%。

口径（改必须写理由，并记进本目录 README）：
 ① **差分消模板**：chat 接口的 `prompt_tokens` 含系统模板 token（L4 实测思考模式
    约 26 个）。做法：同一 messages 形态先发 1 个字符拿 `base`，再发目标段拿
    `sample`，**两边相减**；我方也减掉「1 个字符」的 token 数 ⇒ 模板项完全抵消，
    两边比的都是「内容 token 的增量」。少一步减 base 就会把 26 个模板 token 算成
    我方的误差（2000 字符量级下约 4%，能掩盖真问题）。
 ② **对账文本 = 训练语料同分布的英文段**：语料第 `TRAIN_CHARS` 字符之后的
    `SAMPLE_CHARS` 个字符（held-out，不在训练切片里）。理由：判据比的是**词表压缩
    能力**；中文必然差 4 倍量级（本词表 0.33 字符/token，官方约 1.7 汉字/token），
    那不是 bug 而是词表没覆盖 ⇒ 中文只作为判据 ③ 的证据单独报，不参与判 OK/NG。
 ③ **merge 数扫描 200 / 500 / 1000**：判据原文「>= 200」是**下界**，不是等号。
    三档结果**全报**，取「能进 30% 的最小档」判定通过；不许只报最好那档。
 ④ 相对差 = `|mine - api| / api`。**不接受「按字数折算」的替代口径** ——
    那样测的是字数，不是 token。
 ⑤ 本脚本**会真实计费**（2 次调用，约 ¥0.002），按规定**由本人手动跑**：
        uv run python judge_api.py
    `tokenizer.py` 的 `main()` 不 import 本文件，避免误触发。

判据归属：判据 ② 的原文（唯一家）= 根目录 `README.md` 阶段表 L2 行；
本文件只是它的落地取证。
"""

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from corpus import load_corpus
from tokenizer import TRAIN_CHARS, encode, train_bpe

ENV_FILE: Path = Path(__file__).with_name(".env")
"""三件套的落点。`git check-ignore .env` 必须命中（根 .gitignore 的 `.env` 规则）。"""

REQUIRED: tuple[str, ...] = ("LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL")

MERGE_SCAN: tuple[int, ...] = (200, 500, 1000)
"""扫描三档 merge 数：200 = 判据下界，往上两档看压缩率还差多少（口径 ③）。"""

REL_LIMIT: float = 0.30
"""判据 ② 的门槛：相对差 < 30%。"""

SAMPLE_CHARS: int = 2000
"""对账段落长度：约 600 token（官方口径），我方约 850 —— 模板 token 占比 < 5%。"""

BASE_TEXT: str = "x"
"""差分用的锚点文本（1 个字符）：只为了拿到「模板 + 1 个字符」的底噪。"""

MAX_TOKENS: int = 16
"""只要 `prompt_tokens`，输出越短越省；太小（1）会让思考模式被截断甚至报错。"""


@dataclass(frozen=True, slots=True)
class Settings:
    """一次调用需要三样东西。`api_key` 用 `repr=False` 挡住日志泄漏。"""

    api_key: str = field(repr=False)
    base_url: str
    model: str


def load_settings(env_file: Path = ENV_FILE) -> Settings:
    """读 `.env` 取三变量；缺哪项报哪项（退出码 2 = 配置错误）。"""
    load_dotenv(env_file)
    values = {key: os.environ.get(key, "").strip() for key in REQUIRED}
    missing = [key for key, value in values.items() if not value]
    if missing:
        print(
            f"缺配置：{', '.join(missing)} —— 把 .env.example 复制成 .env 填好再跑"
            f"（Copy-Item .env.example .env）",
            file=sys.stderr,
        )
        raise SystemExit(2)
    return Settings(
        api_key=values["LLM_API_KEY"],
        base_url=values["LLM_BASE_URL"].rstrip("/"),
        model=values["LLM_MODEL"],
    )


def count_tokens(client: OpenAI, model: str, text: str) -> int:
    """发一句话，只取平台回报的 `prompt_tokens`（口径 ①：不自己估）。"""
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
        temperature=0,
        max_tokens=MAX_TOKENS,
    )
    usage = resp.usage
    if usage is None or usage.prompt_tokens is None:
        raise RuntimeError("平台没回 usage.prompt_tokens —— 对账无从取证")
    return usage.prompt_tokens


def judge_api() -> bool:
    """判据 ② 主体：训练 3 档词表，各算一次压缩后的 token 数，与 API 增量比。"""
    settings = load_settings()
    client = OpenAI(api_key=settings.api_key, base_url=settings.base_url, timeout=180.0)

    text = load_corpus()
    train_text = text[:TRAIN_CHARS]
    sample = text[TRAIN_CHARS : TRAIN_CHARS + SAMPLE_CHARS]

    print(f"对账文本 {len(sample)} 字符（held-out：语料第 {TRAIN_CHARS} 字符之后）")
    print(f"训练切片 {len(train_text)} 字符（与 tokenizer.py 的 TRAIN_CHARS 同一口径）")

    base_api = count_tokens(client, settings.model, BASE_TEXT)
    sample_api = count_tokens(client, settings.model, sample)
    api_delta = sample_api - base_api
    print(f"API  base {base_api} · sample {sample_api} => 内容 token {api_delta}")

    rows: list[tuple[int, int, int, float]] = []
    passed: list[tuple[int, float]] = []
    for n in MERGE_SCAN:
        vocab, merges = train_bpe(train_text, n)
        mine_base = len(encode(BASE_TEXT, merges, vocab))
        mine = len(encode(sample, merges, vocab))
        mine_delta = mine - mine_base
        rel = abs(mine_delta - api_delta) / api_delta if api_delta else 1.0
        rows.append((n, len(merges), mine_delta, rel))
        if rel < REL_LIMIT:
            passed.append((n, rel))
        print(
            f"  merge {len(merges):>4}（要求 {n}）· 词表 {len(vocab):>5}"
            f" · 我 {mine_delta:>5} token · 相对差 {rel * 100:>6.2f}%"
            f" {'< 30%' if rel < REL_LIMIT else '超 30%'}"
        )

    ok = bool(passed)
    if ok:
        best_n, best_rel = passed[0]
        print(f"判据 ②：最小达标档 {best_n} merge（相对差 {best_rel * 100:.2f}%）  OK")
    else:
        worst = min(rel for _, _, _, rel in rows)
        print(
            f"判据 ②：三档全部超 30%（最好 {worst * 100:.2f}%）  NG"
            f" —— 按纪律「允许降级通过」写清跳过了什么、为什么，不许改判据"
        )
    return ok


def main() -> None:
    """测试入口（纯胶水）：只跑判据 ②，末行打印一行结论。"""
    judge_api()


if __name__ == "__main__":
    main()
