"""接入探针：不写业务代码，先让服务器回答"地址 / key / 模型名"这三件事对不对。

判据（对着返回码看）：
    200            → 域名对、/v1 对、key 对（下面还会列出可用模型名）
    401            → 地址对，key 错（粘不全 / 带空格 / .env 没读到）
    404            → 地址错（域名错，或 /v1 缺/多）
    403            → 地址与 key 都对，但没开通权限（未实名 / 未充值 / 模型未申请）
    连接超时        → 指向了被墙域名，或本地服务没起

⚠️ 有些厂商没实现 /models 端点（会 404）—— 那时改用 chat.completions 打一发，
   看错误里有没有 "model not found"：有 ⇒ 地址和 key 都对，只有模型名错。

用法：uv run python check_llm.py
"""

import io
import json
import os
import sys
import urllib.error
import urllib.request

from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("LLM_BASE_URL", "").strip().rstrip("/")
API_KEY = os.getenv("LLM_API_KEY", "").strip()
MODEL = os.getenv("LLM_MODEL", "").strip()

HINTS = {
    200: "地址 / key 都对",
    400: "地址和 key 大概率对，参数或模型名有问题",
    401: "地址对，key 错（粘不全 / 带空格 / .env 没读到）",
    403: "地址与 key 都对，但没开通权限（实名 / 充值 / 模型申请）",
    404: "地址错（域名错，或 /v1 缺、多）",
    429: "地址和 key 都对，但触发限流 — 等一会再试",
}


def main() -> None:
    if not (BASE_URL and API_KEY):
        raise SystemExit("缺配置：先填好 .env 里的 LLM_BASE_URL 与 LLM_API_KEY")

    print(f"BASE_URL : {BASE_URL}")
    print(f"MODEL    : {MODEL or '<空>'}")
    print(f"探测地址 : {BASE_URL}/models\n")

    request = urllib.request.Request(
        f"{BASE_URL}/models",
        headers={"Authorization": f"Bearer {API_KEY}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            status = response.status
            body = response.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        status = exc.code
        body = exc.read().decode("utf-8", "replace")
    except Exception as exc:  # 探针要如实报错，不吞
        print(f"连接失败：{type(exc).__name__}: {exc}")
        print("⇒ 看判据：连接超时 = 域名被墙或本地服务没起")
        return

    print(f"HTTP {status} —— {HINTS.get(status, '未收录的返回码，把 body 贴出来分析')}\n")

    try:
        ids = [item["id"] for item in json.loads(body).get("data", [])]
    except (json.JSONDecodeError, KeyError, TypeError):
        print("body（前 400 字）：")
        print(body[:400])
        return

    if ids:
        print(f"可用模型 {len(ids)} 个：")
        for model_id in ids:
            mark = "  ← 你在用的" if model_id == MODEL else ""
            print(f"  - {model_id}{mark}")
        if MODEL and MODEL not in ids:
            print(f"\n⚠️ 你填的 MODEL `{MODEL}` 不在上面这个列表里 —— 抄名字时容易漏字符")


if __name__ == "__main__":
    # Windows 控制台默认 GBK，中文/符号输出会 UnicodeEncodeError ⇒ 强制 UTF-8
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8")
    main()
