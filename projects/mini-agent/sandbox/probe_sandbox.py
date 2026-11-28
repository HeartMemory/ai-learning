"""L5 关 0：真沙箱探针（Windows 受限令牌 + 合成 SID + 受限 ACL，免提权）。

五条判据的唯一家 = `README.md` 阶段表 L5 关 0。本文件按块长大，**当前只到判据 C**。

判据 C（本块）= **反向对照**：同一段「写沙箱外」的代码，**不开受限令牌**直接跑 ⇒ 必须成功。
  为什么它是灵魂：若不开令牌都写不进去，那 B（沙箱外写被拒）的失败只能说明
  「这个路径本来就不能写」，与令牌无关 ⇒ 整条 A/B 实验作废。**C 绿了，B 红才有意义。**

本块只建两件东西：
  1. `child_write.py` —— 被测代码，A/B/C 共用同一份
  2. `run_child()`     —— 起子进程的唯一入口；后续 A/B 只多传一个令牌句柄，
                          别的都不变 —— 这就是「归因成立」的工程保证

跑法：`uv run python sandbox/probe_sandbox.py`
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
CHILD_SCRIPT = SCRIPT_DIR / "child_write.py"

# 受限令牌将来只给这个目录写权限；本块不用，先占位（已在 .gitignore）
SANDBOX_ROOT = SCRIPT_DIR / ".sandbox-root"

# 「沙箱外」= 系统临时目录下的探针目录：当前用户可写，但不在 SANDBOX_ROOT 里
OUTSIDE_TARGET = Path(tempfile.gettempdir()) / "mini-agent-sandbox-probe" / "outside.txt"


def run_child(target: Path, text: str) -> tuple[int, str]:
    """用当前解释器起一个 child_write.py 子进程，返回 (退出码, 合并输出)。

    ⚠️ 这个函数**永远裸跑**（不带令牌）：`subprocess` 没有"指定令牌"的接口。
    它存在的意义 = 判据 C 的对照组。带令牌起进程在 `spawn_restricted.py`
    （`CreateProcessAsUserW`）—— 两条路径、同一份 `child_write.py`，这才是归因结构。
    （原签名里的 `token` 参数已删：那是个死参数，会让人误以为 subprocess 能带令牌。）

    自己写时对着这几条想：
      - child 的 stdout 与 stderr **都要收全**（判据看的是原文，丢了就没证据）
      - 解释器用 `sys.executable`，别写裸 `python`（那会跑到系统解释器，不是 .venv）
      - 起进程本身失败也是信息（`FileNotFoundError` 等）⇒ 别让它没头没尾地崩
      - **本块别动环境变量**（判据 D 才做白名单，提前做会污染 C 的对照）
    """
    argv = [sys.executable, str(CHILD_SCRIPT), str(target), text]
    try:
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            encoding="utf-8",
            errors="backslashreplace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except OSError as exc:
        return 127, f"起进程失败: {type(exc).__name__}: {exc}"
    return proc.returncode, proc.stdout


def judge_c_baseline() -> bool:
    """判据 C：不开令牌，直接写 OUTSIDE_TARGET，必须成功。

    三条都要成立才算绿（缺一条都可能是「假绿」）：
      ① 退出码 == 0
      ② 文件真实存在
      ③ 读回内容 == 写进去的内容（逐字相等）

    自己写时对着这几条想：
      - **先删掉上次跑剩的目标文件**，否则「文件存在 + 内容相同」可能全是上一轮的
        留影 —— 这是本判据最容易造出来的假绿
      - 红的时候把子进程输出**原样打出来**，那是归因的唯一证据
      - 三条各自打印一行，别只打印一个总的 OK / NG
    """
    OUTSIDE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    OUTSIDE_TARGET.unlink(missing_ok=True)
    code, output = run_child(OUTSIDE_TARGET, "hello world")
    if code != 0:
        print(f"判据 C 失败：子进程退出码 {code}，输出：\n{output}")
        return False
    if not OUTSIDE_TARGET.exists():
        print(f"判据 C 失败：目标文件 {OUTSIDE_TARGET} 不存在")
        return False
    if OUTSIDE_TARGET.read_text(encoding="utf-8") != "hello world":
        print("判据 C 失败：读回内容与写进去的内容不一致")
        return False
    return True


def main() -> int:
    """胶水：跑判据 C，末行打印结论。"""
    ok = judge_c_baseline()
    print(f"judge C: {'OK' if ok else 'NG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
