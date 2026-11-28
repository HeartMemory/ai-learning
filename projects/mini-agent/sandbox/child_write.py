"""沙箱探针的「被测代码」：一个只做一件事的子进程 —— 往指定路径写一段文本。

为什么单独一个文件：判据 A / B / C 必须跑**同一段代码**，差别只在「有没有受限令牌」。
把「身份」（令牌）和「动作」（写文件）拆开，归因才成立：
  A 绿 + B 红 + C 绿 ⇒ 「写外部失败」只能归因于令牌，不是代码写错了、也不是路径本身不可写。

用法（由 probe_sandbox.py 用当前解释器拉起，也可自己手动跑）：
    uv run python sandbox/child_write.py <目标路径> <要写的内容>

输出契约（探针靠它判绿红，**别改格式**）：
    成功：打印 `OK <完整路径> <字节数>`，退出码 0
    失败：打印 `FAIL <异常类型> <错误原文>`，退出码 非 0
           —— Windows 拒绝访问时，原文里应能看到 `[WinError 5]`
"""

import sys
from pathlib import Path


def write_and_report(target: Path, text: str) -> int:
    """写 target，按上面的输出契约打印，返回退出码。

    自己写时对着这几条想：
      - 父目录可能不存在 ⇒ 要不要 mkdir？先想清楚本探针要证明的是「能写文件」
        还是「能建目录」，别用 mkdir 把要卡的东西绕过去
      - **只接 OSError**（只有它拿得到 WinError 原文）；其余异常让它冒泡，吞掉就丢了归因
      - 结果用**退出码**表达，不是用 print 表达（探针读的是退出码 + 原文，两者都要）
    """
    try:
        target.write_text(text,encoding="utf-8")
    except OSError as exc:
        print(f"FAIL: {type(exc).__name__} {exc}")
        return 1
    print(f"OK {target} {len(text.encode("utf-8"))}")
    return 0


def main(argv: list[str]) -> int:
    """胶水：argv = [目标路径, 要写的内容] → 交给 write_and_report。"""
    if len(argv) != 2:
        print("用法: child_write.py <目标路径> <要写的内容>", file=sys.stderr)
        return 2
    return write_and_report(Path(argv[0]), argv[1])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
