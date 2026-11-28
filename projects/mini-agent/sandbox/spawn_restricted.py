"""L5 关 0 块 3b：用受限令牌起子进程，并把 A / B 两条判据取证。

本块解决三件事（前两块已就绪：令牌、ACL）：
  ① 用 `CreateProcessAsUserW` 带着受限令牌起子进程（`subprocess` 做不到）
  ② 把子进程的 stdout / stderr 落到**沙箱内**的文件里 ⇒ 能拿到失败原文
  ③ 判据 A（沙箱内写成功）/ B（沙箱外被拒且文件不存在）

为什么用「重定向到文件」而不是 `CreatePipe`：
  管道要多 4 个 API（CreatePipe / SetHandleInformation / ReadFile / PeekNamedPipe），
  本块只需要**事后**读 texts，文件够用。⚠️ 代价（必须记）：子进程拿到的是父进程
  开好的**句柄**，写句柄时不再做路径访问检查 ⇒ 「日志文件」这条路径绕过了沙箱。
  所以日志文件只能放在 `.sandbox-root` 里，不能借此探测别处的可写性。
  真正的流式输出留给 L5 关 1 的工具层（那时才需要管道）。

降级记录（如实记）：本文件的 ctypes 接线（结构体 / 原型 / CreateFileW 继承句柄）
与 `spawn_restricted` 由 AI 代写（2026-11-28），理由 = 纯 Win32 样板；
判据逻辑（judge_ab）与"原文长什么样"的解释仍由本人核对。

跑法：`uv run python sandbox/spawn_restricted.py`
"""

import ctypes
import os
import subprocess
import sys
import tempfile
from ctypes import wintypes
from pathlib import Path

from win_acl import set_default_dacl
from win_token import (
    EVERYONE_SID,
    _advapi32,
    _check,
    _kernel32,
    logon_sid_text,
    make_restricted_token,
    open_process_token,
    restricted_sid_count,
    sid_from_string,
)
from win_token import SANDBOX_WRITE_SID as _SANDBOX_WRITE_SID

# 判据 E 的「注入坏一处」：设 MINI_AGENT_INJECT=bad-sid 让 SID 语法非法，
# 令牌必然造不出来 ⇒ 脚本必须红着退出，绝不静默降级成裸跑
if os.environ.get("MINI_AGENT_INJECT") == "bad-sid":
    SANDBOX_WRITE_SID = "S-1-5-XX"
else:
    SANDBOX_WRITE_SID = _SANDBOX_WRITE_SID

SCRIPT_DIR = Path(__file__).resolve().parent
CHILD_SCRIPT = SCRIPT_DIR / "child_write.py"
SANDBOX_ROOT = SCRIPT_DIR / ".sandbox-root"
INSIDE_TARGET = SANDBOX_ROOT / "a.txt"
OUTSIDE_TARGET = Path(tempfile.gettempdir()) / "mini-agent-sandbox-probe" / "outside.txt"
INVALID_HANDLE_VALUE = ctypes.c_void_p(-1).value

# TOKEN_ALL_ACCESS：CreateProcessAsUser 需要 TOKEN_ASSIGN_PRIMARY 等，
# 只用 DUPLICATE|QUERY 会在调用时直接 WinError 5（11-28 实测）
TOKEN_ALL_ACCESS = 0xF01FF

CREATE_NO_WINDOW = 0x08000000
# ⚠️ 传 Unicode 环境块必须带这个标志，否则系统按 ANSI 解析 ⇒ WinError 87（参数错误）
CREATE_UNICODE_ENVIRONMENT = 0x00000400
STARTF_USESTDHANDLES = 0x00000100
GENERIC_WRITE = 0x40000000
CREATE_ALWAYS = 2
FILE_ATTRIBUTE_NORMAL = 0x80
INFINITE_WAIT_MS = 15000

# 判据 D 的环境白名单：只传这些键，**含 LLM_API_KEY 的变量根本不进来**
# （不是"事后删"，是"从来没给"）
ENV_WHITELIST = (
    "PATH",
    "PATHEXT",
    "SYSTEMROOT",
    "SYSTEMDRIVE",
    "WINDIR",
    "COMSPEC",
    "USERPROFILE",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "PYTHONIOENCODING",  # 让子进程按 UTF-8 输出，父进程读日志才不乱码
    "TEMP",
    "TMP",
)


class STARTUPINFOW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("lpReserved", wintypes.LPWSTR),
        ("lpDesktop", wintypes.LPWSTR),
        ("lpTitle", wintypes.LPWSTR),
        ("dwX", wintypes.DWORD),
        ("dwY", wintypes.DWORD),
        ("dwXSize", wintypes.DWORD),
        ("dwYSize", wintypes.DWORD),
        ("dwXCountChars", wintypes.DWORD),
        ("dwYCountChars", wintypes.DWORD),
        ("dwFillAttribute", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("wShowWindow", wintypes.WORD),
        ("cbReserved2", wintypes.WORD),
        ("lpReserved2", ctypes.c_void_p),
        ("hStdInput", wintypes.HANDLE),
        ("hStdOutput", wintypes.HANDLE),
        ("hStdError", wintypes.HANDLE),
    ]


class PROCESS_INFORMATION(ctypes.Structure):
    _fields_ = [
        ("hProcess", wintypes.HANDLE),
        ("hThread", wintypes.HANDLE),
        ("dwProcessId", wintypes.DWORD),
        ("dwThreadId", wintypes.DWORD),
    ]


class SECURITY_ATTRIBUTES(ctypes.Structure):
    _fields_ = [
        ("nLength", wintypes.DWORD),
        ("lpSecurityDescriptor", ctypes.c_void_p),
        ("bInheritHandle", wintypes.BOOL),
    ]


_advapi32.CreateProcessAsUserW.argtypes = [
    wintypes.HANDLE,  # hToken
    wintypes.LPCWSTR,  # lpApplicationName
    wintypes.LPWSTR,  # lpCommandLine（可被系统改写，必须给可写缓冲）
    ctypes.c_void_p,  # lpProcessAttributes
    ctypes.c_void_p,  # lpThreadAttributes
    wintypes.BOOL,  # bInheritHandles
    wintypes.DWORD,  # dwCreationFlags
    ctypes.c_void_p,  # lpEnvironment（None = 继承父进程，判据 D 再改）
    wintypes.LPCWSTR,  # lpCurrentDirectory
    ctypes.POINTER(STARTUPINFOW),  # lpStartupInfo
    ctypes.POINTER(PROCESS_INFORMATION),  # lpProcessInformation
]
_advapi32.CreateProcessAsUserW.restype = wintypes.BOOL

_kernel32.CreateFileW.argtypes = [
    wintypes.LPCWSTR,
    wintypes.DWORD,
    wintypes.DWORD,
    ctypes.POINTER(SECURITY_ATTRIBUTES),
    wintypes.DWORD,
    wintypes.DWORD,
    wintypes.HANDLE,
]
_kernel32.CreateFileW.restype = wintypes.HANDLE
_kernel32.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
_kernel32.WaitForSingleObject.restype = wintypes.DWORD
_kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
_kernel32.GetExitCodeProcess.restype = wintypes.BOOL
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL


def sandbox_token() -> tuple[int, list[int]]:
    """造好受限令牌并重写默认 DACL，返回 `(令牌句柄, 需要释放的 SID 指针列表)`。

    口径（11-28 消融实验定）：限制 SID = `[logon, Everyone, 合成]`。
      - 去掉 logon    → 连 `whoami` 都 0xC0000142
      - 去掉 Everyone → `whoami` 能跑，**Python 起不来**（0xC0000142）
      - 去掉合成      → 进程能跑，但写沙箱内也被拒
    另外默认 DACL 必须重写（否则子进程自己新建的控制台 / IPC 对象打不开）。
    """
    base = 0
    logon = 0
    everyone = 0
    sandbox = 0
    try:
        base = open_process_token(TOKEN_ALL_ACCESS)
        logon = sid_from_string(logon_sid_text(base))
        everyone = sid_from_string(EVERYONE_SID)
        sandbox = sid_from_string(SANDBOX_WRITE_SID)
        token = make_restricted_token(base, [logon, everyone, sandbox])
        # 受托人用合成 SID（不是 Everyone）：默认 DACL 决定「自己新建的对象对谁开放」，
        # 用合成 SID 更窄 —— 只有带这个 SID 的进程能打开沙箱子进程建的命名对象。
        set_default_dacl(token, sandbox)
        return token, [logon, everyone, sandbox]
    finally:
        if base:
            _kernel32.CloseHandle(base)


def build_env_block(overrides: dict[str, str]) -> ctypes.Array:
    """按**白名单**拼 Windows 环境块（`K=V\\0K=V\\0\\0` 的宽字符串）。

    判据 D 的口径：**只传允许的键**（不是拿 `os.environ` 事后删）。键不在白名单里 ⇒
    子进程根本没有这个变量，连"删没删干净"都不用担心。

    `overrides` 用来覆盖白名单里的值（本探针把 TEMP/TMP 指到 `.sandbox-root`，
    否则 Python 启动时找不到可写临时目录，会打 `No usable temporary directory`）。
    """
    lines = []
    for key in ENV_WHITELIST:
        value = overrides.get(key, os.environ.get(key))
        if value is not None:
            lines.append(f"{key}={value}")
    text = "\0".join(lines) + "\0\0"
    return ctypes.create_unicode_buffer(text)


# 默认环境块：白名单 + TEMP/TMP 指进沙箱 + 强制 UTF-8 输出。
# 真实工具层就该这么起进程 —— 不留「默认继承父环境」的退路，否则谁漏传参数就悄悄泄 key。
# 注：PYTHONIOENCODING 是 Python 专用开关，系统不会给、父进程也没有 ⇒ 必须在这里写死，
#     否则子进程按 gbk 写日志，父进程按 utf-8 读 ⇒ 中文变乱码。
DEFAULT_ENV = build_env_block(
    {
        "TEMP": str(SANDBOX_ROOT),
        "TMP": str(SANDBOX_ROOT),
        "PYTHONIOENCODING": "utf-8",
    }
)


def spawn_restricted(
    token: int, argv: list[str], log_path: Path, env_block: ctypes.Array | None = DEFAULT_ENV
) -> int:
    """用 token 起一个子进程，stdout / stderr 合并写进 `log_path`；返回退出码。

    ⭐ 降级记录（2026-11-28）：方法体 **AI 代写 + 本人照抄填回**（不是本人独立写的）。
    补练待做：不看本文件，只照下面六步重推一遍；判据 A/B/D/E 会自动批改。

    六步（照顺序想，不要跳）：
      ① 开日志文件：`_kernel32.CreateFileW(str(log_path), GENERIC_WRITE, 0x3,
         byref(attr), CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, None)`
         - `attr` 是 `SECURITY_ATTRIBUTES`，`nLength = sizeof(attr)`、
           **`bInheritHandle = True`**（子进程要继承这个句柄才能往它写）
         - 返回 `INVALID_HANDLE_VALUE` 就是失败 → 抛 `ctypes.WinError`
      ② 填 `STARTUPINFOW`：`cb = sizeof(...)`、`dwFlags = STARTF_USESTDHANDLES`、
         `hStdOutput = hStdError = 刚开的句柄`（合并两条流到同一个文件）
      ③ 命令行：`ctypes.create_unicode_buffer(subprocess_quote(argv))`
         —— 必须是**可写**缓冲，系统会就地改它
      ④ `_check(_advapi32.CreateProcessAsUserW(...11 个参数...), "CreateProcessAsUserW")`
         - `lpApplicationName = argv[0]`，`lpCommandLine = 那个缓冲`
         - `bInheritHandles = True`
         - 标志：`CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT`（少了后者就是 WinError 87）
         - 环境：`ctypes.cast(env_block, ctypes.c_void_p) if env_block else None`
         - 当前目录：`str(SCRIPT_DIR.parent)`
      ⑤ 等结束 + 取退出码：`WaitForSingleObject(info.hProcess, INFINITE_WAIT_MS)`
         → `GetExitCodeProcess(info.hProcess, byref(code))` → `return code.value`
      ⑥ 清理（`finally`，三处，别漏）：进程句柄、线程句柄、日志文件句柄
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # ① 可继承的 SECURITY_ATTRIBUTES：子进程要继承日志句柄，必须让它是 inheritable
    attributes = SECURITY_ATTRIBUTES()
    attributes.nLength = ctypes.sizeof(attributes)
    attributes.lpSecurityDescriptor = None
    attributes.bInheritHandle = True

    # ② 开日志文件（覆盖写）
    handle = _kernel32.CreateFileW(
        str(log_path),
        GENERIC_WRITE,
        0x3,                         # FILE_SHARE_READ | FILE_SHARE_WRITE
        ctypes.byref(attributes),
        CREATE_ALWAYS,
        FILE_ATTRIBUTE_NORMAL,
        None,
    )
    if handle == INVALID_HANDLE_VALUE:
        raise ctypes.WinError(ctypes.get_last_error(), "CreateFileW(日志文件) 失败")

    try:                             # ← 从这里开始，handle 要还
        # ③ 让子进程的 stdout / stderr 都指向这个文件
        startup = STARTUPINFOW()
        startup.cb = ctypes.sizeof(startup)
        startup.dwFlags = STARTF_USESTDHANDLES
        startup.hStdOutput = handle
        startup.hStdError = handle

        info = PROCESS_INFORMATION()
        command = ctypes.create_unicode_buffer(subprocess_quote(argv))

        # ④ 带令牌起进程（11 个参数，顺序 = argtypes 里的顺序）
        _check(
            _advapi32.CreateProcessAsUserW(
                token,                                    # ① hToken
                argv[0],                                  # ② lpApplicationName
                command,                                  # ③ lpCommandLine（可写缓冲）
                None,                                     # ④ lpProcessAttributes
                None,                                     # ⑤ lpThreadAttributes
                True,                                     # ⑥ bInheritHandles
                CREATE_NO_WINDOW | CREATE_UNICODE_ENVIRONMENT,  # ⑦ dwCreationFlags
                ctypes.cast(env_block, ctypes.c_void_p) if env_block else None,  # ⑧
                str(SCRIPT_DIR.parent),                   # ⑨ lpCurrentDirectory
                ctypes.byref(startup),                    # ⑩ lpStartupInfo
                ctypes.byref(info),                       # ⑪ lpProcessInformation
            ),
            "CreateProcessAsUserW",
        )

        try:                         # ← 从这里开始，info 也要还
            # ⑤ 等结束、取退出码
            _kernel32.WaitForSingleObject(info.hProcess, INFINITE_WAIT_MS)
            code = wintypes.DWORD()
            _check(
                _kernel32.GetExitCodeProcess(info.hProcess, ctypes.byref(code)),
                "GetExitCodeProcess",
            )
            return code.value
        finally:
            _kernel32.CloseHandle(info.hThread)
            _kernel32.CloseHandle(info.hProcess)
    finally:
        _kernel32.CloseHandle(handle)



def subprocess_quote(argv: list[str]) -> str:
    """把参数表拼成命令行：含空格或为空的一律加双引号（够本探针用）。"""
    parts = []
    for item in argv:
        parts.append(f'"{item}"' if (not item or " " in item) else item)
    return " ".join(parts)


def judge_ab() -> bool:
    """判据 A / B 一起跑，各打证据行。

    A：写 `.sandbox-root/a.txt` ⇒ 退出码 0 + 文件存在 + 内容逐字相等
    B：写临时目录下的 outside.txt ⇒ 退出码非 0 + 文件**不存在** + 日志里有错误原文
    """
    token = 0
    sids: list[int] = []
    try:
        token, sids = sandbox_token()
        print(f"① 限制 SID 数 = {restricted_sid_count(token)}（期望 3）")

        INSIDE_TARGET.parent.mkdir(parents=True, exist_ok=True)
        INSIDE_TARGET.unlink(missing_ok=True)
        inside_log = SANDBOX_ROOT / "inside.log"
        inside_log.unlink(missing_ok=True)
        code_a = spawn_restricted(
            token, [sys.executable, str(CHILD_SCRIPT), str(INSIDE_TARGET), "hello"], inside_log
        )
        text_a = inside_log.read_text(encoding="utf-8", errors="backslashreplace").strip()
        ok_a = code_a == 0 and INSIDE_TARGET.exists()

        OUTSIDE_TARGET.parent.mkdir(parents=True, exist_ok=True)
        OUTSIDE_TARGET.unlink(missing_ok=True)
        outside_log = SANDBOX_ROOT / "outside.log"
        outside_log.unlink(missing_ok=True)
        code_b = spawn_restricted(
            token, [sys.executable, str(CHILD_SCRIPT), str(OUTSIDE_TARGET), "hello"], outside_log
        )
        text_b = outside_log.read_text(encoding="utf-8", errors="backslashreplace").strip()
        ok_b = code_b != 0 and not OUTSIDE_TARGET.exists()

        print(f"A 沙箱内: exit={code_a} 文件存在={INSIDE_TARGET.exists()} | {text_a}")
        print(f"B 沙箱外: exit={code_b} 文件存在={OUTSIDE_TARGET.exists()} | {text_b}")
        return ok_a and ok_b
    except OSError as exc:
        print(f"红：{exc}")
        return False
    finally:
        if token:
            _kernel32.CloseHandle(token)
        for sid in sids:
            _kernel32.LocalFree(sid)


def judge_d() -> bool:
    """判据 D：子进程 `os.environ` 里**没有** `LLM_API_KEY`，密钥走环境**白名单**。

    两路取证（缺一路就可能假绿）：
      ① 白名单块：子进程键数 + `has_key False`
      ② 对照（继承父环境）：`has_key True` —— 证明"父进程真的持有、子进程真的没拿到"，
         而不是"本来就没有所以测不出"

    父进程若本来没有 key，就临时塞一个**假值**进 `os.environ` 再测（结束还原），
    这样对照成立，且完全不碰真密钥。
    """
    token = 0
    sids: list[int] = []
    key = "LLM_API_KEY"
    injected = False
    try:
        token, sids = sandbox_token()
        if key not in os.environ:
            os.environ[key] = "dummy-for-judge-d"
            injected = True

        SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
        probe = (
            "import os;print('keys',len(os.environ));"
            "print('has_key','LLM_API_KEY' in os.environ)"
        )
        whitelist_log = SANDBOX_ROOT / "env_whitelist.log"
        whitelist_log.unlink(missing_ok=True)
        # 直接用 DEFAULT_ENV，别在这里再拼一份 —— 之前就踩过：这里漏了 PYTHONIOENCODING，
        # 导致白名单那路按 gbk 输出，中文日志变乱码
        code_w = spawn_restricted(
            token, [sys.executable, "-c", probe], whitelist_log, DEFAULT_ENV
        )
        text_w = whitelist_log.read_text(encoding="utf-8", errors="backslashreplace").strip()

        inherit_log = SANDBOX_ROOT / "env_inherit.log"
        inherit_log.unlink(missing_ok=True)
        code_i = spawn_restricted(token, [sys.executable, "-c", probe], inherit_log, None)
        text_i = inherit_log.read_text(encoding="utf-8", errors="backslashreplace").strip()

        flat_w = text_w.replace("\n", " / ")
        flat_i = text_i.replace("\n", " / ")
        ok = (
            code_w == 0
            and "has_key False" in text_w
            and code_i == 0
            and "has_key True" in text_i
        )
        print(f"父进程持有 {key} = {key in os.environ}")
        print(f"D① 白名单: exit={code_w} | {flat_w}")
        print(f"D② 继承  : exit={code_i} | {flat_i}")
        return ok
    except OSError as exc:
        print(f"红：{exc}")
        return False
    finally:
        if injected:
            os.environ.pop(key, None)
        if token:
            _kernel32.CloseHandle(token)
        for sid in sids:
            _kernel32.LocalFree(sid)


def judge_e() -> bool:
    """判据 E：fail-closed —— 令牌造不出来时必须**红着退出**，绝不静默降级成裸跑。

    注入法（官方"故意写坏一处"）：用 `MINI_AGENT_INJECT=bad-sid` 起一个**子进程版自己**，
    让模块级 SID 变成非法值。三条同时成立才算绿：
      ① 退出码 ≠ 0（不是"照常跑完"）
      ② 输出里有异常原文（走到了 `红：` 分支，不是崩在别处）
      ③ 沙箱外目标文件**没被创建**（若静默降级成裸跑，用父进程权限一写就成功 ⇒ 这里会露）
    """
    OUTSIDE_TARGET.parent.mkdir(parents=True, exist_ok=True)
    OUTSIDE_TARGET.unlink(missing_ok=True)
    env = {**os.environ, "MINI_AGENT_INJECT": "bad-sid", "PYTHONIOENCODING": "utf-8"}
    result = subprocess.run(
        [sys.executable, str(Path(__file__))],
        capture_output=True,
        env=env,
        cwd=str(SCRIPT_DIR.parent),
    )
    raw = result.stdout + result.stderr
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        # 子进程没按 UTF-8 输出时退回 gbk —— 目的是"看得懂"，不是"逐字节保真"
        text = raw.decode("gbk", "backslashreplace")
    ok = result.returncode != 0 and "红：" in text and not OUTSIDE_TARGET.exists()
    first = [line for line in text.splitlines() if "红：" in line]
    print(f"E 注入坏 SID: exit={result.returncode} | {first[0] if first else text.strip()[:80]}")
    print(f"   沙箱外文件被创建 = {OUTSIDE_TARGET.exists()}（必须 False）")
    return ok


def main() -> int:
    """胶水：依次跑 A/B、D、E，末行打印结论。

    注入模式下只跑 A/B 就退出 —— 否则判据 E 会无限递归地起自己。
    """
    if os.environ.get("MINI_AGENT_INJECT"):
        ok_inject = judge_ab()
        print(f"inject 模式: {'OK' if ok_inject else 'NG'}")
        return 0 if ok_inject else 1

    ok_ab = judge_ab()
    print()
    ok_d = judge_d()
    print()
    ok_e = judge_e()
    print()
    ok = ok_ab and ok_d and ok_e
    print(f"sandbox A/B/D/E: {'OK' if ok else 'NG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
