"""L5 关 0 块 2：合成 SID + 受限令牌（**只造令牌，不用令牌**）。

本块自检两条（**不是**关 0 的判据编号，判据 A–E 的唯一家仍是 `README.md` 阶段表）：
  ① 合成 SID 字符串 → `ConvertStringSidToSidW` 返回非 0，拿到可用的 SID 指针
  ② `CreateRestrictedToken` 造出的令牌，`IsTokenRestricted()` == True
  fail-closed：任一步系统调用返回 0 ⇒ 抛带 WinError 原文的 OSError，绝不返回"半个令牌"

为什么先只造令牌，不建 ACL、不起子进程：
  万一 `CreateProcessAsUserW` 报 1314（缺 SE_INCREASE_QUOTA），必须能一眼分清是
  「**令牌造不出来**」还是「**令牌造出来了但起不了进程**」。三件事分开验，才谈得上归因。

边界（不许超出声称）：
  - 本块**不**证明「令牌卡住了写」—— 那要靠 ACL（块 3）+ 受限子进程（块 3）
  - 令牌只在当前进程内有效，进程退出即失效

跑法：`uv run python sandbox/win_token.py`
"""

import ctypes
import sys
from ctypes import wintypes

# --- Win32 常量（值抄自 securitybaseapi.h / winnt.h） ---
TOKEN_DUPLICATE = 0x0002
TOKEN_QUERY = 0x0008
DISABLE_MAX_PRIVILEGE = 0x0001
WRITE_RESTRICTED = 0x0008

# TOKEN_INFORMATION_CLASS 里的编号（winnt.h 枚举顺序）
# 2  = TokenGroups            → 令牌里的组（用来找 logon SID）
# 11 = TokenRestrictedSids    → 限制 SID 列表
TOKEN_GROUPS_CLASS = 2
TOKEN_RESTRICTED_SIDS = 11

# SID_AND_ATTRIBUTES.Attributes 里的标志位：这一组是本登录会话的 logon SID
SE_GROUP_LOGON_ID = 0xC0000000

# 合成 SID：只要求**语法合法**，不要求系统上真有这个账户
# （官方原文：ConvertStringSidToSidW 只做格式解析，不查 SAM / 域控）
# 用固定值而不是随机生成 —— 判据要可重复，随机 SID 会让每次 ACL 都不同
SANDBOX_WRITE_SID = "S-1-5-21-0-0-0-1000"

# 限制 SID 列表固定三项：logon SID + Everyone + 合成 SID（口径跑前写死，不跑完再挑）
# 为什么必须有 logon SID：WinSta0 / 桌面等 per-logon 对象的 ACL 里只有它，
#   少了它，受限进程在 kernel32/user32 初始化阶段就死（0xC0000142）。
# 为什么不加「用户本人 SID」：%TEMP% 的 ACL 里有 Elysia:(F)，加进去判据 B 立刻失效。
EVERYONE_SID = "S-1-1-0"
EXPECTED_RESTRICTING_SIDS = 3


class SID_AND_ATTRIBUTES(ctypes.Structure):
    """「SID + 属性」结构，CreateRestrictedToken / SetEntriesInAcl 共用。

    Attributes 必须是 0：限制 SID 在访问检查中总是启用的，传别的值是错的。
    """

    _fields_ = [("Sid", ctypes.c_void_p), ("Attributes", wintypes.DWORD)]


class _TOKEN_GROUPS(ctypes.Structure):
    """只为算「数组在缓冲区里的偏移」而存在（TOKEN_GROUPS 变长，真实个数靠 GroupCount）。"""

    _fields_ = [("GroupCount", wintypes.DWORD), ("Groups", SID_AND_ATTRIBUTES * 1)]


_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

# argtypes 必须显式声明：句柄是 64 位，ctypes 默认按 C int 传参会把高位截掉
_kernel32.GetCurrentProcess.argtypes = []
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
_kernel32.CloseHandle.restype = wintypes.BOOL
_kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
_kernel32.LocalFree.restype = wintypes.HLOCAL

_advapi32.OpenProcessToken.argtypes = [
    wintypes.HANDLE,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.HANDLE),
]
_advapi32.OpenProcessToken.restype = wintypes.BOOL

_advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.HANDLE)]
_advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL

_advapi32.ConvertSidToStringSidW.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.LPWSTR)]
_advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL

_advapi32.CreateRestrictedToken.argtypes = [
    wintypes.HANDLE,  # ExistingTokenHandle
    wintypes.DWORD,  # Flags
    wintypes.DWORD,  # DisableSidCount
    ctypes.POINTER(SID_AND_ATTRIBUTES),  # SidsToDisable
    wintypes.DWORD,  # DeletePrivilegeCount
    ctypes.c_void_p,  # PrivilegesToDelete
    wintypes.DWORD,  # RestrictedSidCount
    ctypes.POINTER(SID_AND_ATTRIBUTES),  # SidsToRestrict
    ctypes.POINTER(wintypes.HANDLE),  # NewTokenHandle
]
_advapi32.CreateRestrictedToken.restype = wintypes.BOOL

_advapi32.GetTokenInformation.argtypes = [
    wintypes.HANDLE,  # TokenHandle
    ctypes.c_int,  # TokenInformationClass
    ctypes.c_void_p,  # TokenInformation（传 None 先问大小）
    wintypes.DWORD,  # TokenInformationLength
    ctypes.POINTER(wintypes.DWORD),  # ReturnLength
]
_advapi32.GetTokenInformation.restype = wintypes.BOOL

# ⚠️ IsTokenRestricted 实测**不可靠**（2026-11-28 本机）：令牌里明明有 1 个限制 SID
# （TokenRestrictedSids 的 GroupCount == 1），它却返回 FALSE，且调用本身是成功的。
# ⇒ 自检 ② 改用 GetTokenInformation(TokenRestrictedSids) 数数，别再信 IsTokenRestricted。


def _check(ok: int, what: str) -> None:
    """fail-closed 的唯一出口：系统调用返回 0 ⇒ 抛带 WinError 原文的 OSError。"""
    if ok:
        return
    raise ctypes.WinError(ctypes.get_last_error(), f"{what} 失败")


def _handle_value(h: wintypes.HANDLE) -> int:
    """取出参句柄里的 int（配 `_check` 用的收窄函数）。

    为什么要多这一层：typeshed 把 `c_void_p.value` 标成 `int | None`（c_void_p 也能
    表示 NULL），类型检查器不认「`_check` 已保证成功」这种上下文 ⇒ 直接 `return h.value`
    会被 Pylance 报 `reportReturnType`。这里显式判一次并收窄成 `int`，
    顺带再兜一次底：万一哪天绕过 `_check` 调了，这里仍然会抛，不会把 0/None 当成功。
    """
    value = h.value  # 先落到局部变量：属性访问的收窄，类型检查器不一定记住
    if value is None:
        raise ctypes.WinError(ctypes.get_last_error(), "出参句柄为空")
    return value


def logon_sid_text(token: int) -> str:
    """把令牌里的 **logon SID**（`SE_GROUP_LOGON_ID` 标记的那个组）转成字符串返回。

    为什么必须要它：WinSta0（窗口站）和 Default（桌面）的 ACL 里只认这个 SID。
    限制 SID 只放合成 SID 时，第二遍访问检查在启动阶段就失败 ⇒ 进程"启动成功"
    但立刻死，退出码 0xC0000142（STATUS_DLL_INIT_FAILED）。

    AI 代写（ctypes 双调用 + 变长结构遍历，纯样板）。为什么**转成字符串**而不是
    直接返回指针：TOKEN_GROUPS 的缓冲区是本函数的局部变量，函数一返回就释放，
    里面的 PSID 会变成悬空指针。转成字符串最省事 —— 调用方随后用 `sid_from_string`
    重新分配一份，所有权和合成 SID 一致，都走 `LocalFree`。
    """
    need = wintypes.DWORD(0)
    _advapi32.GetTokenInformation(token, TOKEN_GROUPS_CLASS, None, 0, ctypes.byref(need))
    buffer = ctypes.create_string_buffer(need.value)
    _check(
        _advapi32.GetTokenInformation(
            token, TOKEN_GROUPS_CLASS, buffer, need.value, ctypes.byref(need)
        ),
        "GetTokenInformation(TokenGroups)",
    )
    count = ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD)).contents.value
    entries = (SID_AND_ATTRIBUTES * count).from_buffer_copy(buffer, _TOKEN_GROUPS.Groups.offset)
    for entry in entries:
        if entry.Attributes & SE_GROUP_LOGON_ID:
            text = wintypes.LPWSTR()
            _check(
                _advapi32.ConvertSidToStringSidW(ctypes.c_void_p(entry.Sid), ctypes.byref(text)),
                "ConvertSidToStringSidW",
            )
            try:
                return text.value or ""
            finally:
                _kernel32.LocalFree(ctypes.cast(text, wintypes.HLOCAL))
    raise OSError(0, "令牌里找不到 logon SID（SE_GROUP_LOGON_ID）")


def restricted_sid_count(token: int) -> int:
    """数一数令牌里到底有几个「限制 SID」（自检 ② 的权威依据）。

    AI 代写（纯 ctypes 双调用样板，不是本块练习点）。两趟调用是这个 API 的标准用法：
      第一趟：缓冲区给 None、长度给 0 ⇒ 必然失败，但 ReturnLength 里带回所需字节数
              （WinError 122 = ERROR_INSUFFICIENT_BUFFER，是**预期**的，不是错）
      第二趟：按大小开缓冲区再调一次 ⇒ 拿回 TOKEN_GROUPS 结构
    TOKEN_GROUPS 的头 4 个字节就是 GroupCount（限制 SID 的个数）。

    为什么不用 IsTokenRestricted：实测它对本探针造出来的令牌返回 FALSE（见上面注释）。
    """
    need = wintypes.DWORD(0)
    _advapi32.GetTokenInformation(
        token, TOKEN_RESTRICTED_SIDS, None, 0, ctypes.byref(need)
    )  # 第一趟故意失败，只为拿 size；返回值和错误码都不用管

    buffer = ctypes.create_string_buffer(need.value)
    _check(
        _advapi32.GetTokenInformation(
            token, TOKEN_RESTRICTED_SIDS, buffer, need.value, ctypes.byref(need)
        ),
        "GetTokenInformation(TokenRestrictedSids)",
    )
    return ctypes.cast(buffer, ctypes.POINTER(wintypes.DWORD)).contents.value


def open_process_token(access: int = TOKEN_DUPLICATE | TOKEN_QUERY) -> int:
    """拿当前进程的令牌句柄（后面所有收窄都以它为底）。

    `access`：要申请的访问权。只造令牌时 `DUPLICATE|QUERY` 够；但要用它
    `CreateProcessAsUser` 起进程时，句柄还得有 `TOKEN_ASSIGN_PRIMARY` 等
    ⇒ 传 `0xF01FF`（TOKEN_ALL_ACCESS），否则调用那一步 `WinError 5`（11-28 实测）。

    自己写时对着这几条想：
      - 权限位至少要 `TOKEN_DUPLICATE`（CreateRestrictedToken 要复制它）+ `TOKEN_QUERY`
      - 出参用 `ctypes.byref(...)` 传一个空的 HANDLE
      - 失败必须走 `_check(...)`，不许返回 0 当"没拿到"
      - 返回 `.value`（int），不是 HANDLE 对象
    """
    h = wintypes.HANDLE()
    _check(
        _advapi32.OpenProcessToken(_kernel32.GetCurrentProcess(), access, ctypes.byref(h)),
        "OpenProcessToken",
    )
    return _handle_value(h)


def sid_from_string(sid_text: str) -> int:
    """SID 字符串 → SID 指针（int）。**调用方负责 LocalFree 释放。**

    自己写时对着这几条想：
      - 官方原文：只管语法、不查账户是否存在 ⇒ 合成 SID 能过
      - 两种入参都合法：`S-1-5-21-...` 标准式，或 `BA` / `WD` 这种缩写
      - 失败走 `_check(...)`
    """
    h = wintypes.HANDLE()
    _check(
        _advapi32.ConvertStringSidToSidW(sid_text, ctypes.byref(h)),
        "ConvertStringSidToSidW",
    )
    return _handle_value(h)


def make_restricted_token(base_token: int, sids: list[int]) -> int:
    """造受限令牌，返回新令牌句柄（int）。`sids` = 限制 SID 列表（顺序即传入顺序）。

    ⚠️ 口径改动（2026-11-28，照 DSH / Codex 两个开源实现对齐）：
    限制 SID **不再是单个**，固定三项 `[logon SID, Everyone, 合成 SID]`。
    上一版只放合成 SID ⇒ 子进程启动阶段 0xC0000142（见 `logon_sid_text` 的说明）。

    自己写时对着这几条想：
      - `Flags` 用 `DISABLE_MAX_PRIVILEGE | WRITE_RESTRICTED`（后者 = 限制 SID 只管写）
      - 数组元素是 `SID_AND_ATTRIBUTES(psid, 0)`，`Attributes` 必须 0
      - 删特权 / 禁 SID 两处全给 0 / None（`DISABLE_MAX_PRIVILEGE` 会覆盖删特权列表）
      - 出参同样 `byref(HANDLE())`，失败走 `_check(...)`
    """
    entries = (SID_AND_ATTRIBUTES * len(sids))()
    for index, sid_ptr in enumerate(sids):
        entries[index].Sid = sid_ptr
        entries[index].Attributes = 0

    out = wintypes.HANDLE()
    _check(
        _advapi32.CreateRestrictedToken(
            base_token,
            DISABLE_MAX_PRIVILEGE | WRITE_RESTRICTED,
            0,  # DisableSidCount
            None,
            0,  # DeletePrivilegeCount
            None,
            len(sids),  # RestrictedSidCount
            entries,
            ctypes.byref(out),
        ),
        "CreateRestrictedToken",
    )
    return _handle_value(out)


def judge_token() -> bool:
    """块内自检 ①②，各打一行；红时把 WinError 原文打出来。**句柄必须全部释放。**

    流程：开进程令牌 → 取 logon SID → 转 3 个 SID → 造受限令牌 → 数限制 SID → 清理
    自己写时对着这几条想：
      - 清理：`CloseHandle(base)` / `CloseHandle(token)` / `LocalFree` ×3（三个 SID 都是
        `sid_from_string` 分配的，都要还）
      - 用 `try/finally` 还是"成功路径末尾统一释放"？先想清楚红的时候会不会漏
      - 自检 ② 用 `restricted_sid_count(token) == EXPECTED_RESTRICTING_SIDS`，
        **不要**用 IsTokenRestricted（实测它对本探针的令牌返回 FALSE，见原型区注释）
    """
    base = 0
    logon = 0
    everyone = 0
    sandbox = 0
    token = 0
    try:
        base = open_process_token()
        logon = sid_from_string(logon_sid_text(base))
        everyone = sid_from_string(EVERYONE_SID)
        sandbox = sid_from_string(SANDBOX_WRITE_SID)
        token = make_restricted_token(base, [logon, everyone, sandbox])

        count = restricted_sid_count(token)
        print(f"① logon SID   = {logon_sid_text(base)}")
        print(f"② 限制 SID 数 = {count}（期望 {EXPECTED_RESTRICTING_SIDS}）")
        return count == EXPECTED_RESTRICTING_SIDS

    except OSError as exc:
        print(f"红：{exc}")
        return False

    finally:
        if token:
            _kernel32.CloseHandle(token)
        if base:
            _kernel32.CloseHandle(base)
        if sandbox:
            _kernel32.LocalFree(sandbox)
        if everyone:
            _kernel32.LocalFree(everyone)
        if logon:
            _kernel32.LocalFree(logon)


def main() -> int:
    """胶水：跑块内自检，末行打印结论。"""
    ok = judge_token()
    print(f"token check: {'OK' if ok else 'NG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
