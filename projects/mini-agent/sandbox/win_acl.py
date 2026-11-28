"""L5 关 0 块 3a：给沙箱目录挂一条「允许合成 SID 写」的 ACE。

为什么不用 icacls（2026-11-28 本机实测）：
  `icacls dir /grant "*SID:(OI)(CI)(M)"` 要求 SID **能映射到账户** ——
  `*S-1-5-32-544`、`*S-1-1-0` 成功，但 `*S-1-5-21-0-0-0-1000` 报
  `ERROR_NONE_MAPPED`（帐户名与安全标识间无任何映射完成）。
  ⇒ 合成 SID 只能用 API：`SetEntriesInAclW` 直接吃 PSID，不查账户表。

本块自检（**不是**关 0 判据编号；判据 A–E 的唯一家仍是 `README.md` 阶段表）：
  ① 三个 API 全部返回 0（注意：这套 API **返回错误码，0 = 成功**，跟 BOOL 那套正好相反）
  ② `icacls <目录>` 读回能看到 `S-1-5-21-0-0-0-1000:(OI)(CI)(R,W)` 且**不带 (I)**
  ③ 原有 ACE（SYSTEM / Administrators / 本人）**一条都没丢**
     ⇒ 所以必须把「读出来的旧 DACL」当 OldAcl 传进去，不能直接新建空 DACL

边界：本块只动**一个目录**的 DACL，不碰所有权、不碰 SACL（审计）。

跑法：`uv run python sandbox/win_acl.py`
"""

import ctypes
import sys
from ctypes import wintypes
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SANDBOX_ROOT = SCRIPT_DIR / ".sandbox-root"

# --- 常量 ---
SE_FILE_OBJECT = 1
DACL_SECURITY_INFORMATION = 0x0004
GRANT_ACCESS = 1
TRUSTEE_IS_SID = 0
TRUSTEE_IS_UNKNOWN = 0
OBJECT_INHERIT_ACE = 0x1  # 子文件继承
CONTAINER_INHERIT_ACE = 0x2  # 子目录继承
ACE_INHERIT_BOTH = OBJECT_INHERIT_ACE | CONTAINER_INHERIT_ACE

# FILE_GENERIC_READ(0x120089) | FILE_GENERIC_WRITE(0x120116)
# 对目录来说 FILE_WRITE_DATA 就等于“可以在这里建文件”
ACCESS_READ_WRITE = 0x12019F


class TRUSTEE_W(ctypes.Structure):
    """「被授权人」描述。当 `TrusteeForm = TRUSTEE_IS_SID` 时，`ptstrName` 里放的是 PSID。

    名字看着像字符串指针（LPWSTR），实际按 form 决定装什么 —— 所以这里声明成 c_void_p。
    """


TRUSTEE_W._fields_ = [
    ("pMultipleTrustee", ctypes.POINTER(TRUSTEE_W)),
    ("MultipleTrusteeOperation", ctypes.c_int),
    ("TrusteeForm", ctypes.c_int),
    ("TrusteeType", ctypes.c_int),
    ("ptstrName", ctypes.c_void_p),
]


class EXPLICIT_ACCESS_W(ctypes.Structure):
    """一条要加进 DACL 的 ACE 描述（权限 / 允许还是拒绝 / 继承方式 / 给谁）。"""

    _fields_ = [
        ("grfAccessPermissions", wintypes.DWORD),
        ("grfAccessMode", ctypes.c_int),
        ("grfInheritance", wintypes.DWORD),
        ("Trustee", TRUSTEE_W),
    ]


_advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

_kernel32.LocalFree.argtypes = [wintypes.HLOCAL]
_kernel32.LocalFree.restype = wintypes.HLOCAL

_advapi32.ConvertStringSidToSidW.argtypes = [wintypes.LPCWSTR, ctypes.POINTER(wintypes.HANDLE)]
_advapi32.ConvertStringSidToSidW.restype = wintypes.BOOL

# 注意这三个的返回类型是 DWORD（错误码），不是 BOOL
_advapi32.GetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,  # pObjectName
    ctypes.c_int,  # ObjectType
    wintypes.DWORD,  # SecurityInfo
    ctypes.POINTER(ctypes.c_void_p),  # ppsidOwner
    ctypes.POINTER(ctypes.c_void_p),  # ppsidGroup
    ctypes.POINTER(ctypes.c_void_p),  # ppDacl
    ctypes.POINTER(ctypes.c_void_p),  # ppSacl
    ctypes.POINTER(ctypes.c_void_p),  # ppSecurityDescriptor
]
_advapi32.GetNamedSecurityInfoW.restype = wintypes.DWORD

_advapi32.SetEntriesInAclW.argtypes = [
    wintypes.ULONG,  # cCountOfExplicitEntries
    ctypes.POINTER(EXPLICIT_ACCESS_W),  # pListOfExplicitEntries
    ctypes.c_void_p,  # OldAcl
    ctypes.POINTER(ctypes.c_void_p),  # NewAcl
]
_advapi32.SetEntriesInAclW.restype = wintypes.DWORD

_advapi32.SetNamedSecurityInfoW.argtypes = [
    wintypes.LPWSTR,  # pObjectName
    ctypes.c_int,  # ObjectType
    wintypes.DWORD,  # SecurityInfo
    ctypes.c_void_p,  # psidOwner
    ctypes.c_void_p,  # psidGroup
    ctypes.c_void_p,  # pDacl
    ctypes.c_void_p,  # pSacl
]
_advapi32.SetNamedSecurityInfoW.restype = wintypes.DWORD

# --- 令牌默认 DACL 相关（放本文件是因为要用 EXPLICIT_ACCESS_W，避免与 win_token 循环 import）---
TOKEN_DEFAULT_DACL = 6  # TOKEN_INFORMATION_CLASS
ACCESS_ALL = 0x0F01FF  # 全权，够 conhost / 命名管道这类「子进程自己新建的对象」用

_advapi32.GetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
    ctypes.POINTER(wintypes.DWORD),
]
_advapi32.GetTokenInformation.restype = wintypes.BOOL

_advapi32.SetTokenInformation.argtypes = [
    wintypes.HANDLE,
    ctypes.c_int,
    ctypes.c_void_p,
    wintypes.DWORD,
]
_advapi32.SetTokenInformation.restype = wintypes.BOOL


class TOKEN_DEFAULT_DACL_ENTRY(ctypes.Structure):
    """令牌的「默认 DACL」：子进程新建对象时自动套用这张名单。"""

    _fields_ = [("DefaultDacl", ctypes.c_void_p)]


# 复用块 2 的合成 SID 与检查函数，避免两处写死不一致
from win_token import SANDBOX_WRITE_SID, _check, sid_from_string  # noqa: E402


def _check_rc(rc: int, what: str) -> None:
    """fail-closed：这套 API **返回 0 才算成功**，非 0 就是 WinError 码本身。"""
    if rc == 0:
        return
    raise ctypes.WinError(rc, f"{what} 失败")


def grant_write(target: Path, sid: int) -> None:
    """给 target 的 DACL 追加一条「允许 sid 读+写（含继承）」的 ACE。

    自己写时对着这几条想：
      - 三步不是两步：**先读出旧 DACL**，再
        `SetEntriesInAclW(1, byref(ea), 旧DACL, byref(新DACL))`，
        最后 `SetNamedSecurityInfoW(...)` 写回。少了第一步，原有的 SYSTEM/本人 那几条 ACE 全没
      - `GetNamedSecurityInfoW` 有 **8 个参数**，其中前 4 个出参随便给个 `byref(c_void_p())` 占位，
        但 **旧 DACL** 和 **安全描述符** 两个一定要留住（后面要用 / 要释放）
      - `EXPLICIT_ACCESS_W` 四个字段：权限位 / `GRANT_ACCESS` / `ACE_INHERIT_BOTH` /
        Trustee(Form=SID, Type=UNKNOWN, ptstrName=sid)
      - 释放：**新 DACL**（`LocalFree`）+ **安全描述符**（`LocalFree`）；sid 由调用方释放
      - 每一步都 `_check_rc(...)`，别用 `_check`（那个是 BOOL 那套，判断相反）

    ⚠️ 降级记录：本函数体由 AI 代写（2026-11-28，本人「没有思路」时给出示范）。
       理由 = 纯 ctypes 样板（按文档摆 8 个参数 + 两处释放），不是本关的学习点；
       判据未放松，读取回证据仍靠 `icacls` 外部工具核对。
    """
    # --- 第 1 步：把旧 DACL 抄下来（不抄，写回时旧 ACE 全没）---
    owner = ctypes.c_void_p()
    group = ctypes.c_void_p()
    old_dacl = ctypes.c_void_p()
    sacl = ctypes.c_void_p()
    sd = ctypes.c_void_p()
    _check_rc(
        _advapi32.GetNamedSecurityInfoW(
            str(target),
            SE_FILE_OBJECT,
            DACL_SECURITY_INFORMATION,
            ctypes.byref(owner),
            ctypes.byref(group),
            ctypes.byref(old_dacl),
            ctypes.byref(sacl),
            ctypes.byref(sd),
        ),
        "GetNamedSecurityInfoW",
    )

    # --- 第 2 步：旧 DACL + 我们这一条 → 新 DACL ---
    ea = EXPLICIT_ACCESS_W()
    ea.grfAccessPermissions = ACCESS_READ_WRITE
    ea.grfAccessMode = GRANT_ACCESS
    ea.grfInheritance = ACE_INHERIT_BOTH
    ea.Trustee.TrusteeForm = TRUSTEE_IS_SID
    ea.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN
    ea.Trustee.ptstrName = sid  # TrusteeForm=SID 时，这个字段装的是 PSID

    new_dacl = ctypes.c_void_p()
    try:
        _check_rc(
            _advapi32.SetEntriesInAclW(1, ctypes.byref(ea), old_dacl, ctypes.byref(new_dacl)),
            "SetEntriesInAclW",
        )

        # --- 第 3 步：把新 DACL 贴回目录 ---
        _check_rc(
            _advapi32.SetNamedSecurityInfoW(
                str(target),
                SE_FILE_OBJECT,
                DACL_SECURITY_INFORMATION,
                None,
                None,
                new_dacl,
                None,
            ),
            "SetNamedSecurityInfoW",
        )
    finally:
        # 新 DACL 与安全描述符都是系统分配的，必须还回去
        if new_dacl:
            _kernel32.LocalFree(new_dacl)
        if sd:
            _kernel32.LocalFree(sd)
    


def set_default_dacl(token: int, sid: int) -> None:
    """给受限令牌的**默认 DACL** 追加一条「允许 sid 全权」。

    为什么需要（2026-11-28 补，照 DSH / Codex 对齐）：子进程启动时会**新建对象**
    （控制台、命名管道等），这些对象自动套用令牌的默认 DACL。默认 DACL 里没有
    限制 SID ⇒ 第二遍访问检查拒掉子进程**自己刚建的对象** ⇒ 0xC0000142。

    流程：读 TokenDefaultDacl → SetEntriesInAclW 并入一条 → SetTokenInformation 写回。

    AI 代写（纯样板；与 grant_write 同形，只是目标从「目录」换成「令牌」）。
    注意两套返回值约定同时出现在一个函数里：
      `GetTokenInformation` / `SetTokenInformation` 返回 **BOOL**（非 0 成功）→ `_check`
      `SetEntriesInAclW` 返回 **错误码**（0 成功）→ `_check_rc`
    """
    need = wintypes.DWORD(0)
    _advapi32.GetTokenInformation(token, TOKEN_DEFAULT_DACL, None, 0, ctypes.byref(need))
    buffer = ctypes.create_string_buffer(need.value)
    _check(
        _advapi32.GetTokenInformation(
            token, TOKEN_DEFAULT_DACL, buffer, need.value, ctypes.byref(need)
        ),
        "GetTokenInformation(TokenDefaultDacl)",
    )
    holder = ctypes.cast(buffer, ctypes.POINTER(TOKEN_DEFAULT_DACL_ENTRY)).contents

    ea = EXPLICIT_ACCESS_W()
    ea.grfAccessPermissions = ACCESS_ALL
    ea.grfAccessMode = GRANT_ACCESS
    ea.grfInheritance = 0  # 默认 DACL 用不到继承
    ea.Trustee.TrusteeForm = TRUSTEE_IS_SID
    ea.Trustee.TrusteeType = TRUSTEE_IS_UNKNOWN
    ea.Trustee.ptstrName = sid

    new_dacl = ctypes.c_void_p()
    _check_rc(
        _advapi32.SetEntriesInAclW(1, ctypes.byref(ea), holder.DefaultDacl, ctypes.byref(new_dacl)),
        "SetEntriesInAclW(TokenDefaultDacl)",
    )
    try:
        holder.DefaultDacl = new_dacl
        _check(
            _advapi32.SetTokenInformation(
                token, TOKEN_DEFAULT_DACL, ctypes.byref(holder), ctypes.sizeof(holder)
            ),
            "SetTokenInformation(TokenDefaultDacl)",
        )
    finally:
        if new_dacl:
            _kernel32.LocalFree(new_dacl)


def judge_acl() -> bool:
    """块内自检：建目录 → 造 SID → 挂 ACE → 打印证据（`icacls` 读回留给命令行看）。

    自己写时对着这几条想：
      - 目录用 `SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)`（可重复跑）
      - SID 用 `win_token.sid_from_string(SANDBOX_WRITE_SID)`
      - 打印：目录路径 + SID + "ACE 已写入"，然后 return True
      - 失败时接 `OSError`，打原文，返回 False；sid 用 `finally` 释放
    """
    SANDBOX_ROOT.mkdir(parents=True, exist_ok=True)
    sid = 0
    try:
        sid = sid_from_string(SANDBOX_WRITE_SID)
        grant_write(SANDBOX_ROOT, sid)
        print(f"目录     = {SANDBOX_ROOT}")
        print(f"合成 SID = {SANDBOX_WRITE_SID}")
        print("ACE 已写入 —— 用 icacls 读回核对（不带 (I) 才是我们加的）")
        return True
    except OSError as exc:
        print(f"红：{exc}")
        return False
    finally:
        if sid:
            _kernel32.LocalFree(sid)


def main() -> int:
    """胶水：跑块内自检，末行打印结论。"""
    ok = judge_acl()
    print(f"acl check: {'OK' if ok else 'NG'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
