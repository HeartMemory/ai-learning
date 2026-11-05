"""S0 · 关卡 2：pathlib —— 路径是对象，不是字符串

目的（一条根）：`Path` 把"路径"变成一个**对象**，于是拼接、判存在、读写都成了方法调用，
    不再靠 `"a" + "/" + "b"` 这种手工拼串（在 Windows 上必然踩反斜杠）。`/` 运算符与
    `joinpath()` 完全等价，`read_text` / `write_text` 是读写正文的最短路径。

════════════════════════════════════════════════════════════════════════
 TODO 清单：**一项 = 一个函数 = 一个整体**
════════════════════════════════════════════════════════════════════════
 T5  [实现] roundtrip(root, text)   拼路径 → 建目录 → 写 → 读回 → 返回 (路径, 内容)
 T6  [判据] judge_pathlib()         验内容一致、路径形状、`/` 的拼接语义

跑法：uv run python paths.py
文档依据：docs.python.org/3/library/pathlib.html（`/` 运算符等价于 joinpath；参数是绝对路径时
        前面整段被忽略；`read_text` / `write_text` 的用法与返回值；`exists()` 的 True/False 语义）
════════════════════════════════════════════════════════════════════════
"""

from pathlib import Path


def roundtrip(root: Path, text: str) -> tuple[Path, str]:
    """T5 · 在 `root` 下造 `notes/day.txt`，写入 `text`，再把内容读回来。

    要求（四步，顺序别换）：
      ① 目标路径用 `/` 拼出来：`root / "notes" / "day.txt"`（**不许**用字符串拼接 + `open`）
      ② `notes` 目录可能不存在 ⇒ 先建（`mkdir` 的 `parents` / `exist_ok` 参数要看一眼）
      ③ `write_text(text, encoding="utf-8")` —— 中文必须显式给编码，否则平台默认编码会咬人
      ④ `read_text(encoding="utf-8")` 读回，返回 `(目标路径, 读回的内容)`

    答"为什么要 encoding"：Windows 上默认编码不一定是 utf-8，同一个文件在别人机器上会乱码或报错。
    """
    path = root / "notes" / "day.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    content = path.read_text(encoding="utf-8")
    return path, content


def judge_pathlib() -> bool:
    """T6 · 判据：四条（写成 assert，全过 return True）。

    先造一个用完自动删除的临时根目录：
        `with tempfile.TemporaryDirectory() as tmp:` → `root = Path(tmp)`
    （`tempfile` 是标准库，判据自己负责造和清理，不要写到仓库目录里）

    ① 读回内容严格等于写进去的（用带中文的字符串，才能证明 encoding 那一步没白写）
    ② 返回的路径就是 `root / "notes" / "day.txt"`
    ③ `path.parent.name == "notes"` 且 `path.exists() is True`
    ④ `/` 的拼接语义：`(root / "a" / "b").parent.name == "a"`
       （顺带观察：`root / "/abs"` 会**忽略前面整段** —— 官方文档那条"绝对路径重置"，
        这一条只观察、不进判据）

    提示：`roundtrip` 写在哪个模块就 import 哪个 —— 本函数和它同文件，直接调即可。
    """
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        text = "中文测试"
        path, content = roundtrip(root, text)

        assert content == text
        assert path == root / "notes" / "day.txt"
        assert path.parent.name == "notes" and path.exists() is True
        assert (root / "a" / "b").parent.name == "a"
        assert (root / "/abs").name == "abs"
        assert path.read_bytes() == text.encode("utf-8")

    return True


def main() -> None:
    """测试入口（纯胶水）：末行计数是 sum(results) **真算**的。"""
    results = [judge_pathlib()]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
