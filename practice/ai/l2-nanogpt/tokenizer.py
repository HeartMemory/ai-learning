"""L2 · 注意力 —— 关卡 5：手写 BPE（byte-level，>= 200 merge）

一条根：关 4 的模型吃的是「字符下标」（65 词表、一次一个字符）—— 那是为了把
注意力本身看清楚而做的简化。真模型吃的是 **token**：一个 token 可能是一个字节、
半个汉字、也可能是一整个单词。这一关就是把「文本 -> token -> 文本」这条最底层
的管道自己造一遍：

    byte-level BPE = 从 256 个字节出发，反复把「最常一起出现的相邻两个符号」
    合并成一个新符号；合并 200 次 => 词表 256 + 200 = 456 个 token。

关 4 的 loss 在猜「下一个字符」，关 5 之后才谈得上「猜下一个 token」；
而 token 数直接进账单（L4 已实测），所以这一关的第二个判据是**和 API 对账**。

口径（改必须写理由，并记进本目录 README）：
 ① **byte-level，不是 char-level**：起点 = `text.encode("utf-8")` 的 256 个字节值。
    理由：任何文本都可编码（没有 <unk> 可退回单字节）、中文 3 字节/字、emoji 4 字节/字；
    且服务商侧全是 byte-level BPE ⇒ 判据 ② 才有同口径可比。
 ② **词表 = 256 + merge 数**（线性），`vocab[id]` 存的是**字节串**而不是字符串。
    判据要求 >= 200 merge => `len(merges) >= 200`、`len(vocab) == 256 + len(merges)`。
 ③ **平票口径写死**：同一轮里多个相邻对频次并列最高时，取**扫描时最先出现**的那个
    （Python dict 插入顺序 + `max(stats, key=stats.get)`）。理由：BPE 只规定「最高频」，
    没规定平票；不定死则同语料两次训练可能给出不同 merges，判据 B 就无从写死期望值。
    ⚠️ 官方 minbpe README 的文字描述与它的代码在平票时不一致（它写第 2 次合并 `(a,b)`，
    代码实际给 `(256,97)`）—— 两者 **encode 结果相同**，所以判据只卡 encode 结果。
 ④ **encode 必须按 merge 学到的顺序（rank）逐条应用**，不是「对着待编码文本重新统计最高频」。
    理由：encode 时手上没有语料、只有 merges；训练与编码必须是同一套规则，否则往返不保。
    O(句子长度 x merge 数) 慢但正确，本关句子长度可控，不做优化。
 ⑤ **往返判据含非 ASCII**（英文语料同源句 + 中文 + emoji + 空串）。理由：纯 ASCII 往返
    只是在测 list 操作；中文/emoji 才能测到「多字节字符被拆成多个 token 后能否拼回」。
    ⚠️ `decode` 必须**先拼字节再 `bytes.decode("utf-8")`**，不能逐 id `chr()` 拼字符串 ——
    后者对中文必然红，且 `chr()` 出来的东西根本不再是原文。
 ⑥ **判据 ② 的对账口径（块 3 落地，先写死）**：chat 接口的 `prompt_tokens` 含模板 token
    （L4 实测过思考模式隐藏模板约 26 输入 token）=> 用**差分**消掉它：同一 messages 形态
    先发 1 个字符拿 `base`，再发目标句拿 `prompt_tokens`，API 侧文本 token 数 ≈ 差值。
    相对差 = `|mine - api| / api`。若 200 merge 下差 > 30%，允许把 merge 数往上调
    （判据原文是「>= 200」的下界，不是等号），并把「merge 数 -> 相对差」曲线报出来；
    **禁止回头改判据**。
 ⑦ **merges / vocab 不落盘**：本关只要进程内往返。要复现 = 固定语料切片 + 固定平票口径。
 ⑧ **训练用「小语料」**：判据原文就是小语料 => 取语料前 `TRAIN_CHARS` 个字符。
    全量 1.1 M 字符训 200 merge 要数分钟，收益只是 merges 更贴全分布，对三条判据无影响。

判据（唯一家 = 根目录 README.md 阶段表 L2 行）：
 ① `decode(encode(s)) == s`，10 个句子逐字相等
 ② 同一句话「我切的 token 数」vs API 报的 `prompt_tokens`，差 < 30%
 ③ 说得清「为什么 token 数不等于字数」

本文件是 ① 的落地拆法（不改判据，只拆块）：
 A 往返 10 句 ................ judge_roundtrip()   [块 2]
 B 官方参考值对账 ............ judge_official()    [块 2]  minbpe README 的 "aaabdaaabac"
 C 底层函数手算对账 .......... judge_primitives()  [块 1]  本文件先跑这一组
 ② ③ 落在 judge_api.py（块 3）与本目录 README —— 不进 main()，避免误触发计费。

════════════════════════════════════════════════════════════════════
 TODO 清单（一项 = 一个函数 = 一个整体，把 raise 换成你的实现）
════════════════════════════════════════════════════════════════════
 T1 [实现] to_bytes(text)             str -> list[int]（UTF-8，0..255）
 T2 [实现] from_bytes(ids)            list[int] -> str（T1 的逆）
 T3 [实现] get_stats(ids, counts)     相邻对频次
 T4 [实现] merge(ids, pair, new_id)   把某对替换成新 id（**不重叠、不就地改**）
 T5 [实现] build_vocab(merges)        {id: 字节串}，256 个单字节 + 每条 merge
 T6 [实现] train_bpe(text, n_merges)  -> (vocab, merges)
 T7 [实现] encode(text, merges, vocab) -> list[int]
 T8 [实现] decode(ids, vocab)         -> str
 T9 [已判据] judge_primitives / judge_official / judge_roundtrip（已写好，别改）

写完一块跑一块：
    uv run python tokenizer.py      # 三组判据按序跑，未实现的打 NG 不崩
════════════════════════════════════════════════════════════════════
"""

import sys

from corpus import load_corpus

BYTE_VOCAB: int = 256
"""起点词表：所有单字节值 0..255（口径 ①）。"""

TARGET_MERGES: int = 200
"""判据原文「>= 200 个 merge」。这是**下界**：块 3 对账不够时可以往上调（口径 ⑥）。"""

TRAIN_CHARS: int = 100_000
"""训练用的小语料长度（口径 ⑧）：语料前 10 万字符，200 merge 约 10-20 s。"""

DEMO_TEXT: str = "aaabdaaabac"
"""官方参考例（minbpe README）：11 字符 -> 5 token，3 次 merge。"""

DEMO_MERGES: int = 3
"""参考例只做 3 次 merge => 词表 259。"""

DEMO_IDS: list[int] = [258, 100, 258, 97, 99]
"""官方给出的 encode 结果（与平票口径无关，两条路径都得这个值）。"""

SENTENCES: list[str] = [
    "To be, or not to be, that is the question.",
    "shall I compare thee to a summer's day?",
    "ALL CAPS AND 123 NUMBERS!!!",
    "",
    "  leading and trailing spaces  ",
    "你好，世界。这是一句中文。",
    "混合 mixed 中英 文本 2026",
    "emoji: 😉🚀🎉",
    "标点：，。！？；：“”‘’（）【】",
    "制表符\t与换行\n都在里面",
]
"""判据 A 的 10 句：3 英文（语料同源）+ 空格/大写/数字 + 空串 + 中文 + 中英混排 + emoji。

空串与「只有空白的串」是两个独立边界：前者测 0 token，后者测空格不会被吃掉。
"""

SEQ_STATS: list[int] = [1, 1, 2, 1, 2, 2, 1]
EXPECT_STATS: dict[tuple[int, int], int] = {(1, 1): 1, (1, 2): 2, (2, 1): 2, (2, 2): 1}
"""判据 C2 的手算值：7 个元素 => 6 个相邻对。(1,2) 与 (2,1) 各 2 次，抓着「只数一次」。"""

SEED_COUNTS: dict[tuple[int, int], int] = {(9, 9): 5}
"""判据 C2 的累加语义：传入 counts 时在它上面继续加，不能清空。"""

SEQ_MERGE: list[int] = [1, 1, 2, 1, 2]
PAIR_AB: tuple[int, int] = (1, 2)
NEW_ID: int = 9
EXPECT_MERGE: list[int] = [1, 9, 9]
"""判据 C3 手算：[1,1,2,1,2] 里 (1,2) 出现 2 次且不重叠 => [1,9,9]，长度 5 -> 3。"""

SEQ_OVERLAP: list[int] = [1, 1, 1]
PAIR_AA: tuple[int, int] = (1, 1)
EXPECT_OVERLAP: list[int] = [9, 1]
"""[1,1,1] 里 (1,1) 只能吃前两个 => [9,1]。重叠实现会给 [9,9]，丢一个元素就变 [1,9]。"""

SEQ_NO_HIT: list[int] = [1, 3, 4]
"""判据 C3 的「不进来就原样」：不含 (1,2)（注意 3、4 也防着下标越界的实现）。"""


def to_bytes(text: str) -> list[int]:
    """T1 · 字符串 -> UTF-8 字节值列表（每个元素 0..255）。

    提示：`list(text.encode("utf-8"))`。`bytes` 迭代出来已经是 int，`list()` 就够。

    自检：`len(to_bytes("你好")) == 6`（UTF-8 一个汉字 3 字节）、
          `len(to_bytes("😉")) == 4`、`to_bytes("") == []`。
    """
    return list(text.encode("utf-8"))


def from_bytes(ids: list[int]) -> str:
    """T2 · T1 的逆：字节值列表 -> 字符串。

    提示：`bytes(ids).decode("utf-8")`。**别用 `chr()` 逐个拼**（第 ② 条口径）。
    ⚠️ 若 ids 里有超出 0..255 的值，`bytes()` 会抛 `ValueError` —— 这是想要的
       （说明 vocab 里混进了非法 token），不要 try/except 吞掉。

    自检：`from_bytes(to_bytes("你好")) == "你好"`、`from_bytes([]) == ""`。
    """
    return bytes(ids).decode("utf-8")


def get_stats(
    ids: list[int],
    counts: dict[tuple[int, int], int] | None = None,
) -> dict[tuple[int, int], int]:
    """T3 · 统计相邻对频次：返回 `{(左, 右): 次数}`。

    形状（自己翻译成代码）：
        for i in range(len(ids) - 1):
            pair = (ids[i], ids[i + 1])
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    三条必须注意：
      · `counts=None` 时新建空 dict；**传了就在它上面累加、返回同一个对象**（不要新建后丢掉）
      · `len(ids) <= 1` => 空 dict（`range(-1)` 正好是空的，别写 `range(len(ids))` 再手动截断）
      · 这是 BPE 唯一的「看全局」步骤，块 2 的 train_bpe 每轮都调它一次 =>
        它是整关最热的函数，但本关不做优化（正确优先）

    自检：`get_stats(SEQ_STATS) == EXPECT_STATS`、`get_stats([]) == {}`。
    """
    if counts is None:
        counts = {}
    for i in range(len(ids) - 1):
        pair = (ids[i], ids[i + 1])
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """T4 · 把序列里的 `pair` 替换成 `new_id`（**从左到右、不重叠**），返回**新列表**。

    形状（逐元素扫，别用 `str.replace` 那套思路）：
        i = 0
        新列表 = []
        while i < len(ids):
            若 (ids[i] == pair[0] 且 i+1 < len 且 ids[i+1] == pair[1]):
                追加 new_id; i += 2          ← 吃掉两个，这就是「不重叠」
            否则:
                追加 ids[i]; i += 1
        return 新列表

    三条必须注意：
      · **不就地改 ids**：块 2 的 train_bpe 会反复用它，就地改会把「上一轮序列」也改了，
        之后算出的频次全错（而且错得很像对的）
      · **不重叠**：`[1,1,1]` 合并 `(1,1)` 只能是 `[9,1]`，不是 `[9,9]`（长度守恒）
      · 不命中时返回等长的新列表（不是原对象）

    自检：`merge([1,1,2,1,2], (1,2), 9) == [1,9,9]`、`merge([1,1,1], (1,1), 9) == [9,1]`。
    """
    i = 0
    new_ids = []
    while i < len(ids):
        if (i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]):
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


def build_vocab(merges: list[tuple[int, int]]) -> dict[int, bytes]:
    """T5 · 由 merges 还原词表：`{id: 字节串}`。

    形状（三行）：
        vocab = {i: bytes([i]) for i in range(BYTE_VOCAB)}      ← 256 个单字节
        for k, (a, b) in enumerate(merges):
            vocab[BYTE_VOCAB + k] = vocab[a] + vocab[b]          ← 新 token = 两个旧 token 拼起来
        return vocab

    为什么存字节串而不是字符串：一个 token 可能是半个汉字（UTF-8 被切开），
    它自己 `decode("utf-8")` 会抛 `UnicodeDecodeError` —— 只有把 token 序列整体拼成
    字节串再解码才合法（口径 ⑤）。这也解释了 tiktoken 的 `mergeable_ranks` 是 `dict[bytes, int]`。

    自检：`len(build_vocab([])) == 256`；`len(build_vocab([(97, 97)])) == 257` 且
          `vocab[256] == b"aa"`。
    """
    vocab = {i: bytes([i]) for i in range(BYTE_VOCAB)}
    for k, (a, b) in enumerate(merges):
        vocab[BYTE_VOCAB + k] = vocab[a] + vocab[b]
    return vocab

def train_bpe(
    text: str,
    n_merges: int = TARGET_MERGES,
) -> tuple[dict[int, bytes], list[tuple[int, int]]]:
    """T6 · 训练：跑 `n_merges` 轮「统计 -> 合并 -> 记账」，返回 `(vocab, merges)`。

    形状（一个循环）：
        ids = to_bytes(text)
        merges = []
        for k in range(n_merges):
            stats = get_stats(ids)
            if not stats:                      ← 语料太短/已被合并到底，提前收工
                break
            pair = max(stats, key=stats.get)   ← 平票取最先出现（口径 ③）
            ids = merge(ids, pair, BYTE_VOCAB + k)
            merges.append(pair)
        return build_vocab(merges), merges

    三条必须注意：
      · 新 id **必须**是 `BYTE_VOCAB + k`（k 是第几轮，从 0 起）—— 这条和 build_vocab 的
        `enumerate(merges)` 是同一个约定，两边错一个就整体错位
      · `max(stats, key=stats.get)` 返回的是 **key**（也就是 pair），不是次数
      · merges 的**顺序就是 encode 时的优先级**（口径 ④），不能排序、不能去重

    自检：`train_bpe("aaabdaaabac", 3)` => `len(merges) == 3`、`len(vocab) == 259`；
          块 2 的判据 B 会拿 encode 结果去和官方值对账。
    """
    ids = to_bytes(text)
    merges = []
    for k in range(n_merges):
        stats = get_stats(ids)
        if not stats:
            break
        pair = max(stats, key = lambda p: stats[p])
        ids = merge(ids, pair, BYTE_VOCAB + k)
        merges.append(pair)
    return build_vocab(merges), merges


def encode(text: str, merges: list[tuple[int, int]], vocab: dict[int, bytes]) -> list[int]:
    """T7 · 编码：文本 -> token id 列表。

    形状（三行）：
        ids = to_bytes(text)
        for k, pair in enumerate(merges):
            ids = merge(ids, pair, BYTE_VOCAB + k)
        return ids

    为什么这样写（口径 ④）：merges 是**有序**的，越早学到的越优先；逐条扫一遍就等价于
    tiktoken 的「每轮取全局最小 rank 那一对合并」。**不要**改成「对着这段文本重新统计最高频」
    —— 那样每次 encode 结果都可能不同，往返也就无从保证。

    O(句子长度 x merge 数)：10 万字符的句子 x 200 merge 会明显慢，本关句子都在千字符内。
    ⚠️ 这里没用 vocab 参数 —— 签名留着是为了和 decode 成对、以及将来加「未登录 id 检查」。

    自检：`encode("", ...) == []`；编码结果的每个 id 都 < `256 + len(merges)`。
    """
    ids = to_bytes(text)
    for k, pair in enumerate(merges):
        ids = merge(ids, pair, BYTE_VOCAB + k)
    return ids


def decode(ids: list[int], vocab: dict[int, bytes]) -> str:
    """T8 · 解码：token id 列表 -> 文本。

    形状（两行）：
        raw = b"".join(vocab[i] for i in ids)
        return raw.decode("utf-8")

    为什么必须这样（口径 ⑤）：单个 token 可能是半个汉字，逐个 `decode` 会抛
    `UnicodeDecodeError`；只有**整体拼成字节串**再解码才对。
    空列表 => `b"".join([])` 是 `b""` => `""`。

    自检：`decode([], vocab) == ""`。
    """
    raw = b"".join(vocab[i] for i in ids)
    return raw.decode("utf-8")


def judge_primitives() -> bool:
    """判据 C · 底层函数的**手算**对账（块 1 的完成标志）。

    为什么先卡这一层：块 2 的官方对账只给一个「最终答案」，错了不知道错在哪个函数；
    先把 get_stats / merge 用 3-7 个元素的小序列钉死，错就错在当行。
    """
    checks: list[bool] = []

    # C1 字节层：中文 3 字节/字、emoji 4 字节、空串 0 字节
    try:
        raw = to_bytes("你好")
        emoji = to_bytes("😉")
        empty = to_bytes("")
    except NotImplementedError:
        print("C1 字节层：to_bytes 未实现  NG")
        return False
    ok_len = len(raw) == 6 and len(emoji) == 4 and empty == []
    ok_range = all(0 <= b < BYTE_VOCAB for b in [*raw, *emoji])
    try:
        ok_back = from_bytes(raw) == "你好" and from_bytes(emoji) == "😉" and from_bytes([]) == ""
    except NotImplementedError:
        print("C1 字节层：from_bytes 未实现  NG")
        return False
    checks += [ok_len, ok_range, ok_back]
    tag_c1 = "OK" if (ok_len and ok_range and ok_back) else "NG"
    print(
        _show(
            f"C1 字节层 你好 {len(raw)}B / 😉 {len(emoji)}B / 空串 {len(empty)}B"
            f" · 值域 0..255 {ok_range} · 往返 {ok_back}  {tag_c1}"
        )
    )

    # C2 相邻对频次：手算值 + 累加语义 + 边界
    try:
        stats = get_stats(list(SEQ_STATS))
        with_base = get_stats(list(SEQ_STATS), counts=dict(SEED_COUNTS))
        edge = get_stats([])
        single = get_stats([7])
    except NotImplementedError:
        print("C2 相邻对频次：get_stats 未实现  NG")
        return False
    ok_stats = stats == EXPECT_STATS
    ok_acc = with_base.get((9, 9)) == 5 and with_base.get((1, 2)) == 2
    ok_edge = edge == {} and single == {}
    checks += [ok_stats, ok_acc, ok_edge]
    tag_c2 = "OK" if (ok_stats and ok_acc and ok_edge) else "NG"
    print(
        f"C2 相邻对 {sorted(stats.items())}（期望 {sorted(EXPECT_STATS.items())}）"
        f" · 累加语义 {ok_acc} · 空/单元素 {ok_edge}  {tag_c2}"
    )

    # C3 merge：手算值 + 不重叠 + 不就地改 + 不命中原样
    src = list(SEQ_MERGE)
    try:
        merged = merge(src, PAIR_AB, NEW_ID)
        overlap = merge(list(SEQ_OVERLAP), PAIR_AA, NEW_ID)
        no_hit = merge(list(SEQ_NO_HIT), PAIR_AB, NEW_ID)
    except NotImplementedError:
        print("C3 merge：merge 未实现  NG")
        return False
    ok_val = merged == EXPECT_MERGE and overlap == EXPECT_OVERLAP
    ok_pure = src == SEQ_MERGE and no_hit == SEQ_NO_HIT
    ok_keep = sorted(_expand(merged, NEW_ID, PAIR_AB)) == sorted(SEQ_MERGE)
    checks += [ok_val, ok_pure, ok_keep]
    tag_c3 = "OK" if (ok_val and ok_pure and ok_keep) else "NG"
    print(
        f"C3 merge {SEQ_MERGE} + {PAIR_AB}->{NEW_ID} = {merged}（期望 {EXPECT_MERGE}）"
        f" · 不重叠 {overlap == EXPECT_OVERLAP} · 不就地改 {src == SEQ_MERGE}"
        f" · 元素守恒 {ok_keep}  {tag_c3}"
    )

    return all(checks)


def judge_official() -> bool:
    """判据 B · 官方参考值对账（块 2）。

    minbpe README 的 `"aaabdaaabac"` + 3 次 merge => encode 得 `[258, 100, 258, 97, 99]`。
    这是**独立的第三方期望值**（不是我自己手算），因此能抓「整体口径错但小算例对」
    的那类 bug —— 比如新 id 起点写成 0、merge 顺序被排序、encode 重新统计最高频。
    """
    try:
        vocab, merges = train_bpe(DEMO_TEXT, DEMO_MERGES)
    except NotImplementedError:
        print("B 官方对账：train_bpe 未实现  NG")
        return False
    checks: list[bool] = []
    ok_count = len(merges) == DEMO_MERGES and len(vocab) == BYTE_VOCAB + DEMO_MERGES
    ok_legal = all(
        a in vocab and b in vocab and a < BYTE_VOCAB + k and b < BYTE_VOCAB + k
        for k, (a, b) in enumerate(merges)
    )
    checks += [ok_count, ok_legal]
    try:
        got = encode(DEMO_TEXT, merges, vocab)
        back = decode(got, vocab)
    except NotImplementedError:
        print("B 官方对账：encode / decode 未实现  NG")
        return False
    ok_ids = got == DEMO_IDS
    ok_rt = back == DEMO_TEXT
    checks += [ok_ids, ok_rt]
    tag = "OK" if all(checks) else "NG"
    print(
        f"B 官方对账 merge {len(merges)} 条 / 词表 {len(vocab)} · 引用合法 {ok_legal}"
        f" · encode {got}（期望 {DEMO_IDS}）· 往返 {ok_rt}  {tag}"
    )
    return all(checks)


def judge_roundtrip() -> bool:
    """判据 A · 文本层往返（块 2）：10 句逐字相等 + 报压缩率。

    压缩率（字符/token）是判据 ③ 要用的数字，顺手在这里打出来：
    中文会明显低于英文（汉字 3 字节，词表里还几乎没有中文 merge）。
    """
    text = load_corpus()[:TRAIN_CHARS]
    try:
        vocab, merges = train_bpe(text, TARGET_MERGES)
    except NotImplementedError:
        print("A 往返：train_bpe 未实现  NG")
        return False
    if len(merges) < TARGET_MERGES:
        print(f"A 往返：只训出 {len(merges)} 条 merge（< {TARGET_MERGES}）  NG")
        return False

    bad: list[int] = []
    rows: list[str] = []
    total_chars = 0
    total_tokens = 0
    for i, s in enumerate(SENTENCES):
        try:
            ids = encode(s, merges, vocab)
            back = decode(ids, vocab)
        except NotImplementedError:
            print("A 往返：encode / decode 未实现  NG")
            return False
        total_chars += len(s)
        total_tokens += len(ids)
        if back != s:
            bad.append(i)
        ratio = len(s) / len(ids) if ids else 0.0
        rows.append(f"  #{i} {len(s)} 字符 -> {len(ids)} token（{ratio:.2f} 字符/token）")

    ok_rt = not bad
    tag = "OK" if ok_rt else "NG"
    print(f"A 往返 {len(SENTENCES) - len(bad)}/{len(SENTENCES)} 句逐字相等  {tag}")
    for r in rows:
        print(r)
    if bad:
        for i in bad:
            got_back = decode(encode(SENTENCES[i], merges, vocab), vocab)
            print(_show(f"   第 {i} 句不等：{SENTENCES[i]!r} -> {got_back!r}"))
    overall = total_chars / total_tokens if total_tokens else 0.0
    print(f"  合计 {total_chars} 字符 -> {total_tokens} token（{overall:.2f} 字符/token）")
    return ok_rt


def _show(text: str) -> str:
    """打印用：把当前控制台编码不了的字符转成 `\\U0001f609` 这类转义。

    中文 Windows 控制台是 GBK：emoji（U+1F609）和 `✓/✗`（U+2713）都编不出来，
    `print` 会抛 `UnicodeEncodeError: 'gbk' codec can't encode character`。
    本目录已经记过三次同族坑（`✓/✗`、`−`、`1024³`）—— 所以判据里凡是要打印
    **数据原文**（句子、emoji）的地方都过这一层，标记本身仍然只用 ASCII 的 OK/NG。
    """
    enc = sys.stdout.encoding or "utf-8"
    return text.encode(enc, "backslashreplace").decode(enc, "replace")


def _expand(ids: list[int], new_id: int, pair: tuple[int, int]) -> list[int]:
    """判据辅助：把 new_id 就地展开回 pair，用来验「merge 只换了写法、没丢元素」。"""
    out: list[int] = []
    for v in ids:
        if v == new_id:
            out.extend(pair)
        else:
            out.append(v)
    return out


def main() -> None:
    """测试入口（纯胶水）：三组判据按序跑 + 末行汇总。"""
    results = [judge_primitives(), judge_official(), judge_roundtrip()]
    print(f"判据 {sum(results)}/{len(results)} 绿")


if __name__ == "__main__":
    main()
