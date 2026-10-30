# -*- coding: utf-8 -*-
"""
从打包好的剧情文本里抽出「阿米娅」的全部台词，并做数据勘察。

输入：raw_all\*.txt   （官方解析格式，形如「说话人：内容」）
输出：out\amiya_lines.jsonl、out\speakers.json

用法：python scripts\02_extract_amiya.py
"""
import json
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "raw_all")
OUT = os.path.join(ROOT, "out")
os.makedirs(OUT, exist_ok=True)

TARGET = "阿米娅"

# 「杜宾：可恶......」→ 说话人 + 台词
LINE_RE = re.compile(r"^([^\s：]{1,24})：(.+)$")
MONO_RE = re.compile(r"^【(.+?)·独白】$")
TAG_RE = re.compile(r"^【(旁白|选项|贴图)】")


def parse_file(path):
    """返回该文件里的 [(kind, speaker, text, prev_speaker, prev_text)]"""
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()
    out = []
    prev_s, prev_t = "", ""
    mono_speaker = None
    mono_buf = []

    def flush_mono():
        nonlocal mono_buf, mono_speaker
        if mono_speaker and mono_buf:
            text = "".join(mono_buf).strip()
            if text:
                out.append(("monologue", mono_speaker, text, prev_s, prev_t))
        mono_buf = []
        mono_speaker = None

    for line in raw.splitlines():
        s = line.strip()
        if not s:
            flush_mono()
            continue
        m = MONO_RE.match(s)
        if m:
            flush_mono()
            mono_speaker = m.group(1).strip()
            continue
        if mono_speaker:
            if TAG_RE.match(s):
                flush_mono()
            else:
                mono_buf.append(s)
                continue
        mm = LINE_RE.match(s)
        if mm:
            speaker, text = mm.group(1).strip(), mm.group(2).strip()
            out.append(("dialog", speaker, text, prev_s, prev_t))
            prev_s, prev_t = speaker, text
    flush_mono()
    return out


def main():
    if not os.path.isdir(RAW):
        sys.exit(f"找不到目录：{RAW}")

    files = sorted(f for f in os.listdir(RAW) if f.endswith(".txt"))
    print(f"扫描 {len(files)} 个剧情文本文件……")

    rows = []
    speakers = Counter()
    naked = 0          # 没有归属说话人的行（旁白等）
    for name in files:
        for kind, spk, text, ps, pt in parse_file(os.path.join(RAW, name)):
            speakers[spk] += 1
            rows.append({"file": name, "kind": kind, "speaker": spk,
                         "text": text, "prev_speaker": ps, "prev_text": pt})

    print(f"共解析 {len(rows)} 条台词/独白，涉及 {len(speakers)} 个说话人")

    # 阿米娅的所有变体
    variants = {s: c for s, c in speakers.items() if TARGET in s}
    print(f"\n名字里含「{TARGET}」的写法：")
    for s, c in sorted(variants.items(), key=lambda x: -x[1]):
        print(f"  {s:<18} {c:>5} 条")

    mine = [r for r in rows if r["speaker"] in variants]
    print(f"\n「{TARGET}」合计 {len(mine)} 条")

    # ── 数据勘察 ──────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  数据勘察")
    print("=" * 66)

    kind_cnt = Counter(r["kind"] for r in mine)
    lens = sorted(len(r["text"]) for r in mine)
    n = len(lens)
    print(f"  对白      : {kind_cnt.get('dialog', 0)}")
    print(f"  独白      : {kind_cnt.get('monologue', 0)}")
    if n:
        print(f"  平均字数  : {sum(lens) / n:.1f}")
        print(f"  中位字数  : {lens[n // 2]}")
        print(f"  最短/最长 : {lens[0]} / {lens[-1]}")

    seen = {}
    for r in mine:
        seen.setdefault(r["text"].strip(), r)
    print(f"  完全重复  : {len(mine) - len(seen)} 条")
    uniq = list(seen.values())
    print(f"  去重后    : {len(uniq)} 条")

    print("\n  ── 长度分布 ──")
    buckets = [(0, 6), (6, 12), (12, 20), (20, 35), (35, 60), (60, 100), (100, 10 ** 9)]
    for lo, hi in buckets:
        c = sum(1 for t in lens if lo <= t < hi)
        bar = "█" * round(c / max(n, 1) * 44)
        lab = f"{lo}-{hi if hi < 10**9 else '∞'}"
        print(f"  {lab:>7} 字 {c:>5}  {bar}")

    # 脏数据
    dirty = Counter()
    examples = {}
    for r in mine:
        t = r["text"]
        checks = []
        if re.search(r"[\[\]{}<>]", t):
            checks.append("含括号/占位符")
        if "$" in t or "&amp;" in t or "\\n" in t:
            checks.append("转义残留")
        if len(t) < 4:
            checks.append("过短")
        if re.search(r"(\.\.\.|……)$", t):
            checks.append("以省略号结尾(可能是截断)")
        if "（）" in t or re.search(r"^（.*）$", t):
            checks.append("纯括号旁白")
        for c in checks:
            dirty[c] += 1
            examples.setdefault(c, t)
    print("\n  ── 脏数据排查 ──")
    if not dirty:
        print("  未发现明显问题")
    for k, c in dirty.most_common():
        print(f"  {k:<22} {c:>4} 条   例: {examples[k][:36]!r}")

    # 覆盖
    per_file = Counter(r["file"] for r in mine)
    print(f"\n  分布在 {len(per_file)} 个剧情文件里（平均每文件 {n / max(len(per_file), 1):.1f} 条）")
    print("  ── 阿米娅台词最多的 8 个文件 ──")
    for f, c in per_file.most_common(8):
        print(f"  {f:<34} {c:>4}")

    # 落盘
    with open(os.path.join(OUT, "amiya_lines.jsonl"), "w", encoding="utf-8") as f:
        for r in mine:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "speakers.json"), "w", encoding="utf-8") as f:
        json.dump(speakers.most_common(400), f, ensure_ascii=False, indent=1)

    print(f"\n  已写出 out\\amiya_lines.jsonl（{len(mine)} 条）")

    print("\n  ── 说话最多的 10 个角色（验证解析是否正确）──")
    for s, c in speakers.most_common(10):
        print(f"  {s:<12} {c:>6}")

    # 抽样给人看
    print("\n  ── 随机抽样 8 条 ──")
    import random
    random.seed(42)
    for r in random.sample(uniq, min(8, len(uniq))):
        print(f"  [{r['kind'][:4]}] {r['text'][:56]}")


if __name__ == "__main__":
    main()
