# -*- coding: utf-8 -*-
"""
按方案 B 生成阿米娅的训练语料：去掉战斗指挥 / 剧情设定 / 旁白。

用法：python scripts\04_build_corpus.py
输出：out\amiya_corpus.jsonl、out\amiya_corpus.txt
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

DROP_LABELS = {"战斗指挥", "剧情设定"}

# 第三人称旁白/动作描写：以「阿米娅/她/卡特斯女孩」开头做动作，或整句无「我」
NARR_PATTERNS = [
    re.compile(r"^阿米娅(望着|看向|怔怔|没有|点了|摇了|沉默|伸出|抬起头|低下头)"),
    re.compile(r"^卡特斯女孩"),
    re.compile(r"^(她|你)(还是)?(没有回答|没有应声|看向|望着|忍不住)"),
    re.compile(r"^不管她看向哪里"),
    re.compile(r"^你(忍不住|看见|听见|注意到|发现)"),
    re.compile(r"^[^我你]{0,8}(点了点头|摇了摇头|怔怔地)"),
    re.compile(r"^阿米娅"),
]

def is_narration(t):
    if "阿米娅" in t and not t.startswith("我叫阿米娅"):
        return True
    return any(p.search(t) for p in NARR_PATTERNS)


def main():
    src = os.path.join(OUT, "amiya_tagged.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]

    kept, dropped = [], {"类别剔除": 0, "旁白剔除": 0}
    for r in rows:
        if r["label"] in DROP_LABELS:
            dropped["类别剔除"] += 1
            continue
        if is_narration(r["text"]):
            dropped["旁白剔除"] += 1
            continue
        kept.append(r)

    # 归一化后再去一次重（忽略末尾标点差异）
    seen = {}
    for r in kept:
        key = re.sub(r"[\s。！？!?，,、…\.]+$", "", r["text"])
        key = re.sub(r"\s+", "", key)
        if key and key not in seen:
            seen[key] = r

    final = list(seen.values())
    total_char = sum(len(r["text"]) for r in final)

    print("=" * 60)
    print("  方案 B 语料构建")
    print("=" * 60)
    print(f"  输入        : {len(rows)} 条")
    for k, v in dropped.items():
        print(f"  {k:<10}: -{v}")
    print(f"  二次去重    : -{len(kept) - len(final)}")
    print(f"  ─────────────────────────")
    print(f"  最终        : {len(final)} 条")
    print(f"  总字数      : {total_char:,} 字")
    print(f"  平均长度    : {total_char / max(len(final), 1):.1f} 字")
    print(f"  估算 token  : {int(total_char * 0.75):,}")

    lens = sorted(len(r["text"]) for r in final)
    n = len(lens)
    print(f"  中位数      : {lens[n // 2]} 字")
    print(f"  10% / 90% 分位: {lens[n // 10]} / {lens[n * 9 // 10]} 字")

    with open(os.path.join(OUT, "amiya_corpus.jsonl"), "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps({"text": r["text"]}, ensure_ascii=False) + "\n")
    with open(os.path.join(OUT, "amiya_corpus.txt"), "w", encoding="utf-8") as f:
        for r in final:
            f.write(r["text"] + "\n")
    print(f"\n  已写出 out\\amiya_corpus.jsonl / .txt")

    import random
    rng = random.Random(3)
    print("\n  ── 随机 10 条，感受一下语料质量 ──")
    for r in rng.sample(final, 10):
        print(f"   · {r['text'][:54]}")


if __name__ == "__main__":
    main()
