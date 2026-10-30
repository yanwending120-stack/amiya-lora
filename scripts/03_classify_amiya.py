# -*- coding: utf-8 -*-
"""
给阿米娅的台词自动打标签（纯规则，不调模型，秒出结果）。

用法：python scripts\03_classify_amiya.py
输入：out\amiya_lines.jsonl
输出：out\amiya_tagged.jsonl + 终端报告
"""
import json
import os
import random
import re
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")

# ── 词表：按类别给分，命中一个加分 ──────────────────────
LEX = {
    "对博士": ["博士"],
    "日常关心": ["休息", "睡觉", "睡", "吃", "累", "加油", "谢谢", "早安", "晚安",
             "辛苦", "身体", "健康", "吃饭", "冷", "暖和", "照顾好"],
    "战斗指挥": ["撤退", "进攻", "攻击", "掩护", "防线", "准备", "行动", "命令",
             "支援", "守住", "小心", "警戒", "敌", "射击", "突围", "阵型", "火力"],
    "剧情设定": ["源石", "矿石病", "感染者", "天灾", "罗德岛", "乌萨斯", "整合运动",
             "萨卡兹", "魔王", "卡兹戴尔", "龙门", "切尔诺伯格", "王冠", "特蕾西娅",
             "凯尔希", "干员", "手术", "药剂"],
    "情绪柔软": ["对不起", "抱歉", "没关系", "别怕", "不要紧", "我在", "相信",
             "希望", "谢谢", "温暖", "难过", "伤心", "一起"],
}

# 语气词结尾 → 更像口语 / 日常
CHATTY_RE = re.compile(r"[呢吧啊哦啦呀嘛嘛欸诶嗯哎]$")
QUESTION_RE = re.compile(r"[？?]$")

CATS = list(LEX.keys())


def score(text):
    s = {c: 0 for c in CATS}
    for c, words in LEX.items():
        for w in words:
            if w in text:
                s[c] += 1
    return s


def main():
    src = os.path.join(OUT, "amiya_lines.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]

    # 去重 + 基础过滤
    uniq = {}
    for r in rows:
        t = r["text"].strip()
        if not t:
            continue
        if not re.search(r"[\u4e00-\u9fff]", t):        # 无汉字
            continue
        if len(t) <= 4:                                  # 极短
            continue
        if "{@" in t or "<" in t:                        # 占位符
            continue
        uniq.setdefault(t, r)
    items = list(uniq.values())
    print(f"过滤后剩 {len(items)} 条（原始 {len(rows)}，去重+剔除脏数据）\n")

    tagged = []
    hit_cnt = Counter()
    for r in items:
        t = r["text"]
        s = score(t)
        hits = [c for c in CATS if s[c] > 0]
        for c in hits:
            hit_cnt[c] += 1
        chunky = bool(CHATTY_RE.search(t))
        question = bool(QUESTION_RE.search(t))
        # 判定：先看有没有强特征
        if s["战斗指挥"] > 0:
            label = "战斗指挥"
        elif hits:
            # 多个软类别，取分最高的；平手时优先「对博士」
            best = max(hits, key=lambda c: (s[c] + (1.5 if c == "对博士" else 0)))
            label = best
        elif chunky or question:
            label = "日常语气"
        else:
            label = "剧情陈述"

        tagged.append({**r, "label": label, "hits": hits,
                       "chunky": chunky, "question": question, "len": len(t)})

    print("=" * 68)
    print("  自动打标结果")
    print("=" * 68)
    cnt = Counter(t["label"] for t in tagged)
    total = len(tagged)
    for lab, c in cnt.most_common():
        bar = "█" * round(c / total * 40)
        print(f"  {lab:<8} {c:>5}  {c/total*100:>5.1f}%  {bar}")

    print("\n  ── 多标签重叠（一条台词可能既是A又是B）──")
    for c, n in hit_cnt.most_common():
        print(f"  含「{c}」特征: {n:>5} 条  ({n/total*100:.1f}%)")

    # 核心聊天语料：不是战斗、不是纯设定
    core = [t for t in tagged if t["label"] not in ("战斗指挥", "剧情设定")]
    print(f"\n  ── 可用作「聊天语料」: {len(core)} 条 ({len(core)/total*100:.1f}%) ──")
    core_doc = [t for t in core if "博士" in t["text"]]
    print(f"     其中提到博士的: {len(core_doc)} 条")

    # 每类抽 5 条给人过目
    rng = random.Random(7)
    by = defaultdict(list)
    for t in tagged:
        by[t["label"]].append(t)
    print("\n" + "=" * 68)
    print("  每类抽样（你自己扫一眼，判断这些标签对不对）")
    print("=" * 68)
    for lab, _ in cnt.most_common():
        print(f"\n▍{lab}")
        for t in rng.sample(by[lab], min(5, len(by[lab]))):
            print(f"   · {t['text'][:52]}")

    with open(os.path.join(OUT, "amiya_tagged.jsonl"), "w", encoding="utf-8") as f:
        for t in tagged:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(f"\n已写出 out\\amiya_tagged.jsonl（{len(tagged)} 条）")


if __name__ == "__main__":
    main()
