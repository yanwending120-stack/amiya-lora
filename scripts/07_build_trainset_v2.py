# -*- coding: utf-8 -*-
"""
v2 训练集：
  1. 连续台词合并成长回答（治「话短」）
  2. 混入人物/势力档案问答（治「不知道凯尔希是谁」）

输入：out\amiya_lines.jsonl、repo\arknights-lore-main\{人物档案,势力档案}\*.md
输出：LLaMA-Factory\data\amiya_chat_v2.json

用法：python scripts\07_build_trainset_v2.py
"""
import json
import os
import random
import re
from collections import OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
REPO = os.path.join(ROOT, "repo", "arknights-lore-main")
LF_DATA = os.environ.get("LLAMAFACTORY_DATA", "LLaMA-Factory/data")

MAX_ANSWER = 300          # 合并后的回答最多留多少字
N_KNOWLEDGE = 300         # 档案问答生成多少条

LEX_COMBAT = ["撤退", "进攻", "攻击", "掩护", "防线", "支援", "守住", "警戒",
              "射击", "突围", "阵型", "火力", "敌军", "指挥部"]
LEX_SETTING = ["源石", "矿石病", "感染者", "天灾", "乌萨斯", "整合运动", "萨卡兹",
               "魔王", "卡兹戴尔", "切尔诺伯格", "王冠", "特蕾西娅", "石棺"]

CHATTY_TEMPLATES = {
    "人物": ["{n}是谁？", "介绍一下{n}", "{n}是什么人？"],
    "势力": ["{n}是什么？", "介绍一下{n}", "{n}是做什么的？"],
}


def has_cn(t):
    return bool(re.search(r"[\u4e00-\u9fff]", t))


def clean(t):
    t = (t or "").strip().replace("{@nickname}", "博士")
    return re.sub(r"\s+", "", t)


def truncate(t, limit=MAX_ANSWER):
    if len(t) <= limit:
        return t
    cut = t[:limit]
    pos = max(cut.rfind("。"), cut.rfind("！"), cut.rfind("？"))
    return cut[:pos + 1] if pos > limit * 0.5 else cut + "。"


# ── 1. 合并连续台词 ────────────────────────────────────────
def build_blocks():
    rows = [json.loads(l) for l in open(os.path.join(OUT, "amiya_lines.jsonl"), encoding="utf-8")]
    rows = [r for r in rows if r["speaker"] == "阿米娅"]

    by_file = OrderedDict()
    for r in rows:
        by_file.setdefault(r["file"], []).append(r)

    blocks = []
    for f, rs in by_file.items():
        buf, head = [], None
        for r in rs:
            if r["prev_speaker"] == "阿米娅" and buf:
                buf.append(r["text"])
            else:
                if buf:
                    blocks.append((head, "".join(buf)))
                buf, head = [r["text"]], r
        if buf:
            blocks.append((head, "".join(buf)))
    return blocks


def filter_blocks(blocks):
    kept, stat = [], {"无上下文": 0, "旁白": 0, "脏": 0, "术语太密": 0, "重复": 0}
    seen = set()
    for head, text in blocks:
        prev_s = clean(head["prev_speaker"])
        prev_t = clean(head["prev_text"])
        txt = clean(text)

        if not prev_s or prev_s in ("？？？", "旁白") or not prev_t or len(prev_t) < 2:
            stat["无上下文"] += 1
            continue
        if re.fullmatch(r"（.*）", prev_t) or re.fullmatch(r"（.*）", txt):
            stat["脏"] += 1
            continue
        if "阿米娅" in txt and not txt.startswith("我叫阿米娅"):
            stat["旁白"] += 1
            continue
        if not has_cn(txt) or not has_cn(prev_t):
            stat["脏"] += 1
            continue
        density = sum(txt.count(w) for w in LEX_COMBAT + LEX_SETTING)
        if density >= 3:
            stat["术语太密"] += 1
            continue
        txt = truncate(txt)
        key = (prev_t, txt)
        if key in seen:
            stat["重复"] += 1
            continue
        seen.add(key)
        kept.append({"user": prev_t, "assistant": txt, "src": "对话"})
    return kept, stat


# ── 2. 档案问答 ───────────────────────────────────────────
def build_knowledge():
    # 只保留能描述「这个人/势力是什么样」的字段
    KEEP = {"身份", "阵营", "种族", "形象", "性格", "定位", "立场", "职责", "职能"}
    # 元信息一律丢弃——没人会在对话里说「首次登场：活动《吾导先路》」
    DROP = {"信息来源", "首次登场", "经典台词", "关键信息", "结局", "经历",
            "身世", "理念", "后续", "备注", "待补充"}

    items = []
    for sub, kind in (("人物档案", "人物"), ("势力档案", "势力")):
        folder = os.path.join(REPO, sub)
        if not os.path.isdir(folder):
            continue
        for fn in sorted(os.listdir(folder)):
            if not fn.endswith(".md") or fn == "README.md":
                continue
            path = os.path.join(folder, fn)
            raw = open(path, encoding="utf-8", errors="replace").read()
            lines = [l.strip() for l in raw.splitlines() if l.strip()]

            name = ""
            vals = []
            for l in lines:
                if l.startswith("# ") and not name:
                    name = re.sub(r"[（(].*?[）)]", "", l[2:]).strip()
                    continue
                if not l.startswith("- "):
                    continue
                body = l[2:].strip().replace("**", "")
                m = re.match(r"^([^：:]{1,10})[：:]\s*(.+)$", body)
                if not m:
                    continue
                label = re.sub(r"[（(].*?[）)]", "", m.group(1)).strip()
                if label in DROP or label not in KEEP:
                    continue
                value = re.sub(r"[（(][^）)]*[）)]?", "", m.group(2))
                value = re.sub(r"[\"'“”「」]", "", value)
                for piece in re.split(r"[；;｜|]", value):
                    piece = re.sub(r"^\**([^：:]{1,6})[：:]\**", "", piece)
                    piece = piece.strip(" ，,、。：:\t")
                    if 2 <= len(piece) <= 40:
                        vals.append(piece)
                if len(vals) >= 3:
                    break

            if not name or not vals:
                continue
            # 纯描述句，不带字段名
            answer = f"{name}，" + "，".join(vals[:3]) + "。"
            items.append({"name": name, "answer": truncate(answer, 90), "kind": kind})

    random.seed(9)
    random.shuffle(items)
    items = items[:N_KNOWLEDGE]

    pairs = []
    for i, it in enumerate(items):
        tpl = CHATTY_TEMPLATES[it["kind"]]
        q = tpl[i % len(tpl)].format(n=it["name"])
        pairs.append({"user": q, "assistant": it["answer"], "src": "档案"})
    return pairs


def main():
    print("合并连续台词……")
    blocks = build_blocks()
    print(f"  合并得到 {len(blocks)} 段")

    style, stat = filter_blocks(blocks)
    print(f"  过滤后保留 {len(style)} 段")
    for k, v in stat.items():
        print(f"    剔除·{k:<6}: {v}")

    print("\n生成档案问答……")
    know = build_knowledge()
    print(f"  生成 {len(know)} 条")

    allp = style + know
    random.seed(42)
    random.shuffle(allp)

    print("\n" + "=" * 60)
    print("  v2 训练集")
    print("=" * 60)
    print(f"  对话(合并) : {len(style):>4}   条")
    print(f"  档案问答   : {len(know):>4}   条")
    print(f"  合计       : {len(allp):>4}   条")
    ls = sorted(len(p["assistant"]) for p in style)
    print(f"  对话回答长度: 中位 {ls[len(ls)//2]} 字，平均 {sum(ls)/len(ls):.1f} 字")

    conv = [{"conversations": [
        {"from": "human", "value": p["user"]},
        {"from": "gpt", "value": p["assistant"]},
    ]} for p in allp]

    dst = os.path.join(LF_DATA, "amiya_chat_v2.json")
    json.dump(conv, open(dst, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  已写出 {dst}")

    info_path = os.path.join(LF_DATA, "dataset_info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    info["amiya_chat_v2"] = {
        "file_name": "amiya_chat_v2.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {"role_tag": "from", "content_tag": "value",
                 "user_tag": "human", "assistant_tag": "gpt"},
    }
    json.dump(info, open(info_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("  已注册：amiya_chat_v2")

    print("\n  ── 抽样 ──")
    rng = random.Random(1)
    print("\n▍合并后的对话")
    for p in rng.sample(style, 4):
        print(f"  问：{p['user'][:40]}")
        print(f"  答：{p['assistant'][:90]}")
    print("\n▍档案问答")
    for p in rng.sample(know, 3):
        print(f"  问：{p['user']}")
        print(f"  答：{p['assistant'][:90]}")


if __name__ == "__main__":
    main()

