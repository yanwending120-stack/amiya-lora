# -*- coding: utf-8 -*-
"""
把阿米娅的真实对话回合做成 LLaMA-Factory 训练集。

输入：out\amiya_tagged.jsonl
输出：LLaMA-Factory\data\amiya_chat.json  （sharegpt 格式）

用法：python scripts\06_build_trainset.py
"""
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
LF_DATA = os.environ.get("LLAMAFACTORY_DATA", "LLaMA-Factory/data")

DROP_LABELS = {"战斗指挥", "剧情设定"}

# 前一句如果是这些「说话人」，多半是旁白/群演，不是真对话
NARR_SPEAKERS = {"", "？？？", "旁白"}

# 第三人称描写：以「阿米娅/她/你」开头做动作
NARR_RE = [
    re.compile(r"^阿米娅(侧过|望向|看向|点了|摇了|沉默|伸出|抬起|低下|怔怔|注意)"),
    re.compile(r"^你(忍不住|看见|听见|注意到|发现)"),
    re.compile(r"^卡特斯女孩"),
]

ACT_RE = re.compile(
    r"(摇了摇头|点了点头|笑了笑|叹了口气|皱起|愣住|没有回答|转身|抬起头|低下头|望了望|看了看)"
)


def has_cn(t):
    return bool(re.search(r"[\u4e00-\u9fff]", t))


def clean(t):
    t = t.strip()
    t = t.replace("{@nickname}", "博士")
    t = re.sub(r"\s+", "", t)
    return t


def is_narr(t):
    return any(p.search(t) for p in NARR_RE)


def main():
    src = os.path.join(OUT, "amiya_tagged.jsonl")
    rows = [json.loads(l) for l in open(src, encoding="utf-8")]

    pairs, dropped = [], {"类别": 0, "无上下文": 0, "脏数据": 0, "重复": 0}
    seen = set()
    for r in rows:
        if r["label"] in DROP_LABELS:
            dropped["类别"] += 1
            continue
        prev_s = (r["prev_speaker"] or "").strip()
        prev_t = clean(r["prev_text"] or "")
        ans = clean(r["text"])

        if prev_s in NARR_SPEAKERS or prev_s == "阿米娅" or not prev_t:
            dropped["无上下文"] += 1
            continue
        if len(prev_t) < 2 or len(ans) < 2 or not has_cn(ans) or not has_cn(prev_t):
            dropped["脏数据"] += 1
            continue
        if is_narr(prev_t) or is_narr(ans):
            dropped["脏数据"] += 1
            continue
        if "阿米娅" in ans and not ans.startswith("我叫阿米娅"):
            dropped["脏数据"] += 1
            continue
        # 整句是括号旁白（「（疑惑地歪头）」）
        if re.fullmatch(r"（.*）", prev_t) or re.fullmatch(r"（.*）", ans):
            dropped["脏数据"] += 1
            continue
        # 回答是第三人称动作描写
        if ACT_RE.search(ans) and "我" not in ans:
            dropped["脏数据"] += 1
            continue
        key = (prev_t, ans)
        if key in seen:
            dropped["重复"] += 1
            continue
        seen.add(key)
        pairs.append({
            "conversations": [
                {"from": "human", "value": prev_t},
                {"from": "gpt", "value": ans},
            ]
        })

    print("=" * 56)
    print("  训练集构建")
    print("=" * 56)
    for k, v in dropped.items():
        print(f"  剔除·{k:<6}: {v}")
    print(f"  ────────────────")
    print(f"  最终     : {len(pairs)} 条对话")

    lens = sorted(len(p["conversations"][1]["value"]) for p in pairs)
    print(f"  回答字数 : 中位 {lens[len(lens)//2]}，平均 {sum(lens)/len(lens):.1f}")

    dst = os.path.join(LF_DATA, "amiya_chat.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    print(f"\n  已写出：{dst}")

    # 注册到 dataset_info.json
    info_path = os.path.join(LF_DATA, "dataset_info.json")
    info = json.load(open(info_path, encoding="utf-8"))
    info["amiya_chat"] = {
        "file_name": "amiya_chat.json",
        "formatting": "sharegpt",
        "columns": {"messages": "conversations"},
        "tags": {
            "role_tag": "from",
            "content_tag": "value",
            "user_tag": "human",
            "assistant_tag": "gpt",
        },
    }
    json.dump(info, open(info_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"  已注册数据集名：amiya_chat")

    import random
    rng = random.Random(2)
    print("\n  ── 随机 6 条，训练前最后过目 ──")
    for p in rng.sample(pairs, 6):
        print(f"\n  问：{p['conversations'][0]['value'][:46]}")
        print(f"  答：{p['conversations'][1]['value'][:46]}")


if __name__ == "__main__":
    main()
