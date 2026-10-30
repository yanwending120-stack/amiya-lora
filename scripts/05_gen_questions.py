# -*- coding: utf-8 -*-
"""
反过来生成对话：先有阿米娅的台词，再让模型猜「博士会问什么」。

用法：
    python scripts\05_gen_questions.py --limit 300
    python scripts\05_gen_questions.py --limit 3000 --batch 8

输出：out\amiya_qa.jsonl  （每行 {user, assistant, source}）
特性：断点续跑——重复执行会跳过已经生成的。
"""
import argparse
import json
import os
import random
import re
import sys
import time

os.system("")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "out")
MODEL = os.environ.get("QWEN_MODEL_DIR", "models/Qwen3-1.7B")

SYSTEM = (
    "你是明日方舟的编剧助手。下面给出若干句「阿米娅」对「博士」说的话。"
    "请为每一句反推出一个最可能让阿米娅这样回答的提问。"
    "提问要具体、贴住这句话的内容，不要问「你在说什么」这种空话，"
    "不要出现编号以外的任何解释。"
)

TEMPLATE = """下面有 {n} 句阿米娅说的话：

{lines}

请为每一句反推出一个博士会问的问题。只输出 {n} 行，每行一个问题，格式为「序号. 问题」，不要输出其他内容。"""


def load():
    tk = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()
    return tk, model


def ask(tk, model, lines, temp=0.8):
    n = len(lines)
    body = "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
    msgs = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": TEMPLATE.format(n=n, lines=body)},
    ]
    prompt = tk.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = tk(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=64 + 40 * n, do_sample=temp > 0,
            temperature=temp, top_p=0.9,
            pad_token_id=tk.pad_token_id or tk.eos_token_id,
        )
    text = tk.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return text


def parse(text, n):
    """从模型输出里抠出 n 个问题，失败返回 None"""
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    qs = []
    for l in lines:
        m = re.match(r"^\s*(\d+)[\.、\):：]\s*(.+)$", l)
        if m:
            qs.append(m.group(2).strip())
        elif qs is not None and len(qs) < n and not l.startswith("问题"):
            # 没有编号的行也可能是一个问题
            if l.endswith(("？", "?")) or len(l) < 30:
                qs.append(l.strip())
    qs = [q for q in qs if q]
    if len(qs) < n:
        return None
    return qs[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = [json.loads(l)["text"]
              for l in open(os.path.join(OUT, "amiya_corpus.jsonl"), encoding="utf-8")]
    random.seed(args.seed)
    random.shuffle(corpus)

    out_path = os.path.join(OUT, "amiya_qa.jsonl")
    done = set()
    if os.path.exists(out_path):
        for l in open(out_path, encoding="utf-8"):
            try:
                done.add(json.loads(l)["assistant"])
            except Exception:  # noqa: BLE001
                pass

    pool = [t for t in corpus if t not in done][: args.limit]
    print(f"待生成 {len(pool)} 条（已存在 {len(done)} 条，本次上限 {args.limit}）")

    print("加载模型……")
    tk, model = load()

    fout = open(out_path, "a", encoding="utf-8")
    ok, bad = 0, 0
    t0 = time.time()
    batches = [pool[i:i + args.batch] for i in range(0, len(pool), args.batch)]
    for bi, batch in enumerate(batches, 1):
        raw = ask(tk, model, batch, args.temp)
        qs = parse(raw, len(batch))
        if qs is None:
            bad += 1
            continue
        for line, q in zip(batch, qs):
            fout.write(json.dumps({"user": q, "assistant": line, "source": "反推"},
                                  ensure_ascii=False) + "\n")
        fout.flush()
        ok += len(batch)
        el = time.time() - t0
        eta = el / bi * (len(batches) - bi)
        print(f"  [{bi}/{len(batches)}] 已生成 {ok} 条，失败批次 {bad}，"
              f"已用 {el:.0f}s，剩余约 {eta:.0f}s")
    fout.close()
    print(f"\n完成：成功 {ok} 条，失败 {bad} 批 → {out_path}")


if __name__ == "__main__":
    main()
