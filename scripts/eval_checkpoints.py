# -*- coding: utf-8 -*-
"""
把同一组问题跑遍所有 checkpoint，出一张对比表。

用法：
    python scripts\eval_checkpoints.py
    python scripts\eval_checkpoints.py --ckpt-dir ... --max-new 60

输出：
    reports\ckpt_eval.json（原始结果）
    终端表格（逐问题对比 + 长度/坍缩统计）
"""
import argparse
import json
import os
import re
import time

os.system("")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.environ.get("QWEN_MODEL_DIR", "models/Qwen3-1.7B")
CKPT_DIR = os.path.join(ROOT, "saves", "qwen3-1.7b", "lora", "amiya_v2")
REP = os.path.join(ROOT, "reports")

QUESTIONS = [
    "我今天好累，不想干活了",
    "罗德岛接下来该怎么办",
    "你觉得凯尔希医生是个怎样的人",
    "帮我写一个 Python 快速排序",
    "今天天气不错",
    "凯尔希医生是谁",
    "给我讲讲罗德岛",
    "赫默是谁",
]


def build_prompt(q):
    return f"<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", default=CKPT_DIR)
    ap.add_argument("--max-new", type=int, default=60)
    args = ap.parse_args()

    ckpts = [d for d in os.listdir(args.ckpt_dir)
             if d.startswith("checkpoint-") and os.path.isdir(os.path.join(args.ckpt_dir, d))]
    ckpts.sort(key=lambda s: int(s.split("-")[1]))
    if not ckpts:
        raise SystemExit(f"没找到 checkpoint：{args.ckpt_dir}")
    print(f"共 {len(ckpts)} 个 checkpoint，{len(QUESTIONS)} 个问题\n")

    tk = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    model.eval()

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, os.path.join(args.ckpt_dir, ckpts[0]),
                                      adapter_name="a0")
    model.eval()

    results = {}
    prev = "a0"
    for i, ck in enumerate(ckpts):
        name = f"a{i}"
        if i > 0:
            model.load_adapter(os.path.join(args.ckpt_dir, ck), adapter_name=name)
            model.set_adapter(name)
            try:
                model.delete_adapter(prev)
            except Exception:  # noqa: BLE001
                pass
            prev = name
        else:
            model.set_adapter(name)

        row = []
        t0 = time.time()
        for q in QUESTIONS:
            inputs = tk(build_prompt(q), return_tensors="pt").to(model.device)
            with torch.no_grad():
                out = model.generate(
                    **inputs, max_new_tokens=args.max_new, do_sample=False,
                    pad_token_id=tk.pad_token_id or tk.eos_token_id,
                )
            text = tk.decode(out[0][inputs["input_ids"].shape[1]:],
                             skip_special_tokens=True)
            text = text.replace("<|im_end|>", "").strip()
            row.append(text)
        results[ck] = row
        print(f"  [{i + 1}/{len(ckpts)}] {ck}  用时 {time.time() - t0:.1f}s")

    print("\n" + "=" * 96)
    print("  逐问题对比（贪心解码，同一模型同一问法，唯一变量是训练步数）")
    print("=" * 96)
    for qi, q in enumerate(QUESTIONS):
        print(f"\n▍{q}")
        for ck in ckpts:
            a = results[ck][qi]
            print(f"  {ck:>14} │ {a[:60]}")

    print("\n" + "=" * 96)
    print("  统计")
    print("=" * 96)
    print(f"{'checkpoint':>14} {'平均字数':>8} {'省略号开头':>8} {'复读(重复句)':>10}")
    for ck in ckpts:
        ans = results[ck]
        avg = sum(len(a) for a in ans) / len(ans)
        dot = sum(1 for a in ans if a.startswith(("......", "…", "。。。")))
        dup = len(ans) - len(set(ans))
        print(f"{ck:>14} {avg:>8.1f} {dot:>7}/8 {dup:>10}")

    os.makedirs(REP, exist_ok=True)
    dst = os.path.join(REP, "ckpt_eval.json")
    json.dump({"questions": QUESTIONS, "results": results}, open(dst, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"\n原始结果已存：{dst}")


if __name__ == "__main__":
    main()
