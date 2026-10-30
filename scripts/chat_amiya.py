# -*- coding: utf-8 -*-
"""
和阿米娅 LoRA 聊天。

用法：
    python scripts\chat_amiya.py
    python scripts\chat_amiya.py --adapter D:\...\amiya_v1

注意：这里的提示词格式必须和训练时完全一致（qwen3_nothink），
也就是「<|im_start|>user ... <|im_end|> <|im_start|>assistant」，
不能带空的 think 块——否则会出现「答非所问」，这个坑我们踩过。
"""
import argparse
import os
import sys
import threading
import time

os.system("")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from transformers.utils import logging as hf_logging

hf_logging.disable_progress_bar()

BASE = os.environ.get("QWEN_MODEL_DIR", "models/Qwen3-1.7B")
ADAPTER = os.environ.get("ADAPTER_DIR", "saves/amiya_v1")

DIM, BOLD, RESET = "\033[2m", "\033[1m", "\033[0m"
CYAN, GREEN, GREY, RED = "\033[36m", "\033[32m", "\033[90m", "\033[31m"

CONF = {"temp": 0.7, "max_new": 256, "top_p": 0.9}


def build_prompt(history):
    """严格对齐 LLaMA-Factory 的 qwen3_nothink 模板"""
    parts = []
    for role, content in history:
        if role == "user":
            parts.append(f"<|im_start|>user\n{content}<|im_end|>\n")
        else:
            parts.append(f"<|im_start|>assistant\n{content}<|im_end|>\n")
    parts.append("<|im_start|>assistant\n")
    return "".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", default=ADAPTER)
    ap.add_argument("--base", default=BASE)
    ap.add_argument("--temp", type=float, default=0.7)
    args = ap.parse_args()
    CONF["temp"] = args.temp

    print(f"{GREY}加载底模 + LoRA……{RESET}", flush=True)
    tk = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, dtype=torch.bfloat16, device_map="cuda:0", trust_remote_code=True
    )
    if args.adapter and os.path.isdir(args.adapter):
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"{GREY}已挂载 LoRA：{args.adapter}{RESET}")
    else:
        print(f"{RED}没找到 LoRA 目录，将用原始模型对话：{args.adapter}{RESET}")
    model.eval()

    streamer = TextIteratorStreamer(tk, skip_prompt=True, skip_special_tokens=True)

    print()
    print(f"{CYAN}{'═' * 60}{RESET}")
    print(f"{CYAN}  阿米娅{RESET}  {GREY}· 输入 /clear 清空，/exit 退出{RESET}")
    print(f"{CYAN}{'═' * 60}{RESET}\n")

    history = []
    while True:
        try:
            user = input(f"{GREEN}博士 ▸ {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user:
            continue
        if user.lower() in ("/exit", "exit", "quit", "再见"):
            break
        if user.lower() == "/clear":
            history = []
            print(f"{GREY}（对话已清空）{RESET}\n")
            continue

        history.append(("user", user))
        prompt = build_prompt(history)
        inputs = tk(prompt, return_tensors="pt").to(model.device)

        kw = dict(**inputs, max_new_tokens=CONF["max_new"], streamer=streamer,
                  pad_token_id=tk.pad_token_id or tk.eos_token_id)
        if CONF["temp"] > 0:
            kw.update(do_sample=True, temperature=CONF["temp"], top_p=CONF["top_p"])
        else:
            kw.update(do_sample=False)

        print(f"{BOLD}阿米娅 ▸ {RESET}", end="", flush=True)
        t0 = time.time()
        acc = []
        th = threading.Thread(target=lambda: model.generate(**kw), daemon=True)
        th.start()
        try:
            for piece in streamer:
                acc.append(piece)
                sys.stdout.write(piece)
                sys.stdout.flush()
        except KeyboardInterrupt:
            print(f"\n{RED}已中断{RESET}")
        th.join()
        text = "".join(acc).replace("<|im_end|>", "").strip()
        dt = time.time() - t0
        print(f"\n{GREY}  ── {dt:.1f}s{RESET}\n")
        history.append(("assistant", text))
        if len(history) > 9:
            history = history[-9:]

    print(f"{GREY}再见。{RESET}")


if __name__ == "__main__":
    main()
