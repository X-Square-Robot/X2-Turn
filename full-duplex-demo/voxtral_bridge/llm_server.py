#!/usr/bin/env python3
"""Minimal streaming LLM server compatible with X Square QwenLLM_stream (/chat)."""

from __future__ import annotations

import argparse
import os
from threading import Thread

import torch
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

app = FastAPI(title="Dialogue Demo LLM")
tokenizer = None
model = None
gen_kwargs = {}


def load_model(model_path: str, max_tokens: int = 512, temperature: float = 0.7, top_p: float = 0.9):
    global tokenizer, model, gen_kwargs
    print(f"[LLM] loading {model_path}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
    ).eval()
    gen_kwargs = {
        "max_new_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": True,
    }
    print(f"[LLM] ready on {model.device}", flush=True)


@app.get("/health")
def health():
    return {"ok": model is not None}


@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    messages = data.get("messages", [])
    if not messages:
        return StreamingResponse(iter([]), media_type="text/plain")

    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    kwargs = dict(**inputs, **gen_kwargs, streamer=streamer)
    thread = Thread(target=model.generate, kwargs=kwargs)
    thread.start()

    def gen():
        try:
            for token in streamer:
                yield token
        finally:
            thread.join(timeout=1.0)

    return StreamingResponse(gen(), media_type="text/plain")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model_dir",
        default=os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-3B-Instruct"),
    )
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=6007)
    p.add_argument("--max_tokens", type=int, default=512)
    return p.parse_args()


def main():
    args = parse_args()
    load_model(args.model_dir, max_tokens=args.max_tokens)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
