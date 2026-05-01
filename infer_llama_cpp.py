#!/usr/bin/env python3
"""
Portable chat inference using llama-cpp-python.

Works on:
- Raspberry Pi (CPU)
- Laptops/desktops (CPU or Metal/OpenBLAS builds)
- Android Termux Python (CPU)
"""

from __future__ import annotations

import argparse
from typing import List, Tuple

from llama_cpp import Llama


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="You are a concise and helpful assistant.",
    )
    return parser.parse_args()


def build_prompt(history: List[Tuple[str, str]]) -> str:
    parts = []
    for role, content in history:
        if role == "system":
            parts.append(f"<|system|>\n{content}")
        elif role == "user":
            parts.append(f"<|user|>\n{content}")
        elif role == "assistant":
            parts.append(f"<|assistant|>\n{content}")
    parts.append("<|assistant|>\n")
    return "\n".join(parts)


def main() -> None:
    args = parse_args()

    llm = Llama(
        model_path=args.model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        verbose=False,
    )

    history: List[Tuple[str, str]] = [("system", args.system_prompt)]
    print("Portable chat ready. Type 'exit' to quit.")

    while True:
        user_text = input("\nYou: ").strip()
        if not user_text:
            continue
        if user_text.lower() in {"exit", "quit"}:
            break

        history.append(("user", user_text))
        prompt = build_prompt(history)
        response = llm(
            prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stop=["<|user|>", "<|system|>", "<|assistant|>"],
        )
        answer = response["choices"][0]["text"].strip()
        print(f"Assistant: {answer}")
        history.append(("assistant", answer))


if __name__ == "__main__":
    main()
