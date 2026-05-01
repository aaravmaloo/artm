#!/usr/bin/env python3
"""
Generate synthetic distillation data with microsoft/Phi-3.5-mini-instruct.

Each JSONL record includes:
- prompt
- teacher response
- generated token IDs
- top-k teacher logits per generated token
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, set_seed


@dataclass(frozen=True)
class SourceSpec:
    name: str
    dataset_id: str
    split: str
    ratio: float


SOURCES: List[SourceSpec] = [
    SourceSpec("ultrachat", "HuggingFaceH4/ultrachat_200k", "train_sft", 0.60),
    SourceSpec("openhermes", "teknium/OpenHermes-2.5", "train", 0.25),
    SourceSpec("lmsys", "lmsys/lmsys-chat-1m", "train", 0.15),
]

ROLE_MAP = {
    "human": "user",
    "user": "user",
    "assistant": "assistant",
    "gpt": "assistant",
    "bot": "assistant",
    "system": "system",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--output_jsonl", type=str, default="/kaggle/working/artm_teacher_data.jsonl")
    parser.add_argument("--total_prompts", type=int, default=80_000)
    parser.add_argument("--eval_ratio", type=float, default=0.05)
    parser.add_argument("--max_prompt_chars", type=int, default=1200)
    parser.add_argument("--max_new_tokens", type=int, default=192)
    parser.add_argument("--topk_logits", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache_dir", type=str, default="/kaggle/working/hf_cache")
    parser.add_argument("--load_in_4bit", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def to_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def first_non_empty(*values) -> Optional[str]:
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            return text
    return None


def normalize_messages(raw_messages: Iterable[Dict]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for turn in raw_messages:
        if not isinstance(turn, dict):
            continue
        role = turn.get("role") or turn.get("from") or turn.get("speaker")
        role = ROLE_MAP.get(str(role).lower()) if role is not None else None
        content = first_non_empty(turn.get("content"), turn.get("value"), turn.get("text"))
        if role is None or content is None:
            continue
        out.append({"role": role, "content": content})
    return out


def extract_prompt(row: Dict, max_prompt_chars: int) -> Optional[str]:
    candidates = [
        row.get("messages"),
        row.get("conversations"),
        row.get("conversation"),
        row.get("chat"),
    ]
    for candidate in candidates:
        messages = normalize_messages(to_list(candidate))
        if not messages:
            continue
        for msg in messages:
            if msg["role"] == "user":
                prompt = msg["content"].strip()
                if prompt:
                    return prompt[:max_prompt_chars]

    prompt = first_non_empty(row.get("instruction"), row.get("prompt"), row.get("question"))
    if prompt:
        return prompt[:max_prompt_chars]
    return None


def sample_prompts(
    source: SourceSpec,
    count: int,
    seed: int,
    cache_dir: str,
    max_prompt_chars: int,
) -> List[str]:
    stream = load_dataset(
        source.dataset_id,
        split=source.split,
        streaming=True,
        cache_dir=cache_dir,
    ).shuffle(seed=seed, buffer_size=50_000)

    prompts: List[str] = []
    seen: set[str] = set()
    for row in stream:
        prompt = extract_prompt(row, max_prompt_chars=max_prompt_chars)
        if prompt is None:
            continue
        key = prompt.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        prompts.append(prompt)
        if len(prompts) >= count:
            break

    if len(prompts) < count:
        raise RuntimeError(f"{source.name}: only collected {len(prompts)}/{count} prompts.")
    return prompts


def render_chat(tokenizer, prompt: str) -> torch.Tensor:
    messages = [{"role": "user", "content": prompt}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        input_ids = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
        )
    else:
        fallback = f"User: {prompt}\nAssistant:"
        input_ids = tokenizer(fallback, return_tensors="pt").input_ids
    return input_ids


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"{output_path} exists. Pass --overwrite to replace it.")

    counts: Dict[str, int] = {}
    running = 0
    for i, src in enumerate(SOURCES):
        if i < len(SOURCES) - 1:
            c = int(args.total_prompts * src.ratio)
            running += c
        else:
            c = args.total_prompts - running
        counts[src.name] = c

    all_prompts: List[Dict[str, str]] = []
    for src in SOURCES:
        prompts = sample_prompts(
            source=src,
            count=counts[src.name],
            seed=args.seed + (hash(src.name) % 10_000),
            cache_dir=args.cache_dir,
            max_prompt_chars=args.max_prompt_chars,
        )
        all_prompts.extend({"source": src.name, "prompt": p} for p in prompts)
        print(f"[data] {src.name}: {len(prompts)} prompts")

    random.Random(args.seed).shuffle(all_prompts)
    print(f"[data] total prompts: {len(all_prompts)}")

    quant_cfg = None
    dtype = torch.bfloat16
    if args.load_in_4bit:
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        dtype = None

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        quantization_config=quant_cfg,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()

    eval_cutoff = int(len(all_prompts) * args.eval_ratio)

    with output_path.open("w", encoding="utf-8") as fout:
        written = 0
        for idx, item in enumerate(all_prompts, start=1):
            prompt = item["prompt"]
            source = item["source"]

            input_ids = render_chat(tokenizer, prompt).to(model.device)
            attention_mask = torch.ones_like(input_ids)
            input_len = input_ids.shape[1]

            with torch.no_grad():
                generated = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    output_scores=True,
                    return_dict_in_generate=True,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            seq = generated.sequences[0]
            gen_ids = seq[input_len:].tolist()
            if not gen_ids:
                continue

            response_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
            if not response_text:
                continue

            teacher_topk_ids: List[List[int]] = []
            teacher_topk_logits: List[List[float]] = []
            for step_score in generated.scores:
                step_score = step_score[0].float().cpu()
                k = min(args.topk_logits, step_score.shape[-1])
                vals, ids = torch.topk(step_score, k=k)
                teacher_topk_ids.append(ids.tolist())
                teacher_topk_logits.append(vals.tolist())

            split = "eval" if written < eval_cutoff else "train"
            record = {
                "id": idx,
                "split": split,
                "source": source,
                "prompt": prompt,
                "teacher_response": response_text,
                "teacher_token_ids": gen_ids,
                "teacher_topk_ids": teacher_topk_ids,
                "teacher_topk_logits": teacher_topk_logits,
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

            if written % 100 == 0:
                print(f"[progress] wrote {written} rows")

    print(f"[done] wrote {written} records to: {output_path}")


if __name__ == "__main__":
    main()