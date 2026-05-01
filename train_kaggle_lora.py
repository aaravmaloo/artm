#!/usr/bin/env python3
"""
QLoRA training script for a portable conversational model (Kaggle T4 target).

Outputs:
- LoRA adapter checkpoints
- Final LoRA adapter
- Tokenizer
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import torch
from datasets import Dataset, concatenate_datasets, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


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
    parser.add_argument("--base_model", type=str, default="openai-community/gpt2")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/portable_chat_qlora")
    parser.add_argument("--total_samples", type=int, default=280_000)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument(
        "--dataset_cache_dir",
        type=str,
        default="/kaggle/working/hf_cache",
        help="Cache for downloaded HF dataset shards.",
    )
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


def normalize_role(raw_role: Optional[str]) -> Optional[str]:
    if raw_role is None:
        return None
    return ROLE_MAP.get(str(raw_role).strip().lower())


def normalize_message_list(messages: Iterable[Dict]) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for turn in messages:
        if not isinstance(turn, dict):
            continue
        role = normalize_role(turn.get("role") or turn.get("from") or turn.get("speaker"))
        content = first_non_empty(turn.get("content"), turn.get("value"), turn.get("text"))
        if role is None or content is None:
            continue
        out.append({"role": role, "content": content})
    return out


def extract_messages(row: Dict) -> List[Dict[str, str]]:
    candidates = [
        row.get("messages"),
        row.get("conversations"),
        row.get("conversation"),
        row.get("chat"),
    ]
    for candidate in candidates:
        messages = normalize_message_list(to_list(candidate))
        if messages:
            return messages

    # Fallback for instruction-style records.
    instruction = first_non_empty(row.get("instruction"), row.get("prompt"), row.get("question"))
    output = first_non_empty(row.get("output"), row.get("response"), row.get("answer"), row.get("completion"))
    if instruction and output:
        return [{"role": "user", "content": instruction}, {"role": "assistant", "content": output}]
    return []


def format_chat(messages: List[Dict[str, str]]) -> Optional[str]:
    if not messages:
        return None

    has_user = any(m["role"] == "user" for m in messages)
    has_assistant = any(m["role"] == "assistant" for m in messages)
    if not has_user or not has_assistant:
        return None

    chunks: List[str] = []
    for m in messages:
        if m["role"] == "system":
            chunks.append(f"<|system|>\n{m['content'].strip()}")
        elif m["role"] == "user":
            chunks.append(f"<|user|>\n{m['content'].strip()}")
        elif m["role"] == "assistant":
            chunks.append(f"<|assistant|>\n{m['content'].strip()}")

    text = "\n".join(chunks).strip()
    if not text:
        return None
    return text + "\n<|endoftext|>"


def sample_source(
    source: SourceSpec,
    sample_count: int,
    seed: int,
    cache_dir: str,
) -> Dataset:
    stream = load_dataset(
        source.dataset_id,
        split=source.split,
        streaming=True,
        cache_dir=cache_dir,
    ).shuffle(seed=seed, buffer_size=50_000)

    rows: List[Dict[str, str]] = []
    for row in stream:
        messages = extract_messages(row)
        text = format_chat(messages)
        if text is None:
            continue
        rows.append({"text": text, "source": source.name})
        if len(rows) >= sample_count:
            break

    if len(rows) < sample_count:
        raise RuntimeError(
            f"{source.name}: requested {sample_count} samples, collected {len(rows)}. "
            "Check dataset availability/gated access."
        )
    return Dataset.from_list(rows)


def build_mixed_dataset(total_samples: int, seed: int, cache_dir: str) -> Dataset:
    counts: Dict[str, int] = {}
    running = 0
    for idx, src in enumerate(SOURCES):
        if idx < len(SOURCES) - 1:
            count = int(total_samples * src.ratio)
            running += count
        else:
            count = total_samples - running
        counts[src.name] = count

    datasets = []
    for src in SOURCES:
        ds = sample_source(src, counts[src.name], seed=seed + hash(src.name) % 10_000, cache_dir=cache_dir)
        datasets.append(ds)
        print(f"[data] {src.name}: {len(ds)} samples")

    mixed = concatenate_datasets(datasets).shuffle(seed=seed)
    print(f"[data] mixed total: {len(mixed)}")
    return mixed


def make_sft_config(args: argparse.Namespace) -> SFTConfig:
    # TRL changed field names between versions: use introspection for compatibility.
    sft_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        per_device_train_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        fp16=True,
        bf16=False,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        report_to="none",
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
    )

    sig = inspect.signature(SFTConfig)
    if "max_seq_length" in sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_len
    elif "max_length" in sig.parameters:
        sft_kwargs["max_length"] = args.max_seq_len
    else:
        raise RuntimeError("Your TRL version does not expose max_seq_length/max_length in SFTConfig.")

    return SFTConfig(**sft_kwargs)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = build_mixed_dataset(
        total_samples=args.total_samples,
        seed=args.seed,
        cache_dir=args.dataset_cache_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["c_attn", "c_proj", "c_fc"],
    )

    sft_config = make_sft_config(args)

    trainer_kwargs = dict(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        dataset_text_field="text",
        peft_config=lora_config,
        packing=False,
    )

    # TRL API also changed tokenizer -> processing_class in newer releases.
    try:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kwargs)
    except TypeError:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kwargs)

    resume_path = args.resume_from_checkpoint
    if resume_path is not None and not Path(resume_path).exists():
        raise FileNotFoundError(f"Checkpoint path not found: {resume_path}")

    trainer.train(resume_from_checkpoint=resume_path)

    adapter_dir = output_dir / "lora_adapter_final"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    trainer.save_state()
    print(f"[done] LoRA adapter saved to: {adapter_dir}")


if __name__ == "__main__":
    main()

