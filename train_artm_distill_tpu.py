import os
import math
import time
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# TPU Libraries
import torch_xla.core.xla_model as xm

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
    set_seed
)

from dataclasses import dataclass

@dataclass
class DistillRecord:
    prompt: str
    response: str

class JsonlDistillDataset(Dataset):
    def __init__(self, jsonl_path: str, split: str) -> None:
        self.samples: List[DistillRecord] = []
        with Path(jsonl_path).open("r", encoding="utf-8") as fin:
            for line in fin:
                row = json.loads(line)
                row_split = row.get("split")
                if split != "all" and row_split is not None and row_split != split:
                    continue
                prompt = str(row.get("prompt", "")).strip()
                response = str(row.get("teacher_response", "")).strip()
                if not prompt or not response:
                    continue
                self.samples.append(DistillRecord(prompt=prompt, response=response))
    def __len__(self) -> int: return len(self.samples)
    def __getitem__(self, idx: int) -> DistillRecord: return self.samples[idx]

def _apply_chat_template(tokenizer, prompt: str, response: str | None = None, generation_prompt: bool = False) -> List[int]:
    messages = [{"role": "user", "content": prompt}]
    if response is not None:
        messages.append({"role": "assistant", "content": response})
    
    ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=generation_prompt,
    )
    
    # Extract IDs if returned as a complex object or dictionary
    if hasattr(ids, "input_ids"):
        ids = ids.input_ids
    elif isinstance(ids, dict):
        ids = ids.get("input_ids", ids.get("input", ids))
    
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    
    # Ensure it's a flat list of integers
    if not isinstance(ids, list):
        ids = list(ids)
    
    return [int(x) for x in ids]

class DistillCollator:
    def __init__(self, tokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
    def __call__(self, items: Sequence[DistillRecord]) -> Dict[str, torch.Tensor]:
        batch_input_ids, batch_labels = [], []
        for sample in items:
            ids = _apply_chat_template(self.tokenizer, sample.prompt, sample.response)
            if len(ids) > self.max_seq_len: ids = ids[:self.max_seq_len]
            lbls = ids[:]
            batch_input_ids.append(torch.tensor(ids))
            batch_labels.append(torch.tensor(lbls))
        
        input_ids = torch.nn.utils.rnn.pad_sequence(batch_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(batch_labels, batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": (input_ids != self.tokenizer.pad_token_id).long(), "labels": labels}

def shift_for_lm(logits, labels):
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous(), (labels[:, 1:] != -100)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--data_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./tpu_output")
    parser.add_argument("--epochs", type=float, default=3.5)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--student_layers", type=int, default=36)
    parser.add_argument("--student_hidden", type=int, default=1536)
    parser.add_argument("--student_heads", type=int, default=24)
    parser.add_argument("--student_ffn", type=int, default=6144)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = xm.xla_device()
    
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Build Student
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=args.context_length, n_ctx=args.context_length,
        n_embd=args.student_hidden, n_layer=args.student_layers, n_head=args.student_heads,
        n_inner=args.student_ffn, activation_function="gelu_new", use_cache=False
    )
    student = GPT2LMHeadModel(config).to(device)
    
    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model, torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    teacher.eval()

    train_ds = JsonlDistillDataset(args.data_jsonl, split="train")
    train_loader = DataLoader(
        train_ds, batch_size=args.per_device_batch_size, shuffle=True,
        collate_fn=DistillCollator(tokenizer, args.context_length), num_workers=2
    )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(total_steps * 0.03), int(total_steps))

    print(f"[system] Starting Single-Core TPU Distillation (v5e-1)")
    
    for epoch in range(math.ceil(args.epochs)):
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            mask_orig = batch["attention_mask"].to(device)
            
            # Forward Student
            s_out = student(input_ids=input_ids, attention_mask=mask_orig)
            
            # Forward Teacher
            with torch.no_grad():
                t_out = teacher(input_ids=input_ids, attention_mask=mask_orig)
            
            s_logits, s_labels, mask = shift_for_lm(s_out.logits, labels)
            t_logits, _, _ = shift_for_lm(t_out.logits, labels)

            loss_ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)), s_labels.view(-1), ignore_index=-100)
            
            loss_kd = F.kl_div(
                F.log_softmax(s_logits / args.temperature, dim=-1),
                F.softmax(t_logits / args.temperature, dim=-1),
                reduction="batchmean"
            ) * (args.temperature ** 2)

            loss = loss_ce + loss_kd
            loss.backward()
            
            xm.optimizer_step(optimizer)
            scheduler.step()

            if step % 20 == 0:
                print(f"[epoch {epoch}] step {step} loss: {loss.item():.4f}")

            if step > 0 and step % 500 == 0:
                print(f"[save] Saving checkpoint at step {step}...")
                ckpt_path = Path(args.output_dir) / f"checkpoint-{step}"
                student.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)

    student.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
