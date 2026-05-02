#!/usr/bin/env python3
"""
ARTM distillation training (student from scratch) for Kaggle T4.

Core objectives:
- Train a new 1B+ student model from config (no LoRA, no fine-tuning adapters)
- Distill from microsoft/Phi-3.5-mini-instruct using:
  1) CE loss on teacher outputs
  2) Logit MSE distillation with temperature scaling
  3) Layer-wise hidden-state distillation
  4) Optional attention-pattern distillation
- Optional fake-quantization-aware training for 4-bit robustness
- Optional structured head pruning + short recovery fine-tune
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers.pytorch_utils import Conv1D
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    GPT2Config,
    GPT2LMHeadModel,
    get_cosine_schedule_with_warmup,
    set_seed,
)


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
        if not self.samples:
            raise RuntimeError(f"No rows loaded from {jsonl_path} for split={split}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> DistillRecord:
        return self.samples[idx]


def _apply_chat_template(tokenizer, prompt: str, response: str | None = None, generation_prompt: bool = False) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation_prompt,
        )
        if isinstance(ids, dict):
            # Some tokenizers return a dict with input_ids and attention_mask
            ids = ids.get("input_ids", ids.get("input", ids))

        if isinstance(ids, str):
            # Fallback if tokenize=True returned a rendered string
            ids = tokenizer.encode(ids, add_special_tokens=False)
        elif isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        
        # Ensure every element is an integer to prevent "too many dimensions 'str'" errors
        return [int(x) for x in ids]

    if response is None:
        text = f"User: {prompt}\nAssistant:"
    else:
        text = f"User: {prompt}\nAssistant: {response}"
    return tokenizer(text, add_special_tokens=True).input_ids


class DistillCollator:
    def __init__(self, tokenizer, max_seq_len: int) -> None:
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pad_token_id = tokenizer.pad_token_id
        if self.pad_token_id is None:
            self.pad_token_id = tokenizer.eos_token_id

    def __call__(self, items: Sequence[DistillRecord]) -> Dict[str, torch.Tensor]:
        batch_input_ids: List[torch.Tensor] = []
        batch_labels: List[torch.Tensor] = []

        for sample in items:
            prompt_ids = _apply_chat_template(
                self.tokenizer,
                prompt=sample.prompt,
                response=None,
                generation_prompt=True,
            )
            full_ids = _apply_chat_template(
                self.tokenizer,
                prompt=sample.prompt,
                response=sample.response,
                generation_prompt=False,
            )

            if len(full_ids) <= 1:
                continue

            overflow = max(0, len(full_ids) - self.max_seq_len)
            if overflow > 0:
                full_ids = full_ids[overflow:]
            prompt_len = max(0, len(prompt_ids) - overflow)
            prompt_len = min(prompt_len, len(full_ids) - 1)

            labels = full_ids.copy()
            for i in range(prompt_len):
                labels[i] = -100

            if all(x == -100 for x in labels):
                continue

            batch_input_ids.append(torch.tensor(full_ids, dtype=torch.long))
            batch_labels.append(torch.tensor(labels, dtype=torch.long))

        if not batch_input_ids:
            raise RuntimeError("All examples in this batch were empty after tokenization/truncation")

        max_len = max(x.shape[0] for x in batch_input_ids)
        input_ids = torch.full((len(batch_input_ids), max_len), self.pad_token_id, dtype=torch.long)
        labels = torch.full((len(batch_labels), max_len), -100, dtype=torch.long)
        attention_mask = torch.zeros((len(batch_input_ids), max_len), dtype=torch.long)

        for i, (ids, lab) in enumerate(zip(batch_input_ids, batch_labels)):
            n = ids.shape[0]
            input_ids[i, :n] = ids
            labels[i, :n] = lab
            attention_mask[i, :n] = 1

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--data_jsonl", type=str, default="/kaggle/working/artm_teacher_data.jsonl")
    parser.add_argument("--output_dir", type=str, default="/kaggle/working/artm_distilled")

    parser.add_argument("--student_layers", type=int, default=36)
    parser.add_argument("--student_hidden", type=int, default=1536)
    parser.add_argument("--student_heads", type=int, default=24)
    parser.add_argument("--student_ffn", type=int, default=6144)
    parser.add_argument("--context_length", type=int, default=2048)

    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--loss_weight_ce", type=float, default=1.0)
    parser.add_argument("--loss_weight_kd", type=float, default=1.0)
    parser.add_argument("--loss_weight_hidden", type=float, default=0.25)
    parser.add_argument("--loss_weight_attn", type=float, default=0.0)
    parser.add_argument("--attn_window", type=int, default=128)
    parser.add_argument("--distill_layers", type=int, default=8)

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--per_device_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=5e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--teacher_load_in_4bit", action="store_true")

    parser.add_argument("--enable_qat", action="store_true")
    parser.add_argument("--qat_bits", type=int, default=4)

    parser.add_argument("--prune_heads_ratio", type=float, default=0.0)
    parser.add_argument("--post_prune_epochs", type=float, default=1.0)

    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_eval_samples", type=int, default=2000)
    return parser.parse_args()


def fake_quantize_tensor(x: torch.Tensor, bits: int = 4, eps: float = 1e-8) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = x.detach().abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(scale / float(qmax), min=eps)
    x_q = torch.round(x / scale).clamp(-qmax, qmax) * scale
    return x + (x_q - x).detach()


class FakeQuantLinear(nn.Module):
    def __init__(self, linear: nn.Linear, bits: int = 4) -> None:
        super().__init__()
        self.linear = linear
        self.bits = bits

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = fake_quantize_tensor(x, bits=self.bits)
        w_q = fake_quantize_tensor(self.linear.weight, bits=self.bits)
        b_q = None
        if self.linear.bias is not None:
            b_q = fake_quantize_tensor(self.linear.bias.unsqueeze(0), bits=self.bits).squeeze(0)
        return F.linear(x_q, w_q, b_q)


class FakeQuantConv1D(nn.Module):
    def __init__(self, conv: Conv1D, bits: int = 4) -> None:
        super().__init__()
        self.conv = conv
        self.bits = bits

    @property
    def weight(self) -> torch.Tensor:
        return self.conv.weight

    @property
    def bias(self) -> torch.Tensor:
        return self.conv.bias

    @property
    def nf(self) -> int:
        return self.conv.nf

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = fake_quantize_tensor(x, bits=self.bits)
        w_q = fake_quantize_tensor(self.conv.weight, bits=self.bits)
        b_q = fake_quantize_tensor(self.conv.bias.unsqueeze(0), bits=self.bits).squeeze(0)
        size_out = x_q.size()[:-1] + (self.conv.nf,)
        out = torch.addmm(b_q, x_q.view(-1, x_q.size(-1)), w_q)
        return out.view(size_out)


def apply_qat_wrappers(module: nn.Module, bits: int = 4) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, FakeQuantLinear(child, bits=bits))
        elif isinstance(child, Conv1D):
            setattr(module, name, FakeQuantConv1D(child, bits=bits))
        else:
            apply_qat_wrappers(child, bits=bits)


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def build_student(tokenizer, args: argparse.Namespace) -> GPT2LMHeadModel:
    config = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=args.context_length,
        n_ctx=args.context_length,
        n_embd=args.student_hidden,
        n_layer=args.student_layers,
        n_head=args.student_heads,
        n_inner=args.student_ffn,
        activation_function="gelu_new",
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        layer_norm_epsilon=1e-5,
        initializer_range=0.02,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )
    return GPT2LMHeadModel(config)


def choose_layer_map(student_layers: int, teacher_layers: int, pairs: int) -> List[tuple[int, int]]:
    pairs = max(1, min(pairs, student_layers, teacher_layers))
    mapping: List[tuple[int, int]] = []
    for i in range(pairs):
        s_idx = round((i + 1) * student_layers / pairs) - 1
        t_idx = round((i + 1) * teacher_layers / pairs) - 1
        mapping.append((max(0, s_idx), max(0, t_idx)))
    return mapping


def shift_for_lm(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shifted_logits = logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    valid = shifted_labels != -100
    return shifted_logits, shifted_labels, valid


def masked_mse(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if mask.any():
        diff = a[mask] - b[mask]
        return (diff * diff).mean()
    return torch.zeros((), device=a.device, dtype=a.dtype)


def eval_perplexity(
    student: nn.Module,
    teacher: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int,
    use_bf16: bool,
) -> tuple[float, float]:
    student.eval()
    teacher.eval()

    ce_sum_student = 0.0
    ce_sum_teacher = 0.0
    token_count = 0

    autocast_dtype = torch.bfloat16 if use_bf16 else None
    autocast_enabled = bool(use_bf16 and device.type == "cuda")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step >= max_batches:
                break
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=autocast_enabled):
                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
                t_out = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )

                s_logits, s_labels, s_valid = shift_for_lm(s_out.logits.float(), labels)
                t_logits, t_labels, _ = shift_for_lm(t_out.logits.float(), labels)

                s_loss = F.cross_entropy(
                    s_logits.view(-1, s_logits.size(-1)),
                    s_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )
                t_loss = F.cross_entropy(
                    t_logits.view(-1, t_logits.size(-1)),
                    t_labels.view(-1),
                    ignore_index=-100,
                    reduction="sum",
                )

                valid_tokens = int(s_valid.sum().item())
                ce_sum_student += float(s_loss.item())
                ce_sum_teacher += float(t_loss.item())
                token_count += valid_tokens

    if token_count == 0:
        return float("inf"), float("inf")

    ppl_student = math.exp(ce_sum_student / token_count)
    ppl_teacher = math.exp(ce_sum_teacher / token_count)
    return ppl_student, ppl_teacher


def prune_gpt2_heads(student: GPT2LMHeadModel, ratio: float) -> Dict[int, List[int]]:
    if ratio <= 0.0:
        return {}

    pruned: Dict[int, List[int]] = {}
    n_layer = student.config.n_layer
    n_head = student.config.n_head
    head_dim = student.config.n_embd // n_head
    prune_per_layer = int(n_head * ratio)
    if prune_per_layer <= 0:
        return {}

    for layer_idx in range(n_layer):
        block = student.transformer.h[layer_idx]
        c_attn_mod = block.attn.c_attn
        c_proj_mod = block.attn.c_proj
        if isinstance(c_attn_mod, FakeQuantConv1D):
            c_attn_mod = c_attn_mod.conv
        if isinstance(c_proj_mod, FakeQuantConv1D):
            c_proj_mod = c_proj_mod.conv

        c_attn = c_attn_mod.weight.data
        q_w, k_w, v_w = c_attn.split(student.config.n_embd, dim=1)

        scores: List[tuple[int, float]] = []
        for head_idx in range(n_head):
            start = head_idx * head_dim
            end = start + head_dim
            score = (
                q_w[:, start:end].abs().mean()
                + k_w[:, start:end].abs().mean()
                + v_w[:, start:end].abs().mean()
            ).item()
            scores.append((head_idx, score))

        scores.sort(key=lambda x: x[1])
        to_prune = [h for h, _ in scores[:prune_per_layer]]
        if to_prune:
            pruned[layer_idx] = to_prune
            for head_idx in to_prune:
                start = head_idx * head_dim
                end = start + head_dim

                q_start = start
                q_end = end
                k_start = student.config.n_embd + start
                k_end = student.config.n_embd + end
                v_start = 2 * student.config.n_embd + start
                v_end = 2 * student.config.n_embd + end

                c_attn_mod.weight.data[:, q_start:q_end] = 0
                c_attn_mod.weight.data[:, k_start:k_end] = 0
                c_attn_mod.weight.data[:, v_start:v_end] = 0
                c_attn_mod.bias.data[q_start:q_end] = 0
                c_attn_mod.bias.data[k_start:k_end] = 0
                c_attn_mod.bias.data[v_start:v_end] = 0

                # c_proj consumes concatenated head outputs in its input rows.
                c_proj_mod.weight.data[start:end, :] = 0
    return pruned


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_ds = JsonlDistillDataset(args.data_jsonl, split="train")
    eval_ds = JsonlDistillDataset(args.data_jsonl, split="eval")

    if args.max_train_samples > 0:
        train_ds.samples = train_ds.samples[: args.max_train_samples]
    if args.max_eval_samples > 0:
        eval_ds.samples = eval_ds.samples[: args.max_eval_samples]

    collator = DistillCollator(tokenizer=tokenizer, max_seq_len=args.context_length)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.per_device_batch_size,
        shuffle=True,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=max(1, args.per_device_batch_size // 2),
        shuffle=False,
        collate_fn=collator,
        num_workers=2,
        pin_memory=True,
    )

    student = build_student(tokenizer, args)
    if args.gradient_checkpointing:
        student.gradient_checkpointing_enable()
    if args.enable_qat:
        apply_qat_wrappers(student, bits=args.qat_bits)

    teacher_quant = None
    teacher_dtype = torch.bfloat16 if args.bf16 else torch.float16
    if args.teacher_load_in_4bit:
        teacher_quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
        )
        teacher_dtype = None

    teacher = AutoModelForCausalLM.from_pretrained(
        args.teacher_model,
        trust_remote_code=True,
        device_map="auto" if args.teacher_load_in_4bit else None,
        quantization_config=teacher_quant,
        torch_dtype=teacher_dtype,
    )
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.device_count() > 1:
        print(f"[system] using {torch.cuda.device_count()} GPUs for training")
        student = nn.DataParallel(student)
        if not args.teacher_load_in_4bit:
            teacher = nn.DataParallel(teacher)

    student.to(device)
    if not args.teacher_load_in_4bit:
        teacher.to(device)

    # Use .module when accessing attributes if wrapped in DataParallel
    student_obj = getattr(student, "module", student)
    teacher_obj = getattr(teacher, "module", teacher)

    total_params = parameter_count(student_obj)
    print(f"[model] student parameters: {total_params:,} ({total_params / 1e9:.3f}B)")

    student_layers = student_obj.config.n_layer
    teacher_layers = int(getattr(teacher_obj.config, "num_hidden_layers", student_layers))
    layer_map = choose_layer_map(student_layers, teacher_layers, args.distill_layers)

    teacher_hidden = int(getattr(teacher_obj.config, "hidden_size", args.student_hidden))
    student_hidden = student_obj.config.n_embd
    projectors = nn.ModuleList([nn.Linear(student_hidden, teacher_hidden, bias=False) for _ in layer_map]).to(device)

    optim_params = list(student.parameters()) + list(projectors.parameters())
    optimizer = AdamW(optim_params, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / args.gradient_accumulation_steps)
    total_steps = max(1, int(steps_per_epoch * args.epochs))
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    autocast_dtype = torch.bfloat16 if args.bf16 else None
    autocast_enabled = bool(args.bf16 and device.type == "cuda")
    autocast_device = "cuda" if device.type == "cuda" else "cpu"

    micro_step = 0
    update_step = 0
    running_loss = 0.0
    student.train()

    train_start_time = time.time()
    total_tokens_processed = 0
    step_start_time = time.time()

    stop_training = False
    for epoch in range(math.ceil(args.epochs)):
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=autocast_enabled):
                s_out = student(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    output_attentions=args.loss_weight_attn > 0.0,
                    use_cache=False,
                )

                with torch.no_grad():
                    t_out = teacher(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        output_attentions=args.loss_weight_attn > 0.0,
                        use_cache=False,
                    )

                s_logits, s_labels, valid = shift_for_lm(s_out.logits.float(), labels)
                t_logits, _, _ = shift_for_lm(t_out.logits.float(), labels)

                ce_loss = F.cross_entropy(
                    s_logits.view(-1, s_logits.size(-1)),
                    s_labels.view(-1),
                    ignore_index=-100,
                )

                temp = args.temperature
                kd_loss = masked_mse(s_logits / temp, t_logits / temp, valid)

                hid_loss = torch.zeros((), device=device)
                for i, (s_idx, t_idx) in enumerate(layer_map):
                    s_h = s_out.hidden_states[s_idx + 1]
                    t_h = t_out.hidden_states[t_idx + 1].float()
                    s_h_proj = projectors[i](s_h.float())
                    hid_loss = hid_loss + masked_mse(s_h_proj, t_h, attention_mask.bool())
                hid_loss = hid_loss / len(layer_map)

                attn_loss = torch.zeros((), device=device)
                if args.loss_weight_attn > 0.0:
                    for s_idx, t_idx in layer_map:
                        s_a = s_out.attentions[s_idx].float().mean(dim=1)
                        t_a = t_out.attentions[t_idx].float().mean(dim=1)
                        w = min(args.attn_window, s_a.size(-1), t_a.size(-1))
                        s_local = s_a[:, -w:, -w:]
                        t_local = t_a[:, -w:, -w:]
                        attn_loss = attn_loss + F.mse_loss(s_local, t_local)
                    attn_loss = attn_loss / len(layer_map)

                loss = (
                    args.loss_weight_ce * ce_loss
                    + args.loss_weight_kd * kd_loss
                    + args.loss_weight_hidden * hid_loss
                    + args.loss_weight_attn * attn_loss
                )
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            running_loss += float(loss.item())
            total_tokens_processed += input_ids.numel()

            if (micro_step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(optim_params, max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                update_step += 1

                if update_step % args.logging_steps == 0:
                    lr = scheduler.get_last_lr()[0]
                    elapsed = time.time() - step_start_time
                    tok_per_sec = total_tokens_processed / elapsed if elapsed > 0 else 0
                    print(f"[train] step={update_step} loss={running_loss / args.logging_steps:.4f} lr={lr:.6e} tok/s={tok_per_sec:.1f}")
                    running_loss = 0.0
                    total_tokens_processed = 0
                    step_start_time = time.time()

                if update_step % args.save_steps == 0:
                    ckpt = out_dir / f"checkpoint-step-{update_step}"
                    ckpt.mkdir(parents=True, exist_ok=True)
                    student_obj.save_pretrained(ckpt)
                    tokenizer.save_pretrained(ckpt)
                    torch.save(projectors.state_dict(), ckpt / "distill_projectors.pt")
                    print(f"[save] {ckpt}")

                if update_step >= total_steps:
                    stop_training = True
                    break

            micro_step += 1
        if stop_training:
            break

    if args.prune_heads_ratio > 0.0:
        pruned_heads = prune_gpt2_heads(student_obj, args.prune_heads_ratio)
        print(f"[prune] pruned heads in {len(pruned_heads)} layers")

        if args.post_prune_epochs > 0:
            student.train()
            for _ in range(max(1, int(args.post_prune_epochs))):
                for batch in train_loader:
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)

                    with torch.autocast(device_type=autocast_device, dtype=autocast_dtype, enabled=autocast_enabled):
                        out = student(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            use_cache=False,
                        )
                        logits, shifted_labels, _ = shift_for_lm(out.logits.float(), labels)
                        loss = F.cross_entropy(
                            logits.view(-1, logits.size(-1)),
                            shifted_labels.view(-1),
                            ignore_index=-100,
                        )

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

    ppl_student, ppl_teacher = eval_perplexity(
        student=student,
        teacher=teacher,
        dataloader=eval_loader,
        device=device,
        max_batches=max(1, args.max_eval_samples // max(1, args.per_device_batch_size)),
        use_bf16=args.bf16,
    )

    final_dir = out_dir / "final_student"
    final_dir.mkdir(parents=True, exist_ok=True)
    student_obj.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)

    metrics = {
        "student_params": total_params,
        "student_params_billions": total_params / 1e9,
        "eval_ppl_student": ppl_student,
        "eval_ppl_teacher": ppl_teacher,
        "perplexity_gap_pct": ((ppl_student - ppl_teacher) / max(ppl_teacher, 1e-6)) * 100.0,
    }
    with (final_dir / "distill_metrics.json").open("w", encoding="utf-8") as fout:
        json.dump(metrics, fout, indent=2)

    print(f"[done] student checkpoint: {final_dir}")
    print(f"[eval] student ppl={ppl_student:.3f} teacher ppl={ppl_teacher:.3f} gap={metrics['perplexity_gap_pct']:.2f}%")


if __name__ == "__main__":
    main()
