#!/usr/bin/env python3
"""
Benchmark ARTM quality + speed.

Quality metrics:
- Perplexity (student vs teacher) on held-out JSONL split
- Perplexity gap percentage
- BLEU against teacher responses

Speed metric:
- Tokens/sec using llama-cpp-python on GGUF
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

try:
    from llama_cpp import Llama
except Exception:
    Llama = None


@dataclass
class EvalSample:
    prompt: str
    response: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_hf_dir", type=str, default="/kaggle/working/artm_distilled/final_student")
    parser.add_argument("--teacher_model", type=str, default="microsoft/Phi-3.5-mini-instruct")
    parser.add_argument("--teacher_load_in_4bit", action="store_true")
    parser.add_argument("--eval_jsonl", type=str, default="/kaggle/working/artm_teacher_data.jsonl")
    parser.add_argument("--max_eval_samples", type=int, default=512)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)

    parser.add_argument("--gguf_model_path", type=str, default="")
    parser.add_argument("--n_ctx", type=int, default=2048)
    parser.add_argument("--n_threads", type=int, default=4)
    parser.add_argument("--speed_runs", type=int, default=5)
    parser.add_argument("--speed_max_tokens", type=int, default=128)

    parser.add_argument("--report_json", type=str, default="/kaggle/working/artm_benchmark.json")
    return parser.parse_args()


def load_eval_samples(path: str, limit: int) -> List[EvalSample]:
    rows: List[EvalSample] = []
    with Path(path).open("r", encoding="utf-8") as fin:
        for line in fin:
            obj = json.loads(line)
            if obj.get("split") not in {"eval", None}:
                continue
            prompt = str(obj.get("prompt", "")).strip()
            response = str(obj.get("teacher_response", "")).strip()
            if not prompt or not response:
                continue
            rows.append(EvalSample(prompt=prompt, response=response))
            if len(rows) >= limit:
                break
    if not rows:
        raise RuntimeError("No eval rows found in eval_jsonl")
    return rows


def apply_chat(tokenizer, prompt: str, response: str | None, generation_prompt: bool) -> List[int]:
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        if response is not None:
            messages.append({"role": "assistant", "content": response})
        ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=generation_prompt,
        )
        if isinstance(ids, torch.Tensor):
            return ids.tolist()
        return list(ids)

    if response is None:
        text = f"User: {prompt}\nAssistant:"
    else:
        text = f"User: {prompt}\nAssistant: {response}"
    return tokenizer(text, add_special_tokens=True).input_ids


def build_batch(tokenizer, samples: List[EvalSample], max_len: int = 2048) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    seqs: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []

    for sample in samples:
        prompt_ids = apply_chat(tokenizer, sample.prompt, None, True)
        full_ids = apply_chat(tokenizer, sample.prompt, sample.response, False)
        overflow = max(0, len(full_ids) - max_len)
        if overflow > 0:
            full_ids = full_ids[overflow:]
        prompt_len = max(0, len(prompt_ids) - overflow)
        prompt_len = min(prompt_len, len(full_ids) - 1)

        lab = full_ids.copy()
        for i in range(prompt_len):
            lab[i] = -100

        seqs.append(torch.tensor(full_ids, dtype=torch.long))
        labels.append(torch.tensor(lab, dtype=torch.long))

    mx = max(x.shape[0] for x in seqs)
    input_ids = torch.full((len(seqs), mx), pad_id, dtype=torch.long)
    attn = torch.zeros((len(seqs), mx), dtype=torch.long)
    labs = torch.full((len(seqs), mx), -100, dtype=torch.long)

    for i, (ids, lab) in enumerate(zip(seqs, labels)):
        n = ids.shape[0]
        input_ids[i, :n] = ids
        attn[i, :n] = 1
        labs[i, :n] = lab

    return {"input_ids": input_ids, "attention_mask": attn, "labels": labs}


def shift(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous()


def perplexity(model, tokenizer, samples: List[EvalSample], device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for i in range(0, len(samples), 4):
            batch_rows = samples[i : i + 4]
            batch = build_batch(tokenizer, batch_rows)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            logits, shifted_labels = shift(out.logits.float(), labels)
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                shifted_labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )
            total_loss += float(loss.item())
            total_tokens += int((shifted_labels != -100).sum().item())

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)


def generate_student_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: torch.device) -> str:
    input_ids = apply_chat(tokenizer, prompt, None, True)
    x = torch.tensor([input_ids], dtype=torch.long, device=device)
    attn = torch.ones_like(x)

    with torch.no_grad():
        out = model.generate(
            input_ids=x,
            attention_mask=attn,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-6),
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    gen = out[0, x.shape[1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


def speed_benchmark_gguf(
    gguf_path: str,
    n_ctx: int,
    n_threads: int,
    runs: int,
    max_tokens: int,
) -> Dict[str, float] | None:
    if not gguf_path:
        return None
    if Llama is None:
        print("[warn] llama_cpp is unavailable, skipping GGUF speed benchmark")
        return None

    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        verbose=False,
    )

    prompt = "<|user|>\nExplain edge inference in 3 lines.\n<|assistant|>\n"

    _ = llm(prompt, max_tokens=min(32, max_tokens), temperature=0.0)

    rates: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        out = llm(prompt, max_tokens=max_tokens, temperature=0.0)
        elapsed = max(time.perf_counter() - start, 1e-6)
        usage = out.get("usage", {})
        toks = usage.get("completion_tokens")
        if toks is None:
            txt = out["choices"][0]["text"]
            toks = len(llm.tokenize(txt.encode("utf-8"), add_bos=False))
        rates.append(float(toks) / elapsed)

    return {
        "mean_tok_s": float(statistics.mean(rates)),
        "median_tok_s": float(statistics.median(rates)),
        "min_tok_s": float(min(rates)),
        "max_tok_s": float(max(rates)),
    }


def main() -> None:
    args = parse_args()
    samples = load_eval_samples(args.eval_jsonl, args.max_eval_samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.student_hf_dir, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    student = AutoModelForCausalLM.from_pretrained(args.student_hf_dir, trust_remote_code=True).to(device)
    teacher_kwargs = {
        "trust_remote_code": True,
    }
    if args.teacher_load_in_4bit:
        teacher_kwargs["device_map"] = "auto"
        teacher_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    else:
        teacher_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32

    teacher = AutoModelForCausalLM.from_pretrained(args.teacher_model, **teacher_kwargs)
    if not args.teacher_load_in_4bit:
        teacher = teacher.to(device)

    ppl_student = perplexity(student, tokenizer, samples, device)
    ppl_teacher = perplexity(teacher, tokenizer, samples, device)
    ppl_gap_pct = ((ppl_student - ppl_teacher) / max(ppl_teacher, 1e-6)) * 100.0

    preds: List[str] = []
    refs: List[str] = []
    for sample in samples:
        pred = generate_student_text(
            model=student,
            tokenizer=tokenizer,
            prompt=sample.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            device=device,
        )
        preds.append(pred)
        refs.append(sample.response)

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score

    speed = speed_benchmark_gguf(
        gguf_path=args.gguf_model_path,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        runs=args.speed_runs,
        max_tokens=args.speed_max_tokens,
    )

    gguf_size_gb = None
    if args.gguf_model_path and Path(args.gguf_model_path).exists():
        gguf_size_gb = Path(args.gguf_model_path).stat().st_size / (1024**3)

    report = {
        "eval_samples": len(samples),
        "perplexity_student": ppl_student,
        "perplexity_teacher": ppl_teacher,
        "perplexity_gap_pct": ppl_gap_pct,
        "bleu": bleu,
        "gguf_size_gb": gguf_size_gb,
        "speed": speed,
    }

    out_path = Path(args.report_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fout:
        json.dump(report, fout, indent=2)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
