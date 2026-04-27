"""
ARTM Phase 1 Training Script (ARTM-10M setup)

General-purpose assistant decoder-only Transformer with reasoning/emotion/math data mixing.
Default configuration follows the requested ARTM-10M shape.
Designed for quantization-friendly deployment later.

Includes a minimal generate() helper for quick post-training checks.
"""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import nullcontext
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, Dataset, IterableDataset


# ============================================================
# CONFIG
# ============================================================


@dataclass
class ARTMConfig:
    vocab_size: int
    max_seq_len: int = 384
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 10
    ffn_mult: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    rope_base: float = 10000.0
    gradient_checkpointing: bool = True
    label_smoothing: float = 0.05
    pad_token_id: int = 0


# ============================================================
# TOKENIZER
# ============================================================


class MathCharTokenizer:
    PAD = "<pad>"
    BOS = "<bos>"
    EOS = "<eos>"
    UNK = "<unk>"
    TOKEN_PATTERN = re.compile(r"\d+|[A-Za-z]+|[^\w\s]|\s+")

    def __init__(self, vocab_tokens: Optional[List[str]] = None):
        if vocab_tokens is None:
            vocab_tokens = self._base_tokens()

        seen = set()
        ordered_tokens: List[str] = []
        for tok in vocab_tokens:
            if tok not in seen:
                seen.add(tok)
                ordered_tokens.append(tok)

        self.special_tokens = [self.PAD, self.BOS, self.EOS, self.UNK]
        self.itos = self.special_tokens + ordered_tokens
        self.stoi = {token: idx for idx, token in enumerate(self.itos)}

        self.pad_id = self.stoi[self.PAD]
        self.bos_id = self.stoi[self.BOS]
        self.eos_id = self.stoi[self.EOS]
        self.unk_id = self.stoi[self.UNK]

    @classmethod
    def _tokenize_text(cls, text: str) -> List[str]:
        return cls.TOKEN_PATTERN.findall(text)

    @classmethod
    def _base_tokens(cls) -> List[str]:
        # Keep character fallback tokens so unseen pieces can still be decomposed.
        base_chars = [chr(i) for i in range(32, 127)] + ["\n", "\t"]
        math_tokens = [
            "Question",
            "Answer",
            "Step",
            "If",
            "what",
            "is",
            "x",
            "y",
            "+",
            "-",
            "*",
            "/",
            "^",
            "=",
            "(",
            ")",
        ]
        number_tokens = [str(i) for i in range(10)]
        return number_tokens + math_tokens + base_chars

    @classmethod
    def build_from_texts(
        cls,
        texts: List[str],
        target_vocab_size: int = 4096,
        min_freq: int = 2,
    ) -> "MathCharTokenizer":
        if target_vocab_size < 256:
            raise ValueError("target_vocab_size should be >= 256 for this tokenizer")

        token_counts: Counter[str] = Counter()
        for text in texts:
            token_counts.update(cls._tokenize_text(text))

        base_tokens = cls._base_tokens()
        base_set = set(base_tokens)
        extra_tokens = [tok for tok, freq in token_counts.most_common() if freq >= min_freq and tok not in base_set]

        max_non_special = max(1, target_vocab_size - 4)
        vocab_tokens = (base_tokens + extra_tokens)[:max_non_special]

        # Pad vocabulary so embedding matrix reaches the requested size.
        extra_id = 0
        while len(vocab_tokens) < max_non_special:
            synthetic_token = f"<extra_{extra_id}>"
            extra_id += 1
            if synthetic_token not in base_set:
                vocab_tokens.append(synthetic_token)

        return cls(vocab_tokens=vocab_tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        for tok in self._tokenize_text(text):
            idx = self.stoi.get(tok)
            if idx is not None:
                ids.append(idx)
                continue
            # Fallback: decompose unknown token into characters.
            for ch in tok:
                ids.append(self.stoi.get(ch, self.unk_id))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: List[int], skip_special: bool = True) -> str:
        chars: List[str] = []
        for idx in ids:
            if idx < 0 or idx >= len(self.itos):
                continue
            token = self.itos[idx]
            if skip_special and token in self.special_tokens:
                continue
            chars.append(token)
        return "".join(chars)

    def to_dict(self) -> Dict[str, object]:
        return {
            "itos": self.itos,
            "special_tokens": self.special_tokens,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "MathCharTokenizer":
        obj = cls(vocab_tokens=[])
        obj.itos = list(data["itos"])
        obj.special_tokens = list(data["special_tokens"])
        obj.stoi = {token: idx for idx, token in enumerate(obj.itos)}
        obj.pad_id = obj.stoi[obj.PAD]
        obj.bos_id = obj.stoi[obj.BOS]
        obj.eos_id = obj.stoi[obj.EOS]
        obj.unk_id = obj.stoi[obj.UNK]
        return obj

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")


# ============================================================
# MODEL
# ============================================================


def build_rope_cache(
    seq_len: int,
    head_dim: int,
    base: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even for RoPE")

    inv_freq = 1.0 / (
        base ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    positions = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cos(freqs).to(dtype=dtype)
    sin = torch.sin(freqs).to(dtype=dtype)
    return cos, sin


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x_even = x[..., ::2]
    x_odd = x[..., 1::2]

    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    rot_even = x_even * cos - x_odd * sin
    rot_odd = x_even * sin + x_odd * cos
    return torch.stack((rot_even, rot_odd), dim=-1).flatten(-2)


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ARTMConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.n_heads = cfg.n_heads
        self.head_dim = cfg.d_model // cfg.n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=True)
        self.out_proj = nn.Linear(cfg.d_model, cfg.d_model, bias=True)
        self.attn_dropout_p = cfg.dropout
        self.resid_dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, dim = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(bsz, seqlen, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rope(q, rope_cos, rope_sin)
        k = apply_rope(k, rope_cos, rope_sin)

        if hasattr(F, "scaled_dot_product_attention"):
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.attn_dropout_p if self.training else 0.0,
                is_causal=True,
            )
        else:
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            mask = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=x.device))
            scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)

        out = out.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.resid_dropout(self.out_proj(out))


class FeedForward(nn.Module):
    def __init__(self, cfg: ARTMConfig):
        super().__init__()
        hidden = cfg.d_model * cfg.ffn_mult
        self.fc1 = nn.Linear(cfg.d_model, hidden, bias=True)
        self.fc2 = nn.Linear(hidden, cfg.d_model, bias=True)
        self.dropout = nn.Dropout(cfg.dropout)
        if cfg.activation == "relu":
            self.act_fn = F.relu
        elif cfg.activation == "gelu":
            self.act_fn = F.gelu
        else:
            raise ValueError("activation must be either 'relu' or 'gelu'")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act_fn(self.fc1(x))))


class TransformerBlock(nn.Module):
    def __init__(self, cfg: ARTMConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg)

    def forward(self, x: torch.Tensor, rope_cos: torch.Tensor, rope_sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), rope_cos, rope_sin)
        x = x + self.ffn(self.ln2(x))
        return x


class ARTM(nn.Module):
    def __init__(self, cfg: ARTMConfig):
        super().__init__()
        self.cfg = cfg

        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.emb_dropout = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.gradient_checkpointing = cfg.gradient_checkpointing
        self.rope_cache: Dict[Tuple[int, torch.device, torch.dtype], Tuple[torch.Tensor, torch.Tensor]] = {}

        self.apply(self._init_weights)
        self.tie_weights()

    def tie_weights(self) -> None:
        if self.lm_head.weight.shape != self.token_emb.weight.shape:
            raise ValueError("lm_head and token_emb shapes are incompatible for weight tying")
        self.lm_head.weight = self.token_emb.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        _, seqlen = input_ids.shape
        if seqlen > self.cfg.max_seq_len:
            raise ValueError(f"Sequence length {seqlen} exceeds max_seq_len {self.cfg.max_seq_len}")

        x = self.emb_dropout(self.token_emb(input_ids))
        head_dim = self.cfg.d_model // self.cfg.n_heads
        rope_key = (seqlen, x.device, x.dtype)
        if rope_key not in self.rope_cache:
            self.rope_cache[rope_key] = build_rope_cache(
                seq_len=seqlen,
                head_dim=head_dim,
                base=self.cfg.rope_base,
                device=input_ids.device,
                dtype=x.dtype,
            )
        rope_cos, rope_sin = self.rope_cache[rope_key]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(block, x, rope_cos, rope_sin)
            else:
                x = block(x, rope_cos, rope_sin)

        x = self.ln_f(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                ignore_index=self.cfg.pad_token_id,
                label_smoothing=self.cfg.label_smoothing,
            )

        return logits, loss


def count_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in unwrap_model(model).parameters())


def unwrap_model(model: nn.Module) -> ARTM:
    return model._orig_mod if hasattr(model, "_orig_mod") else model


# ============================================================
# DATASET LOADING
# ============================================================


def format_math_example(question: str, steps: List[str], answer: str) -> str:
    lines = [f"Question: {question}"]
    for i, step in enumerate(steps, start=1):
        lines.append(f"Step {i}: {step}")
    lines.append(f"Answer: {answer}")
    return "\n".join(lines)


GENERAL_SYSTEM_PROMPT = (
    "You are a helpful, intelligent, and emotionally aware assistant. "
    "You respond clearly, step-by-step when needed, and adapt tone to the user."
)
MATH_SYSTEM_PROMPT = "You are a precise reasoning model. Always solve step-by-step."
EMOTION_SYSTEM_PROMPT = (
    "You are a supportive and empathetic assistant. You validate feelings and respond calmly."
)

DATASET_GROUP_WEIGHTS: Dict[str, float] = {
    "gsm8k": 0.35,
    "hendrycks_math": 0.23,
    "svamp_mawps": 0.15,
    "wikipedia": 0.08,
    "openwebtext": 0.04,
    "the_stack": 0.05,
    "instruction_chat": 0.10,
}

HENDRYCKS_MATH_CONFIGS: List[str] = [
    "algebra",
    "counting_and_probability",
    "geometry",
]


def normalize_sample(user_text: str, assistant_text: str, mode: str) -> str:
    user_text = _to_clean_text(user_text)
    assistant_text = _to_clean_text(assistant_text)
    if not user_text or not assistant_text:
        return ""

    if mode == "math":
        system_prompt = MATH_SYSTEM_PROMPT
        if "Step 1:" not in assistant_text:
            assistant_text = f"Step 1: Solve carefully.\nFinal Answer: {assistant_text}"
    elif mode == "emotion":
        system_prompt = EMOTION_SYSTEM_PROMPT
    else:
        system_prompt = GENERAL_SYSTEM_PROMPT

    return (
        "<system>\n"
        f"{system_prompt}\n\n"
        "<user>\n"
        f"{user_text}\n\n"
        "<assistant>\n"
        f"{assistant_text}"
    )


def _record_to_text(record: Dict[str, object]) -> str:
    text = record.get("text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    question = record.get("question")
    answer = record.get("answer") or record.get("final_answer")
    steps = record.get("steps") or record.get("reasoning") or record.get("solution_steps")

    if isinstance(question, str) and question.strip() and isinstance(answer, str) and answer.strip():
        if isinstance(steps, list):
            clean_steps = [str(s).strip() for s in steps if str(s).strip()]
        elif isinstance(steps, str):
            clean_steps = [line.strip() for line in steps.split("\n") if line.strip()]
        else:
            clean_steps = []

        if not clean_steps:
            clean_steps = ["Solve carefully."]

        return format_math_example(question.strip(), clean_steps, answer.strip())

    raise ValueError(
        "Each JSON record must include either 'text' OR ('question', 'steps' or 'reasoning', 'answer')."
    )


def _to_clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _pick_first_text(record: Dict[str, Any], keys: List[str]) -> str:
    for key in keys:
        if key in record:
            text = _to_clean_text(record[key])
            if text:
                return text
    return ""


def _split_reasoning_and_final(answer_text: str) -> Tuple[List[str], str]:
    text = _to_clean_text(answer_text)
    if not text:
        return [], ""

    separators = ["####", "Final Answer:", "final answer:", "Answer:", "answer:"]
    for sep in separators:
        if sep in text:
            reasoning, final = text.rsplit(sep, 1)
            steps = [line.strip() for line in reasoning.replace("\r", "").split("\n") if line.strip()]
            return steps, final.strip()

    lines = [line.strip() for line in text.replace("\r", "").split("\n") if line.strip()]
    if len(lines) > 1:
        return lines[:-1], lines[-1]
    return [], text


def _extract_boxed_answer(text: str) -> str:
    # Keep this lightweight and robust for MATH-style \boxed{...} outputs.
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return ""


def _distill_reasoning_steps(steps: List[str], max_steps: int = 4) -> List[str]:
    cleaned: List[str] = []
    for raw in steps:
        line = _to_clean_text(raw)
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        line = line.replace("Therefore,", "So,").replace("Hence,", "So,")
        line = line.strip(" -")
        if line:
            cleaned.append(line)
        if len(cleaned) >= max_steps:
            break
    return cleaned


def _format_reasoning_answer(steps: List[str], final_answer: str) -> str:
    lines: List[str] = []
    for i, step in enumerate(steps, start=1):
        lines.append(f"Step {i}: {step}")
    lines.append(f"Final Answer: {final_answer}")
    return "\n".join(lines)


def _hf_record_to_text(dataset_name: str, record: Dict[str, Any]) -> Optional[str]:
    if dataset_name == "openai/gsm8k":
        question = _pick_first_text(record, ["question", "Question"])
        raw_answer = _pick_first_text(record, ["answer", "Answer"])
        if not question:
            raise ValueError("Missing question in GSM8K record")

        steps, final = _split_reasoning_and_final(raw_answer)
        if not steps:
            steps = ["Solve carefully, step by step."]
        if not final:
            final = raw_answer if raw_answer else "Unknown"
        return format_math_example(question, steps, final)

    if dataset_name == "ChilleD/SVAMP":
        body = _pick_first_text(record, ["Body", "body", "context"])
        question_part = _pick_first_text(record, ["Question", "question", "problem"])
        question = f"{body} {question_part}".strip() if body and question_part else (question_part or body)
        answer = _pick_first_text(record, ["Answer", "answer", "final_answer"])
        equation = _pick_first_text(record, ["Equation", "equation"])

        if not question or not answer:
            raise ValueError("Missing question/answer in SVAMP record")

        steps = [f"Form equation: {equation}"] if equation else []
        steps.append("Compute step by step.")
        return format_math_example(question, steps, answer)

    if dataset_name in ("hendrycks/competition_math", "EleutherAI/hendrycks_math"):
        question = _pick_first_text(record, ["problem", "question"])
        solution = _pick_first_text(record, ["solution", "answer"])
        if not question or not solution:
            raise ValueError("Missing problem/solution in Hendrycks MATH record")

        steps, final = _split_reasoning_and_final(solution)
        boxed = _extract_boxed_answer(solution)
        if boxed:
            final = boxed
        if not final:
            final = "Unknown"
        steps = _distill_reasoning_steps(steps, max_steps=4)
        if not steps:
            steps = ["Break the problem into smaller math steps."]
        return format_math_example(question, steps, final)

    if dataset_name in ("mu-nlpc/mawps", "mwpt5/MAWPS"):
        body = _pick_first_text(record, ["Body", "body", "context"])
        question_part = _pick_first_text(record, ["Question", "question", "sQuestion", "problem"])
        question = f"{body} {question_part}".strip() if body and question_part else (question_part or body)
        answer = _pick_first_text(record, ["Answer", "answer", "final_answer", "lSolutions", "solution"])
        equation = _pick_first_text(record, ["Equation", "equation", "template", "equation_template"])
        if not question or not answer:
            raise ValueError("Missing question/answer in MAWPS record")
        steps = [f"Form equation: {equation}"] if equation else ["Translate words into arithmetic."]
        return format_math_example(question, steps, str(answer))

    return _record_to_text(record)


def _load_hf_split_any(
    dataset_name: str,
    config_name: Optional[str],
    cache_dir: Optional[str],
):
    from datasets import load_dataset

    preferred_splits = ("train", "validation", "test")

    for split in preferred_splits:
        try:
            dataset_split = load_dataset(
                dataset_name,
                config_name,
                split=split,
                cache_dir=cache_dir,
            )
            return dataset_split, split
        except Exception:
            pass

    dataset_obj = load_dataset(dataset_name, config_name, cache_dir=cache_dir)
    if hasattr(dataset_obj, "keys"):
        for split in preferred_splits:
            if split in dataset_obj:
                return dataset_obj[split], split
        first_split = next(iter(dataset_obj.keys()))
        return dataset_obj[first_split], first_split

    return dataset_obj, "all"


def load_hf_math_texts(
    max_samples_per_dataset: int = 0,
    cache_dir: Optional[str] = None,
) -> List[str]:
    try:
        import datasets  # noqa: F401
    except Exception as exc:
        raise ImportError(
            "Hugging Face datasets is not installed. Install with: pip install datasets"
        ) from exc

    sources = [
        ("openai/gsm8k", "main"),
        ("openai/gsm8k", "socratic"),
        ("EleutherAI/hendrycks_math", "algebra"),
        ("EleutherAI/hendrycks_math", "counting_and_probability"),
        ("EleutherAI/hendrycks_math", "geometry"),
        ("ChilleD/SVAMP", None),
    ]

    combined_texts: List[str] = []
    for dataset_name, config_name in sources:
        split_dataset, split_name = _load_hf_split_any(dataset_name, config_name, cache_dir)
        source_texts: List[str] = []

        for i, record in enumerate(split_dataset):
            if max_samples_per_dataset > 0 and i >= max_samples_per_dataset:
                break
            if not isinstance(record, dict):
                continue
            text = _hf_record_to_text(dataset_name, record)
            if text:
                source_texts.append(text)

        if not source_texts:
            raise ValueError(f"No usable samples in {dataset_name} ({config_name or 'default'})")

        combined_texts.extend(source_texts)
        print(
            f"Loaded {len(source_texts)} samples from {dataset_name}"
            f"{'/' + config_name if config_name else ''} ({split_name})."
        )

    return combined_texts


def _cap_samples(samples: List[str], max_samples: int, seed: int) -> List[str]:
    if max_samples <= 0 or len(samples) <= max_samples:
        return samples
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)
    return [samples[i] for i in idx[:max_samples]]


def _as_list_of_strings(value: Any) -> List[str]:
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _extract_pairs_from_conversation(messages: Any) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    if not isinstance(messages, list) or len(messages) < 2:
        return pairs

    if all(isinstance(item, str) for item in messages):
        for i in range(0, len(messages) - 1, 2):
            user = messages[i].strip()
            assistant = messages[i + 1].strip()
            if user and assistant:
                pairs.append((user, assistant))
        return pairs

    normalized: List[Tuple[str, str]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = _pick_first_text(msg, ["role", "from", "speaker", "author"]).lower()
        content = _pick_first_text(msg, ["content", "text", "value", "message"])
        if content:
            normalized.append((role, content))

    for i in range(len(normalized) - 1):
        role_a, text_a = normalized[i]
        role_b, text_b = normalized[i + 1]
        if role_a in ("user", "human", "prompter") and role_b in ("assistant", "gpt", "bot", "model"):
            pairs.append((text_a, text_b))
    return pairs


def _extract_user_assistant_from_math_text(text: str) -> Tuple[str, str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return "", ""
    if lines[0].startswith("Question:"):
        user = lines[0].replace("Question:", "", 1).strip()
        assistant = "\n".join(lines[1:]).strip()
        return user, assistant
    return "", text.strip()


def _normalize_math_pair(question: str, steps: List[str], final_answer: str) -> str:
    q = _to_clean_text(question)
    final = _to_clean_text(final_answer)
    if not q or not final:
        return ""
    distilled_steps = _distill_reasoning_steps(steps, max_steps=4)
    if not distilled_steps:
        distilled_steps = ["Solve this carefully, one step at a time."]
    user_text = f"Q: {q}\nA:"
    assistant_text = _format_reasoning_answer(distilled_steps, final)
    return normalize_sample(user_text, assistant_text, mode="math")


def _load_math_component_texts(
    dataset_name: str,
    config_name: Optional[str],
    max_samples: int,
    cache_dir: Optional[str],
    seed: int,
) -> List[str]:
    split_dataset, split_name = _load_hf_split_any(dataset_name, config_name, cache_dir)
    local: List[str] = []

    for i, record in enumerate(split_dataset):
        if max_samples > 0 and i >= max_samples:
            break
        if not isinstance(record, dict):
            continue

        if dataset_name == "openai/gsm8k":
            question = _pick_first_text(record, ["question", "Question"])
            answer = _pick_first_text(record, ["answer", "Answer"])
            steps, final = _split_reasoning_and_final(answer)
            text = _normalize_math_pair(question, steps, final or answer)
        elif dataset_name in ("hendrycks/competition_math", "EleutherAI/hendrycks_math"):
            question = _pick_first_text(record, ["problem", "question"])
            solution = _pick_first_text(record, ["solution", "answer"])
            steps, final = _split_reasoning_and_final(solution)
            boxed = _extract_boxed_answer(solution)
            if boxed:
                final = boxed
            text = _normalize_math_pair(question, steps, final)
        elif dataset_name == "ChilleD/SVAMP":
            body = _pick_first_text(record, ["Body", "body", "context"])
            q_part = _pick_first_text(record, ["Question", "question", "problem"])
            question = f"{body} {q_part}".strip() if body and q_part else (q_part or body)
            answer = _pick_first_text(record, ["Answer", "answer", "final_answer"])
            equation = _pick_first_text(record, ["Equation", "equation"])
            steps = [f"Form equation: {equation}"] if equation else ["Translate to arithmetic."]
            text = _normalize_math_pair(question, steps, answer)
        elif dataset_name in ("mu-nlpc/mawps", "mwpt5/MAWPS"):
            body = _pick_first_text(record, ["Body", "body", "context"])
            q_part = _pick_first_text(record, ["Question", "question", "sQuestion", "problem"])
            question = f"{body} {q_part}".strip() if body and q_part else (q_part or body)
            answer = _pick_first_text(record, ["Answer", "answer", "final_answer", "lSolutions", "solution"])
            equation = _pick_first_text(record, ["Equation", "equation", "template", "equation_template"])
            steps = [f"Form equation: {equation}"] if equation else ["Translate words into arithmetic."]
            text = _normalize_math_pair(question, steps, str(answer))
        else:
            text = ""

        if text:
            local.append(text)

    print(
        f"Loaded {len(local)} samples from {dataset_name}"
        f"{'/' + config_name if config_name else ''} ({split_name})."
    )
    return _cap_samples(local, max_samples, seed)


def _load_text_component_texts(
    dataset_name: str,
    config_name: Optional[str],
    max_samples: int,
    cache_dir: Optional[str],
    seed: int,
    source_key_candidates: List[str],
) -> List[str]:
    candidates: List[Tuple[str, Optional[str], List[str]]] = [(dataset_name, config_name, source_key_candidates)]
    if dataset_name == "wikipedia":
        # 'wikipedia' is script-based and may fail on newer datasets versions.
        candidates = [
            ("wikimedia/wikipedia", "20231101.en", ["text"]),
            ("wikimedia/wikipedia", "20220301.en", ["text"]),
            ("wikipedia", "20220301.en", ["text"]),
        ]
    elif dataset_name == "Skylion007/openwebtext":
        candidates = [
            ("Skylion007/openwebtext", None, ["text", "content"]),
            ("stas/openwebtext-10k", None, ["text", "content"]),
            # Last-resort general text fallback to avoid full run failure.
            ("roneneldan/TinyStories", None, ["text"]),
        ]

    for cand_name, cand_cfg, cand_keys in candidates:
        try:
            split_dataset, split_name = _load_hf_split_any(cand_name, cand_cfg, cache_dir)
            local: List[str] = []
            for i, record in enumerate(split_dataset):
                if max_samples > 0 and i >= max_samples:
                    break
                if not isinstance(record, dict):
                    continue
                text = _pick_first_text(record, cand_keys)
                if not text:
                    continue
                user_text = "Write a concise explanation in clear natural language."
                assistant_text = text
                normalized = normalize_sample(user_text, assistant_text, mode="general")
                if normalized:
                    local.append(normalized)

            print(
                f"Loaded {len(local)} samples from {cand_name}"
                f"{'/' + cand_cfg if cand_cfg else ''} ({split_name})."
            )
            return _cap_samples(local, max_samples, seed)
        except Exception as exc:
            print(
                f"Warning: failed to load {cand_name}"
                f"{'/' + cand_cfg if cand_cfg else ''} ({exc})"
            )

    raise ValueError(f"Unable to load text dataset from candidates for: {dataset_name}")


def _load_instruction_chat_component_texts(
    max_samples: int,
    cache_dir: Optional[str],
    seed: int,
) -> List[str]:
    # Real conversations first; instruction-style FLAN as fallback.
    try:
        split, _ = _load_hf_split_any("OpenAssistant/oasst1", None, cache_dir)
        id_to_record: Dict[str, Dict[str, str]] = {}
        for record in split:
            if not isinstance(record, dict):
                continue
            lang = _pick_first_text(record, ["lang", "language"]).lower()
            if lang and lang != "en":
                continue
            msg_id = _pick_first_text(record, ["message_id", "id"])
            if not msg_id:
                continue
            id_to_record[msg_id] = {
                "parent_id": _pick_first_text(record, ["parent_id"]),
                "role": _pick_first_text(record, ["role"]).lower(),
                "text": _pick_first_text(record, ["text", "message"]),
            }

        local: List[str] = []
        for rec in id_to_record.values():
            if rec["role"] not in ("assistant", "assistant_reply", "bot"):
                continue
            parent = id_to_record.get(rec["parent_id"], {})
            user_text = _to_clean_text(parent.get("text", ""))
            assistant_text = _to_clean_text(rec.get("text", ""))
            normalized = normalize_sample(user_text, assistant_text, mode="general")
            if normalized:
                local.append(normalized)

        print(f"Loaded {len(local)} samples from OpenAssistant/oasst1.")
        return _cap_samples(local, max_samples, seed)
    except Exception as exc:
        print(f"Warning: failed to load OpenAssistant/oasst1 ({exc})")

    try:
        split, split_name = _load_hf_split_any("Muennighoff/flan", None, cache_dir)
        local = []
        for i, record in enumerate(split):
            if max_samples > 0 and i >= max_samples:
                break
            if not isinstance(record, dict):
                continue
            instruction = _pick_first_text(record, ["instruction", "inputs", "input", "question"])
            response = _pick_first_text(record, ["targets", "target", "output", "answer"])
            normalized = normalize_sample(instruction, response, mode="general")
            if normalized:
                local.append(normalized)

        print(f"Loaded {len(local)} samples from Muennighoff/flan ({split_name}).")
        return _cap_samples(local, max_samples, seed)
    except Exception as exc:
        print(f"Warning: failed to load Muennighoff/flan ({exc})")

    raise ValueError("Unable to load OpenAssistant/oasst1 or FLAN instruction data.")


def _load_code_component_texts(
    max_samples: int,
    cache_dir: Optional[str],
    seed: int,
) -> List[str]:
    # Prefer The Stack; keep a fallback so training still works without special access.
    candidates: List[Tuple[str, Optional[str], Dict[str, Any]]] = [
        ("bigcode/the-stack", None, {"data_dir": "python"}),
        ("codeparrot/github-code-clean", None, {}),
    ]

    from datasets import load_dataset

    for dataset_name, config_name, extra_kwargs in candidates:
        try:
            split = load_dataset(
                dataset_name,
                config_name,
                split="train",
                cache_dir=cache_dir,
                streaming=True,
                **extra_kwargs,
            )
            local: List[str] = []
            for i, record in enumerate(split):
                if max_samples > 0 and i >= max_samples:
                    break
                if not isinstance(record, dict):
                    continue
                code = _pick_first_text(record, ["content", "code", "text"])
                if not code:
                    continue
                normalized = normalize_sample(
                    "Explain the logic in this code briefly.",
                    code,
                    mode="general",
                )
                if normalized:
                    local.append(normalized)

            print(f"Loaded {len(local)} samples from {dataset_name} (train/streaming).")
            return _cap_samples(local, max_samples, seed)
        except Exception as exc:
            print(f"Warning: failed to load {dataset_name} ({exc})")

    raise ValueError("Unable to load The Stack or fallback code dataset.")


def mix_grouped_samples(
    grouped_texts: Dict[str, List[str]],
    seed: int,
    total_samples: int = 0,
) -> List[str]:
    required_groups = list(DATASET_GROUP_WEIGHTS.keys())
    missing = [g for g in required_groups if len(grouped_texts.get(g, [])) == 0]
    if missing:
        raise ValueError(f"Missing required dataset groups: {missing}")

    rng = random.Random(seed)
    weights_sum = sum(DATASET_GROUP_WEIGHTS.values())

    if total_samples <= 0:
        capacity = min(len(grouped_texts[g]) / DATASET_GROUP_WEIGHTS[g] for g in required_groups)
        total_samples = int(capacity * weights_sum)

    total_samples = max(total_samples, 1)
    mixed: List[str] = []

    for group in required_groups:
        group_weight = DATASET_GROUP_WEIGHTS[group] / weights_sum
        target_count = max(1, int(round(total_samples * group_weight)))
        pool = grouped_texts[group]
        if target_count <= len(pool):
            samples = rng.sample(pool, target_count)
        else:
            samples = [pool[rng.randrange(len(pool))] for _ in range(target_count)]
        mixed.extend(samples)

    rng.shuffle(mixed)
    return mixed


def load_hf_general_stack_texts(
    max_samples_per_dataset: int,
    cache_dir: Optional[str],
    seed: int,
    mixed_total_samples: int = 0,
) -> Tuple[List[str], Dict[str, int]]:
    per_dataset_cap = max_samples_per_dataset if max_samples_per_dataset > 0 else 60000
    per_hendrycks_config_cap = max(1, per_dataset_cap // max(1, len(HENDRYCKS_MATH_CONFIGS)))

    mawps: List[str] = []
    try:
        mawps = _load_math_component_texts(
            dataset_name="mu-nlpc/mawps",
            config_name=None,
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed + 4,
        )
    except Exception:
        mawps = _load_math_component_texts(
            dataset_name="mwpt5/MAWPS",
            config_name=None,
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed + 4,
        )

    hendrycks_math: List[str] = []
    for i, cfg_name in enumerate(HENDRYCKS_MATH_CONFIGS):
        try:
            hendrycks_math.extend(
                _load_math_component_texts(
                    dataset_name="EleutherAI/hendrycks_math",
                    config_name=cfg_name,
                    max_samples=per_hendrycks_config_cap,
                    cache_dir=cache_dir,
                    seed=seed + 1 + i,
                )
            )
        except Exception as exc:
            print(f"Warning: failed to load EleutherAI/hendrycks_math/{cfg_name} ({exc})")
    if not hendrycks_math:
        raise ValueError("Failed to load any EleutherAI/hendrycks_math configs.")

    grouped: Dict[str, List[str]] = {
        "gsm8k": _load_math_component_texts(
            dataset_name="openai/gsm8k",
            config_name="main",
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed,
        ),
        "hendrycks_math": hendrycks_math,
        "svamp_mawps": _load_math_component_texts(
            dataset_name="ChilleD/SVAMP",
            config_name=None,
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed + 2,
        ) + mawps,
        "wikipedia": _load_text_component_texts(
            dataset_name="wikipedia",
            config_name="20220301.en",
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed + 10,
            source_key_candidates=["text"],
        ),
        "openwebtext": _load_text_component_texts(
            dataset_name="Skylion007/openwebtext",
            config_name=None,
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed + 11,
            source_key_candidates=["text", "content"],
        ),
        "the_stack": _load_code_component_texts(
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed + 12,
        ),
        "instruction_chat": _load_instruction_chat_component_texts(
            max_samples=per_dataset_cap,
            cache_dir=cache_dir,
            seed=seed + 13,
        ),
    }

    counts = {group: len(samples) for group, samples in grouped.items()}
    mixed = mix_grouped_samples(grouped, seed=seed, total_samples=mixed_total_samples)
    return mixed, counts


def generate_synthetic_linear_equation_texts(
    num_samples: int,
    seed: int,
    x_min: int = 1,
    x_max: int = 80,
    a_min: int = 1,
    a_max: int = 20,
    b_min: int = 1,
    b_max: int = 40,
) -> List[str]:
    rng = random.Random(seed)
    texts: List[str] = []

    for _ in range(num_samples):
        x = rng.randint(x_min, x_max)
        a = rng.randint(a_min, a_max)
        b = rng.randint(b_min, b_max)
        result = a * x + b

        question = f"If {a}x + {b} = {result}, what is x?"
        texts.append(f"Question: {question}\nAnswer: x = {x}")

    return texts


def load_math_texts(dataset_path: str | Path) -> List[str]:
    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    texts: List[str] = []

    if path.suffix.lower() == ".jsonl":
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError("Each JSONL line must be an object")
            texts.append(_record_to_text(record))

    elif path.suffix.lower() == ".json":
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            raise ValueError("JSON dataset must be a list of objects")
        for record in raw:
            if not isinstance(record, dict):
                raise ValueError("Each JSON item must be an object")
            texts.append(_record_to_text(record))

    elif path.suffix.lower() == ".txt":
        # Expect each sample as a block separated by a blank line.
        blocks = [b.strip() for b in path.read_text(encoding="utf-8").split("\n\n") if b.strip()]
        texts.extend(blocks)

    else:
        raise ValueError("Unsupported dataset format. Use .jsonl, .json, or .txt")

    if not texts:
        raise ValueError("No training examples found in dataset")

    return texts


def split_train_val(texts: List[str], val_fraction: float, seed: int) -> Tuple[List[str], List[str]]:
    if not 0.0 <= val_fraction < 1.0:
        raise ValueError("val_fraction must be in [0.0, 1.0)")

    if val_fraction == 0.0 or len(texts) < 2:
        return texts, []

    idx = list(range(len(texts)))
    rng = random.Random(seed)
    rng.shuffle(idx)

    val_size = max(1, int(len(texts) * val_fraction))
    val_idx = set(idx[:val_size])

    train_texts = [texts[i] for i in range(len(texts)) if i not in val_idx]
    val_texts = [texts[i] for i in range(len(texts)) if i in val_idx]

    return train_texts, val_texts


# ============================================================
# TOKEN STREAM DATASET (teacher forcing)
# ============================================================


class PackedCausalDataset(Dataset):
    def __init__(self, token_stream: torch.Tensor, seq_len: int, pad_id: int):
        if token_stream.ndim != 1:
            raise ValueError("token_stream must be 1D")
        if token_stream.numel() < 2:
            raise ValueError("token_stream must have at least 2 tokens")

        self.tokens = token_stream
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.num_samples = max(1, math.ceil((token_stream.numel() - 1) / seq_len))

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.seq_len
        chunk = self.tokens[start : start + self.seq_len + 1]

        x = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)
        y = torch.full((self.seq_len,), self.pad_id, dtype=torch.long)

        n = chunk.numel() - 1
        if n > 0:
            x[:n] = chunk[:-1]
            y[:n] = chunk[1:]

        return x, y


def _chunk_to_example(chunk: List[int], seq_len: int, pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    x = torch.full((seq_len,), pad_id, dtype=torch.long)
    y = torch.full((seq_len,), pad_id, dtype=torch.long)

    n = len(chunk) - 1
    if n > 0:
        x[:n] = torch.tensor(chunk[:-1], dtype=torch.long)
        y[:n] = torch.tensor(chunk[1:], dtype=torch.long)
    return x, y


class StreamingPackedCausalDataset(IterableDataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer: MathCharTokenizer,
        seq_len: int,
        pad_id: int,
        shuffle: bool,
        seed: int,
        estimated_num_samples: int,
    ):
        if seq_len < 1:
            raise ValueError("seq_len must be >= 1")
        self.texts = texts
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.pad_id = pad_id
        self.shuffle = shuffle
        self.seed = seed
        self.estimated_num_samples = max(1, estimated_num_samples)

    def __len__(self) -> int:
        return self.estimated_num_samples

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        worker_id = worker_info.id if worker_info else 0
        num_workers = worker_info.num_workers if worker_info else 1

        indices = list(range(worker_id, len(self.texts), num_workers))
        if self.shuffle and len(indices) > 1:
            iteration_seed = (torch.initial_seed() + self.seed + worker_id) % (2**31)
            rng = random.Random(iteration_seed)
            rng.shuffle(indices)

        token_buffer: List[int] = []
        start = 0

        for text_idx in indices:
            token_buffer.extend(self.tokenizer.encode(self.texts[text_idx], add_bos=True, add_eos=True))

            while len(token_buffer) - start > self.seq_len:
                chunk = token_buffer[start : start + self.seq_len + 1]
                start += self.seq_len
                yield _chunk_to_example(chunk, self.seq_len, self.pad_id)

                if start >= 4096:
                    token_buffer = token_buffer[start:]
                    start = 0

        remaining = token_buffer[start:]
        if len(remaining) > 1:
            yield _chunk_to_example(remaining[: self.seq_len + 1], self.seq_len, self.pad_id)


def build_token_stream(texts: List[str], tokenizer: MathCharTokenizer) -> torch.Tensor:
    ids: List[int] = []
    for text in texts:
        ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return torch.tensor(ids, dtype=torch.long)


def estimate_packed_num_samples(texts: List[str], tokenizer: MathCharTokenizer, seq_len: int) -> int:
    total_tokens = 0
    for text in texts:
        total_tokens += len(tokenizer.encode(text, add_bos=True, add_eos=True))
    if total_tokens < 2:
        return 1
    return max(1, math.ceil((total_tokens - 1) / seq_len))


def build_loader(
    texts: List[str],
    tokenizer: MathCharTokenizer,
    seq_len: int,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    streaming: bool,
) -> DataLoader:
    if streaming:
        dataset = StreamingPackedCausalDataset(
            texts=texts,
            tokenizer=tokenizer,
            seq_len=seq_len,
            pad_id=tokenizer.pad_id,
            shuffle=shuffle,
            seed=seed,
            estimated_num_samples=estimate_packed_num_samples(texts, tokenizer, seq_len),
        )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )

    stream = build_token_stream(texts, tokenizer)
    dataset = PackedCausalDataset(stream, seq_len=seq_len, pad_id=tokenizer.pad_id)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=False,
    )


# ============================================================
# TRAINING
# ============================================================


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    warmup_steps: int,
    total_steps: int,
) -> torch.optim.lr_scheduler.LambdaLR:
    def lr_lambda(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, max(0.0, progress))))
        return max(0.01, cosine)

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: ARTM,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LambdaLR],
    device: torch.device,
    grad_clip: float,
    grad_accum_steps: int = 1,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    use_fp16: bool = False,
    on_periodic_checkpoint: Optional[Callable[[int], None]] = None,
    save_every_seconds: float = 0.0,
) -> Tuple[float, int]:
    if grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be >= 1")

    model.train()
    total_loss = 0.0
    steps = 0
    update_steps = 0
    accum_micro_steps = 0
    last_periodic_ckpt = time.monotonic()

    optimizer.zero_grad(set_to_none=True)

    def _optimizer_step() -> None:
        nonlocal update_steps, last_periodic_ckpt
        if scaler is not None and use_fp16:
            scaler.unscale_(optimizer)
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        if scaler is not None and use_fp16:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        optimizer.zero_grad(set_to_none=True)
        update_steps += 1
        if on_periodic_checkpoint is not None and save_every_seconds > 0:
            now = time.monotonic()
            if (now - last_periodic_ckpt) >= save_every_seconds:
                on_periodic_checkpoint(update_steps)
                last_periodic_ckpt = now

    for batch_idx, (input_ids, targets) in enumerate(loader):
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        autocast_ctx = torch.cuda.amp.autocast(dtype=torch.float16) if use_fp16 else nullcontext()
        with autocast_ctx:
            _, loss = model(input_ids, targets)
            raw_loss = float(loss.item())
            loss = loss / grad_accum_steps

        if scaler is not None and use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accum_micro_steps += 1
        if accum_micro_steps == grad_accum_steps:
            _optimizer_step()
            accum_micro_steps = 0

        total_loss += raw_loss
        steps += 1

    if accum_micro_steps > 0 and steps > 0:
        _optimizer_step()

    return total_loss / max(1, steps), update_steps


@torch.no_grad()
def evaluate(model: ARTM, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    steps = 0

    for input_ids, targets in loader:
        input_ids = input_ids.to(device)
        targets = targets.to(device)
        _, loss = model(input_ids, targets)
        total_loss += float(loss.item())
        steps += 1

    return total_loss / max(1, steps)


@torch.no_grad()
def generate(
    model: ARTM,
    tokenizer: MathCharTokenizer,
    prompt: str,
    max_new_tokens: int = 80,
    temperature: float = 0.0,
    device: Optional[torch.device] = None,
) -> str:
    if device is None:
        device = next(model.parameters()).device

    model.eval()
    token_ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)

    for _ in range(max_new_tokens):
        context = token_ids[-model.cfg.max_seq_len :]
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        logits, _ = model(input_ids)
        next_logits = logits[0, -1]

        if temperature > 0:
            probs = F.softmax(next_logits / temperature, dim=-1)
            next_id = int(torch.multinomial(probs, num_samples=1).item())
        else:
            next_id = int(torch.argmax(next_logits).item())

        token_ids.append(next_id)
        if next_id == tokenizer.eos_id:
            break

    return tokenizer.decode(token_ids, skip_special=True)


def save_checkpoint(
    out_dir: Path,
    filename: str,
    model: nn.Module,
    tokenizer: MathCharTokenizer,
    cfg: ARTMConfig,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    step: int,
) -> None:
    raw_model = unwrap_model(model)
    payload = {
        "model_state": raw_model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "step": step,
        "model_config": asdict(cfg),
        "tokenizer": tokenizer.to_dict(),
    }
    torch.save(payload, out_dir / filename)


def pack_int4_per_channel(weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, int]:
    if weight.ndim != 2:
        raise ValueError("Expected 2D linear weight")

    weight = weight.detach().to(torch.float32)
    in_features_original = weight.size(1)

    max_abs = weight.abs().amax(dim=1, keepdim=True).clamp(min=1e-8)
    scales = (max_abs / 7.0).squeeze(1).to(torch.float16)

    q = torch.round(weight / max_abs * 7.0).clamp(-8, 7).to(torch.int8)
    q_u = (q + 8).to(torch.uint8)

    if q_u.size(1) % 2 == 1:
        q_u = torch.cat([q_u, torch.full((q_u.size(0), 1), 8, dtype=torch.uint8)], dim=1)

    packed = q_u[:, 0::2] | (q_u[:, 1::2] << 4)
    return packed.contiguous(), scales.contiguous(), in_features_original


def export_int4_linear_weights(model: nn.Module, out_path: Path) -> None:
    model_cpu = unwrap_model(model).cpu().eval()
    packed_state: Dict[str, Dict[str, object]] = {}

    for name, module in model_cpu.named_modules():
        if isinstance(module, nn.Linear):
            packed, scales, in_features_original = pack_int4_per_channel(module.weight)
            packed_state[name] = {
                "packed_weight": packed,
                "scales": scales,
                "in_features_original": in_features_original,
                "out_features": module.weight.size(0),
                "bias": module.bias.detach().to(torch.float32) if module.bias is not None else None,
            }

    torch.save(packed_state, out_path)


# ============================================================
# CLI
# ============================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARTM tiny reasoning model on weighted HF datasets")

    parser.add_argument("--dataset_path", type=str, default="")
    parser.add_argument("--use_hf_general_stack", action="store_true")
    parser.add_argument("--use_hf_math_datasets", action="store_true")
    parser.add_argument("--hf_max_samples_per_dataset", type=int, default=0)
    parser.add_argument("--hf_mixed_total_samples", type=int, default=0)
    parser.add_argument("--hf_cache_dir", type=str, default="")
    parser.add_argument("--synthetic_equation_samples", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="./artm_checkpoints")
    parser.add_argument("--resume_from", type=str, default="")

    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=100)

    parser.add_argument("--max_seq_len", type=int, default=384)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=8)
    parser.add_argument("--ffn_mult", type=int, default=4)
    parser.add_argument("--vocab_size", type=int, default=8192)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu", choices=["relu", "gelu"])
    parser.add_argument("--rope_base", type=float, default=10000.0)
    parser.add_argument("--label_smoothing", type=float, default=0.05)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--val_fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu_threads", type=int, default=0)
    parser.add_argument(
        "--streaming_loader",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--grad_accum_steps", type=int, default=1)

    parser.add_argument("--save_every", type=int, default=1)
    parser.add_argument("--save_every_minutes", type=float, default=0.0)
    parser.add_argument("--export_int4", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.grad_accum_steps < 1:
        raise ValueError("--grad_accum_steps must be >= 1")
    if not 0.0 <= args.label_smoothing < 1.0:
        raise ValueError("--label_smoothing must be in [0.0, 1.0)")

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.cpu_threads > 0:
        torch.set_num_threads(args.cpu_threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    use_fp16 = bool(args.fp16 and device.type == "cuda")
    if args.fp16 and not use_fp16:
        print("Warning: --fp16 requested but CUDA is unavailable. Running in fp32.")
    # PyTorch 2.4+ prefers torch.amp.GradScaler("cuda", ...).
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)
    except Exception:
        scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_texts: List[str] = []

    if args.dataset_path:
        all_texts.extend(load_math_texts(args.dataset_path))

    if args.use_hf_general_stack:
        hf_cache_dir = args.hf_cache_dir.strip() or None
        mixed_hf_texts, group_counts = load_hf_general_stack_texts(
            max_samples_per_dataset=args.hf_max_samples_per_dataset,
            mixed_total_samples=args.hf_mixed_total_samples,
            cache_dir=hf_cache_dir,
            seed=args.seed,
        )
        all_texts.extend(mixed_hf_texts)
        print(
            "Added mixed HF general stack samples: "
            f"{len(mixed_hf_texts)} | group counts={group_counts}"
        )
    elif args.use_hf_math_datasets:
        hf_cache_dir = args.hf_cache_dir.strip() or None
        raw_math_texts = load_hf_math_texts(
            max_samples_per_dataset=args.hf_max_samples_per_dataset,
            cache_dir=hf_cache_dir,
        )
        local_math = []
        for text in raw_math_texts:
            user_text, assistant_text = _extract_user_assistant_from_math_text(text)
            normalized = normalize_sample(user_text, assistant_text, mode="math")
            if normalized:
                local_math.append(normalized)
        if args.synthetic_equation_samples > 0:
            synthetic_raw = generate_synthetic_linear_equation_texts(
                num_samples=args.synthetic_equation_samples,
                seed=args.seed,
            )
            for text in synthetic_raw:
                user_text, assistant_text = _extract_user_assistant_from_math_text(text)
                normalized = normalize_sample(user_text, assistant_text, mode="math")
                if normalized:
                    local_math.append(normalized)
        all_texts.extend(local_math)
        print(f"Added HF math-only stack samples: {len(local_math)}")

    if not all_texts:
        raise ValueError(
            "Provide --dataset_path and/or use --use_hf_general_stack (or --use_hf_math_datasets)."
        )

    train_texts, val_texts = split_train_val(all_texts, args.val_fraction, args.seed)

    resume_payload: Optional[Dict[str, Any]] = None
    start_epoch = 1
    global_step = 0

    if args.resume_from.strip():
        resume_path = Path(args.resume_from.strip())
        if not resume_path.exists():
            raise FileNotFoundError(f"--resume_from not found: {resume_path}")
        resume_payload = torch.load(resume_path, map_location="cpu")
        tokenizer = MathCharTokenizer.from_dict(resume_payload["tokenizer"])
        cfg_dict = resume_payload.get("model_config") or resume_payload.get("config")
        if cfg_dict is None:
            raise ValueError("Checkpoint missing model config.")
        cfg = ARTMConfig(**cfg_dict)
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        global_step = int(resume_payload.get("step", 0))
        print(f"Resuming from checkpoint: {resume_path} | start_epoch={start_epoch} | step={global_step}")
    else:
        tokenizer = MathCharTokenizer.build_from_texts(
            texts=train_texts,
            target_vocab_size=args.vocab_size,
        )

        cfg = ARTMConfig(
            vocab_size=tokenizer.vocab_size,
            max_seq_len=args.max_seq_len,
            d_model=args.d_model,
            n_heads=args.n_heads,
            n_layers=args.n_layers,
            ffn_mult=args.ffn_mult,
            dropout=args.dropout,
            activation=args.activation,
            rope_base=args.rope_base,
            gradient_checkpointing=args.gradient_checkpointing,
            label_smoothing=args.label_smoothing,
            pad_token_id=tokenizer.pad_id,
        )

    model = ARTM(cfg).to(device)
    if resume_payload is not None:
        model.load_state_dict(resume_payload["model_state"])

    compile_enabled = False
    if hasattr(torch, "compile"):
        try:
            model = torch.compile(model)
            compile_enabled = True
        except Exception as exc:
            print(f"Warning: torch.compile failed ({exc}). Continuing without compile.")

    param_count = count_parameters(model)
    fp32_mb = param_count * 4 / (1024 * 1024)
    int8_mb = param_count * 1 / (1024 * 1024)
    int4_mb = param_count * 0.5 / (1024 * 1024)

    print(f"Device: {device}")
    print(f"Train samples: {len(train_texts)}")
    print(f"Val samples: {len(val_texts)}")
    print(f"Vocab size: {tokenizer.vocab_size}")
    print(f"Parameters: {param_count:,}")
    print(f"Estimated size FP32: {fp32_mb:.2f} MB | INT8: {int8_mb:.2f} MB | INT4: {int4_mb:.2f} MB")
    print(f"FP16 training: {'on' if use_fp16 else 'off'} | Grad accumulation: {args.grad_accum_steps}")
    print(f"Streaming loader: {'on' if args.streaming_loader else 'off'}")
    print(f"torch.compile: {'on' if compile_enabled else 'off'}")
    print(
        f"Activation: {cfg.activation} | RoPE base: {cfg.rope_base} | "
        f"Label smoothing: {cfg.label_smoothing} | Gradient checkpointing: {cfg.gradient_checkpointing}"
    )

    train_loader = build_loader(
        texts=train_texts,
        tokenizer=tokenizer,
        seq_len=cfg.max_seq_len,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        seed=args.seed,
        streaming=args.streaming_loader,
    )

    val_loader = None
    if val_texts:
        val_loader = build_loader(
            texts=val_texts,
            tokenizer=tokenizer,
            seq_len=cfg.max_seq_len,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            seed=args.seed + 1,
            streaming=args.streaming_loader,
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.99),
        weight_decay=args.weight_decay,
    )
    if resume_payload is not None and "optimizer_state" in resume_payload:
        optimizer.load_state_dict(resume_payload["optimizer_state"])

    updates_per_epoch = math.ceil(len(train_loader) / args.grad_accum_steps)
    total_steps = updates_per_epoch * args.epochs
    scheduler = build_scheduler(optimizer, args.warmup_steps, total_steps)
    if global_step > 0:
        for _ in range(global_step):
            scheduler.step()
    best_val_loss = float("inf")
    for epoch in range(start_epoch, args.epochs + 1):
        def _periodic_ckpt(epoch_update_steps: int) -> None:
            step_now = global_step + epoch_update_steps
            filename = f"artm_epoch_{epoch}_step_{step_now}.pt"
            save_checkpoint(
                out_dir=out_dir,
                filename=filename,
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                optimizer=optimizer,
                epoch=epoch,
                step=step_now,
            )
            print(f"[Autosave] Wrote checkpoint: {out_dir / filename}")

        train_loss, update_steps = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            scaler=scaler,
            use_fp16=use_fp16,
            on_periodic_checkpoint=_periodic_ckpt if args.save_every_minutes > 0 else None,
            save_every_seconds=max(0.0, args.save_every_minutes * 60.0),
        )
        global_step += update_steps
        train_ppl = math.exp(min(20.0, train_loss))

        if val_loader is not None:
            val_loss = evaluate(model, val_loader, device)
            val_ppl = math.exp(min(20.0, val_loss))
            print(
                f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | train_ppl={train_ppl:.2f} | "
                f"val_loss={val_loss:.4f} | val_ppl={val_ppl:.2f}"
            )
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    out_dir=out_dir,
                    filename="artm_best.pt",
                    model=model,
                    tokenizer=tokenizer,
                    cfg=cfg,
                    optimizer=optimizer,
                    epoch=epoch,
                    step=global_step,
                )
        else:
            print(f"Epoch {epoch}/{args.epochs} | train_loss={train_loss:.4f} | train_ppl={train_ppl:.2f}")

        sample = generate(
            unwrap_model(model),
            tokenizer,
            "Question: If 2x + 5 = 13, what is x?\nAnswer:",
            max_new_tokens=60,
            temperature=0.0,
            device=device,
        )
        print("\n[Sample Output]\n", sample, "\n")

        if epoch % args.save_every == 0:
            save_checkpoint(
                out_dir=out_dir,
                filename=f"artm_epoch_{epoch}.pt",
                model=model,
                tokenizer=tokenizer,
                cfg=cfg,
                optimizer=optimizer,
                epoch=epoch,
                step=global_step,
            )

  
    save_checkpoint(
        out_dir=out_dir,
        filename="artm_final.pt",
        model=model,
        tokenizer=tokenizer,
        cfg=cfg,
        optimizer=optimizer,
        epoch=args.epochs,
        step=global_step,
    )
    tokenizer.save(out_dir / "tokenizer.json")
    (out_dir / "model_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    if args.export_int4:
        export_int4_linear_weights(model, out_dir / "linear_int4.pt")
        print(f"Saved int4 linear weights to: {out_dir / 'linear_int4.pt'}")

    print(f"Done. Checkpoints saved in: {out_dir}")


if __name__ == "__main__":
    main()


# ============================================================
# HOW TO TRAIN ON GOOGLE COLAB (COMMANDS)
# ============================================================
# 1) Install dependencies:
#    !pip install -q torch datasets accelerate
#
# 2) Train tiny ARTM (about 25M-35M params with defaults) on weighted stack:
#    gsm8k 35%, hendrycks_math 23%, svamp+mawps 15%, wikipedia 8%, openwebtext 4%, the_stack 5%, instruction_chat 10%
#    !python artm_phase1.py \
#      --use_hf_general_stack \
#      --hf_max_samples_per_dataset 40000 \
#      --hf_mixed_total_samples 120000 \
#      --out_dir /content/artm_ckpts \
#      --epochs 6 \
#      --batch_size 8 \
#      --grad_accum_steps 8 \
#      --max_seq_len 384 \
#      --d_model 512 \
#      --n_heads 8 \
#      --n_layers 8 \
#      --ffn_mult 4 \
#      --vocab_size 8192 \
#      --dropout 0.1 \
#      --activation gelu \
#      --label_smoothing 0.05 \
#      --lr 2e-4 \
#      --val_fraction 0.05 \
#      --save_every 1 \
#      --fp16 \
#      --gradient_checkpointing
#
# 3) Quick debug run:
#    !python artm_phase1.py \
#      --use_hf_general_stack \
#      --hf_max_samples_per_dataset 2000 \
#      --hf_mixed_total_samples 20000 \
#      --out_dir /content/artm_ckpts \
#      --epochs 1 \
#      --batch_size 8 \
#      --grad_accum_steps 2 \
#      --max_seq_len 384 \
#      --d_model 512 \
#      --n_heads 8 \
#      --n_layers 8 \
#      --ffn_mult 4 \
#      --vocab_size 8192 \
#      --dropout 0.1 \
#      --activation gelu \
#      --label_smoothing 0.05 \
#      --lr 2e-4 \
#      --val_fraction 0.05 \
#      --save_every 1 \
#      --fp16 \
#      --gradient_checkpointing
#
# 4) Optional: math-only mode:
#    !python artm_phase1.py \
#      --use_hf_math_datasets \
#      --synthetic_equation_samples 0 \
#      --out_dir /content/artm_ckpts
#
# 5) Optional: combine local + weighted HF stack:
#    !python artm_phase1.py \
#      --dataset_path /content/my_extra_data.jsonl \
#      --use_hf_general_stack \
#      --out_dir /content/artm_ckpts
#
# 6) Test generation after training:
#    !python -c "import torch, artm_phase1 as a; ckpt=torch.load('/content/artm_ckpts/artm_final.pt', map_location='cpu'); tok=a.MathCharTokenizer.from_dict(ckpt['tokenizer']); cfg=a.ARTMConfig(**ckpt['config']); m=a.ARTM(cfg); m.load_state_dict(ckpt['model']); print(a.generate(m, tok, 'Q: If 3x + 7 = 22, what is x?\\nA:', max_new_tokens=80, temperature=0.0, device='cpu'))"
#
# 7) Optional int4 export:
#    add --export_int4 to training command.
