from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

try:
    from datasets import load_dataset
except ImportError:
    load_dataset = None

try:
    from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
except ImportError:
    Tokenizer = None
    decoders = None
    models = None
    normalizers = None
    pre_tokenizers = None
    trainers = None

SPECIAL = ["<pad>", "<unk>", "<bos>", "<eos>", "<user>", "<assistant>"]


def req_deps() -> None:
    if load_dataset is None:
        raise RuntimeError("Missing dependency: datasets (pip install datasets)")
    if Tokenizer is None:
        raise RuntimeError("Missing dependency: tokenizers (pip install tokenizers)")


def clean(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def parse_csv(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def parse_ratios(s: str, n: int) -> List[float]:
    if not s:
        return [1.0 / n] * n
    vals = [float(x.strip()) for x in s.split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"mix_ratios must have {n} values")
    z = sum(vals)
    if z <= 0:
        raise ValueError("mix_ratios must sum > 0")
    return [v / z for v in vals]


def mem_report(num_params: int) -> Dict[str, float]:
    mib = 1024**2
    return {
        "fp32": (num_params * 4) / mib,
        "fp16": (num_params * 2) / mib,
        "int8": (num_params * 1) / mib,
        "int4": (num_params * 0.5) / mib,
    }


@dataclass
class GPTConfig:
    vocab_size: int
    seq_len: int = 256
    d_model: int = 640
    n_heads: int = 8
    n_layers: int = 10
    ffn_mult: int = 4
    dropout: float = 0.1
    activation: str = "gelu"
    rope_base: float = 10000.0
    gradient_checkpointing: bool = False


class ChatTokenizer:
    def __init__(self, tok: Tokenizer):
        self.tok = tok
        self.tok.decoder = decoders.ByteLevel()
        self.pad_id = self._id("<pad>")
        self.unk_id = self._id("<unk>")
        self.bos_id = self._id("<bos>")
        self.eos_id = self._id("<eos>")
        self.user_id = self._id("<user>")
        self.assistant_id = self._id("<assistant>")

    def _id(self, token: str) -> int:
        idx = self.tok.token_to_id(token)
        if idx is None:
            raise ValueError(f"Tokenizer missing special token: {token}")
        return idx

    @property
    def vocab_size(self) -> int:
        return self.tok.get_vocab_size()

    def encode_text(self, text: str) -> List[int]:
        return self.tok.encode(text).ids

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=skip_special_tokens)

    def encode_chat(self, user: str, assistant: str, max_len: int) -> List[int]:
        user = clean(user)
        assistant = clean(assistant)
        if not user or not assistant:
            return []
        ids = [self.bos_id, self.user_id] + self.encode_text(user) + [self.assistant_id] + self.encode_text(assistant) + [self.eos_id]
        if len(ids) > max_len:
            ids = ids[: max_len - 1] + [self.eos_id]
        return ids

    def encode_prompt(self, user: str) -> List[int]:
        return [self.bos_id, self.user_id] + self.encode_text(clean(user)) + [self.assistant_id]

    def save(self, d: Path) -> None:
        d.mkdir(parents=True, exist_ok=True)
        self.tok.save(str(d / "tokenizer.json"))
        (d / "tokenizer_meta.json").write_text(
            json.dumps(
                {
                    "special_tokens": SPECIAL,
                    "pad_id": self.pad_id,
                    "unk_id": self.unk_id,
                    "bos_id": self.bos_id,
                    "eos_id": self.eos_id,
                    "user_id": self.user_id,
                    "assistant_id": self.assistant_id,
                    "vocab_size": self.vocab_size,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, p: Path) -> "ChatTokenizer":
        return cls(Tokenizer.from_file(str(p)))

    @classmethod
    def train_new(cls, text_iter: Iterable[str], vocab_size: int, min_freq: int = 2) -> "ChatTokenizer":
        tok = Tokenizer(models.BPE(unk_token="<unk>"))
        tok.normalizer = normalizers.Sequence([normalizers.NFKC(), normalizers.Replace("\u00A0", " "), normalizers.Strip()])
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tr = trainers.BpeTrainer(vocab_size=vocab_size, min_frequency=min_freq, special_tokens=SPECIAL, limit_alphabet=1000)
        tok.train_from_iterator(text_iter, trainer=tr)
        tok.decoder = decoders.ByteLevel()
        return cls(tok)


def story_prompt(rng: random.Random) -> str:
    opts = [
        "Tell me a short story.",
        "Write a simple bedtime story.",
        "Share a short friendly story.",
    ]
    return opts[rng.randrange(len(opts))]


def iter_oasst(rows: Iterable[dict], seed: int, english_only: bool = True) -> Iterator[Tuple[str, str]]:
    prompts: Dict[str, str] = {}
    rng = random.Random(seed)
    for r in rows:
        if r.get("deleted"):
            continue
        role = str(r.get("role", "")).strip().lower()
        text = clean(str(r.get("text", "")))
        if not text:
            continue
        if english_only:
            lang = str(r.get("lang", "")).strip().lower()
            if lang and not lang.startswith("en"):
                continue
        mid = r.get("message_id")
        pid = r.get("parent_id")
        if role == "prompter" and mid:
            prompts[str(mid)] = text
            if len(prompts) > 120000:
                prompts.pop(next(iter(prompts)), None)
            continue
        if role != "assistant" or not pid:
            continue
        parent = prompts.get(str(pid))
        if not parent:
            continue
        rank = r.get("rank")
        if rank is not None:
            try:
                if int(rank) > 1:
                    continue
            except Exception:
                pass
        yield parent, text
        if len(prompts) > 80000 and rng.random() < 0.002:
            prompts.pop(next(iter(prompts)), None)


def iter_tinystories(rows: Iterable[dict], seed: int) -> Iterator[Tuple[str, str]]:
    rng = random.Random(seed)
    for r in rows:
        t = clean(str(r.get("text", "")))
        if t:
            yield story_prompt(rng), t


def extract_generic(ex: dict) -> Optional[Tuple[str, str]]:
    msgs = ex.get("messages")
    if isinstance(msgs, list):
        u = None
        a = None
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = str(m.get("role", m.get("from", ""))).lower()
            c = clean(str(m.get("content", m.get("text", m.get("value", "")))))
            if not c:
                continue
            if u is None and role in {"user", "human", "prompter"}:
                u = c
            elif u is not None and a is None and role in {"assistant", "gpt", "bot"}:
                a = c
                break
        if u and a:
            return u, a

    ukeys = ["instruction", "prompt", "question", "input", "user"]
    akeys = ["response", "output", "answer", "assistant", "completion"]
    u = ""
    a = ""
    for k in ukeys:
        v = ex.get(k)
        if isinstance(v, str) and clean(v):
            u = clean(v)
            break
    for k in akeys:
        v = ex.get(k)
        if isinstance(v, str) and clean(v):
            a = clean(v)
            break
    if u and a:
        return u, a

    t = ex.get("text") or ex.get("content")
    if isinstance(t, str):
        t = clean(t)
        if t:
            return "Respond in a helpful concise way.", t
    return None


def iter_generic(rows: Iterable[dict]) -> Iterator[Tuple[str, str]]:
    for r in rows:
        p = extract_generic(r)
        if p is not None:
            yield p

class MixedChat(IterableDataset):
    def __init__(
        self,
        names: Sequence[str],
        ratios: Sequence[float],
        tok: ChatTokenizer,
        seq_len: int,
        target_samples: int,
        split: str,
        streaming: bool,
        seed: int,
        shuffle_buffer: int,
        val_fraction: float,
        oasst_english_only: bool,
    ):
        super().__init__()
        self.names = list(names)
        self.ratios = list(ratios)
        self.tok = tok
        self.seq_len = seq_len
        self.target = target_samples
        self.split = split
        self.streaming = streaming
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer
        self.val_fraction = val_fraction
        self.oasst_english_only = oasst_english_only
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _rows(self, name: str, seed: int) -> Iterable[dict]:
        ds = load_dataset(name, split="train", streaming=self.streaming)
        return ds.shuffle(seed=seed, buffer_size=self.shuffle_buffer) if self.streaming else ds.shuffle(seed=seed)

    def _pairs(self, name: str, seed: int) -> Iterator[Tuple[str, str]]:
        rows = self._rows(name, seed)
        lname = name.lower()
        if "oasst1" in lname:
            yield from iter_oasst(rows, seed, self.oasst_english_only)
        elif "tinystories" in lname:
            yield from iter_tinystories(rows, seed)
        else:
            yield from iter_generic(rows)

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        wi = get_worker_info()
        wid = wi.id if wi is not None else 0
        base = self.seed + self.epoch * 100003 + wid * 7919
        rng = random.Random(base)
        seeds = [base + i * 9973 for i in range(len(self.names))]
        iters = [self._pairs(self.names[i], seeds[i]) for i in range(len(self.names))]

        produced = 0
        seen = 0
        while produced < self.target:
            i = rng.choices(range(len(iters)), weights=self.ratios, k=1)[0]
            try:
                u, a = next(iters[i])
            except StopIteration:
                seeds[i] += 1
                iters[i] = self._pairs(self.names[i], seeds[i])
                try:
                    u, a = next(iters[i])
                except StopIteration:
                    continue

            seen += 1
            is_val = ((hash((seen, self.seed)) % 1000000) / 1000000.0) < self.val_fraction
            if self.split == "train" and is_val:
                continue
            if self.split == "val" and not is_val:
                continue

            ids = self.tok.encode_chat(u, a, self.seq_len + 1)
            if len(ids) < 2:
                continue
            x = ids[:-1]
            y = ids[1:]
            if len(x) < self.seq_len:
                pad = self.seq_len - len(x)
                x += [self.tok.pad_id] * pad
                y += [self.tok.pad_id] * pad
            else:
                x = x[: self.seq_len]
                y = y[: self.seq_len]

            produced += 1
            yield torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w


class FFN(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        h = cfg.d_model * cfg.ffn_mult
        self.fc1 = nn.Linear(cfg.d_model, h, bias=False)
        self.fc2 = nn.Linear(h, cfg.d_model, bias=False)
        self.drop = nn.Dropout(cfg.dropout)
        a = cfg.activation.lower()
        if a == "gelu":
            self.act = nn.GELU()
        elif a == "relu":
            self.act = nn.ReLU()
        elif a == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {cfg.activation}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


class Attn(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        if cfg.d_model % cfg.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.h = cfg.n_heads
        self.d = cfg.d_model // cfg.n_heads
        if self.d % 2 != 0:
            raise ValueError("head_dim must be even for RoPE")
        self.dm = cfg.d_model
        self.drop = cfg.dropout
        self.base = cfg.rope_base
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        cos, sin = self._rope_cache(cfg.seq_len)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    def _rope_cache(self, T: int) -> Tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d))
        pos = torch.arange(T).float()
        f = torch.outer(pos, inv)
        return torch.cos(f)[None, None, :, :], torch.sin(f)[None, None, :, :]

    @staticmethod
    def _apply_rope(x: torch.Tensor, c: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        y = torch.stack((x1 * c - x2 * s, x1 * s + x2 * c), dim=-1)
        return y.flatten(-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.h, self.d).transpose(1, 2)
        k = k.view(B, T, self.h, self.d).transpose(1, 2)
        v = v.view(B, T, self.h, self.d).transpose(1, 2)

        if T > self.cos.size(2):
            c, s = self._rope_cache(T)
            c = c.to(x.device)
            s = s.to(x.device)
        else:
            c = self.cos[:, :, :T, :].to(x.device)
            s = self.sin[:, :, :T, :].to(x.device)

        q = self._apply_rope(q, c, s)
        k = self._apply_rope(k, c, s)

        if hasattr(F, "scaled_dot_product_attention"):
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.drop if self.training else 0.0)
        else:
            score = (q @ k.transpose(-2, -1)) / math.sqrt(self.d)
            mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            score = score.masked_fill(mask, float("-inf"))
            p = F.dropout(torch.softmax(score, dim=-1), p=self.drop, training=self.training)
            y = p @ v

        y = y.transpose(1, 2).contiguous().view(B, T, self.dm)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n1 = RMSNorm(cfg.d_model)
        self.a = Attn(cfg)
        self.n2 = RMSNorm(cfg.d_model)
        self.f = FFN(cfg)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.a(self.n1(x))
        x = x + self.f(self.n2(x))
        return x


class ChatGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.nf = RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.emb.weight
        self._init()

    def _init(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.drop(self.emb(idx))
        for b in self.blocks:
            if self.cfg.gradient_checkpointing and self.training:
                x = checkpoint(b, x, use_reentrant=False)
            else:
                x = b(x)
        return self.head(self.nf(x))

class CosineWarmup:
    def __init__(self, opt: torch.optim.Optimizer, warmup: int, total: int, min_ratio: float = 0.1):
        self.opt = opt
        self.warmup = max(1, warmup)
        self.total = max(self.warmup + 1, total)
        self.min_ratio = min_ratio
        self.base = [g["lr"] for g in opt.param_groups]
        self.t = 0

    def step(self) -> None:
        self.t += 1
        if self.t <= self.warmup:
            s = self.t / self.warmup
        else:
            p = (self.t - self.warmup) / (self.total - self.warmup)
            s = self.min_ratio + (1 - self.min_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
        for b, g in zip(self.base, self.opt.param_groups):
            g["lr"] = b * s

    def state_dict(self) -> Dict[str, int]:
        return {"t": self.t}


def ce_loss(logits: torch.Tensor, y: torch.Tensor, pad_id: int) -> torch.Tensor:
    v = logits.size(-1)
    return F.cross_entropy(logits.view(-1, v), y.view(-1), ignore_index=pad_id)


def top_p_filter(logits: torch.Tensor, top_p: float) -> torch.Tensor:
    if top_p >= 1.0:
        return logits
    sl, si = torch.sort(logits, descending=True)
    cp = torch.cumsum(torch.softmax(sl, dim=-1), dim=-1)
    m = cp > top_p
    m[..., 1:] = m[..., :-1].clone()
    m[..., 0] = False
    sl[m] = float("-inf")
    return torch.full_like(logits, float("-inf")).scatter(-1, si, sl)


def generate(
    model: nn.Module,
    tok: ChatTokenizer,
    prompt: str,
    device: torch.device,
    seq_len: int,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 0.95,
    max_new_tokens: int = 80,
) -> str:
    model.eval()
    ids = tok.encode_prompt(prompt)
    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        xc = x[:, -seq_len:]
        with torch.no_grad():
            logits = model(xc)[:, -1, :]
            if temperature <= 0:
                nt = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
                    cut = vals[:, -1].unsqueeze(-1)
                    logits = torch.where(logits < cut, torch.full_like(logits, float("-inf")), logits)
                logits = top_p_filter(logits, top_p)
                nt = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1)
        x = torch.cat([x, nt], dim=1)
        if int(nt.item()) == tok.eos_id:
            break

    out = x[0].tolist()
    try:
        st = out.index(tok.assistant_id) + 1
    except ValueError:
        st = 0
    resp = []
    for t in out[st:]:
        if t == tok.eos_id:
            break
        if t in {tok.bos_id, tok.user_id, tok.assistant_id, tok.pad_id}:
            continue
        resp.append(t)
    return clean(tok.decode(resp, skip_special_tokens=True))


def eval_loss(model: nn.Module, loader: DataLoader, device: torch.device, tok: ChatTokenizer, fp16: bool, max_batches: Optional[int]) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for i, (x, y) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=fp16 and device.type == "cuda"):
                l = ce_loss(model(x), y, tok.pad_id)
            losses.append(l.item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))


def save_ckpt(out_dir: Path, epoch: int, model: nn.Module, opt: torch.optim.Optimizer, sched: CosineWarmup, cfg: GPTConfig, args: argparse.Namespace, tok: ChatTokenizer) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"chat_epoch_{epoch}.pt"
    torch.save(
        {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": opt.state_dict(),
            "scheduler_state": sched.state_dict(),
            "model_config": asdict(cfg),
            "train_args": vars(args),
            "tokenizer_vocab_size": tok.vocab_size,
        },
        p,
    )
    return p


def tok_corpus(names: Sequence[str], ratios: Sequence[float], streaming: bool, seed: int, shuffle_buffer: int, max_texts: int, oasst_english_only: bool) -> Iterator[str]:
    rng = random.Random(seed)

    def rows(name: str, s: int) -> Iterable[dict]:
        ds = load_dataset(name, split="train", streaming=streaming)
        return ds.shuffle(seed=s, buffer_size=shuffle_buffer) if streaming else ds.shuffle(seed=s)

    seeds = [seed + i * 1231 for i in range(len(names))]
    its: List[Iterator[Tuple[str, str]]] = []
    for i, n in enumerate(names):
        lname = n.lower()
        r = rows(n, seeds[i])
        if "oasst1" in lname:
            it = iter_oasst(r, seeds[i], oasst_english_only)
        elif "tinystories" in lname:
            it = iter_tinystories(r, seeds[i])
        else:
            it = iter_generic(r)
        its.append(it)

    got = 0
    while got < max_texts:
        i = rng.choices(range(len(its)), weights=ratios, k=1)[0]
        try:
            u, a = next(its[i])
        except StopIteration:
            seeds[i] += 1
            lname = names[i].lower()
            r = rows(names[i], seeds[i])
            if "oasst1" in lname:
                its[i] = iter_oasst(r, seeds[i], oasst_english_only)
            elif "tinystories" in lname:
                its[i] = iter_tinystories(r, seeds[i])
            else:
                its[i] = iter_generic(r)
            continue
        t = clean(f"<user> {u}\n<assistant> {a}")
        if t:
            got += 1
            yield t


def make_tokenizer(args: argparse.Namespace, names: Sequence[str], ratios: Sequence[float], out_dir: Path) -> ChatTokenizer:
    td = out_dir / "tokenizer"
    tf = td / "tokenizer.json"
    if args.tokenizer_path:
        tf = Path(args.tokenizer_path)

    if tf.exists() and not args.force_retrain_tokenizer:
        print(f"Loading tokenizer from {tf}")
        return ChatTokenizer.load(tf)

    print("Training tokenizer...")
    it = tok_corpus(names, ratios, args.dataset_streaming, args.seed, args.shuffle_buffer, args.tokenizer_samples, args.oasst_english_only)
    tok = ChatTokenizer.train_new(it, args.vocab_size, args.tokenizer_min_freq)
    tok.save(td)
    print(f"Tokenizer saved to {td}")
    return tok

def train(args: argparse.Namespace) -> None:
    req_deps()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    names = parse_csv(args.hf_datasets)
    if not names:
        raise ValueError("--hf_datasets is empty")
    ratios = parse_ratios(args.mix_ratios, len(names))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tok = make_tokenizer(args, names, ratios, out_dir)
    if tok.vocab_size != args.vocab_size:
        print(f"Adjusting vocab_size from {args.vocab_size} to tokenizer size {tok.vocab_size}")
        args.vocab_size = tok.vocab_size

    cfg = GPTConfig(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        ffn_mult=args.ffn_mult,
        dropout=args.dropout,
        activation=args.activation,
        rope_base=args.rope_base,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = bool(args.fp16 and dev.type == "cuda")
    if args.fp16 and dev.type != "cuda":
        print("FP16 requested but CUDA unavailable. Using FP32.")

    train_target = max(1, int(args.max_samples * (1.0 - args.val_fraction)))
    val_target = max(1, args.max_samples - train_target)

    train_ds = MixedChat(names, ratios, tok, args.seq_len, train_target, "train", args.dataset_streaming, args.seed, args.shuffle_buffer, args.val_fraction, args.oasst_english_only)
    val_ds = MixedChat(names, ratios, tok, args.seq_len, val_target, "val", args.dataset_streaming, args.seed + 999, args.shuffle_buffer, args.val_fraction, args.oasst_english_only)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, num_workers=0)

    model = ChatGPT(cfg).to(dev)

    if args.resume_from and args.resume_weights and args.resume_from != args.resume_weights:
        raise ValueError("Use either --resume_from or --resume_weights (or the same path for both).")
    resume_path = args.resume_from if args.resume_from else args.resume_weights
    if resume_path:
        print(f"Loading model weights from {resume_path}")
        ck = torch.load(resume_path, map_location="cpu")
        if "model_state" in ck:
            model.load_state_dict(ck["model_state"], strict=False)
        else:
            model.load_state_dict(ck, strict=False)

    if args.compile and hasattr(torch, "compile"):
        print("Compiling model...")
        model = torch.compile(model)

    nparams = sum(p.numel() for p in model.parameters())
    mem = mem_report(nparams)
    print("\n=== Model Report ===")
    print(f"Parameters: {nparams:,}")
    print(f"Memory FP32: {mem['fp32']:.2f} MiB")
    print(f"Memory FP16: {mem['fp16']:.2f} MiB")
    print(f"Memory INT8: {mem['int8']:.2f} MiB")
    print(f"Memory INT4: {mem['int4']:.2f} MiB")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=args.weight_decay)
    micro_steps = math.ceil(train_target / args.batch_size)
    opt_steps = math.ceil(micro_steps / args.grad_accum) * args.epochs
    sched = CosineWarmup(opt, args.warmup_steps, opt_steps, args.min_lr_ratio)
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    (out_dir / "train_config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "model_config": asdict(cfg),
                "dataset_names": names,
                "mix_ratios": ratios,
                "train_target": train_target,
                "val_target": val_target,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    gstep = 0
    t0 = time.time()
    model.train()

    for epoch in range(1, args.epochs + 1):
        train_ds.set_epoch(epoch)
        val_ds.set_epoch(epoch)
        opt.zero_grad(set_to_none=True)
        run_loss = 0.0
        run_n = 0
        step = 0

        for step, (x, y) in enumerate(train_dl, start=1):
            x = x.to(dev, non_blocking=True)
            y = y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, dtype=torch.float16, enabled=use_fp16):
                loss = ce_loss(model(x), y, tok.pad_id)
                back = loss / args.grad_accum

            scaler.scale(back).backward()
            run_loss += loss.item()
            run_n += 1

            if step % args.grad_accum == 0:
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
                gstep += 1

                if gstep % args.log_every == 0:
                    lr = opt.param_groups[0]["lr"]
                    print(f"epoch={epoch} step={gstep} train_loss={run_loss / max(1, run_n):.4f} lr={lr:.6g} elapsed_sec={time.time() - t0:.1f}")
                    run_loss = 0.0
                    run_n = 0

                if gstep % args.sample_every == 0:
                    s = generate(model, tok, args.sample_prompt, dev, args.seq_len, args.sample_temperature, args.sample_top_k, args.sample_top_p, args.sample_max_new_tokens)
                    print("--- sample ---")
                    print(f"user: {args.sample_prompt}")
                    print(f"assistant: {s}")
                    print("-------------")

        if step > 0 and step % args.grad_accum != 0:
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            gstep += 1

        vl = eval_loss(model, val_dl, dev, tok, use_fp16, args.val_batches)
        print(f"epoch={epoch} validation_loss={vl:.4f}")

        if args.save_every_epoch:
            p = save_ckpt(out_dir, epoch, model, opt, sched, cfg, args, tok)
            print(f"Saved checkpoint: {p}")

    print(f"Training completed in {time.time() - t0:.1f} seconds")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Efficient chat foundation GPT trainer")

    p.add_argument("--dataset_streaming", action="store_true")
    p.add_argument("--hf_datasets", type=str, default="roneneldan/TinyStories,OpenAssistant/oasst1")
    p.add_argument("--mix_ratios", type=str, default="")
    p.add_argument("--max_samples", type=int, default=120000)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--shuffle_buffer", type=int, default=10000)
    p.add_argument("--oasst_english_only", action="store_true", default=True)

    p.add_argument("--vocab_size", type=int, default=8192)
    p.add_argument("--tokenizer_path", type=str, default="")
    p.add_argument("--force_retrain_tokenizer", action="store_true")
    p.add_argument("--tokenizer_samples", type=int, default=200000)
    p.add_argument("--tokenizer_min_freq", type=int, default=2)

    p.add_argument("--seq_len", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup_steps", type=int, default=200)

    p.add_argument("--d_model", type=int, default=640)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=10)
    p.add_argument("--ffn_mult", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--activation", type=str, default="gelu", choices=["gelu", "relu", "silu"])
    p.add_argument("--rope_base", type=float, default=10000.0)

    p.add_argument("--fp16", action="store_true")
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--compile", action="store_true")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--min_lr_ratio", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)

    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--sample_every", type=int, default=500)
    p.add_argument("--sample_prompt", type=str, default="Hi! Introduce yourself in one short paragraph.")
    p.add_argument("--sample_temperature", type=float, default=0.8)
    p.add_argument("--sample_top_k", type=int, default=40)
    p.add_argument("--sample_top_p", type=float, default=0.95)
    p.add_argument("--sample_max_new_tokens", type=int, default=80)
    p.add_argument("--val_batches", type=int, default=80)

    p.add_argument("--save_every_epoch", action="store_true")
    p.add_argument("--out_dir", type=str, default="./checkpoints_chat")
    p.add_argument("--resume_from", type=str, default="", help="Load model weights only for refinement.")
    p.add_argument("--resume_weights", type=str, default="", help="Backward-compatible alias for --resume_from.")
    p.add_argument("--seed", type=int, default=42)

    return p


def main() -> None:
    args = parser().parse_args()
    if not (0.0 < args.val_fraction < 0.5):
        raise ValueError("val_fraction must be in (0.0, 0.5)")
    if args.max_samples < 100:
        raise ValueError("max_samples should be >= 100")
    if args.grad_accum < 1:
        raise ValueError("grad_accum must be >= 1")
    train(args)


if __name__ == "__main__":
    main()
