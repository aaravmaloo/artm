from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from train_chat import ChatGPT, ChatTokenizer, GPTConfig, generate


def _merge_system_prompt(system_prompt: str, user_prompt: str) -> str:
    system_prompt = system_prompt.strip()
    user_prompt = user_prompt.strip()
    if not system_prompt:
        return user_prompt
    # This model has <user>/<assistant> tags but no dedicated <system> token,
    # so we fold system instructions into the user turn.
    return f"System instruction: {system_prompt}\n\nUser: {user_prompt}"


def _clean_state_dict_keys(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    # Handle checkpoints saved from torch.compile / DDP wrappers.
    keys = list(state.keys())
    if keys and all(k.startswith("_orig_mod.") for k in keys):
        state = {k[len("_orig_mod.") :]: v for k, v in state.items()}
    if keys and all(k.startswith("module.") for k in keys):
        state = {k[len("module.") :]: v for k, v in state.items()}
    return state


def _pick_tokenizer_path(ckpt_path: Path, tokenizer_path: str) -> Path:
    def as_tokenizer_json(p: Path) -> Path:
        return p / "tokenizer.json" if p.is_dir() else p

    if tokenizer_path:
        p = as_tokenizer_json(Path(tokenizer_path))
        if not p.exists():
            raise FileNotFoundError(f"Tokenizer path not found: {p}")
        return p

    candidates = [
        ckpt_path.parent / "tokenizer" / "tokenizer.json",
        ckpt_path.parent / "tokenizer.json",
        Path("tokenizer.json"),
        Path("checkpoints_chat/tokenizer/tokenizer.json"),
    ]
    for p in candidates:
        p = as_tokenizer_json(p)
        if p.exists():
            return p

    raise FileNotFoundError(
        "Tokenizer not found in default locations.\n"
        "Pass --tokenizer_path explicitly."
    )


def _build_config(ckpt: dict, seq_len_override: int | None, vocab_size: int) -> GPTConfig:
    raw_cfg = ckpt.get("model_config")
    if not isinstance(raw_cfg, dict):
        raise ValueError("Checkpoint missing model_config. Use a chat_epoch_X.pt from training.")

    allowed = GPTConfig.__annotations__.keys()
    cfg_kwargs = {k: v for k, v in raw_cfg.items() if k in allowed}
    cfg = GPTConfig(**cfg_kwargs)
    cfg.vocab_size = vocab_size
    if seq_len_override is not None:
        cfg.seq_len = seq_len_override
    return cfg


def load_model_and_tokenizer(
    ckpt_path: Path,
    tokenizer_path: str,
    device: torch.device,
    seq_len_override: int | None,
) -> tuple[ChatGPT, ChatTokenizer, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    tok_path = _pick_tokenizer_path(ckpt_path, tokenizer_path)
    tok = ChatTokenizer.load(tok_path)

    cfg = _build_config(ckpt, seq_len_override=seq_len_override, vocab_size=tok.vocab_size)
    model = ChatGPT(cfg).to(device)

    state = ckpt.get("model_state", ckpt)
    if not isinstance(state, dict):
        raise ValueError("Checkpoint format is invalid. Expected state_dict or dict with model_state.")
    state = _clean_state_dict_keys(state)

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Warning: missing keys while loading: {len(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys while loading: {len(unexpected)}")

    model.eval()
    return model, tok, cfg.seq_len


def run_interactive(
    model: ChatGPT,
    tok: ChatTokenizer,
    device: torch.device,
    seq_len: int,
    system_prompt: str,
    temperature: float,
    top_k: int,
    top_p: float,
    max_new_tokens: int,
) -> None:
    print("Interactive mode. Type `exit` to stop.")
    while True:
        user = input("you> ").strip()
        if user.lower() in {"exit", "quit"}:
            break
        if not user:
            continue
        merged_prompt = _merge_system_prompt(system_prompt, user)
        out = generate(
            model=model,
            tok=tok,
            prompt=merged_prompt,
            device=device,
            seq_len=seq_len,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
        )
        print(f"assistant> {out}")


def main() -> None:
    p = argparse.ArgumentParser(description="Inference for phase-1 chat checkpoint")
    p.add_argument("--ckpt", type=str, required=True, help="Path to chat_epoch_X.pt")
    p.add_argument("--tokenizer_path", type=str, default="", help="Path to tokenizer.json or a folder containing it")
    p.add_argument("--prompt", type=str, default="", help="Single prompt; if empty, starts interactive mode")
    p.add_argument("--system_prompt", type=str, default="", help="Behavior instruction injected before the user prompt")
    p.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seq_len", type=int, default=0, help="Optional override; 0 means use checkpoint value")
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--max_new_tokens", type=int, default=80)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    seq_override = args.seq_len if args.seq_len > 0 else None
    model, tok, seq_len = load_model_and_tokenizer(
        ckpt_path=ckpt_path,
        tokenizer_path=args.tokenizer_path,
        device=device,
        seq_len_override=seq_override,
    )

    if args.prompt.strip():
        merged_prompt = _merge_system_prompt(args.system_prompt, args.prompt)
        out = generate(
            model=model,
            tok=tok,
            prompt=merged_prompt,
            device=device,
            seq_len=seq_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )
        print(out)
    else:
        run_interactive(
            model=model,
            tok=tok,
            device=device,
            seq_len=seq_len,
            system_prompt=args.system_prompt,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
        )


if __name__ == "__main__":
    main()
