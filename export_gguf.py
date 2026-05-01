#!/usr/bin/env python3
"""
Convert distilled HF checkpoint to GGUF and quantize to Q4_K_M.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student_dir", type=str, default="/kaggle/working/artm_distilled/final_student")
    parser.add_argument("--gguf_out_dir", type=str, default="/kaggle/working/gguf")
    parser.add_argument("--llama_cpp_dir", type=str, default="/kaggle/working/llama.cpp")
    parser.add_argument("--skip_build", action="store_true")
    parser.add_argument("--quant_type", type=str, default="Q4_K_M")
    return parser.parse_args()


def ensure_llama_cpp(llama_cpp_dir: Path, skip_build: bool) -> None:
    if not llama_cpp_dir.exists():
        run(["git", "clone", "https://github.com/ggerganov/llama.cpp", str(llama_cpp_dir)])

    if skip_build:
        return

    build_dir = llama_cpp_dir / "build"
    run(["cmake", "-S", str(llama_cpp_dir), "-B", str(build_dir), "-DCMAKE_BUILD_TYPE=Release"])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "-j"])


def find_quantize_binary(llama_cpp_dir: Path) -> Path:
    candidates = [
        llama_cpp_dir / "build" / "bin" / "llama-quantize",
        llama_cpp_dir / "build" / "bin" / "quantize",
        llama_cpp_dir / "build" / "Release" / "llama-quantize.exe",
        llama_cpp_dir / "build" / "Release" / "quantize.exe",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError("Unable to find llama.cpp quantize binary after build")


def main() -> None:
    args = parse_args()

    student_dir = Path(args.student_dir)
    gguf_out = Path(args.gguf_out_dir)
    gguf_out.mkdir(parents=True, exist_ok=True)

    if not student_dir.exists():
        raise FileNotFoundError(f"Student checkpoint not found: {student_dir}")

    llama_cpp_dir = Path(args.llama_cpp_dir)
    ensure_llama_cpp(llama_cpp_dir, skip_build=args.skip_build)

    convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"convert_hf_to_gguf.py not found in {llama_cpp_dir}")

    f16_gguf = gguf_out / "artm-f16.gguf"
    q4_gguf = gguf_out / f"artm-{args.quant_type.lower()}.gguf"

    run(
        [
            "python",
            str(convert_script),
            str(student_dir),
            "--outfile",
            str(f16_gguf),
            "--outtype",
            "f16",
        ]
    )

    quant_bin = find_quantize_binary(llama_cpp_dir)
    run([str(quant_bin), str(f16_gguf), str(q4_gguf), args.quant_type])

    print(f"[done] fp16 gguf: {f16_gguf}")
    print(f"[done] quant gguf: {q4_gguf}")


if __name__ == "__main__":
    main()