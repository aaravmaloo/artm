#!/usr/bin/env bash
set -euo pipefail

# ARTM end-to-end TPU pipeline for v5e-1 (Single Core)
# PERSISTENT VERSION: Saves to Google Drive
echo "=========================================================="
echo "      ARTM TPU DISTILLATION PIPELINE - V3.3 (PERSISTENT)"
echo "=========================================================="

# Create output directory on Google Drive if it doesn't exist
DRIVE_OUT="/content/drive/MyDrive/artm_output"
mkdir -p "$DRIVE_OUT"

# 1) Install dependencies
echo "[system] Installing TPU dependencies..."
python -m pip install --upgrade pip
python -m pip install --upgrade transformers accelerate
python -m pip install -r requirements_kaggle.txt
python -m pip install torch-xla

# 2) Dataset Setup
DATA_PATH="/content/jaqua_teacher_data.jsonl"
if [ -f "$DATA_PATH" ]; then
    echo "[system] Found dataset at $DATA_PATH."
else
    echo "[error] Dataset not found at $DATA_PATH. Please upload it to /content/"
    exit 1
fi

# 3) TPU Distillation Training
echo "[system] Starting TPU Training (v5e-1 Single-Core)..."
# We save to DRIVE_OUT so it persists even if Colab kills the session
python train_artm_distill_tpu.py \
  --teacher_model microsoft/Phi-3.5-mini-instruct \
  --data_jsonl "$DATA_PATH" \
  --output_dir "$DRIVE_OUT" \
  --epochs 3.5 \
  --learning_rate 5e-4 \
  --per_device_batch_size 4 \
  --student_layers 36 \
  --student_hidden 1536 \
  --student_heads 24 \
  --student_ffn 6144 \
  --context_length 256

# 4) Export to GGUF
echo "[system] Converting to GGUF..."
python export_gguf.py \
  --student_dir "$DRIVE_OUT" \
  --gguf_out_dir "$DRIVE_OUT/gguf" \
  --llama_cpp_dir /content/llama.cpp \
  --quant_type Q4_K_M

# 5) Benchmarking
echo "[system] Running Benchmarks..."
python benchmark_tokens.py \
  --student_hf_dir "$DRIVE_OUT" \
  --teacher_model microsoft/Phi-3.5-mini-instruct \
  --teacher_load_in_4bit \
  --eval_jsonl "$DATA_PATH" \
  --max_eval_samples 512 \
  --gguf_model_path "$DRIVE_OUT/gguf/jaqua-q4_k_m.gguf" \
  --n_ctx 2048 \
  --report_json "$DRIVE_OUT/jaqua_tpu_benchmark.json"

echo "=========================================================="
echo "PIPELINE COMPLETE!"
echo "All files saved permanently to: $DRIVE_OUT"
echo "=========================================================="
