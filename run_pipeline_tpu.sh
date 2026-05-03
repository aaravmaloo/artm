#!/usr/bin/env bash
set -euo pipefail

# ARTM end-to-end TPU pipeline
# Optimized for Kaggle TPU v3-8

echo "=========================================================="
echo "      ARTM TPU DISTILLATION PIPELINE - V3.0"
echo "=========================================================="

# 1) Install dependencies
echo "[system] Installing TPU dependencies..."
python -m pip install --upgrade pip
python -m pip install -r requirements_kaggle.txt
# Ensure TPU-specific libraries are present
python -m pip install torch-xla

# 2) Dataset Setup
BACKUP_PATH="/kaggle/input/datasets/aaravmaloo6/dataset-rows/jaqua_teacher_data.jsonl"
DATA_PATH="/kaggle/working/jaqua_teacher_data.jsonl"

if [ -f "$BACKUP_PATH" ]; then
    echo "[system] Found existing dataset at $BACKUP_PATH."
    ln -sf "$BACKUP_PATH" "$DATA_PATH"
else
    echo "[error] Dataset not found at $BACKUP_PATH"
    exit 1
fi

# 3) TPU Distillation Training
echo "[system] Starting TPU Training (3.5 Epochs)..."
# We start from scratch for TPU optimization
python train_artm_distill_tpu.py \
  --teacher_model microsoft/Phi-3.5-mini-instruct \
  --data_jsonl "$DATA_PATH" \
  --output_dir /kaggle/working/jaqua_distilled_tpu \
  --epochs 3.5 \
  --learning_rate 5e-4 \
  --per_device_batch_size 2 \
  --student_layers 36 \
  --student_hidden 1536 \
  --student_heads 24 \
  --student_ffn 6144 \
  --context_length 256

# 4) Export to GGUF
echo "[system] Converting to GGUF..."
python export_gguf.py \
  --student_dir /kaggle/working/jaqua_distilled_tpu \
  --gguf_out_dir /kaggle/working/gguf_tpu \
  --llama_cpp_dir /kaggle/working/llama.cpp \
  --quant_type Q4_K_M

# 5) Benchmarking
echo "[system] Running Benchmarks..."
python benchmark_tokens.py \
  --student_hf_dir /kaggle/working/jaqua_distilled_tpu \
  --teacher_model microsoft/Phi-3.5-mini-instruct \
  --teacher_load_in_4bit \
  --eval_jsonl "$DATA_PATH" \
  --max_eval_samples 512 \
  --gguf_model_path /kaggle/working/gguf_tpu/jaqua-q4_k_m.gguf \
  --n_ctx 2048 \
  --report_json /kaggle/working/jaqua_tpu_benchmark.json

echo "=========================================================="
echo "PIPELINE COMPLETE!"
echo "Model Location: /kaggle/working/jaqua_distilled_tpu"
echo "GGUF Location:  /kaggle/working/gguf_tpu/jaqua-q4_k_m.gguf"
echo "Benchmark:       /kaggle/working/jaqua_tpu_benchmark.json"
echo "=========================================================="
