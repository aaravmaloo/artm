#!/usr/bin/env bash
set -euo pipefail

# ARTM Interactive TPU Pipeline - V4.3 (Native Discovery)
echo "=========================================================="
echo "      ARTM INTERACTIVE TPU PIPELINE - V4.3"
echo "=========================================================="

# 2) Dataset Setup
DATA_PATH="/kaggle/working/jaqua_teacher_data.jsonl"
ln -sf "/kaggle/input/datasets/aaravmaloo6/final-dataset/jaqua_teacher_data.jsonl" "$DATA_PATH"

# 3) TPU Training (Native Discovery)
# No manual env vars here - the Python script will clean itself.
echo "[system] Starting 8-Core TPU v5e-8 Training..."

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
