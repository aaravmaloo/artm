#!/usr/bin/env bash
set -euo pipefail

# ARTM Interactive TPU Pipeline - V4.1 (Purified PJRT)
echo "=========================================================="
echo "      ARTM INTERACTIVE TPU PIPELINE - V4.1"
echo "=========================================================="

# 1) Sync versions
python -m pip install torch==2.8.0 torch_xla[tpu]==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html
python -m pip install --upgrade transformers accelerate

# --- PURIFIED PJRT ENV ---
# We UNSET the old XRT variables that trigger the "Slice Builder" error
unset XRT_TPU_CONFIG
unset TPU_CONFIG
unset TPU_CHIPS_PER_HOST_BOUNDS

# We set the ONLY variable needed for modern PJRT
export PJRT_DEVICE=TPU
# -------------------------

# 2) Dataset Setup
DATA_PATH="/kaggle/working/jaqua_teacher_data.jsonl"
ln -sf "/kaggle/input/datasets/aaravmaloo6/final-dataset/jaqua_teacher_data.jsonl" "$DATA_PATH"

# 3) TPU Training
echo "[system] Starting Training (PJRT Mode)..."

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
