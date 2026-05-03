#!/usr/bin/env bash
set -euo pipefail

# ARTM Interactive TPU Pipeline - V4.0 (Local Handshake)
echo "=========================================================="
echo "      ARTM INTERACTIVE TPU PIPELINE - V4.0"
echo "=========================================================="

# 1) Sync versions (Match 2.8.0)
python -m pip install torch==2.8.0 torch_xla[tpu]==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html
python -m pip install --upgrade transformers accelerate

# --- THE MAGIC FLAGS ---
export PJRT_DEVICE=TPU
export TPU_NAME=local
export CLOUD_TPU_TASK_ID=0
export TPU_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# -----------------------

# 2) Dataset Setup
DATA_PATH="/kaggle/working/jaqua_teacher_data.jsonl"
ln -sf "/kaggle/input/datasets/aaravmaloo6/final-dataset/jaqua_teacher_data.jsonl" "$DATA_PATH"

# 3) TPU Training (Back to standard execution)
echo "[system] Starting 8-Core Training..."

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
python export_gguf.py \
  --student_dir /kaggle/working/jaqua_distilled_tpu \
  --gguf_out_dir /kaggle/working/gguf_tpu \
  --llama_cpp_dir /kaggle/working/llama.cpp \
  --quant_type Q4_K_M
