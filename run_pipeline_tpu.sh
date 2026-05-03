#!/usr/bin/env bash
set -euo pipefail

# ARTM Interactive TPU Pipeline
export PJRT_DEVICE=TPU

echo "=========================================================="
echo "      ARTM INTERACTIVE TPU PIPELINE - V3.8"
echo "=========================================================="

# 1) Install dependencies (Force matching versions)
echo "[system] Synchronizing Torch & XLA versions (2.8.1)..."
python -m pip install --upgrade pip
python -m pip install torch==2.8.1 torch_xla[tpu]==2.8.1 -f https://storage.googleapis.com/libtpu-releases/index.html
python -m pip install --upgrade transformers accelerate

# 2) Dataset Setup
BACKUP_PATH="/kaggle/input/datasets/aaravmaloo6/final-dataset/jaqua_teacher_data.jsonl"
DATA_PATH="/kaggle/working/jaqua_teacher_data.jsonl"

if [ -f "$BACKUP_PATH" ]; then
    echo "[system] Linking dataset: $BACKUP_PATH -> $DATA_PATH"
    ln -sf "$BACKUP_PATH" "$DATA_PATH"
else
    echo "[error] Dataset not found! Please check your Kaggle Input path."
    exit 1
fi

# 3) TPU Training
# We use per_device_batch_size 2 for safety
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
