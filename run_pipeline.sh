#!/usr/bin/env bash
set -euo pipefail

# ARTM end-to-end pipeline for Kaggle T4:
# 1) Install deps
# 2) Generate synthetic teacher dataset
# 3) Train student from scratch with distillation (+ optional QAT)
# 4) Convert to GGUF and quantize to Q4_K_M
# 5) Benchmark quality + speed

# 1) Install deps
python -m pip install --upgrade pip
python -m pip install -r requirements_kaggle.txt

# Path detection for the pre-generated dataset
BACKUP_PATH="/kaggle/input/datasets/aaravmaloo6/final-dataaset/jaqua_teacher_data.jsonl"
DATA_PATH="/kaggle/working/jaqua_teacher_data.jsonl"

if [ -f "$BACKUP_PATH" ]; then
    echo "[system] Found existing dataset at $BACKUP_PATH."
    echo "[system] Linking to local workspace to save time..."
    ln -sf "$BACKUP_PATH" "$DATA_PATH"
else
    echo "=========================================================="
    echo "CRITICAL ERROR: DATASET NOT FOUND AT:"
    echo "$BACKUP_PATH"
    echo "----------------------------------------------------------"
    echo "Skipping generation to save your 12-hour quota."
    echo "Please ensure the dataset is attached to this notebook."
    echo "=========================================================="
    exit 1
fi

echo "[system] Starting distillation training..."
python train_artm_distill.py \
  --teacher_model microsoft/Phi-3.5-mini-instruct \
  --data_jsonl "$DATA_PATH" \
  --output_dir /kaggle/working/jaqua_distilled \
  --student_layers 36 \
  --student_hidden 1536 \
  --student_heads 24 \
  --student_ffn 6144 \
  --context_length 384 \
  --temperature 2.0 \
  --loss_weight_ce 1.0 \
  --loss_weight_kd 1.0 \
  --loss_weight_hidden 0.0 \
  --epochs 3.5 \
  --learning_rate 5e-4 \
  --per_device_batch_size 1 \
  --gradient_accumulation_steps 64 \
  --save_steps 50 \
  --bf16 \
  --gradient_checkpointing \
  --teacher_load_in_4bit \
  --post_prune_epochs 1.0

python export_gguf.py \
  --student_dir /kaggle/working/jaqua_distilled/final_student \
  --gguf_out_dir /kaggle/working/gguf \
  --llama_cpp_dir /kaggle/working/llama.cpp \
  --quant_type Q4_K_M

python benchmark_tokens.py \
  --student_hf_dir /kaggle/working/jaqua_distilled/final_student \
  --teacher_model microsoft/Phi-3.5-mini-instruct \
  --teacher_load_in_4bit \
  --eval_jsonl /kaggle/working/jaqua_teacher_data.jsonl \
  --max_eval_samples 512 \
  --gguf_model_path /kaggle/working/gguf/jaqua-q4_k_m.gguf \
  --n_ctx 2048 \
  --n_threads 4 \
  --speed_runs 5 \
  --speed_max_tokens 128 \
  --report_json /kaggle/working/jaqua_benchmark.json

echo "Pipeline complete."
echo "Student HF: /kaggle/working/jaqua_distilled/final_student"
echo "GGUF Q4_K_M: /kaggle/working/gguf/jaqua-q4_k_m.gguf"
echo "Benchmark report: /kaggle/working/jaqua_benchmark.json"
