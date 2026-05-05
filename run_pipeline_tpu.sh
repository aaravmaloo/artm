#!/usr/bin/env bash
set -euo pipefail

# ARTM Interactive TPU Pipeline - V4.3 (Native Discovery)
echo "=========================================================="
echo "      ARTM INTERACTIVE TPU PIPELINE - V4.3"
echo "=========================================================="

# 1) System Dependencies
echo "[system] Installing dependencies..."
python -m pip install torch==2.8.0 torch_xla[tpu]==2.8.0 -f https://storage.googleapis.com/libtpu-releases/index.html
python -m pip install --upgrade transformers accelerate cmake

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
  --per_device_batch_size 1 \
  --student_layers 36 \
  --student_hidden 1536 \
  --student_heads 24 \
  --student_ffn 6144 \
  --context_length 256

echo "=========================================================="
echo "      TPU TRAINING COMPLETE - STARTING POST-PROCESSING"
echo "=========================================================="

python export_gguf.py \
  --student_dir /kaggle/working/jaqua_distilled_tpu \
  --gguf_out_dir /kaggle/working \
  --llama_cpp_dir /kaggle/working/llama.cpp \
  --quant_type Q4_K_M

python benchmark_tokens.py \
  --student_hf_dir /kaggle/working/jaqua_distilled_tpu \
  --teacher_model microsoft/Phi-3.5-mini-instruct \
  --teacher_load_in_4bit \
  --eval_jsonl /kaggle/working/jaqua_teacher_data.jsonl \
  --max_eval_samples 512 \
  --gguf_model_path /kaggle/working/jaqua-q4_k_m.gguf \
  --n_ctx 2048 \
  --n_threads 4 \
  --speed_runs 5 \
  --speed_max_tokens 128 \
  --report_json /kaggle/working/jaqua_benchmark_tpu.json

echo "=========================================================="
echo "      CLEANING UP KAGGLE OUTPUT SPACE"
echo "=========================================================="
rm -rf /kaggle/working/llama.cpp
rm -rf /kaggle/working/jaqua_distilled_tpu

echo "Pipeline complete."
echo "Final GGUF Q4_K_M: /kaggle/working/jaqua-q4_k_m.gguf"
echo "Benchmark report: /kaggle/working/jaqua_benchmark_tpu.json"
