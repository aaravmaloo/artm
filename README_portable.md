# ARTM Distillation Pipeline (Kaggle T4 -> Raspberry Pi 4)

ARTM is a distilled conversational model trained from scratch with a student architecture and Phi-3.5 as teacher.

## What this pipeline does
1. Generates synthetic teacher conversations from:
   - `HuggingFaceH4/ultrachat_200k`
   - `teknium/OpenHermes-2.5`
   - `lmsys/lmsys-chat-1m`
2. Trains a new student model from config (default ~1.12B params GPT-style):
   - CE loss on teacher outputs
   - Logit MSE distillation (`T=2.0`)
   - Layer-wise hidden-state distillation
   - Optional attention-pattern distillation
3. Applies optional fake-quantization-aware training for 4-bit robustness.
4. Optionally prunes low-importance attention heads and runs a short recovery epoch.
5. Converts to GGUF and quantizes to `Q4_K_M`.
6. Benchmarks quality and speed.

## Scripts
- `artm_generate_teacher_data.py`: synthetic dataset generation with teacher outputs + top-k logits.
- `train_artm_distill.py`: student-from-scratch distillation + optional QAT + optional pruning.
- `export_gguf.py`: HF checkpoint -> GGUF -> `Q4_K_M` quantization.
- `benchmark_tokens.py`: perplexity gap, BLEU, GGUF tok/s benchmark.
- `run_pipeline.sh`: full end-to-end command sequence.

## One-command run (Kaggle)
```bash
bash run_pipeline.sh
```

## Key default training settings
- Student: `36` layers, `1536` hidden, `24` heads, context `2048` (~1.12B params)
- Batch: `4`, grad accumulation: `16`
- LR: `5e-4`, cosine schedule
- Epochs: `3`
- Precision: `bf16`
- QAT: enabled (fake 4-bit)
- Pruning: 25% heads + 1 recovery epoch

## Outputs
- Student HF checkpoint:
  - `/kaggle/working/artm_distilled/final_student`
- Quantized GGUF:
  - `/kaggle/working/gguf/artm-q4_k_m.gguf`
- Benchmark report:
  - `/kaggle/working/artm_benchmark.json`

## Raspberry Pi 4 inference
```bash
python infer_llama_cpp.py \
  --model_path /path/to/artm-q4_k_m.gguf \
  --n_ctx 2048 \
  --n_threads 4
```

## Notes
- This pipeline trains a new student model from random initialization. It does not use LoRA/adapter fine-tuning.
- If `llama-cpp-python` is unavailable in the benchmark environment, quality metrics still run and speed benchmarking is skipped automatically.