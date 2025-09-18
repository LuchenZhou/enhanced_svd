model=$1
task=$2
attn_pattern=$3
sparsity=$4
python -u eval/LongBench/pred_sedpa.py \
  --model "$model" --task "$task" \
  --method duo_attn \
  --attn_load_dir "$attn_pattern" \
  --sparsity "$sparsity" \
  --compress_svd \
  --svd_rank 1024 \
  --svd_eta_retr 0.98 --svd_eta_stream 0.95 \
  --svd_skip_first_layers 8 --lambda_skip 0.5 \
  --window_s 256 --window_m 512 --window_w 4096
