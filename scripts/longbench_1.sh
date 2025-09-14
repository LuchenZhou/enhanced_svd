model=$1
task=$2
attn_pattern=$3
sparsity=$4
python -u eval/LongBench/pred_weights.py \
    --model $model --task $task \
    --method duo_attn \
    --attn_load_dir ${attn_pattern} \
    --sparsity $sparsity \
    --sink_size 64 \
    --recent_size 256 \
    --compress_svd \
    --svd_rank 1024 \
    --compress_kv_cache \
    --kv_max_length 10000 \
    --kv_sliding_window 4096 \
    --kv_sink_tokens 256 \
    --kv_compression_ratio 0.0625
