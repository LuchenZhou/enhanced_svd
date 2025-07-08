#!/usr/bin/env bash
set -euo pipefail

models="Llama-2-7B-32K-Instruct Llama-3-8B-Instruct-Gradient-1048k"
HEAD64="28,29,30,31"

for model in $models; do
    ckpt="svd_head64_sel/${model}/compressed_model.pth"
    if [[ -f "$ckpt" ]]; then
        echo "✓ [$model] ckpt exists — skip"
    else
        CUDA_VISIBLE_DEVICES=0 bash scripts/svd.sh "$model" "$HEAD64"
    fi
done

echo "✓ All requested models have compressed weights."
