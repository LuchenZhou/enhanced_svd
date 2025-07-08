#!/usr/bin/env bash
# scripts/longbench.sh   （最终版）
set -euo pipefail

model=$1                             # 必填
head64_layers=${2:-"28,29,30,31"}    # 默认为末 4 层

python -u eval/LongBench/pred_svdhead.py \
        --model "$model" \
        --task dummy \
        --compress_svd \
        --svd_rank 512 \
        --head64_layers "$head64_layers" \
        --compress_only          # ← 最后一行不加反斜杠

