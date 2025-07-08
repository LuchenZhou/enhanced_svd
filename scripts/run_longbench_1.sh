#!/bin/bash

attn_pattern_name="lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
models="Llama-3-8B-Instruct-Gradient-1048k"
sparsities="0 0.5 0.75"

tasks="qasper triviaqa"

for model in $models; do
    for task in $tasks; do
        for sparsity in $sparsities; do
            bash scripts/longbench.sh $model $task "attn_patterns/${model}/${attn_pattern_name}" $sparsity
        done
    done
done

# 可选：运行评估脚本
cd eval/LongBench
for model in $models; do
    python -u eval.py --model $model &
done
