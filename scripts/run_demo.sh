export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=4 python demo/run_duo_w8a8kv4_1.py --len 1000000 --sparsity 0.5
