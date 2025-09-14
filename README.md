# SEDPA: SVD-ENHANCED DUAL PATH ATTENTION FOR EFFICIENT INFERENCE


## Abstract
Long context large language model inference is constrained by key value (KV) cache memory and latency. Existing methods such as windowing and uniform low rank compression reduce computation or the KV footprint, but they often impair long range retrieval or require a subset of heads to keep full context, which limits efficiency at long lengths. We present SEDPA, a Singular Value Decomposition (SVD) enhanced dual path attention framework with two variants: SEDPA S, which applies SVD to compress projections and speed up inference; and SEDPA W, which extends SEDPA S with a sliding window to further lower latency at small accuracy cost. SEDPA supports task dependent tradeoffs between accuracy and speed. On Llama 2 7B 32K and Llama 3 8B 1048K, SEDPA S reduces parameters by 11% to 15% while keeping Needle in a Haystack (NIAH) accuracy within 1.1 percentage points of DuoAttention and achieving comparable LongBench macro scores. SEDPA W maintains accuracy and reduces decode latency, KV cache memory, and peak memory by 10% to 25%.

#### Environment

```bash
conda create -yn duo python=3.10
conda activate duo

conda install -y git
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y nvidia::cuda-cudart-dev
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia

pip install transformers==4.45.2 accelerate sentencepiece datasets wandb zstandard matplotlib huggingface_hub==0.25.2
pip install tensor_parallel==2.0.0

pip install ninja packaging
pip install flash-attn==2.6.3 --no-build-isolation

pip install seaborn rouge_score einops pandas

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

pip install -e .

git clone https://github.com:mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
python setup.py install
```
**Model**
```bash
huggingface-cli download togethercomputer/Llama-2-7B-32K-Instruct --local-dir Llama-2-7B-32K-Instruct
huggingface-cli download gradientai/Llama-3-8B-Instruct-Gradient-1048k --local-dir Llama-3-8B-Instruct-Gradient-1048k
```

## Experiment
#motivation
```bash
python needle_in_haystack_with_mask.py --mask_top 30 --s 1000 --e 100000  --model_path $path_to_model  
python needle_in_haystack_with_mask.py --mask_top -30 --s 1000 --e 100000  --model_path $path_to_model  
python head_score.py
```
#train
```bash
bash scripts/run_train.sh
```
### Needle-in-a-Haystack (NIAH)

```bash
bash scripts/run_niah.sh
```

### LongBench

```bash
bash scripts/run_longbench.sh

bash scripts/run_longbench_1.sh
```

### Efficiency

```bash
bash scripts/run_efficiency.sh
```

