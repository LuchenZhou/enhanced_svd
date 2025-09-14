# SEDPA: SVD-ENHANCED DUAL PATH ATTENTION FOR EFFICIENT INFERENCE


## Abstract
Long context large language model inference is constrained by key value (KV) cache memory and latency. Existing methods such as windowing and uniform low rank compression reduce computation or the KV footprint, but they often impair long range retrieval or require a subset of heads to keep full context, which limits efficiency at long lengths. We present SEDPA, a Singular Value Decomposition (SVD) enhanced dual path attention framework with two variants: SEDPA S, which applies SVD to compress projections and speed up inference; and SEDPA W, which extends SEDPA S with a sliding window to further lower latency at small accuracy cost. SEDPA supports task dependent tradeoffs between accuracy and speed. On Llama 2 7B 32K and Llama 3 8B 1048K, SEDPA S reduces parameters by 11% to 15% while keeping Needle in a Haystack (NIAH) accuracy within 1.1 percentage points of DuoAttention and achieving comparable LongBench macro scores. SEDPA W maintains accuracy and reduces decode latency, KV cache memory, and peak memory by 10% to 25%.

#### Environment Setup

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

# LongBench evaluation
pip install seaborn rouge_score einops pandas

pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/

# Install DuoAttention
pip install -e .

# Install Block Sparse Streaming Attention
git clone https://github.com:mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
python setup.py install
```


## Quick Start for DuoAttention
We offer a simple one-click patch to enable DuoAttention optimization on HuggingFace models, including Llama and Mistral. Pretrained retrieval head patterns for five long-context models are available in the `attn_patterns` directory: `Llama-2-7B-32K-Instruct`, `Llama-3-8B-Instruct-Gradient-1048k`, `Llama-3-8B-Instruct-Gradient-4194k`, `Mistral-7B-Instruct-v0.2`, `Mistral-7B-Instruct-v0.3`, and `Meta-Llama-3.1-8B-Instruct`. If you'd like to train your own retrieval head patterns, you can use the training script provided in the scripts directory. Below is an example of how to enable DuoAttention on the `Llama-3-8B-Instruct-Gradient-1048k` model.


```python
from duo_attn.utils import load_attn_pattern, sparsify_attention_heads
from duo_attn.patch import enable_duo_attention_eval
import transformers
import torch

# Load the model
model = transformers.AutoModelForCausalLM.from_pretrained(
    "models/Llama-3-8B-Instruct-Gradient-1048k",
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="eager",
)

# Load the attention pattern
attn_heads, sink_size, recent_size = load_attn_pattern(
    "attn_patterns/Llama-3-8B-Instruct-Gradient-1048k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
)

# Sparsify attention heads
attn_heads, sparsity = sparsify_attention_heads(attn_heads, sparsity=0.5)

# Enable DuoAttention
enable_duo_attention_eval(
    model,
    attn_heads,
    sink_size=64,
    recent_size=256,
)

# Move model to GPU
model = model.cuda()

# Ready for inference!
```

## Demo
After setting up the environment, you can run the following script to execute the W4A8KV4 with DuoAttention demo on the `Llama-3-8B-Instruct-Gradient-4194k` model. The demo is designed to run on a single A100 GPU and supports a context length of up to 3.3 million tokens.

```bash
bash scripts/run_demo.sh
```

## Results 

### Retrieval Head Identification
After preparing the dataset and models, you can run the training script to identify the retrieval heads. For the models we evaluated, the corresponding attention patterns are available in the `attn_patterns` directory.

```bash
bash scripts/run_train.sh
```

### Needle in a Haystack (NIAH)
DuoAttention provides comparable accuracy as full attention on the Needle-in-a-Haystack benchmark using 25% full attention ratio on the MHA model and 50% full attention ratio on the GQA model.

```bash
bash scripts/run_niah.sh
```

![niah](figures/niah.jpg)

### LongBench

```bash
bash scripts/run_longbench.sh
```

DuoAttention provides better KV budget and accuracy trade-off on LongBench benchmarks.

![longbench](figures/longbench.jpg)

### Efficiency

```bash
bash scripts/run_efficiency.sh
```

- Per-token decoding latency and memory usage of DuoAttention compared to full attention across varying context sizes. DuoAttention uses a 25% retrieval head ratio for Llama-2-7B (MHA) and 50% for Llama-3-8B (GQA). DuoAttention achieves up to 2.45× memory reduction for MHA and 1.65× for GQA models, along with up to 2.13× latency reduction for MHA and 1.5× for GQA models. These reductions approach the inverse of the retrieval head ratios as context length increases. Out-of-memory (OOM) results are linearly extrapolated from measured data.

![efficiency_decoding](figures/efficiency_decoding.jpg)

- Pre-filling latency and memory usage of DuoAttention compared to full attention across varying
pre-filling chunk sizes. DuoAttention uses a 25% retrieval head ratio for Llama-2-7B (MHA), pre-filling a context of 100K tokens, and a 50% ratio for Llama-3-8B (GQA), pre-filling a context of 320K tokens. As the pre-filling chunk size decreases, DuoAttention achieves up to 1.73× latency reduction for MHA and 1.63× for GQA models, with memory reductions up to 2.38× for MHA and 1.53× for GQA models.

![efficiency_prefilling](figures/efficiency_prefilling.jpg)

- DuoAttention’s decoding memory and latency vs. KV budget with a fixed context length. Memory and latency are reduced linearly when the ratio of retrieval heads is reduced. DuoAttention
achieves up to 2.55× memory reduction for MHA and 1.67× for GQA models, along with up to 2.18× latency reduction for MHA and 1.50× for GQA models.

![efficiency_curve](figures/efficiency_curve.jpg)

- Combined with 8-bit weight and 4-bit KV cache quantization, DuoAttention can accommodate 3.3 million tokens on a single A100-80G GPU for the Llama-3-8B model.

<p align="center">
<img src="figures/kv_capacity.jpg" alt="kv_capacity" width="400"/>
</p>

## Citation

If you find DuoAttention useful or relevant to your project and research, please kindly cite our paper:

```bibtex
@article{xiao2024duo,
        title={DuoAttention: Efficient Long-Context LLM Inference with Retrieval and Streaming Heads},
        author={Xiao, Guangxuan and Tang, Jiaming and Zuo, Jingwei and Guo, Junxian and Yang, Shang and Tang, Haotian and Fu, Yao and Han, Song},
        journal={arXiv},
        year={2024}
}
```
