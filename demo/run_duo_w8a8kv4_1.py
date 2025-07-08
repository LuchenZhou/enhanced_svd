import warnings

warnings.filterwarnings("ignore")

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoConfig,
    AutoTokenizer,
    GenerationConfig,
)
import time
import argparse
from tqdm import tqdm, trange
from w8a8kv4_llama import (
    enable_llama_duo_attention_eval,
    LlamaForCausalLM as LlamaForCausalLMW8A8,
)
from qserve import SamplingParams
from int4_kv import DuoAttentionStaticINT4KVCache
from qserve.utils.input_metadata import InputMetadata

from duo_attn.utils import (
    load_attn_pattern,
    sparsify_attention_heads,
)
from transformers.utils import logging

logging.set_verbosity_error()

import shutil

torch.cuda.memory._record_memory_history()

model_name = "models/Llama-3-8B-Instruct-Gradient-4194k-w8a8kv4-per-channel"
hf_config = AutoConfig.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=50
)
model = (
    LlamaForCausalLMW8A8(hf_config, sampling_params, quant_path=model_name)
    .half()
    .to("cuda")
)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
generation_config = GenerationConfig.from_pretrained(model_name)
eos_token_ids = generation_config.eos_token_id
if not isinstance(eos_token_ids, list):
    eos_token_ids = [eos_token_ids]

# add some tokens like "</user>" and </s> to eos ids
eos_token_ids += tokenizer.encode("</user>", add_special_tokens=False)
eos_token_ids += tokenizer.encode("</s>", add_special_tokens=False)
eos_token_ids += tokenizer.encode("</", add_special_tokens=False)

attn_load_dir = "attn_patterns/Llama-3-8B-Instruct-Gradient-4194k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"

full_attention_heads, sink_size, recent_size = load_attn_pattern(attn_load_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, default=100)
parser.add_argument("--sparsity", type=float, default=0.5)
parser.add_argument("--insertion_point", type=float, default=0.75)
args = parser.parse_args()

full_attention_heads, sparsity = sparsify_attention_heads(
    full_attention_heads, None, args.sparsity
)

if sparsity > 0:
    print(f"Using DuoAttention with {sparsity} sparsity.")
else:
    print("Using Full Attention.")

enable_llama_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
)

context = "A quick brown fox jumps over the lazy dog. \n"
with open("demo/duo_attention.txt", "r") as f:
    needle = f.read()

num_tokens_context = len(tokenizer.encode(context, add_special_tokens=False))
num_repetitions = args.len // num_tokens_context

text = (
    "This is a very long story book: <book> "
    + context * int(num_repetitions * args.insertion_point)
    + needle
    + context * int(num_repetitions * (1 - args.insertion_point))
    + "</book>\n Based on the content of the book, please briefly tell me about DuoAttention.\nAnswer:"
)

input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")

print(f"Input sequence length: {input_ids.size(1)}\n")


# SVD压缩注意力矩阵的函数
def apply_svd_to_attention_heads(attn_weights, rank=32):
    """
    使用SVD压缩注意力头，减少内存和计算复杂度。
    
    Parameters:
    attn_weights (Tensor): 原始的注意力权重矩阵，形状为 (num_heads, seq_len, seq_len)
    rank (int): SVD的秩，用于控制压缩的程度，通常设置为比原始秩小的值
    
    Returns:
    Tensor: 压缩后的注意力权重矩阵
    """
    U, S, V = torch.svd(attn_weights.view(-1, attn_weights.size(-1)))  # 先展开为2D矩阵
    U = U[:, :rank]  # 取前rank个奇异值对应的左奇异矩阵
    S = S[:rank]  # 取前rank个奇异值
    V = V[:, :rank]  # 取前rank个右奇异矩阵
    compressed_weights = torch.mm(U, torch.mm(torch.diag(S), V.t()))  # 重建压缩后的矩阵
    return compressed_weights.view(attn_weights.size())


# 将SVD压缩功能加入KV缓存
class OptimizedDuoAttentionCache(DuoAttentionStaticINT4KVCache):
    def __init__(self, *args, svd_rank=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.svd_rank = svd_rank

    def compress_kv_cache(self, kv_cache):
        """
        使用SVD压缩kv_cache中的注意力权重。
        """
        for key, value in kv_cache.items():
            if key == 'attn_weights':  # 假设'kv_cache'字典中存储了注意力权重
                kv_cache[key] = apply_svd_to_attention_heads(value, rank=self.svd_rank)
        return kv_cache

    def forward(self, input_ids, kv_cache):
        kv_cache = self.compress_kv_cache(kv_cache)
        return super().forward(input_ids, kv_cache)


# 修改生成函数，使用经过SVD压缩的KV缓存
@torch.no_grad()
def generate_with_kv_cache(model, kv_cache, pred_token_idx, eos_token_ids, tokenizer):
    total_latency = 0
    generated_content = [pred_token_idx.item()]
    previous_lines = 0

    print("Generated text (Mem: N/A | Time: N/A):", end=" ", flush=True)

    for _ in range(500):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        logits = model(
            input_ids=pred_token_idx,
            kv_cache=kv_cache,
        )
        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)
        total_latency += elapsed_time

        pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
        if pred_token_idx.item() in eos_token_ids:
            break
        generated_content += [pred_token_idx.item()]
        # Capture memory usage using torch.cuda.max_memory_allocated()
        used_mem = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        latency_per_token = total_latency / (
            len(generated_content) - 1
        )  # Latency in ms

        generated_text = tokenizer.decode(
            generated_content,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        ).strip()

        output = f"Decoding (Mem: {used_mem:.1f} GB | Latency: {latency_per_token:.1f} ms/tok): {generated_text}"

        terminal_width = shutil.get_terminal_size().columns

        lines = (len(output) + terminal_width - 1) // terminal_width - 1

        print("\r" + "\033[K", end="")
        for _ in range(previous_lines):
            print("\033[F\033[K", end="")

        print(
            output,
            end="",
            flush=True,
        )

        previous_lines = lines

    print(
        f"\n\nPer-token decoding latency: {total_latency / (len(generated_content) - 1):.2f} ms"
    )
    return tokenizer.decode(generated_content, skip_special_tokens=False).strip()


# with chunked prefilling
def test_with_chunked_prefilling(chunk_size=32000):
    kv_cache = OptimizedDuoAttentionCache(
        model=model,
        full_attention_heads=full_attention_heads,
        batch_size=1,
        max_size=input_ids.size(1) + 550,
        sink_size=sink_size,
        recent_size=recent_size,
        prefilling_chunk_size=32000,
        svd_rank=32,  # 设置SVD秩
    )

    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(
            range(0, input_ids.size(1), chunk_size),
            desc=f"Pre-filling ({0}/{input_ids.size(1)})",
        )
        for i in pbar:
            chunk_input_ids = input_ids[:, i : i + chunk_size]
            logits = model(
                input_ids=chunk_input_ids,
                kv_cache=kv_cache,
            )
            used_mem = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
            pbar.set_description(
                f"Pre-filling ({min(i + chunk_size, input_ids.size(1)) // 1000}K/{input_ids.size(1)//1000}K, Mem: {used_mem:.1f} GB)"
            )
        pbar.close()
    end_time = time.time()
    print(f"Pre-filling time: {end_time - start_time:.2f}s\n")

    pred_token_idx = logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
    _ = generate_with_kv_cache(
        model, kv_cache, pred_token_idx, eos_token_ids, tokenizer
    )


torch.cuda.reset_peak_memory_stats()
test_with_chunked_prefilling(32000)
used_mem = torch.cuda.max_memory_allocated()
print(f"Peak memory: {used_mem / 1024 ** 3:.2f} GB")
