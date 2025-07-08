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
from w8a8kv4_llama_2 import (
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

# Model setup
model_name = "models/Llama-3-8B-Instruct-Gradient-4194k-w8a8kv4-per-channel"
hf_config = AutoConfig.from_pretrained(model_name)
sampling_params = SamplingParams(
    temperature=0.0, top_p=1.0, stop_token_ids=[128001, 128009], max_tokens=50
)

# Specify the path to the precomputed SVD right singular vectors
svd_path = "models/llama_svd_right_vectors.pt"

# Initialize the model with the SVD path and quantization path
model = LlamaForCausalLMW8A8(
    hf_config, sampling_params, quant_path=model_name, svd_path=svd_path
).half().to("cuda")

# Load tokenizer and generation config
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
generation_config = GenerationConfig.from_pretrained(model_name)
eos_token_ids = generation_config.eos_token_id
if not isinstance(eos_token_ids, list):
    eos_token_ids = [eos_token_ids]

# Add special tokens like "</user>" and "</s>" to eos ids
eos_token_ids += tokenizer.encode("</user>", add_special_tokens=False)
eos_token_ids += tokenizer.encode("</s>", add_special_tokens=False)
eos_token_ids += tokenizer.encode("</", add_special_tokens=False)

# Load attention patterns
attn_load_dir = "attn_patterns/Llama-3-8B-Instruct-Gradient-4194k/lr=0.02-reg=0.05-ctx=1000_32000-multi_passkey10"
full_attention_heads, sink_size, recent_size = load_attn_pattern(attn_load_dir)

# Argument parsing for sparsity and insertion point
parser = argparse.ArgumentParser()
parser.add_argument("--len", type=int, default=100)
parser.add_argument("--sparsity", type=float, default=0.5)
parser.add_argument("--insertion_point", type=float, default=0.75)
args = parser.parse_args()

# Sparsify attention heads
full_attention_heads, sparsity = sparsify_attention_heads(
    full_attention_heads, None, args.sparsity
)

if sparsity > 0:
    print(f"Using DuoAttention with {sparsity} sparsity.")
else:
    print("Using Full Attention.")

# Enable DuoAttention evaluation
enable_llama_duo_attention_eval(
    model,
    full_attention_heads,
    sink_size,
    recent_size,
)

# Context setup for text generation
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

        used_mem = torch.cuda.max_memory_allocated() / (1024**3)  # Convert to GB
        latency_per_token = total_latency / (len(generated_content) - 1)

        generated_text = tokenizer.decode(
            generated_content,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
            spaces_between_special_tokens=False,
        ).strip()

        output = f"Decoding (Mem: {used_mem:.1f} GB | Latency: {latency_per_token:.1f} ms/tok): {generated_text}"

        terminal_width = shutil.get_terminal_size().columns
        lines = (len(output) + terminal_width - 1) // terminal_width - 1

        print("\r" + "\033[K", end="")  # Clear current line
        for _ in range(previous_lines):
            print("\033[F\033[K", end="")  # Move cursor up and clear line

        print(output, end="", flush=True)
        previous_lines = lines

    print(f"\n\nPer-token decoding latency: {total_latency / (len(generated_content) - 1):.2f} ms")
    return tokenizer.decode(generated_content, skip_special_tokens=False).strip()


# 修改 `split_sizes` 确保其总和为12288
def fix_split_sizes(input_tensor, split_sizes):
    total_size = input_tensor.size(-1)  # 获取输入张量的最后一维大小
    if sum(split_sizes) != total_size:
        print(f"Warning: Split sizes sum ({sum(split_sizes)}) does not match input tensor size ({total_size}). Adjusting.")
        # 动态计算拆分大小，确保其总和为 total_size
        num_parts = len(split_sizes)
        adjusted_split_sizes = [total_size // num_parts] * num_parts
        adjusted_split_sizes[-1] += total_size % num_parts  # 确保总和正确
        return adjusted_split_sizes
    return split_sizes


# Run the model with chunked prefilling
def test_with_chunked_prefilling(chunk_size=32000):
    kv_cache = DuoAttentionStaticINT4KVCache(
        model=model,
        full_attention_heads=full_attention_heads,
        batch_size=1,
        max_size=input_ids.size(1) + 550,
        sink_size=sink_size,
        recent_size=recent_size,
        prefilling_chunk_size=32000,
    )

    start_time = time.time()
    with torch.no_grad():
        pbar = tqdm(
            range(0, input_ids.size(1), chunk_size),
            desc=f"Pre-filling ({0}/{input_ids.size(1)})",
        )
        for i in pbar:
            chunk_input_ids = input_ids[:, i : i + chunk_size]
            
            # 确保在调用 model 之前调整 split_sizes
            split_sizes = [4096, 1024, 1024]  # 假设这是原始的 split_sizes
            adjusted_split_sizes = fix_split_sizes(chunk_input_ids, split_sizes)  # 调整 split_sizes
            
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
