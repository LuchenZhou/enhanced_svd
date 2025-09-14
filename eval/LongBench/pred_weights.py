
# -*- coding: utf-8 -*-
import os
from datasets import load_dataset
import torch
import json
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
)
from tqdm import tqdm
import numpy as np
import random
import argparse
import signal
from contextlib import contextmanager
import math

from duo_attn.patch import enable_duo_attention_eval
from duo_attn.patch.llama import enable_llama_duo_attention_static_kv_cache_eval
from duo_attn.patch.static_kv_cache import DuoAttentionStaticKVCache
from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache
import sys, pathlib
#  .../eval/LongBench/pred_weights.py
root_dir = pathlib.Path(__file__).resolve().parents[1]    # -> .../eval/
eff_dir  = root_dir / "efficiency"                        # .../eval/efficiency
sys.path.append(str(eff_dir))

from utils import bench_func

class TimeoutException(Exception): pass

# KV-Cache压缩类
class CompressedKVCache:

    def __init__(self, 
                 max_cache_length=8192,
                 quantize_bits=8,
                 sliding_window=4096,
                 sink_tokens=256,
                 compression_ratio=0.5,
                 verbose=False):
        self.max_cache_length = max_cache_length
        self.quantize_bits = quantize_bits  # 8 for INT8, 4 for INT4
        self.sliding_window = sliding_window
        self.sink_tokens = sink_tokens  
        self.compression_ratio = compression_ratio
        self.verbose = verbose
        
        self.memory_usage = 0   # MiB
        
        # 缓存存储
        self.compressed_cache = {}
        self.cache_metadata = {}
        self.current_length = 0

        # 安全检查
        assert self.sink_tokens   < self.max_cache_length
        assert self.sliding_window <= self.max_cache_length
    def quantize_tensor(self, tensor, bits=8):

        if bits == 8:
            # INT8量化
            scale = tensor.abs().max() / 127.0
            quantized = torch.round(tensor / scale).clamp(-128, 127).to(torch.int8)
            return quantized, scale
        elif bits == 4:
            # INT4量化
            scale = tensor.abs().max() / 7.0
            quantized = torch.round(tensor / scale).clamp(-8, 7).to(torch.int8)
            return quantized, scale
        else:
            return tensor, 1.0
    
    def dequantize_tensor(self, quantized_tensor, scale):

        return quantized_tensor.float() * scale
    
    def compress_kv_pair(self, key, value, layer_idx):

        # 量化K和V
        k_quantized, k_scale = self.quantize_tensor(key, self.quantize_bits)
        v_quantized, v_scale = self.quantize_tensor(value, self.quantize_bits)
        
        # 存储压缩后的KV和metadata
        compressed_kv = {
            'k_quantized': k_quantized,
            'v_quantized': v_quantized,
            'k_scale': k_scale,
            'v_scale': v_scale,
            'original_shape': key.shape
        }
        
        return compressed_kv
    
    def decompress_kv_pair(self, compressed_kv):
        k_dequantized = self.dequantize_tensor(
            compressed_kv['k_quantized'], 
            compressed_kv['k_scale']
        )
        v_dequantized = self.dequantize_tensor(
            compressed_kv['v_quantized'], 
            compressed_kv['v_scale']
        )
        
        return k_dequantized, v_dequantized
    
    def should_keep_token(self, position, total_length):
        # sink tokens
        if position < self.sink_tokens:
            return True
            
        # sliding_window
        if position >= total_length - self.sliding_window:
            return True
            
        # compression_ratio
        if position % int(1 / math.sqrt(self.compression_ratio)) == 0:
            return True
            
        return False
    
    def update_cache(self, past_key_values, new_key_values):
        if past_key_values is None:
            return new_key_values

        compressed_cache = []

        for layer_idx, ((past_k, past_v), (new_k, new_v)) in enumerate(
            zip(past_key_values, new_key_values)
        ):
            # 1
            combined_k = torch.cat([past_k, new_k], dim=2)          # [b,h,seq,d]
            combined_v = torch.cat([past_v, new_v], dim=2)
            seq_len    = combined_k.size(2)
            self.current_length = seq_len

            # keep_positions（python list）
            if seq_len > self.max_cache_length:
                keep_positions = [
                    p for p in range(seq_len) if self.should_keep_token(p, seq_len)
                ]

                # sink / recent / middle
                if len(keep_positions) > self.max_cache_length:
                    sink   = [p for p in keep_positions if p < self.sink_tokens]
                    recent = [p for p in keep_positions if p >= seq_len - self.sliding_window]
                    middle = [p for p in keep_positions
                            if p not in sink and p not in recent]
                    keep_positions = (sink + recent + middle)[: self.max_cache_length]
            else:
                keep_positions = list(range(seq_len))

            # 3
            keep_positions = [p for p in set(keep_positions) if 0 <= p < seq_len]
            keep_positions.sort()
            keep_positions = keep_positions[: self.max_cache_length]

            # 4 tensor
            kp_tensor = torch.tensor(
                keep_positions, device=combined_k.device, dtype=torch.long
            )

            # 5
            compressed_k = combined_k.index_select(2, kp_tensor)
            compressed_v = combined_v.index_select(2, kp_tensor)

            if self.verbose and seq_len > self.max_cache_length:
                print(f"Layer {layer_idx}: {seq_len} → {compressed_k.size(2)} "
                    f"tokens ({100*compressed_k.size(2)/seq_len:.1f}%)")

            compressed_cache.append((compressed_k, compressed_v))
        
        dtype_bytes = compressed_cache[0][0].element_size()      
        self.memory_usage = sum(k_.numel() + v_.numel()
                            for k_, v_ in compressed_cache) \
                        * dtype_bytes / 1024 / 1024          # → MiB
        return compressed_cache


# SVD
class SVDLinear(torch.nn.Module):
    def __init__(self, original_linear, rank):
        super().__init__()
        weight = original_linear.weight.data
        bias = original_linear.bias.data if original_linear.bias is not None else None
        
        d_out, d_in = weight.shape
        
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        r = min(rank, S.size(0), d_out, d_in)
        
        sqrt_s = torch.diag(S[:r].sqrt())
        A = U[:, :r] @ sqrt_s
        B = sqrt_s @ Vh[:r, :]
        
        self.lora_down = torch.nn.Linear(d_in, r, bias=False)
        self.lora_up = torch.nn.Linear(r, d_out, bias=bias is not None)
        
        with torch.no_grad():
            self.lora_down.weight.data = B.to(weight.dtype)
            self.lora_up.weight.data = A.to(weight.dtype)
        
        if bias is not None:
            self.lora_up.bias.data = bias.to(weight.dtype)
    
    def forward(self, x):
        return self.lora_up(self.lora_down(x))

def select_device(model_name):
    if "1048k" in model_name:
        for dev in [0]:
            if torch.cuda.memory_allocated(dev) < 0.8 * torch.cuda.get_device_properties(dev).total_memory:
                return f"cuda:{dev}"
    else:
        for dev in [0,1,3]:
            if torch.cuda.memory_allocated(dev) < 0.7 * torch.cuda.get_device_properties(dev).total_memory:
                return f"cuda:{dev}"
    return "cuda:0"

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def measure_efficiency(model, kv_cache, input_ids, prefilling_chunk_size):
    """ benchmark_static.py """
    # -------- prefill --------
    def prefill():
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                _ = model(
                    input_ids=input_ids[:, i:i+prefilling_chunk_size],
                    past_key_values=kv_cache,
                    use_cache=True,
                )
        kv_cache.clear()

    ctx_lat, ctx_mem = bench_func(prefill, num_steps=10, num_warmup_steps=3)

    # -------- prefill--------
    with torch.no_grad():
        for i in range(0, input_ids.size(1), prefilling_chunk_size):
            _ = model(
                input_ids=input_ids[:, i:i+prefilling_chunk_size],
                past_key_values=kv_cache,
                use_cache=True,
            )
    pred = _.logits[:, -1:].argmax(dim=-1)

    # -------- evict --------
    def generate():
        with torch.no_grad():
            _ = model(
                input_ids=pred,
                past_key_values=kv_cache,
                use_cache=True,
            )
        kv_cache.evict_last(1)

    gen_lat, gen_mem = bench_func(generate, num_steps=100, num_warmup_steps=50)

    kv_mem = kv_cache.memory_usage / 1024 / 1024
    return ctx_lat, ctx_mem, gen_lat, gen_mem, kv_mem

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--task", type=str, help="task name", required=True)
    parser.add_argument("--method", type=str, default="full")

    # duo attention
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--decoding_simulation_length", type=int, default=50)

    # SVD
    parser.add_argument("--compress_svd", action="store_true")
    parser.add_argument("--svd_rank", type=int, default=1024)
    
    # KV-Cache
    parser.add_argument("--compress_kv_cache", action="store_true", help="Enable KV cache compression")
    parser.add_argument("--kv_max_length", type=int, default=4096, help="Max KV cache length")
    parser.add_argument("--kv_sliding_window", type=int, default=1024, help="Sliding window size")
    parser.add_argument("--kv_sink_tokens", type=int, default=256, help="Always keep sink tokens")
    parser.add_argument("--kv_compression_ratio", type=float, default=0.3, help="Compression ratio for middle tokens")
    parser.add_argument("--kv_quantize_bits", type=int, default=16, help="Quantization bits (16/8/4)")
    
    return parser.parse_args(args)

def build_chat(tokenizer, prompt, model_name):
    if "llama-2" in model_name:
        prompt = f"[INST]{prompt}[/INST]"
    return prompt

def post_process(response, model_name):
    if "xgen" in model_name:
        response = response.strip().replace("Assistant:", "")
    elif "internlm" in model_name:
        response = response.split("<eoa>")[0]
    elif "llama-3" in model_name.lower():
        response = (
            response.split(".assistant")[0]
            .split("\n\nQuestion")[0]
            .split("</s>")[0]
            .strip()
        )
    elif "Llama-2-7B-32K-Instruct" in model_name:
        response = (
            response.split("(Document")[0]
            .split("\n\nQuestion")[0]
            .split("\n\nAnswer")[0]
            .split("(Passage")[0]
            .strip()
        )
    return response

def get_pred_with_kv_compression(model, tokenizer, eos_token_ids, data, max_length, max_gen, prompt_format, dataset, model_name, decoding_simulation_length, kv_compressor=None):
    print("first data...")
    test_input = "This is a test"
    test_tokens = tokenizer(test_input, return_tensors="pt").to(device)
    print(f"Tokenization finish, shape: {test_tokens.input_ids.shape}")

    preds = []
    pbar = tqdm(data)
    
    SAFE_CTX_LEN = min(max_length, 8192)
    SAFE_GEN_LEN = min(max_gen, 512)
    
    for idx, json_obj in enumerate(pbar):
        try:
            prompt = prompt_format.format(**json_obj)
            tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
            
            if len(tokenized_prompt) > SAFE_CTX_LEN:
                keep_ratio = 0.3
                keep_start = int(SAFE_CTX_LEN * keep_ratio)
                keep_end = SAFE_CTX_LEN - keep_start
                prompt = (
                    tokenizer.decode(tokenized_prompt[:keep_start], skip_special_tokens=True) + 
                    tokenizer.decode(tokenized_prompt[-keep_end:], skip_special_tokens=True)
                )
            
            if dataset not in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
                prompt = build_chat(tokenizer, prompt, model_name)

            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            current_len = input.input_ids.shape[-1]
            pbar.set_description(f"Generating {idx}, len={min(current_len, SAFE_CTX_LEN)}")
            
            simulation_start_idx = max(0, current_len - min(decoding_simulation_length, 512))
            
            memory_before = torch.cuda.memory_allocated() / 1024**3
            print(f"memory: {memory_before:.2f}GB")
            
            with torch.no_grad(), time_limit(300):
                assert input.input_ids.device == torch.device(device), "error"
                
                # forward pass
                output = model(
                    input_ids=input.input_ids[:, :simulation_start_idx],
                    past_key_values=None,
                    use_cache=True,
                )
                past_key_values = output.past_key_values
                
                if decoding_simulation_length > 0:
                    sim_tokens = input.input_ids[0, simulation_start_idx:simulation_start_idx+512]
                    for i, input_id in enumerate(sim_tokens):
                        if i >= 512:
                            break
                        input_id = input_id.to(device)
                        
                        
                        past_key_values = tuple(
                            (k.to(device), v.to(device)) 
                            for k, v in past_key_values
                        ) if past_key_values else None
                        
                        output = model(
                            input_ids=input_id.unsqueeze(0).unsqueeze(0),
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        new_key_values = output.past_key_values
                        
                        # 
                        if kv_compressor is not None and i % 10 == 0:  
                            past_key_values = kv_compressor.update_cache(past_key_values, 
                                                                       [(new_key_values[j][0][:,:,-1:,:], new_key_values[j][1][:,:,-1:,:]) 
                                                                        for j in range(len(new_key_values))])
                        else:
                            past_key_values = new_key_values
                
                # gen
                pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1).to("cuda:0")
                generated_content = [pred_token_idx.item()]
                
                for step in range(SAFE_GEN_LEN - 1):
                    past_key_values = tuple(
                        (k.to(device), v.to(device)) 
                        for k, v in past_key_values
                    ) if past_key_values else None
                    
                    outputs = model(
                        input_ids=pred_token_idx,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    new_key_values = outputs.past_key_values
                    past_key_values = new_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1).to(device)
                    generated_content.append(pred_token_idx.item())
                    
                    # 
                    if kv_compressor is not None and step % 40 == 0:
                        past_key_values = kv_compressor.update_cache(past_key_values, 
                                                                   [(new_key_values[j][0][:,:,-1:,:], new_key_values[j][1][:,:,-1:,:]) 
                                                                    for j in range(len(new_key_values))])
                    MIN_GEN = 32
                    if step >= MIN_GEN and pred_token_idx.item() in eos_token_ids:
                        break

            memory_after = torch.cuda.memory_allocated() / 1024**3
            print(f"生成后显存: {memory_after:.2f}GB (增加: {memory_after-memory_before:.2f}GB)")
            
            pred = tokenizer.decode(generated_content, skip_special_tokens=True)
            pred = post_process(pred, model_name)
            print(f"Sample {idx} Prediction: {pred}")
            print(f"Prediction: {pred[:100]}...")
            
            preds.append({
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            })
            
        except (TimeoutException, RuntimeError) as e:
            print(f"处理样本{idx}时出错: {str(e)}")
            torch.cuda.empty_cache()
            continue
            
        if idx % 2 == 0:
            torch.cuda.empty_cache()
    
    return preds

import math

def compress_attention_weights(
        model,
        base_rank: int = 1024,
        keep_first_n: int = 8,
        min_rank: int = 32,
):
    """
    """
    print("→ Layer-wise SVD compression (Q/O only)")
    total_layers = len(model.model.layers)
    tot_before = sum(p.numel() for p in model.parameters())
    tot_reduced = 0

    for idx, layer in enumerate(model.model.layers):
        if idx < keep_first_n:
            # skip
            continue

        depth_ratio = (idx + 1) / total_layers          # 0-1
        rank = int(base_rank * math.sqrt(depth_ratio))
        rank = max(rank, min_rank)

        attn = layer.self_attn
        for proj_name in ("q_proj", "o_proj"):           # 只改 Q / O
            lin = getattr(attn, proj_name)

            if isinstance(lin, SVDLinear):
                continue

            old_n = lin.weight.numel() + (lin.bias.numel() if lin.bias is not None else 0)

            setattr(attn, proj_name, SVDLinear(lin, rank))

            new_lin = getattr(attn, proj_name)
            new_n = sum(p.numel() for p in new_lin.parameters())
            tot_reduced += old_n - new_n

            print(f"  layer {idx:02d} {proj_name}: "
                  f"{old_n:,} → {new_n:,} (rank {rank})")

    tot_after = sum(p.numel() for p in model.parameters())
    print(f"\n=== SVD summary ===")
    print(f"params: {tot_before:,} → {tot_after:,}  "
          f"(-{tot_reduced:,}, {100*tot_reduced/tot_before:.1f}% ↓)")

    return model

def quick_kv_benchmark(model, kv_compressor, seq_len=4096, gen_steps=64):

    import time, torch
    dummy_input = torch.ones((1, seq_len), dtype=torch.long, device=device)

    model.eval()
    torch.cuda.reset_peak_memory_stats(device)

    # Prefill
    with torch.no_grad():
        out = model(input_ids=dummy_input, past_key_values=None, use_cache=True)
        past_kv = out.past_key_values

    # Generate
    start = time.perf_counter()
    next_id = dummy_input[:, -1:]
    for _ in range(gen_steps):
        with torch.no_grad():
            out = model(input_ids=next_id, past_key_values=past_kv, use_cache=True)
        next_id = out.logits[:, -1:].argmax(dim=-1)
        new_kv  = out.past_key_values
        if kv_compressor is not None:
            past_kv = kv_compressor.update_cache(
                past_kv,
                [(new_kv[i][0][:, :, -1:, :], new_kv[i][1][:, :, -1:, :])
                 for i in range(len(new_kv))]
            )
        else:
            past_kv = new_kv
    latency_ms = (time.perf_counter() - start) * 1000 / gen_steps

    peak_mem = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    kv_mem   = kv_compressor.memory_usage if kv_compressor else 0

    print("\n====== Quick KV Benchmark ======")
    print(f"Context length        : {seq_len}")
    print(f"Generate steps        : {gen_steps}")
    print(f"Peak CUDA memory      : {peak_mem:.2f} MB")
    print(f"KV-cache memory       : {kv_mem:.2f} MB")
    print(f"Avg per-token latency : {latency_ms:.2f} ms")
    print("================================\n")
    

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    device = select_device(model_name)
    print(f"【resource】{model_name} → {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="flash_attention_2" if not args.compress_svd else "eager",
        low_cpu_mem_usage=True,
    )

    generation_config = GenerationConfig.from_pretrained(path)
    

    generation_config.do_sample   = True
    generation_config.repetition_penalty = 1.2      
    generation_config.no_repeat_ngram_size = 4      
    generation_config.temperature = 0.7             
    generation_config.top_p = 0.9                   # nucleus sampling

    
    model.generation_config = generation_config
    
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()

    if args.method == "duo_attn":
        assert args.attn_load_dir is not None, "attn_load_dir must be provided"
        print(f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}")
        full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)

        if args.sink_size is not None:
            sink_size = args.sink_size
        if args.recent_size is not None:
            recent_size = args.recent_size

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, None, sparsity=args.sparsity
        )
        print(f"True sparsity: {sparsity}")

        enable_duo_attention_eval(model, full_attention_heads, sink_size, recent_size)
    else:
        enable_tuple_kv_cache(model)
    
    model.config.pretraining_tp = 1
    
    for name, param in model.named_parameters():
        if param.device != torch.device(device):
            raise RuntimeError(f"parameter {name} device is different: {param.device}")
        
    return model, tokenizer, eos_token_ids

if __name__ == "__main__":
    global device
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LongBench/config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    device = select_device(args.model)
    
    # load
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name
    )

    # param
    original_params = sum(p.numel() for p in model.parameters())
    print(f"Original total parameters: {original_params:,}")

    if args.compress_svd:
        model = compress_attention_weights(model,
                                            base_rank=args.svd_rank,
                                            keep_first_n=8)

    kv_compressor = None
    if args.compress_kv_cache:
        kv_compressor = CompressedKVCache(
            max_cache_length=args.kv_max_length,
            quantize_bits=args.kv_quantize_bits,
            sliding_window=args.kv_sliding_window,
            sink_tokens=args.kv_sink_tokens,
            compression_ratio=args.kv_compression_ratio,
            verbose=False
        )
        print(f"\n🗜️ KV Cache Compression Enabled:")
        print(f"Max length: {args.kv_max_length}")
        print(f"Sliding window: {args.kv_sliding_window}")
        print(f"Sink tokens: {args.kv_sink_tokens}")
        print(f"Compression ratio: {args.kv_compression_ratio}")
        print(f"Quantization: {args.kv_quantize_bits} bits")

    final_params = sum(p.numel() for p in model.parameters())
    overall_ratio = original_params / final_params if final_params < original_params else final_params / original_params
    reduction_pct = (original_params - final_params) / original_params * 100
    
    print(f"\n🎯 Final Model Summary:")
    print(f"Parameters: {original_params:,} -> {final_params:,}")
    print(f"Reduction: {reduction_pct:.1f}%")
    
    quick_kv_benchmark(model, kv_compressor, seq_len=4096, gen_steps=64)
    
    max_length = model2maxlen[model_name]
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", 
                   "multi_news", "trec", "triviaqa", "samsum", "passage_count", 
                   "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = [args.task]
    
    dataset2prompt = json.load(open("eval/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/LongBench/config/dataset2maxlen.json", "r"))
    
    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")
    if not os.path.exists("eval/LongBench/pred_e"):
        os.makedirs("eval/LongBench/pred_e")
    
    for dataset in datasets:
        data = load_dataset(
            "/home/xuezeyu/llm/duo-attention/datasets/THUDM/LongBench",
            name=dataset,
            split="test",
            num_proc=1
        )
        
        print(data)
        if not os.path.exists(f"eval/LongBench/pred/{model_name}"):
            os.makedirs(f"eval/LongBench/pred/{model_name}")
        
        if args.method == "duo_attn":
            suffix_parts = []
            if args.compress_svd:
                suffix_parts.append(f"svd{args.svd_rank}")
            if args.compress_kv_cache:
                suffix_parts.append(f"kvcache{args.kv_max_length}")
            
            if suffix_parts:
                suffix = "-".join(suffix_parts)
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}-{suffix}.jsonl"
            else:
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}.jsonl"
        else:
            if args.compress_svd:
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-svd_compressed-rank_{args.svd_rank}.jsonl"
            else:
                out_path = f"eval/LongBench/pred/{model_name}/{dataset}-full.jsonl"
            
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred_with_kv_compression(
            model, tokenizer, eos_token_ids, data, max_length, max_gen,
            prompt_format, dataset, model_name, args.decoding_simulation_length,
            kv_compressor=kv_compressor
        )
        
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")