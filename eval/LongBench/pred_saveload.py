# -*- coding: utf-8 -*-
import os
import json
import random
import argparse
import signal
from contextlib import contextmanager

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig

from duo_attn.patch import enable_duo_attention_eval
from duo_attn.utils import to_device, load_attn_pattern, sparsify_attention_heads
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache

# === 通用设置 ===

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame): raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try: yield
    finally: signal.alarm(0)

def select_device(model_name):
    if "1048k" in model_name:
        for dev in [0]:
            if torch.cuda.memory_allocated(dev) < 0.8 * torch.cuda.get_device_properties(dev).total_memory:
                return f"cuda:{dev}"
    else:
        for dev in [0, 1, 3]:
            if torch.cuda.memory_allocated(dev) < 0.7 * torch.cuda.get_device_properties(dev).total_memory:
                return f"cuda:{dev}"
    return "cuda:0"

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--method", type=str, default="full")
    parser.add_argument("--attn_load_dir", type=str, default=None)
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument("--sparsity", type=float, default=0.5)
    parser.add_argument("--decoding_simulation_length", type=int, default=50)
    parser.add_argument("--compress_svd", action="store_true")  # OPT: 启用 SVD 压缩
    parser.add_argument("--svd_rank", type=int, default=1024)  # OPT: SVD 基础秩
    return parser.parse_args(args)

# === 模型加载与 SVD ===

def compress_attention_weights(model, svd_rank=1024):
    print("🔧 SVD 压缩中...")
    for name, module in model.named_modules():
        if all(hasattr(module, attr) for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj']):
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                weight = getattr(module, proj).weight.data
                try:
                    dim1, dim2 = weight.shape
                    svd_rank_max = int(0.85 * max(dim1, dim2))
                    base = svd_rank
                    current_rank = min(int(base * 3.3), svd_rank_max)

                    weight_f = weight.float()
                    U, S, Vh = torch.linalg.svd(weight_f, full_matrices=False)
                    total_energy = (S**2).sum()
                    retained_energy = torch.cumsum(S**2, dim=0)
                    rank = (retained_energy <= 0.997 * total_energy).sum().item() + 1
                    rank = min(rank, current_rank, S.size(0))
                    rank = max(rank, base)

                    Uw = U[:, :rank]
                    Sw = S[:rank]
                    Vw = Vh[:rank, :]
                    compressed = (Uw @ torch.diag(Sw)) @ Vw
                    getattr(module, proj).weight.data = compressed.to(weight.dtype)

                    print(f"✅ {name}.{proj}: {weight.shape} → {compressed.shape}, rank={rank}")
                except Exception as e:
                    print(f"❌ SVD failed on {name}.{proj}: {e}")

def save_compressed_weights(model, model_version):
    save_path = os.path.join("./svd_compressed", model_version)
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "compressed_model.pth"))
    print(f"💾 Saved compressed model to {save_path}/compressed_model.pth")

def load_svd_compressed_weights(model, model_version):
    path = os.path.join("./svd_compressed", model_version, "compressed_model.pth")
    if os.path.exists(path):
        print(f"📦 Loading compressed weights from {path}")
        model.load_state_dict(torch.load(path, map_location="cuda"), strict=False)
    else:
        print(f"⚠️ Compressed weights not found for {model_version}")

def build_chat(tokenizer, prompt, model_name):
    return f"[INST]{prompt}[/INST]" if "llama-2" in model_name else prompt

def post_process(resp, model_name):
    if "llama-3" in model_name.lower():
        return resp.split(".assistant")[0].split("</s>")[0].strip()
    if "Llama-2-7B-32K-Instruct" in model_name:
        return resp.split("(Document")[0].split("\n\nQuestion")[0].strip()
    return resp.strip()

def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
    device = select_device(model_name)
    print(f"📦 Loading model on {device}")

    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        attn_implementation="eager",  # OPT: eager 模式支持 SVD 替换
        low_cpu_mem_usage=True,
    )

    if args.compress_svd:
        compress_attention_weights(model, svd_rank=args.svd_rank)
        save_compressed_weights(model, model_name)
    model = model.eval()

    if args.method == "duo_attn":
        assert args.attn_load_dir is not None
        full_heads, sink, recent = load_attn_pattern(args.attn_load_dir)
        sink = args.sink_size or sink
        recent = args.recent_size or recent
        full_heads, sparsity = sparsify_attention_heads(full_heads, None, args.sparsity)
        enable_duo_attention_eval(model, full_heads, sink, recent)
    else:
        enable_tuple_kv_cache(model)

    return model, tokenizer, GenerationConfig.from_pretrained(path).eos_token_id

# === 推理逻辑 ===

def get_pred(model, tokenizer, eos_ids, data, max_length, max_gen, prompt_fmt, dataset, model_name, sim_len):
    preds, SAFE_CTX, SAFE_GEN = [], min(max_length, 8192), min(max_gen, 512)
    for idx, sample in enumerate(tqdm(data)):
        try:
            prompt = prompt_fmt.format(**sample)
            ids = tokenizer(prompt, return_tensors="pt").input_ids[0]
            if len(ids) > SAFE_CTX:
                keep = SAFE_CTX
                prompt = tokenizer.decode(ids[:keep//3], skip_special_tokens=True) + tokenizer.decode(ids[-(keep*2//3):], skip_special_tokens=True)
            if dataset not in ["triviaqa", "trec", "lcc"]: prompt = build_chat(tokenizer, prompt, model_name)
            input = tokenizer(prompt, return_tensors="pt").to(model.device)
            out = model(input_ids=input.input_ids[:, :-sim_len], use_cache=True)
            kv = out.past_key_values
            sim = input.input_ids[0, -sim_len:]
            for token in sim:
                out = model(input_ids=token.view(1,1), past_key_values=kv, use_cache=True)
                kv = out.past_key_values
            pred = [out.logits[:, -1, :].argmax(dim=-1).item()]
            for _ in range(SAFE_GEN - 1):
                out = model(input_ids=torch.tensor([[pred[-1]]], device=model.device), past_key_values=kv, use_cache=True)
                kv = out.past_key_values
                next_id = out.logits[:, -1, :].argmax(dim=-1).item()
                pred.append(next_id)
                if next_id in eos_ids: break
            pred_text = post_process(tokenizer.decode(pred, skip_special_tokens=True), model_name)
            preds.append({"pred": pred_text, "answers": sample["answers"], "all_classes": sample["all_classes"], "length": sample["length"]})
        except Exception as e:
            print(f"⚠️ Error on sample {idx}: {e}")
            torch.cuda.empty_cache()
            continue
        if idx % 2 == 0: torch.cuda.empty_cache()
    return preds

# === 主入口 ===

if __name__ == "__main__":
    global device
    seed_everything(42)
    args = parse_args()
    device = select_device(args.model)
    model_paths = json.load(open("eval/LongBench/config/model2path.json"))
    model_maxlen = json.load(open("eval/LongBench/config/model2maxlen.json"))
    model, tokenizer, eos_ids = load_model_and_tokenizer(model_paths[args.model], args.model)
    max_len = model_maxlen[args.model]
    dataset = load_dataset("/home/xuezeyu/llm/duo-attention/datasets/THUDM/LongBench", name=args.task, split="test")
    os.makedirs(f"eval/LongBench/pred/{args.model}", exist_ok=True)
    prompt_map = json.load(open("eval/LongBench/config/dataset2prompt.json"))
    genlen_map = json.load(open("eval/LongBench/config/dataset2maxlen.json"))
    out_path = f"eval/LongBench/pred/{args.model}/{args.task}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}.jsonl" if args.method == "duo_attn" else f"eval/LongBench/pred/{args.model}/{args.task}-full.jsonl"
    results = get_pred(model, tokenizer, eos_ids if isinstance(eos_ids, list) else [eos_ids], dataset, max_len, genlen_map[args.task], prompt_map[args.task], args.task, args.model, args.decoding_simulation_length)
    with open(out_path, "w", encoding="utf-8") as f:
        for item in results: json.dump(item, f, ensure_ascii=False); f.write("\n")
