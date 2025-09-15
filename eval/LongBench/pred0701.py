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

from duo_attn.patch import enable_duo_attention_eval

from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache

class TimeoutException(Exception): pass

def select_device(model_name):
    if "1048k" in model_name:  # H100
        for dev in [0]:  # H100
            if torch.cuda.memory_allocated(dev) < 0.8 * torch.cuda.get_device_properties(dev).total_memory:
                return f"cuda:{dev}"
    else:  # RTX4090
        for dev in [0,1,3]:  # 4090
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

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="task name", required=True)

    parser.add_argument(
        "--method",
        type=str,
        default="full",
    )

    # duo attention
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="attention pattern directory"
    )
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)

    parser.add_argument("--sparsity", type=float, default=0.5)

    parser.add_argument("--decoding_simulation_length", type=int, default=50)

    # 新增 SVD 压缩相关参数
    parser.add_argument(
        "--compress_svd",
        action="store_true",
        help="Enable SVD compression of attention weights"
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=None,
        help="Base SVD rank for compression (default: 1024)"
    )
    '''
    parser.add_argument(
        "--load_compressed",
        action="store_true",
        help="Load SVD compressed weights if available"
    )
    '''
    return parser.parse_args(args)


# This is the customized building prompt for chat models
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


def get_pred(
    model,
    tokenizer,
    eos_token_ids,
    data,
    max_length,
    max_gen,
    prompt_format,
    dataset,
    model_name,
    decoding_simulation_length,
):
    print("Processing first data...")
    test_input = "This is a test"
    # 1：cuda:0
    test_tokens = tokenizer(test_input, return_tensors="pt").to(device)
    print(f"Tokenization finish，shape: {test_tokens.input_ids.shape}")

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

            # 2：cuda:0
            input = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
            current_len = input.input_ids.shape[-1]
            pbar.set_description(f"Generating {idx}, len={min(current_len, SAFE_CTX_LEN)}")
            
            simulation_start_idx = max(0, current_len - min(decoding_simulation_length, 512))
            
            print(f"memory: {torch.cuda.memory_allocated()/1024**3:.2f}GB")
            torch.cuda.current_stream().synchronize()
            
            with torch.no_grad(), time_limit(300):
                # 3：
                assert input.input_ids.device == torch.device(device), "error"
                
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
                        # 4：cuda:0
                        input_id = input_id.to(device)
                        
                        # 5
                        past_key_values = tuple(
                            (k.to(device), v.to(device)) 
                            for k, v in past_key_values
                        ) if past_key_values else None
                        
                        output = model(
                            input_ids=input_id.unsqueeze(0).unsqueeze(0),
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                        past_key_values = output.past_key_values
                
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
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1).to(device)
                    generated_content.append(pred_token_idx.item())
                    
                    if pred_token_idx.item() in eos_token_ids:
                        print(f"stop at {step} ")
                        break

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
            print(f"processing {idx} error: {str(e)}")
            torch.cuda.empty_cache()
            continue
            
        if idx % 2 == 0:
            torch.cuda.empty_cache()
    
    return preds

def compress_attention_weights(model, svd_rank=None):
    """
    SVD
    differrnt svd_rank
    """
    print("Starting SVD compression of attention weights...")

    base_svd_rank = svd_rank if svd_rank is not None else 1024  # 默认值为1024

    for name, module in model.named_modules():
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj') and hasattr(module, 'o_proj'):
            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                weight = getattr(module, proj).weight.data
                try:
                    dim1, dim2 = weight.shape

                    # svd_rank_max
                    svd_rank_max = int(0.85 * max(dim1, dim2))

                    # svd_rank
                    if proj in ['q_proj', 'k_proj', 'v_proj']:
                        # higher svd_rank
                        current_svd_rank = min(int(base_svd_rank * 3.3), svd_rank_max)  # 例如2048/3072
                    elif proj == 'o_proj':
                        # o_proj
                        current_svd_rank = min(int(base_svd_rank * 3.3), svd_rank_max)  # 例如3072
                    else:
                        current_svd_rank = min(base_svd_rank, svd_rank_max)

                    # SVD
                    weight_float = weight.float()
                    U, S, Vh = torch.linalg.svd(weight_float, full_matrices=False)

                    # energy
                    energy = torch.sum(S ** 2)
                    cumulative_energy = torch.cumsum(S ** 2, dim=0)
                    energy_threshold = 0.997 * energy

                    # rank
                    rank_to_use = (cumulative_energy <= energy_threshold).sum().item() + 1  # +1以包含第一个超过阈值的奇异值

                    rank_to_use = min(rank_to_use, current_svd_rank, S.size(0))

                    # rank_to_use>=base svd_rank
                    if rank_to_use < base_svd_rank:
                        rank_to_use = base_svd_rank
                        rank_to_use = min(rank_to_use, current_svd_rank, S.size(0))

                    # SVD result
                    U_truncated = U[:, :rank_to_use]
                    S_truncated = S[:rank_to_use]
                    Vh_truncated = Vh[:rank_to_use, :]

                    # reconstruction of the compressed weight matrix
                    compressed_weight = torch.mm(U_truncated, torch.diag(S_truncated)).mm(Vh_truncated)

                    # conversion back to the original data type
                    compressed_weight = compressed_weight.to(weight.dtype)

                    # replacement of the original weights
                    getattr(module, proj).weight.data = compressed_weight

                    # calculation of the energy retention ratio
                    energy_retained = torch.sum(S_truncated ** 2).item() / energy.item()

                    # output of compression information
                    print(f"SVD compressed {proj} for {name}: original shape {weight.shape}, compressed shape {compressed_weight.shape}, rank={rank_to_use}, energy retained={energy_retained:.4f}")

                except Exception as e:
                    print(f"Failed to compress {proj} for {name}: {e}")

# save
def save_compressed_weights(model, model_version):
    save_dir = os.path.join("./svd_compressed", model_version)
    os.makedirs(save_dir, exist_ok=True)
    try:
        torch.save(model.state_dict(), os.path.join(save_dir, "compressed_model.pth"))
        print(f"Saved compressed model weights to {os.path.join(save_dir, 'compressed_model.pth')}")
    except Exception as e:
        print(f"Failed to save compressed model weights: {e}")

# load
def load_svd_compressed_weights(model, model_version):
    compressed_path = os.path.join("./svd_compressed", model_version, "compressed_model.pth")
    if os.path.exists(compressed_path):
        try:
            print(f"Loading compressed weights from {compressed_path}")
            model.load_state_dict(torch.load(compressed_path, map_location=model.device), strict=False)
            print("Successfully loaded compressed weights.")
        except Exception as e:
            print(f"Failed to load compressed weights: {e}")
    else:
        print("No compressed weights found. Proceeding without loading.")


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def load_model_and_tokenizer(path, model_name):
    tokenizer = AutoTokenizer.from_pretrained(
        path, 
        trust_remote_code=True,
        use_fast=False,
    )
    device = select_device(model_name)
    print(f"【资源分配】{model_name} → {device}")
    
    model = AutoModelForCausalLM.from_pretrained(  
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": device},  
        attn_implementation="flash_attention_2" if not args.compress_svd else "eager",  
        #max_memory={i: "80GiB" for i in range(torch.cuda.device_count())}, 
        low_cpu_mem_usage=True,
    )
    
    #if load_compressed:
    #    compressed_path = f"./svd_compressed/{model_name}/compressed_model.pth"
    #if os.path.exists(compressed_path):
    #    model.load_state_dict(torch.load(compressed_path, map_location="cuda"), strict=False)
    #    print("Loaded compressed weights.")
    #else:
    #    print("Compressed weights not found. Using original weights.")

    generation_config = GenerationConfig.from_pretrained(path)
    eos_token_ids = generation_config.eos_token_id
    if not isinstance(eos_token_ids, list):
        eos_token_ids = [eos_token_ids]

    model = model.eval()

    if args.method == "duo_attn":
        assert args.attn_load_dir is not None, "attn_load_dir must be provided"
        print(
            f"Loading attention pattern from {args.attn_load_dir} with sparsity {args.sparsity}"
        )
        full_attention_heads, sink_size, recent_size = load_attn_pattern(
            args.attn_load_dir
        )

        if args.sink_size is not None:
            sink_size = args.sink_size
        if args.recent_size is not None:
            recent_size = args.recent_size

        full_attention_heads, sparsity = sparsify_attention_heads(
            full_attention_heads, None, sparsity=args.sparsity
        )
        print(f"True sparsity: {sparsity}")

        enable_duo_attention_eval(
            model,
            full_attention_heads,
            sink_size,
            recent_size,
        )
    else:
        enable_tuple_kv_cache(model)
    model.config.pretraining_tp = 1  
    # 设备一致性检查
    for name, param in model.named_parameters():
        if param.device != torch.device(device):
            raise RuntimeError(f"param {name} device is different: {param.device}")
        
    return model, tokenizer, eos_token_ids


if __name__ == "__main__":
     # init
    global device
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LongBench/config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    device = select_device(args.model)
    # define your model args.load_compressed
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name #, load_compressed=False
    )
    #model = to_device(model, device_list, enable_tp=True)

    #if args.load_compressed:
    #    load_svd_compressed_weights(model, model_name)

    if args.compress_svd:
        compress_attention_weights(model, svd_rank=args.svd_rank)
        save_compressed_weights(model, model_name)

    max_length = model2maxlen[model_name]
    if args.e:
        datasets = [
            "qasper",
            "multifieldqa_en",
            "hotpotqa",
            "2wikimqa",
            "gov_report",
            "multi_news",
            "trec",
            "triviaqa",
            "samsum",
            "passage_count",
            "passage_retrieval_en",
            "lcc",
            "repobench-p",
        ]
    else:
        datasets = [args.task]
    # we design specific prompt format and max generation length for each task, feel free to modify them to optimize model output
    dataset2prompt = json.load(open("eval/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/LongBench/config/dataset2maxlen.json", "r"))
    # predict on each dataset
    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")
    if not os.path.exists("eval/LongBench/pred_e"):
        os.makedirs("eval/LongBench/pred_e")
    for dataset in datasets:
        #data = load_dataset("THUDM/LongBench", dataset, split="test")
        #data = load_dataset("/datasets/THUDM/LongBench/LongBench.py", name=dataset, split="test")
        #data = load_dataset("/datasets/THUDM/LongBench/longbench.py", name="hotpotqa", split="test")
        #data = load_dataset("/home/xuezeyu/llm/duo-attention/datasets/THUDM/LongBench", name=dataset, split="test")
        data = load_dataset(
            "/home/xuezeyu/llm/duo-attention/datasets/THUDM/LongBench",
            name=dataset,
            split="test",
            num_proc=1  # NEW
        )
        print(data)
        if not os.path.exists(f"eval/LongBench/pred/{model_name}"):
            os.makedirs(f"eval/LongBench/pred/{model_name}")
        #if args.method == "duo_attn":
        #    out_path = f"eval/LongBench/pred/{model_name}/{dataset}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}.jsonl"
        #else:
        #    out_path = f"eval/LongBench/pred/{model_name}/{dataset}-full.jsonl"
        if args.method == "duo_attn":
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-duo_attn-pattern-{args.attn_load_dir.split('/')[-1]}-sp-{args.sparsity}.jsonl"
        else:
            out_path = f"eval/LongBench/pred/{model_name}/{dataset}-full.jsonl"
        prompt_format = dataset2prompt[dataset]
        max_gen = dataset2maxlen[dataset]
        preds = get_pred(
            model,
            tokenizer,
            eos_token_ids,
            data,
            max_length,
            max_gen,
            prompt_format,
            dataset,
            model_name,
            args.decoding_simulation_length,
        )
        with open(out_path, "w", encoding="utf-8") as f:
            for pred in preds:
                json.dump(pred, f, ensure_ascii=False)
                f.write("\n")
