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

from duo_attn.patch import enable_duo_attention_eval

from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)
from duo_attn.patch.tuple_kv_cache import enable_tuple_kv_cache


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        required=True,
        help="Model name"
    )
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")

    parser.add_argument("--task", type=str, help="Task name", required=True)

    parser.add_argument(
        "--method",
        type=str,
        default="full",
        help="Attention method to use (e.g., full, duo_attn)"
    )

    # Duo Attention
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="Attention pattern directory"
    )
    parser.add_argument("--sink_size", type=int, default=None, help="Sink size")
    parser.add_argument("--recent_size", type=int, default=None, help="Recent size")

    parser.add_argument("--sparsity", type=float, default=0.5, help="Sparsity level")

    parser.add_argument("--decoding_simulation_length", type=int, default=50, help="Decoding simulation length")

    # 新增 SVD 压缩相关参数
    parser.add_argument(
        "--compress_svd",
        action="store_true",
        help="Enable SVD compression of attention weights"
    )
    parser.add_argument(
        "--svd_rank",
        type=int,
        default=1024,
        help="Base SVD rank for compression (default: 1024)"
    )
    parser.add_argument(
        "--load_compressed",
        action="store_true",
        help="Load SVD compressed weights if available"
    )
    return parser.parse_args(args)


# Customized building prompt for chat models
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
    preds = []
    pbar = tqdm(data)
    for idx, json_obj in enumerate(pbar):
        prompt = prompt_format.format(**json_obj)
        # Truncate to fit max_length (suggest truncating in the middle)
        tokenized_prompt = tokenizer(
            prompt, truncation=False, return_tensors="pt"
        ).input_ids[0]
        if len(tokenized_prompt) > max_length:
            half = int(max_length / 2)
            prompt = tokenizer.decode(
                tokenized_prompt[:half], skip_special_tokens=True
            ) + tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
        if dataset not in [
            "trec",
            "triviaqa",
            "samsum",
            "lsht",
            "lcc",
            "repobench-p",
        ]:  # Chat models are better off without build prompts on these tasks
            prompt = build_chat(tokenizer, prompt, model_name)

        input = tokenizer(prompt, truncation=False, return_tensors="pt").to("cuda")
        pbar.set_description(
            f"Generating for {idx}, len = {input.input_ids.shape[-1]}"
        )
        simulation_start_idx = input.input_ids.shape[-1] - decoding_simulation_length
        with torch.no_grad():
            output = model(
                input_ids=input.input_ids[:, :simulation_start_idx],
                past_key_values=None,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            if decoding_simulation_length > 0:
                for idx, input_id in enumerate(
                    input.input_ids[0, simulation_start_idx:]
                ):
                    output = model(
                        input_ids=input_id.unsqueeze(0).unsqueeze(0),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
            pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
            generated_content = [pred_token_idx.item()]
            for _ in range(max_gen - 1):
                outputs = model(
                    input_ids=pred_token_idx,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

                past_key_values = outputs.past_key_values
                pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                generated_content += [pred_token_idx.item()]
                if pred_token_idx.item() in eos_token_ids:
                    break

        pred = tokenizer.decode(generated_content, skip_special_tokens=True)
        pred = post_process(pred, model_name)
        print(f"Prediction: {pred}")
        preds.append(
            {
                "pred": pred,
                "answers": json_obj["answers"],
                "all_classes": json_obj["all_classes"],
                "length": json_obj["length"],
            }
        )
    return preds
def compress_attention_weights(model, svd_rank=1024):
    """
    改进点：
    1. 层深感知的动态秩调整
    2. TP并行分片适配
    3. 秩下限保护
    4. 能量保留优化
    保持原有参数名称和接口不变
    """
    print("Starting optimized SVD compression...")

    def _get_layer_depth(name):
        """解析层号，支持常见模型结构"""
        parts = name.split('.')
        for p in parts:
            if p.isdigit():
                return int(p)
        return 0  # 默认处理无法解析层号的情况

    # 配置参数（可调整）
    TP_SIZE = 8                     # 根据实际并行度调整
    MIN_RANK_RATIO = 0.15          # 最小保留基础秩的比例
    DEPTH_DECAY = 0.97              # 每层衰减率

    base_svd_rank = svd_rank       # 保持参数名不变

    for name, module in model.named_modules():
        if hasattr(module, 'q_proj') and hasattr(module, 'k_proj') and hasattr(module, 'v_proj') and hasattr(module, 'o_proj'):
            # 动态调整因子计算
            depth = _get_layer_depth(name)
            depth_factor = DEPTH_DECAY ** depth  # 指数衰减
            
            # TP分片检测
            is_tp_module = "tp_wrapped_module" in name
            tp_factor = 1.0 / TP_SIZE if is_tp_module else 1.0

            for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                try:
                    weight = getattr(module, proj).weight.data
                    dim1, dim2 = weight.shape
                    
                    # 动态计算基础秩
                    dynamic_base = base_svd_rank * depth_factor * tp_factor
                    proj_factor = 3.5 if proj != 'o_proj' else 2.8  # 区分投影类型
                    current_base = int(dynamic_base * proj_factor)
                    
                    # 秩边界保护
                    max_dim = max(dim1, dim2)
                    svd_rank_max = int(0.88 * max_dim)
                    min_rank = max(int(base_svd_rank * MIN_RANK_RATIO), 64)  # 双重保护
                    current_svd_rank = min(max(current_base, min_rank), svd_rank_max)

                    # SVD分解核心逻辑（保持原始结构）
                    weight_float = weight.float()
                    U, S, Vh = torch.linalg.svd(weight_float, full_matrices=False)
                    
                    # 能量计算优化
                    cumulative_energy = S.pow(2).cumsum(dim=0)
                    energy_threshold = 0.995 * cumulative_energy[-1]  # 保留99.5%能量
                    
                    # 秩选择逻辑
                    rank = (cumulative_energy <= energy_threshold).sum().item()
                    rank = min(rank + 1, current_svd_rank)  # +1补偿阈值
                    rank = max(rank, min_rank)  # 确保不低于下限

                    # 重构矩阵
                    U_trunc = U[:, :rank]
                    S_trunc = S[:rank]
                    Vh_trunc = Vh[:rank, :]
                    compressed = (U_trunc @ torch.diag(S_trunc)) @ Vh_trunc

                    # 替换权重
                    getattr(module, proj).weight.data = compressed.to(weight.dtype)
                    
                    # 增强日志输出
                    energy_ratio = cumulative_energy[rank-1].item() / cumulative_energy[-1].item()
                    print(f"Compressed {proj} in {name}: "
                          f"base={base_svd_rank}→dyn={int(current_base)}→final={rank} "
                          f"energy={energy_ratio:.3%} "
                          f"shape={dim1}x{dim2}→{U_trunc.shape[1]}")

                except Exception as e:
                    print(f"Skip {proj} in {name}: {str(e)[:50]}")  # 简略错误信息
                    continue

def save_compressed_weights(model, model_version):
    """
    保存压缩后的模型权重。
    """
    save_dir = os.path.join("./svd_compressed", model_version)
    os.makedirs(save_dir, exist_ok=True)
    try:
        torch.save(model.state_dict(), os.path.join(save_dir, "compressed_model.pth"))
        print(f"Saved compressed model weights to {os.path.join(save_dir, 'compressed_model.pth')}")
    except Exception as e:
        print(f"Failed to save compressed model weights: {e}")


def load_svd_compressed_weights(model, model_version):
    """
    加载已压缩的注意力权重。
    """
    compressed_path = os.path.join("./svd_compressed", model_version, "compressed_model.pth")
    if os.path.exists(compressed_path):
        try:
            print(f"Loading compressed weights from {compressed_path}")
            model.load_state_dict(torch.load(compressed_path, map_location="cuda"), strict=False)
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


def load_model_and_tokenizer(path, model_name, load_compressed=False):
    """
    加载模型和分词器，并根据需要加载压缩后的权重。
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )

    if load_compressed:
        compressed_path = f"./svd_compressed/{model_name}/compressed_model.pth"
        if os.path.exists(compressed_path):
            print(f"释放现存内存...")
            torch.cuda.empty_cache()
            print(f"当前可用显存: {torch.cuda.mem_get_info()[0]/1024**3:.2f} GB")
            
            print(f"加载压缩权重到CPU...")
            state_dict = torch.load(compressed_path, map_location="cpu")
            
            print(f"逐步转移权重到GPU...")
            for key in list(state_dict.keys()):
                val = state_dict.pop(key)
                state_dict[key] = val.to("cuda")
                del val
                torch.cuda.empty_cache()
            
            print(f"加载到模型...")
            model.load_state_dict(state_dict, strict=False)
            del state_dict
            torch.cuda.empty_cache()

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

    return model, tokenizer, eos_token_ids


if __name__ == "__main__":
    seed_everything(42)
    args = parse_args()
    model2path = json.load(open("eval/LongBench/config/model2path.json", "r"))
    model2maxlen = json.load(open("eval/LongBench/config/model2maxlen.json", "r"))
    device_list = [i for i in range(torch.cuda.device_count())]
    model_name = args.model
    # Define your model
    model, tokenizer, eos_token_ids = load_model_and_tokenizer(
        model2path[model_name], model_name, load_compressed=args.load_compressed
    )
    model = to_device(model, device_list, enable_tp=True)

    # 如果选择进行 SVD 压缩
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
    # Design specific prompt format and max generation length for each task
    dataset2prompt = json.load(open("eval/LongBench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("eval/LongBench/config/dataset2maxlen.json", "r"))
    # Predict on each dataset
    if not os.path.exists("eval/LongBench/pred"):
        os.makedirs("eval/LongBench/pred")
    if not os.path.exists("eval/LongBench/pred_e"):
        os.makedirs("eval/LongBench/pred_e")
    for dataset in datasets:
        data = load_dataset("/home/xuezeyu/llm/duo-attention/datasets/THUDM/LongBench", name=dataset, split="test")
        print(data)
        if not os.path.exists(f"eval/LongBench/pred/{model_name}"):
            os.makedirs(f"eval/LongBench/pred/{model_name}")
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
