import torch
import os
import argparse
import duo_attn.patch.static_kv_cache
print("[DEBUG] Loaded DuoAttentionStaticKVCache from:", duo_attn.patch.static_kv_cache.__file__)

from duo_attn.utils import (
    get_model,
    get_tokenizer,
    to_device,
    load_attn_pattern,
    seed_everything,
    sparsify_attention_heads,
)
from duo_attn.patch.llama import (
    enable_llama_duo_attention_static_kv_cache_eval,
    DuoAttentionStaticKVCache,
)
from utils import bench_func


def parse_args():
    parser = argparse.ArgumentParser(description="kv_reduction")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--attn_load_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=4096)
    parser.add_argument("--prefilling_chunk_size", type=int, default=4096)
    parser.add_argument("--sparsity", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--use_svd", action="store_true", help="Enable SVD-compressed weight loading")
    parser.add_argument("--head64_layers", type=str, default="28,29,30,31",
    help="哪些层把 k/v head_dim 改到 64；空串则不改")
    return parser.parse_args()


def load_svd_compressed_weights(model, model_version):
    compressed_path = os.path.join("./svd_head64_sel", model_version, "compressed_model.pth")
    if os.path.exists(compressed_path):
        try:
            print(f"[✓] Loading compressed weights from {compressed_path}")
            model.load_state_dict(torch.load(compressed_path, map_location="cuda"), strict=False)
            print("[✓] Successfully loaded compressed weights.")
        except Exception as e:
            print(f"[×] Failed to load compressed weights: {e}")
    else:
        print(f"[×] No compressed weights found for {model_version}")
        
def patch_head64(model, layers, new_head_dim=64, change_head_dim=False):
    if not layers:
        return

    # 1) 找到 decoder layers 列表
    dec_layers = None
    for attr in ["model.layers", "layers", "model.decoder.layers"]:
        obj = model
        ok = True
        for seg in attr.split("."):
            if hasattr(obj, seg):
                obj = getattr(obj, seg)
            else:
                ok = False
                break
        if ok:
            dec_layers = obj
            break
    if dec_layers is None:
        raise RuntimeError("❌ cannot locate decoder layers path")

    # 2) 计算目标行
    num_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)
    target_out = num_kv * new_head_dim

    for idx in layers:
        sa = dec_layers[idx].self_attn
        for proj in ["k_proj", "v_proj"]:
            old = getattr(sa, proj)
            lin = torch.nn.Linear(
                old.in_features,
                target_out,
                bias=False,
                dtype=old.weight.dtype,
                device=old.weight.device,
            )
            setattr(sa, proj, lin)
            
        if change_head_dim:
            sa.head_dim = new_head_dim
            sa.scale    = 1 / (new_head_dim ** 0.5)


def infer_compressed_head_dim(model):
    for name, module in model.named_modules():
        if hasattr(module, 'k_proj'):
            weight = getattr(module, 'k_proj').weight.data
            return weight.shape[0] // model.config.num_key_value_heads
    return None


if __name__ == "__main__":
    args = parse_args()

    if args.seed is not None:
        seed_everything(args.seed)

    tokenizer = get_tokenizer(args.model_name)

    with torch.no_grad():
        model = get_model(args.model_name)
            # ★ ① 解析 head-64 层列表（空串 → 不改）
        print("num_kv =", model.config.num_key_value_heads)
        layers_to_compress = [int(x) for x in args.head64_layers.split(",") if x]
        if layers_to_compress:                       # ★ 先把模块行数改到 512
            patch_head64(model, layers_to_compress, new_head_dim=64, change_head_dim=True)
            print("after patch:", model.model.layers[28].self_attn.k_proj.weight.shape)
        if args.use_svd:                           # 与上行顺序不能反
            model_version = args.model_name.split("/")[-1]
            load_svd_compressed_weights(model, model_version)
            print("after load :", model.model.layers[28].self_attn.k_proj.weight.shape)
    model.eval()

    if isinstance(args.device, str) and args.device.isdigit():
        args.device = f"cuda:{args.device}"
    model = to_device(model, args.device)

    if args.attn_load_dir is not None:
        full_attention_heads, sink_size, recent_size = load_attn_pattern(args.attn_load_dir)
        full_attention_heads, sparsity = sparsify_attention_heads(full_attention_heads, None, args.sparsity)
        print(f"[✓] True Sparsity: {sparsity}")
        enable_llama_duo_attention_static_kv_cache_eval(model, full_attention_heads)
        patch_head64(model, layers_to_compress, new_head_dim=64, change_head_dim=True)
        kv_cache_head_dim = 64      # 记得同步 KV-cache 维度
    else:
        raise ValueError("You must provide --attn_load_dir to evaluate DuoAttention with static KV cache.")

    text = "a\n\n" * args.max_length
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")[:, : args.max_length - 1]
    print(f"[✓] Input shape: {input_ids.shape}")

    max_size = input_ids.size(1) + 5
    prefilling_chunk_size = args.prefilling_chunk_size
    print(f"[✓] Max size: {max_size}, Prefilling chunk size: {prefilling_chunk_size}")

    kv_cache = DuoAttentionStaticKVCache(
        model,
        full_attention_heads,
        1,
        max_size,
        sink_size,
        recent_size,
        head_dim=kv_cache_head_dim  # ✅ pass compressed head_dim
    )

    def func1():
        with torch.no_grad():
            for i in range(0, input_ids.size(1), prefilling_chunk_size):
                input_chunk = input_ids[:, i: i + prefilling_chunk_size]
                _ = model(
                    input_ids=input_chunk,
                    past_key_values=kv_cache,
                    use_cache=True,
                )
            kv_cache.clear()

    ctx_latency, ctx_memory = bench_func(func1, num_steps=10, num_warmup_steps=3)

    kv_cache.clear()
    with torch.no_grad():
        for i in range(0, input_ids.size(1), prefilling_chunk_size):
            input_chunk = input_ids[:, i: i + prefilling_chunk_size]
            outputs = model(
                input_ids=input_chunk,
                past_key_values=kv_cache,
                use_cache=True,
            )

    print(f"[✓] Peak memory usage during prefill: {torch.cuda.max_memory_allocated() / 1024 / 1024:.2f} MB")

    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)

    def func2():
        with torch.no_grad():
            _ = model(
                input_ids=pred_token_idx,
                past_key_values=kv_cache,
                use_cache=True,
            )
        kv_cache.evict_last(1)

    gen_latency, gen_memory = bench_func(func2, num_steps=100, num_warmup_steps=50)
    kv_cache_memory_usage = kv_cache.memory_usage / 1024 / 1024

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "benchmark_result.txt"), "w") as f:
        print(f"Average generation time: {gen_latency:.4f} ms", file=f)
        print(f"Peak generation memory usage: {gen_memory:.4f} MB", file=f)
        print(f"Average context time: {ctx_latency:.4f} ms", file=f)
        print(f"Peak context memory usage: {ctx_memory:.4f} MB", file=f)
        print(f"Model name: {args.model_name}", file=f)
        print(f"Context length: {args.max_length}", file=f)
        print(f"Sparsity: {sparsity}", file=f)
        print(f"Prefilling chunk size: {prefilling_chunk_size}", file=f)
        print(f"KV cache memory usage: {kv_cache_memory_usage:.4f} MB", file=f)
