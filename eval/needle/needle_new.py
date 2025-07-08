"""
This script is adapted from 
https://github.com/gkamradt/LLMTest_NeedleInAHaystack
"""

import os
import glob
import json
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

import numpy as np
import argparse
from rouge_score import rouge_scorer

from datetime import datetime, timezone
import time
import torch
import torch.nn as nn
from duo_attn.patch import enable_duo_attention_eval
from duo_attn.utils import (
    to_device,
    load_attn_pattern,
    sparsify_attention_heads,
)

# 过滤 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.load")


class LLMNeedleHaystackTester:
    """
    This class is used to test the LLM Needle Haystack.
    """

    def __init__(
        self,
        args,
        needle="\n\nRemember, the best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n",
        haystack_dir="PaulGrahamEssays",
        retrieval_question="what is the best thing to do in San Francisco?\n\nAnswer: The best thing to do in San Francisco is",
        results_version=1,
        context_lengths_min=1000,
        context_lengths_max=1048000,
        context_lengths_num_intervals=40,
        context_lengths=None,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_intervals=10,
        document_depth_percents=None,
        document_depth_percent_interval_type="linear",
        model_provider="LLaMa",
        model_name="",
        model_name_suffix=None,
        num_concurrent_requests=1,
        save_results=True,
        save_contexts=True,
        final_context_length_buffer=200,
        seconds_to_sleep_between_completions=None,
        print_ongoing_status=True,
        attn_load_dir=None,
        sparsity=0.5,
        simulation_length=50,
        svd_rank=256,  # 新增 SVD 压缩等级参数
    ):
        """
        初始化方法
        """
        if not needle or not haystack_dir or not retrieval_question:
            raise ValueError(
                "Needle, haystack, and retrieval_question must be provided."
            )

        self.args = args
        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.results_version = results_version
        self.num_concurrent_requests = num_concurrent_requests
        self.save_results = save_results
        self.final_context_length_buffer = final_context_length_buffer
        self.save_contexts = save_contexts
        self.seconds_to_sleep_between_completions = seconds_to_sleep_between_completions
        self.print_ongoing_status = print_ongoing_status
        self.model_provider = model_provider
        self.testing_results = []

        self.svd_rank = svd_rank  # 保存 SVD 压缩级别

        if "/" in model_name:
            self.model_version = model_name.split("/")[-1]
        else:
            self.model_version = model_name
        if model_name_suffix is not None:
            self.model_version += "_" + model_name_suffix

        if context_lengths is None:
            if (
                context_lengths_min is None
                or context_lengths_max is None
                or context_lengths_num_intervals is None
            ):
                raise ValueError(
                    "Either context_lengths_min, context_lengths_max, context_lengths_num_intervals need to be filled out OR the context_lengths_list needs to be supplied."
                )
            else:
                # 后续会调整 context_lengths_max
                self.context_lengths = np.round(
                    np.linspace(
                        context_lengths_min,
                        context_lengths_max,
                        num=context_lengths_num_intervals,
                        endpoint=True,
                    )
                ).astype(int)
        else:
            self.context_lengths = context_lengths

        # 无论 context_lengths 是否被提供，都需要设置 context_lengths_min 和 context_lengths_max
        self.context_lengths_min = min(self.context_lengths)
        self.context_lengths_max = max(self.context_lengths)

        if document_depth_percents is None:
            if (
                document_depth_percent_min is None
                or document_depth_percent_max is None
                or document_depth_percent_intervals is None
            ):
                raise ValueError(
                    "Either document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals need to be filled out OR the document_depth_percents needs to be supplied."
                )
            else:
                if document_depth_percent_interval_type == "linear":
                    self.document_depth_percents = np.round(
                        np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            num=document_depth_percent_intervals,
                            endpoint=True,
                        )
                    ).astype(int)
                elif document_depth_percent_interval_type == "sigmoid":
                    self.document_depth_percents = [
                        self.logistic(x)
                        for x in np.linspace(
                            document_depth_percent_min,
                            document_depth_percent_max,
                            document_depth_percent_intervals,
                        )
                    ]
        else:
            self.document_depth_percents = document_depth_percents

        if document_depth_percent_interval_type not in [None, "linear", "sigmoid"]:
            raise ValueError(
                "document_depth_percent_interval_type must be either None, 'linear' or 'sigmoid'. If you'd like your own distribution give a list of ints in via document_depth_percent_intervals"
            )

        self.model_name = model_name

        self.enc = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        self.generation_config = GenerationConfig.from_pretrained(model_name)
        self.eos_token_ids = self.generation_config.eos_token_id
        if not isinstance(self.eos_token_ids, list):
            self.eos_token_ids = [self.eos_token_ids]

        if self.enc.pad_token_id is None:
            if self.enc.eos_token_id is not None:
                self.enc.pad_token_id = self.enc.eos_token_id
            else:
                self.enc.pad_token_id = 0
        print("Loading from %s" % model_name)

        self.model_to_test = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,  # 保持原有数据类型
            attn_implementation="eager",
        ).eval()

        # 定义 RougeScorer
        self.scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer=True)

        # 调用加载或压缩SVD矩阵的函数
        self.load_svd_compressed_matrices()

        print(f"attn_load_dir: {attn_load_dir}")
        # 打印矩阵形状确认是否正确加载

        if attn_load_dir is not None:
            print(
                f"Loading attention pattern from {attn_load_dir} with sparsity {sparsity}"
            )
            full_attention_heads, sink_size, recent_size = load_attn_pattern(
                attn_load_dir
            )
            if args.sink_size is not None:
                sink_size = args.sink_size
            if args.recent_size is not None:
                recent_size = args.recent_size
            full_attention_heads, sparsity = sparsify_attention_heads(
                full_attention_heads, None, sparsity
            )
            enable_duo_attention_eval(
                self.model_to_test,
                full_attention_heads,
                sink_size,
                recent_size,
            )

        # 列出所有可用的 GPU 设备
        device_list = [i for i in range(torch.cuda.device_count())]
        self.model_to_test = to_device(self.model_to_test, device_list, enable_tp=True)

        self.model_to_test_description = model_name

        self.evaluation_model = None
        self.debug = "debug"
        self.simulation_length = simulation_length
        model_name = model_name.split("/")[-1]

        # 获取模型的最大上下文长度
        self.max_context_length = self.enc.model_max_length
        if self.context_lengths_max > self.max_context_length:
            print(f"Setting context_lengths_max to model's max context length: {self.max_context_length}")
            self.context_lengths_max = self.max_context_length
            self.context_lengths = np.round(
                np.linspace(
                    context_lengths_min,
                    self.context_lengths_max,
                    num=context_lengths_num_intervals,
                    endpoint=True,
                )
            ).astype(int)

    def logistic(self, x, L=100, x0=50, k=0.1):
        if x == 0:
            return 0
        if x == 100:
            return 100
        return np.round(L / (1 + np.exp(-k * (x - x0))), 3)

    def is_attention_module(self, module):
        """
        检查模块是否是注意力模块，通过检查是否有 q_proj, k_proj, v_proj, o_proj 属性。
        """
        return all(hasattr(module, attr) for attr in ['q_proj', 'k_proj', 'v_proj', 'o_proj'])

    def svd_compress_attention_weights(self, attn_layer: nn.Module, rank: int, module_name: str):
        """
        对注意力层的权重进行SVD压缩。
        :param attn_layer: 多头注意力层。
        :param rank: 压缩的秩（SVD的秩）。
        :param module_name: 模块的名称，用于保存压缩后的权重。
        """
        # 获取 Q, K, V, O 权重矩阵并转换为 float32
        q_weight = attn_layer.q_proj.weight.data.to(torch.float32)
        k_weight = attn_layer.k_proj.weight.data.to(torch.float32)
        v_weight = attn_layer.v_proj.weight.data.to(torch.float32)
        o_weight = attn_layer.o_proj.weight.data.to(torch.float32)

        # 获取权重矩阵的形状
        q_shape = q_weight.shape
        k_shape = k_weight.shape
        v_shape = v_weight.shape
        o_shape = o_weight.shape

        # 确保 SVD 压缩的秩不超过权重矩阵的最小维度
        rank_q = min(rank, q_shape[0], q_shape[1])
        rank_k = min(rank, k_shape[0], k_shape[1])
        rank_v = min(rank, v_shape[0], v_shape[1])
        rank_o = min(rank, o_shape[0], o_shape[1])

        # 对每个矩阵进行SVD压缩
        try:
            # Perform SVD on float32 tensors
            U_q, S_q, Vh_q = torch.linalg.svd(q_weight, full_matrices=False)
            U_k, S_k, Vh_k = torch.linalg.svd(k_weight, full_matrices=False)
            U_v, S_v, Vh_v = torch.linalg.svd(v_weight, full_matrices=False)
            U_o, S_o, Vh_o = torch.linalg.svd(o_weight, full_matrices=False)
        except RuntimeError as e:
            print(f"SVD failed for module {module_name}: {e}")
            return  # 不保存压缩后的权重

        # 截取前 rank 个奇异值并重构压缩后的权重矩阵
        S_q = torch.diag(S_q[:rank_q])
        compressed_q = torch.mm(torch.mm(U_q[:, :rank_q], S_q), Vh_q[:rank_q, :])

        S_k = torch.diag(S_k[:rank_k])
        compressed_k = torch.mm(torch.mm(U_k[:, :rank_k], S_k), Vh_k[:rank_k, :])

        S_v = torch.diag(S_v[:rank_v])
        compressed_v = torch.mm(torch.mm(U_v[:, :rank_v], S_v), Vh_v[:rank_v, :])

        S_o = torch.diag(S_o[:rank_o])
        compressed_o = torch.mm(torch.mm(U_o[:, :rank_o], S_o), Vh_o[:rank_o, :])

        # 确保压缩后的权重形状与原始权重一致
        assert compressed_q.shape == q_shape, f"Compressed q_weight shape {compressed_q.shape} does not match original shape {q_shape}"
        assert compressed_k.shape == k_shape, f"Compressed k_weight shape {compressed_k.shape} does not match original shape {k_shape}"
        assert compressed_v.shape == v_shape, f"Compressed v_weight shape {compressed_v.shape} does not match original shape {v_shape}"
        assert compressed_o.shape == o_shape, f"Compressed o_weight shape {compressed_o.shape} does not match original shape {o_shape}"

        # 替换原始权重并转换回原始数据类型
        attn_layer.q_proj.weight.data.copy_(compressed_q.to(attn_layer.q_proj.weight.dtype))
        attn_layer.k_proj.weight.data.copy_(compressed_k.to(attn_layer.k_proj.weight.dtype))
        attn_layer.v_proj.weight.data.copy_(compressed_v.to(attn_layer.v_proj.weight.dtype))
        attn_layer.o_proj.weight.data.copy_(compressed_o.to(attn_layer.o_proj.weight.dtype))

        # 打印压缩后的矩阵形状（调试用）
        print(f"Compressed q_weight for {module_name}: {compressed_q.shape}")
        print(f"Compressed k_weight for {module_name}: {compressed_k.shape}")
        print(f"Compressed v_weight for {module_name}: {compressed_v.shape}")
        print(f"Compressed o_weight for {module_name}: {compressed_o.shape}")

        # 保存压缩后的权重到模型独立的目录
        save_dir = os.path.join("./svd_compressed", self.model_version)
        os.makedirs(save_dir, exist_ok=True)
        try:
            torch.save(compressed_q, os.path.join(save_dir, f"{module_name}_q_weight.pth"))
            torch.save(compressed_k, os.path.join(save_dir, f"{module_name}_k_weight.pth"))
            torch.save(compressed_v, os.path.join(save_dir, f"{module_name}_v_weight.pth"))
            torch.save(compressed_o, os.path.join(save_dir, f"{module_name}_o_weight.pth"))
            print(f"Saved compressed weights for {module_name} to {save_dir}")
        except Exception as e:
            print(f"Failed to save compressed weights for {module_name}: {e}")

    def load_svd_compressed_matrices(self):
        """
        加载压缩矩阵，或者如果没有压缩矩阵则进行实时SVD压缩。
        """
        save_dir = os.path.join("./svd_compressed", self.model_version)  # 每个模型独立的压缩目录

        # 尝试加载预先压缩的矩阵
        if os.path.exists(save_dir) and os.listdir(save_dir):
            print(f"Loading compressed matrices from {save_dir}")
            for name, module in self.model_to_test.named_modules():
                if self.is_attention_module(module):
                    print(f"Identified attention module: {name}")
                    try:
                        q_weight_path = os.path.join(save_dir, f"{name}_q_weight.pth")
                        k_weight_path = os.path.join(save_dir, f"{name}_k_weight.pth")
                        v_weight_path = os.path.join(save_dir, f"{name}_v_weight.pth")
                        o_weight_path = os.path.join(save_dir, f"{name}_o_weight.pth")

                        if all(os.path.exists(p) for p in [q_weight_path, k_weight_path, v_weight_path, o_weight_path]):
                            # 加载压缩后的权重
                            compressed_q = torch.load(q_weight_path, map_location=torch.device('cpu'))
                            compressed_k = torch.load(k_weight_path, map_location=torch.device('cpu'))
                            compressed_v = torch.load(v_weight_path, map_location=torch.device('cpu'))
                            compressed_o = torch.load(o_weight_path, map_location=torch.device('cpu'))

                            # 替换模型中的权重
                            module.q_proj.weight.data.copy_(compressed_q)
                            module.k_proj.weight.data.copy_(compressed_k)
                            module.v_proj.weight.data.copy_(compressed_v)
                            module.o_proj.weight.data.copy_(compressed_o)
                            print(f"Loaded compressed weights for {name}")
                        else:
                            print(f"Compressed weights for {name} not found, skipping.")
                    except Exception as e:
                        print(f"Failed to load compressed weights for {name}: {e}")
        else:
            print("No pre-compressed matrices found. Performing real-time SVD compression.")
            # 实时进行 SVD 压缩并保存
            for name, module in self.model_to_test.named_modules():
                if self.is_attention_module(module):
                    print(f"Compressing attention weights for {name}")
                    self.svd_compress_attention_weights(module, self.svd_rank, name)

    def generate_context(self, context_length, depth_percent):
        """
        生成上下文
        """
        # 获取 Paul Graham 的文本文件内容
        context = self.read_context_files()

        # 截断上下文到指定长度
        context = self.encode_and_trim(context, context_length)

        # 插入 needle
        context = self.insert_needle(context, depth_percent, context_length)

        return context

    def encode_text_to_tokens(self, text):
        return self.enc.encode(text, add_special_tokens=False)

    def insert_needle(self, context, depth_percent, context_length):
        """
        在上下文中插入 needle
        """
        tokens_needle = self.encode_text_to_tokens(self.needle)
        tokens_context = self.encode_text_to_tokens(context)

        # 减去缓冲区长度
        context_length -= self.final_context_length_buffer

        # 如果上下文 + needle 超过指定长度，则截断上下文
        if len(tokens_context) + len(tokens_needle) > context_length:
            tokens_context = tokens_context[: context_length - len(tokens_needle)]

        if depth_percent == 100:
            # 在末尾插入 needle
            tokens_new_context = tokens_context + tokens_needle
        else:
            insertion_point = int(len(tokens_context) * (depth_percent / 100))
            tokens_new_context = tokens_context[:insertion_point]
            print(f"Insertion at {insertion_point} / {len(tokens_context)}")
            tokens_new_context += tokens_needle + tokens_context[insertion_point:]

        # 转换回字符串
        new_context = self.decode_tokens(tokens_new_context)
        return new_context

    def get_context_length_in_tokens(self, context):
        return len(self.enc.encode(context))

    def read_context_files(self):
        """
        读取 haystack 目录下的所有文本文件并拼接成一个字符串
        """
        context = ""
        max_context_length = max(self.context_lengths)

        while self.get_context_length_in_tokens(context) < max_context_length:
            for file in glob.glob(f"{self.haystack_dir}/*.txt"):
                with open(file, "r", encoding="utf-8") as f:
                    context += f.read()
        return context

    def get_tokens_from_context(self, context):
        return self.enc.encode(context)

    def decode_tokens(self, tokens, context_length=None):
        return self.enc.decode(tokens[:context_length], skip_special_tokens=True)

    def encode_and_trim(self, context, context_length):
        tokens = self.get_tokens_from_context(context)
        if len(tokens) > context_length:
            context = self.decode_tokens(tokens, context_length)
        return context

    def get_results(self):
        return self.testing_results

    def print_start_test_summary(self):
        """
        打印测试开始的摘要信息
        """
        print("\n")
        print("Starting Needle In A Haystack Testing...")
        print(f"- Model: {self.model_name}")
        print(
            f"- Context Lengths: {len(self.context_lengths)}, Min: {min(self.context_lengths)}, Max: {max(self.context_lengths)}"
        )
        print(
            f"- Document Depths: {len(self.document_depth_percents)}, Min: {min(self.document_depth_percents)}%, Max: {max(self.document_depth_percents)}%"
        )
        print(f"- Needle: {self.needle.strip()}")
        print("\n\n")

    def start_test(self, args):
        """
        启动测试
        """
        if self.print_ongoing_status:
            self.print_start_test_summary()
        self.run_test(args)

    def run_test(self, args):
        """
        运行所有测试任务
        """
        # 遍历所有上下文长度和深度百分比
        for context_length in self.context_lengths:
            if self.args.s_len is not None and context_length < self.args.s_len:
                continue
            if self.args.e_len is not None and context_length > self.args.e_len:
                continue
            for depth_percent in self.document_depth_percents:
                self.bound_evaluate_and_log(context_length, depth_percent)

    def bound_evaluate_and_log(self, *args):
        """
        绑定 evaluate_and_log 方法
        """
        self.evaluate_and_log(*args)

    def evaluate_and_log(self, context_length, depth_percent):
        """
        评估并记录结果
        # 检查是否已经评估过
        if self.save_results:
            if self.result_exists(context_length, depth_percent):
                print("Result exists, skipping")
                return
            else:
                print("Result does not exist, testing")
        """
        if self.save_results:
            print("Result does not exist, testing")
            
        # 生成上下文和提示
        context = self.generate_context(context_length, depth_percent)
        prompt = self.generate_prompt(context)

        test_start_time = time.time()

        # 编码提示
        encoded_prompt = self.enc(prompt, return_tensors="pt")
        prompt_input_ids = encoded_prompt["input_ids"].to(self.model_to_test.device)

        # 确保 prompt_input_ids 不超过模型的最大上下文长度
        if prompt_input_ids.size(1) > self.max_context_length:
            print(f"Truncating prompt_input_ids from {prompt_input_ids.size(1)} to {self.max_context_length}")
            prompt_input_ids = prompt_input_ids[:, -self.max_context_length:]

        simulation_start_idx = prompt_input_ids.size(1) - self.simulation_length
        if simulation_start_idx < 0:
            simulation_start_idx = 0
        question_input_ids = prompt_input_ids[:, simulation_start_idx:]
        prompt_input_ids = prompt_input_ids[:, :simulation_start_idx]

        pred_token_idx = None  # 初始化

        with torch.no_grad():
            if self.args.prefilling_chunk_size is not None:
                past_key_values = None
                for i in range(0, prompt_input_ids.size(1), self.args.prefilling_chunk_size):
                    chunk = prompt_input_ids[:, i : i + self.args.prefilling_chunk_size]
                    output = self.model_to_test(
                        input_ids=chunk,
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
            else:
                output = self.model_to_test(
                    input_ids=prompt_input_ids, past_key_values=None, use_cache=True
                )
                past_key_values = output.past_key_values

            # 继续生成 tokens
            generated_content = []
            for input_id in question_input_ids[0]:
                try:
                    input_id_tensor = input_id.unsqueeze(0).unsqueeze(0)
                    output = self.model_to_test(
                        input_ids=input_id_tensor.to(self.model_to_test.device),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = output.past_key_values
                    pred_token_idx = output.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content.append(pred_token_idx.item())
                    print(f"Generated token: {pred_token_idx.item()}")
                except Exception as e:
                    print(f"Error during token generation: {e}")
                    break  # 退出循环

            # 检查 pred_token_idx 是否被赋值
            if pred_token_idx is None:
                print("Error: pred_token_idx was not set during token generation.")
                return

            # 生成后续 tokens
            for _ in range(50):
                try:
                    outputs = self.model_to_test(
                        input_ids=pred_token_idx.to(self.model_to_test.device),
                        past_key_values=past_key_values,
                        use_cache=True,
                    )
                    past_key_values = outputs.past_key_values
                    pred_token_idx = outputs.logits[:, -1, :].argmax(dim=-1).unsqueeze(1)
                    generated_content.append(pred_token_idx.item())
                    if pred_token_idx.item() in self.eos_token_ids:
                        break
                except Exception as e:
                    print(f"Error during further token generation: {e}")
                    break

        # 解码生成的内容
        response = self.enc.decode(generated_content, skip_special_tokens=True).strip()

        test_end_time = time.time()
        test_elapsed_time = test_end_time - test_start_time

        # 计算 Rouge 分数
        score = self.scorer.score(self.needle, response)["rouge1"].fmeasure * 10

        # 记录结果
        results = {
            "model": self.model_to_test_description,
            "context_length": int(context_length),
            "depth_percent": float(depth_percent),
            "version": self.results_version,
            "needle": self.needle,
            "model_response": response,
            "score": score,
            "test_duration_seconds": test_elapsed_time,
            "test_timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S%z"),
        }

        self.testing_results.append(results)

        if self.print_ongoing_status:
            print(f"-- Test Summary -- ")
            print(f"Duration: {test_elapsed_time:.1f} seconds")
            print(f"Context: {context_length} tokens")
            print(f"Depth: {depth_percent}%")
            print(f"Score: {score}")
            print(f"Response: {response}\n")

        context_file_location = f'{self.model_version.replace(".", "_")}_len_{context_length}_depth_{int(depth_percent)}'

        if self.save_contexts:
            results["file_name"] = context_file_location

            # 保存上下文到文件以便重新测试
            if not os.path.exists("contexts"):
                os.makedirs("contexts")

            if not os.path.exists(f"contexts/{self.model_version}"):
                os.makedirs(f"contexts/{self.model_version}")

            with open(
                f"contexts/{self.model_version}/{context_file_location}_context.txt",
                "w",
                encoding="utf-8",
            ) as f:
                f.write(context)

        if self.save_results:
            # 保存结果到文件以便重新测试
            if not os.path.exists("results"):
                os.makedirs("results")

            if not os.path.exists(f"results/{self.model_version}"):
                os.makedirs(f"results/{self.model_version}")

            # 保存结果到 JSON 文件
            p = f"results/{self.model_version}/{context_file_location}_results.json"
            print("Writing at %s" % p)
            with open(p, "w", encoding="utf-8") as f:
                json.dump(results, f)

    def result_exists(self, context_length, depth_percent):
        """
        检查结果是否已经存在
        """
        results_dir = "results/" + self.model_version
        print("Searching existing results at %s" % results_dir)

        # 检查结果目录是否存在
        if not os.path.exists(results_dir):
            return False

        # 遍历结果目录中的文件
        for filename in os.listdir(results_dir):
            if filename.endswith(".json"):
                with open(os.path.join(results_dir, filename), "r") as f:
                    result = json.load(f)
                    context_length_met = result["context_length"] == context_length
                    depth_percent_met = result["depth_percent"] == depth_percent
                    version_met = result.get("version", 1) == self.results_version
                    model_met = result["model"] == self.model_name

                    # 检查是否匹配
                    if context_length_met and depth_percent_met and version_met and model_met:
                        return True  # 找到匹配的结果

        return False

    def generate_prompt(self, context):
        """
        生成提示语
        """
        test_format = f"<|im_start|> This is a very long story book: <book> {context} </book>.\n\nQuestion: Based on the content of the book, {self.retrieval_question}"
        return test_format


# 在类外部定义主执行逻辑
if __name__ == "__main__":
    # 解析命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--s_len", metavar="N", type=int, help="a number")
    parser.add_argument("-e", "--e_len", metavar="N", type=int, help="a number")
    parser.add_argument("--model_path", type=str, default=None, help="path to model")
    parser.add_argument("--model_name", type=str, default=None, help="name of model")
    parser.add_argument(
        "--model_name_suffix", type=str, default=None, help="name of model"
    )
    parser.add_argument(
        "--model_provider", type=str, default="LLaMA", help="which model to use"
    )
    parser.add_argument(
        "--attn_load_dir", type=str, default=None, help="attention pattern directory"
    )
    parser.add_argument("--sink_size", type=int, default=None)
    parser.add_argument("--recent_size", type=int, default=None)
    parser.add_argument("--simulation_length", type=int, default=50)
    parser.add_argument("--context_lengths_num_intervals", type=int, default=40)
    parser.add_argument("--document_depth_percent_intervals", type=int, default=10)
    parser.add_argument("--context_lengths_min", type=int, default=1000)
    parser.add_argument("--context_lengths_max", type=int, default=1048000)
    parser.add_argument("--document_depth_percent_min", type=int, default=0)
    parser.add_argument("--document_depth_percent_max", type=int, default=100)

    parser.add_argument("--prefilling_chunk_size", type=int, default=None)

    parser.add_argument("--sparsity", type=float, default=0.5)

    parser.add_argument(
        "--method",
        type=str,
        default=None,
    )
    parser.add_argument("--svd_rank", type=int, default=256, help="SVD compression rank")

    args = parser.parse_args()

    if args.model_path is not None:
        assert args.model_name is None
        model_name = args.model_path
    else:
        assert args.model_name is not None
        model_name = args.model_name

    ht = LLMNeedleHaystackTester(
        args=args,
        model_name=model_name,
        model_name_suffix=args.model_name_suffix,
        model_provider=args.model_provider,
        save_contexts=True,
        save_results=True,
        attn_load_dir=args.attn_load_dir,
        sparsity=args.sparsity,
        simulation_length=args.simulation_length,
        svd_rank=args.svd_rank,
        context_lengths_min=args.context_lengths_min,
        context_lengths_max=args.context_lengths_max,
        context_lengths_num_intervals=args.context_lengths_num_intervals,
        document_depth_percent_intervals=args.document_depth_percent_intervals,
        document_depth_percent_min=args.document_depth_percent_min,
        document_depth_percent_max=args.document_depth_percent_max,
    )

    ht.start_test(args)
