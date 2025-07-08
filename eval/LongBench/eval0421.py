import os
import json
import argparse
import numpy as np

from metrics import (
    qa_f1_score,
    rouge_zh_score,
    qa_f1_zh_score,
    rouge_score,
    classification_score,
    retrieval_score,
    retrieval_zh_score,
    count_score,
    code_sim_score,
)

dataset2metric = {
    "narrativeqa": qa_f1_score,
    "qasper": qa_f1_score,
    "multifieldqa_en": qa_f1_score,
    "multifieldqa_zh": qa_f1_zh_score,
    "hotpotqa": qa_f1_score,
    "2wikimqa": qa_f1_score,
    "musique": qa_f1_score,
    "dureader": rouge_zh_score,
    "gov_report": rouge_score,
    "qmsum": rouge_score,
    "multi_news": rouge_score,
    "vcsum": rouge_zh_score,
    "trec": classification_score,
    "triviaqa": qa_f1_score,
    "samsum": rouge_score,
    "lsht": classification_score,
    "passage_retrieval_en": retrieval_score,
    "passage_count": count_score,
    "passage_retrieval_zh": retrieval_zh_score,
    "lcc": code_sim_score,
    "repobench-p": code_sim_score,
}


def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--e", action="store_true", help="Evaluate on LongBench-E")
    parser.add_argument("--results_path", type=str, default=None)
    parser.add_argument("--verbose", 
                        action="store_true",
                        help="Enable debug logging")
    return parser.parse_args(args)


def scorer_e(dataset, predictions, answers, lengths, all_classes):
    scores = {"0-4k": [], "4-8k": [], "8k+": []}
    for prediction, ground_truths, length in zip(predictions, answers, lengths):
        score = 0.0
        if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
            prediction = prediction.lstrip("\n").split("\n")[0]
        for ground_truth in ground_truths:
            score = max(
                score,
                dataset2metric[dataset](
                    prediction, ground_truth, all_classes=all_classes
                ),
            )
        if length < 4000:
            scores["0-4k"].append(score)
        elif length < 8000:
            scores["4-8k"].append(score)
        else:
            scores["8k+"].append(score)
    for key in scores.keys():
        scores[key] = round(100 * np.mean(scores[key]), 2)
    return scores


def scorer(dataset, predictions, answers, all_classes):
    total_score = 0.0
    valid_count = 0  # 新增：有效样本计数器
    
    # 新增：空预测保护
    if not predictions:
        return 0.0
    
    for idx, (prediction, ground_truths) in enumerate(zip(predictions, answers)):
        try:
            # ==== 保留原有清理逻辑 ====
            prediction = prediction.split(".assistant")[0] \
                .split("\n\nQuestion")[0] \
                .split("</s>")[0] \
                .split("(Document")[0] \
                .split("\n\nQuestion")[0] \
                .split("\n\nAnswer")[0] \
                .split("(Passage")[0] \
                .strip()
                
            # ==== 保留任务特定清理 ====
            if dataset in ["trec", "triviaqa", "samsum", "lsht"]:
                prediction = prediction.lstrip("\n").split("\n")[0]
            if dataset in ["multifieldqa_zh", "dureader"]:
                prediction = prediction.split("问题：")[0].strip()
            if dataset == "lsht":
                prediction = prediction.split("新闻内容：")[0].strip()
            if dataset == "passage_retrieval_zh":
                prediction = prediction.split("请问")[0].split("提示")[0].strip()
            
            # ==== 新增输入验证 ====
            if not isinstance(ground_truths, list) or len(ground_truths) == 0:
                print(f"样本 {idx} 答案格式错误: {ground_truths}")
                continue
                
            score = 0.0
            for gt in ground_truths:
                current_score = dataset2metric[dataset](
                    str(prediction),  # 确保prediction为字符串
                    str(gt),          # 确保ground truth为字符串
                    all_classes=all_classes
                )
                score = max(score, current_score)
            
            total_score += score
            valid_count += 1  # 仅统计有效样本
            
        except Exception as e:
            print(f"处理样本 {idx} 失败: {str(e)}")
            print(f"问题预测: {prediction[:50]}...")
            continue
    
    # ==== 保留原有返回逻辑 ====
    if valid_count == 0:
        print(f"警告: 任务 {dataset} 无有效样本")
        return 0.0
        
    return round(100 * total_score / valid_count, 2)  # 分母改为有效样本数


if __name__ == "__main__":
    args = parse_args()
    scores = dict()
    
    # ===== 1. 路径验证与创建 =====
    if args.results_path:
        path = args.results_path
        os.makedirs(path, exist_ok=True)  # 确保路径存在
    else:
        base_dir = "pred_e" if args.e else "pred"  # 添加完整路径
        path = os.path.join(base_dir, args.model)
        os.makedirs(path, exist_ok=True)  # 自动创建目录
    
    print(f"结果存储路径: {os.path.abspath(path)}")
    
    # ===== 2. 增强文件遍历逻辑 =====
    try:
        all_files = [f for f in os.listdir(path) if f.endswith(".jsonl")]
        all_files.sort()
        print("待评估文件列表:", all_files)
        
        if not all_files:
            raise FileNotFoundError(f"目录 {path} 下无.jsonl文件")
            
    except FileNotFoundError as e:
        print(f"致命错误: {str(e)}")
        sys.exit(1)

    # ===== 3. 带错误捕获的文件处理 =====
    for filename in all_files:
        file_path = os.path.join(path, filename)
        print(f"\n{'='*30} 处理文件 {filename} {'='*30}")
        
        try:
            # ==== 3.1 读取数据 ====
            predictions, answers, lengths = [], [], []
            all_classes = None
            with open(file_path, "r", encoding="utf-8") as f:
                for line_idx, line in enumerate(f):
                    try:
                        data = json.loads(line)
                        # ==== 关键字段检查 ====
                        if "pred" not in data or "answers" not in data:
                            print(f"行 {line_idx} 缺少必要字段，已跳过")
                            continue
                            
                        predictions.append(data["pred"])
                        answers.append(data.get("answers", []))
                        if "length" in data:
                            lengths.append(data["length"])
                        if "all_classes" in data:
                            all_classes = data["all_classes"]  # 保持最后一个值
                            
                    except json.JSONDecodeError as e:
                        print(f"行 {line_idx} JSON解析失败: {str(e)}")
                        continue
                        
            # ==== 3.2 空数据检查 ====
            if not predictions:
                print(f"警告: 文件 {filename} 无有效预测数据")
                scores[filename] = 0.0
                continue
                
            # ==== 3.3 数据集名称修正 ====
            dataset = filename.split("-")[0].replace("_e", "")  # 处理LongBench-E后缀
            if dataset == "repobench":
                dataset = "repobench-p"
                
            # ==== 3.4 执行评分 ====
            score = (
                scorer_e(dataset, predictions, answers, lengths, all_classes)
                if args.e
                else scorer(dataset, predictions, answers, all_classes)
            )
            scores[filename] = score
            print(f"评分完成: {filename} => {score}")
            
        except Exception as e:
            print(f"处理文件 {filename} 时发生严重错误: {str(e)}")
            scores[filename] = "ERROR"
            continue

    # ===== 4. 确保结果写入 =====
    out_path = os.path.join(path, "result.json")
    
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(scores, f, ensure_ascii=False, indent=4)
        print(f"\n{'='*30} 结果已保存至 {out_path} {'='*30}")
        
    except IOError as e:
        print(f"无法写入结果文件: {str(e)}")
        sys.exit(1)
