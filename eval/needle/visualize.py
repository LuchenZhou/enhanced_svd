import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import pandas as pd
import json
import glob
import argparse
import shutil
import os
import datetime  # 添加 datetime 模块

parser = argparse.ArgumentParser()
parser.add_argument(
    "" "--folder_path",
    type=str,
    default="results/LLaMA-2-7B-32K/",
    help="Path to the directory containing JSON results",
)
parser.add_argument(
    "--model_name",
    type=str,
    default="LLaMA-2-7B-32K",
    help="Name of the model",
)
parser.add_argument(
    "--pretrained_len",
    type=int,
    default=32000,
    help="Length of the pretrained model",
)
args = parser.parse_args()


FOLDER_PATH = args.folder_path
MODEL_NAME = args.model_name
PRETRAINED_LEN = args.pretrained_len


def main():
    # 获取当前时间戳作为运行标识符
    run_id = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
                
    # 创建一个新的文件夹用于存储当前运行的结果
    current_run_folder = os.path.join(FOLDER_PATH, f"run_{run_id}/")
    os.makedirs(current_run_folder, exist_ok=True)

    # Path to the directory containing JSON results
    folder_path = FOLDER_PATH

    if "/" in folder_path:
        model_name = folder_path.split("/")[-2]
    else:
        model_name = MODEL_NAME
    print("model_name = %s" % model_name)

    # Using glob to find all json files in the directory
    json_files = glob.glob(f"{folder_path}*.json")
    print(f"Found {len(json_files)} JSON files in {folder_path}")

    if not json_files:
        print("No JSON files found. Exiting.")
        return
    # import ipdb; ipdb.set_trace()

    # List to hold the data
    data = []

    # Iterating through each file and extract the 3 columns we need
    for file in json_files:
        with open(file, "r") as f:
            json_data = json.load(f)
            # Extracting the required fields
            document_depth = json_data.get("depth_percent", None)
            context_length = json_data.get("context_length", None)
            # score = json_data.get("score", None)
            model_response = json_data.get("model_response", None).lower()
            needle = json_data.get("needle", None).lower()
            expected_answer = (
                "eat a sandwich and sit in Dolores Park on a sunny day.".lower().split()
            )
            
            # 确保model_response和needle存在
            if model_response and needle:
                # 检查 'context_length' 是否存在且不为 None
                if context_length is not None:
                    score = len(
                        set(model_response.split()).intersection(set(expected_answer))
                    ) / len(expected_answer)
                    # Appending to the list
                    data.append(
                        {
                            "Document Depth": document_depth,
                            "Context Length": context_length,
                            "Score": score,
                        }
                    )
                    print(f"Added score: {score}")
                else:
                    print(f"'context_length' is missing or None in {file}. Skipping.")
            else:
                print(f"Missing 'model_response' or 'needle' in {file}. Skipping.")               

    # Creating a DataFrame
    #df = pd.DataFrame(data)
    #locations = list(df["Context Length"].unique())
    #locations.sort()
    # 创建仅包含当前运行测试数据的 DataFrame
    df = pd.DataFrame(data)
    print(f"DataFrame columns: {df.columns.tolist()}")
    print(f"DataFrame size: {df.shape}")

    # 确保 'Context Length' 列存在
    if "Context Length" in df.columns:
        locations = list(df["Context Length"].unique())
        locations.sort()
        print(f"Unique Context Lengths: {locations}")
    else:
        locations = []
        print("Warning: 'Context Length' column is missing in DataFrame.")

        # 计算 pretrained_len
    if locations:
        pretrained_len = 0  # 默认值
        for li, l in enumerate(locations):
            if l > PRETRAINED_LEN:
                pretrained_len = li
                break
        else:
            pretrained_len = len(locations)  # 如果没有长度超过 PRETRAINED_LEN
        print(f"pretrained_len: {pretrained_len}")
    else:
        pretrained_len = 0
        print("No 'Context Length' data available. Setting 'pretrained_len' to 0.")
    #for li, l in enumerate(locations):
    #    if l > PRETRAINED_LEN:
    #        break
    #pretrained_len = li

        # 检查df是否为空，以避免除以零
    if not df.empty:
        overall_score = df["Score"].mean()
    else:
        overall_score = 0.0

    print(df.head())
    print("Overall score %.3f" % overall_score)

    pivot_table = pd.pivot_table(
        df, values="Score", index=["Document Depth", "Context Length"], aggfunc="mean"
    ).reset_index()  # This will aggregate
    pivot_table = pivot_table.pivot(
        index="Document Depth", columns="Context Length", values="Score"
    )  # This will turn into a proper pivot
    pivot_table.iloc[:5, :5]

    # Create a custom colormap. Go to https://coolors.co/ and pick cool colors
    cmap = LinearSegmentedColormap.from_list(
        "custom_cmap", ["#F0496E", "#EBB839", "#0CD79F"]
    )

    # Create the heatmap with better aesthetics
    f = plt.figure(figsize=(17.5, 8))  # Can adjust these dimensions as needed
    heatmap = sns.heatmap(
        pivot_table,
        vmin=0,
        vmax=1,
        cmap=cmap,
        cbar_kws={"label": "Score"},
        linewidths=0.5,  # Adjust the thickness of the grid lines here
        linecolor="grey",  # Set the color of the grid lines
        linestyle="--",
    )

    # More aesthetics
    model_name_ = MODEL_NAME
    plt.title(
        f'NIAH {model_name_} \n Overall Score: {overall_score:.3f}'
    )  # Adds a title
    plt.xlabel("Token Limit")  # X-axis label
    plt.ylabel("Depth Percent")  # Y-axis label
    plt.xticks(rotation=45)  # Rotates the x-axis labels to prevent overlap
    plt.yticks(rotation=0)  # Ensures the y-axis labels are horizontal
    plt.tight_layout()  # Fits everything neatly into the figure area

    # Add a vertical line at the desired column index
    plt.axvline(x=pretrained_len + 0.8, color="white", linestyle="--", linewidth=4)

    #save_path = "img/%s.png" % model_name
    save_path = os.path.join("img", f"{model_name}.png")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    print("saving at %s" % save_path)
    plt.savefig(save_path, dpi=150)


if __name__ == "__main__":
    main()
