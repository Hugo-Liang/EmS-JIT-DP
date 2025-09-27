# import json
# import pickle
# import pandas as pd
#
#
#
# # 输入文件
# changes_file = "./fine-tuning/JITDefectPrediction/Unified/jdt/changes_test.jsonl"   # jsonl 或 pkl 格式
# features_file = "./fine-tuning/JITDefectPrediction/Unified/jdt/features_test.pkl" # pkl 格式
# output_file = "./fine-tuning/JITDefectPrediction/Unified/jdt/jitdp_alpaca_academic.jsonl"
#
# # 读取 changes 数据
# try:
#     changes = pd.read_json(changes_file, lines=True)
# except ValueError:
#     with open(changes_file, "rb") as f:
#         changes = pickle.load(f)
#     changes = pd.DataFrame(changes)
#
# # 读取 features 数据
# with open(features_file, "rb") as f:
#     features = pickle.load(f)
# features = pd.DataFrame(features)
#
# # 保留需要的列
# features = features.rename(columns={"commit_hash": "hash"})
# feature_cols = [
#     "la","ld","lt","ns","nd","nf","entropy","fix",
#     "ndev","age","nuc","exp","rexp","sexp"
# ]
# features = features[["hash"] + feature_cols]
#
# # 合并
# data = pd.merge(changes, features, on="hash", how="inner")
#
# # 转换成 Alpaca academic style 格式
# with open(output_file, "w", encoding="utf-8") as f:
#     for _, row in data.iterrows():
#         instruction = (
#             "As an intelligent assistant for Just-in-Time Defect Prediction. Based on the commit message, code changes, and change-level features, predict whether this commit is defective (1) or clean (0)."
#         )
#
#         # 输入文本
#         input_text = (
#             f"Message: {row['msg']}\n"
#             f"'<add>': {row['added_code']}\n"
#             f"'<del>': {row['removed_code']}\n"
#             f"Features:" + " " +
#             ", ".join([f"{col}={row[col]}" for col in feature_cols])
#         )
#
#         output_text = str(row["y"])  # 标签 (0 或 1)
#
#         alpaca_item = {
#             "instruction": instruction,
#             "input": input_text,
#             "output": output_text
#         }
#         f.write(json.dumps(alpaca_item, ensure_ascii=False) + "\n")
#
# print(f"✅ Conversion completed. Saved to {output_file}")
#



import os
import json
import pickle
import pandas as pd
from tqdm import tqdm

# -------- 配置部分 --------
# 数据目录结构: base_dir / project / subset / (changes_xxx.pkl, features_xxx.pkl)
base_dir = "./fine-tuning/JITDefectPrediction/Unified"  # 根目录，例如 datasets/Project1/train/...
# output_dir = "./fine-tuning/JITDefectPrediction/Unified"
# os.makedirs(output_dir, exist_ok=True)


# Instruction 模板
INSTRUCTION_TEMPLATES = {
    "academic": (

    ),
    "engineering": (
            "As an intelligent assistant for Just-in-Time Defect Prediction. Based on the commit message, code changes, and change-level features, predict whether this commit is defective (1) or clean (0).\n"
            "These change-level features are: \n"
            "- la: Lines of code added\n"
            "- ld: Lines of code deleted\n"
            "- lt: Lines of code in a file before the change\n"
            "- ns: Number of modified subsystems\n"
            "- nd: Number of modified directories\n"
            "- nf: Number of modified files\n"
            "- entropy: Distribution of modified code across each file\n"
            "- fix: Whether or not the change is a defect fix\n"
            "- ndev: The number of developers that changed the modified files\n"
            "- age: The average time interval between the last and the current change\n"
            "- nuc: The number of unique changes to the modified files\n"
            "- exp: Developer experience\n"
            "- rexp: Recent developer experience\n"
            "- sexp: Developer experience on a subsystem\n"
        )
}

# 专家特征列
FEATURE_COLS = [
    "la", "ld", "lt", "ns", "nd", "nf", "entropy", "fix",
    "ndev", "age", "nuc", "exp", "rexp", "sexp"
]


# -------- 核心函数 --------
def load_changes(changes_file):
    """加载 changes 文件，支持 pkl/jsonl"""
    try:
        return pd.read_json(changes_file, lines=True)
    except ValueError:
        with open(changes_file, "rb") as f:
            changes = pickle.load(f)
        return pd.DataFrame(changes)


def load_features(features_file):
    """加载 features 文件"""
    with open(features_file, "rb") as f:
        features = pickle.load(f)
    df = pd.DataFrame(features)
    df = df.rename(columns={"commit_hash": "hash"})
    return df[["hash"] + FEATURE_COLS]


def convert_to_alpaca(changes_file, features_file, style="engineering"):
    """合并 changes 和 features，返回 alpaca 格式列表"""
    changes = load_changes(changes_file)
    features = load_features(features_file)
    data = pd.merge(changes, features, on="hash", how="inner")

    alpaca_data = []
    for _, row in tqdm(data.iterrows()):
        instruction = INSTRUCTION_TEMPLATES[style]
        input_text = (
            f"Message: {row['msg']}\n"
            f"'<add>': {row['added_code']}\n"
            f"'<del>': {row['removed_code']}\n"
            f"Features:" + " " +
            ", ".join([f"{col}={row[col]}" for col in FEATURE_COLS])
        )
        output_text = str(row["y"])
        alpaca_data.append({
            "instruction": instruction,
            "input": input_text,
            "output": output_text
        })
    return alpaca_data


def process_all_projects(base_dir, style="academic"):
    """遍历所有项目和子集，生成对应的 JSON 文件"""
    for project in os.listdir(base_dir):
        project_path = os.path.join(base_dir, project)
        if not os.path.isdir(project_path):
            continue

        for subset in ["train", "valid", "test"]:
            # subset_path = os.path.join(project_path, subset)
            # if not os.path.isdir(subset_path):
            #     continue

            # 寻找文件
            # changes_file = None
            # features_file = None
            # for fname in os.listdir(project_path):
            #     if fname.startswith("changes") and fname.endswith(".pkl"):
            #         changes_file = os.path.join(project_path, subset, fname)
            #     elif fname.startswith("features") and fname.endswith(".pkl"):
            #         features_file = os.path.join(project_path, subset, fname)
            changes_file = os.path.join(project_path, f"changes_{subset}.jsonl")
            features_file = os.path.join(project_path, f"features_{subset}.pkl")

            if not changes_file or not features_file:
                print(f"⚠️ Missing files in {project_path}, skipped.")
                continue

            # 转换
            alpaca_data = convert_to_alpaca(changes_file, features_file, style=style)

            # 输出文件
            out_file = os.path.join(project_path, f"{project}_{subset}_{style}.json")
            with open(out_file, "w", encoding="utf-8") as f:
                json.dump(alpaca_data, f, ensure_ascii=False, indent=4)

            print(f"✅ Saved {len(alpaca_data)} samples to {out_file}")


# -------- 运行 --------
if __name__ == "__main__":
    # 选择 instruction 风格: "academic" 或 "engineering"
    # process_all_projects(base_dir, output_dir, style="academic")
    process_all_projects(base_dir, style="engineering")
