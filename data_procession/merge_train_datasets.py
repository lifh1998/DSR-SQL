import pandas as pd
import os

# 定义输入文件路径列表
input_paths = [
    './outputs/1_5b/bird/train/error_analysis/sql_refinement_training_data.csv',
    './outputs/1_5b/bird/dev/error_analysis/sql_refinement_training_data.csv',
    './outputs/1_5b/spider/train/error_analysis/sql_refinement_training_data.csv',
    './outputs/1_5b/spider/dev/error_analysis/sql_refinement_training_data.csv',
    './outputs/1_5b/spider-syn/train/error_analysis/sql_refinement_training_data.csv',
    './outputs/1_5b/spider-syn/dev/error_analysis/sql_refinement_training_data.csv',
]

# 定义每个文件要抽取的数量，如果文件路径不在字典中，则默认抽取所有
sample_counts = {
    './outputs/1_5b/bird/dev/error_analysis/sql_refinement_training_data.csv': 20,
    './outputs/1_5b/spider/dev/error_analysis/sql_refinement_training_data.csv': 20,
    './outputs/1_5b/spider-syn/dev/error_analysis/sql_refinement_training_data.csv': 20,
}

# 定义输出文件路径
output_dir = './preprocess_data/combined/error_analysis'
output_file_name = 'sql_refinement_training_data.csv'
output_path = os.path.join(output_dir, output_file_name)

# 读取CSV文件并合并
all_dfs = []
for path in input_paths:
    try:
        df = pd.read_csv(path)
        if path in sample_counts:
            # 随机抽取指定数量的数据
            df = df.sample(n=sample_counts[path], random_state=42)
            print(f"Successfully loaded and randomly sampled {sample_counts[path]} entries from {path}")
        else:
            print(f"Successfully loaded all entries from {path}")
        all_dfs.append(df)
    except FileNotFoundError as e:
        print(f"Error: Input file not found: {path}. {e}")
        exit() # 如果文件未找到，则直接退出
    except Exception as e:
        print(f"Error reading file {path}: {e}")
        exit() # 如果读取文件时发生其他错误，也直接退出

# 如果所有文件都加载失败，all_dfs 将为空
if not all_dfs:
    print("No dataframes were loaded successfully. Exiting.")
    exit()

# 合并所有数据帧
# 假设所有CSV文件有相同的列结构，可以直接拼接
merged_df = pd.concat(all_dfs, ignore_index=True)

# 统计合并后的数据条数
print(f"Total number of merged data entries: {len(merged_df)}")

# 保存合并后的数据帧到新的CSV文件
merged_df.to_csv(output_path, index=False)

print(f"Successfully merged datasets from {len(input_paths)} files into {output_path}")
