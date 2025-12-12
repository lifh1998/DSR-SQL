import pandas as pd
import os

# 定义文件路径
BASE_DIR = "./preprocess_data/combined"
MERGED_TRAIN_INPUT_FILE = os.path.join(BASE_DIR, "processed_dataset.csv")
SQL_REFINEMENT_INPUT_FILE = os.path.join(BASE_DIR, "error_analysis", "sql_refinement_training_data.csv")

TABLE_SELECTOR_OUTPUT_FILE = os.path.join(BASE_DIR, "table_selector_training_data.csv")
UNIFIED_SQL_OUTPUT_FILE = os.path.join(BASE_DIR, "unified_sql_training_data.csv")

def process_table_selector_data():
    """
    从原始CSV文件提取table selector微调所需要的列并保存到新的csv文件。
    """
    print(f"Processing table selector data from {MERGED_TRAIN_INPUT_FILE}...")
    try:
        df_merged_train = pd.read_csv(MERGED_TRAIN_INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {MERGED_TRAIN_INPUT_FILE} not found. Please ensure the file exists.")
        return

    # 提取table_extracter所需的列: database_schema, question, correct_tables
    required_columns = ['database_schema', 'question', 'correct_tables', 'is_error']
    
    # 检查所有必需列是否存在
    missing_columns = [col for col in required_columns if col not in df_merged_train.columns]
    if missing_columns:
        print(f"Warning: Missing columns in {MERGED_TRAIN_INPUT_FILE}: {missing_columns}. Skipping table selector data processing.")
        return

    df_table_selector = df_merged_train[required_columns].copy()

    # 过滤掉is_error为True的行，以及correct_tables为空的行
    df_table_selector = df_table_selector[~df_table_selector['is_error']]
    df_table_selector = df_table_selector.dropna(subset=['correct_tables'])
    df_table_selector = df_table_selector[df_table_selector['correct_tables'] != '']

    # 移除is_error列，因为它只是用于过滤
    df_table_selector = df_table_selector.drop(columns=['is_error'])

    df_table_selector.to_csv(TABLE_SELECTOR_OUTPUT_FILE, index=False)
    print(f"Table selector data saved to {TABLE_SELECTOR_OUTPUT_FILE}")

def merge_sql_generator_refiner_data():
    """
    通过两个原csv合并得到sql generator 以及 sql refiner都能使用的单个csv文件。
    """
    print(f"Merging SQL generator and refiner data...")
    try:
        df_generator = pd.read_csv(MERGED_TRAIN_INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {MERGED_TRAIN_INPUT_FILE} not found. Please ensure the file exists.")
        return

    try:
        df_refiner = pd.read_csv(SQL_REFINEMENT_INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: {SQL_REFINEMENT_INPUT_FILE} not found. Please ensure the file exists.")
        return

    # --- 处理 df_generator ---
    # 过滤掉 is_error 为 True 的行
    df_generator_processed = df_generator[~df_generator['is_error']].copy()
    df_generator_processed = df_generator_processed.rename(columns={'query': 'final_sql'})
    df_generator_processed['candidate_sql'] = None # sql_generator没有candidate_sql
    df_generator_processed['error_message'] = None
    df_generator_processed['task_type'] = 'sql_generation'
    
    # --- 处理 df_refiner ---
    df_refiner_processed = df_refiner.rename(columns={'sql': 'candidate_sql', 'refinement_sql': 'final_sql'})
    df_refiner_processed['task_type'] = 'sql_refinement'

    # 确保所有列都存在于两个DataFrame中，并保持一致的顺序
    common_columns = ['question', 'database_schema', 'candidate_sql', 'error_message', 'final_sql', 'task_type']
    
    # 检查并添加缺失列
    for col in common_columns:
        if col not in df_generator_processed.columns:
            df_generator_processed[col] = None
        if col not in df_refiner_processed.columns:
            df_refiner_processed[col] = None

    # 重新排序列
    df_generator_processed = df_generator_processed[common_columns]
    df_refiner_processed = df_refiner_processed[common_columns]

    # 合并数据集
    df_unified = pd.concat([df_generator_processed, df_refiner_processed], ignore_index=True)
    
    df_unified.to_csv(UNIFIED_SQL_OUTPUT_FILE, index=False)
    print(f"Unified SQL generator and refiner data saved to {UNIFIED_SQL_OUTPUT_FILE}")

if __name__ == "__main__":
    os.makedirs(BASE_DIR, exist_ok=True) # 确保目录存在
    process_table_selector_data()
    merge_sql_generator_refiner_data()
