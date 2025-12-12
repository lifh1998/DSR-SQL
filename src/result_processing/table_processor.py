import logging
import pandas as pd
import os

def save_table_extraction_results(all_results, output_base_dir, dataset_name):
    """
    将表格提取结果写入CSV文件。

    Args:
        all_results (list): 包含所有管道执行结果的列表。
        output_base_dir (str): 基础输出目录。
        dataset_name (str): 数据集名称，用于构建输出路径。
    """
    table_output_dir = os.path.join(output_base_dir, dataset_name, 'table_results')
    os.makedirs(table_output_dir, exist_ok=True)

    table_results_list = []
    for result in all_results:
        related_tables = result.get('related_tables', '')
        correct_tables = result.get('correct_tables', '')
        
        table_results_list.append({
            'predicted_tables': related_tables,
            'reference_tables': correct_tables
        })
    if table_results_list:
        table_output_file_path = os.path.join(table_output_dir, "table_extraction_results.csv")
        table_df = pd.DataFrame(table_results_list)
        table_df.to_csv(table_output_file_path, index=False, encoding='utf-8')
        logging.info(f"表格提取结果已写入: {table_output_file_path}")
    else:
        logging.warning("没有表格提取结果可写入。")
