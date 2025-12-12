import logging
import pandas as pd
import os

def save_additional_data(all_results, output_base_dir, dataset_name, save_additional_data_flag=True):
    """
    保存额外的CSV数据。

    Args:
        all_results (list): 包含所有管道执行结果的列表。
        output_base_dir (str): 基础输出目录。
        dataset_name (str): 数据集名称，用于构建输出路径。
        save_additional_data_flag (bool): 是否保存额外数据的标志。
    """
    sql_output_dir = os.path.join(output_base_dir, dataset_name, 'sql_results')
    os.makedirs(sql_output_dir, exist_ok=True)

    if save_additional_data_flag:
        logging.info("正在保存额外的CSV数据...")
        data_for_file1 = []
        data_for_file2 = []

        for result in all_results:
            question = result.get('question', '')
            database_schema = result.get('database_schema', '')
            scaled_down_db_schema = result.get('scaled_down_db_schema', '')
            candidate_sqls_1 = result.get('candidate_sql_1', '')
            candidate_sqls_2 = result.get('candidate_sql_2', '')

            data_for_file1.append({
                'question': question,
                'database_schema': scaled_down_db_schema,
                'candidate_sql': candidate_sqls_1
            })
            data_for_file2.append({
                'question': question,
                'database_schema': database_schema,
                'candidate_sql': candidate_sqls_2
            })

        if data_for_file1:
            df1 = pd.DataFrame(data_for_file1)
            file1_path = os.path.join(sql_output_dir, "additional_data_1.csv")
            df1.to_csv(file1_path, index=False, encoding='utf-8')
            logging.info(f"额外数据文件1已写入: {file1_path}")
        else:
            logging.warning("没有额外数据文件1可写入。")

        if data_for_file2:
            df2 = pd.DataFrame(data_for_file2)
            file2_path = os.path.join(sql_output_dir, "additional_data_2.csv")
            df2.to_csv(file2_path, index=False, encoding='utf-8')
            logging.info(f"额外数据文件2已写入: {file2_path}")
        else:
            logging.warning("没有额外数据文件2可写入。")
    else:
        logging.info("额外数据保存功能已禁用。")
