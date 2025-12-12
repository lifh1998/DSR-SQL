import logging
import os

def save_sql_results(all_results, output_base_dir, dataset_name, save_additional_data_flag: bool = False):
    """
    将SQL结果写入文件。

    Args:
        all_results (list): 包含所有管道执行结果的列表。
        output_base_dir (str): 基础输出目录。
        dataset_name (str): 数据集名称，用于构建输出路径。
    """
    sql_output_dir = os.path.join(output_base_dir, dataset_name, 'sql_results')
    os.makedirs(sql_output_dir, exist_ok=True)

    gold_sql_file_path = os.path.join(sql_output_dir, "gold_sqls.sql")
    candidate_sql_1_file_path = os.path.join(sql_output_dir, "candidate_sqls_1.sql")
    candidate_sql_2_file_path = os.path.join(sql_output_dir, "candidate_sqls_2.sql")
    refined_sql_1_file_path = os.path.join(sql_output_dir, "refined_sqls_1.sql")
    refined_sql_2_file_path = os.path.join(sql_output_dir, "refined_sqls_2.sql")
    selected_sql_file_path = os.path.join(sql_output_dir, "selected_sqls.sql")
    question_ids_file_path = os.path.join(sql_output_dir, "question_ids.txt")

    with open(gold_sql_file_path, 'w', encoding='utf-8') as gold_f, \
         open(candidate_sql_1_file_path, 'w', encoding='utf-8') as cand1_f, \
         open(candidate_sql_2_file_path, 'w', encoding='utf-8') as cand2_f, \
         open(question_ids_file_path, 'w', encoding='utf-8') as qid_f:
         for result in all_results:
            question_id = result.get('question_id', '')
            db_id = result.get('db_id', '')
            gold_sql = result.get('query', '')
            candidate_sql_1 = result.get('candidate_sql_1', '')
            candidate_sql_2 = result.get('candidate_sql_2', '')

            # 移除SQL字符串中的所有换行符，确保每条SQL只占一行
            gold_sql = gold_sql.replace('\n', ' ').replace('\r', ' ')
            candidate_sql_1 = candidate_sql_1.replace('\n', ' ').replace('\r', ' ')
            candidate_sql_2 = candidate_sql_2.replace('\n', ' ').replace('\r', ' ')
            
            gold_f.write(f"{gold_sql}\t{db_id}\n")
            cand1_f.write(f"{candidate_sql_1}\n")
            cand2_f.write(f"{candidate_sql_2}\n")
            qid_f.write(f"{question_id}\n")

    if not save_additional_data_flag:
        with open(refined_sql_1_file_path, 'w', encoding='utf-8') as refined1_f, \
            open(refined_sql_2_file_path, 'w', encoding='utf-8') as refined2_f, \
            open(selected_sql_file_path, 'w', encoding='utf-8') as selected_f:
            for result in all_results:
                refined_sql_1 = result.get('refined_sql_1', '')
                refined_sql_2 = result.get('refined_sql_2', '')
                selected_sql = result.get('selected_sql', '')

                # 移除SQL字符串中的所有换行符，确保每条SQL只占一行
                refined_sql_1 = refined_sql_1.replace('\n', ' ').replace('\r', ' ')
                refined_sql_2 = refined_sql_2.replace('\n', ' ').replace('\r', ' ')
                selected_sql = selected_sql.replace('\n', ' ').replace('\r', ' ')

                refined1_f.write(f"{refined_sql_1}\n")
                refined2_f.write(f"{refined_sql_2}\n")
                selected_f.write(f"{selected_sql}\n")

    logging.info(f"SQL生成结果已写入: {sql_output_dir}")
