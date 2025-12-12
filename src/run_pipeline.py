import logging
import json
import pandas as pd
from pipeline import Pipeline, Task
import os 
from result_processing import process_and_save_all_results
from transformers import AutoTokenizer
import argparse
from pipeline.managers.database_manager import DatabaseManager
from pipeline.managers.pipeline_manager import PipelineManager

def filter_dataframe_by_schema_token_length(df: pd.DataFrame, tokenizer, max_token_length: int = 8192):
    """
    根据database_schema的token序列长度过滤DataFrame。
    
    Args:
        df (pd.DataFrame): 原始DataFrame。
        tokenizer: 用于计算token长度的tokenizer。
        max_token_length (int): 允许的最大token长度。
        
    Returns:
        tuple: 包含过滤后的DataFrame、原始行数、过滤掉的行数和保留的行数。
    """
    original_rows = len(df)
    filtered_df = pd.DataFrame(columns=df.columns)
    filtered_out_count = 0

    for index, row in df.iterrows():
        schema_text = str(row['database_schema'])
        token_length = len(tokenizer.encode(schema_text, add_special_tokens=False))
        
        if token_length <= max_token_length:
            filtered_df = pd.concat([filtered_df, pd.DataFrame([row])], ignore_index=True)
        else:
            filtered_out_count += 1

    remaining_rows = len(filtered_df)
    logging.info(f"总共数据条数: {original_rows}")
    logging.info(f"过滤掉的数据条数: {filtered_out_count}")
    logging.info(f"最终保留的数据条数: {remaining_rows}")
    
    return filtered_df, original_rows, filtered_out_count, remaining_rows

def main():
    # 配置日志
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    parser = argparse.ArgumentParser(description="运行Text-to-SQL管道流")
    parser.add_argument('--output_base_dir', type=str, default='../outputs/7b',
                        help='输出文件的基础目录')
    parser.add_argument('--dataset_name', type=str, default='spider',
                        help='数据集名称 (例如: spider, bird)')
    parser.add_argument('--save_additional_data', action='store_true',
                        help='是否保存额外的中间数据。如果设置为True，则在生成候选SQL后停止，无需SQL refine和SQL select。')
    parser.add_argument('--db_schema_dir', type=str, default='../preprocess_data/spider/db_schemas',
                        help='数据库schema文件目录')
    parser.add_argument('--db_root_dir', type=str, default='../../datasets/Spider/database',
                        help='数据库根目录')
    parser.add_argument('--csv_file_path', type=str, default='../preprocess_data/spider/dev/processed_dataset.csv',
                        help='包含任务数据的CSV文件路径')
    parser.add_argument('--pipeline_configs_path', type=str, default=None,
                        help='PipelineManager配置的JSON文件路径。如果未提供，将使用默认配置。')
    parser.add_argument('--max_schema_token_length', type=int, default=None,
                        help='数据库schema的最大token长度。如果未提供，则不进行过滤。')
    parser.add_argument('--use_optimal_table_selection', action='store_true',
                        help='是否使用最优的表选择结果。如果设置为True，则表选择直接返回正确的表。')
    parser.add_argument('--disable_refine_sql', action='store_true',
                        help='是否禁用SQL refine模块。如果设置为True，则refined_sql将直接使用candidate_sql。')
    
    args = parser.parse_args()

    output_base_dir = args.output_base_dir
    dataset_name = args.dataset_name
    SAVE_ADDITIONAL_DATA = args.save_additional_data
    db_schema_dir = args.db_schema_dir
    db_root_dir = args.db_root_dir
    csv_file_path = args.csv_file_path
    pipeline_configs_path = args.pipeline_configs_path
    max_schema_token_length = args.max_schema_token_length

    table_output_dir = os.path.join(output_base_dir, dataset_name, 'table_results')
    sql_output_dir = os.path.join(output_base_dir, dataset_name, 'sql_results')
    
    os.makedirs(sql_output_dir, exist_ok=True)
    logging.info(f"SQL结果将写入目录: {sql_output_dir}")
    os.makedirs(table_output_dir, exist_ok=True)
    logging.info(f"表格提取结果将写入目录: {table_output_dir}")
    
    error_log_file_path = os.path.join(output_base_dir, dataset_name, 'error_log.jsonl')
    os.makedirs(os.path.dirname(error_log_file_path), exist_ok=True)
    logging.info(f"错误日志将写入: {error_log_file_path}")

    # 实例化DatabaseManager
    DatabaseManager(db_schema_dir=db_schema_dir, db_root_dir=db_root_dir)
    logging.info(f"DatabaseManager已初始化，db_schema_dir设置为：{db_schema_dir}，db_root_dir设置为：{db_root_dir}")

    # 实例化PipelineManager
    pipeline_configs = None
    if pipeline_configs_path:
        try:
            with open(pipeline_configs_path, 'r', encoding='utf-8') as f:
                pipeline_configs = json.load(f)
            logging.info(f"成功从 {pipeline_configs_path} 加载PipelineManager配置。")
        except FileNotFoundError:
            logging.error(f"PipelineManager配置文件未找到: {pipeline_configs_path}，将使用默认配置。")
        except json.JSONDecodeError:
            logging.error(f"PipelineManager配置文件 {pipeline_configs_path} 格式错误，将使用默认配置。")
    
    PipelineManager(configs=pipeline_configs)
    logging.info("PipelineManager已初始化。")

    # 使用pandas读取CSV文件
    try:
        df = pd.read_csv(csv_file_path)
        logging.info(f"成功读取CSV文件: {csv_file_path}")
        df = df.sort_values(by='question_id', ascending=True)
        logging.info("CSV文件已按照'question_id'升序排序。")
    except FileNotFoundError:
        logging.error(f"文件未找到: {csv_file_path}")
        return
    except Exception as e:
        logging.error(f"读取CSV文件时发生错误: {e}")
        return

    # 初始化tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", trust_remote_code=True)
    logging.info("Qwen/Qwen2.5-Coder-7B-Instruct tokenizer已加载。")

    # 过滤数据
    if max_schema_token_length is not None:
        logging.info(f"将根据database_schema的token长度过滤数据，最大长度为: {max_schema_token_length}")
        df, original_rows, filtered_out_count, remaining_rows = filter_dataframe_by_schema_token_length(df, tokenizer, max_schema_token_length)
    else:
        logging.info("未提供--max_schema_token_length参数，跳过数据过滤。")

    # 构建Task集合
    tasks = []
    for index, row in df.iterrows():
        task = Task(
            question_id=row['question_id'],
            db_id=row['db_id'],
            question=row['question'],
            database_schema=row['database_schema'], 
            query=row['query'],
            correct_tables=row['correct_tables'],
        )
        tasks.append(task)
    
    if not tasks:
        logging.warning("没有从CSV文件中构建任何任务。")
        return

    # 创建并执行管道流
    pipeline = Pipeline(
        output_base_dir=output_base_dir, 
        dataset_name=dataset_name,
        use_optimal_table_selection=args.use_optimal_table_selection, # 传递新参数
        disable_refine_sql=args.disable_refine_sql # 传递新参数
    )
    
    # 批量执行管道流
    logging.info(f"开始批量执行 {len(tasks)} 个任务...")
    try:
        final_pipeline_results = pipeline.execute_batch(tasks, save_additional_data=SAVE_ADDITIONAL_DATA)
        pipeline_results_file_path = os.path.join(output_base_dir, dataset_name, 'pipeline_results.jsonl')
        with open(pipeline_results_file_path, 'w', encoding='utf-8') as f:
            for res in final_pipeline_results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        logging.info(f"所有任务批量执行完成，最终结果已写入: {pipeline_results_file_path}")
    except Exception as e:
        logging.error(f"批量执行管道流时发生错误: {e}")
        return

    # 调用 result_processing 处理结果
    logging.info("所有任务执行完毕，开始处理和保存结果...")
    process_and_save_all_results(pipeline_results_file_path, output_base_dir, dataset_name, SAVE_ADDITIONAL_DATA)
    logging.info("结果处理和保存完成。")
    
if __name__ == "__main__":
    main()
