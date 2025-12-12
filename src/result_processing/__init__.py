import logging
import json
import os
from .table_processor import save_table_extraction_results
from .sql_processor import save_sql_results
from .additional_data_processor import save_additional_data

def process_and_save_all_results(pipeline_results_file_path, output_base_dir, dataset_name, save_additional_data_flag=True):
    """
    处理管道执行结果并将其保存到指定文件。

    Args:
        pipeline_results_file_path (str): 存储管道中间结果的 JSONL 文件路径。
        output_base_dir (str): 基础输出目录。
        dataset_name (str): 数据集名称，用于构建输出路径。
    """
    logging.info(f"开始处理管道结果文件: {pipeline_results_file_path}")

    all_results = []
    try:
        with open(pipeline_results_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                all_results.append(json.loads(line.strip()))
        logging.info(f"成功从 {pipeline_results_file_path} 读取 {len(all_results)} 条结果。")
    except FileNotFoundError:
        logging.error(f"结果文件未找到: {pipeline_results_file_path}")
        return
    except Exception as e:
        logging.error(f"读取结果文件时发生错误: {e}")
        return

    if not all_results:
        logging.warning("没有结果可供处理。")
        return

    # 调用各个处理器保存结果
    save_table_extraction_results(all_results, output_base_dir, dataset_name)
    
    save_sql_results(all_results, output_base_dir, dataset_name, save_additional_data_flag)
        
    save_additional_data(all_results, output_base_dir, dataset_name, save_additional_data_flag)

    logging.info("所有结果处理和保存完成。")
