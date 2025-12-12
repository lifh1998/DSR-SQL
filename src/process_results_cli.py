import os
import logging
from result_processing import process_and_save_all_results

def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 定义参数
    # 请根据实际情况修改这些路径
    pipeline_results_file_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs', 'train', 'bird', 'pipeline_results.jsonl'
    )
    output_base_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'outputs'
    )
    dataset_name = "train/bird"
    save_additional_data = True

    logging.info(f"开始处理文件: {pipeline_results_file_path}")
    logging.info(f"输出基础目录: {output_base_dir}")
    logging.info(f"数据集名称: {dataset_name}")
    logging.info(f"保存额外数据: {save_additional_data}")

    process_and_save_all_results(
        pipeline_results_file_path,
        output_base_dir,
        dataset_name,
        save_additional_data
    )
    logging.info("结果处理完成。")

if __name__ == "__main__":
    main()
