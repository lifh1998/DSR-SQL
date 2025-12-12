#!/bin/bash

# 进入脚本所在的目录，确保相对路径正确
SCRIPT_DIR=$(dirname "$0")
cd "$SCRIPT_DIR"

# 定义默认参数
OUTPUT_BASE_DIR="outputs/"
DATASET_NAME="bird/dev"
DB_SCHEMA_DIR="preprocess_data/bird/dev/db_schemas"
DB_ROOT_DIR="/path/to/BIRD/dev/dev_databases"
CSV_FILE_PATH="preprocess_data/bird/dev/processed_dataset.csv"
PIPELINE_CONFIGS_PATH="config/pipeline_configs.json"
MAX_SCHEMA_TOKEN_LENGTH=8192
USE_OPTIMAL_TABLE_SELECTION=false # 新增参数，默认不使用最优表选择
ENABLE_REFINE_SQL=true # 新增参数，默认启用refine_sql模块

# 解析命令行参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        --output_base_dir) OUTPUT_BASE_DIR="$2"; shift ;;
        --dataset_name) DATASET_NAME="$2"; shift ;;
        --db_schema_dir) DB_SCHEMA_DIR="$2"; shift ;;
        --db_root_dir) DB_ROOT_DIR="$2"; shift ;;
        --csv_file_path) CSV_FILE_PATH="$2"; shift ;;
        --pipeline_configs_path) PIPELINE_CONFIGS_PATH="$2"; shift ;;
        --max_schema_token_length) MAX_SCHEMA_TOKEN_LENGTH="$2"; shift ;;
        --use_optimal_table_selection) USE_OPTIMAL_TABLE_SELECTION=true ;;
        --disable_refine_sql) ENABLE_REFINE_SQL=false ;;
        *) echo "未知参数: $1"; exit 1 ;;
    esac
    shift
done

# 检查是否存在自定义的PipelineManager配置
if [ ! -f "$PIPELINE_CONFIGS_PATH" ]; then
    echo "警告: 未找到PipelineManager配置文件 $PIPELINE_CONFIGS_PATH，将使用默认配置。"
    PIPELINE_CONFIGS_ARG=""
else
    PIPELINE_CONFIGS_ARG="--pipeline_configs_path $PIPELINE_CONFIGS_PATH"
fi

# 检查是否提供了MAX_SCHEMA_TOKEN_LENGTH参数
if [ -n "$MAX_SCHEMA_TOKEN_LENGTH" ]; then
    MAX_SCHEMA_TOKEN_LENGTH_ARG="--max_schema_token_length $MAX_SCHEMA_TOKEN_LENGTH"
else
    MAX_SCHEMA_TOKEN_LENGTH_ARG=""
fi

# 根据USE_OPTIMAL_TABLE_SELECTION变量设置参数
if [ "$USE_OPTIMAL_TABLE_SELECTION" = true ]; then
    USE_OPTIMAL_TABLE_SELECTION_ARG="--use_optimal_table_selection"
else
    USE_OPTIMAL_TABLE_SELECTION_ARG=""
fi

# 根据ENABLE_REFINE_SQL变量设置参数
if [ "$ENABLE_REFINE_SQL" = true ]; then
    ENABLE_REFINE_SQL_ARG="" # 默认启用，不需要传递额外参数
else
    ENABLE_REFINE_SQL_ARG="--disable_refine_sql" # 禁用时传递参数
fi

# 执行Python脚本
python src/run_pipeline.py \
    --output_base_dir "$OUTPUT_BASE_DIR" \
    --dataset_name "$DATASET_NAME" \
    --db_schema_dir "$DB_SCHEMA_DIR" \
    --db_root_dir "$DB_ROOT_DIR" \
    --csv_file_path "$CSV_FILE_PATH" \
    $PIPELINE_CONFIGS_ARG \
    $MAX_SCHEMA_TOKEN_LENGTH_ARG \
    $USE_OPTIMAL_TABLE_SELECTION_ARG \
    $ENABLE_REFINE_SQL_ARG \
    # --save_additional_data

echo "管道流执行完成。"
