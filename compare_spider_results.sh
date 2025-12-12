#!/bin/bash

OUTPUT_BASE_DIR="./outputs/spider"
data_mode="dev"
DB_PATH="/path/to/Spider/database"

PRED1_PATH="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/refined_sqls_1.sql"
PRED2_PATH="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/refined_sqls_2.sql"
PRED3_PATH="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/selected_sqls.sql"
GOLD_PATH="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/gold_sqls.sql"
QUESTION_IDS_PATH="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/question_ids.txt"
META_TIME_OUT=30.0 # 默认超时时间

python ./evaluation/compare_evaluation_spider.py \
    --db "${DB_PATH}" \
    --pred1 "${PRED1_PATH}" \
    --pred2 "${PRED2_PATH}" \
    --pred3 "${PRED3_PATH}" \
    --gold "${GOLD_PATH}" \
    --question_ids_path "${QUESTION_IDS_PATH}" \
    --meta_time_out "${META_TIME_OUT}" \
    --etype "exec"
