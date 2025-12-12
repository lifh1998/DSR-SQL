#!/bin/bash

OUTPUT_BASE_DIR="./outputs/bird"
data_mode="dev"
db_root_path="/path/to/BIRD/dev/dev_databases"
diff_json_path="/path/to/BIRD/dev/dev.json"

predicted_sql_json_path_1="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/refined_sqls_1.sql"
predicted_sql_json_path_2="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/refined_sqls_2.sql"
predicted_sql_json_path_3="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/selected_sqls.sql"
ground_truth_sql_path="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/gold_sqls.sql"
question_ids_path="${OUTPUT_BASE_DIR}/${data_mode}/sql_results/question_ids.txt"
num_cpus=12
meta_time_out=60.0
mode_gt="gt"
mode_predict="gpt"

# compare EX
echo "Compare BIRD EX begin!"
CMD="python ./evaluation/compare_evaluation_bird.py \
    --predicted_sql_json_path_1 \"$predicted_sql_json_path_1\" \
    --predicted_sql_json_path_2 \"$predicted_sql_json_path_2\""

if [ -n "$predicted_sql_json_path_3" ]; then
    CMD="$CMD --predicted_sql_json_path_3 \"$predicted_sql_json_path_3\""
fi

CMD="$CMD \
    --ground_truth_sql_path \"$ground_truth_sql_path\" \
    --question_ids_path \"$question_ids_path\" \
    --data_mode \"$data_mode\" \
    --db_root_path \"$db_root_path\" \
    --diff_json_path \"$diff_json_path\" \
    --num_cpus \"$num_cpus\" \
    --meta_time_out \"$meta_time_out\" \
    --mode_predict \"$mode_predict\""

eval $CMD
echo "Compare EX done!"
