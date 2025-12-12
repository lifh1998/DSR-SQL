#!/bin/bash

MODEL_TYPE="qwen"

# MODEL_TYPE=${MODEL_TYPE} AGENT_TYPE=table_extracter python3 supervised_finetuning.py
# echo "表选择模型微调完成!"

MODEL_TYPE=${MODEL_TYPE} AGENT_TYPE=sql_generator python3 supervised_finetuning.py
echo "SQL生成模型微调完成!"

# MODEL_TYPE=${MODEL_TYPE} AGENT_TYPE=sql_refiner python3 supervised_finetuning.py
# echo "SQL精炼模型微调完成!"

# MODEL_TYPE=${MODEL_TYPE} AGENT_TYPE=sql_selector_classifier python3 supervised_finetuning.py
# echo "SQL选择模型微调完成!"
