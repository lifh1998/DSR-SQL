import logging
from typing import Any, Dict, List
from tqdm import tqdm # 导入 tqdm

from ..utils.prompts import sql_generation_prompt, cscsql_generation_prompt, cscsql_system_prompt
from ..core.task import Task # 导入 Task 类

logger = logging.getLogger(__name__)

def candidate_generate(tasks: List[Task], chat_model: Any) -> List[Dict[str, Any]]:
    """候选SQL生成节点"""
    results = []
    for task in tqdm(tasks, desc="生成候选SQL"): # 添加进度条
        # 生成第一个候选SQL（使用精简schema）
        ans1 = chat_model.get_ans(_build_messages(task.scaled_down_db_schema, task.question))
        candidate_sql_1 = _extract_ans(ans1)

        # 生成第二个候选SQL（使用完整schema）
        ans2 = chat_model.get_ans(_build_messages(task.database_schema, task.question))
        candidate_sql_2 = _extract_ans(ans2)

        response = {
            "question_id": task.question_id, # 确保包含 question_id
            "candidate_sql_1": candidate_sql_1,
            "candidate_sql_2": candidate_sql_2
        }
        results.append(response)
    return results

def _build_messages(database_schema, question):
    return sql_generation_prompt.format(
        database_schema=database_schema, 
        question=question
    )
    # return [{
    #     "role": "system",
    #     "content": cscsql_system_prompt
    # }, {
    #     "role": "user",
    #     "content": cscsql_generation_prompt.format(
    #         database_schema=database_schema, 
    #         question=question
    #     )
    # }]

def _extract_ans(ans):
    try:
        return ans.split('<answer>\n<sql>')[1].split('</sql>\n</answer>')[0].strip()
    except IndexError:
        logger.error(f"无法从LLM响应中提取SQL: {ans}")
        return "Extraction Error: Could not parse SQL from LLM response."

# def _extract_ans(ans):
#     try:
#         # 先提取 <answer> 标签内的内容
#         extracted = ans.split('<answer>')[1].split('</answer>')[0].strip()
        
#         # 去掉 ```sql 和 ``` 标记
#         if extracted.startswith('```sql'):
#             extracted = extracted[6:]  # 去掉开头的 ```sql
#         elif extracted.startswith('```'):
#             extracted = extracted[3:]  # 去掉开头的 ```
            
#         if extracted.endswith('```'):
#             extracted = extracted[:-3]  # 去掉结尾的 ```
            
#         return extracted.strip()
#     except IndexError:
#         logger.error(f"无法从LLM响应中提取SQL: {ans}")
#         return "Extraction Error: Could not parse SQL from LLM response."
