import logging
import time
from typing import Any, Dict, List, Tuple
from ..core.task import Task
from tqdm import tqdm # 导入 tqdm
from ..managers.database_manager import DatabaseManager
from ..utils.prompts import sql_refinement_prompt
from ..utils.db_utils import execute_sql_query, convert_row_to_list
import logging
import time

logger = logging.getLogger(__name__)

def refine_candidate(tasks: List[Task], chat_model: Any, disable_refine_sql: bool = False) -> List[Dict[str, Any]]:
    """
    精炼候选SQL。
    :param tasks: 当前任务对象列表。
    :param chat_model: 用于SQL精炼的聊天模型实例。
    :param disable_refine_sql: 是否禁用SQL精炼模块。
    :return: 包含精炼结果的字典列表。
    """
    database_manager = DatabaseManager()
    max_refine_iterations = 3
    logger.info("开始精炼候选SQL。")

    results = []
    if disable_refine_sql:
        logger.info("disable_refine_sql 为 True，跳过SQL精炼步骤，直接使用candidate_sql作为refined_sql。")
        for task in tqdm(tasks, desc="跳过SQL精炼"):
            result = {
                "question_id": task.question_id,
                "refined_sql_1": task.candidate_sql_1,
                "refined_sql_2": task.candidate_sql_2,
                "sql1_final_error": "", # 禁用精炼时，没有错误
                "sql2_final_error": "",
                "sql1_exec_results": [], # 禁用精炼时，没有执行结果
                "sql2_exec_results": [],
                "sql1_exec_time": 0.0, # 禁用精炼时，没有执行时间
                "sql2_exec_time": 0.0,
                "status": "success"
            }
            results.append(result)
        return results

    for task in tqdm(tasks, desc="精炼候选SQL"): # 添加进度条

        db_path = database_manager.get_db_path(task.db_id)
        question = task.question

        # 精炼第一个候选SQL
        refined_sql_1, sql1_final_error, sql1_exec_results, sql1_exec_time = _iterative_refine_sql(
            db_path, question, task.candidate_sql_1, task.scaled_down_db_schema, max_refine_iterations, chat_model, "第一个候选SQL"
        )
        # 确保 exec_results 可序列化，递归处理
        sql1_exec_results = convert_row_to_list(sql1_exec_results)

        # 精炼第二个候选SQL
        refined_sql_2, sql2_final_error, sql2_exec_results, sql2_exec_time = _iterative_refine_sql(
            db_path, question, task.candidate_sql_2, task.database_schema, max_refine_iterations, chat_model, "第二个候选SQL"
        )
        # 确保 exec_results 可序列化，递归处理
        sql2_exec_results = convert_row_to_list(sql2_exec_results)

        result = {
            "question_id": task.question_id,
            "refined_sql_1": refined_sql_1,
            "refined_sql_2": refined_sql_2,
            "sql1_final_error": sql1_final_error,
            "sql2_final_error": sql2_final_error,
            "sql1_exec_results": sql1_exec_results,
            "sql2_exec_results": sql2_exec_results,
            "sql1_exec_time": sql1_exec_time,
            "sql2_exec_time": sql2_exec_time,
            "status": "success"
        }
        results.append(result)
    return results

def _iterative_refine_sql(
    db_path: str,
    question: str,
    current_sql: str,
    db_schema: str,
    max_refine_iterations: int,
    chat_model: Any,
    sql_type_label: str # 用于日志输出，例如 "第一个候选SQL"
) -> Tuple[str, str, List[Any], float]: # 修改返回类型以适应更通用的列表
    """
    辅助函数：迭代精炼SQL。
    """
    final_sql = current_sql
    final_error = ""
    exec_results = []
    exec_time = 0.0

    for i in range(max_refine_iterations):
        results, sql_exec_error, current_exec_time = execute_sql_query(db_path, final_sql)
        exec_time = current_exec_time # 记录每次执行的时间

        if not sql_exec_error: # 如果没有错误，则精炼成功，跳出循环
            exec_results = results
            final_error = ""
            # logger.info(f"{sql_type_label}精炼成功，结果: {final_sql}")
            break
        else:
            # logger.info(f"精炼{sql_type_label} (迭代 {i+1}/{max_refine_iterations}): {final_sql}")
            final_error = sql_exec_error
            # logger.warning(f"{sql_type_label}执行错误: {final_error}")
            prompt = sql_refinement_prompt.format(
                database_schema=db_schema,
                question=question,
                candidate_sql=final_sql,
                error_message=final_error
            )
            try:
                ans = chat_model.get_ans(prompt)
                final_sql = _extract_ans(ans)
            except Exception as model_e:
                logger.error(f"调用模型时发生错误: {str(model_e)}")
                final_sql = "Error during model call."
                final_error = f"Model call error: {str(model_e)}"
                break # 发生模型错误，停止精炼
    else: # 如果循环结束仍未成功，则使用最后一次精炼的结果
        pass
        # logger.warning(f"{sql_type_label}达到最大精炼次数，最终结果: {final_sql}")

    return final_sql, final_error, exec_results, exec_time

def _extract_ans(ans):
    try:
        return ans.split('<answer>\n<sql>')[1].split('</sql>\n</answer>')[0].strip()
    except IndexError:
        logger.error(f"无法从LLM响应中提取SQL: {ans}")
        return "Extraction Error: Could not parse SQL from LLM response."
