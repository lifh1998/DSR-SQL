import logging
from typing import Any, Dict, List, Iterable # 导入 Iterable
from ..managers.pipeline_manager import PipelineManager
from ..managers.database_manager import DatabaseManager
from ..utils.model_utils import model_chose
from ..utils.prompts import sql_selection_prompt, cscsql_merge_prompt, cscsql_system_prompt
from ..utils.db_utils import execute_sql_query, convert_row_to_list # 导入 convert_row_to_list
from ..core.task import Task
from tqdm import tqdm # 导入 tqdm

logger = logging.getLogger(__name__)

def select_sql(tasks: List[Task], chat_model: Any) -> List[Dict[str, Any]]:
    """
    SQL选择节点。
    :param tasks: 当前任务对象列表。
    :param chat_model: 用于SQL选择的聊天模型实例。
    :param execution_history: 管道流的执行历史。
    :return: 包含选择结果的字典列表。
    """
    database_manager = DatabaseManager()

    logger.info("开始选择最终SQL。")

    results = []
    for task in tqdm(tasks, desc="选择最终SQL"): # 添加进度条
        refined_sql_1 = task.refined_sql_1
        refined_sql_2 = task.refined_sql_2
        
        sql1_final_error = task.sql1_final_error
        sql2_final_error = task.sql2_final_error
        
        sql1_exec_results = task.sql1_exec_results
        sql2_exec_results = task.sql2_exec_results
        
        sql1_exec_time = task.sql1_exec_time
        sql2_exec_time = task.sql2_exec_time

        try:
            selected_sql = _merge_sql_with_llm(
                chat_model, task, database_manager,
                refined_sql_1, refined_sql_2,
                sql1_final_error, sql2_final_error, 
                sql1_exec_results, sql2_exec_results,
                sql1_exec_time, sql2_exec_time
            )
        except Exception as e:
            logger.error(f"调用 _merge_sql_with_llm 时发生异常: {e}")
            selected_sql = _fallback_sql_selection(
                refined_sql_1, refined_sql_2,
                sql1_final_error, sql2_final_error,
                sql1_exec_results, sql2_exec_results,
                sql1_exec_time, sql2_exec_time
            )

        result = {
            "question_id": task.question_id,
            "selected_sql": selected_sql,
            "status": "success"
        }
        results.append(result)
    return results

def _merge_sql_with_llm(
    chat_model: Any, task: Task, database_manager,
    refined_sql_1, refined_sql_2, 
    sql1_final_error, sql2_final_error, 
    sql1_exec_results, sql2_exec_results,
    sql1_exec_time, sql2_exec_time
):
    """
    使用LLM合并或修正SQL。
    """
    prompt_content = cscsql_merge_prompt.format(
        question=task.question,
        database_schema=task.database_schema,
        candidate_sql_1=refined_sql_1,
        sql1_exec_results=sql1_final_error if sql1_final_error else truncated_str(sql1_exec_results, 10),
        candidate_sql_2=refined_sql_2,
        sql2_exec_results=sql2_final_error if sql2_final_error else truncated_str(sql2_exec_results, 10),
    )

    messages = [{
        "role": "system",
        "content": cscsql_system_prompt
    }, {
        "role": "user",
        "content": prompt_content
    }]

    ans = chat_model.get_ans(messages)
    merged_sql = _extract_ans(ans)
    # logger.info(f"LLM merged SQL: {merged_sql}")

    db_path = database_manager.get_db_path(task.db_id)
    results, sql_exec_error, current_exec_time = execute_sql_query(db_path, merged_sql)
    
    # 确保 results 可序列化
    results = convert_row_to_list(results)

    if sql_exec_error:
        logger.info("merged_sql执行失败，降级为选择任务。")
        return _fallback_sql_selection(
            refined_sql_1, refined_sql_2,
            sql1_final_error, sql2_final_error,
            sql1_exec_results, sql2_exec_results,
            sql1_exec_time, sql2_exec_time
        )
    else:
        return merged_sql

def truncated_str(sql1_exec_results, n):
    """
    返回 sql1_exec_results 的字符串表示：
      - 如果长度 <= n：返回完整的 str(sql1_exec_results)
      - 否则：返回前 n 个元素的 str(...) 并在末尾追加 "..."
    假设传入值合法（序列或可迭代），且 n 为非负整数。
    """
    if n is None or n < 0:
        return str(sql1_exec_results)

    # 优先尝试使用 len 和切片（适用于 list/tuple/str 等）
    try:
        if len(sql1_exec_results) <= n:
            return str(sql1_exec_results)
        # 若支持切片，则取前 n 项
        try:
            prefix = sql1_exec_results[:n]
            return str(prefix) + "..."
        except Exception:
            # 回退：构造前 n 项的列表
            prefix = []
            it = iter(sql1_exec_results)
            for _ in range(n):
                prefix.append(next(it))
            return str(prefix) + "..."
    except Exception:
        # 对于没有 len 的可迭代对象，取 n+1 项判断是否需要 "..."
        if isinstance(sql1_exec_results, Iterable):
            it = iter(sql1_exec_results)
            prefix = []
            for _ in range(n + 1):
                try:
                    prefix.append(next(it))
                except StopIteration:
                    return str(prefix)
            return str(prefix[:n]) + "..."
        # 其它情况直接返回完整字符串
        return str(sql1_exec_results)

def _fallback_sql_selection(
    refined_sql_1, refined_sql_2,
    sql1_final_error, sql2_final_error,
    sql1_exec_results, sql2_exec_results,
    sql1_exec_time, sql2_exec_time
):
    """
    降级SQL选择逻辑，当合并的SQL执行失败时使用。
    """
    if sql1_final_error and sql2_final_error:
        logger.info("情况1: 两个SQL都执行失败。")
        if sql1_final_error == "Empty result" and sql2_final_error != "Empty result":
            selected_sql = refined_sql_1
            logger.info("优先选择错误信息为'Empty result'的SQL1。")
        elif sql2_final_error == "Empty result" and sql1_final_error != "Empty result":
            selected_sql = refined_sql_2
            logger.info("优先选择错误信息为'Empty result'的SQL2。")
        else: # 都是Empty result或都有其他错误
            selected_sql = refined_sql_1 # 选第一个返回
            logger.info("两个SQL错误类型相同或都为'Empty result'，选择SQL1。")
    elif sql1_final_error and not sql2_final_error:
        selected_sql = refined_sql_2
        logger.info("情况2: SQL1执行失败，选择SQL2。")
    elif not sql1_final_error and sql2_final_error:
        selected_sql = refined_sql_1
        logger.info("情况2: SQL2执行失败，选择SQL1。")
    else:
        # 检查结果是否一致
        if sql1_exec_results == sql2_exec_results:
            logger.info("情况3: 两个SQL都执行成功且结果一致。")
            if sql1_exec_time <= sql2_exec_time:
                selected_sql = refined_sql_1
                logger.info("选择查询时间更短的SQL1。")
            else:
                selected_sql = refined_sql_2
                logger.info("选择查询时间更短的SQL2。")
        else:
            # 策略二：
            logger.info("情况4: 两个SQL执行成功但结果不一致，直接选择Model1。")
            selected_sql = refined_sql_1
    
    return selected_sql

def _extract_ans(ans):
    try:
        # 先提取 <answer> 标签内的内容
        extracted = ans.split('<answer>')[1].split('</answer>')[0].strip()
        
        # 去掉 ```sql 和 ``` 标记
        if extracted.startswith('```sql'):
            extracted = extracted[6:]  # 去掉开头的 ```sql
        elif extracted.startswith('```'):
            extracted = extracted[3:]  # 去掉开头的 ```
            
        if extracted.endswith('```'):
            extracted = extracted[:-3]  # 去掉结尾的 ```
            
        return extracted.strip()
    except IndexError:
        logger.error(f"无法从LLM响应中提取SQL: {ans}")
        return "Extraction Error: Could not parse SQL from LLM response."
