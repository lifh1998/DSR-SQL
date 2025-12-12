import sqlite3
import logging
import time
import threading
from typing import Any, List, Tuple, Dict

logger = logging.getLogger(__name__)

# 辅助函数：递归地将包含 Row 对象的列表或嵌套结构转换为包含列表的列表，以便 JSON 序列化。
def convert_row_to_list(data: Any) -> Any:
    """
    递归地将包含 Row 对象的列表或嵌套结构转换为包含列表的列表，以便 JSON 序列化。
    """
    if isinstance(data, list):
        return [convert_row_to_list(item) for item in data]
    elif hasattr(data, '__iter__') and not isinstance(data, (str, bytes, dict)):
        # 如果是可迭代对象（如 Row），但不是字符串、字节或字典，则转换为列表
        return list(data)
    elif isinstance(data, dict):
        return {k: convert_row_to_list(v) for k, v in data.items()}
    else:
        return data

# 用于在线程间传递结果的辅助类
class QueryResult:
    def __init__(self):
        self.results = []
        self.error = ""
        self.execution_time = -1.0

def _query_worker(db_path: str, query: str, result_obj: QueryResult):
    """
    在单独的线程中执行SQL查询。
    """
    conn = None
    try:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        start_time = time.time()
        cursor.execute(query)
        result_obj.results = cursor.fetchall()
        end_time = time.time()
        result_obj.execution_time = end_time - start_time

        if not result_obj.results:
            result_obj.error = "Empty result."
    except sqlite3.Error as e:
        result_obj.error = str(e)
        logger.error(f"SQL执行错误: {e}")
    except Exception as e:
        result_obj.error = str(e)
        logger.error(f"发生意外错误: {e}")
    finally:
        if conn:
            conn.close()

def execute_sql_query(db_path: str, query: str, timeout: float = 300.0) -> Tuple[List[Tuple[Any, ...]], str, float]:
    """
    执行SQL查询并返回结果、错误信息和执行时间，支持超时机制。
    :param db_path: 数据库文件的路径。
    :param query: 要执行的SQL查询。
    :param timeout: 查询时间阈值（秒）。如果查询时间超过此值，将返回超时错误。
    :return: 一个元组，包含查询结果（如果成功）、错误信息（如果失败）和执行时间。
             如果成功且有结果，返回 (results, "", execution_time)。
             如果成功但结果为空，返回 ([], "Empty result", execution_time)。
             如果失败，返回 ([], error_message, execution_time)。
             如果超时，返回 ([], "Query timed out.", execution_time)。
    """
    result_obj = QueryResult()
    query_thread = threading.Thread(target=_query_worker, args=(db_path, query, result_obj))
    query_thread.start()
    query_thread.join(timeout=timeout)

    if query_thread.is_alive():
        # 线程仍然存活，表示超时
        logger.warning(f"SQL查询超时 (>{timeout}秒): {query}")
        # 尝试中断线程（SQLite连接可能无法被外部中断，但我们可以返回超时错误）
        # 在实际应用中，对于SQLite，可能需要更复杂的机制来终止长时间运行的查询，
        # 例如关闭连接，但这可能导致数据库文件锁定或损坏，因此通常不推荐。
        # 这里我们只返回超时错误，并让线程自行完成（或在Python解释器关闭时终止）。
        return [], "Query timed out.", timeout
    else:
        # 线程已完成
        if result_obj.error:
            return [], result_obj.error, result_obj.execution_time
        return result_obj.results, "", result_obj.execution_time
