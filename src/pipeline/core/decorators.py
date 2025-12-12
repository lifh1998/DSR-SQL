import logging
from functools import wraps

def node_decorator(check_schema_status=True):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            logging.info(f"执行节点: {func.__name__}")
            try:
                result = func(*args, **kwargs)
                logging.info(f"节点 {func.__name__} 执行成功")
                return result
            except Exception as e:
                logging.error(f"节点 {func.__name__} 执行失败: {str(e)}")
                raise
        return wrapper
    return decorator