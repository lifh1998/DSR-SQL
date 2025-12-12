from typing import Optional, Dict, Any

class Task:
    """任务对象"""
    def __init__(
        self, 
        question_id: int, 
        db_id: str, 
        question: str, 
        database_schema: str, 
        query: str, 
        correct_tables: str,
        # 新增中间结果属性，并设置默认值
        related_tables: Optional[str] = None,
        scaled_down_db_schema: Optional[str] = None,
        candidate_sql_1: Optional[str] = None,
        candidate_sql_2: Optional[str] = None,
        refined_sql_1: Optional[str] = None,
        refined_sql_2: Optional[str] = None,
        selected_sql: Optional[str] = None,
        **kwargs # 允许接收额外参数，以防未来扩展
    ):
        self.question_id = question_id
        self.db_id = db_id
        self.question = question
        self.database_schema = database_schema
        self.query = query
        self.correct_tables = correct_tables
        
        # 初始化中间结果属性
        self.related_tables = related_tables
        self.scaled_down_db_schema = scaled_down_db_schema
        self.candidate_sql_1 = candidate_sql_1
        self.candidate_sql_2 = candidate_sql_2
        self.refined_sql_1 = refined_sql_1
        self.refined_sql_2 = refined_sql_2
        self.selected_sql = selected_sql

        # 处理任何额外的 kwargs，以支持从字典创建Task对象时包含所有字段
        for k, v in kwargs.items():
            setattr(self, k, v)
        
    def __repr__(self):
        return f"Task(question_id={self.question_id}, db_id='{self.db_id}', question='{self.question[:50]}...')"

    def to_dict(self) -> Dict[str, Any]:
        # 动态获取所有属性，包括新添加的中间结果属性
        return {attr: getattr(self, attr) for attr in self.__dict__ if not attr.startswith('_')}
