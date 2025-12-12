import threading
import os

class DatabaseManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs): # 接受任意参数，但不处理
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(DatabaseManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, db_schema_dir=None, db_root_dir=None):
        if not hasattr(self, '_initialized'): # 避免重复初始化
            self._initialized = True

            self.db_schema_dir = db_schema_dir
            self.db_root_dir = db_root_dir

    def get_db_path(self, db_id):
        return os.path.join(self.db_root_dir, db_id, f"{db_id}.sqlite")
