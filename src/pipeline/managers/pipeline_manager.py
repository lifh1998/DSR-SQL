import threading
from typing import Any, Dict, List, Optional

class PipelineManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(PipelineManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, configs: Optional[Dict[str, Any]] = None):
        if not hasattr(self, '_initialized'):
            self._initialized = True

            if configs is None:
                # 定义默认的模型名称
                default_model_name = "Qwen/Qwen2.5-Coder-7B-Instruct"
                
                self.configs = {
                    "table_extraction": { # 对应 extract_related_table 节点
                        "model_name": default_model_name,
                        "lora_path": "./final_checkpoint/7b/table_extracter/qwen",
                        "device": "cuda:1",
                        "temperature": 0,
                        "top_p": None,
                        "n": 1,
                        "single": True,
                        "model_type": "causal"
                    },
                    "sql_generation": { # 对应 candidate_generate 节点
                        "model_name": default_model_name,
                        "lora_path": "./final_checkpoint/7b/sql_generator/qwen",
                        "device": "cuda:1",
                        "temperature": 0,
                        "top_p": None,
                        "n": 1,
                        "single": True,
                        "model_type": "causal"
                    },
                    "sql_refinement": { # 对应 refine_candidate 节点
                        "model_name": default_model_name,
                        "lora_path": "./final_checkpoint/7b/sql_refiner/qwen",
                        "device": "cuda:1",
                        "temperature": 0,
                        "top_p": None,
                        "n": 1,
                        "single": True,
                        "model_type": "causal"
                    },
                    "sql_selection": { # 对应 select_sql 节点，实际使用的是 merge_sql 模型
                        "model_name": "cycloneboy/CscSQL-Merge-Qwen2.5-Coder-7B-Instruct",
                        "lora_path": None, # sql_merger 没有 LoRA
                        "device": "cuda:1",
                        "temperature": 0,
                        "top_p": None,
                        "n": 1,
                        "single": True,
                        "model_type": "causal"
                    },
                    # 如果有分类器，可以取消注释并配置
                    # "select_sql_classifier": {
                    #     "model_name": default_model_name,
                    #     "lora_path": "./final_checkpoint/7b/sql_selector_classifier/qwen",
                    #     "device": "cuda:1",
                    #     "temperature": 0,
                    #     "top_p": None,
                    #     "n": 1,
                    #     "single": True,
                    #     "model_type": "classification",
                    #     "num_labels": 2
                    # },
                }
            else:
                self.configs = configs
    
    def get_model_config(self, node_name: str) -> Dict[str, Any]:
        """获取指定节点的完整模型配置"""
        config = self.configs.get(node_name)
        if not config:
            raise ValueError(f"未找到节点 '{node_name}' 的模型配置。")
        return config

    def get_last_node_result(self, execution_history: List[Dict[str, Any]], node_name: str) -> Dict[str, Any]:
        """从执行历史中获取指定节点的最后结果"""
        for record in reversed(execution_history):
            if record.get("node_name") == node_name:
                return record.get("result", {})
        return {}
