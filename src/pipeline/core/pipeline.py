import logging
import json
import os
from typing import Any, Dict, List, Callable, Optional
from ..nodes.table_extraction import extract_related_table
from ..nodes.sql_generation import candidate_generate
from ..nodes.sql_refinement import refine_candidate
from ..nodes.sql_selection import select_sql
from .task import Task
from ..managers.pipeline_manager import PipelineManager # 导入 PipelineManager
from ..utils.model_utils import model_chose # 导入 model_chose

class Pipeline:
    """管道流主类"""
    def __init__(self, output_base_dir: str = '../outputs', dataset_name: str = 'default_dataset', 
                 use_optimal_table_selection: bool = False, disable_refine_sql: bool = False): # 添加 disable_refine_sql 参数
        self.logger = logging.getLogger(__name__)
        self.output_base_dir = output_base_dir
        self.dataset_name = dataset_name
        self.use_optimal_table_selection = use_optimal_table_selection # 接收新参数
        self.disable_refine_sql = disable_refine_sql # 接收新参数
        self.intermediate_results_dir = os.path.join(output_base_dir, dataset_name, 'intermediate_results')
        os.makedirs(self.intermediate_results_dir, exist_ok=True)
        self.logger.info(f"中间结果将保存到: {self.intermediate_results_dir}")
        self._model_cache: Dict[str, Any] = {} # 用于缓存模型实例
        self.pipeline_manager = PipelineManager() # 初始化 PipelineManager
        
    def _load_model(self, node_name: str) -> Any:
        """加载指定节点的模型"""
        if node_name not in self._model_cache:
            self.logger.info(f"正在加载阶段 '{node_name}' 的模型...")
            model_config = self.pipeline_manager.get_model_config(node_name) # 获取完整的模型配置
            chat_model = model_chose(model_config) # 传入完整的配置字典
            self._model_cache[node_name] = chat_model
            self.logger.info(f"阶段 '{node_name}' 的模型已加载。")
        return self._model_cache[node_name]

    def _unload_model(self, node_name: str):
        """卸载指定节点的模型，并调用其 release 方法清理资源"""
        if node_name in self._model_cache:
            self.logger.info(f"正在卸载阶段 '{node_name}' 的模型...")
            model_instance = self._model_cache[node_name]
            
            if hasattr(model_instance, 'release') and callable(model_instance.release):
                self.logger.info(f"调用阶段 '{node_name}' 模型的 release 方法。")
                model_instance.release()
            else:
                self.logger.warning(f"阶段 '{node_name}' 的模型没有可用的 release 方法。")
            
            del self._model_cache[node_name]
            self.logger.info(f"阶段 '{node_name}' 的模型已卸载。")

    def _save_intermediate_results(self, stage_name: str, results: List[Dict[str, Any]]):
        """保存中间结果到文件"""
        file_path = os.path.join(self.intermediate_results_dir, f"{stage_name}_results.jsonl")
        with open(file_path, 'w', encoding='utf-8') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        self.logger.info(f"阶段 '{stage_name}' 的 {len(results)} 条中间结果已保存到 {file_path}")

    def _load_intermediate_results(self, stage_name: str) -> List[Dict[str, Any]]:
        """从文件加载中间结果"""
        file_path = os.path.join(self.intermediate_results_dir, f"{stage_name}_results.jsonl")
        results = []
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        self.logger.error(f"加载阶段 '{stage_name}' 中间结果时解析错误: {line.strip()} - {e}")
            self.logger.info(f"已从 {file_path} 加载 {len(results)} 条阶段 '{stage_name}' 的中间结果。")
            self.logger.debug(f"阶段 '{stage_name}' 从文件加载的中间结果: {json.dumps(results, ensure_ascii=False, indent=2)}") # 添加详细日志
        return results

    def _process_stage(self, 
                       stage_name: str, 
                       processor_func: Callable, 
                       input_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理管道流中的一个阶段，支持批量处理和断点续传。
        input_data: 上一个阶段的输出，作为当前阶段的输入。
        """
        self.logger.info(f"开始处理阶段: {stage_name}")
        self.logger.debug(f"阶段 '{stage_name}' 接收到的 input_data 长度: {len(input_data)}")
        self.logger.debug(f"阶段 '{stage_name}' 接收到的 input_data: {json.dumps(input_data, ensure_ascii=False, indent=2)}") # 详细打印 input_data
        
        # 尝试加载已完成的中间结果
        completed_results = self._load_intermediate_results(stage_name)
        self.logger.debug(f"阶段 '{stage_name}' 加载的 completed_results 长度: {len(completed_results)}")
        self.logger.debug(f"阶段 '{stage_name}' 加载的 completed_results: {json.dumps(completed_results, ensure_ascii=False, indent=2)}") # 详细打印 completed_results
        
        completed_question_ids = {res.get('question_id') for res in completed_results if 'question_id' in res}
        self.logger.debug(f"阶段 '{stage_name}' 的 completed_question_ids: {completed_question_ids}")
        
        # 过滤掉已完成的任务，只处理未完成的部分
        tasks_to_process = []
        all_tasks = {}
        for item in input_data:
            question_id = item.get('question_id')
            if question_id is not None: # 修改此处，确保0也被正确识别
                updated_task = Task(**item)
                all_tasks[question_id] = updated_task
                if question_id not in completed_question_ids:
                    tasks_to_process.append(updated_task)
                    self.logger.debug(f"阶段 '{stage_name}' 添加到 tasks_to_process: question_id={question_id}")
                else:
                    self.logger.info(f"阶段 '{stage_name}' 跳过 question_id: {question_id}，因为结果已存在。")
            else:
                self.logger.warning(f"阶段 '{stage_name}' 输入项缺少 question_id，跳过: {json.dumps(item, ensure_ascii=False)}")

        self.logger.debug(f"阶段 '{stage_name}' all_tasks 键: {list(all_tasks.keys())}")
        self.logger.info(f"阶段 '{stage_name}' tasks_to_process 长度: {len(tasks_to_process)}")
        self.logger.debug(f"阶段 '{stage_name}' tasks_to_process 内容: {json.dumps([t.to_dict() for t in tasks_to_process], ensure_ascii=False, indent=2)}") # 详细打印 tasks_to_process

        if tasks_to_process:
            self.logger.info(f"阶段 '{stage_name}' 正在处理 {len(tasks_to_process)} 个任务...")
            
            # 加载模型
            chat_model = self._load_model(stage_name)
            
            try:
                # 调用批量处理函数，传入模型实例
                batch_output = processor_func(tasks_to_process, chat_model)
                self.logger.debug(f"阶段 '{stage_name}' 批量处理函数返回 {len(batch_output)} 条结果。")
                self.logger.debug(f"阶段 '{stage_name}' 批量处理函数返回结果: {json.dumps(batch_output, ensure_ascii=False, indent=2)}") # 详细打印 batch_output
            except Exception as e:
                self.logger.error(f"阶段 '{stage_name}' 批量处理失败: {e}")
                raise
            finally:
                # 卸载模型
                self._unload_model(stage_name)

            self.logger.debug(f"阶段 '{stage_name}' 合并前 completed_results 长度: {len(completed_results)}")
            self.logger.debug(f"阶段 '{stage_name}' 合并前 batch_output 长度: {len(batch_output)}")
            # 将当前阶段处理的结果与之前已完成的结果合并
            current_stage_results = completed_results + batch_output
            self._save_intermediate_results(stage_name, current_stage_results)
        else:
            self.logger.info(f"阶段 '{stage_name}' 没有新的任务需要处理，直接使用已加载结果。")
            current_stage_results = completed_results

        self.logger.debug(f"阶段 '{stage_name}' 最终 current_stage_results (合并后) 长度: {len(current_stage_results)}")
        self.logger.debug(f"阶段 '{stage_name}' 最终 current_stage_results (合并后): {json.dumps(current_stage_results, ensure_ascii=False, indent=2)}") # 详细打印 current_stage_results

        # 将 current_stage_results 中的结果与原始任务信息合并，确保所有中间结果都保留
        all_results_for_stage = []
        for res_dict in current_stage_results:
            qid = res_dict.get('question_id')
            if qid is not None: # 确保0也被正确识别
                original_task = all_tasks.get(qid)
                if original_task is None:
                    self.logger.warning(f"没有找到 question_id={qid} 对应的 Task，跳过或其他处理。all_tasks 键: {list(all_tasks.keys())}")
                    continue
                merged_result = original_task.to_dict()
                merged_result.update(res_dict)
                all_results_for_stage.append(merged_result)
            else:
                self.logger.warning(f"result missing question_id, skipping: {json.dumps(res_dict, ensure_ascii=False)}") # 详细打印缺失 question_id 的结果
            
        self.logger.info(f"阶段 '{stage_name}' 处理完成。返回 {len(all_results_for_stage)} 条结果。")
        self.logger.debug(f"阶段 '{stage_name}' 处理完成。返回结果: {json.dumps(all_results_for_stage, ensure_ascii=False, indent=2)}") # 详细打印 all_results_for_stage
        return all_results_for_stage

    def execute_batch(self, tasks: List[Task], save_additional_data: bool = False) -> List[Dict[str, Any]]:
        """
        执行完整的批量管道流。
        每个阶段处理完所有任务后，将结果保存到文件，并作为下一个阶段的输入。
        """
        self.logger.info("开始执行批量SQL生成管道流")
        
        # 初始输入是原始任务列表，转换为字典列表以便统一处理
        current_input_data = [task.to_dict() for task in tasks]

        try:
            # 步骤1: 提取相关表格
            current_input_data = self._process_stage(
                "table_extraction", 
                lambda tasks, chat_model: extract_related_table(tasks, chat_model, self.use_optimal_table_selection), # 传递新参数
                current_input_data
            )
            
            # 步骤2: 生成候选SQL
            current_input_data = self._process_stage(
                "sql_generation", 
                candidate_generate,
                current_input_data
            )
            
            if save_additional_data:
                self.logger.info("SAVE_ADDITIONAL_DATA 为 True，跳过SQL精炼和SQL选择步骤。")
                return current_input_data

            # 步骤3: 精炼SQL
            current_input_data = self._process_stage(
                "sql_refinement", 
                lambda tasks, chat_model: refine_candidate(tasks, chat_model, self.disable_refine_sql), # 传递 disable_refine_sql 参数
                current_input_data
            )

            # 步骤4: 选择最终SQL
            current_input_data = self._process_stage(
                "sql_selection", 
                select_sql,
                current_input_data
            )

            self.logger.info("批量管道流执行完成")
            return current_input_data
            
        except Exception as e:
            self.logger.error(f"批量管道流执行失败: {str(e)}")
            raise
        
    def execute(self, task: Task) -> Dict[str, Any]:
        """
        执行完整的管道流 (单任务模式)。
        此方法现在作为 execute_batch 的包装器，以利用批量处理的优化和断点续传。
        """
        self.logger.info("开始执行SQL生成管道流 (单任务模式)")
        try:
            # 将单个任务包装成列表，调用批量执行方法
            batch_results = self.execute_batch([task])
            
            # 从批量结果中找到当前任务的结果
            for result in batch_results:
                if result.get("question_id") == task.question_id:
                    self.logger.info("管道流执行完成 (单任务模式)")
                    return result
            
            # 如果没有找到结果，可能是处理失败或 question_id 不匹配
            self.logger.error(f"单任务模式执行失败: 未找到 question_id 为 {task.question_id} 的结果。")
            raise ValueError(f"未找到 question_id 为 {task.question_id} 的结果。")
            
        except Exception as e:
            self.logger.error(f"管道流执行失败 (单任务模式): {str(e)}")
            raise
