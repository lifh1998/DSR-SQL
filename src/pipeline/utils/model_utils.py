import requests, time
import torch
import json
import re
from typing import Any, Dict

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from peft import PeftModel

def model_chose(config: Dict[str, Any]) -> Any:
    """
    根据配置选择并加载模型。
    config 字典应包含 'model_name', 'lora_path', 'device', 'temperature', 'top_p', 'n', 'single' 等参数。
    """
    model_type = config.get("model_type", "causal") # 默认为因果模型

    if model_type == "causal":
        return CausalModel(config)
    elif model_type == "classification":
        # 如果有分类模型，可以在这里实现
        # return ClassificationModel(config)
        raise NotImplementedError("分类模型尚未实现。")
    else:
        raise ValueError(f"未知模型类型: {model_type}")

class HFModel:
    """Hugging Face 模型基类"""
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer"],
            trust_remote_code=True,
            padding_side="right",
            use_fast=True
        )
        self.model = None # 模型实例将在子类中加载

    def release(self):
        """释放模型资源，包括清理GPU内存"""
        if self.model is not None:
            self.model.to('cpu') # 将模型移动到CPU
            del self.model # 删除模型实例
            self.model = None # 清空引用
            if torch.cuda.is_available():
                torch.cuda.empty_cache() # 清理CUDA缓存
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

class CausalModel(HFModel):
    """因果语言模型类"""
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.target_device = config["device"]
        
        # 动态加载基模型
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name"],
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.bfloat16,
            device_map=self.target_device, # 直接加载到目标设备
        ).eval()

        # 如果提供了 LoRA 路径，则合并 LoRA 模型
        lora_path = config.get("lora_path")
        if lora_path and lora_path.strip(): # 确保 lora_path 不为空字符串或 None
            model = PeftModel.from_pretrained(
                model, 
                lora_path, 
                torch_dtype = torch.bfloat16
            )
            model = model.merge_and_unload()
        
        self.model = model

    def get_ans(self, content: Any) -> Any:
        """
        获取模型答案。
        temperature, top_p, n, single 等参数从 config 中获取。
        """
        temperature = self.config.get("temperature", 0.0)
        top_p = self.config.get("top_p")
        n = self.config.get("n", 1)
        single = self.config.get("single", True)

        if isinstance(content, str):
            messages = [{
                "role": "user", 
                "content": content
            }]
        else:
            messages = content

        inputs = self.tokenizer.apply_chat_template(
            messages, 
            return_tensors="pt", 
            add_generation_prompt=True, 
            tokenize=True
        ).to(self.target_device)
        
        generate_kwargs = {
            "max_new_tokens": 2048,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.pad_token_id,
            "num_return_sequences": n if not single else 1
        }
        
        if temperature > 0:
            generate_kwargs.update({
                "do_sample": True,
                "temperature": temperature
            })
            if top_p is not None:
                generate_kwargs["top_p"] = top_p
        else:
            generate_kwargs["do_sample"] = False
        
        output_tokens = self.model.generate(inputs, **generate_kwargs)

        if single or n == 1:
            return self.tokenizer.decode(
                output_tokens[0][len(inputs[0]):], 
                skip_special_tokens=True
            ).strip()
        else:
            results = []
            for i in range(n):
                decoded = self.tokenizer.decode(
                    output_tokens[i][len(inputs[0]):], 
                    skip_special_tokens=True
                ).strip()
                results.append(decoded)
            return results

# class ClassificationModel(HFModel):
#     """分类模型类 (待实现)"""
#     def __init__(self, config: Dict[str, Any]):
#         super().__init__(config)
#         self.target_device = config["device"]

#         model = AutoModelForSequenceClassification.from_pretrained(
#             config["model_name"], 
#             num_labels=config.get("num_labels", 2), # 默认2个标签
#             torch_dtype=torch.bfloat16,
#             device_map=self.target_device,
#         ).eval()

#         lora_path = config.get("lora_path")
#         if lora_path:
#             model = PeftModel.from_pretrained(
#                 model, 
#                 lora_path,
#                 torch_dtype = torch.bfloat16
#             )
#             model = model.merge_and_unload()
        
#         self.model = model

#     def get_ans(self, content: Any) -> Any:
#         inputs = self.tokenizer(
#             content,
#             return_tensors="pt",
#             padding=True,
#             truncation=True
#         ).to(self.target_device)

#         with torch.no_grad():
#             outputs = self.model(**inputs)
        
#         logits = outputs.logits
#         probabilities = torch.softmax(logits, dim=-1)
#         predicted_class_id = torch.argmax(probabilities, dim=-1).item()
#         return predicted_class_id
