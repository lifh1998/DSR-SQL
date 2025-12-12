import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import random
import torch
import numpy as np
import pandas as pd
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer, TrainingArguments
from peft import LoraConfig, TaskType
from datasets import load_dataset, Value, Features
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import Trainer, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from extract_model.qwen_extract_model import Qwen2ForExtractLM
from extract_model.deepseek_extract_model import DeepseekForExtractLM

from prompts import (
    table_extraction_prompt, 
    table_extraction_response, 
    sql_generation_prompt, 
    sql_generation_response,
    sql_refinement_prompt,
    sql_selection_prompt,
    sql_selection_classifier_prompt,
)

agent_type = os.getenv("AGENT_TYPE", "sql_generator")
model_type = os.getenv("MODEL_TYPE", "qwen")
task_type = TaskType.SEQ_CLS if agent_type == "sql_selector_classifier" else TaskType.CAUSAL_LM

lora_r = 128
lora_alpha = 64
lora_dropout = 0.1
output_dir = "./SFT"
num_train_epochs = 3
bf16 = True
overwrite_output_dir = True
per_device_train_batch_size = 1
per_device_eval_batch_size = 8
gradient_accumulation_steps = 16
gradient_checkpointing = True
evaluation_strategy = "no"
save_strategy = "steps"
learning_rate = 5e-5
weight_decay = 0.01
lr_scheduler_type = "cosine"
warmup_ratio = 0.01
max_grad_norm = 0.3
group_by_length = True
auto_find_batch_size = False
eval_steps = 50
save_steps = 50
logging_steps = 50
load_best_model_at_end= False
metric_for_best_model="f1" if agent_type == "sql_selector_classifier" else None
packing = False
save_total_limit=3
neftune_noise_alpha=5
report_to="wandb"
max_seq_length = 8192 #set based on the maximum number of tokens

seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(seed)
os.environ["WANDB_PROJECT"]="msrsql"
os.environ["WANDB_MODE"] = "offline"

# device = "torch.device("cuda" if torch.cuda.is_available() else "cpu")"
device = "auto"
if model_type == "qwen":
    model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
else:
    model_name = "deepseek-ai/deepseek-coder-6.7b-instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
tokenizer.padding_side = "right"
# print(tokenizer)

if agent_type == "table_extracter":
    if model_type == "qwen":
        model = Qwen2ForExtractLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype = torch.bfloat16,
            device_map=device,
        )
    else:
        model = DeepseekForExtractLM.from_pretrained(
            model_name,
            attn_implementation="flash_attention_2",
            torch_dtype = torch.bfloat16,
            device_map=device,
        )
    # ExtractLM setting
    model.set_tokenizer(tokenizer)
    model.set_answer_start_ids(text="<answer>\n<table>")
    model.set_answer_end_ids(text=" </table>\n</answer>\n")
    model.set_split_ids(text=" </table>\n<table>")
    # model.set_perms_limit(6)
elif agent_type == "sql_selector_classifier":
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

model.config.use_cache = False
# print(model)

# 定义新的数据集文件路径
TABLE_SELECTOR_DATASET = "datasets/table_selector_training_data.csv"
UNIFIED_SQL_DATASET = "datasets/unified_sql_training_data.csv"

if agent_type == "table_extracter":
    data_files = {
        "train": TABLE_SELECTOR_DATASET,
    }
    dataset = load_dataset('csv', data_files=data_files, features=Features({
        'question': Value('string'),
        'database_schema': Value('string'),
        'correct_tables': Value('string')
    }))
elif agent_type in ["sql_generator", "sql_refiner"]: # sql_generator 和 sql_refiner 使用统一的数据集
    data_files = {
        "train": UNIFIED_SQL_DATASET,
    }
    # 明确指定所有相关列为字符串类型
    dataset = load_dataset('csv', data_files=data_files, features=Features({
        'question': Value('string'),
        'database_schema': Value('string'),
        'task_type': Value('string'),
        'final_sql': Value('string'),
        'candidate_sql': Value('string'),
        'error_message': Value('string')
    }))
elif agent_type in ["sql_selector", "sql_selector_classifier"]:
    data_files = {
        "train": "datasets/sql_selection_training_data.csv" # 保持不变，如果用户需要修改，再进行
    }
    dataset = load_dataset('csv', data_files=data_files, features=Features({
        'question': Value('string'),
        'database_schema': Value('string'),
        'candidate_sql1': Value('string'),
        'candidate_sql2': Value('string'),
        'label': Value('int32')
    }))

# Apply filtering directly to the training dataset
def _prepare_messages(example):
    question = example['question']
    database_schema = example['database_schema']
    if agent_type == "sql_selector_classifier":
        return {
            "user_message": sql_selection_classifier_prompt.format(
                database_schema=database_schema,
                question=question,
                candidate_sql_1=example['candidate_sql1'],
                candidate_sql_2=example['candidate_sql2']
            ),
            "label": example['label']
        }
    if agent_type == "table_extracter":
        user_message = table_extraction_prompt.format(database_schema=database_schema, question=question)
        assistant_message = table_extraction_response.format(
            correct_tables=" </table>\n<table> ".join(example['correct_tables'].split(", "))
        )
    elif agent_type in ["sql_generator", "sql_refiner"]: # 统一处理 sql_generator 和 sql_refiner
        task_type_example = example['task_type']
        if task_type_example == "sql_generation":
            user_message = sql_generation_prompt.format(database_schema=database_schema, question=question)
            assistant_message = sql_generation_response.format(query=example['final_sql'])
        elif task_type_example == "sql_refinement":
            user_message = sql_refinement_prompt.format(
                database_schema=database_schema,
                question=question,
                candidate_sql=example['candidate_sql'],
                error_message=example['error_message']
            )
            assistant_message = sql_generation_response.format(query=example['final_sql'])
        else:
            raise ValueError(f"Unknown task_type: {task_type_example} for agent_type: {agent_type}")
    elif agent_type == "sql_selector":
        user_message = sql_selection_prompt.format(
            database_schema=database_schema,
            question=question,
            candidate_sql_1=example['candidate_sql1'],
            candidate_sql_2=example['candidate_sql2']
        )
        query = example['candidate_sql1'] if example['label'] == 0 else example['candidate_sql2']
        assistant_message = sql_generation_response.format(query=query)
    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": assistant_message},
    ]
    return messages

def filter_long_sequences(example):
    prepared_data = _prepare_messages(example)
    if agent_type == "sql_selector_classifier":
        return len(tokenizer(prepared_data['user_message']).input_ids) <= max_seq_length
    else:
        # 对于统一的sql_generator/refiner，需要确保example['task_type']存在
        if agent_type in ["sql_generator", "sql_refiner"]:
            if 'task_type' not in example:
                return False # 如果没有task_type，则过滤掉
            # 对于sql_refinement，需要确保candidate_sql和error_message存在
            if example['task_type'] == 'sql_refinement' and (example['candidate_sql'] is None or example['error_message'] is None):
                return False
            # 对于sql_generation，需要确保final_sql存在
            if example['task_type'] == 'sql_generation' and example['final_sql'] is None:
                return False
        return len(tokenizer.apply_chat_template(prepared_data, tokenize=True)) <= max_seq_length
    
# 统一过滤逻辑
if agent_type == "table_extracter":
    dataset["train"] = dataset["train"].filter(lambda example: example['correct_tables'] is not None and example['correct_tables'] != '')
elif agent_type in ["sql_generator", "sql_refiner"]:
    # 统一数据集的过滤逻辑，主要依赖于data_preprocessing.py中已经完成的过滤
    # 这里只需要确保final_sql不为空即可
    dataset["train"] = dataset["train"].filter(lambda example: example['final_sql'] is not None and example['final_sql'] != '')
elif agent_type in ["sql_selector", "sql_selector_classifier"]:
    dataset["train"] = dataset["train"].filter(lambda example: example['candidate_sql1'] is not None and example['candidate_sql1'] != '')
    dataset["train"] = dataset["train"].filter(lambda example: example['candidate_sql2'] is not None and example['candidate_sql2'] != '')
    dataset["train"] = dataset["train"].filter(lambda example: example['label'] is not None)
dataset["train"] = dataset["train"].filter(filter_long_sequences)

peft_config = LoraConfig(
    lora_alpha=lora_alpha,
    lora_dropout=lora_dropout,
    r=lora_r,
    target_modules=[
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        # "lm_head"
    ],
    task_type=task_type,
)

training_arguments = TrainingArguments(
    output_dir=output_dir,
    overwrite_output_dir=overwrite_output_dir,
    num_train_epochs=num_train_epochs,
    load_best_model_at_end=load_best_model_at_end,
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_eval_batch_size,
    evaluation_strategy=evaluation_strategy,
    save_strategy=save_strategy,
    max_grad_norm=max_grad_norm,
    auto_find_batch_size=auto_find_batch_size,
    save_total_limit=save_total_limit,
    gradient_accumulation_steps=gradient_accumulation_steps,
    eval_steps=eval_steps,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    weight_decay=weight_decay,
    bf16=bf16,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    report_to=report_to,
    neftune_noise_alpha=neftune_noise_alpha,
    metric_for_best_model=metric_for_best_model,
)

if agent_type == "sql_selector_classifier":
    from peft import get_peft_model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    class SQLSelectorDataset(Dataset):
        def __init__(self, dataset, tokenizer, max_seq_length):
            self.dataset = dataset
            self.tokenizer = tokenizer
            self.max_seq_length = max_seq_length
            self.encoded_data = []
            for i in tqdm(range(len(self.dataset)), desc="Encoding SQLSelectorDataset"):
                example = self.dataset[i]
                
                # 原始样本
                user_message_original = sql_selection_prompt.format(
                    database_schema=example['database_schema'],
                    question=example['question'],
                    candidate_sql_1=example['candidate_sql1'],
                    candidate_sql_2=example['candidate_sql2']
                )
                label_original = example['label']

                encoded_input_original = self.tokenizer(
                    user_message_original,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                self.encoded_data.append({
                    'input_ids': encoded_input_original['input_ids'].squeeze(),
                    'attention_mask': encoded_input_original['attention_mask'].squeeze(),
                    'labels': torch.tensor(label_original, dtype=torch.long)
                })

                # 交换样本
                user_message_swapped = sql_selection_prompt.format(
                    database_schema=example['database_schema'],
                    question=example['question'],
                    candidate_sql_1=example['candidate_sql2'],
                    candidate_sql_2=example['candidate_sql1']
                )
                label_swapped = 1 - example['label'] # 翻转 label

                encoded_input_swapped = self.tokenizer(
                    user_message_swapped,
                    max_length=self.max_seq_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt"
                )
                self.encoded_data.append({
                    'input_ids': encoded_input_swapped['input_ids'].squeeze(),
                    'attention_mask': encoded_input_swapped['attention_mask'].squeeze(),
                    'labels': torch.tensor(label_swapped, dtype=torch.long)
                })

        def __len__(self):
            return len(self.encoded_data)

        def __getitem__(self, idx):
            return self.encoded_data[idx]

    def compute_metrics(p):
        predictions, labels = p
        predictions = np.argmax(predictions, axis=1)
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average='binary') # For binary classification
        precision = precision_score(labels, predictions, average='binary')
        recall = recall_score(labels, predictions, average='binary')
        return {"accuracy": accuracy, "f1": f1, "precision": precision, "recall": recall}

    train_dataset = SQLSelectorDataset(dataset["train"], tokenizer, max_seq_length)
    # 假设有验证集，如果没有则 eval_dataset 为 None
    eval_dataset = SQLSelectorDataset(dataset["validation"], tokenizer, max_seq_length) if 'validation' in dataset else None
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
else:
    def formatting_prompts_func(training_dataset):
        output_texts = []
        for i in range(len(training_dataset['question'])):
            example = {
                'question': training_dataset['question'][i],
                'database_schema': training_dataset['database_schema'][i],
            }
            if agent_type == "table_extracter":
                example['correct_tables'] = training_dataset['correct_tables'][i]
            elif agent_type in ["sql_generator", "sql_refiner"]:
                example['task_type'] = training_dataset['task_type'][i]
                example['final_sql'] = training_dataset['final_sql'][i]
                if example['task_type'] == 'sql_refinement':
                    example['candidate_sql'] = training_dataset['candidate_sql'][i]
                    example['error_message'] = training_dataset['error_message'][i]
            elif agent_type == "sql_selector":
                example['candidate_sql1'] = training_dataset['candidate_sql1'][i]
                example['candidate_sql2'] = training_dataset['candidate_sql2'][i]
                example['label'] = training_dataset['label'][i]
            
            messages = _prepare_messages(example)
            text = tokenizer.apply_chat_template(messages, tokenize=False)
            output_texts.append(text)

            if agent_type == "sql_selector":
                # 对于sql_selector，需要生成交换后的样本
                swapped_example = {
                    'question': training_dataset['question'][i],
                    'database_schema': training_dataset['database_schema'][i],
                    'candidate_sql1': training_dataset['candidate_sql2'][i],
                    'candidate_sql2': training_dataset['candidate_sql1'][i],
                    'label': 1 - training_dataset['label'][i]
                }
                messages = _prepare_messages(swapped_example)
                text = tokenizer.apply_chat_template(messages, tokenize=False)
                output_texts.append(text)

        return output_texts

    if model_type == "qwen":
        response_template = "<|im_start|>assistant\n"  #qwen
    else:
        response_template = "### Response:" # deepseek
    collator = DataCollatorForCompletionOnlyLM(response_template, tokenizer=tokenizer)

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset['train'],
        eval_dataset=dataset['validation'] if 'validation' in dataset else None,
        peft_config=peft_config,
        formatting_func=formatting_prompts_func,
        data_collator=collator,
        args=training_arguments,
        max_seq_length=max_seq_length,
        packing=packing
    )

trainer.train()

output_dir = os.path.join("./", "final_checkpoint", agent_type, model_type)
trainer.model.save_pretrained(output_dir)
