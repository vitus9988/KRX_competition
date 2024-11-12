from datasets import Dataset
import pandas as pd
from transformers import DataCollatorForLanguageModeling,AutoTokenizer, AutoModelForCausalLM
import torch
from trl import SFTTrainer,SFTConfig
from peft import LoraConfig, TaskType, get_peft_model
import os


os.environ["NCCL_P2P_DISABLE"] = "1"
os.environ["NCCL_IB_DISABLE"] = "1"

model_id = "Qwen/Qwen2.5-7B"

tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False, trust_remote_code=True)

df = pd.read_csv("Dataset.csv")
ds = Dataset.from_pandas(df)

def process_func(example):
    MAX_LENGTH = 2048
    input_ids, attention_mask, labels = [], [], []
    instruction = tokenizer(f"""<|im_start|>system
    금융관련 어시스턴트로써 다음 지문과 질문에 대해 잘 생각한 후 한국어로만 답변하세요.
    필요하다고 판단된다면 도출과정을 통해 설득력있는 답변을 출력하세요.<|im_end|>
    <|im_start|>user
    ##{example['query']}
    ##{example['instruction']}<|im_end|>
    <|im_start|>assistant
    """, add_special_tokens=False)

    response = tokenizer(f"{example['output']}", add_special_tokens=False)
    input_ids = instruction["input_ids"] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"] + response["attention_mask"] + [1]
    labels = [-100] * len(instruction["input_ids"]) + response["input_ids"] + [-100]
    
    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]
        
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }

tokenized_ds = ds.map(process_func, remove_columns=ds.column_names)

train_test_split = tokenized_ds.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16
)
model.enable_input_require_grads()

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.01,
    bias="none",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, config)

sft_config = SFTConfig(
    output_dir="./results",
    num_train_epochs=1,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    eval_strategy="steps",
    eval_steps=0.1,
    logging_dir="./logs",
    logging_steps=100,
    warmup_steps=10,
    logging_strategy="steps",
    learning_rate=1e-4,
    bf16=True,
)


trainer = SFTTrainer(
    model=model,
    args=sft_config,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
)

trainer.train()
trainer.model.save_pretrained("./qwen-lora-adaptor")
eval_results = trainer.evaluate()
print(eval_results)
