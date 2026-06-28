from datasets import load_dataset
import bitsandbytes, accelerate
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training
import torch

model_name = "Qwen/Qwen2.5-Coder-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)


def get_dataset() :
  dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")
  return dataset

def preprocess(example) :
  instruction = example["problem"]  
  solution   =  example["solution"]
  message = tokenizer.apply_chat_template(
    [{"role":"system" , "content" :"You are a helpful coding assistant that can code asolution for any given problem."},{"role": "user", "content": instruction}],
    tokenize=False,
    add_generation_prompt=True
  )
  prompt_len = len(tokenizer(message,add_special_tokens=False)["input_ids"])
  text = message+solution+tokenizer.eos_token
  tokens = tokenizer(text = text , truncation=True  , max_length=780)
  
  tokens["labels"] = [-100 if i<prompt_len else token_id
        for i, token_id in enumerate(tokens["input_ids"])
    ]
  return tokens

def create_tokenize_dataset(dataset) :
  tokenize_dataset =dataset.map(preprocess,batched=False ,remove_columns=dataset["train"].column_names)
  return tokenize_dataset

def model_quantization() :
  bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,   # extra ~0.4 bits/param saved — meaningful at this VRAM budget
)
  
  model = AutoModelForCausalLM.from_pretrained(model_name , device_map = "auto" , quantization_config=bnb_config)
  model = prepare_model_for_kbit_training(model) 
  lora_config=LoraConfig(task_type=TaskType.CAUSAL_LM , r= 8  , lora_alpha = 16 , target_modules = ["q_proj" , "v_proj"],lora_dropout=0.05 , bias = "none")
  lora_model = get_peft_model(model ,peft_config=lora_config )
  return lora_model


def get_training_args():
    return TrainingArguments(
        output_dir="./qwen-coder-lora",
        per_device_train_batch_size=1,        
        gradient_accumulation_steps=16,        
        num_train_epochs=1,
        learning_rate=2e-4,                    
        bf16=True,                             
        gradient_checkpointing=True,           
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",              
        logging_steps=20,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        report_to="none",                      
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
    )


def build_trainer():
    dataset = get_dataset()
    tokenized_dataset = create_tokenize_dataset(dataset)
    model = model_quantization()
    training_args = get_training_args()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
    )
    return trainer


if __name__ == "__main__":
    trainer = build_trainer()
    trainer.train()