import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)

from data_process import (
    build_dataset,
    build_dataloader,
)


model_name = "Qwen/Qwen2.5-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


dataset = build_dataset(
    tokenizer
)


train_dataloader = build_dataloader(
    dataset,
    tokenizer,
    batch_size=2,
)


base_model = AutoModelForCausalLM.from_pretrained(
    model_name
)


lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)


model = get_peft_model(
    base_model,
    lora_config,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)