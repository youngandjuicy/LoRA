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


# =========================
# Tokenizer
# =========================

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


# =========================
# Dataset / DataLoader
# =========================

train_dataset = build_dataset(
    tokenizer
)

train_dataloader = build_dataloader(
    train_dataset,
    tokenizer,
    batch_size=2,
    shuffle=True,
)


# =========================
# Base Model
# =========================

base_model = AutoModelForCausalLM.from_pretrained(
    model_name
)


# =========================
# LoRA
# =========================

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


# =========================
# Device
# =========================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

model = model.to(device)


# =========================
# Optimizer
# =========================

optimizer = torch.optim.AdamW(
    (
        param
        for param in model.parameters()
        if param.requires_grad
    ),
    lr=1e-3,
    weight_decay=0.0,
)


# =========================
# Training
# =========================

num_epochs = 20

model.train()

for epoch in range(num_epochs):

    total_loss = 0.0
    num_steps = 0

    for batch in train_dataloader:

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad()

        outputs = model(**batch)

        loss = outputs.loss

        loss.backward()

        optimizer.step()

        total_loss += loss.item()
        num_steps += 1

    avg_loss = total_loss / num_steps

    print(
        f"Epoch {epoch + 1:02d} "
        f"| loss = {avg_loss:.4f}"
    )

output_dir = (
    "/root/autodl-tmp/projects/LoRA/"
    "outputs/qwen2.5-0.5b-ie-lora"
)

model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"LoRA adapter saved to: {output_dir}")