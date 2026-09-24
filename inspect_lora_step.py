import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    LoraConfig,
    get_peft_model,
)

from data_process import (
    load_cluener_dataset,
)

from dataset_utils import (
    normalize_cluener_sample,
)

from preprocess_sft_sample import (
    preprocess_sft_sample,
)

from sft_data_collator import (
    sft_data_collator,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# 1. tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# 2. 构造一个真实 CLUENER batch
# ============================================================

dataset = load_cluener_dataset()

train_dataset = dataset["train"]

indices = [0, 1, 2]

features = []


for idx in indices:

    raw_sample = train_dataset[idx]

    sample = normalize_cluener_sample(
        raw_sample
    )

    feature = preprocess_sft_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_length=512,
    )

    features.append(feature)


batch = sft_data_collator(
    features,
    tokenizer,
)

batch = {
    key: value.to(DEVICE)
    for key, value in batch.items()
}


print("=" * 80)
print("BATCH")
print("=" * 80)

print(
    "input_ids:",
    batch["input_ids"].shape,
)

print(
    "attention_mask:",
    batch["attention_mask"].shape,
)

print(
    "labels:",
    batch["labels"].shape,
)


# ============================================================
# 3. 加载 Base Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(DEVICE)


# ============================================================
# 4. 注入 LoRA
# ============================================================

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "v_proj",
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)


model = get_peft_model(
    model,
    lora_config,
)

model.train()


print("\n" + "=" * 80)
print("TRAINABLE PARAMETERS")
print("=" * 80)

model.print_trainable_parameters()


# ============================================================
# 5. 检查到底哪些参数 trainable
# ============================================================

trainable_names = []

for name, param in model.named_parameters():

    if param.requires_grad:
        trainable_names.append(name)


print(
    "\nNumber of trainable parameter tensors:",
    len(trainable_names),
)

print("\nFirst few trainable parameters:")

for name in trainable_names[:10]:
    print(name)


# ============================================================
# 6. optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    (
        param
        for param in model.parameters()
        if param.requires_grad
    ),
    lr=2e-4,
)


# ============================================================
# 7. 第一次 forward
# ============================================================

outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)


loss = outputs.loss


print("\n" + "=" * 80)
print("FIRST FORWARD")
print("=" * 80)

print(
    "loss:",
    loss.item(),
)


# ============================================================
# 8. 第一次 backward
# ============================================================

optimizer.zero_grad()

loss.backward()


# ============================================================
# 9. 分析 gradient
# ============================================================

def grad_statistics(model):

    stats = {
        "base_grad": 0,
        "lora_A_grad": 0,
        "lora_A_nonzero": 0,
        "lora_B_grad": 0,
        "lora_B_nonzero": 0,
    }

    for name, param in model.named_parameters():

        if "lora_A" in name:

            if param.grad is not None:

                stats["lora_A_grad"] += 1

                if torch.any(
                    param.grad != 0
                ):
                    stats[
                        "lora_A_nonzero"
                    ] += 1

        elif "lora_B" in name:

            if param.grad is not None:

                stats["lora_B_grad"] += 1

                if torch.any(
                    param.grad != 0
                ):
                    stats[
                        "lora_B_nonzero"
                    ] += 1

        else:

            if param.grad is not None:
                stats["base_grad"] += 1

    return stats


stats_1 = grad_statistics(model)


print("\n" + "=" * 80)
print("GRADIENTS AFTER FIRST BACKWARD")
print("=" * 80)

print(stats_1)


# ============================================================
# 10. 更新一次参数
# ============================================================

optimizer.step()


# ============================================================
# 11. 第二次 forward/backward
# ============================================================

optimizer.zero_grad()


outputs = model(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    labels=batch["labels"],
)


loss_2 = outputs.loss

loss_2.backward()


stats_2 = grad_statistics(model)


print("\n" + "=" * 80)
print("SECOND FORWARD")
print("=" * 80)

print(
    "loss:",
    loss_2.item(),
)


print("\n" + "=" * 80)
print("GRADIENTS AFTER SECOND BACKWARD")
print("=" * 80)

print(stats_2)