import os
import random

import torch

from torch.utils.data import DataLoader

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


# ============================================================
# Config
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_DIR = (
    "checkpoints/"
    "s1_lora_smoke"
)

SEED = 42

NUM_TRAIN_SAMPLES = 1000

BATCH_SIZE = 8

NUM_EPOCHS = 1

LEARNING_RATE = 2e-4

MAX_LENGTH = 512


# ============================================================
# Seed
# ============================================================

random.seed(SEED)

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("device:", device)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Dataset
# ============================================================

dataset = load_cluener_dataset()

train_dataset = dataset["train"]


train_features = []


for i in range(NUM_TRAIN_SAMPLES):

    sample = normalize_cluener_sample(
        train_dataset[i]
    )

    feature = preprocess_sft_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_features.append(feature)


print(
    "number of training samples:",
    len(train_features),
)


# ============================================================
# DataLoader
# ============================================================

generator = torch.Generator()

generator.manual_seed(SEED)


train_loader = DataLoader(
    train_features,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    generator=generator,
    collate_fn=lambda features:
        sft_data_collator(
            features,
            tokenizer,
        ),
)


print(
    "number of training steps:",
    len(train_loader),
)


# ============================================================
# Base Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

model.config.use_cache = False


# ============================================================
# LoRA
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

model.print_trainable_parameters()

model.train()


# ============================================================
# Optimizer
# ============================================================

optimizer = torch.optim.AdamW(
    (
        param
        for param in model.parameters()
        if param.requires_grad
    ),
    lr=LEARNING_RATE,
    weight_decay=0.0,
)


# ============================================================
# Training
# ============================================================

for epoch in range(NUM_EPOCHS):

    running_loss = 0.0

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Epoch {epoch + 1}"
        f"/{NUM_EPOCHS}"
    )

    print(
        "=" * 80
    )

    for step, batch in enumerate(
        train_loader,
        start=1,
    ):

        batch = {
            key: value.to(device)
            for key, value in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        outputs = model(
            **batch
        )

        loss = outputs.loss

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss: "
                f"{loss.item()}"
            )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
        )

        if (
            step == 1
            or step % 20 == 0
            or step == len(train_loader)
        ):

            average_loss = (
                running_loss / step
            )

            print(
                f"step "
                f"{step:4d}"
                f"/{len(train_loader)}"
                f" | "
                f"loss="
                f"{loss.item():.6f}"
                f" | "
                f"avg_loss="
                f"{average_loss:.6f}"
            )


# ============================================================
# Save Adapter
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True,
)


model.save_pretrained(
    OUTPUT_DIR
)


print(
    "\nAdapter saved to:",
    OUTPUT_DIR,
)