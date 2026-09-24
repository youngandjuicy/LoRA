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

OUTPUT_ROOT = "checkpoints/s1_lora"

SEED = 42

BATCH_SIZE = 8

NUM_EPOCHS = 3

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


print(
    "device:",
    device,
)


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


print(
    "number of raw training samples:",
    len(train_dataset),
)


# 先将每一条真实 CLUENER 样本预处理成
# input_ids / attention_mask / labels

train_features = []


for i in range(len(train_dataset)):

    raw_sample = train_dataset[i]

    sample = normalize_cluener_sample(
        raw_sample
    )

    feature = preprocess_sft_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
    )

    train_features.append(
        feature
    )


print(
    "number of processed training samples:",
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
    "steps per epoch:",
    len(train_loader),
)

print(
    "total optimizer steps:",
    len(train_loader)
    * NUM_EPOCHS,
)


# ============================================================
# Base Model
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(
    device
)

# 训练时不需要 KV cache
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


print(
    "\n"
    + "=" * 80
)

print(
    "TRAINABLE PARAMETERS"
)

print(
    "=" * 80
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

for epoch in range(
    1,
    NUM_EPOCHS + 1,
):

    model.train()

    epoch_loss_sum = 0.0


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Epoch "
        f"{epoch}"
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


        # RTX 4080 SUPER 支持 BF16
        # Base model 权重仍然保持 FP32，
        # forward 时使用 BF16 autocast。
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=(
                device.type == "cuda"
            ),
        ):

            outputs = model(
                **batch
            )

            loss = outputs.loss


        if not torch.isfinite(loss):

            raise RuntimeError(
                "Non-finite loss "
                f"at epoch={epoch}, "
                f"step={step}: "
                f"{loss.item()}"
            )


        loss.backward()

        optimizer.step()


        loss_value = (
            loss.item()
        )

        epoch_loss_sum += (
            loss_value
        )


        if (
            step == 1
            or step % 50 == 0
            or step == len(train_loader)
        ):

            average_loss = (
                epoch_loss_sum
                / step
            )

            print(
                f"epoch="
                f"{epoch}"
                f" | "
                f"step="
                f"{step:4d}"
                f"/{len(train_loader)}"
                f" | "
                f"loss="
                f"{loss_value:.6f}"
                f" | "
                f"avg_loss="
                f"{average_loss:.6f}"
            )


    # ========================================================
    # End of epoch
    # ========================================================

    epoch_average_loss = (
        epoch_loss_sum
        / len(train_loader)
    )


    print(
        f"\nEpoch {epoch} "
        f"average loss: "
        f"{epoch_average_loss:.6f}"
    )


    # ========================================================
    # Save Adapter
    # ========================================================

    checkpoint_dir = os.path.join(
        OUTPUT_ROOT,
        f"epoch_{epoch}",
    )


    os.makedirs(
        checkpoint_dir,
        exist_ok=True,
    )


    model.save_pretrained(
        checkpoint_dir
    )


    print(
        "Adapter saved to:",
        checkpoint_dir,
    )


print(
    "\n"
    + "=" * 80
)

print(
    "Training finished."
)

print(
    "=" * 80
)