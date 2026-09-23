import torch

from transformers import AutoTokenizer

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


MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


dataset = load_cluener_dataset()

train_dataset = dataset["train"]


# =========================
# 1. 取三个真实样本
# =========================

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


# =========================
# 2. 查看 padding 前长度
# =========================

print("=" * 80)
print("BEFORE COLLATION")
print("=" * 80)


for i, feature in enumerate(features):

    print(
        f"sample {i}: "
        f"length={len(feature['input_ids'])}"
    )


# =========================
# 3. collate
# =========================

batch = sft_data_collator(
    features,
    tokenizer,
)


print("\n" + "=" * 80)
print("BATCH")
print("=" * 80)

print(
    "input_ids shape:",
    batch["input_ids"].shape
)

print(
    "attention_mask shape:",
    batch["attention_mask"].shape
)

print(
    "labels shape:",
    batch["labels"].shape
)


# =========================
# 4. 检查每条样本
# =========================

max_length = batch["input_ids"].shape[1]


for i, feature in enumerate(features):

    original_length = len(
        feature["input_ids"]
    )

    padding_length = (
        max_length - original_length
    )

    print(
        f"\nsample {i}"
    )

    print(
        "original length:",
        original_length
    )

    print(
        "padding length:",
        padding_length
    )

    print(
        "attention mask tail:",
        batch["attention_mask"][
            i,
            original_length:
        ].tolist()
    )

    print(
        "labels tail:",
        batch["labels"][
            i,
            original_length:
        ].tolist()
    )


# =========================
# 5. 自动 sanity check
# =========================

for i, feature in enumerate(features):

    original_length = len(
        feature["input_ids"]
    )

    # 原始部分不能被 collator 改掉
    assert torch.equal(
        batch["input_ids"][
            i,
            :original_length
        ],
        torch.tensor(
            feature["input_ids"]
        ),
    )

    assert torch.equal(
        batch["attention_mask"][
            i,
            :original_length
        ],
        torch.tensor(
            feature["attention_mask"]
        ),
    )

    assert torch.equal(
        batch["labels"][
            i,
            :original_length
        ],
        torch.tensor(
            feature["labels"]
        ),
    )

    # padding 部分
    if original_length < max_length:

        assert torch.all(
            batch["input_ids"][
                i,
                original_length:
            ]
            == tokenizer.pad_token_id
        )

        assert torch.all(
            batch["attention_mask"][
                i,
                original_length:
            ]
            == 0
        )

        assert torch.all(
            batch["labels"][
                i,
                original_length:
            ]
            == -100
        )


print(
    "\nAll batch sanity checks passed."
)