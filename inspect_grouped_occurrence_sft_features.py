from transformers import (
    AutoTokenizer,
)

from data_process import (
    load_cluener_dataset,
)

from dataset_utils import (
    normalize_cluener_sample,
)

from preprocess_grouped_occurrence_sft_sample import (
    build_grouped_occurrence_messages,
    build_grouped_occurrence_target_content,
    preprocess_grouped_occurrence_sft_sample,
    get_occurrence_index,
)


MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)


tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
)


dataset = load_cluener_dataset()

train_dataset = dataset["train"]


# ============================================================
# 找一个 occurrence > 1 的真实样本
# ============================================================

selected_sample = None
selected_index = None


for i in range(
    len(train_dataset)
):

    sample = normalize_cluener_sample(
        train_dataset[i]
    )

    found = False

    for entity in sample["entities"]:

        occurrence = (
            get_occurrence_index(
                sample["text"],
                entity,
            )
        )

        if occurrence > 1:

            selected_sample = sample
            selected_index = i

            found = True
            break

    if found:
        break


if selected_sample is None:

    raise RuntimeError(
        "No repeated occurrence found."
    )


print(
    "train index:",
    selected_index,
)


print(
    "\nTEXT:"
)

print(
    selected_sample["text"]
)


print(
    "\nORIGINAL GOLD:"
)

print(
    selected_sample["entities"]
)


print(
    "\nGROUPED OCCURRENCE TARGET:"
)

print(
    build_grouped_occurrence_target_content(
        selected_sample
    )
)


print(
    "\nMESSAGES:"
)

for message in (
    build_grouped_occurrence_messages(
        selected_sample
    )
):

    print(
        message
    )


# ============================================================
# preprocessing
# ============================================================

feature = (
    preprocess_grouped_occurrence_sft_sample(
        selected_sample,
        tokenizer,
        max_length=512,
    )
)


print(
    "\nfeature length:",
    len(feature["input_ids"]),
)


# ============================================================
# 分离 prompt / target
# ============================================================

first_supervised_index = (
    next(
        i
        for i, label
        in enumerate(
            feature["labels"]
        )
        if label != -100
    )
)


prompt_ids = (
    feature["input_ids"][
        :first_supervised_index
    ]
)

target_ids = (
    feature["input_ids"][
        first_supervised_index:
    ]
)


print(
    "\nPROMPT:"
)

print(
    tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
    )
)


print(
    "\nTARGET:"
)

print(
    tokenizer.decode(
        target_ids,
        skip_special_tokens=False,
    )
)