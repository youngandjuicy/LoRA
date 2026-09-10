from datasets import load_dataset
from dataset_utils import normalize_cluener_sample
from preprocess_sft_sample import build_messages
from transformers import AutoTokenizer

dataset = load_dataset(
    "json",
    data_files={
        "train": "data/raw/cluener/train.json",
        "validation": "data/raw/cluener/dev.json",
    },
)

print(dataset)
print(dataset["train"][0])

sample = dataset["train"][0]

normalized_sample = normalize_cluener_sample(sample)

print("\n===== Raw Sample =====")
print(sample)

print("\n===== Normalized Sample =====")
print(normalized_sample)

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

messages = build_messages(normalized_sample)

print("\n===== Normalized Sample =====")
print(normalized_sample)

print("\n===== Messages =====")
print(messages)

print("\n===== Rendered Training Text =====")
print(
    tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
)

for i in range(5):

    sample = normalize_cluener_sample(
        dataset["train"][i]
    )

    print("\n" + "=" * 80)
    print("TEXT:")
    print(sample["text"])

    print("\nENTITIES:")
    print(sample["entities"])

    print("\nTARGET:")
    messages = build_messages(sample)
    print(messages[-1]["content"])