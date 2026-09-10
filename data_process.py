from datasets import load_dataset
from dataset_utils import normalize_cluener_sample

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