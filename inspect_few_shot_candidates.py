import random

from data_process import load_cluener_dataset
from dataset_utils import normalize_cluener_sample


dataset = load_cluener_dataset()
train_dataset = dataset["train"]

rng = random.Random(42)

candidate_indices = rng.sample(
    range(len(train_dataset)),
    20,
)

for idx in candidate_indices:

    sample = normalize_cluener_sample(
        train_dataset[idx]
    )

    labels = sorted(
        {
            entity["type"]
            for entity in sample["entities"]
        }
    )

    print("\n" + "=" * 80)
    print("INDEX:", idx)
    print("TEXT:")
    print(sample["text"])

    print("LABELS:")
    print(labels)

    print("ENTITIES:")
    print(sample["entities"])