from collections import Counter

from data_process import load_cluener_dataset
from dataset_utils import normalize_cluener_sample


dataset = load_cluener_dataset()


def get_label_statistics(dataset_split):

    counter = Counter()

    total_entities = 0

    for raw_sample in dataset_split:

        sample = normalize_cluener_sample(
            raw_sample
        )

        for entity in sample["entities"]:

            counter[entity["type"]] += 1
            total_entities += 1

    return counter, total_entities


all_labels = [
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene",
]


for split_name in [
    "train",
    "validation",
    "test",
]:

    counter, total_entities = get_label_statistics(
        dataset[split_name]
    )

    print("\n" + "=" * 70)
    print(split_name)
    print("=" * 70)

    print(
        "samples:",
        len(dataset[split_name]),
    )

    print(
        "entities:",
        total_entities,
    )

    for label in all_labels:

        count = counter[label]

        ratio = (
            count / total_entities
            if total_entities > 0
            else 0
        )

        print(
            f"{label:15s}"
            f"{count:6d}"
            f"  {ratio:.2%}"
        )