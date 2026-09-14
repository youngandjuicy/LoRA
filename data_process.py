from datasets import load_dataset, DatasetDict


def load_cluener_dataset():

    raw_dataset = load_dataset(
        "json",
        data_files={
            "train": "data/raw/cluener/train.json",
            "dev": "data/raw/cluener/dev.json",
        },
    )

    split_dataset = raw_dataset["train"].train_test_split(
        test_size=0.1,
        seed=42,
        shuffle=True,
    )

    dataset = DatasetDict(
        {
            "train": split_dataset["train"],
            "validation": split_dataset["test"],
            "test": raw_dataset["dev"],
        }
    )

    return dataset

if __name__ == "__main__":

    dataset = load_cluener_dataset()

    print(dataset)