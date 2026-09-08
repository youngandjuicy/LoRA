from datasets import Dataset
from torch.utils.data import DataLoader

from preprocess_sft_sample import preprocess_sft_sample
from sft_data_collator import sft_data_collator


def build_dataset(tokenizer):

    data = {
        "text": [
            "张明于2024年加入浙江大学人工智能研究所。",
            "李华昨天前往北京大学参加会议。",
            "王强将在下周加入腾讯公司。",
        ],
        "entities": [
            [
                {"text": "张明", "type": "person"},
                {"text": "2024年", "type": "time"},
                {
                    "text": "浙江大学人工智能研究所",
                    "type": "organization",
                },
            ],
            [
                {"text": "李华", "type": "person"},
                {"text": "昨天", "type": "time"},
                {"text": "北京大学", "type": "organization"},
            ],
            [
                {"text": "王强", "type": "person"},
                {"text": "下周", "type": "time"},
                {"text": "腾讯公司", "type": "organization"},
            ],
        ],
    }

    dataset = Dataset.from_dict(data)

    processed_dataset = dataset.map(
        lambda sample: preprocess_sft_sample(
            sample,
            tokenizer,
        )
    )

    return processed_dataset


def build_dataloader(
    dataset,
    tokenizer,
    batch_size=2,
    shuffle=True,
):

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda features: sft_data_collator(
            features,
            tokenizer,
        ),
    )