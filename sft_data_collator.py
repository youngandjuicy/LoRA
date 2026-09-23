import torch


def sft_data_collator(features, tokenizer):

    if tokenizer.pad_token_id is None:
        raise ValueError(
            "tokenizer.pad_token_id is None"
        )

    max_length = max(
        len(feature["input_ids"])
        for feature in features
    )

    batch_input_ids = []
    batch_attention_mask = []
    batch_labels = []

    for feature in features:

        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        labels = feature["labels"]

        padding_length = (
            max_length - len(input_ids)
        )

        padded_input_ids = (
            input_ids
            + [tokenizer.pad_token_id]
            * padding_length
        )

        padded_attention_mask = (
            attention_mask
            + [0] * padding_length
        )

        padded_labels = (
            labels
            + [-100] * padding_length
        )

        batch_input_ids.append(
            padded_input_ids
        )

        batch_attention_mask.append(
            padded_attention_mask
        )

        batch_labels.append(
            padded_labels
        )

    return {
        "input_ids":
            torch.tensor(batch_input_ids),

        "attention_mask":
            torch.tensor(batch_attention_mask),

        "labels":
            torch.tensor(batch_labels),
    }