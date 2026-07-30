import json
from typing import Any
from transformers import AutoTokenizer

def build_messages(sample):
    instruction = (
    "请从给定文本中抽取人名、时间和组织机构，"
    "并严格按照 JSON 格式输出。"
    )

    user_content = (
        f"{instruction}\n\n"
        f"文本：{sample['text']}"
    )

    assistant_content = json.dumps(
        {"entities": sample["entities"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )

    messages = [
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": assistant_content,
        },
    ]

    return messages

def preprocess_sft_sample(
    sample: dict[str, Any],
    tokenizer,
    max_length: int = 512,
) -> dict[str, list[int]]:
    messages = build_messages(sample)

    prompt_encoding = tokenizer.apply_chat_template(
        messages[:1],
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
    )

    full_encoding = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        return_dict=True,
    )

    prompt_ids = prompt_encoding["input_ids"]
    full_ids = full_encoding["input_ids"]
    full_attention_mask = full_encoding["attention_mask"]

    # 当前 Qwen 模板满足这个关系，但最好主动检查
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(
            "prompt_ids 不是 full_ids 的前缀，"
            "无法根据 prompt 长度构造 labels。"
        )

    # 截断完整训练序列
    input_ids = full_ids[:max_length]
    attention_mask = full_attention_mask[:max_length]

    prompt_length = min(len(prompt_ids), len(input_ids))

    labels = (
        [-100] * prompt_length
        + input_ids[prompt_length:]
    )

    # 如果截断后答案一个 token 都没留下，这条数据没有训练价值
    if all(label == -100 for label in labels):
        raise ValueError(
            "截断后没有留下任何 assistant 答案 token，"
            "请增大 max_length 或缩短输入。"
        )

    assert len(input_ids) == len(attention_mask) == len(labels)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

if __name__ == "__main__":

    model_name = "Qwen/Qwen2.5-0.5B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    sample = {
    "text": "张明于2024年加入浙江大学人工智能研究所。",
    "entities": [
        {
            "text": "张明",
            "type": "person",
        },
        {
            "text": "2024年",
            "type": "time",
        },
        {
            "text": "浙江大学人工智能研究所",
            "type": "organization",
        },
    ],
    }

    features = preprocess_sft_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_length=512,
    )

    print(features.keys())
    print(len(features["input_ids"]))
    print(len(features["attention_mask"]))
    print(len(features["labels"]))

