import json

from transformers import AutoTokenizer


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

print(messages)

prompt_text = tokenizer.apply_chat_template(
    messages[:1],
    tokenize=False,
    add_generation_prompt=True,
)

full_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)

print("===== PROMPT TEXT =====")
print(prompt_text)

print("===== FULL TEXT =====")
print(full_text)

prompt_ids = tokenizer.apply_chat_template(
    messages[:1],
    tokenize=True,
    add_generation_prompt=True,
)

full_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=False,
)

prompt_ids = prompt_ids.input_ids
full_ids = full_ids.input_ids

print("===== PROMPT TEXT TOKENS =====")
print(prompt_ids)

print("===== FULL TEXT TOKENS =====")
print(full_ids)

print("prompt token 数：", len(prompt_ids))
print("full token 数：", len(full_ids))

is_prefix = full_ids[: len(prompt_ids)] == prompt_ids
print("prompt_ids 是否为 full_ids 的前缀：", is_prefix)

answer_ids = full_ids[len(prompt_ids):]

print("\n===== DECODED PROMPT =====")
print(
    tokenizer.decode(
        prompt_ids,
        skip_special_tokens=False,
    )
)

print("\n===== DECODED ANSWER =====")
print(
    tokenizer.decode(
        answer_ids,
        skip_special_tokens=False,
    )
)

if not is_prefix:
    raise ValueError(
        "prompt_ids 不是 full_ids 的前缀，"
        "不能直接按 prompt 长度构造 labels。"
    )

input_ids = full_ids.copy()

attention_mask = [1] * len(input_ids)

labels = (
    [-100] * len(prompt_ids)
    + full_ids[len(prompt_ids):]
)

print("\ninput_ids 长度：", len(input_ids))
print("attention_mask 长度：", len(attention_mask))
print("labels 长度：", len(labels))

print("参与 loss 的 token 数：", sum(x != -100 for x in labels))
print("被 mask 的 token 数：", sum(x == -100 for x in labels))

supervised_ids = [
    token_id
    for token_id in labels
    if token_id != -100
]

print("\n===== SUPERVISED TEXT =====")
print(
    tokenizer.decode(
        supervised_ids,
        skip_special_tokens=False,
    )
)
