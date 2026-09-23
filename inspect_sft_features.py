from transformers import AutoTokenizer
from preprocess_sft_sample import preprocess_sft_sample, build_messages


model_name = "Qwen/Qwen2.5-0.5B-Instruct"


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


sample = {
    "text": "浙商银行企业信贷部叶老桂博士",

    "entities": [
        {
            "text": "浙商银行",
            "type": "company",
            "start": 0,
            "end": 3,
        },
        {
            "text": "叶老桂",
            "type": "name",
            "start": 9,
            "end": 11,
        },
    ],
}


# =========================
# 1. 查看 messages
# =========================

messages = build_messages(sample)


print("=" * 80)
print("MESSAGES")
print("=" * 80)

for message in messages:
    print("\nROLE:")
    print(message["role"])

    print("CONTENT:")
    print(message["content"])



# =========================
# 2. 查看 chat template 后文本
# =========================

rendered_text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=False,
)


print("\n" + "=" * 80)
print("RENDERED CHAT TEMPLATE")
print("=" * 80)

print(rendered_text)



# =========================
# 3. 测试 preprocess
# =========================

features = preprocess_sft_sample(
    sample=sample,
    tokenizer=tokenizer,
    max_length=512,
)


print("\n" + "=" * 80)
print("FEATURES")
print("=" * 80)

print("keys:")
print(features.keys())

print(
    "input length:",
    len(features["input_ids"])
)

print(
    "attention length:",
    len(features["attention_mask"])
)

print(
    "label length:",
    len(features["labels"])
)



# =========================
# 4. 查看 token 和 label
# =========================

input_ids = features["input_ids"]

prompt_ids = []

target_ids = []


for token_id, label in zip(
    input_ids,
    features["labels"]
):

    if label == -100:
        prompt_ids.append(token_id)

    else:
        target_ids.append(token_id)


print("PROMPT:")
print(
    tokenizer.decode(prompt_ids)
)


print("\nTARGET:")
print(
    tokenizer.decode(target_ids)
)