import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

adapter_path = (
    "/root/autodl-tmp/projects/LoRA/"
    "outputs/qwen2.5-0.5b-ie-lora"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


def build_inputs(text):

    instruction = (
        "请从给定文本中抽取人名、时间和组织机构，"
        "并严格按照 JSON 格式输出。"
    )

    messages = [
        {
            "role": "user",
            "content": (
                f"{instruction}\n\n"
                f"文本：{text}"
            ),
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    return {
        key: value.to(device)
        for key, value in inputs.items()
    }


def generate(model, text):

    inputs = build_inputs(text)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )

    prompt_length = inputs["input_ids"].shape[1]

    generated_ids = output_ids[
        0,
        prompt_length:
    ]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

base_model = AutoModelForCausalLM.from_pretrained(
    model_name
).to(device)

base_model.eval()

text = "赵敏明天将在阿里巴巴集团参加技术会议。"

base_answer = generate(
    base_model,
    text,
)

print("===== Base Model =====")
print(base_answer)

lora_base_model = AutoModelForCausalLM.from_pretrained(
    model_name
).to(device)

lora_model = PeftModel.from_pretrained(
    lora_base_model,
    adapter_path,
)

lora_model.eval()

lora_answer = generate(
    lora_model,
    text,
)

print("\n===== LoRA Model =====")
print(lora_answer)

test_texts = [
    "赵敏明天将在阿里巴巴集团参加技术会议。",

    "2026年9月，陈浩加入字节跳动人工智能实验室。",

    "清华大学将于下周举办学术论坛，王伟将出席会议。",

    "李雷离开上海后加入腾讯公司。",

    "昨天北京下了一场大雨。",
]

for text in test_texts:

    print("\n" + "=" * 60)
    print("文本：", text)

    print("\nBase Model:")
    print(generate(base_model, text))

    print("\nLoRA Model:")
    print(generate(lora_model, text))