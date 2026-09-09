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
    model_name,
)

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
)

base_model = base_model.to(device)

model = PeftModel.from_pretrained(
    base_model,
    adapter_path,
)

model.eval()


text = "李华昨天前往北京大学参加会议。"
text = "赵敏明天将在阿里巴巴集团参加技术会议。"

instruction = (
    "请从给定文本中抽取人名、时间和组织机构，"
    "并严格按照 JSON 格式输出。"
)

user_content = (
    f"{instruction}\n\n"
    f"文本：{text}"
)

messages = [
    {
        "role": "user",
        "content": user_content,
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)

inputs = {
    key: value.to(device)
    for key, value in inputs.items()
}

with torch.no_grad():

    output_ids = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=False,
    )

prompt_length = inputs["input_ids"].shape[1]

generated_ids = output_ids[0, prompt_length:]

answer = tokenizer.decode(
    generated_ids,
    skip_special_tokens=True,
)

print(answer)