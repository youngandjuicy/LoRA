import json

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from data_process import load_cluener_dataset
from dataset_utils import normalize_cluener_sample
from preprocess_sft_sample import build_messages
from evaluate import (
    evaluate_dataset,
    print_evaluation_result,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

OUTPUT_PATH = (
    "outputs/base_validation_predictions.jsonl"
)

NUM_SAMPLES = 5


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME
)

model = model.to(device)

model.eval()

# 该函数接收一条样本，生成一条输出
def generate_prediction(
    model,
    tokenizer,
    sample,
):

    messages = build_messages(sample)

    # 推理阶段不能把 assistant 标准答案给模型
    prompt_messages = messages[:1]

    inputs = tokenizer.apply_chat_template(
        prompt_messages,
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
            max_new_tokens=256, # train target max = 204 tokens, validation target max = 178 tokens
            do_sample=False,
        )

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    generated_ids = output_ids[
        0,
        prompt_length:
    ]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return raw_output

dataset = load_cluener_dataset()

validation_dataset = dataset["validation"]


gold_samples = []
raw_outputs = []


for i in range(NUM_SAMPLES):

    raw_sample = validation_dataset[i]

    sample = normalize_cluener_sample(
        raw_sample
    )

    raw_output = generate_prediction(
        model,
        tokenizer,
        sample,
    )

    gold_samples.append(sample)
    raw_outputs.append(raw_output)

    print("\n" + "=" * 70)
    print(f"Sample {i}")
    print("=" * 70)

    print("TEXT:")
    print(sample["text"])

    print("\nGOLD:")
    print(sample["entities"])

    print("\nPREDICTION:")
    print(raw_output)

result = evaluate_dataset(
    gold_samples,
    raw_outputs,
)

print_evaluation_result(result)

with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as f:

    for sample, raw_output in zip(
        gold_samples,
        raw_outputs,
    ):

        record = {
            "text": sample["text"],
            "gold_entities": sample["entities"],
            "raw_output": raw_output,
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )