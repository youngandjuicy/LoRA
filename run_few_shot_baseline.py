import os

import json

import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from data_process import load_cluener_dataset
from dataset_utils import normalize_cluener_sample
from few_shot import build_few_shot_messages
from evaluate import (
    evaluate_dataset,
    print_evaluation_result,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEMO_INDICES = [
    1824,
    4012,
    2286,
]

OUTPUT_PATH = (
    "outputs/"
    "base_validation_predictions_3shot.jsonl"
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

dataset = load_cluener_dataset()

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

demo_samples = [
    normalize_cluener_sample(
        train_dataset[idx]
    )
    for idx in DEMO_INDICES
]

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
    query_sample,
    demo_samples,
):

    messages = build_few_shot_messages(
        query_sample,
        demo_samples,
    )

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
            max_new_tokens=256,
            do_sample=False,
        )

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    generated_ids = output_ids[
        0,
        prompt_length:
    ]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

NUM_SAMPLES = 5

gold_samples = []
raw_outputs = []


for i in tqdm(
    range(NUM_SAMPLES),
    desc="Generating Predictions"
):

    raw_sample = validation_dataset[i]

    query_sample = normalize_cluener_sample(
        raw_sample
    )

    raw_output = generate_prediction(
        model,
        tokenizer,
        query_sample,
        demo_samples,
    )

    gold_samples.append(query_sample)
    raw_outputs.append(raw_output)

    if i < 5:
        print("\n" + "=" * 70)
        print(f"Sample {i}")
        print("=" * 70)

        print("TEXT:")
        print(query_sample["text"])

        print("\nGOLD:")
        print(query_sample["entities"])

        print("\nPREDICTION:")
        print(raw_output)

result = evaluate_dataset(
    gold_samples,
    raw_outputs,
)

print_evaluation_result(result)

os.makedirs(
    os.path.dirname(OUTPUT_PATH),
    exist_ok=True,
)

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