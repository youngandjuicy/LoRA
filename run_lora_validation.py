import os
import json
import argparse

import torch
from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

from data_process import (
    load_cluener_dataset,
)

from dataset_utils import (
    normalize_cluener_sample,
)

from preprocess_sft_sample import (
    build_messages,
)

from evaluate import (
    evaluate_dataset,
    print_evaluation_result,
)


# ============================================================
# Config
# ============================================================

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

MAX_NEW_TOKENS = 256


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--adapter_path",
    type=str,
    required=True,
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
)

parser.add_argument(
    "--num_samples",
    type=int,
    default=None,
)

args = parser.parse_args()


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("device:", device)

print(
    "adapter:",
    args.adapter_path,
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Base Model + LoRA
# ============================================================

base_model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )
)

model = PeftModel.from_pretrained(
    base_model,
    args.adapter_path,
)

model = model.to(device)

model.eval()


# ============================================================
# Dataset
# ============================================================

dataset = load_cluener_dataset()

validation_dataset = dataset[
    "validation"
]


if args.num_samples is None:

    num_samples = len(
        validation_dataset
    )

else:

    num_samples = min(
        args.num_samples,
        len(validation_dataset),
    )


print(
    "number of validation samples:",
    num_samples,
)


# ============================================================
# Generation
# ============================================================

def generate_prediction(
    sample,
):

    messages = build_messages(
        sample
    )

    # 去掉 gold assistant answer
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
            max_new_tokens=MAX_NEW_TOKENS,
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


# ============================================================
# Run validation
# ============================================================

gold_samples = []

raw_outputs = []


for i in tqdm(
    range(num_samples),
    desc="Generating Predictions",
):

    raw_sample = validation_dataset[i]

    sample = normalize_cluener_sample(
        raw_sample
    )

    raw_output = generate_prediction(
        sample
    )

    gold_samples.append(
        sample
    )

    raw_outputs.append(
        raw_output
    )


# ============================================================
# Save predictions
# ============================================================

output_dir = os.path.dirname(
    args.output_path
)

if output_dir:

    os.makedirs(
        output_dir,
        exist_ok=True,
    )


with open(
    args.output_path,
    "w",
    encoding="utf-8",
) as f:

    for sample, raw_output in zip(
        gold_samples,
        raw_outputs,
    ):

        record = {
            "text":
                sample["text"],

            "gold_entities":
                sample["entities"],

            "raw_output":
                raw_output,
        }

        f.write(
            json.dumps(
                record,
                ensure_ascii=False,
            )
            + "\n"
        )


print(
    "\nPredictions saved to:",
    args.output_path,
)


# ============================================================
# Evaluation
# ============================================================

result = evaluate_dataset(
    gold_samples,
    raw_outputs,
)

print_evaluation_result(
    result
)