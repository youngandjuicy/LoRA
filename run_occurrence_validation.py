import os
import json
import argparse

import torch

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import (
    PeftModel,
)

from data_process import (
    load_cluener_dataset,
)

from dataset_utils import (
    normalize_cluener_sample,
)

from preprocess_occurrence_sft_sample import (
    build_occurrence_messages,
)

from occurrence_reconstruction import (
    reconstruct_occurrence_prediction,
)

from evaluate import (
    evaluate_dataset,
    print_evaluation_result,
)


MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)

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


print(
    "device:",
    device,
)

print(
    "adapter:",
    args.adapter_path,
)


# ============================================================
# Model
# ============================================================

tokenizer = (
    AutoTokenizer.from_pretrained(
        MODEL_NAME
    )
)


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

validation_dataset = (
    dataset["validation"]
)


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

    messages = (
        build_occurrence_messages(
            sample
        )
    )


    # inference 时去掉 gold assistant
    prompt_messages = (
        messages[:1]
    )


    inputs = (
        tokenizer.apply_chat_template(
            prompt_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )
    )


    inputs = {
        key: value.to(device)
        for key, value
        in inputs.items()
    }


    with torch.no_grad():

        output_ids = (
            model.generate(
                **inputs,
                max_new_tokens=
                    MAX_NEW_TOKENS,
                do_sample=False,
            )
        )


    prompt_length = (
        inputs[
            "input_ids"
        ].shape[1]
    )


    generated_ids = (
        output_ids[
            0,
            prompt_length:
        ]
    )


    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )


# ============================================================
# Validation
# ============================================================

gold_samples = []

reconstructed_outputs = []

output_records = []


for i in tqdm(
    range(num_samples),
    desc="Generating Predictions",
):

    raw_sample = (
        validation_dataset[i]
    )


    sample = (
        normalize_cluener_sample(
            raw_sample
        )
    )


    # 模型原始 occurrence 输出
    occurrence_output = (
        generate_prediction(
            sample
        )
    )


    # occurrence -> absolute span
    reconstructed_output = (
        reconstruct_occurrence_prediction(
            occurrence_output,
            sample["text"],
        )
    )


    gold_samples.append(
        sample
    )


    reconstructed_outputs.append(
        reconstructed_output
    )


    output_records.append(
        {
            "text":
                sample["text"],

            "gold_entities":
                sample[
                    "entities"
                ],

            "occurrence_raw_output":
                occurrence_output,

            "raw_output":
                reconstructed_output,
        }
    )


# ============================================================
# Save
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

    for record in output_records:

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
# Standard span evaluation
# ============================================================

result = evaluate_dataset(
    gold_samples,
    reconstructed_outputs,
)


print_evaluation_result(
    result
)