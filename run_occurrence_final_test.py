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


# ============================================================
# Frozen configuration
# ============================================================

MODEL_NAME = (
    "Qwen/Qwen2.5-0.5B-Instruct"
)

MAX_NEW_TOKENS = 256

EXPECTED_TEST_SAMPLES = 1343


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
    "--seed",
    type=int,
    required=True,
)


args = parser.parse_args()


# ============================================================
# Final-test protocol banner
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL HELD-OUT EVALUATION"
)

print(
    "This split must NOT be used for "
    "model selection or further tuning."
)

print(
    "=" * 80
)

print(
    "training seed:",
    args.seed,
)

print(
    "adapter:",
    args.adapter_path,
)

print(
    "output:",
    args.output_path,
)


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


model = model.to(
    device
)

model.eval()


# ============================================================
# Dataset
# ============================================================

dataset = (
    load_cluener_dataset()
)


# IMPORTANT:
# project test == official CLUENER dev
test_dataset = (
    dataset["test"]
)


print(
    "number of final-test samples:",
    len(test_dataset),
)


# Protection against accidentally
# evaluating the wrong split
if (
    len(test_dataset)
    != EXPECTED_TEST_SAMPLES
):

    raise RuntimeError(
        "Unexpected final-test size.\n"
        f"Expected: {EXPECTED_TEST_SAMPLES}\n"
        f"Actual:   {len(test_dataset)}\n"
        "STOP: check data_process.py "
        "before continuing."
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


    # IMPORTANT:
    # only user prompt;
    # never expose gold assistant answer
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
        key:
            value.to(device)

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


    raw_output = (
        tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )
    )


    return raw_output


# ============================================================
# Final-test inference
# ============================================================

gold_samples = []

reconstructed_outputs = []

output_records = []


for i in tqdm(
    range(
        len(test_dataset)
    ),
    desc="Final Test Inference",
):

    raw_sample = (
        test_dataset[i]
    )


    sample = (
        normalize_cluener_sample(
            raw_sample
        )
    )


    # ----------------------------------------
    # Native S2 prediction
    # ----------------------------------------

    occurrence_raw_output = (
        generate_prediction(
            sample
        )
    )


    # ----------------------------------------
    # occurrence -> standard span
    # ----------------------------------------

    reconstructed_output = (
        reconstruct_occurrence_prediction(
            occurrence_raw_output,
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
            "sample_index":
                i,

            "text":
                sample["text"],

            "gold_entities":
                sample["entities"],

            "occurrence_raw_output":
                occurrence_raw_output,

            "raw_output":
                reconstructed_output,
        }
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
# Final evaluation
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "FINAL TEST RESULT"
)

print(
    f"Training seed: {args.seed}"
)

print(
    "=" * 80
)


result = evaluate_dataset(
    gold_samples,
    reconstructed_outputs,
)


print_evaluation_result(
    result
)