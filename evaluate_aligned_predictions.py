import json
import os
import argparse

from span_alignment import (
    align_prediction,
)

from evaluate import (
    evaluate_dataset,
    print_evaluation_result,
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--input_path",
    type=str,
    required=True,
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
)

args = parser.parse_args()


# ============================================================
# Load + align
# ============================================================

gold_samples = []

aligned_outputs = []

output_records = []


with open(
    args.input_path,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        record = json.loads(
            line
        )

        source_text = record[
            "text"
        ]

        gold_entities = record[
            "gold_entities"
        ]

        original_raw_output = (
            record["raw_output"]
        )


        aligned_raw_output = (
            align_prediction(
                original_raw_output,
                source_text,
            )
        )


        gold_samples.append(
            {
                "text":
                    source_text,

                "entities":
                    gold_entities,
            }
        )


        aligned_outputs.append(
            aligned_raw_output
        )


        # 同时保存原始输出，
        # 方便之后追查 alignment 做了什么。
        output_records.append(
            {
                "text":
                    source_text,

                "gold_entities":
                    gold_entities,

                "original_raw_output":
                    original_raw_output,

                "raw_output":
                    aligned_raw_output,
            }
        )


# ============================================================
# Save aligned predictions
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
    "Aligned predictions saved to:",
    args.output_path,
)


# ============================================================
# Evaluate
# ============================================================

result = evaluate_dataset(
    gold_samples,
    aligned_outputs,
)

print_evaluation_result(
    result
)