import json

from evaluate import (
    evaluate_dataset,
    print_evaluation_result,
)


gold_samples = []
raw_outputs = []


with open(
    "outputs/base_validation_predictions.jsonl",
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        record = json.loads(line)

        gold_samples.append(
            {
                "text": record["text"],
                "entities": record[
                    "gold_entities"
                ],
            }
        )

        raw_outputs.append(
            record["raw_output"]
        )


result = evaluate_dataset(
    gold_samples,
    raw_outputs,
)

print_evaluation_result(result)