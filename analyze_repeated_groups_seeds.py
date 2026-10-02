import json
from collections import defaultdict

import numpy as np

from span_alignment import (
    find_all_occurrences,
)


EXPERIMENTS = {
    42: {
        "s2": (
            "outputs/"
            "s2_epoch3_validation_predictions.jsonl"
        ),
        "s3": (
            "outputs/"
            "s3_epoch3_validation_predictions.jsonl"
        ),
    },

    43: {
        "s2": (
            "outputs/"
            "s2_seed43_epoch3_validation_predictions.jsonl"
        ),
        "s3": (
            "outputs/"
            "s3_seed43_epoch3_validation_predictions.jsonl"
        ),
    },

    44: {
        "s2": (
            "outputs/"
            "s2_seed44_epoch3_validation_predictions.jsonl"
        ),
        "s3": (
            "outputs/"
            "s3_seed44_epoch2_validation_predictions.jsonl"
        ),
    },
}


def load_jsonl(path):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        return [
            json.loads(line)
            for line in f
        ]


def get_gold_occurrence(
    source_text,
    entity,
):

    occurrences = find_all_occurrences(
        source_text,
        entity["text"],
    )

    span = (
        entity["start"],
        entity["end"],
    )

    return (
        occurrences.index(span)
        + 1
    )


def parse_s2(raw_output):

    try:
        data = json.loads(
            raw_output
        )
    except json.JSONDecodeError:
        return []

    if not isinstance(data, dict):
        return []

    entities = data.get(
        "entities"
    )

    if not isinstance(
        entities,
        list,
    ):
        return []

    result = []

    for entity in entities:

        if not isinstance(
            entity,
            dict,
        ):
            continue

        if not all(
            key in entity
            for key in (
                "text",
                "type",
                "occurrence",
            )
        ):
            continue

        if (
            not isinstance(
                entity["text"],
                str,
            )
            or not isinstance(
                entity["type"],
                str,
            )
            or type(
                entity["occurrence"]
            ) is not int
        ):
            continue

        result.append(
            (
                entity["text"],
                entity["type"],
                entity["occurrence"],
            )
        )

    return result


def parse_s3(raw_output):

    try:
        data = json.loads(
            raw_output
        )
    except json.JSONDecodeError:
        return []

    if not isinstance(data, dict):
        return []

    entities = data.get(
        "entities"
    )

    if not isinstance(
        entities,
        list,
    ):
        return []

    result = []

    for entity in entities:

        if not isinstance(
            entity,
            dict,
        ):
            continue

        if not all(
            key in entity
            for key in (
                "text",
                "type",
                "occurrences",
            )
        ):
            continue

        if (
            not isinstance(
                entity["text"],
                str,
            )
            or not isinstance(
                entity["type"],
                str,
            )
            or not isinstance(
                entity["occurrences"],
                list,
            )
        ):
            continue

        for occurrence in (
            entity["occurrences"]
        ):

            if type(occurrence) is int:

                result.append(
                    (
                        entity["text"],
                        entity["type"],
                        occurrence,
                    )
                )

    return result


def evaluate_all_groups(
    records,
    formulation,
):

    gold_mentions = 0
    pred_mentions = 0
    tp = 0

    exact_groups = 0
    total_groups = 0


    for record in records:

        text = record[
            "text"
        ]

        gold_groups = (
            defaultdict(set)
        )


        for entity in (
            record[
                "gold_entities"
            ]
        ):

            surface = entity[
                "text"
            ]

            source_occurrences = (
                find_all_occurrences(
                    text,
                    surface,
                )
            )

            if len(
                source_occurrences
            ) <= 1:
                continue

            key = (
                surface,
                entity["type"],
            )

            gold_groups[
                key
            ].add(
                get_gold_occurrence(
                    text,
                    entity,
                )
            )


        if formulation == "s2":

            predictions = parse_s2(
                record[
                    "occurrence_raw_output"
                ]
            )

        else:

            predictions = parse_s3(
                record[
                    "grouped_raw_output"
                ]
            )


        pred_groups = (
            defaultdict(set)
        )


        for (
            surface,
            entity_type,
            occurrence,
        ) in predictions:

            if len(
                find_all_occurrences(
                    text,
                    surface,
                )
            ) <= 1:
                continue

            pred_groups[
                (
                    surface,
                    entity_type,
                )
            ].add(
                occurrence
            )


        for key, gold_set in (
            gold_groups.items()
        ):

            surface, _ = key

            all_occurrences = set(
                range(
                    1,
                    len(
                        find_all_occurrences(
                            text,
                            surface,
                        )
                    )
                    + 1,
                )
            )

            # 只保留 ALL
            if (
                gold_set
                != all_occurrences
            ):
                continue


            pred_set = (
                pred_groups.get(
                    key,
                    set(),
                )
            )


            total_groups += 1

            gold_mentions += len(
                gold_set
            )

            pred_mentions += len(
                pred_set
            )

            tp += len(
                gold_set
                & pred_set
            )


            if pred_set == gold_set:

                exact_groups += 1


    fp = (
        pred_mentions
        - tp
    )

    fn = (
        gold_mentions
        - tp
    )


    precision = (
        tp
        / (
            tp + fp
        )
        if (
            tp + fp
        ) > 0
        else 0.0
    )

    recall = (
        tp
        / (
            tp + fn
        )
        if (
            tp + fn
        ) > 0
        else 0.0
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
        if (
            precision
            + recall
        ) > 0
        else 0.0
    )

    exact_accuracy = (
        exact_groups
        / total_groups
    )


    return {
        "groups":
            total_groups,

        "gold_mentions":
            gold_mentions,

        "pred_mentions":
            pred_mentions,

        "precision":
            precision,

        "recall":
            recall,

        "f1":
            f1,

        "exact":
            exact_accuracy,
    }


results = {
    "s2": [],
    "s3": [],
}


for seed, paths in (
    EXPERIMENTS.items()
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        f"SEED {seed}"
    )

    print(
        "=" * 80
    )


    for formulation in (
        "s2",
        "s3",
    ):

        records = load_jsonl(
            paths[
                formulation
            ]
        )

        result = (
            evaluate_all_groups(
                records,
                formulation,
            )
        )

        results[
            formulation
        ].append(
            result
        )


        print(
            f"\n{formulation.upper()}"
        )

        print(
            "Groups:",
            result["groups"],
        )

        print(
            "Predicted mentions:",
            result[
                "pred_mentions"
            ],
        )

        print(
            "Precision:",
            f"{result['precision']:.4f}",
        )

        print(
            "Recall:",
            f"{result['recall']:.4f}",
        )

        print(
            "F1:",
            f"{result['f1']:.4f}",
        )

        print(
            "Exact group accuracy:",
            f"{result['exact']:.4f}",
        )


print(
    "\n"
    + "=" * 80
)

print(
    "MEAN ± SAMPLE STD"
)

print(
    "=" * 80
)


for formulation in (
    "s2",
    "s3",
):

    print(
        f"\n{formulation.upper()}"
    )


    for metric in (
        "recall",
        "f1",
        "exact",
    ):

        values = np.asarray(
            [
                result[metric]
                for result
                in results[
                    formulation
                ]
            ]
        )


        print(
            f"{metric}: "
            f"{values.mean():.4f} "
            f"± "
            f"{values.std(ddof=1):.4f}"
        )