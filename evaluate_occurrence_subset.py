import json
from collections import Counter

from span_alignment import (
    find_all_occurrences,
)


INPUT_PATH = (
    "outputs/"
    "s2_epoch3_validation_predictions.jsonl"
)


def get_gold_occurrence(
    source_text,
    entity,
):
    occurrences = find_all_occurrences(
        source_text,
        entity["text"],
    )

    gold_span = (
        entity["start"],
        entity["end"],
    )

    return (
        occurrences.index(gold_span)
        + 1
    )


def parse_occurrence_output(
    raw_output,
):
    try:
        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError:
        return []

    if not isinstance(data, dict):
        return []

    if not isinstance(
        data.get("entities"),
        list,
    ):
        return []

    parsed = []

    for entity in data["entities"]:

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

        parsed.append(
            {
                "text":
                    entity["text"],

                "type":
                    entity["type"],

                "occurrence":
                    entity[
                        "occurrence"
                    ],
            }
        )

    return parsed


# ============================================================
# Counters
# ============================================================

repeated_tp = 0
repeated_fp = 0
repeated_fn = 0

surface_type_matched = 0
surface_type_and_occurrence_correct = 0

nonfirst_gold = 0
nonfirst_pred = 0
nonfirst_tp = 0


with open(
    INPUT_PATH,
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

        pred_entities = (
            parse_occurrence_output(
                record[
                    "occurrence_raw_output"
                ]
            )
        )


        # ====================================================
        # Gold repeated-surface mentions
        # ====================================================

        gold_repeated = []

        for entity in gold_entities:

            occurrences = (
                find_all_occurrences(
                    source_text,
                    entity["text"],
                )
            )

            if len(occurrences) <= 1:
                continue

            occurrence = (
                get_gold_occurrence(
                    source_text,
                    entity,
                )
            )

            gold_repeated.append(
                (
                    entity["text"],
                    entity["type"],
                    occurrence,
                )
            )

            if occurrence > 1:
                nonfirst_gold += 1


        # ====================================================
        # Predicted repeated-surface mentions
        # ====================================================

        pred_repeated = []

        for entity in pred_entities:

            occurrences = (
                find_all_occurrences(
                    source_text,
                    entity["text"],
                )
            )

            if len(occurrences) <= 1:
                continue

            key = (
                entity["text"],
                entity["type"],
                entity["occurrence"],
            )

            pred_repeated.append(
                key
            )

            if entity["occurrence"] > 1:
                nonfirst_pred += 1


        # ====================================================
        # Exact repeated mention F1
        # ====================================================

        gold_counter = Counter(
            gold_repeated
        )

        pred_counter = Counter(
            pred_repeated
        )

        exact_tp = sum(
            (
                gold_counter
                & pred_counter
            ).values()
        )

        repeated_tp += exact_tp

        repeated_fp += (
            sum(
                pred_counter.values()
            )
            - exact_tp
        )

        repeated_fn += (
            sum(
                gold_counter.values()
            )
            - exact_tp
        )


        # ====================================================
        # occurrence accuracy conditional on
        # surface + type being matched
        # ====================================================

        gold_st = Counter(
            (text, entity_type)
            for (
                text,
                entity_type,
                occurrence,
            )
            in gold_repeated
        )

        pred_st = Counter(
            (text, entity_type)
            for (
                text,
                entity_type,
                occurrence,
            )
            in pred_repeated
        )

        matched_st = sum(
            (
                gold_st
                & pred_st
            ).values()
        )

        surface_type_matched += (
            matched_st
        )

        surface_type_and_occurrence_correct += (
            exact_tp
        )


        # ====================================================
        # non-first exact TP
        # ====================================================

        gold_nonfirst = Counter(
            key
            for key in gold_repeated
            if key[2] > 1
        )

        pred_nonfirst = Counter(
            key
            for key in pred_repeated
            if key[2] > 1
        )

        nonfirst_tp += sum(
            (
                gold_nonfirst
                & pred_nonfirst
            ).values()
        )


# ============================================================
# Metrics
# ============================================================

precision = (
    repeated_tp
    / (
        repeated_tp
        + repeated_fp
    )
    if repeated_tp
    + repeated_fp > 0
    else 0.0
)

recall = (
    repeated_tp
    / (
        repeated_tp
        + repeated_fn
    )
    if repeated_tp
    + repeated_fn > 0
    else 0.0
)

f1 = (
    2 * precision * recall
    / (
        precision
        + recall
    )
    if precision
    + recall > 0
    else 0.0
)


occurrence_accuracy = (
    surface_type_and_occurrence_correct
    / surface_type_matched
    if surface_type_matched > 0
    else 0.0
)


nonfirst_precision = (
    nonfirst_tp
    / nonfirst_pred
    if nonfirst_pred > 0
    else 0.0
)

nonfirst_recall = (
    nonfirst_tp
    / nonfirst_gold
    if nonfirst_gold > 0
    else 0.0
)


print(
    "=" * 80
)

print(
    "REPEATED-SURFACE END-TO-END"
)

print(
    "=" * 80
)

print(
    "TP:",
    repeated_tp,
)

print(
    "FP:",
    repeated_fp,
)

print(
    "FN:",
    repeated_fn,
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall:    {recall:.4f}"
)

print(
    f"F1:        {f1:.4f}"
)


print(
    "\n"
    + "=" * 80
)

print(
    "OCCURRENCE GIVEN CORRECT SURFACE-TYPE"
)

print(
    "=" * 80
)

print(
    "Surface-Type matched:",
    surface_type_matched,
)

print(
    "Occurrence correct:",
    surface_type_and_occurrence_correct,
)

print(
    f"Accuracy: {occurrence_accuracy:.4f}"
)


print(
    "\n"
    + "=" * 80
)

print(
    "NON-FIRST OCCURRENCE"
)

print(
    "=" * 80
)

print(
    "Gold occurrence > 1:",
    nonfirst_gold,
)

print(
    "Pred occurrence > 1:",
    nonfirst_pred,
)

print(
    "Exact TP:",
    nonfirst_tp,
)

print(
    f"Precision: {nonfirst_precision:.4f}"
)

print(
    f"Recall:    {nonfirst_recall:.4f}"
)