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


    entities = data.get(
        "entities"
    )

    if not isinstance(
        entities,
        list,
    ):
        return []


    parsed = []


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


        parsed.append(
            (
                entity["text"],
                entity["type"],
                entity["occurrence"],
            )
        )


    return parsed


gold_occurrence_distribution = Counter()

pred_occurrence_distribution = Counter()


gold_counter = Counter()

pred_counter = Counter()


# 只统计 source 中 surface 确实重复出现的 mention
repeated_gold_counter = Counter()

repeated_pred_counter = Counter()


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

        raw_output = record[
            "occurrence_raw_output"
        ]


        # =====================================
        # Gold
        # =====================================

        for entity in gold_entities:

            occurrence = (
                get_gold_occurrence(
                    source_text,
                    entity,
                )
            )


            key = (
                entity["text"],
                entity["type"],
                occurrence,
            )


            gold_counter[key] += 1

            gold_occurrence_distribution[
                occurrence
            ] += 1


            all_occurrences = (
                find_all_occurrences(
                    source_text,
                    entity["text"],
                )
            )


            if len(all_occurrences) > 1:

                repeated_gold_counter[
                    key
                ] += 1


        # =====================================
        # Prediction
        # =====================================

        pred_entities = (
            parse_occurrence_output(
                raw_output
            )
        )


        for (
            surface,
            entity_type,
            occurrence,
        ) in pred_entities:

            key = (
                surface,
                entity_type,
                occurrence,
            )


            pred_counter[key] += 1

            pred_occurrence_distribution[
                occurrence
            ] += 1


            all_occurrences = (
                find_all_occurrences(
                    source_text,
                    surface,
                )
            )


            if len(all_occurrences) > 1:

                repeated_pred_counter[
                    key
                ] += 1


# ============================================================
# 注意：
# 上面的 Counter key 还缺 sample identity。
#
# 所以不能直接拿整个数据集的相同 surface 合并计算
# exact TP。
#
# 这里的 distribution 可以使用，
# exact repeated F1 我们下一步按 sample 计算。
# ============================================================


print(
    "=" * 80
)

print(
    "GOLD OCCURRENCE DISTRIBUTION"
)

print(
    "=" * 80
)

for k in sorted(
    gold_occurrence_distribution
):

    print(
        f"occurrence={k}: "
        f"{gold_occurrence_distribution[k]}"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "PREDICTED OCCURRENCE DISTRIBUTION"
)

print(
    "=" * 80
)

for k in sorted(
    pred_occurrence_distribution
):

    print(
        f"occurrence={k}: "
        f"{pred_occurrence_distribution[k]}"
    )