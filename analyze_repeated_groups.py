import json
from collections import defaultdict

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
            entity
        )

    return result


stats = {
    "all": {
        "groups": 0,
        "exact_groups": 0,
        "gold_mentions": 0,
        "pred_mentions": 0,
        "tp": 0,
    },

    "partial": {
        "groups": 0,
        "exact_groups": 0,
        "gold_mentions": 0,
        "pred_mentions": 0,
        "tp": 0,
    },
}


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


        # ============================================
        # gold 按 (surface, type) 分组
        # ============================================

        gold_groups = defaultdict(
            list
        )

        for entity in gold_entities:

            surface = entity["text"]

            occurrences = (
                find_all_occurrences(
                    source_text,
                    surface,
                )
            )

            # 只研究原文中 surface 重复出现的 group
            if len(occurrences) <= 1:
                continue

            occurrence = (
                get_gold_occurrence(
                    source_text,
                    entity,
                )
            )

            key = (
                surface,
                entity["type"],
            )

            gold_groups[key].append(
                occurrence
            )


        # ============================================
        # prediction 按 (surface, type) 分组
        # ============================================

        pred_groups = defaultdict(
            list
        )

        for entity in pred_entities:

            surface = entity["text"]

            occurrences = (
                find_all_occurrences(
                    source_text,
                    surface,
                )
            )

            if len(occurrences) <= 1:
                continue

            key = (
                surface,
                entity["type"],
            )

            pred_groups[key].append(
                entity["occurrence"]
            )


        # ============================================
        # 分析每个 gold repeated group
        # ============================================

        for (
            surface,
            entity_type,
        ), gold_occurrences in (
            gold_groups.items()
        ):

            source_occurrences = (
                find_all_occurrences(
                    source_text,
                    surface,
                )
            )

            total_source_count = len(
                source_occurrences
            )

            gold_set = set(
                gold_occurrences
            )

            pred_set = set(
                pred_groups.get(
                    (
                        surface,
                        entity_type,
                    ),
                    [],
                )
            )


            all_occurrence_set = set(
                range(
                    1,
                    total_source_count + 1,
                )
            )


            if (
                gold_set
                == all_occurrence_set
            ):

                category = "all"

            else:

                category = "partial"


            stats[category][
                "groups"
            ] += 1


            stats[category][
                "gold_mentions"
            ] += len(
                gold_set
            )


            stats[category][
                "pred_mentions"
            ] += len(
                pred_set
            )


            tp = len(
                gold_set
                & pred_set
            )


            stats[category][
                "tp"
            ] += tp


            if pred_set == gold_set:

                stats[category][
                    "exact_groups"
                ] += 1


# ============================================================
# Print
# ============================================================

for category in (
    "all",
    "partial",
):

    result = stats[
        category
    ]

    groups = result[
        "groups"
    ]

    exact_groups = result[
        "exact_groups"
    ]

    gold_mentions = result[
        "gold_mentions"
    ]

    pred_mentions = result[
        "pred_mentions"
    ]

    tp = result[
        "tp"
    ]


    group_accuracy = (
        exact_groups / groups
        if groups > 0
        else 0.0
    )


    precision = (
        tp / pred_mentions
        if pred_mentions > 0
        else 0.0
    )


    recall = (
        tp / gold_mentions
        if gold_mentions > 0
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


    print(
        "\n"
        + "=" * 80
    )

    print(
        category.upper()
    )

    print(
        "=" * 80
    )

    print(
        "Groups:",
        groups,
    )

    print(
        "Exact group matches:",
        exact_groups,
    )

    print(
        "Exact group accuracy:",
        f"{group_accuracy:.4f}",
    )

    print(
        "Gold mentions:",
        gold_mentions,
    )

    print(
        "Pred mentions:",
        pred_mentions,
    )

    print(
        "TP:",
        tp,
    )

    print(
        "Precision:",
        f"{precision:.4f}",
    )

    print(
        "Recall:",
        f"{recall:.4f}",
    )

    print(
        "F1:",
        f"{f1:.4f}",
    )