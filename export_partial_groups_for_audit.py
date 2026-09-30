import csv
import json

from span_alignment import (
    find_all_occurrences,
)


S2_PATH = (
    "outputs/"
    "s2_epoch3_validation_predictions.jsonl"
)

S3_PATH = (
    "outputs/"
    "s3_epoch3_validation_predictions.jsonl"
)

OUTPUT_PATH = (
    "outputs/"
    "partial_groups_audit.csv"
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


def parse_s2_output(
    raw_output,
):
    """
    S2:
    text / type / occurrence
    """

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


def parse_s3_output(
    raw_output,
):
    """
    S3:
    text / type / occurrences[]
    """

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

        valid = True

        for occurrence in (
            entity["occurrences"]
        ):

            if type(occurrence) is not int:
                valid = False
                break

        if not valid:
            continue

        result.append(
            entity
        )

    return result


# ============================================================
# Load predictions
# ============================================================

with open(
    S2_PATH,
    "r",
    encoding="utf-8",
) as f:

    s2_records = [
        json.loads(line)
        for line in f
    ]


with open(
    S3_PATH,
    "r",
    encoding="utf-8",
) as f:

    s3_records = [
        json.loads(line)
        for line in f
    ]


assert (
    len(s2_records)
    == len(s3_records)
)


audit_rows = []


# ============================================================
# Build partial groups
# ============================================================

for sample_index, (
    s2_record,
    s3_record,
) in enumerate(
    zip(
        s2_records,
        s3_records,
    )
):

    source_text = s2_record[
        "text"
    ]

    assert (
        source_text
        == s3_record["text"]
    )


    gold_entities = s2_record[
        "gold_entities"
    ]


    s2_predictions = (
        parse_s2_output(
            s2_record[
                "occurrence_raw_output"
            ]
        )
    )


    s3_predictions = (
        parse_s3_output(
            s3_record[
                "grouped_raw_output"
            ]
        )
    )


    # ========================================
    # Gold 按 (surface, type) 分组
    # ========================================

    gold_groups = {}


    for entity in gold_entities:

        surface = entity["text"]

        source_occurrences = (
            find_all_occurrences(
                source_text,
                surface,
            )
        )

        # 只关心 repeated surface
        if len(source_occurrences) <= 1:
            continue


        key = (
            surface,
            entity["type"],
        )


        if key not in gold_groups:

            gold_groups[key] = []


        occurrence = (
            get_gold_occurrence(
                source_text,
                entity,
            )
        )


        gold_groups[key].append(
            occurrence
        )


    # ========================================
    # 检查 partial groups
    # ========================================

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


        all_occurrences = list(
            range(
                1,
                len(source_occurrences)
                + 1,
            )
        )


        gold_occurrences = sorted(
            set(
                gold_occurrences
            )
        )


        # ALL group 跳过
        if (
            gold_occurrences
            == all_occurrences
        ):
            continue


        missing_occurrences = [
            occurrence
            for occurrence
            in all_occurrences
            if occurrence
            not in gold_occurrences
        ]


        # ====================================
        # S2 prediction for this group
        # ====================================

        s2_occurrences = []

        for entity in s2_predictions:

            if (
                entity["text"]
                == surface
                and entity["type"]
                == entity_type
            ):

                s2_occurrences.append(
                    entity[
                        "occurrence"
                    ]
                )


        s2_occurrences = sorted(
            set(
                s2_occurrences
            )
        )


        # ====================================
        # S3 prediction for this group
        # ====================================

        s3_occurrences = []

        for entity in s3_predictions:

            if (
                entity["text"]
                == surface
                and entity["type"]
                == entity_type
            ):

                s3_occurrences.extend(
                    entity[
                        "occurrences"
                    ]
                )


        s3_occurrences = sorted(
            set(
                s3_occurrences
            )
        )


        # ====================================
        # Human-readable occurrence spans
        # ====================================

        occurrence_info = []

        for occurrence_index, (
            start,
            end,
        ) in enumerate(
            source_occurrences,
            start=1,
        ):

            occurrence_info.append(
                (
                    f"{occurrence_index}:"
                    f"[{start},{end}]"
                )
            )


        # S3 相比 gold 多出来的 occurrence
        s3_extra_occurrences = [
            occurrence
            for occurrence
            in s3_occurrences
            if occurrence
            not in gold_occurrences
        ]


        audit_rows.append(
            {
                "sample_index":
                    sample_index,

                "text":
                    source_text,

                "surface":
                    surface,

                "type":
                    entity_type,

                "source_occurrence_count":
                    len(
                        source_occurrences
                    ),

                "occurrence_spans":
                    " | ".join(
                        occurrence_info
                    ),

                "gold_occurrences":
                    str(
                        gold_occurrences
                    ),

                "unannotated_occurrences":
                    str(
                        missing_occurrences
                    ),

                "s2_predicted_occurrences":
                    str(
                        s2_occurrences
                    ),

                "s3_predicted_occurrences":
                    str(
                        s3_occurrences
                    ),

                "s3_extra_vs_gold":
                    str(
                        s3_extra_occurrences
                    ),

                # 人工填写
                "audit_label":
                    "",

                "audit_notes":
                    "",
            }
        )


# ============================================================
# Save CSV
# ============================================================

fieldnames = [
    "sample_index",
    "text",
    "surface",
    "type",
    "source_occurrence_count",
    "occurrence_spans",
    "gold_occurrences",
    "unannotated_occurrences",
    "s2_predicted_occurrences",
    "s3_predicted_occurrences",
    "s3_extra_vs_gold",
    "audit_label",
    "audit_notes",
]


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8-sig",
    newline="",
) as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )

    writer.writeheader()

    writer.writerows(
        audit_rows
    )


print(
    "Partial groups:",
    len(audit_rows),
)

print(
    "Saved to:",
    OUTPUT_PATH,
)