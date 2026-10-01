import json
import numpy as np

from span_alignment import (
    find_all_occurrences,
)

from evaluate import (
    CLUENER_LABELS,
)


S2_PATH = (
    "outputs/"
    "s2_epoch3_validation_predictions.jsonl"
)

S3_PATH = (
    "outputs/"
    "s3_epoch3_validation_predictions.jsonl"
)

NUM_BOOTSTRAP = 10000
BOOTSTRAP_SEED = 42


# ============================================================
# Basic utilities
# ============================================================

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


def safe_divide(
    numerator,
    denominator,
):

    if denominator == 0:
        return 0.0

    return (
        numerator
        / denominator
    )


def calculate_f1(
    tp,
    fp,
    fn,
):

    precision = safe_divide(
        tp,
        tp + fp,
    )

    recall = safe_divide(
        tp,
        tp + fn,
    )

    if (
        precision
        + recall
        == 0
    ):
        f1 = 0.0

    else:

        f1 = (
            2
            * precision
            * recall
            / (
                precision
                + recall
            )
        )

    return (
        precision,
        recall,
        f1,
    )


# ============================================================
# Standard span parser
#
# Reproduce the main strict evaluation logic:
#
# prediction entity identity:
#
#     (start, end, type)
#
# Invalid schema -> empty prediction for that sample.
# ============================================================

def parse_standard_prediction(
    raw_output,
    source_text,
):

    try:

        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        return set()


    if not isinstance(
        data,
        dict,
    ):
        return set()


    if set(data.keys()) != {
        "entities"
    }:
        return set()


    entities = data[
        "entities"
    ]


    if not isinstance(
        entities,
        list,
    ):
        return set()


    parsed = set()


    for entity in entities:

        if not isinstance(
            entity,
            dict,
        ):
            return set()


        if set(
            entity.keys()
        ) != {
            "text",
            "type",
            "start",
            "end",
        }:
            return set()


        text = entity[
            "text"
        ]

        entity_type = entity[
            "type"
        ]

        start = entity[
            "start"
        ]

        end = entity[
            "end"
        ]


        if not isinstance(
            text,
            str,
        ):
            return set()


        if (
            not isinstance(
                entity_type,
                str,
            )
            or entity_type
            not in CLUENER_LABELS
        ):
            return set()


        # bool 是 int 的子类，因此必须用 type(...) is int
        if type(start) is not int:
            return set()

        if type(end) is not int:
            return set()


        if (
            start < 0
            or end < start
            or end >= len(
                source_text
            )
        ):
            return set()


        parsed.add(
            (
                start,
                end,
                entity_type,
            )
        )


    return parsed


def gold_span_set(
    gold_entities,
):

    return {
        (
            entity["start"],
            entity["end"],
            entity["type"],
        )
        for entity
        in gold_entities
    }


# ============================================================
# Overall sample-level statistics
# ============================================================

def build_sample_statistics(
    records,
):

    statistics = []


    for record in records:

        source_text = record[
            "text"
        ]


        gold = gold_span_set(
            record[
                "gold_entities"
            ]
        )


        pred = (
            parse_standard_prediction(
                record[
                    "raw_output"
                ],
                source_text,
            )
        )


        tp = len(
            gold & pred
        )

        fp = len(
            pred - gold
        )

        fn = len(
            gold - pred
        )


        statistics.append(
            (
                tp,
                fp,
                fn,
            )
        )


    return np.asarray(
        statistics,
        dtype=np.int64,
    )


# ============================================================
# Native S2 / S3 parsers
# Used only for repeated-group analysis
# ============================================================

def parse_s2_output(
    raw_output,
):

    try:

        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        return []


    if not isinstance(
        data,
        dict,
    ):
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

    try:

        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        return []


    if not isinstance(
        data,
        dict,
    ):
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


        if not all(
            type(x) is int
            for x
            in entity[
                "occurrences"
            ]
        ):
            continue


        result.append(
            entity
        )


    return result


def get_gold_occurrence(
    source_text,
    entity,
):

    occurrences = (
        find_all_occurrences(
            source_text,
            entity["text"],
        )
    )


    gold_span = (
        entity["start"],
        entity["end"],
    )


    return (
        occurrences.index(
            gold_span
        )
        + 1
    )


# ============================================================
# ALL repeated groups
#
# Each row:
#
# gold_n,
# S2 pred_n,
# S2 tp,
# S2 exact,
# S3 pred_n,
# S3 tp,
# S3 exact
# ============================================================

def build_all_group_statistics(
    s2_records,
    s3_records,
):

    group_statistics = []


    for s2_record, s3_record in zip(
        s2_records,
        s3_records,
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


        # ----------------------------------------
        # Gold groups
        # ----------------------------------------

        gold_groups = {}


        for entity in gold_entities:

            surface = entity[
                "text"
            ]


            source_occurrences = (
                find_all_occurrences(
                    source_text,
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


            if key not in gold_groups:

                gold_groups[
                    key
                ] = set()


            gold_groups[
                key
            ].add(
                get_gold_occurrence(
                    source_text,
                    entity,
                )
            )


        # ----------------------------------------
        # S2 predictions
        # ----------------------------------------

        s2_groups = {}


        for entity in (
            parse_s2_output(
                s2_record[
                    "occurrence_raw_output"
                ]
            )
        ):

            surface = entity[
                "text"
            ]


            if len(
                find_all_occurrences(
                    source_text,
                    surface,
                )
            ) <= 1:
                continue


            key = (
                surface,
                entity["type"],
            )


            if key not in s2_groups:

                s2_groups[
                    key
                ] = set()


            s2_groups[
                key
            ].add(
                entity[
                    "occurrence"
                ]
            )


        # ----------------------------------------
        # S3 predictions
        # ----------------------------------------

        s3_groups = {}


        for entity in (
            parse_s3_output(
                s3_record[
                    "grouped_raw_output"
                ]
            )
        ):

            surface = entity[
                "text"
            ]


            if len(
                find_all_occurrences(
                    source_text,
                    surface,
                )
            ) <= 1:
                continue


            key = (
                surface,
                entity["type"],
            )


            if key not in s3_groups:

                s3_groups[
                    key
                ] = set()


            s3_groups[
                key
            ].update(
                entity[
                    "occurrences"
                ]
            )


        # ----------------------------------------
        # Only ALL groups
        # ----------------------------------------

        for (
            surface,
            entity_type,
        ), gold_set in (
            gold_groups.items()
        ):

            source_occurrences = (
                find_all_occurrences(
                    source_text,
                    surface,
                )
            )


            all_occurrences = set(
                range(
                    1,
                    len(
                        source_occurrences
                    )
                    + 1,
                )
            )


            if (
                gold_set
                != all_occurrences
            ):
                continue


            key = (
                surface,
                entity_type,
            )


            s2_pred = (
                s2_groups.get(
                    key,
                    set(),
                )
            )


            s3_pred = (
                s3_groups.get(
                    key,
                    set(),
                )
            )


            group_statistics.append(
                (
                    len(
                        gold_set
                    ),

                    len(
                        s2_pred
                    ),

                    len(
                        gold_set
                        & s2_pred
                    ),

                    int(
                        s2_pred
                        == gold_set
                    ),

                    len(
                        s3_pred
                    ),

                    len(
                        gold_set
                        & s3_pred
                    ),

                    int(
                        s3_pred
                        == gold_set
                    ),
                )
            )


    return np.asarray(
        group_statistics,
        dtype=np.int64,
    )


# ============================================================
# Bootstrap helpers
# ============================================================

def percentile_ci(
    values,
):

    lower, upper = (
        np.percentile(
            values,
            [
                2.5,
                97.5,
            ],
        )
    )


    return (
        lower,
        upper,
    )


def print_delta_summary(
    name,
    observed_delta,
    bootstrap_deltas,
):

    lower, upper = (
        percentile_ci(
            bootstrap_deltas
        )
    )


    positive_rate = np.mean(
        bootstrap_deltas
        > 0
    )


    print(
        f"\n{name}"
    )

    print(
        f"Observed delta: "
        f"{observed_delta:.6f}"
    )

    print(
        "95% percentile CI: "
        f"[{lower:.6f}, "
        f"{upper:.6f}]"
    )

    print(
        "Bootstrap P(delta > 0): "
        f"{positive_rate:.4f}"
    )


# ============================================================
# Main
# ============================================================

def main():

    rng = (
        np.random.default_rng(
            BOOTSTRAP_SEED
        )
    )


    s2_records = load_jsonl(
        S2_PATH
    )

    s3_records = load_jsonl(
        S3_PATH
    )


    assert (
        len(s2_records)
        == len(s3_records)
    )


    assert all(
        s2["text"]
        == s3["text"]
        for s2, s3
        in zip(
            s2_records,
            s3_records,
        )
    )


    # ========================================================
    # Part 1:
    # Overall Strict Micro F1
    # ========================================================

    s2_stats = (
        build_sample_statistics(
            s2_records
        )
    )

    s3_stats = (
        build_sample_statistics(
            s3_records
        )
    )


    s2_total = s2_stats.sum(
        axis=0
    )

    s3_total = s3_stats.sum(
        axis=0
    )


    s2_p, s2_r, s2_f1 = (
        calculate_f1(
            *s2_total
        )
    )

    s3_p, s3_r, s3_f1 = (
        calculate_f1(
            *s3_total
        )
    )


    print(
        "=" * 80
    )

    print(
        "OBSERVED OVERALL"
    )

    print(
        "=" * 80
    )

    print(
        f"S2 Strict F1: "
        f"{s2_f1:.6f}"
    )

    print(
        f"S3 Strict F1: "
        f"{s3_f1:.6f}"
    )

    print(
        f"Delta: "
        f"{s3_f1 - s2_f1:.6f}"
    )


    num_samples = len(
        s2_stats
    )


    overall_deltas = []


    for _ in range(
        NUM_BOOTSTRAP
    ):

        indices = rng.integers(
            0,
            num_samples,
            size=num_samples,
        )


        s2_boot = (
            s2_stats[
                indices
            ].sum(
                axis=0
            )
        )


        s3_boot = (
            s3_stats[
                indices
            ].sum(
                axis=0
            )
        )


        _, _, s2_boot_f1 = (
            calculate_f1(
                *s2_boot
            )
        )

        _, _, s3_boot_f1 = (
            calculate_f1(
                *s3_boot
            )
        )


        overall_deltas.append(
            s3_boot_f1
            - s2_boot_f1
        )


    overall_deltas = np.asarray(
        overall_deltas
    )


    print_delta_summary(
        "OVERALL STRICT F1 DELTA (S3 - S2)",
        s3_f1 - s2_f1,
        overall_deltas,
    )


    # ========================================================
    # Part 2:
    # ALL repeated groups
    # ========================================================

    group_stats = (
        build_all_group_statistics(
            s2_records,
            s3_records,
        )
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "OBSERVED ALL REPEATED GROUPS"
    )

    print(
        "=" * 80
    )

    print(
        "Number of groups:",
        len(
            group_stats
        ),
    )


    def group_metrics(
        rows,
        method,
    ):

        gold_n = rows[
            :,
            0
        ].sum()


        if method == "s2":

            pred_n = rows[
                :,
                1
            ].sum()

            tp = rows[
                :,
                2
            ].sum()

            exact = rows[
                :,
                3
            ].mean()


        else:

            pred_n = rows[
                :,
                4
            ].sum()

            tp = rows[
                :,
                5
            ].sum()

            exact = rows[
                :,
                6
            ].mean()


        fp = (
            pred_n
            - tp
        )

        fn = (
            gold_n
            - tp
        )


        precision, recall, f1 = (
            calculate_f1(
                tp,
                fp,
                fn,
            )
        )


        return {
            "precision":
                precision,

            "recall":
                recall,

            "f1":
                f1,

            "exact":
                exact,
        }


    s2_group = group_metrics(
        group_stats,
        "s2",
    )

    s3_group = group_metrics(
        group_stats,
        "s3",
    )


    for name, result in (
        (
            "S2",
            s2_group,
        ),
        (
            "S3",
            s3_group,
        ),
    ):

        print(
            f"\n{name}"
        )

        print(
            f"Recall: "
            f"{result['recall']:.6f}"
        )

        print(
            f"F1: "
            f"{result['f1']:.6f}"
        )

        print(
            "Exact group accuracy: "
            f"{result['exact']:.6f}"
        )


    num_groups = len(
        group_stats
    )


    recall_deltas = []

    f1_deltas = []

    exact_deltas = []


    for _ in range(
        NUM_BOOTSTRAP
    ):

        indices = rng.integers(
            0,
            num_groups,
            size=num_groups,
        )


        sampled = (
            group_stats[
                indices
            ]
        )


        s2_result = group_metrics(
            sampled,
            "s2",
        )

        s3_result = group_metrics(
            sampled,
            "s3",
        )


        recall_deltas.append(
            s3_result[
                "recall"
            ]
            - s2_result[
                "recall"
            ]
        )


        f1_deltas.append(
            s3_result[
                "f1"
            ]
            - s2_result[
                "f1"
            ]
        )


        exact_deltas.append(
            s3_result[
                "exact"
            ]
            - s2_result[
                "exact"
            ]
        )


    print_delta_summary(
        "ALL RECALL DELTA (S3 - S2)",
        (
            s3_group["recall"]
            - s2_group["recall"]
        ),
        np.asarray(
            recall_deltas
        ),
    )


    print_delta_summary(
        "ALL F1 DELTA (S3 - S2)",
        (
            s3_group["f1"]
            - s2_group["f1"]
        ),
        np.asarray(
            f1_deltas
        ),
    )


    print_delta_summary(
        "ALL EXACT GROUP ACCURACY DELTA (S3 - S2)",
        (
            s3_group["exact"]
            - s2_group["exact"]
        ),
        np.asarray(
            exact_deltas
        ),
    )


if __name__ == "__main__":
    main()