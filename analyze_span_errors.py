import json
import argparse

from collections import (
    defaultdict,
    Counter,
)


# ============================================================
# Arguments
# ============================================================

parser = argparse.ArgumentParser()

parser.add_argument(
    "--prediction_path",
    type=str,
    required=True,
)

parser.add_argument(
    "--analysis_output",
    type=str,
    default=None,
)

parser.add_argument(
    "--show_examples",
    type=int,
    default=10,
)

args = parser.parse_args()


# ============================================================
# Relaxed prediction parser
# ============================================================

def relaxed_entities(raw_output):

    try:
        data = json.loads(
            raw_output
        )

    except Exception:
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


        text = entity.get(
            "text"
        )

        entity_type = entity.get(
            "type"
        )

        start = entity.get(
            "start"
        )

        end = entity.get(
            "end"
        )


        if (
            isinstance(text, str)
            and isinstance(
                entity_type,
                str,
            )
            and type(start) is int
            and type(end) is int
        ):

            result.append(
                {
                    "text": text,
                    "type": entity_type,
                    "start": start,
                    "end": end,
                }
            )


    return result


# ============================================================
# Find all exact occurrences
# ============================================================

def all_occurrences(
    text,
    surface,
):

    positions = []

    search_start = 0


    while True:

        idx = text.find(
            surface,
            search_start,
        )


        if idx == -1:
            break


        positions.append(
            (
                idx,
                idx + len(surface) - 1,
            )
        )


        # 允许 overlapping occurrence
        search_start = idx + 1


    return positions


# ============================================================
# Group by (text, type)
# ============================================================

def group_by_surface_type(
    entities,
):

    groups = defaultdict(list)


    for entity in entities:

        key = (
            entity["text"],
            entity["type"],
        )

        groups[key].append(
            entity
        )


    return groups


# ============================================================
# Match gold / prediction
# ============================================================

def match_surface_type_entities(
    gold_entities,
    pred_entities,
):

    gold_groups = (
        group_by_surface_type(
            gold_entities
        )
    )

    pred_groups = (
        group_by_surface_type(
            pred_entities
        )
    )


    pairs = []


    common_keys = (
        gold_groups.keys()
        & pred_groups.keys()
    )


    for key in common_keys:

        gold_group = sorted(
            gold_groups[key],
            key=lambda entity: (
                entity["start"],
                entity["end"],
            ),
        )

        pred_group = sorted(
            pred_groups[key],
            key=lambda entity: (
                entity["start"],
                entity["end"],
            ),
        )


        for gold_entity, pred_entity in zip(
            gold_group,
            pred_group,
        ):

            pairs.append(
                (
                    gold_entity,
                    pred_entity,
                )
            )


    return pairs


# ============================================================
# Length bucket
# ============================================================

def length_bucket(
    length,
):

    if length <= 2:
        return "1-2"

    elif length <= 4:
        return "3-4"

    elif length <= 8:
        return "5-8"

    else:
        return "9+"


# ============================================================
# Position bucket
# ============================================================

def position_bucket(
    normalized_position,
):

    if normalized_position < 1 / 3:
        return "first_third"

    elif normalized_position < 2 / 3:
        return "middle_third"

    else:
        return "last_third"


# ============================================================
# Load prediction file
# ============================================================

records = []


with open(
    args.prediction_path,
    "r",
    encoding="utf-8",
) as f:

    for line in f:

        line = line.strip()

        if not line:
            continue


        records.append(
            json.loads(
                line
            )
        )


print(
    "number of samples:",
    len(records),
)


# ============================================================
# Build matched-pair records
# ============================================================

pairs = []

total_gold_entities = 0


for sample_index, record in enumerate(
    records
):

    source_text = record[
        "text"
    ]

    gold_entities = record[
        "gold_entities"
    ]

    raw_output = record[
        "raw_output"
    ]


    total_gold_entities += len(
        gold_entities
    )


    pred_entities = relaxed_entities(
        raw_output
    )


    matched_pairs = (
        match_surface_type_entities(
            gold_entities,
            pred_entities,
        )
    )


    for (
        gold_entity,
        pred_entity,
    ) in matched_pairs:

        # 保留误差方向
        start_error = (
            pred_entity["start"]
            - gold_entity["start"]
        )

        end_error = (
            pred_entity["end"]
            - gold_entity["end"]
        )


        max_abs_error = max(
            abs(start_error),
            abs(end_error),
        )


        gold_length = (
            gold_entity["end"]
            - gold_entity["start"]
            + 1
        )


        normalized_position = (
            gold_entity["start"]
            / max(
                1,
                len(source_text) - 1,
            )
        )


        occurrences = all_occurrences(
            source_text,
            gold_entity["text"],
        )


        repeated = (
            len(occurrences) >= 2
        )


        pred_start = (
            pred_entity["start"]
        )

        pred_end = (
            pred_entity["end"]
        )


        pred_span_in_range = (
            0
            <= pred_start
            <= pred_end
            < len(source_text)
        )


        if pred_span_in_range:

            pred_span_text_matches = (
                source_text[
                    pred_start:
                    pred_end + 1
                ]
                == pred_entity["text"]
            )

        else:

            pred_span_text_matches = False


        pairs.append(
            {
                "sample_index":
                    sample_index,

                "source_text":
                    source_text,

                "surface":
                    gold_entity["text"],

                "type":
                    gold_entity["type"],

                "gold_start":
                    gold_entity["start"],

                "gold_end":
                    gold_entity["end"],

                "pred_start":
                    pred_entity["start"],

                "pred_end":
                    pred_entity["end"],

                "start_error":
                    start_error,

                "end_error":
                    end_error,

                "max_abs_error":
                    max_abs_error,

                "gold_length":
                    gold_length,

                "normalized_position":
                    normalized_position,

                "length_bucket":
                    length_bucket(
                        gold_length
                    ),

                "position_bucket":
                    position_bucket(
                        normalized_position
                    ),

                "repeated":
                    repeated,

                "num_occurrences":
                    len(occurrences),

                "pred_span_in_range":
                    pred_span_in_range,

                "pred_span_text_matches":
                    pred_span_text_matches,
            }
        )


# ============================================================
# Basic statistics
# ============================================================

n = len(
    pairs
)


print(
    "\n"
    + "=" * 80
)

print(
    "BASIC STATISTICS"
)

print(
    "=" * 80
)


print(
    "total gold entities:",
    total_gold_entities,
)

print(
    "surface/type matched entities:",
    n,
)


if total_gold_entities > 0:

    print(
        "matched / gold:",
        f"{n / total_gold_entities:.4f}",
    )


if n == 0:

    raise RuntimeError(
        "No matched entities found."
    )


# ============================================================
# Span error distribution
# ============================================================

exact = sum(
    item["max_abs_error"] == 0
    for item in pairs
)

d1 = sum(
    item["max_abs_error"] == 1
    for item in pairs
)

d2 = sum(
    item["max_abs_error"] == 2
    for item in pairs
)

gt2 = sum(
    item["max_abs_error"] > 2
    for item in pairs
)


within1 = sum(
    item["max_abs_error"] <= 1
    for item in pairs
)

within2 = sum(
    item["max_abs_error"] <= 2
    for item in pairs
)


mae_start = (
    sum(
        abs(
            item["start_error"]
        )
        for item in pairs
    )
    / n
)

mae_end = (
    sum(
        abs(
            item["end_error"]
        )
        for item in pairs
    )
    / n
)


mean_max_error = (
    sum(
        item["max_abs_error"]
        for item in pairs
    )
    / n
)


print(
    "\n"
    + "=" * 80
)

print(
    "SPAN ERROR DISTRIBUTION"
)

print(
    "=" * 80
)


for name, count in [
    ("d = 0", exact),
    ("d = 1", d1),
    ("d = 2", d2),
    ("d > 2", gt2),
]:

    print(
        f"{name:8s}"
        f" | count={count:4d}"
        f" | rate={count / n:.2%}"
    )


print(
    "\nd <= 1:",
    f"{within1 / n:.2%}",
)

print(
    "d <= 2:",
    f"{within2 / n:.2%}",
)

print(
    "\nMAE start:",
    f"{mae_start:.4f}",
)

print(
    "MAE end:",
    f"{mae_end:.4f}",
)

print(
    "mean max error:",
    f"{mean_max_error:.4f}",
)


# ============================================================
# Signed start/end error distribution
# ============================================================

start_error_counter = Counter(
    item["start_error"]
    for item in pairs
)

end_error_counter = Counter(
    item["end_error"]
    for item in pairs
)


print(
    "\n"
    + "=" * 80
)

print(
    "SIGNED START ERROR"
)

print(
    "=" * 80
)


for error in sorted(
    start_error_counter.keys()
):

    count = start_error_counter[
        error
    ]

    print(
        f"{error:+4d}"
        f" | {count:4d}"
        f" | {count / n:.2%}"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "SIGNED END ERROR"
)

print(
    "=" * 80
)


for error in sorted(
    end_error_counter.keys()
):

    count = end_error_counter[
        error
    ]

    print(
        f"{error:+4d}"
        f" | {count:4d}"
        f" | {count / n:.2%}"
    )

joint_error_counter = Counter(
    (
        item["start_error"],
        item["end_error"],
    )
    for item in pairs
)

print(
    "\n"
    + "=" * 80
)

print(
    "JOINT START/END ERROR"
)

print(
    "=" * 80
)


for (
    start_error,
    end_error,
), count in (
    joint_error_counter.most_common(20)
):

    print(
        f"({start_error:+d}, "
        f"{end_error:+d})"
        f" | count={count:4d}"
        f" | rate={count / n:.2%}"
    )

# ============================================================
# Length analysis
# ============================================================

length_groups = defaultdict(
    list
)


for item in pairs:

    length_groups[
        item["length_bucket"]
    ].append(
        item
    )


print(
    "\n"
    + "=" * 80
)

print(
    "LENGTH ANALYSIS"
)

print(
    "=" * 80
)


for bucket in [
    "1-2",
    "3-4",
    "5-8",
    "9+",
]:

    items = length_groups[
        bucket
    ]


    if not items:
        continue


    count = len(
        items
    )


    exact_count = sum(
        item["max_abs_error"] == 0
        for item in items
    )


    within1_count = sum(
        item["max_abs_error"] <= 1
        for item in items
    )


    within2_count = sum(
        item["max_abs_error"] <= 2
        for item in items
    )


    mean_error = (
        sum(
            item["max_abs_error"]
            for item in items
        )
        / count
    )


    print(
        f"{bucket:4s}"
        f" | n={count:4d}"
        f" | exact={exact_count / count:.2%}"
        f" | d<=1={within1_count / count:.2%}"
        f" | d<=2={within2_count / count:.2%}"
        f" | mean={mean_error:.4f}"
    )


# ============================================================
# Position analysis
# ============================================================

position_groups = defaultdict(
    list
)


for item in pairs:

    position_groups[
        item["position_bucket"]
    ].append(
        item
    )


print(
    "\n"
    + "=" * 80
)

print(
    "POSITION ANALYSIS"
)

print(
    "=" * 80
)


for bucket in [
    "first_third",
    "middle_third",
    "last_third",
]:

    items = position_groups[
        bucket
    ]


    if not items:
        continue


    count = len(
        items
    )


    exact_count = sum(
        item["max_abs_error"] == 0
        for item in items
    )


    within1_count = sum(
        item["max_abs_error"] <= 1
        for item in items
    )


    within2_count = sum(
        item["max_abs_error"] <= 2
        for item in items
    )


    mean_max_error = (
        sum(
            item["max_abs_error"]
            for item in items
        )
        / count
    )


    mean_start_error = (
        sum(
            item["start_error"]
            for item in items
        )
        / count
    )


    mean_end_error = (
        sum(
            item["end_error"]
            for item in items
        )
        / count
    )


    print(
        f"{bucket:13s}"
        f" | n={count:4d}"
        f" | exact={exact_count / count:.2%}"
        f" | d<=1={within1_count / count:.2%}"
        f" | d<=2={within2_count / count:.2%}"
        f" | mean_max={mean_max_error:.4f}"
        f" | mean_start={mean_start_error:+.4f}"
        f" | mean_end={mean_end_error:+.4f}"
    )


# ============================================================
# Repeated mention analysis
# ============================================================

repeated_groups = defaultdict(
    list
)


for item in pairs:

    repeated_groups[
        item["repeated"]
    ].append(
        item
    )


print(
    "\n"
    + "=" * 80
)

print(
    "REPEATED MENTION ANALYSIS"
)

print(
    "=" * 80
)


for repeated in [
    False,
    True,
]:

    items = repeated_groups[
        repeated
    ]


    if not items:
        continue


    count = len(
        items
    )


    exact_count = sum(
        item["max_abs_error"] == 0
        for item in items
    )


    within1_count = sum(
        item["max_abs_error"] <= 1
        for item in items
    )


    within2_count = sum(
        item["max_abs_error"] <= 2
        for item in items
    )


    mean_error = (
        sum(
            item["max_abs_error"]
            for item in items
        )
        / count
    )


    name = (
        "repeated"
        if repeated
        else "non-repeated"
    )


    print(
        f"{name:12s}"
        f" | n={count:4d}"
        f" | exact={exact_count / count:.2%}"
        f" | d<=1={within1_count / count:.2%}"
        f" | d<=2={within2_count / count:.2%}"
        f" | mean={mean_error:.4f}"
    )


# ============================================================
# Show examples
# ============================================================

print(
    "\n"
    + "=" * 80
)

print(
    "EXAMPLES BY ERROR CATEGORY"
)

print(
    "=" * 80
)


categories = [
    (
        "exact",
        lambda item:
            item["max_abs_error"] == 0,
    ),
    (
        "d1",
        lambda item:
            item["max_abs_error"] == 1,
    ),
    (
        "d2",
        lambda item:
            item["max_abs_error"] == 2,
    ),
    (
        "gt2",
        lambda item:
            item["max_abs_error"] > 2,
    ),
]


for category_name, condition in categories:

    print(
        "\n---",
        category_name,
        "---",
    )


    shown = 0


    for item in pairs:

        if not condition(
            item
        ):
            continue


        print(
            {
                "sample_index":
                    item["sample_index"],

                "surface":
                    item["surface"],

                "type":
                    item["type"],

                "gold":
                    (
                        item["gold_start"],
                        item["gold_end"],
                    ),

                "pred":
                    (
                        item["pred_start"],
                        item["pred_end"],
                    ),

                "start_error":
                    item["start_error"],

                "end_error":
                    item["end_error"],

                "text":
                    item["source_text"],
            }
        )


        shown += 1


        if shown >= args.show_examples:
            break


# ============================================================
# Save detailed records
# ============================================================

if args.analysis_output is not None:

    with open(
        args.analysis_output,
        "w",
        encoding="utf-8",
    ) as f:

        for item in pairs:

            f.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                )
                + "\n"
            )


    print(
        "\nDetailed analysis saved to:",
        args.analysis_output,
    )