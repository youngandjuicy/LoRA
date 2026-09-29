from collections import Counter

from data_process import (
    load_cluener_dataset,
)

from dataset_utils import (
    normalize_cluener_sample,
)

from span_alignment import (
    find_all_occurrences,
)


def get_occurrence_index(
    source_text,
    entity,
):
    surface = entity["text"]

    occurrences = find_all_occurrences(
        source_text,
        surface,
    )

    gold_span = (
        entity["start"],
        entity["end"],
    )

    if gold_span not in occurrences:
        return None

    return (
        occurrences.index(
            gold_span
        )
        + 1
    )


def convert_to_grouped_occurrence(
    sample,
):
    """
    standard span representation

        ->

    grouped occurrence representation
    """

    source_text = sample["text"]

    groups = {}

    for entity in sample["entities"]:

        occurrence = get_occurrence_index(
            source_text,
            entity,
        )

        if occurrence is None:
            raise ValueError(
                f"Cannot map entity: {entity}"
            )

        key = (
            entity["text"],
            entity["type"],
        )

        if key not in groups:

            groups[key] = {
                "text":
                    entity["text"],

                "type":
                    entity["type"],

                "occurrences":
                    [],
            }

        groups[key][
            "occurrences"
        ].append(
            occurrence
        )


    result = []

    for group in groups.values():

        group["occurrences"].sort()

        result.append(
            group
        )

    return result


def reconstruct_from_grouped_occurrence(
    source_text,
    grouped_entities,
):
    """
    grouped occurrence representation

        ->

    standard span representation
    """

    reconstructed = []


    for group in grouped_entities:

        surface = group["text"]

        entity_type = group["type"]

        source_occurrences = (
            find_all_occurrences(
                source_text,
                surface,
            )
        )


        for occurrence in (
            group["occurrences"]
        ):

            if (
                occurrence < 1
                or occurrence
                > len(source_occurrences)
            ):
                return None

            start, end = (
                source_occurrences[
                    occurrence - 1
                ]
            )

            reconstructed.append(
                {
                    "text":
                        surface,

                    "type":
                        entity_type,

                    "start":
                        start,

                    "end":
                        end,
                }
            )


    reconstructed.sort(
        key=lambda entity: (
            entity["start"],
            entity["end"],
            entity["type"],
            entity["text"],
        )
    )

    return reconstructed


def entity_counter(
    entities,
):
    return Counter(
        (
            entity["text"],
            entity["type"],
            entity["start"],
            entity["end"],
        )
        for entity in entities
    )


def analyze_split(
    dataset,
    split_name,
):

    total_samples = len(dataset)

    total_mentions = 0
    total_groups = 0

    repeated_groups = 0

    reconstruction_failures = 0

    group_size_distribution = Counter()


    for i in range(
        len(dataset)
    ):

        sample = normalize_cluener_sample(
            dataset[i]
        )

        grouped = (
            convert_to_grouped_occurrence(
                sample
            )
        )


        reconstructed = (
            reconstruct_from_grouped_occurrence(
                sample["text"],
                grouped,
            )
        )


        total_mentions += len(
            sample["entities"]
        )

        total_groups += len(
            grouped
        )


        for group in grouped:

            size = len(
                group["occurrences"]
            )

            group_size_distribution[
                size
            ] += 1

            if size > 1:
                repeated_groups += 1


        if (
            reconstructed is None
            or entity_counter(
                reconstructed
            )
            != entity_counter(
                sample["entities"]
            )
        ):

            reconstruction_failures += 1


    print(
        "\n"
        + "=" * 80
    )

    print(
        f"Split: {split_name}"
    )

    print(
        "=" * 80
    )

    print(
        "Samples:",
        total_samples,
    )

    print(
        "Mentions:",
        total_mentions,
    )

    print(
        "Grouped entity objects:",
        total_groups,
    )

    print(
        "Repeated groups:",
        repeated_groups,
    )

    print(
        "Reconstruction failures:",
        reconstruction_failures,
    )


    if reconstruction_failures == 0:

        print(
            "Lossless reconstruction rate: "
            "100.000000%"
        )


    print(
        "\nGroup size distribution:"
    )

    for size in sorted(
        group_size_distribution
    ):

        print(
            f"mentions_per_group={size}: "
            f"{group_size_distribution[size]}"
        )


    return (
        reconstruction_failures
    )


def main():

    dataset = load_cluener_dataset()


    train_failures = (
        analyze_split(
            dataset["train"],
            "train",
        )
    )


    validation_failures = (
        analyze_split(
            dataset["validation"],
            "validation",
        )
    )


    print(
        "\n"
        + "=" * 80
    )

    print(
        "FINAL CHECK"
    )

    print(
        "=" * 80
    )


    if (
        train_failures == 0
        and validation_failures == 0
    ):

        print(
            "PASS: grouped occurrence "
            "representation is lossless."
        )

    else:

        print(
            "FAIL: grouped occurrence "
            "representation is not lossless."
        )


if __name__ == "__main__":
    main()