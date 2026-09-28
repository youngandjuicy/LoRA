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


def span_to_occurrence(
    source_text,
    entity,
):
    """
    将 gold entity 的绝对 span 转成 occurrence index。

    occurrence 从 1 开始计数。

    例如：
        text = "郭田勇...郭田勇"

    第二个“郭田勇”：
        occurrence = 2
    """

    surface = entity["text"]

    gold_span = (
        entity["start"],
        entity["end"],
    )

    occurrences = find_all_occurrences(
        source_text,
        surface,
    )

    if gold_span not in occurrences:
        return None

    occurrence_index = (
        occurrences.index(
            gold_span
        )
        + 1
    )

    return occurrence_index


def occurrence_to_span(
    source_text,
    surface,
    occurrence,
):
    """
    根据：
        surface + occurrence

    恢复：
        start + end
    """

    occurrences = find_all_occurrences(
        source_text,
        surface,
    )

    if (
        occurrence < 1
        or occurrence > len(occurrences)
    ):
        return None

    return occurrences[
        occurrence - 1
    ]


def analyze_split(
    dataset,
    split_name,
):

    total_samples = len(dataset)

    total_entities = 0

    conversion_failures = 0
    reconstruction_failures = 0

    repeated_surface_entities = 0

    occurrence_counter = Counter()

    failure_examples = []


    for sample_index in range(
        len(dataset)
    ):

        raw_sample = dataset[
            sample_index
        ]

        sample = normalize_cluener_sample(
            raw_sample
        )

        source_text = sample["text"]


        for entity in sample["entities"]:

            total_entities += 1

            surface = entity["text"]

            gold_span = (
                entity["start"],
                entity["end"],
            )


            # ========================================
            # 找出 surface 的所有 occurrence
            # ========================================

            occurrences = (
                find_all_occurrences(
                    source_text,
                    surface,
                )
            )


            if len(occurrences) > 1:

                repeated_surface_entities += 1


            # ========================================
            # span -> occurrence
            # ========================================

            occurrence = (
                span_to_occurrence(
                    source_text,
                    entity,
                )
            )


            if occurrence is None:

                conversion_failures += 1

                if len(
                    failure_examples
                ) < 10:

                    failure_examples.append(
                        {
                            "sample_index":
                                sample_index,

                            "text":
                                source_text,

                            "entity":
                                entity,

                            "occurrences":
                                occurrences,
                        }
                    )

                continue


            occurrence_counter[
                occurrence
            ] += 1


            # ========================================
            # occurrence -> span
            # ========================================

            reconstructed_span = (
                occurrence_to_span(
                    source_text,
                    surface,
                    occurrence,
                )
            )


            if (
                reconstructed_span
                != gold_span
            ):

                reconstruction_failures += 1

                if len(
                    failure_examples
                ) < 10:

                    failure_examples.append(
                        {
                            "sample_index":
                                sample_index,

                            "text":
                                source_text,

                            "entity":
                                entity,

                            "occurrence":
                                occurrence,

                            "gold_span":
                                gold_span,

                            "reconstructed_span":
                                reconstructed_span,
                        }
                    )


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
        "Entities:",
        total_entities,
    )

    print(
        "Repeated-surface entities:",
        repeated_surface_entities,
    )

    print(
        "Conversion failures:",
        conversion_failures,
    )

    print(
        "Reconstruction failures:",
        reconstruction_failures,
    )


    successful_entities = (
        total_entities
        - conversion_failures
        - reconstruction_failures
    )

    success_rate = (
        successful_entities
        / total_entities
        if total_entities > 0
        else 0.0
    )


    print(
        "Lossless reconstruction rate:",
        f"{success_rate:.6%}",
    )


    print(
        "\nOccurrence distribution:"
    )

    for occurrence in sorted(
        occurrence_counter
    ):

        count = occurrence_counter[
            occurrence
        ]

        print(
            f"occurrence={occurrence}: "
            f"{count}"
        )


    if failure_examples:

        print(
            "\nFailure examples:"
        )

        for example in (
            failure_examples
        ):

            print(
                example
            )


    return {
        "total_samples":
            total_samples,

        "total_entities":
            total_entities,

        "repeated_surface_entities":
            repeated_surface_entities,

        "conversion_failures":
            conversion_failures,

        "reconstruction_failures":
            reconstruction_failures,

        "success_rate":
            success_rate,
    }


def main():

    dataset = (
        load_cluener_dataset()
    )


    train_result = analyze_split(
        dataset["train"],
        "train",
    )


    validation_result = (
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


    all_lossless = (
        train_result[
            "conversion_failures"
        ] == 0

        and train_result[
            "reconstruction_failures"
        ] == 0

        and validation_result[
            "conversion_failures"
        ] == 0

        and validation_result[
            "reconstruction_failures"
        ] == 0
    )


    if all_lossless:

        print(
            "PASS: occurrence representation "
            "is lossless on train + validation."
        )

    else:

        print(
            "FAIL: occurrence representation "
            "is NOT lossless."
        )


if __name__ == "__main__":
    main()