import json

from evaluate import (
    CLUENER_LABELS,
)

from span_alignment import (
    find_all_occurrences,
)


def reconstruct_grouped_occurrence_prediction(
    raw_output,
    source_text,
):
    """
    将：

        text / type / occurrences

    确定性转换成：

        text / type / start / end

    grouped schema 若非法，则不尝试修复，
    直接返回原始 raw_output。
    """

    try:
        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError:
        return raw_output


    # ========================================================
    # Strict grouped schema
    # ========================================================

    if not isinstance(data, dict):
        return raw_output

    if set(data.keys()) != {
        "entities"
    }:
        return raw_output


    entities = data["entities"]

    if not isinstance(
        entities,
        list,
    ):
        return raw_output


    reconstructed_entities = []


    for entity in entities:

        if not isinstance(
            entity,
            dict,
        ):
            return raw_output


        if set(entity.keys()) != {
            "text",
            "type",
            "occurrences",
        }:
            return raw_output


        surface = entity["text"]

        entity_type = entity["type"]

        predicted_occurrences = (
            entity["occurrences"]
        )


        if not isinstance(
            surface,
            str,
        ):
            return raw_output


        if (
            not isinstance(
                entity_type,
                str,
            )
            or entity_type
            not in CLUENER_LABELS
        ):
            return raw_output


        if not isinstance(
            predicted_occurrences,
            list,
        ):
            return raw_output


        # 空 occurrences 没有实际 mention
        if len(predicted_occurrences) == 0:
            return raw_output


        # 必须都是严格整数，并且 >= 1
        for occurrence in (
            predicted_occurrences
        ):

            if type(occurrence) is not int:
                return raw_output

            if occurrence < 1:
                return raw_output


        # instruction 要求升序；
        # 同时不允许 [1,1] 这种重复 occurrence
        if predicted_occurrences != sorted(
            set(predicted_occurrences)
        ):
            return raw_output


        source_occurrences = (
            find_all_occurrences(
                source_text,
                surface,
            )
        )


        # surface 不存在，或者 occurrence 越界
        if len(source_occurrences) == 0:
            return raw_output


        if (
            predicted_occurrences[-1]
            > len(source_occurrences)
        ):
            return raw_output


        # ====================================================
        # grouped occurrence -> standard mentions
        # ====================================================

        for occurrence in (
            predicted_occurrences
        ):

            start, end = (
                source_occurrences[
                    occurrence - 1
                ]
            )

            reconstructed_entities.append(
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


    reconstructed_entities.sort(
        key=lambda entity: (
            entity["start"],
            entity["end"],
            entity["type"],
            entity["text"],
        )
    )


    reconstructed_data = {
        "entities":
            reconstructed_entities
    }


    return json.dumps(
        reconstructed_data,
        ensure_ascii=False,
        separators=(",", ":"),
    )

if __name__ == "__main__":

    text = (
        "波尔图主场3比1取胜。"
        "受注盘费内巴切让出平/半中水，"
        "从积分上看，费内巴切仍然有出线机会，"
    )


    raw_output = (
        '{"entities":['
        '{"text":"费内巴切",'
        '"type":"organization",'
        '"occurrences":[1,2]}'
        ']}'
    )


    print(
        reconstruct_grouped_occurrence_prediction(
            raw_output,
            text,
        )
    )