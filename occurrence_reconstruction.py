import json

from evaluate import (
    CLUENER_LABELS,
)

from span_alignment import (
    find_all_occurrences,
)


def reconstruct_occurrence_prediction(
    raw_output,
    source_text,
):
    """
    将：

        text / type / occurrence

    确定性转换成：

        text / type / start / end

    如果 occurrence prediction 本身不合法，
    返回原始 raw_output。

    这样后续原有 span evaluator 会把该样本
    判为 schema invalid，而不是偷偷修复它。
    """

    try:
        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError:
        return raw_output


    # ==========================================
    # Strict occurrence schema validation
    # ==========================================

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
            "occurrence",
        }:
            return raw_output


        surface = entity["text"]

        entity_type = entity["type"]

        occurrence = entity[
            "occurrence"
        ]


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


        # bool 是 int 的子类，所以不能写
        # isinstance(occurrence, int)
        if type(occurrence) is not int:
            return raw_output


        if occurrence < 1:
            return raw_output


        # ======================================
        # occurrence -> span
        # ======================================

        occurrences = (
            find_all_occurrences(
                source_text,
                surface,
            )
        )


        if (
            occurrence
            > len(occurrences)
        ):
            return raw_output


        start, end = occurrences[
            occurrence - 1
        ]


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
        '"occurrence":2}'
        ']}'
    )

    print(
        reconstruct_occurrence_prediction(
            raw_output,
            text,
        )
    )