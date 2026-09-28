import json


def find_all_occurrences(
    source_text,
    surface,
):
    """
    找到 surface 在 source_text 中的所有精确出现位置。

    返回：
    [
        (start, end),
        ...
    ]

    end 为闭区间。
    """

    if not isinstance(surface, str):
        return []

    if surface == "":
        return []

    occurrences = []

    search_start = 0

    while True:

        start = source_text.find(
            surface,
            search_start,
        )

        if start == -1:
            break

        end = (
            start
            + len(surface)
            - 1
        )

        occurrences.append(
            (start, end)
        )

        # +1，因此可以找到潜在的重叠 occurrence
        search_start = start + 1

    return occurrences


def choose_nearest_occurrence(
    occurrences,
    predicted_start,
    predicted_end,
):
    """
    如果同一个 surface 在原文中出现多次，
    使用模型原始预测的 span 作为粗定位 anchor，
    选择距离最近的 occurrence。
    """

    return min(
        occurrences,
        key=lambda span: (
            abs(
                span[0]
                - predicted_start
            )
            +
            abs(
                span[1]
                - predicted_end
            ),

            # tie-break
            abs(
                span[0]
                - predicted_start
            ),

            span[0],
        ),
    )


def align_entity_span(
    source_text,
    entity,
):
    """
    只修正一个预测实体的 start/end。

    不修改：
    - text
    - type
    - 其他字段
    """

    if not isinstance(entity, dict):
        return entity

    aligned_entity = dict(entity)

    surface = entity.get("text")

    predicted_start = entity.get(
        "start"
    )

    predicted_end = entity.get(
        "end"
    )

    # alignment 只处理我们能够理解的实体
    if not isinstance(surface, str):
        return aligned_entity

    # bool 是 int 的子类，因此显式排除
    if (
        type(predicted_start) is not int
        or type(predicted_end) is not int
    ):
        return aligned_entity

    occurrences = find_all_occurrences(
        source_text,
        surface,
    )

    # 模型生成的 surface 根本不存在于原文
    # 不强行修改。
    if len(occurrences) == 0:
        return aligned_entity

    # 唯一 occurrence：直接确定
    if len(occurrences) == 1:

        aligned_start, aligned_end = (
            occurrences[0]
        )

    # repeated mention：
    # 用原始 coarse span 消除歧义
    else:

        aligned_start, aligned_end = (
            choose_nearest_occurrence(
                occurrences,
                predicted_start,
                predicted_end,
            )
        )

    aligned_entity["start"] = (
        aligned_start
    )

    aligned_entity["end"] = (
        aligned_end
    )

    return aligned_entity


def align_prediction(
    raw_output,
    source_text,
):
    """
    对一整条模型输出做 span alignment。

    注意：
    这里只修正 span。

    非法 JSON、非法结构、非法 type 等问题
    全部留给原来的 evaluator 处理。
    """

    try:

        data = json.loads(
            raw_output
        )

    except json.JSONDecodeError:

        return raw_output

    if not isinstance(data, dict):
        return raw_output

    entities = data.get(
        "entities"
    )

    if not isinstance(
        entities,
        list,
    ):
        return raw_output

    aligned_entities = []

    for entity in entities:

        aligned_entity = (
            align_entity_span(
                source_text,
                entity,
            )
        )

        aligned_entities.append(
            aligned_entity
        )

    aligned_data = dict(data)

    aligned_data["entities"] = (
        aligned_entities
    )

    return json.dumps(
        aligned_data,
        ensure_ascii=False,
        separators=(",", ":"),
    )