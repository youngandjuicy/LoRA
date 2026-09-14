import json


CLUENER_LABELS = {
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene",
}


def parse_prediction(
    raw_output,
    source_text,
):

    result = {
        "json_valid": False,
        "schema_valid": False,
        "entities": [],
        "span_text_consistency": [],
    }

    # 第一步：是不是严格合法的 JSON
    try:
        data = json.loads(raw_output)

    except json.JSONDecodeError:
        return result

    result["json_valid"] = True

    # 第二步：顶层结构
    if not isinstance(data, dict):
        return result

    if set(data.keys()) != {"entities"}:
        return result

    if not isinstance(data["entities"], list):
        return result

    parsed_entities = []
    consistency = []

    # 第三步：逐个检查 entity
    for entity in data["entities"]:

        if not isinstance(entity, dict):
            return result

        required_keys = {
            "text",
            "type",
            "start",
            "end",
        }

        if set(entity.keys()) != required_keys:
            return result

        entity_text = entity["text"]
        entity_type = entity["type"]
        start = entity["start"]
        end = entity["end"]

        if not isinstance(entity_text, str):
            return result

        if entity_type not in CLUENER_LABELS:
            return result

        if type(start) is not int:
            return result

        if type(end) is not int:
            return result

        if not (
            0 <= start <= end < len(source_text)
        ):
            return result

        parsed_entities.append(
            {
                "text": entity_text,
                "type": entity_type,
                "start": start,
                "end": end,
            }
        )

        consistency.append(
            source_text[start:end + 1]
            == entity_text
        )

    result["schema_valid"] = True
    result["entities"] = parsed_entities
    result["span_text_consistency"] = consistency

    return result

if __name__ == "__main__":

    text = "浙商银行企业信贷部叶老桂博士"

    outputs = [
        # 1. 完全正确
        '{"entities":[{"text":"浙商银行","type":"company","start":0,"end":3}]}',

        # 2. JSON 正确，但 type 非法
        '{"entities":[{"text":"浙商银行","type":"bank","start":0,"end":3}]}',

        # 3. JSON 和 schema 正确，但 text 与 span 不一致
        '{"entities":[{"text":"中国银行","type":"company","start":0,"end":3}]}',

        # 4. Markdown code fence
        '```json\n{"entities":[]}\n```',

        # 5. 合法的空实体输出
        '{"entities":[]}',
    ]

    for output in outputs:

        print("\n" + "=" * 60)
        print(output)

        result = parse_prediction(
            output,
            text,
        )

        print(result)