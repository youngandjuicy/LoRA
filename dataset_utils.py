# 该函数用于将CLUENER数据集中的样本转换为统一的格式，便于后续处理和分析。它将原始样本中的实体信息提取出来，并按照实体的起始位置和结束位置进行排序，以确保模型在训练时能够按顺序输出实体，减少不必要的复杂度。
# 输入sample的格式为：
# {
#     "text": str,
#     "label": Dict[str, Dict[str, List[Tuple[int, int]]]]
# }
# 处理后的格式为：
# {
#     "text": str,
#     "entities": List[Dict[str, Union[str, int]]]
# }

def normalize_cluener_sample(sample):

    entities = []

    # 一个label可能对应多个type
    for entity_type, entity_dict in sample["label"].items():
        # 一个type可能对应多个实体
        for entity_text, spans in entity_dict.items():
            # 一个实体可能对应多个span（出现多次）
            for start, end in spans:

                entities.append(
                    {
                        "text": entity_text,
                        "type": entity_type,
                        "start": start,
                        "end": end,
                    }
                )
    # 按照实体的起始位置和结束位置进行排序，旨在让模型按顺序输出，避免不必要的复杂度
    entities.sort(
        key=lambda entity: (
            entity["start"],
            entity["end"],
        )
    )

    return {
        "text": sample["text"],
        "entities": entities,
    }

CLUENER_LABELS = [
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
]