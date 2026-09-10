def normalize_cluener_sample(sample):

    entities = []

    # 一个label可能对应多个label
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
