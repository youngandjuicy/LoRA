import json

from span_alignment import (
    find_all_occurrences,
)


GROUPED_OCCURRENCE_INSTRUCTION = (
    "从给定文本中抽取命名实体。"
    "实体类型只能为："
    "address、book、company、game、government、movie、"
    "name、organization、position、scene。"
    "输出必须是一个 JSON 对象，顶层唯一字段为 entities。"
    "entities 是列表，每个元素必须且只能包含 "
    "text、type、occurrences 三个字段。"
    "occurrences 是整数列表，表示该实体文本在原始文本中"
    "哪些出现位置被标注为该类型，从 1 开始计数。"
    "例如 occurrences 为 [1] 表示第一次出现，"
    "occurrences 为 [1,2] 表示第一次和第二次出现都被标注。"
    "相同 text 和 type 的多个标注必须合并到同一个实体对象中，"
    "并将对应的 occurrence 编号按升序放入 occurrences。"
    "输出内容必须能够直接被 Python json.loads() 解析为字典。"
    "输出的第一个字符必须是 {，最后一个字符必须是 }。"
    '没有实体时输出 {"entities":[]}。'
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
        raise ValueError(
            "Gold span cannot be mapped "
            "to an occurrence.\n"
            f"text={source_text}\n"
            f"entity={entity}\n"
            f"occurrences={occurrences}"
        )

    return (
        occurrences.index(
            gold_span
        )
        + 1
    )


def build_grouped_occurrence_target_content(
    sample,
):
    source_text = sample["text"]

    groups = {}

    for entity in sample["entities"]:

        key = (
            entity["text"],
            entity["type"],
        )

        occurrence = get_occurrence_index(
            source_text,
            entity,
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


    target_entities = []

    for group in groups.values():

        group[
            "occurrences"
        ].sort()

        target_entities.append(
            group
        )


    target = {
        "entities":
            target_entities
    }


    return json.dumps(
        target,
        ensure_ascii=False,
        separators=(",", ":"),
    )


def build_grouped_occurrence_messages(
    sample,
):
    user_content = (
        GROUPED_OCCURRENCE_INSTRUCTION
        + "\n"
        + "文本："
        + sample["text"]
    )

    assistant_content = (
        build_grouped_occurrence_target_content(
            sample
        )
    )

    return [
        {
            "role":
                "user",

            "content":
                user_content,
        },

        {
            "role":
                "assistant",

            "content":
                assistant_content,
        },
    ]


def preprocess_grouped_occurrence_sft_sample(
    sample,
    tokenizer,
    max_length=512,
):
    messages = (
        build_grouped_occurrence_messages(
            sample
        )
    )


    prompt_encoding = (
        tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    )


    full_encoding = (
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )
    )


    prompt_ids = (
        prompt_encoding["input_ids"]
    )

    full_ids = (
        full_encoding["input_ids"]
    )

    full_attention_mask = (
        full_encoding[
            "attention_mask"
        ]
    )


    if (
        full_ids[:len(prompt_ids)]
        != prompt_ids
    ):
        raise ValueError(
            "Prompt tokens are not "
            "a prefix of full tokens."
        )


    input_ids = (
        full_ids[:max_length]
    )

    attention_mask = (
        full_attention_mask[
            :max_length
        ]
    )


    prompt_length = min(
        len(prompt_ids),
        len(input_ids),
    )


    labels = (
        [-100] * prompt_length
        + input_ids[prompt_length:]
    )


    if all(
        label == -100
        for label in labels
    ):
        raise ValueError(
            "All labels are masked."
        )


    assert (
        len(input_ids)
        == len(attention_mask)
        == len(labels)
    )


    return {
        "input_ids":
            input_ids,

        "attention_mask":
            attention_mask,

        "labels":
            labels,
    }