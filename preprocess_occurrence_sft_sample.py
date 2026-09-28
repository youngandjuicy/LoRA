import json

from span_alignment import (
    find_all_occurrences,
)


OCCURRENCE_INSTRUCTION = (
    "从给定文本中抽取命名实体。"
    "实体类型只能为："
    "address、book、company、game、government、movie、"
    "name、organization、position、scene。"
    "输出必须是一个 JSON 对象，顶层唯一字段为 entities。"
    "entities 是列表，每个元素必须且只能包含 "
    "text、type、occurrence 三个字段。"
    "occurrence 表示该实体文本在原始文本中第几次出现，"
    "从 1 开始计数。"
    "例如 occurrence 为 1 表示第一次出现，"
    "occurrence 为 2 表示第二次出现。"
    "输出内容必须能够直接被 Python json.loads() 解析为字典。"
    "输出的第一个字符必须是 {，最后一个字符必须是 }。"
    '没有实体时输出 {"entities":[]}。'
)


def get_occurrence_index(
    source_text,
    entity,
):
    """
    根据 gold span 确定该实体是 surface 的第几次出现。

    返回 1-based occurrence index。
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


def build_occurrence_target_content(
    sample,
):

    source_text = sample["text"]

    target_entities = []

    for entity in sample["entities"]:

        occurrence = get_occurrence_index(
            source_text,
            entity,
        )

        target_entities.append(
            {
                "text":
                    entity["text"],

                "type":
                    entity["type"],

                "occurrence":
                    occurrence,
            }
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


def build_occurrence_messages(
    sample,
):

    user_content = (
        OCCURRENCE_INSTRUCTION
        + "\n"
        + "文本："
        + sample["text"]
    )

    assistant_content = (
        build_occurrence_target_content(
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


def preprocess_occurrence_sft_sample(
    sample,
    tokenizer,
    max_length=512,
):

    messages = (
        build_occurrence_messages(
            sample
        )
    )

    # user prompt + assistant generation header
    prompt_encoding = (
        tokenizer.apply_chat_template(
            messages[:1],
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )
    )

    # 完整 user + gold assistant
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


    # 确保 full sequence 的前缀
    # 真的是 prompt sequence
    if (
        full_ids[:len(prompt_ids)]
        != prompt_ids
    ):

        raise ValueError(
            "Prompt tokens are not "
            "a prefix of full tokens."
        )


    # truncate
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


    # completion-only SFT
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