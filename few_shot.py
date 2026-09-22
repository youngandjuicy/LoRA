from preprocess_sft_sample import (
    BASE_INSTRUCTION,
    build_target_content,
)


def build_few_shot_messages(
    query_sample,
    demo_samples,
):

    messages = []

    # demonstrations
    for demo in demo_samples:

        messages.append(
            {
                "role": "user",
                "content": (
                    f"文本：{demo['text']}"
                ),
            }
        )

        messages.append(
            {
                "role": "assistant",
                "content": build_target_content(
                    demo
                ),
            }
        )

    # 真正的 query：
    # 与 zero-shot 的 user prompt 保持完全一致
    messages.append(
        {
            "role": "user",
            "content": (
                f"{BASE_INSTRUCTION}\n\n"
                f"文本：{query_sample['text']}"
            ),
        }
    )

    return messages