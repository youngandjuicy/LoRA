# 该脚本旨在分析 CLUENER 数据集的基本情况，包括：
# - 样本数
# - 实体数
# - 类别分布
# - 文本长度
# - 实体数分布
# - 检查标注的实体 span 是否与原文一致
# - 检查某个已标注实体文本是否在原文中出现多次（可能导致模型训练时的歧义）

from collections import Counter

from datasets import load_dataset

from dataset_utils import normalize_cluener_sample


dataset = load_dataset(
    "json",
    data_files={
        "train": "data/raw/cluener/train.json",
        "validation": "data/raw/cluener/dev.json",
    },
)


def analyze_split(dataset_split, split_name):

    label_counter = Counter()

    num_samples = len(dataset_split)
    num_entities = 0
    num_empty_samples = 0

    text_lengths = []
    entity_counts = []

    span_errors = []

    repeated_surface_samples = []

    for idx, raw_sample in enumerate(dataset_split):

        sample = normalize_cluener_sample(raw_sample)

        text = sample["text"]
        entities = sample["entities"]

        text_lengths.append(len(text))
        entity_counts.append(len(entities))

        if len(entities) == 0:
            num_empty_samples += 1

        num_entities += len(entities)

        # 统计类别
        for entity in entities:
            label_counter[entity["type"]] += 1

            start = entity["start"]
            end = entity["end"]

            # CLUENER 的 end 是闭区间，所以 Python slicing 要 end + 1
            span_text = text[start:end + 1]

            if span_text != entity["text"]:
                span_errors.append(
                    {
                        "sample_idx": idx,
                        "text": text,
                        "entity": entity,
                        "span_text": span_text,
                    }
                )

        # 检查：某个已标注实体文本是否在原文中出现多次
        ambiguous_entities = []

        for entity in entities:
            surface = entity["text"]

            if text.count(surface) > 1:
                ambiguous_entities.append(
                    {
                        "text": surface,
                        "type": entity["type"],
                        "count_in_text": text.count(surface),
                    }
                )

        if ambiguous_entities:
            repeated_surface_samples.append(
                {
                    "sample_idx": idx,
                    "text": text,
                    "entities": ambiguous_entities,
                }
            )

    print("\n" + "=" * 70)
    print(f"Split: {split_name}")
    print("=" * 70)

    print("样本数:", num_samples)
    print("实体总数:", num_entities)

    print(
        "平均每条实体数:",
        num_entities / num_samples,
    )

    print(
        "无实体样本数:",
        num_empty_samples,
    )

    print(
        "无实体样本比例:",
        num_empty_samples / num_samples,
    )

    print(
        "平均文本字符数:",
        sum(text_lengths) / num_samples,
    )

    print(
        "最长文本字符数:",
        max(text_lengths),
    )

    print("\n===== Label Distribution =====")

    for label, count in label_counter.most_common():
        print(label, count)

    print("\n===== Span Check =====")
    print("span 不一致数量:", len(span_errors))

    print("\n===== Repeated Surface Form =====")
    print(
        "存在重复实体文本的样本数:",
        len(repeated_surface_samples),
    )

    print(
        "比例:",
        len(repeated_surface_samples) / num_samples,
    )

    print("\n前 5 个重复实体案例:")

    for example in repeated_surface_samples[:5]:
        print(example)


analyze_split(
    dataset["train"],
    "train",
)

analyze_split(
    dataset["validation"],
    "validation",
)