from statistics import mean

from datasets import load_dataset
from transformers import AutoTokenizer

from dataset_utils import normalize_cluener_sample
from preprocess_sft_sample import build_messages


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(
    model_name
)


dataset = load_dataset(
    "json",
    data_files={
        "train": "data/raw/cluener/train.json",
        "validation": "data/raw/cluener/dev.json",
    },
)

# 输入messages的数据类型是 List[Dict[str, str]]，其中每个字典表示一条消息，包含两个键值对：
# "role"：表示消息的角色，可以是 "system"、"user" 或 "assistant"。
# "content"：表示消息的内容，是一个字符串。
# 该函数把 messages 转换为 input_ids，方便后续计算 token 长度
def get_input_ids(
    tokenizer,
    messages,
    add_generation_prompt,
):

    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=True,
    )

    return encoded["input_ids"]

# 该函数用于计算values列表中指定百分位数p的值。它首先对values进行排序，然后根据百分位数计算索引，并返回对应的值。
def percentile(values, p):

    values = sorted(values)

    index = int(
        (len(values) - 1) * p
    )

    return values[index]

# 该函数用于打印给定名称和长度列表的统计信息，包括平均值、50th、90th、95th、99th百分位数以及最大值。
def print_statistics(
    name,
    lengths,
):

    print(f"\n===== {name} Length =====")

    print(
        "mean:",
        mean(lengths),
    )

    print(
        "P50:",
        percentile(lengths, 0.50),
    )

    print(
        "P90:",
        percentile(lengths, 0.90),
    )

    print(
        "P95:",
        percentile(lengths, 0.95),
    )

    print(
        "P99:",
        percentile(lengths, 0.99),
    )

    print(
        "max:",
        max(lengths),
    )

# 该函数调用前三个工具函数，分析数据集的token长度，包括prompt长度、target长度和full长度。它遍历数据集中的每个样本，计算相应的长度，并打印统计信息和截断候选样本数量。
def analyze_token_lengths(
    dataset_split,
    split_name,
):

    prompt_lengths = []
    target_lengths = []
    full_lengths = []

    for raw_sample in dataset_split:

        sample = normalize_cluener_sample(
            raw_sample
        )

        messages = build_messages(sample)

        prompt_ids = get_input_ids(
            tokenizer,
            messages[:1],
            add_generation_prompt=True,
        )

        full_ids = get_input_ids(
            tokenizer,
            messages,
            add_generation_prompt=False,
        )

        # 和 preprocess_sft_sample 中的假设保持一致
        assert (
            full_ids[:len(prompt_ids)]
            == prompt_ids
        )

        prompt_length = len(prompt_ids)
        full_length = len(full_ids)

        target_length = (
            full_length
            - prompt_length
        )

        prompt_lengths.append(
            prompt_length
        )

        target_lengths.append(
            target_length
        )

        full_lengths.append(
            full_length
        )

    print("\n" + "=" * 70)
    print(f"Split: {split_name}")
    print("=" * 70)

    print_statistics(
        "Prompt",
        prompt_lengths,
    )

    print_statistics(
        "Target",
        target_lengths,
    )

    print_statistics(
        "Full",
        full_lengths,
    )

    print("\n===== Truncation Candidates =====")

    for max_length in [
        128,
        192,
        256,
        384,
        512,
    ]:

        count = sum(
            length > max_length
            for length in full_lengths
        )

        print(
            f"> {max_length}: "
            f"{count} samples "
            f"({count / len(full_lengths):.4%})"
        )

analyze_token_lengths(
    dataset["train"],
    "train",
)

analyze_token_lengths(
    dataset["validation"],
    "validation",
)