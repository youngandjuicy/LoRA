from transformers import AutoTokenizer

from data_process import load_cluener_dataset
from dataset_utils import normalize_cluener_sample
from few_shot import build_few_shot_messages


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

DEMO_INDICES = [
    1824,
    4012,
    2286,
]


dataset = load_cluener_dataset()

train_dataset = dataset["train"]
validation_dataset = dataset["validation"]

demo_samples = [
    normalize_cluener_sample(
        train_dataset[idx]
    )
    for idx in DEMO_INDICES
]

query_sample = normalize_cluener_sample(
    validation_dataset[0]
)

messages = build_few_shot_messages(
    query_sample,
    demo_samples,
)


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

rendered = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

print(rendered)