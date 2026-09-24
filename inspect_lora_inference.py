import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from peft import PeftModel

from data_process import (
    load_cluener_dataset,
)

from dataset_utils import (
    normalize_cluener_sample,
)

from preprocess_sft_sample import (
    build_messages,
)


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

ADAPTER_PATH = (
    "checkpoints/s1_lora_smoke"
)

NUM_SAMPLES = 5


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Tokenizer
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# Base model
# ============================================================

base_model = (
    AutoModelForCausalLM.from_pretrained(
        MODEL_NAME
    )
)


# ============================================================
# Load LoRA adapter
# ============================================================

model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
)

model = model.to(device)

model.eval()


# ============================================================
# Dataset
# ============================================================

dataset = load_cluener_dataset()

validation_dataset = dataset[
    "validation"
]


# ============================================================
# Generation
# ============================================================

def generate_prediction(
    sample,
):

    messages = build_messages(
        sample
    )

    # inference 时不能把 gold assistant 放进去
    prompt_messages = messages[:1]

    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():

        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
        )

    prompt_length = (
        inputs["input_ids"].shape[1]
    )

    generated_ids = output_ids[
        0,
        prompt_length:
    ]

    raw_output = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return raw_output


# ============================================================
# Inspect first few validation samples
# ============================================================

for i in range(NUM_SAMPLES):

    sample = normalize_cluener_sample(
        validation_dataset[i]
    )

    raw_output = generate_prediction(
        sample
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"Sample {i}"
    )

    print(
        "=" * 80
    )

    print("\nTEXT:")
    print(
        sample["text"]
    )

    print("\nGOLD:")
    print(
        sample["entities"]
    )

    print("\nPREDICTION:")
    print(
        raw_output
    )