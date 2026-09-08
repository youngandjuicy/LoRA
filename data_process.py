import torch
from transformers import AutoModelForCausalLM
from datasets import Dataset
from preprocess_sft_sample import preprocess_sft_sample
from transformers import AutoTokenizer
from sft_data_collator import sft_data_collator
from torch.utils.data import DataLoader
from peft import LoraConfig, TaskType, get_peft_model


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

data = {
    "text": [
        "张明于2024年加入浙江大学人工智能研究所。",
        "李华昨天前往北京大学参加会议。",
        "王强将在下周加入腾讯公司。",
    ],
    "entities": [
        [
            {"text": "张明", "type": "person"},
            {"text": "2024年", "type": "time"},
            {
                "text": "浙江大学人工智能研究所",
                "type": "organization",
            },
        ],
        [
            {"text": "李华", "type": "person"},
            {"text": "昨天", "type": "time"},
            {"text": "北京大学", "type": "organization"},
        ],
        [
            {"text": "王强", "type": "person"},
            {"text": "下周", "type": "time"},
            {"text": "腾讯公司", "type": "organization"},
        ],
    ],
}

dataset = Dataset.from_dict(data)
print(dataset)
print(dataset[0])

processed_dataset = dataset.map(
    lambda sample: preprocess_sft_sample(
        sample,
        tokenizer,
    )
)

print(processed_dataset)
print(processed_dataset[0])
print(processed_dataset[0].keys())

for i in range(len(processed_dataset)):
    print(
        i,
        len(processed_dataset[i]["input_ids"]),
        len(processed_dataset[i]["attention_mask"]),
        len(processed_dataset[i]["labels"]),
    )

features = [
    processed_dataset[0],
    processed_dataset[1],
    processed_dataset[2],
]

batch = sft_data_collator(
    features,
    tokenizer,
)

print(batch["input_ids"].shape)
print(batch["attention_mask"].shape)
print(batch["labels"].shape)

print(batch["attention_mask"])
print(batch["labels"])

train_dataloader = DataLoader(
    processed_dataset,
    batch_size=2,
    shuffle=False,
    collate_fn=lambda features: sft_data_collator(
        features,
        tokenizer,
    ),
)

for batch_idx, batch in enumerate(train_dataloader):

    print(f"\n===== Batch {batch_idx} =====")

    print("input_ids:")
    print(batch["input_ids"].shape)

    print("attention_mask:")
    print(batch["attention_mask"].shape)

    print("labels:")
    print(batch["labels"].shape)

print("===== Testing model forward pass =====")

base_model = AutoModelForCausalLM.from_pretrained(model_name)

device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

base_model = base_model.to(device)

batch = {
    key: value.to(device)
    for key, value in batch.items()
}

base_model.eval()

with torch.no_grad():
    outputs = base_model(**batch)

print(outputs.keys())

print("loss:")
print(outputs.loss)

print("logits shape:")
print(outputs.logits.shape)

print(
    "有效监督 token 数：",
    (batch["labels"] != -100).sum().item()
)

print(
    "总 token 位置数：",
    batch["labels"].numel()
)

print("\n===== Projection Layer Information =====")
for name, module in base_model.named_modules():
    if "proj" in name:
        print(name, type(module))

layer0_attn = base_model.model.layers[0].self_attn

for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
    module = getattr(layer0_attn, name)

    print(f"\n{name}")
    print(module)
    print("weight shape:", module.weight.shape)
    print("参数量:", module.weight.numel())

layer0_mlp = base_model.model.layers[0].mlp

for name in ["gate_proj", "up_proj", "down_proj"]:
    module = getattr(layer0_mlp, name)

    print(f"\n{name}")
    print(module)
    print("weight shape:", module.weight.shape)
    print("参数量:", module.weight.numel())

print("\n===== LoRA Configuration =====")

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(
    base_model,
    lora_config,
)

model.print_trainable_parameters()

print("\n===== Trainable Parameters =====")

for name, param in model.named_parameters():
    if param.requires_grad:
        print(
            name,
            param.shape,
            param.numel(),
        )

q_proj = model.base_model.model.model.layers[0].self_attn.q_proj

print("\n===== Model Structure =====")
print(type(model))
print(type(model.base_model))
print(type(model.base_model.model))
print(type(model.base_model.model.model))
print(type(model.base_model.model.model.layers[0]))
print(type(model.base_model.model.model.layers[0].self_attn))
print(type(model.base_model.model.model.layers[0].self_attn.q_proj))

print("\n===== q_proj after LoRA =====")
print(q_proj)

print(
    "base weight requires_grad:",
    q_proj.base_layer.weight.requires_grad,
)

print(
    "LoRA A requires_grad:",
    q_proj.lora_A["default"].weight.requires_grad,
)

print(
    "LoRA B requires_grad:",
    q_proj.lora_B["default"].weight.requires_grad,
)

print(
    "A abs sum:",
    q_proj.lora_A["default"].weight.abs().sum().item(),
)

print(
    "B abs sum:",
    q_proj.lora_B["default"].weight.abs().sum().item(),
)

print("\n===== first backward =====")

optimizer = torch.optim.AdamW(
    (
        param
        for param in model.parameters()
        if param.requires_grad
    ),
    lr=1e-3,
    weight_decay=0.0,
)

model.train()

q_proj = (
    model
    .base_model
    .model
    .model
    .layers[0]
    .self_attn
    .q_proj
)

base_weight = q_proj.base_layer.weight

lora_A = q_proj.lora_A["default"].weight
lora_B = q_proj.lora_B["default"].weight

base_before = base_weight.detach().clone()
A_before = lora_A.detach().clone()
B_before = lora_B.detach().clone()

optimizer.zero_grad()

outputs = model(**batch)

loss = outputs.loss

print("loss:", loss.item())

loss.backward()

print("\n===== Gradients after first backward =====")

print("base weight grad:")
print(base_weight.grad)

print(
    "A grad abs sum:",
    lora_A.grad.abs().sum().item()
)

print(
    "B grad abs sum:",
    lora_B.grad.abs().sum().item()
)

optimizer.step()

print("\n===== Parameter changes after first step =====")

print(
    "base weight change:",
    (base_weight - base_before).abs().sum().item()
)

print(
    "A change:",
    (lora_A - A_before).abs().sum().item()
)

print(
    "B change:",
    (lora_B - B_before).abs().sum().item()
)

print("\n===== second backward =====")

optimizer.zero_grad()

outputs = model(**batch)
loss = outputs.loss

loss.backward()

print("\n===== Gradients after second backward =====")

print(
    "A grad abs sum:",
    lora_A.grad.abs().sum().item()
)

print(
    "B grad abs sum:",
    lora_B.grad.abs().sum().item()
)