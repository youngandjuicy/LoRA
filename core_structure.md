LoRA/
│
├── preprocess_sft_sample.py
│      build_messages()
│      preprocess_sft_sample()
│
├── sft_data_collator.py
│      sft_data_collator()
│
├── data_process.py
│      构造 Dataset
│      Dataset.map()
│      创建 DataLoader
│
└── train.py
       加载模型
       配置 LoRA
       optimizer
       training loop

# Day 2026-09-10 开始正式项目
LoRA/
├── data/
│   └── raw/
│       └── cluener/
├── outputs/
├── preprocess_sft_sample.py
├── sft_data_collator.py
├── data_process.py
├── train.py
├── inference.py
└── ...