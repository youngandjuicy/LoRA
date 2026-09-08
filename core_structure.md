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