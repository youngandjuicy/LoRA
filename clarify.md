# data
- 传入一个字典，有两个键。
- "text"键的值是一个Python列表，列表的每个元素是一句话（字符串）
- "entities"键的值也是一个python列表，列表的每个元素也是一个列表，对应每句话应该输出的内容。列表中的元素分别是是人物时间地点对应的字典
- 数据的条数 = "text"键的值的长度 = "entities"键的值的长度

# dataset

- `dataset` 是 HuggingFace `datasets` 库中的 `Dataset` 对象，不是普通的 Python 字典。
- 由 `Dataset.from_dict(data)` 根据 `data` 创建。
- 可以把 `dataset` 理解成一张表：
  - 每个键对应一列，例如 `"text"` 和 `"entities"`。
  - 相同下标的数据组成一行，也就是一条 sample。
- 当前 `dataset` 一共有 3 条数据，因此 `len(dataset) == 3`。
- `dataset[0]` 表示取出第 0 条 sample。
- `dataset[i]` 得到的是一个 Python 字典：
  - `"text"` 的值是一个字符串。
  - `"entities"` 的值是一个列表。
  - `"entities"` 列表中的每个元素是一个字典，描述一个实体的 `"text"` 和 `"type"`。
- 因此可以把数据层级理解为：

  dataset
  → sample（dict）
  → text（str） / entities（list）
  → entity（dict）

# processed_dataset

- `processed_dataset` 仍然是 HuggingFace 的 `Dataset` 对象。
- 由 `dataset.map(...)` 得到。
- `map()` 会逐条取出 `dataset` 中的 sample，并将 sample 传给
  `preprocess_sft_sample(sample, tokenizer)`。
- `preprocess_sft_sample` 返回一个字典，返回字典的 key 决定 Dataset
  中新增或更新的列名。
- 如果返回：
  `"input_ids"`、`"attention_mask"`、`"labels"`，
  那么 processed_dataset 就会增加这三列。
- 原来的 `"text"` 和 `"entities"` 列默认仍然保留。
- 注意，新增的input_ids是对原句增加完整提示语以及期望回答之后再进行tokenize的结果，并不是直接tokenize

# features

- `features` 是一个普通 Python 列表。
- 列表中的每个元素都是 `processed_dataset` 中的一条 sample。
- 因此：
  - `features` 的类型可以理解为 `list[dict]`。
  - `features[i]` 是一个字典，表示一条经过预处理的样本。
- 每条 sample 中主要包含：
  - `"input_ids"`：token id 构成的 Python 列表。
  - `"attention_mask"`：Python 列表。
  - `"labels"`：Python 列表。
  - 原来的 `"text"`、`"entities"` 等字段通常也仍然保留。
- 不同 sample 的 `"input_ids"`、`"attention_mask"`、`"labels"` 长度可以不同。
- `features` 表示准备组成一个 batch 的若干条样本。
- 这里手动从 `processed_dataset` 取出 3 条数据，是为了测试
  `sft_data_collator`。
- `sft_data_collator(features, tokenizer)` 会进一步对这些样本进行
  padding，并整理成 Tensor batch。

# batch

- `batch` 是 `sft_data_collator(features, tokenizer)` 的返回值。
- `batch` 是一个 Python 字典。
- 字典中主要包含三个键：
  - `"input_ids"`
  - `"attention_mask"`
  - `"labels"`
- 这三个键对应的值都是 PyTorch Tensor。
- Tensor 的 shape 一般为：

  `[batch_size, sequence_length]`

- `batch_size` 表示这一批有多少条样本。
- `sequence_length` 是这一批样本经过 padding 后统一的序列长度。
- `sft_data_collator` 会把 `features` 中若干条长度可能不同的样本进行 padding，
  然后转换成可以直接输入模型的 Tensor。

- 数据结构发生了：

  `features: list[dict]`

  → `batch: dict[str, Tensor]`

- `"input_ids"`：
  模型的输入 token id。

- `"attention_mask"`：
  区分真实 token 和 padding token，通常真实位置为 1，padding 位置为 0。

- `"labels"`：
  训练时用于计算 loss 的目标 token id。
  SFT 中不希望参与 loss 计算的位置通常会被设置成 `-100`。


# train_dataloader

- `train_dataloader` 是 PyTorch 的 `DataLoader` 对象。
- 它负责从 `processed_dataset` 中自动按 batch 取出数据。
- 它本身不是一个 batch，而是一个可以不断产生 batch 的可迭代对象。

- `processed_dataset`
  - DataLoader 的数据来源。
  - `processed_dataset[i]` 是一条 sample，类型为字典。

- `batch_size=2`
  - 每次最多取 2 条 sample。
  - 如果最后剩余的数据不足 2 条，默认仍然会组成最后一个 batch。

- `shuffle=False`
  - 不打乱样本顺序。
  - 因此会按照 dataset 中原来的顺序取数据。

- `collate_fn`
  - 决定“一组 sample 如何整理成一个 batch”。
  - DataLoader 会先把若干 sample 放入一个 Python 列表：
    `features: list[dict]`
  - 然后调用 `collate_fn(features)`。

- 当前的 collate_fn：

  `lambda features: sft_data_collator(features, tokenizer)`

  等价于定义一个只接受 `features` 的函数，
  再在内部调用 `sft_data_collator(features, tokenizer)`。

- `sft_data_collator` 会完成 padding、Tensor 转换等操作，
  最终产生：

  `batch: dict[str, Tensor]`

- 因此整体过程是：

  `processed_dataset`
  → 若干 `sample`
  → `features: list[dict]`
  → `sft_data_collator`
  → `batch: dict[str, Tensor]`