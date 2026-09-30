# Experiments

本文件记录 CLUENER2020 生成式 NER 项目的主要实验配置、结果与观察。

---

## Experimental Protocol

### Dataset

Dataset: CLUENER2020

原始有标签数据：

| Original Split | Samples |
|---|---:|
| train | 10748 |
| dev | 1343 |

固定随机划分：

- split seed: 42
- official train → 90% train + 10% validation
- official dev → final test

| Project Split | Samples | Usage |
|---|---:|---|
| train | 9673 | 模型训练 |
| validation | 1075 | 开发、调参与模型选择 |
| test | 1343 | 最终评测 |

从本实验开始，不使用 test split 进行 prompt、超参数或模型选择。

### Task

生成式中文命名实体识别（Generative NER）。

CLUENER 实体类别：

`address, book, company, game, government, movie, name, organization, position, scene`

Span-aware 输出格式：

```json
{
  "entities": [
    {
      "text": "实体文本",
      "type": "实体类型",
      "start": 0,
      "end": 3
    }
  ]
}
```

其中：

* `start` 和 `end` 为从 0 开始的字符索引；
* `end` 为闭区间；
* 主 NER 指标以 `(start, end, type)` 为实体身份；
* `text` 不参与 strict TP 判断，但用于 span-text consistency 检查。

### Evaluation

主要指标：

**Strict Span Micro-F1**

一个实体只有 `(start, end, type)` 全部正确时才算 TP。

辅助指标：

**Surface-Type F1**

忽略字符 offset，仅比较 `(text, type)`。
使用 Counter 保留重复 mention 数量。
该指标只用于诊断，不作为主 benchmark 指标。

生成质量指标：

| Metric                | Definition                                           |
| --------------------- | ---------------------------------------------------- |
| JSON validity         | 输出能否直接被 `json.loads()` 解析                            |
| Schema validity       | JSON 是否满足完整任务 schema，包括合法字段、type 和 span 范围           |
| Span-text consistency | `source_text[start:end+1] == predicted_text` 的预测实体比例 |
| Duplicate sample rate | 包含重复 `(start,end,type)` 预测的样本比例                      |

### Inference Configuration

* decoding: greedy
* `do_sample=False`
* `max_new_tokens=256`
* model mode: `eval()`

---

# Experiment B0 — Zero-shot Base Model

## Goal

建立未经任务微调的 Base Model validation baseline。

后续 LoRA SFT、schema ablation 等实验均与该结果比较。

## Model

`Qwen/Qwen2.5-0.5B-Instruct`

未进行任何 CLUENER 训练。

## Data

Split: validation

Samples: 1075

## Prompt

```text
从给定文本中抽取命名实体。
实体类型只能为：address、book、company、game、government、movie、name、organization、position、scene。
输出必须是一个 JSON 对象，顶层唯一字段为 entities。
entities 是列表，每个元素必须且只能包含 text、type、start、end 四个字段。
start 和 end 是实体在原始文本中的字符索引，从 0 开始，end 为闭区间。
输出内容必须能够直接被 Python json.loads() 解析为字典。
输出的第一个字符必须是 {，最后一个字符必须是 }。
没有实体时输出 {"entities":[]}。
```

## Results

### Overall

| Metric                 |     Result |
| ---------------------- | ---------: |
| JSON validity          |   80.6512% |
| Schema validity        |   44.0930% |
| Span-text consistency  |    0.0000% |
| Strict Micro Precision |     0.0028 |
| Strict Micro Recall    |     0.0004 |
| **Strict Micro F1**    | **0.0007** |
| Strict Macro F1        |     0.0004 |
| Surface-Type Precision |     0.1061 |
| Surface-Type Recall    |     0.0828 |
| **Surface-Type F1**    | **0.0930** |

### Strict Per-Class

| Class        | Precision | Recall |     F1 | TP |  FP |  FN |
| ------------ | --------: | -----: | -----: | -: | --: | --: |
| address      |    0.0000 | 0.0000 | 0.0000 |  0 |   5 | 289 |
| book         |    0.0000 | 0.0000 | 0.0000 |  0 |   1 | 139 |
| company      |    0.0000 | 0.0000 | 0.0000 |  0 |  14 | 291 |
| game         |    0.0000 | 0.0000 | 0.0000 |  0 |  48 | 223 |
| government   |    0.0000 | 0.0000 | 0.0000 |  0 |  18 | 187 |
| movie        |    0.0000 | 0.0000 | 0.0000 |  0 |   6 | 121 |
| name         |    0.0064 | 0.0028 | 0.0039 |  1 | 155 | 353 |
| organization |    0.0000 | 0.0000 | 0.0000 |  0 |  83 | 324 |
| position     |    0.0000 | 0.0000 | 0.0000 |  0 |   3 | 293 |
| scene        |    0.0000 | 0.0000 | 0.0000 |  0 |  18 | 183 |

## Observations

Base Model 已具备一定的 JSON 输出和实体语义能力，但尚未掌握 CLUENER 的任务协议。

JSON validity 达到约 80.7%，说明模型通常能够产生 JSON 风格的结构化输出；但 schema validity 仅约 44.1%，说明字段、合法实体类型或 span 范围仍经常违反任务约束。

Surface-Type F1 为 0.0930，说明 Base Model 能够识别少量正确的 `(text, type)` 实体。

Strict Span F1 仅为 0.0007，而 span-text consistency 为 0%，表明绝对字符 offset 生成是当前 zero-shot Base Model 的主要失败点之一。

该结果作为后续 LoRA SFT 的正式 validation baseline。

## Artifacts

Predictions:

`outputs/base_validation_predictions_full.jsonl`

Evaluator:

`evaluate.py`

Saved-prediction evaluator:

`evaluate_saved_predictions.py`


# Experiment B1 — Fixed 3-shot Base Model

## Setup

模型：Qwen/Qwen2.5-0.5B-Instruct

训练：无

验证集样本数：1075

Demonstrations：
- train[1824]
- train[4012]
- train[2286]

生成设置：
- greedy decoding
- do_sample=False
- max_new_tokens=256

## Results

| Metric | Result |
|---|---:|
| JSON validity | 63.1628% |
| Schema validity | 5.1163% |
| Span-text consistency | 0.0000% |
| Strict Micro Precision | 0.0000 |
| Strict Micro Recall | 0.0000 |
| Strict Micro F1 | 0.0000 |
| Strict Macro F1 | 0.0000 |
| Surface-Type Precision | 0.0560 |
| Surface-Type Recall | 0.0686 |
| Surface-Type F1 | 0.0617 |

## Observation

固定 3-shot 的上下文示例并没有提升
Qwen2.5-0.5B-Instruct 在该任务上的表现。

与 B0 相比，JSON 格式合法率、Schema 合法率以及
Surface-Type F1 均有所下降。

模型仍然频繁生成不属于 CLUENER 固定标签集合的语义类别，
例如 `person`、`location`、`verb` 和 `team`。

在 JSON 格式合法的预测结果中，大约 75% 的预测实体使用了
十个合法 CLUENER 标签之外的实体类型。

Span-text consistency 仍然为 0%，说明这些 demonstrations
并没有使模型学会可靠地生成实体对应的绝对字符位置。

该实验表明，在当前模型与提示词配置下，
固定 3-shot in-context learning 不足以让
Qwen2.5-0.5B-Instruct 学会任务特定的实体类别体系，
也不足以使其稳定遵循 span-aware 的结构化输出协议。


# Experiment S1 — LoRA SFT Baseline

## 实验设置

模型：
Qwen/Qwen2.5-0.5B-Instruct

训练数据：
9673 条 CLUENER project-train 样本

验证集：
1075 条样本

LoRA 配置：
- r = 8
- alpha = 16
- dropout = 0.05
- target modules = q_proj, v_proj
- 可训练参数量 = 540,672（占总参数量 0.1093%）

训练配置：
- epochs = 3
- batch size = 8
- learning rate = 2e-4
- optimizer = AdamW
- weight decay = 0
- max length = 512
- 使用 BF16 autocast
- seed = 42

## 训练损失

| Epoch | Train Loss |
|---|---:|
| 1 | 0.197499 |
| 2 | 0.121982 |
| 3 | 0.101447 |

## 验证集结果

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| JSON validity | 99.9070% | 99.9070% | 100.0000% |
| Schema validity | 94.1395% | 95.8140% | 95.9070% |
| Span-text consistency | 35.3029% | 44.0412% | 50.8377% |
| Strict Micro F1 | 0.2318 | 0.3016 | **0.3606** |
| Strict Macro F1 | 0.2225 | 0.2951 | **0.3647** |
| Surface-Type F1 | 0.6526 | 0.6771 | **0.6988** |

最佳验证集 checkpoint：
`checkpoints/s1_lora/epoch_3`

## 实验观察

与 zero-shot baseline 和固定 3-shot baseline 相比，
LoRA SFT 显著提升了模型在各项任务相关能力上的表现。

模型很快学会了遵循结构化输出要求：
经过 1 个 epoch 后，JSON 合法率已经接近 100%，
Schema 合法率也超过了 94%。

Surface-Type F1 在第 1 个 epoch 后就达到了 0.6526，
之后提升速度相对放缓，到第 3 个 epoch 时达到 0.6988。
这说明实体识别能力以及 CLUENER 特定实体类别的映射关系，
主要在训练早期就已经被模型学会。

相比之下，Span-text consistency 从 35.30% 持续提升到 50.84%。
因此，即使后期 Surface-Type F1 的提升幅度已经变小，
Strict Span F1 仍然保持了较为明显的增长。

这表明，在当前 span-aware 生成式任务设计下，
绝对字符位置（absolute character offset）的生成
仍然是模型当前最主要的性能瓶颈。

## Analyze S1
诊断span错误类型。

1、先按 (text,type) 判断“语义上至少像是在说同一种实体 mention”，再用位置排序建立一个不依赖误差最小化的确定性配对，最后定义并计算最大边界误差，统计误差为0,1,2,大于2的比例，发现绝大多数的误差都比较小，因此前面实验中得到的50.84%的span-text consistency并不严重，也就是说如果放宽一点标准，模型基本上知道预测出来的实体在哪里，而不是只知道50.84%。

2、进一步统计span左端点和右端点误差的联合分布，发现最多的几种错误是整体左移或者右移1-2个字符，即实体长度和粗位置基本知道，只是整个 absolute index 偏了。

3、将每个文本分为三段，分别计算三段的span精确率和平均最大误差，发现规律“实体越靠后，absolute offset 越容易漂”

4、将实体按长度分类，分别计算不同长度的实体span精确率和平均最大误差，发现规律“1–8 字符差不多，但超长实体（≥9）明显困难”

5、将实体分为只出现一次的实体和repeated mention实体，分别计算二者的精确率，发现重复实体的精确率稍低，但是由于重复实体比较少，因此说明不了什么

6、经过上述5点的统计计算，初步确定下一步实验设计：LLM负责语义理解 + 类型判断 + 粗定位；Python负责精确字符串定位

# A1 — Deterministic Span Alignment

基于 checkpoint：
S1 epoch 3

方法：
对于每一个预测实体，首先在原始文本中查找 predicted surface 的所有精确出现位置。

如果该 surface 在原文中只出现一次，则直接使用该 occurrence 对应的精确 span。

如果该 surface 在原文中出现多次，则使用模型原始生成的粗略 predicted span 作为位置 anchor，选择与其距离最近的 occurrence，并将该 occurrence 的 start/end 作为最终 span。

该过程只使用原始输入文本和模型预测结果，不使用 gold annotation，也不重新训练模型。

结果：

| Metric | S1 | S1 + A1 |
|---|---:|---:|
| Schema validity | 95.9070% | 99.5349% |
| Span-text consistency | 50.8377% | 99.3593% |
| Strict Micro F1 | 0.3606 | 0.6970 |
| Strict Macro F1 | 0.3647 | 0.6965 |
| Surface-Type F1 | 0.6988 | 0.6988 |

实验观察：

Deterministic span alignment 几乎消除了 Surface-Type F1（0.6988）与 Strict Span F1（0.6970）之间的差距。

这表明，在 S1 已经正确预测出实体 surface 和 type 的情况下，strict-span 指标上的大量额外性能损失主要来自 absolute character offset 的生成误差，而不是 surface/type 层面的错误。

进一步的 span error analysis 表明，大多数 offset 错误属于较小范围的位置偏移，因此模型生成的粗略 span 仍然具有有效的位置信息。对于 repeated mention，可以利用该粗略 span 作为 anchor，在多个候选 occurrence 中进行消歧。


# Experiment S2 — Occurrence-aware LoRA SFT

## 实验动机

S1 使用如下 span-aware 输出表示：

```json
{
  "entities": [
    {
      "text": "...",
      "type": "...",
      "start": 0,
      "end": 3
    }
  ]
}
```

虽然 S1 已经能够较好地学习实体识别和类别判断
（Surface-Type F1 = 0.6988），但直接生成绝对字符位置
（absolute character offset）仍然是一个主要瓶颈：

- Strict Micro F1 = 0.3606
- Span-text consistency = 50.8377%

A1 表明，在不改变 Surface-Type F1 的情况下，通过将模型预测出的实体 surface
确定性地对齐回原始文本，可以将 Strict Micro F1 提升至 0.6970。

这一结果说明，直接生成绝对字符位置可能并不适合当前的生成式语言模型。

因此，S2 将 `(start, end)` 替换为 occurrence index：

```json
{
  "entities": [
    {
      "text": "...",
      "type": "...",
      "occurrence": 2
    }
  ]
}
```

`occurrence` 从 1 开始计数，表示 predicted surface 在原始文本中的第几次出现
对应当前实体 mention。

在推理阶段，`(text, occurrence)` 会通过确定性方法重新转换为
`(start, end)`，因此最终评测仍然使用与 S1 完全相同的 strict span 指标。


## 表示方式验证

正式训练之前，首先验证 occurrence representation 是否能够实现无损转换：

```text
(start, end)
    ->
occurrence
    ->
(start, end)
```

### Train

- Samples: 9673
- Entities: 21567
- Conversion failures: 0
- Reconstruction failures: 0
- Lossless reconstruction rate: 100%

Occurrence 分布：

| Occurrence | Count |
|---|---:|
| 1 | 20798 |
| 2 | 732 |
| 3 | 37 |

### Validation

- Samples: 1075
- Entities: 2404
- Conversion failures: 0
- Reconstruction failures: 0
- Lossless reconstruction rate: 100%

Occurrence 分布：

| Occurrence | Count |
|---|---:|
| 1 | 2312 |
| 2 | 88 |
| 3 | 4 |

因此，在当前 project train 和 validation split 上，
occurrence-aware representation 能够完整保留 mention-level 信息，
不存在由表示方式转换本身造成的信息损失。


## 实验设置

模型：

Qwen/Qwen2.5-0.5B-Instruct

训练数据：

9673 条 CLUENER project-train 样本

验证集：

1075 条样本

LoRA：

- r = 8
- alpha = 16
- dropout = 0.05
- target modules = q_proj, v_proj
- 可训练参数量 = 540,672（0.1093%）

训练：

- epochs = 3
- batch size = 8
- learning rate = 2e-4
- optimizer = AdamW
- weight decay = 0
- max length = 512
- BF16 autocast
- seed = 42

所有训练超参数均与 S1 保持一致。

S2 相比 S1 的主要实验变量是输出表示方式：

S1：

`text / type / start / end`

S2：

`text / type / occurrence`


## 训练损失

| Epoch | Train Loss |
|---|---:|
| 1 | 0.096001 |
| 2 | 0.060123 |
| 3 | 0.048798 |

由于 S1 与 S2 的 target token sequence 不同，因此两种表示方式下的
loss 绝对值不能直接进行横向比较。


## 验证集结果

模型生成 occurrence-aware prediction 后，
首先通过确定性方法将其重新构造为标准的
`(text, type, start, end)` 实体，
随后使用与 S1 完全相同的 strict evaluator 进行评测。

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| JSON validity | 100.0000% | 100.0000% | 100.0000% |
| Schema validity | 97.4884% | 98.5116% | **98.6977%** |
| Span-text consistency | 100.0000% | 100.0000% | 100.0000% |
| Strict Micro F1 | 0.6807 | 0.7069 | **0.7153** |
| Strict Macro F1 | 0.6762 | 0.7074 | **0.7139** |
| Surface-Type F1 | 0.6847 | 0.7098 | **0.7184** |

最佳 validation checkpoint：

`checkpoints/s2_occurrence_lora/epoch_3`


## 与 S1 的对比

| Experiment | Representation | Strict Micro F1 | Surface-Type F1 |
|---|---|---:|---:|
| S1 epoch 3 | absolute span | 0.3606 | 0.6988 |
| S1 + A1 | absolute span + deterministic alignment | 0.6970 | 0.6988 |
| S2 epoch 3 | occurrence + deterministic reconstruction | **0.7153** | **0.7184** |

与直接生成 absolute span 相比，
occurrence-aware training 显著提升了 strict span performance：

`0.3606 -> 0.7153`

S2 也略高于 S1 + A1：

`0.6970 -> 0.7153`

更值得注意的是，Surface-Type F1 同样有所提升：

`0.6988 -> 0.7184`

由于 deterministic span reconstruction 本身不会改善 Surface-Type F1，
因此这一现象说明：去除 absolute-offset generation 的负担后，
模型可能也能够更有效地学习实体发现和类别判断。

不过，目前该结果只来自一个 training seed，
因此应将其视为当前实验配置下的观察结果，而不是一般性结论。


## Occurrence 分布诊断

Validation gold occurrence 分布：

| Occurrence | Count |
|---|---:|
| 1 | 2312 |
| 2 | 88 |
| 3 | 4 |

S2 epoch 3 的 predicted occurrence 分布：

| Occurrence | Count |
|---|---:|
| 1 | 2136 |
| 2 | 33 |
| 3 | 1 |

模型明显偏向预测 `occurrence=1`。

Gold 中的 non-first mentions：

`92`

模型预测出的 non-first mentions：

`30`

这说明模型对后续重复 mention 存在明显的漏预测现象。


## Repeated-Surface 诊断

### Repeated-surface mentions 的端到端结果

- TP = 96
- FP = 35
- FN = 94
- Precision = 0.7328
- Recall = 0.5053
- F1 = 0.5981

### 在 Surface-Type 已正确匹配条件下的 Occurrence 准确率

在 `(text, type)` 已经正确匹配的 repeated-surface mentions 中：

- Surface-Type matched = 100
- Correct occurrence = 96
- Conditional occurrence accuracy = 0.9600

这说明，当模型已经成功生成正确的 repeated mention 时，
occurrence number 本身通常能够被正确预测。

因此，当前 repeated mention 的主要错误并不是将 occurrence 1 与 occurrence 2
相互混淆，而是部分 repeated mentions 根本没有被模型生成出来。


### Non-first occurrences

对于 gold 中满足 `occurrence > 1` 的 mentions：

- Gold = 92
- Predicted = 30
- Exact TP = 21
- Precision = 0.7000
- Recall = 0.2283

模型在生成后续 occurrence 时表现较为保守：
一旦模型预测了 non-first occurrence，其 precision 尚可，
但 recall 很低，说明大量后续 repeated mentions 被漏掉。


## Repeated Group Analysis

将 repeated `(surface, type)` group 分为两类：

- `ALL`：该 surface 在原始文本中的所有 occurrence 都被标注为实体。
- `PARTIAL`：只有部分 occurrence 被标注为实体。

### ALL

- Groups = 71
- Exact group matches = 20
- Exact group accuracy = 0.2817
- Gold mentions = 143
- Predicted mentions = 77
- TP = 77
- Precision = 1.0000
- Recall = 0.5385
- F1 = 0.7000

这些 gold groups 平均包含约：

`143 / 71 = 2.01`

个 mention，而模型平均只预测：

`77 / 71 = 1.08`

个 mention。

因此，即使在所有 repeated occurrences 都被一致标注的 `ALL` 情形下，
模型仍然经常只生成一个 entity object，而没有生成全部需要的 mentions。


### PARTIAL

- Groups = 44
- Exact group matches = 14
- Exact group accuracy = 0.3182
- Gold mentions = 47
- Predicted mentions = 26
- TP = 19
- Precision = 0.7308
- Recall = 0.4043
- F1 = 0.5205

部分标注确实会进一步增加 repeated-mention 任务的难度，
但它并不能完全解释当前较低的 recall，
因为在所有 occurrence 都被完整标注的 `ALL` 子集中，
recall 也只有 0.5385。


## 主要实验观察

S2 的实验结果支持在生成式 NER 中使用 occurrence-aware output representation。

将 absolute character offset 替换为 occurrence index 后，
避免了模型直接进行精确字符计数，
Strict Micro F1 从 0.3606 提升到 0.7153。

不过，进一步的 error analysis 暴露出了新的限制。

当模型已经成功生成正确的 repeated surface-type mention 时，
其 occurrence number 通常能够预测正确，
条件 occurrence accuracy 达到 96%。

但是，模型经常没有为后续 occurrence 生成额外的 entity object。

因此，S2 当前 repeated-mention 场景下最主要的问题是：

`repeated-mention under-generation`

而不是：

`occurrence-index confusion`

这一结果进一步启发我们考虑 grouped occurrence representation：
对于同一个 `(text, type)`，将所有被标注的 occurrence index
放在同一个 entity object 中，例如：

```json
{
  "text": "费内巴切",
  "type": "organization",
  "occurrences": [1, 2]
}
```

这种表示方式可能在保留 mention identity 的同时，
避免要求模型生成多个内容几乎完全相同的 JSON entity object。

项目的 official test split 仍保持封存，尚未使用。


# Experiment S3 — Grouped Occurrence Representation

## 实验动机

S2 将每一个被标注的 mention 表示为一个独立的 entity object：

```json
{"text":"A","type":"organization","occurrence":1}
{"text":"A","type":"organization","occurrence":2}
```

Error analysis 表明，当模型已经成功生成一个 repeated mention 时，
其 occurrence index 通常能够预测正确，
conditional occurrence accuracy 达到 96%。

但是，模型经常无法继续生成同一 `(text, type)` 下后续的 repeated mentions。
因此，S2 的主要 repeated-mention 问题并不是 occurrence index 预测错误，
而是 repeated mention 本身没有被生成。

S3 因此进一步修改输出表示方式，将相同 `(text, type)` 下
所有被标注的 occurrences 合并到同一个 entity object 中：

```json
{
  "text":"A",
  "type":"organization",
  "occurrences":[1,2]
}
```

这一设计的目标是减少生成多个近乎重复 entity objects
所带来的 repeated-mention under-generation，
同时继续完整保留 mention-level 信息。


## 表示方式验证

在训练之前，对 grouped occurrence representation 进行了
`span -> grouped occurrence -> span` 的无损重构验证。

Grouped representation 在 project train 和 validation split 上
均可以实现 100% 无损转换。

### Train

- Samples: 9673
- Mentions: 21567
- Grouped entity objects: 21009
- Repeated groups: 547
- Reconstruction failures: 0
- Lossless reconstruction rate: 100%

Group size 分布：

| Mentions per group | Count |
|---|---:|
| 1 | 20462 |
| 2 | 536 |
| 3 | 11 |

原始 21567 个 mention 被压缩为 21009 个 grouped entity objects。

其中绝大多数 group 仍然只包含一个 mention，
因此 S3 实际上只修改了 repeated mention 对应的少量训练目标。


### Validation

- Samples: 1075
- Mentions: 2404
- Grouped entity objects: 2329
- Repeated groups: 74
- Reconstruction failures: 0
- Lossless reconstruction rate: 100%

Group size 分布：

| Mentions per group | Count |
|---|---:|
| 1 | 2255 |
| 2 | 73 |
| 3 | 1 |

因此，S3 并没有改变任务本身的 mention-level 信息，
而只是改变 repeated mentions 的序列化方式。


## 实验设置

所有训练超参数均与 S2 保持一致。

模型：

`Qwen/Qwen2.5-0.5B-Instruct`

LoRA：

- r = 8
- alpha = 16
- dropout = 0.05
- target modules = `q_proj`, `v_proj`
- trainable parameters = 540,672

训练设置：

- train samples = 9673
- validation samples = 1075
- epochs = 3
- batch size = 8
- learning rate = 2e-4
- optimizer = AdamW
- weight decay = 0
- max length = 512
- BF16 autocast
- seed = 42

唯一具有实质性的改动是输出表示方式：

S2：

`text / type / occurrence`

S3：

`text / type / occurrences[]`


## 训练损失

| Epoch | Train Loss |
|---|---:|
| 1 | 0.097146 |
| 2 | 0.060820 |
| 3 | 0.049225 |

训练损失持续下降，未观察到异常。

由于 S2 与 S3 的 target token sequence 不同，
二者的 absolute training loss 不应直接进行数值比较。


## 验证集结果

Grouped occurrence output 在推理后被确定性重构为标准的：

`text / type / start / end`

随后继续使用与 S1、S2 相同的 strict evaluator 进行评测。

| Metric | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---:|---:|---:|
| JSON validity | 99.9070% | 100.0000% | 100.0000% |
| Schema validity | 97.7674% | 98.9767% | **99.3488%** |
| Span-text consistency | 100.0000% | 100.0000% | 100.0000% |
| Strict Micro F1 | 0.6627 | 0.6698 | **0.7153** |
| Strict Macro F1 | 0.6573 | 0.6688 | **0.7149** |
| Surface-Type F1 | 0.6672 | 0.6722 | **0.7171** |

最佳 checkpoint：

`checkpoints/s3_grouped_occurrence_lora/epoch_3`


## 与 S2 的整体对比

| Experiment | Strict P | Strict R | Strict F1 | Surface-Type F1 |
|---|---:|---:|---:|---:|
| S2 | 0.7626 | **0.6735** | **0.7153** | **0.7184** |
| S3 | **0.7777** | 0.6622 | **0.7153** | 0.7171 |

S2 和 S3 的整体 Strict Micro F1 完全相同，均为 0.7153。

相比 S2，S3 的 precision 更高，但 recall 略低：

`P: 0.7626 -> 0.7777`

`R: 0.6735 -> 0.6622`

Surface-Type F1 也基本保持不变：

`0.7184 -> 0.7171`

因此，仅从整体 validation 指标来看，
S3 并没有带来进一步提升。

不过，S3 的设计目标本身就是针对 repeated mentions，
而 validation 中 grouped representation 实际只合并了 75 个 mention。
因此，仅使用 overall F1 不足以判断 S3 是否实现了其设计目标。


## Repeated-Group Analysis

为了进一步分析 repeated mentions，
将 validation 中的 repeated-surface groups 按如下规则划分：

### ALL

对于某个 `(surface, type)` group，
该 surface 在原文中的所有字符串 occurrence
都对应 gold 中该 type 的 mention。

### PARTIAL

对于某个 `(surface, type)` group，
只有部分字符串 occurrence
对应 gold 中该 type 的 mention。

需要注意的是：

`PARTIAL` 只是一个基于字符串 occurrence 的操作性定义，
并不意味着这些样本一定存在漏标或 annotation noise。

后续人工审计表明，
PARTIAL group 可能同时包含：

1. 真正的 annotation omission；
2. 同一 surface 在不同位置具有不同语义；
3. surface 只是更长实体或词语的一部分；
4. mention boundary 不同；
5. 无法明确判断的模糊情况。

因此，PARTIAL 不能直接等同于 noisy-label subset。


### ALL repeated groups

| Metric | S2 | S3 |
|---|---:|---:|
| Exact group accuracy | 0.2817 | **0.3944** |
| Predicted mentions | 77 | **84** |
| Recall | 0.5385 | **0.5874** |
| F1 | 0.7000 | **0.7401** |

在 ALL repeated groups 上，
S3 带来了较为明确的改善。

Exact group accuracy：

`0.2817 -> 0.3944`

Recall：

`0.5385 -> 0.5874`

F1：

`0.7000 -> 0.7401`

Gold 中平均每个 group 包含：

`143 / 71 ≈ 2.01`

个 mention。

S2 平均每个 group 预测：

`77 / 71 ≈ 1.08`

个 mention。

S3 提升为：

`84 / 71 ≈ 1.18`

个 mention。

因此，将多个 repeated mentions 合并为一个
`occurrences[]` list，
确实使模型更容易生成后续 occurrences。

这一结果支持此前的假设：

要求模型生成多个几乎完全相同的 entity objects，
会加重 repeated-mention under-generation。

不过，S3 平均每组仍然只生成约 1.18 个 mention，
距离 gold 的约 2.01 个仍有明显差距。

因此，grouped representation 只能认为
**缓解了（alleviated）**
repeated-mention under-generation，
而不能认为已经解决这一问题。


### PARTIAL repeated groups

| Metric | S2 | S3 |
|---|---:|---:|
| Exact group accuracy | **0.3182** | 0.2273 |
| Precision | **0.7308** | 0.6333 |
| Recall | 0.4043 | 0.4043 |
| F1 | **0.5205** | 0.4935 |

在 PARTIAL groups 上，
S3 生成了更多 predicted occurrences：

S2：

- Predicted mentions = 26
- TP = 19

S3：

- Predicted mentions = 30
- TP = 19

按照 benchmark gold 计算，
新增预测没有增加 TP，
因此 precision 从 0.7308 下降到 0.6333。


## PARTIAL Groups 人工审计

为了判断 PARTIAL subset 是否可以直接理解为 annotation noise，
对 validation 中全部 44 个 PARTIAL repeated groups
进行了人工语义审计。

人工将其分为三类：

### A — likely missing annotation / annotation noise

未标注的 occurrence 在上下文中仍明显属于相同实体类型，
很可能属于漏标。

### B — legitimate partial annotation

虽然字符串 surface 再次出现，
但该 occurrence 在语义或 mention boundary 上
并不应该被标为当前 type。

常见情况包括：

- 同一 surface 在不同位置具有不同语义；
- surface 是更长实体的一部分；
- surface 是普通词或其他实体中的子串。

### C — ambiguous

仅根据当前文本无法可靠判断是否应该标注。

人工审计结果：

| Category | Groups | Ratio |
|---|---:|---:|
| A — likely missing annotation | 14 | 31.8% |
| B — legitimate partial | 22 | 50.0% |
| C — ambiguous | 8 | 18.2% |

这一结果说明：

PARTIAL subset 中确实存在一定数量疑似漏标的情况，
但 PARTIAL 并不能整体视为错误数据。

实际上，一半的 PARTIAL groups 可以较明确地解释为
合理的 partial annotation。


## 对 S3 PARTIAL False Positives 的进一步检查

S3 在 PARTIAL groups 中：

- Predicted mentions = 30
- TP = 19

因此 benchmark 将其中 11 个 occurrence 判定为 false positives。

进一步结合人工审计后发现：

| 所属人工类别 | S3 extra occurrences |
|---|---:|
| A — likely missing annotation | 7 |
| B — legitimate partial | 2 |
| C — ambiguous | 2 |

也就是说，S3 的 11 个 benchmark false positives 中，
有 7 个发生在人工判断为 likely missing annotation 的 groups 中。

因此，S3 在 PARTIAL subset 上观察到的 precision drop
不能全部解释为模型真实的 over-generation。

其中一部分预测可能在语义上是合理的，
只是因为 gold annotation 不完整而被 benchmark 计为 false positive。

需要强调的是：

这一人工审计只能作为误差分析，
不能替代官方 gold，也不应直接修改 benchmark 分数。

因此，S3 的官方 PARTIAL precision 仍然保持为 0.6333，
但在解释该结果时需要考虑 annotation incompleteness。


## 对 ALL / PARTIAL 划分的重新理解

最初的分析容易将：

`ALL`

理解为“标注完整”，将：

`PARTIAL`

理解为“存在漏标”。

人工审计表明，这种理解过于简单。

更准确地说：

### ALL

所有匹配到的字符串 occurrence
都恰好对应当前 `(surface, type)` 的 gold mentions。

因此，该 subset 中字符串 occurrence 与 mention identity
之间关系较为干净。

### PARTIAL

字符串 occurrence 与 gold mention 并非一一对应。

造成这一现象的原因可能包括：

- annotation omission；
- semantic ambiguity；
- substring overlap；
- mention-boundary difference。

因此，ALL subset 是评估 grouped representation
是否缓解 repeated-object under-generation
相对更干净的诊断集合。

而 PARTIAL subset 更适合用于研究
representation 与数据标注特性之间的相互作用，
不应被简单解释为模型错误。


## 结论

S3 对 grouped-occurrence hypothesis 提供了明确但有限的支持。

在整体 validation 上：

`Strict Micro F1: 0.7153 -> 0.7153`

S3 并没有进一步超过 S2。

但是，在更直接对应 S3 设计目标的 ALL repeated-group subset 上，
S3 表现出一致改善：

- Exact group accuracy: `0.2817 -> 0.3944`
- Recall: `0.5385 -> 0.5874`
- F1: `0.7000 -> 0.7401`
- Predicted mentions: `77 -> 84`

因此，有证据支持：

将相同 `(text, type)` 下的 repeated mentions
合并为一个 `occurrences[]` list，
能够缓解由于重复生成近似 entity objects
造成的 repeated-mention under-generation。

不过，这一问题并没有被彻底解决。

另一方面，S3 在 PARTIAL repeated groups 上的
benchmark precision 出现下降。

人工审计进一步表明，
PARTIAL subset 本身并不等价于 annotation noise：

- 31.8% 的 groups 疑似存在漏标；
- 50.0% 可以解释为合理的 partial annotation；
- 18.2% 无法可靠判断。

同时，S3 在 PARTIAL subset 中新增的 11 个 benchmark false positives 中，
有 7 个发生在疑似漏标的 groups 上。

因此，PARTIAL subset 上观察到的性能下降
不能全部归因于模型真实的 over-generation，
其中一部分可能来自 benchmark annotation incompleteness。

综合来看：

S3 没有提升整体 validation F1，
但成功验证了 grouped representation
对 fully matched repeated-surface cases 的针对性收益。

S2 仍然是当前更简单、直接的主要 occurrence-aware formulation；
S3 则作为一个重要的 representation ablation，
展示了 grouped repeated mentions 的收益，
同时也暴露了 repeated-surface evaluation
与实际 mention identity、数据标注完整性之间的复杂关系。
