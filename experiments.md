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
