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
````

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
