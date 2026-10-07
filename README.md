# LoRA Generative NER：基于 Occurrence-aware Representation 的中文命名实体识别

本项目以 CLUENER2020 为 benchmark，探索使用小型指令模型
`Qwen2.5-0.5B-Instruct` + LoRA 进行生成式中文命名实体识别（NER）。

项目重点并不是单纯追求更高的 F1，
而是围绕生成式 NER 中一个实际问题展开：

> 模型能够识别实体，但直接生成精确字符位置 `start/end`
> 时容易出现字符计数偏移。

围绕这一问题，项目经历了：

`baseline → LoRA SFT → error analysis → deterministic alignment → occurrence-aware representation → repeated-mention ablation → annotation audit → bootstrap → multi-seed robustness → held-out final evaluation`

最终主方法采用 **Occurrence-aware Representation**，
在 3 个 training seeds 的 held-out evaluation 上取得：

- **Strict Micro F1: 0.7238 ± 0.0117**
- **Surface-Type F1: 0.7327 ± 0.0146**

---

## 1. Task

CLUENER2020 包含 10 类中文命名实体：

- `address`
- `book`
- `company`
- `game`
- `government`
- `movie`
- `name`
- `organization`
- `position`
- `scene`

项目内部统一使用 mention-level span 表示：

```json
{
  "text": "费内巴切",
  "type": "organization",
  "start": 14,
  "end": 17
}
```

其中：

- `start/end` 为从 0 开始的字符索引；
- `end` 为闭区间；
- 主评测采用严格 `(start, end, type)` 匹配。

数据划分：

| Split | Samples | 来源 |
|---|---:|---|
| Train | 9673 | official train 的 90% |
| Validation | 1075 | official train 的 10% |
| Final held-out | 1343 | official dev |

Train / Validation 使用固定 split seed = 42。

Official dev 在项目早期曾用于数据统计和少量样本观察，
但在实验协议冻结后不再用于 prompt、representation、
hyperparameter 或 checkpoint selection，
最终作为 held-out final evaluation split。

---

## 2. Baseline：小模型直接生成 span 的问题

最初使用 `Qwen2.5-0.5B-Instruct` 进行 zero-shot generative NER。

模型需要直接生成：

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

Zero-shot baseline 表现很差：

| Metric | B0 Zero-shot |
|---|---:|
| JSON validity | 80.65% |
| Schema validity | 44.09% |
| Strict Micro F1 | 0.0007 |
| Surface-Type F1 | 0.0930 |

固定 3-shot ICL 后性能没有改善，
因此后续转向 LoRA SFT。

---

## 3. S1：Absolute-span LoRA SFT

使用 Qwen2.5-0.5B-Instruct + LoRA：

```text
LoRA rank       = 8
LoRA alpha      = 16
LoRA dropout    = 0.05
target modules  = q_proj, v_proj
trainable params = 540,672
```

训练配置：

```text
epochs       = 3
batch size   = 8
learning rate= 2e-4
optimizer    = AdamW
weight decay = 0
max length   = 512
BF16
```

S1 继续使用：

`text / type / start / end`

作为训练目标。

最佳 validation 结果：

| Metric | S1 |
|---|---:|
| JSON validity | 100.00% |
| Schema validity | 95.91% |
| Strict Micro F1 | 0.3606 |
| Surface-Type F1 | 0.6988 |

这里出现了明显异常：

> Surface-Type F1 已接近 0.70，
> 但 Strict F1 只有 0.36。

说明模型已经具备较强的实体语义识别能力，
真正的主要瓶颈可能来自 span offset generation。

---

## 4. Error Analysis：Absolute Offset 是主要瓶颈

对 S1 epoch-3 validation prediction 进行误差分析。

在 `text + type` 已经正确匹配的实体中：

| 最大 offset error | 比例 |
|---|---:|
| 0 | 51.62% |
| ≤ 1 | 84.76% |
| ≤ 2 | 93.91% |

大量错误并不是实体识别错误，
而是整体偏移 1～2 个字符。

同时，越靠近文本后部，offset error 越明显：

```text
front  exact ≈ 71%
middle exact ≈ 37%
back   exact ≈ 20%
```

这一现象支持：

> autoregressive language model 对实体语义本身掌握得较好，
> 但不擅长直接进行精确 absolute character counting。

---

## 5. A1：Deterministic Span Alignment

基于上述观察，设计确定性后处理：

1. 在原文中搜索模型预测的 entity surface；
2. 若只出现一次，直接使用该 span；
3. 若出现多次，使用模型原始 offset 选择距离最近的 occurrence；
4. 不修改 entity text 或 type。

A1 不重新训练模型，只修改 span reconstruction。

结果：

| Method | Strict Micro F1 | Surface-Type F1 |
|---|---:|---:|
| S1 | 0.3606 | 0.6988 |
| S1 + A1 | **0.6970** | 0.6988 |

Strict F1 几乎追平 Surface-Type F1。

这个实验进一步验证：

> S1 的主要 strict-span 损失确实来自 absolute offset generation，
> 而不是实体语义识别本身。

---

## 6. S2：Occurrence-aware Representation

A1 仍然属于推理阶段 patch。

S2 进一步从训练目标本身移除 absolute offset，
将：

```json
{
  "text": "费内巴切",
  "type": "organization",
  "start": 32,
  "end": 35
}
```

改为：

```json
{
  "text": "费内巴切",
  "type": "organization",
  "occurrence": 2
}
```

`occurrence=2` 表示该 surface 在原文中的第二次出现。

推理阶段再确定性恢复：

`(text, occurrence) -> (start, end)`

该表示在 train / validation 上均经过完整验证，
能够实现 **100% lossless reconstruction**。

### Validation

| Method | Strict Micro F1 | Surface-Type F1 |
|---|---:|---:|
| S1 | 0.3606 | 0.6988 |
| S1 + A1 | 0.6970 | 0.6988 |
| **S2** | **0.7153** | **0.7184** |

S2 不仅解决了 span reconstruction，
Surface-Type F1 也略有提高。

这表明去除 absolute-offset generation burden
可能同时改善了模型对核心 NER 任务的学习。

---

## 7. Repeated Mention Analysis

Occurrence-aware representation 引出了一个新的问题：

> 当同一个 surface 在原文中出现多次时，
> 模型是否真的能够学习 occurrence number？

分析发现：

```text
Gold occurrence > 1:      92
Predicted occurrence > 1: 30
Exact TP:                 21
```

但在 repeated mention 的 `text + type`
已经匹配正确的情况下：

```text
Occurrence conditional accuracy = 96%
```

因此真正的问题不是：

> 模型不会区分 occurrence=1 和 occurrence=2。

而是：

> 模型经常没有继续生成后续 repeated mention。

---

## 8. S3：Grouped Occurrence Ablation

为了减少多个近乎相同 entity objects 的重复生成，
S3 将：

```json
{"text":"A","type":"organization","occurrence":1}
{"text":"A","type":"organization","occurrence":2}
```

改为：

```json
{
  "text":"A",
  "type":"organization",
  "occurrences":[1,2]
}
```

该表示同样可以 100% lossless reconstruction。

### Overall Validation

| Method | Strict F1 | Surface-Type F1 |
|---|---:|---:|
| S2 | 0.7153 | 0.7184 |
| S3 | 0.7153 | 0.7171 |

整体性能基本没有变化。

### Fully Matched Repeated Groups

在所有字符串 occurrence 均对应 gold mention 的 `ALL` subset 上：

| Metric | S2 | S3 |
|---|---:|---:|
| Exact Group Accuracy | 0.2817 | **0.3944** |
| Recall | 0.5385 | **0.5874** |
| F1 | 0.7000 | **0.7401** |

说明 grouped representation
确实能够一定程度缓解 repeated-object under-generation。

但这一局部收益没有转化为整体 validation F1 的提升。

---

## 9. PARTIAL Repeated Groups 与 Annotation Audit

Repeated-surface cases 中还存在另一类情况：

同一个字符串在原文中出现多次，
但 gold 只将其中部分 occurrence 标为当前 entity type。

最初容易将这类 `PARTIAL` group 直接理解为 annotation noise。

因此对 validation 中全部 44 个 PARTIAL groups
进行了人工语义审计。

结果：

| Category | Groups | Ratio |
|---|---:|---:|
| Likely missing annotation | 14 | 31.8% |
| Legitimate partial | 22 | 50.0% |
| Ambiguous | 8 | 18.2% |

说明：

> repeated surface ≠ repeated mention。

PARTIAL cases 同时包含：

- annotation omission；
- semantic ambiguity；
- substring overlap；
- mention-boundary difference。

因此不能直接将 PARTIAL subset 等同于 noisy-label subset。

这一分析也说明，
单纯的字符串 occurrence matching
与真实 mention identity 之间仍存在差异。

---

## 10. Paired Bootstrap

由于 ALL repeated subset 只有 71 个 groups，
进一步进行了 10,000 次 paired bootstrap。

### Overall Strict F1

```text
Delta(S3-S2) = +0.000081

95% CI:
[-0.012675, 0.012472]

Bootstrap P(Delta > 0):
0.4974
```

Overall 上没有观察到稳定的 S2 / S3 差异。

### ALL Repeated Groups

| Metric | Δ(S3-S2) | 95% Bootstrap CI | P(Δ>0) |
|---|---:|---:|---:|
| Recall | +0.0490 | [-0.0282, 0.1268] | 0.8658 |
| F1 | +0.0401 | [-0.0244, 0.1050] | 0.8658 |
| Exact Group Accuracy | +0.1127 | [-0.0282, 0.2535] | 0.9348 |

S3 在 repeated subset 上表现出正向趋势，
但由于 subset 较小，
95% bootstrap interval 仍覆盖 0。

因此该结果被视为有方向性的局部收益，
而不是强统计结论。

---

## 11. Multi-seed Robustness

固定 data split seed = 42，
改变 training seed：

```text
42 / 43 / 44
```

每个 seed 均按 validation Strict Micro F1
选择 best checkpoint。

### Overall Validation

| Formulation | Strict Micro F1 | Surface-Type F1 |
|---|---:|---:|
| **S2** | **0.7242 ± 0.0077** | **0.7292 ± 0.0094** |
| S3 | 0.7195 ± 0.0044 | 0.7236 ± 0.0065 |

S2 在整体任务上的平均表现略高。

### ALL Repeated Groups

| Formulation | Recall | F1 | Exact Group Accuracy |
|---|---:|---:|---:|
| S2 | 0.5781 ± 0.0422 | 0.7288 ± 0.0324 | 0.3239 ± 0.0732 |
| **S3** | **0.5991 ± 0.0107** | **0.7492 ± 0.0084** | **0.3803 ± 0.0244** |

S3 在 repeated-group 指标上的平均值更高，
且跨 seed 波动更小。

但这一收益仍然局限于 repeated subset，
没有转化为整体 validation improvement。

因此最终选择：

- **S2：main formulation**
- **S3：targeted representation ablation**

不再继续设计 S4/S5。

---

## 12. Final Held-out Evaluation

完成 representation design、error analysis、
bootstrap 和 multi-seed robustness 后，
冻结模型与实验协议。

最终使用 S2 的三个 training seeds：

```text
42 / 43 / 44
```

在 project final held-out split
（official CLUENER dev，1343 samples）
上进行一次最终评测。

### Per-seed

| Seed | Strict P | Strict R | Strict F1 | Surface-Type F1 |
|---|---:|---:|---:|---:|
| 42 | 0.7686 | 0.6650 | 0.7131 | 0.7182 |
| 43 | 0.7621 | 0.7122 | 0.7363 | 0.7474 |
| 44 | 0.7662 | 0.6826 | 0.7220 | 0.7325 |

### Mean ± Sample Std

| Metric | Final Held-out |
|---|---:|
| Strict Precision | 0.7656 ± 0.0033 |
| Strict Recall | 0.6866 ± 0.0239 |
| **Strict Micro F1** | **0.7238 ± 0.0117** |
| Strict Macro F1 | 0.7203 ± 0.0156 |
| **Surface-Type F1** | **0.7327 ± 0.0146** |
| Schema Validity | 97.4187% ± 0.2275% |

Validation 上：

```text
Strict Micro F1 = 0.7242 ± 0.0077
```

Final held-out：

```text
Strict Micro F1 = 0.7238 ± 0.0117
```

平均差异仅为：

```text
-0.0004
```

没有观察到明显的 validation-to-test degradation。

---

## 13. Main Findings

本项目最终得到以下几个主要结论：

1. 对小型 generative LLM 而言，
   直接生成 absolute character offsets
   是中文 generative NER 的明显瓶颈。

2. Surface-Type evaluation 和 span-error analysis 表明，
   模型往往已经识别出了正确实体，
   但会在字符索引上发生小范围偏移。

3. Deterministic alignment 可以显著恢复 strict-span performance，
   说明大量错误来自 representation / decoding，
   而不是实体语义理解。

4. Occurrence-aware representation
   可以在保留 mention identity 的同时避免直接字符计数，
   并在 held-out evaluation 上取得：

   `Strict Micro F1 = 0.7238 ± 0.0117`

5. Grouped occurrence representation
   能改善 fully matched repeated-surface cases，
   但没有提高整体任务性能。

6. Repeated-surface statistics
   不能直接等同于 repeated-mention statistics；
   annotation audit 表明其中同时存在
   漏标、多义、substring overlap 和 boundary difference。

7. Paired bootstrap 和 multi-seed experiments
   分别用于检查 evaluation uncertainty
   和 training randomness。

---

## 14. Project Structure

主要文件：

```text
.
├── dataset_utils.py
├── data_process.py
│
├── preprocess_sft_sample.py
├── preprocess_occurrence_sft_sample.py
├── preprocess_grouped_occurrence_sft_sample.py
│
├── sft_data_collator.py
│
├── train_lora.py
├── train_lora_occurrence.py
├── train_lora_grouped_occurrence.py
│
├── run_baseline.py
├── run_few_shot_baseline.py
├── run_occurrence_validation.py
├── run_grouped_occurrence_validation.py
├── run_occurrence_final_test.py
│
├── evaluate.py
├── span_alignment.py
├── occurrence_reconstruction.py
├── grouped_occurrence_reconstruction.py
│
├── analyze_occurrence_predictions.py
├── analyze_repeated_groups.py
├── analyze_s3_repeated_groups.py
├── paired_bootstrap_s2_s3.py
│
├── experiments.md
└── README.md
```

具体实验过程、完整 per-class 结果和 error analysis
见：

`experiments.md`

---

## 15. Reproduction

环境示例：

```text
Python 3.12
PyTorch 2.8.0
Transformers 5.16.1
Datasets 5.0.1
PEFT 0.20.0
Accelerate 1.14.0
```

训练 S2：

```bash
HF_HUB_OFFLINE=1 python train_lora_occurrence.py --seed 42
```

Validation：

```bash
HF_HUB_OFFLINE=1 python run_occurrence_validation.py \
    --adapter_path checkpoints/s2_occurrence_lora/epoch_3 \
    --output_path outputs/s2_epoch3_validation_predictions.jsonl
```

Final held-out evaluation：

```bash
./run_s2_final_test.sh
```

---

## 16. Limitations

本项目仍存在以下限制：

- Base model 仅使用 Qwen2.5-0.5B-Instruct；
- 未进行大规模 hyperparameter search；
- multi-seed robustness 仅包含 3 个 training seeds；
- repeated-group subset 规模较小；
- official dev 在项目早期曾用于数据统计和少量样本观察，
  因此不属于严格意义上的 completely unseen test set；
- 项目关注 generative NER representation，
  未系统比较 BERT / CRF 等 discriminative NER systems。

这些限制均保留在最终报告中，
不再通过进一步 test-set tuning 修正。
