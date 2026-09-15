import json


CLUENER_LABELS = {
    "address",
    "book",
    "company",
    "game",
    "government",
    "movie",
    "name",
    "organization",
    "position",
    "scene",
}

# 该函数尝试解析 raw_output 字符串为 JSON，并检查其是否符合预期的 schema。它返回一个字典，包含以下键：
# - "json_valid": 布尔值，表示 raw_output 是否是合法的 JSON
# - "schema_valid": 布尔值，表示 JSON 是否符合预期的 schema
# - "entities": 如果 schema_valid 为 True，则包含解析后的实体列表；否则为空列表
# - "span_text_consistency": 如果 schema_valid 为 True，则包含一个布尔值列表，表示每个实体的 text 是否与 source_text 中的 span 一致；否则为空列表
def parse_prediction(
    raw_output,
    source_text,
):

    result = {
        "json_valid": False,
        "schema_valid": False,
        "entities": [],
        "span_text_consistency": [],
    }

    # 第一步：是不是严格合法的 JSON
    try:
        data = json.loads(raw_output)

    except json.JSONDecodeError:
        return result

    result["json_valid"] = True

    # 第二步：顶层结构
    if not isinstance(data, dict):
        return result

    if set(data.keys()) != {"entities"}:
        return result

    if not isinstance(data["entities"], list):
        return result

    parsed_entities = []
    consistency = []

    # 第三步：逐个检查 entity
    for entity in data["entities"]:

        if not isinstance(entity, dict):
            return result

        required_keys = {
            "text",
            "type",
            "start",
            "end",
        }

        if set(entity.keys()) != required_keys:
            return result

        entity_text = entity["text"]
        entity_type = entity["type"]
        start = entity["start"]
        end = entity["end"]

        if not isinstance(entity_text, str):
            return result

        if entity_type not in CLUENER_LABELS:
            return result

        if type(start) is not int:
            return result

        if type(end) is not int:
            return result

        if not (
            0 <= start <= end < len(source_text)
        ):
            return result

        parsed_entities.append(
            {
                "text": entity_text,
                "type": entity_type,
                "start": start,
                "end": end,
            }
        )

        consistency.append(
            source_text[start:end + 1]
            == entity_text
        )

    result["schema_valid"] = True
    result["entities"] = parsed_entities
    result["span_text_consistency"] = consistency

    return result

# 该函数将实体字典转换为一个唯一的键，用于集合操作。键由实体的起始位置、结束位置和类型组成。
def entity_to_key(entity):

    return (
        entity["start"],
        entity["end"],
        entity["type"],
    )

# 该函数评估预测结果与真实实体的匹配情况，计算真正例（TP）、假正例（FP）和假负例（FN）的数量。它首先将真实实体和预测实体转换为集合，然后计算交集和差集来确定 TP、FP 和 FN 的数量。
def evaluate_sample(
    gold_entities,
    parsed_prediction,
):

    gold_set = {
        entity_to_key(entity)
        for entity in gold_entities
    }

    if not parsed_prediction["schema_valid"]:

        pred_set = set()

    else:

        pred_set = {
            entity_to_key(entity)
            for entity in parsed_prediction["entities"]
        }

    tp = len(
        gold_set & pred_set
    )

    fp = len(
        pred_set - gold_set
    )

    fn = len(
        gold_set - pred_set
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

# 该函数计算精确率（Precision）、召回率（Recall）和 F1 分数。它根据 TP、FP 和 FN 的数量计算这些指标，并处理除零的情况。
def compute_prf(
    tp,
    fp,
    fn,
):

    precision = (
        tp / (tp + fp)
        if tp + fp > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if tp + fn > 0
        else 0.0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

def evaluate_dataset(
    gold_samples,
    raw_outputs,
):

    assert len(gold_samples) == len(raw_outputs)

    total_tp = 0
    total_fp = 0
    total_fn = 0

    json_valid_count = 0
    schema_valid_count = 0

    consistency_correct = 0
    consistency_total = 0

    duplicate_sample_count = 0
    duplicate_entity_count = 0

    class_counts = {
        label: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
        }
        for label in CLUENER_LABELS
    }

    for sample, raw_output in zip(
        gold_samples,
        raw_outputs,
    ):

        source_text = sample["text"]
        gold_entities = sample["entities"]

        parsed = parse_prediction(
            raw_output,
            source_text,
        )

        if parsed["json_valid"]:
            json_valid_count += 1

        if parsed["schema_valid"]:
            schema_valid_count += 1

        # -------------------------
        # span-text consistency
        # -------------------------

        consistency_correct += sum(
            parsed["span_text_consistency"]
        )

        consistency_total += len(
            parsed["span_text_consistency"]
        )

        # -------------------------
        # gold / prediction sets
        # -------------------------

        gold_set = {
            entity_to_key(entity)
            for entity in gold_entities
        }

        if parsed["schema_valid"]:

            pred_keys = [
                entity_to_key(entity)
                for entity in parsed["entities"]
            ]

            pred_set = set(pred_keys)

            num_duplicates = (
                len(pred_keys)
                - len(pred_set)
            )

            if num_duplicates > 0:
                duplicate_sample_count += 1
                duplicate_entity_count += num_duplicates

        else:

            pred_set = set()

        tp_set = gold_set & pred_set
        fp_set = pred_set - gold_set
        fn_set = gold_set - pred_set

        total_tp += len(tp_set)
        total_fp += len(fp_set)
        total_fn += len(fn_set)

        # -------------------------
        # per-class counts
        # -------------------------

        for start, end, label in tp_set:
            class_counts[label]["tp"] += 1

        for start, end, label in fp_set:
            class_counts[label]["fp"] += 1

        for start, end, label in fn_set:
            class_counts[label]["fn"] += 1

    # =============================
    # Micro metrics
    # =============================

    micro = compute_prf(
        total_tp,
        total_fp,
        total_fn,
    )

    # =============================
    # Per-class metrics
    # =============================

    per_class = {}

    class_f1_values = []

    for label in sorted(CLUENER_LABELS):

        counts = class_counts[label]

        metrics = compute_prf(
            counts["tp"],
            counts["fp"],
            counts["fn"],
        )

        per_class[label] = {
            **counts,
            **metrics,
        }

        class_f1_values.append(
            metrics["f1"]
        )

    macro_f1 = (
        sum(class_f1_values)
        / len(class_f1_values)
    )

    # =============================
    # Diagnostic metrics
    # =============================

    num_samples = len(gold_samples)

    json_valid_rate = (
        json_valid_count / num_samples
    )

    schema_valid_rate = (
        schema_valid_count / num_samples
    )

    span_text_consistency_rate = (
        consistency_correct
        / consistency_total
        if consistency_total > 0
        else 0.0
    )

    duplicate_sample_rate = (
        duplicate_sample_count
        / num_samples
    )

    return {
        "num_samples": num_samples,

        "json_valid_rate":
            json_valid_rate,

        "schema_valid_rate":
            schema_valid_rate,

        "span_text_consistency_rate":
            span_text_consistency_rate,

        "duplicate_sample_rate":
            duplicate_sample_rate,

        "duplicate_entity_count":
            duplicate_entity_count,

        "micro": micro,

        "macro_f1": macro_f1,

        "per_class": per_class,
    }

def print_evaluation_result(result):

    print("\n" + "=" * 70)
    print("Evaluation Result")
    print("=" * 70)

    print(
        f"Samples: "
        f"{result['num_samples']}"
    )

    print(
        f"JSON validity: "
        f"{result['json_valid_rate']:.4%}"
    )

    print(
        f"Schema validity: "
        f"{result['schema_valid_rate']:.4%}"
    )

    print(
        f"Span-text consistency: "
        f"{result['span_text_consistency_rate']:.4%}"
    )

    print(
        f"Duplicate sample rate: "
        f"{result['duplicate_sample_rate']:.4%}"
    )

    print(
        f"Duplicate entities: "
        f"{result['duplicate_entity_count']}"
    )

    print("\n===== Micro =====")

    print(
        f"Precision: "
        f"{result['micro']['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{result['micro']['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{result['micro']['f1']:.4f}"
    )

    print(
        f"\nMacro F1: "
        f"{result['macro_f1']:.4f}"
    )

    print("\n===== Per Class =====")

    for label, metrics in result["per_class"].items():

        print(
            f"{label:15s}"
            f"P={metrics['precision']:.4f}  "
            f"R={metrics['recall']:.4f}  "
            f"F1={metrics['f1']:.4f}  "
            f"TP={metrics['tp']}  "
            f"FP={metrics['fp']}  "
            f"FN={metrics['fn']}"
        )

if __name__ == "__main__":

    text = "浙商银行企业信贷部叶老桂博士"

    outputs = [
        # 1. 完全正确
        '{"entities":[{"text":"浙商银行","type":"company","start":0,"end":3}]}',

        # 2. JSON 正确，但 type 非法
        '{"entities":[{"text":"浙商银行","type":"bank","start":0,"end":3}]}',

        # 3. JSON 和 schema 正确，但 text 与 span 不一致
        '{"entities":[{"text":"中国银行","type":"company","start":0,"end":3}]}',

        # 4. Markdown code fence
        '```json\n{"entities":[]}\n```',

        # 5. 合法的空实体输出
        '{"entities":[]}',
    ]

    for output in outputs:

        print("\n" + "=" * 60)
        print(output)

        result = parse_prediction(
            output,
            text,
        )

        print(result)

    def test_entity_metrics():

        gold_entities = [
            {
                "text": "浙商银行",
                "type": "company",
                "start": 0,
                "end": 3,
            },
            {
                "text": "叶老桂",
                "type": "name",
                "start": 9,
                "end": 11,
            },
        ]

        test_predictions = [
            # Case 1：全部正确
            {
                "schema_valid": True,
                "entities": [
                    {
                        "text": "浙商银行",
                        "type": "company",
                        "start": 0,
                        "end": 3,
                    },
                    {
                        "text": "叶老桂",
                        "type": "name",
                        "start": 9,
                        "end": 11,
                    },
                ],
            },

            # Case 2：一个正确，一个类型错误
            {
                "schema_valid": True,
                "entities": [
                    {
                        "text": "浙商银行",
                        "type": "company",
                        "start": 0,
                        "end": 3,
                    },
                    {
                        "text": "叶老桂",
                        "type": "position",
                        "start": 9,
                        "end": 11,
                    },
                ],
            },

            # Case 3：只预测出一个
            {
                "schema_valid": True,
                "entities": [
                    {
                        "text": "浙商银行",
                        "type": "company",
                        "start": 0,
                        "end": 3,
                    },
                ],
            },

            # Case 4：格式非法
            {
                "schema_valid": False,
                "entities": [],
            },
        ]

        for i, prediction in enumerate(
            test_predictions,
            start=1,
        ):

            counts = evaluate_sample(
                gold_entities,
                prediction,
            )

            metrics = compute_prf(
                counts["tp"],
                counts["fp"],
                counts["fn"],
            )

            print(
                f"\nCase {i}"
            )

            print(counts)
            print(metrics)

    test_entity_metrics()

    # 多样本假数据测试
    gold_samples = [
        {
            "text": "浙商银行企业信贷部叶老桂博士",
            "entities": [
                {
                    "text": "浙商银行",
                    "type": "company",
                    "start": 0,
                    "end": 3,
                },
                {
                    "text": "叶老桂",
                    "type": "name",
                    "start": 9,
                    "end": 11,
                },
            ],
        },

        {
            "text": "王伟加入腾讯公司",
            "entities": [
                {
                    "text": "王伟",
                    "type": "name",
                    "start": 0,
                    "end": 1,
                },
                {
                    "text": "腾讯公司",
                    "type": "company",
                    "start": 4,
                    "end": 7,
                },
            ],
        },
    ]

    raw_outputs = [
        # 第一条两个全对
        (
            '{"entities":['
            '{"text":"浙商银行","type":"company","start":0,"end":3},'
            '{"text":"叶老桂","type":"name","start":9,"end":11}'
            ']}'
        ),

        # 第二条只预测对王伟
        (
            '{"entities":['
            '{"text":"王伟","type":"name","start":0,"end":1}'
            ']}'
        ),
    ]

    result = evaluate_dataset(
        gold_samples,
        raw_outputs,
    )

    print_evaluation_result(result)