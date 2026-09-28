from span_alignment import (
    align_prediction,
)


# ============================================================
# Case 1
# 唯一 occurrence
# ============================================================

source_text = (
    "我在浙江大学学习。"
)

raw_output = (
    '{"entities":['
    '{"text":"浙江大学",'
    '"type":"organization",'
    '"start":100,'
    '"end":200}'
    ']}'
)


aligned = align_prediction(
    raw_output,
    source_text,
)


print("=" * 80)
print("CASE 1")
print("=" * 80)

print(aligned)


# ============================================================
# Case 2
# repeated mention
# ============================================================

source_text = (
    "建行今天发布公告，"
    "随后建行再次回应。"
)

raw_output = (
    '{"entities":['
    '{"text":"建行",'
    '"type":"company",'
    '"start":10,'
    '"end":11}'
    ']}'
)


aligned = align_prediction(
    raw_output,
    source_text,
)


print("\n" + "=" * 80)
print("CASE 2")
print("=" * 80)

print(aligned)


# ============================================================
# Case 3
# surface 不存在
# ============================================================

source_text = (
    "今天北京天气很好。"
)

raw_output = (
    '{"entities":['
    '{"text":"上海",'
    '"type":"address",'
    '"start":50,'
    '"end":51}'
    ']}'
)


aligned = align_prediction(
    raw_output,
    source_text,
)


print("\n" + "=" * 80)
print("CASE 3")
print("=" * 80)

print(aligned)