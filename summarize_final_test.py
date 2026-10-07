import re

import numpy as np


LOG_PATHS = {
    42:
        "logs/final_test/"
        "s2_seed42_test.log",

    43:
        "logs/final_test/"
        "s2_seed43_test.log",

    44:
        "logs/final_test/"
        "s2_seed44_test.log",
}


def extract_metrics(
    path,
):

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as f:

        text = f.read()


    # ========================================================
    # Basic validity
    # ========================================================

    json_validity = re.search(
        r"JSON validity:\s*"
        r"([0-9.]+)%",
        text,
    )


    schema_validity = re.search(
        r"Schema validity:\s*"
        r"([0-9.]+)%",
        text,
    )


    # ========================================================
    # Strict Micro
    # ========================================================

    micro = re.search(
        r"===== Micro ====="
        r"\s*"
        r"Precision:\s*([0-9.]+)"
        r"\s*"
        r"Recall:\s*([0-9.]+)"
        r"\s*"
        r"F1:\s*([0-9.]+)",
        text,
    )


    # ========================================================
    # Macro
    # ========================================================

    macro = re.search(
        r"Macro F1:\s*"
        r"([0-9.]+)",
        text,
    )


    # ========================================================
    # Surface-Type
    # ========================================================

    surface = re.search(
        r"===== Surface-Type Diagnostic ====="
        r"\s*"
        r"Precision:\s*([0-9.]+)"
        r"\s*"
        r"Recall:\s*([0-9.]+)"
        r"\s*"
        r"F1:\s*([0-9.]+)",
        text,
    )


    if (
        json_validity is None
        or schema_validity is None
        or micro is None
        or macro is None
        or surface is None
    ):

        raise RuntimeError(
            f"Could not parse metrics from:\n"
            f"{path}"
        )


    return {
        "json_validity":
            float(
                json_validity.group(1)
            ),

        "schema_validity":
            float(
                schema_validity.group(1)
            ),

        "strict_precision":
            float(
                micro.group(1)
            ),

        "strict_recall":
            float(
                micro.group(2)
            ),

        "strict_f1":
            float(
                micro.group(3)
            ),

        "macro_f1":
            float(
                macro.group(1)
            ),

        "surface_precision":
            float(
                surface.group(1)
            ),

        "surface_recall":
            float(
                surface.group(2)
            ),

        "surface_f1":
            float(
                surface.group(3)
            ),
    }


results = {}


for seed, path in (
    LOG_PATHS.items()
):

    results[seed] = (
        extract_metrics(
            path
        )
    )


print(
    "=" * 80
)

print(
    "FINAL TEST — PER SEED"
)

print(
    "=" * 80
)


for seed in sorted(
    results
):

    result = results[
        seed
    ]


    print(
        f"\nSeed {seed}"
    )


    print(
        "Strict P/R/F1: "
        f"{result['strict_precision']:.4f} / "
        f"{result['strict_recall']:.4f} / "
        f"{result['strict_f1']:.4f}"
    )


    print(
        "Surface-Type F1: "
        f"{result['surface_f1']:.4f}"
    )


    print(
        "Macro F1: "
        f"{result['macro_f1']:.4f}"
    )


    print(
        "Schema validity: "
        f"{result['schema_validity']:.4f}%"
    )


print(
    "\n"
    + "=" * 80
)

print(
    "FINAL TEST — MEAN ± SAMPLE STD"
)

print(
    "=" * 80
)


METRICS = [
    "strict_precision",
    "strict_recall",
    "strict_f1",
    "macro_f1",
    "surface_f1",
    "schema_validity",
]


for metric in METRICS:

    values = np.asarray(
        [
            results[seed][metric]
            for seed
            in sorted(results)
        ],
        dtype=float,
    )


    mean = values.mean()

    std = values.std(
        ddof=1
    )


    print(
        f"{metric:20s}: "
        f"{mean:.4f} ± {std:.4f}"
    )