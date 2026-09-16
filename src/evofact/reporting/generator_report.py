from __future__ import annotations


def generator_report(*, real_only, robustness, generation_quality, safety) -> dict:
    return {
        "real_only_detection": real_only,
        "generated_robustness": robustness,
        "generation_quality": generation_quality,
        "safety": safety,
        "formal_promotion_metric": "real_only_detection",
    }
