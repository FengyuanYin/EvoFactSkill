from __future__ import annotations


def generator_report(*, real_only, robustness, generation_quality, safety) -> dict:
    return {
        "authentic_data_detection": real_only,
        "generated_robustness": robustness,
        "generation_quality": generation_quality,
        "safety": safety,
        "formal_promotion_metric": "authentic_data_detection",
    }
