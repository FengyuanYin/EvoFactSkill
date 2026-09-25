"""Plot the gate's recorded validation Brier scores without new model calls."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

FIELDS = (
    "batch_index",
    "proposal_id",
    "disposition",
    "baseline_brier",
    "candidate_brier",
    "baseline_macro_f1",
    "candidate_macro_f1",
    "validation_evaluations",
    "baseline_cost_available",
    "candidate_cost_available",
)
ACTIVE_FIELDS = (
    "batch_index",
    "package_bank_digest",
    "brier",
    "macro_f1",
    "accuracy",
    "coverage",
    "n",
    "cost_available",
)


def gate_validation_rows(checkpoint: dict) -> list[dict]:
    """Join gate decisions to their training batch through proposal source traces."""
    batches = checkpoint.get("batch_audit", [])
    sample_batch = {
        sample_id: batch["batch_index"]
        for batch in batches
        for sample_id in batch.get("sample_ids", [])
    }
    accumulated = checkpoint.get("accumulated", {})
    trace_batch = {
        trace["trace_id"]: sample_batch.get(trace["sample_id"])
        for trace in accumulated.get("traces", [])
    }
    proposals = accumulated.get("proposals", [])
    decisions = accumulated.get("gate_decisions", [])
    if len(proposals) != len(decisions):
        raise ValueError("checkpoint proposals and gate decisions are not aligned")
    rows = []
    for proposal, decision in zip(proposals, decisions, strict=True):
        source_batches = {
            trace_batch.get(trace_id)
            for trace_id in proposal.get("source_trace_ids", [])
        }
        source_batches.discard(None)
        if len(source_batches) != 1:
            continue
        baseline = decision["baseline_result"]["aggregate_metrics"]
        candidate = decision["candidate_result"]["aggregate_metrics"]
        if "brier" not in baseline or "brier" not in candidate:
            continue
        rows.append(
            {
                "batch_index": source_batches.pop(),
                "proposal_id": proposal["proposal_id"],
                "disposition": decision["disposition"],
                "baseline_brier": baseline["brier"],
                "candidate_brier": candidate["brier"],
                "baseline_macro_f1": baseline.get("macro_f1_all"),
                "candidate_macro_f1": candidate.get("macro_f1_all"),
                "validation_evaluations": baseline.get("n"),
                "baseline_cost_available": baseline.get("cost_available"),
                "candidate_cost_available": candidate.get("cost_available"),
            }
        )
    return sorted(rows, key=lambda row: (row["batch_index"], row["proposal_id"]))


def active_validation_rows(checkpoint: dict) -> list[dict]:
    """Read optional post-batch validation of the committed active Skill Bank."""
    rows = []
    for batch in checkpoint.get("batch_audit", []):
        metrics = batch.get("validation_metrics") or {}
        if "brier" not in metrics:
            continue
        rows.append(
            {
                "batch_index": batch["batch_index"],
                "package_bank_digest": batch.get("package_bank_digest"),
                "brier": metrics["brier"],
                "macro_f1": metrics.get("macro_f1_all"),
                "accuracy": metrics.get("accuracy_all"),
                "coverage": metrics.get("coverage"),
                "n": metrics.get("n"),
                "cost_available": metrics.get("cost_available"),
            }
        )
    return sorted(rows, key=lambda row: row["batch_index"])


def _svg(rows: list[dict], active_rows: list[dict], total_batches: int) -> str:
    width, height = 860, 460
    left, right, top, bottom = 70, 30, 75, 70
    plot_width = width - left - right
    plot_height = height - top - bottom
    values = [float(row[key]) for row in rows for key in ("baseline_brier", "candidate_brier")]
    values.extend(float(row["brier"]) for row in active_rows)
    low = max(0.0, min(values) - 0.05) if values else 0.0
    high = min(1.0, max(values) + 0.05) if values else 1.0
    if high - low < 0.01:
        low = max(0.0, low - 0.01)
        high = min(1.0, high + 0.01)
    last_batch = max(total_batches, max((row["batch_index"] for row in rows), default=1), 2)

    def x(batch: int) -> float:
        return left + (batch - 1) * plot_width / (last_batch - 1)

    def y(value: float) -> float:
        return top + (high - value) * plot_height / (high - low)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="70" y="30" font-family="sans-serif" font-size="20">Validation Brier score by evolve batch</text>',
        '<text x="70" y="50" font-family="sans-serif" font-size="12" fill="#555">Lower is better. Gray points have unavailable cost/partial inference; missing batches were not evaluated.</text>',
        f'<path d="M {left} {top} V {top + plot_height} H {left + plot_width}" fill="none" stroke="#555"/>',
    ]
    for tick in range(5):
        value = low + (high - low) * tick / 4
        py = y(value)
        parts.append(
            f'<path d="M {left} {py:.1f} H {left + plot_width}" stroke="#ddd"/>'
            f'<text x="{left - 10}" y="{py + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.2f}</text>'
        )
    for batch in range(1, last_batch + 1):
        if batch == 1 or batch == last_batch or batch % 5 == 0:
            parts.append(
                f'<text x="{x(batch):.1f}" y="{top + plot_height + 20}" text-anchor="middle" font-family="sans-serif" font-size="11">{batch}</text>'
            )
    for row in rows:
        px = x(row["batch_index"])
        base_y = y(float(row["baseline_brier"]))
        candidate_y = y(float(row["candidate_brier"]))
        complete = (
            row["baseline_cost_available"] == 1.0
            and row["candidate_cost_available"] == 1.0
        )
        base_color = "#2563eb" if complete else "#888"
        candidate_color = "#ea580c" if complete else "#888"
        parts.append(
            f'<path d="M {px:.1f} {base_y:.1f} V {candidate_y:.1f}" stroke="#aaa"/>'
            f'<circle cx="{px:.1f}" cy="{base_y:.1f}" r="4" fill="{base_color}"/>'
            f'<circle cx="{px:.1f}" cy="{candidate_y:.1f}" r="4" fill="{candidate_color}"/>'
        )
    valid_active = [row for row in active_rows if row["cost_available"] == 1.0]
    if len(valid_active) > 1:
        points = " ".join(
            f'{x(row["batch_index"]):.1f},{y(float(row["brier"])):.1f}'
            for row in valid_active
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="#16a34a" stroke-width="2"/>')
    for row in active_rows:
        color = "#16a34a" if row["cost_available"] == 1.0 else "#888"
        parts.append(
            f'<circle cx="{x(row["batch_index"]):.1f}" cy="{y(float(row["brier"])):.1f}" r="5" fill="{color}"/>'
        )
    parts.extend(
        [
            f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="13">Training batch</text>',
            '<circle cx="490" cy="50" r="5" fill="#16a34a"/><text x="502" y="54" font-family="sans-serif" font-size="12">Active bank</text>',
            '<circle cx="610" cy="50" r="4" fill="#2563eb"/><text x="621" y="54" font-family="sans-serif" font-size="12">Baseline</text>',
            '<circle cx="715" cy="50" r="4" fill="#ea580c"/><text x="726" y="54" font-family="sans-serif" font-size="12">Candidate</text>',
            "</svg>",
        ]
    )
    return "\n".join(parts) + "\n"


def write_validation_curve(checkpoint_path: Path, output_dir: Path) -> dict[str, str]:
    """Export CSV and SVG from a saved evolve checkpoint, without rerunning validation."""
    checkpoint = json.loads(Path(checkpoint_path).read_text(encoding="utf-8"))
    rows = gate_validation_rows(checkpoint)
    active_rows = active_validation_rows(checkpoint)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "validation-curve.csv"
    active_csv_path = output_dir / "validation-active.csv"
    svg_path = output_dir / "validation-curve.svg"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with active_csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=ACTIVE_FIELDS)
        writer.writeheader()
        writer.writerows(active_rows)
    svg_path.write_text(
        _svg(rows, active_rows, int(checkpoint.get("total_batches", 0))), encoding="utf-8"
    )
    return {"csv": str(csv_path), "active_csv": str(active_csv_path), "svg": str(svg_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot saved evolve gate validation scores")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    paths = write_validation_curve(
        args.checkpoint,
        args.output_dir or args.checkpoint.parent,
    )
    print(json.dumps(paths, ensure_ascii=False))


if __name__ == "__main__":
    main()
