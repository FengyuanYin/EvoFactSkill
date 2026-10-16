"""Read-only paired trace analysis; write derived JSON and Markdown under docs."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def features(t):
    reports = t.get("specialist_reports", [])
    return {
        "label": t["decision"]["label"],
        "confidence": t["decision"].get("confidence"),
        "rationale": t["decision"].get("rationale"),
        "reports": [r.get("report_type", r["skill_id"]) for r in reports],
        "assessments": [r.get("assessment") for r in reports],
        "routing_reasons": t.get("routing", {}).get("reasons"),
        "fallback": t.get("routing", {}).get("fallback_used"),
        "execution": t.get("execution_summary"),
        "errors": t.get("errors"),
        "evidence_n": len(t["sample_public"].get("evidence", [])),
    }


def summarize(pairs):
    out = {"n": len(pairs)}
    out["transitions"] = dict(Counter(p["transition"] for p in pairs))
    for side in ["baseline", "evolved"]:
        out[side] = {
            "confusion": dict(Counter(p["gold"] + "->" + p[side]["label"] for p in pairs)),
            "report_counts": dict(Counter(r for p in pairs for r in p[side]["reports"])),
            "report_patterns": dict(Counter("+".join(p[side]["reports"]) for p in pairs)),
            "avg_reports": sum(len(p[side]["reports"]) for p in pairs) / len(pairs),
            "errors_samples": sum(bool(p[side]["errors"]) for p in pairs),
            "execution_incomplete": sum(
                p[side]["execution"].get("required_complete") is False
                for p in pairs
                if p[side]["execution"]
            ),
            "no_external_evidence": sum(p[side]["evidence_n"] == 0 for p in pairs),
        }
    return out


def main():
    pairs = []
    for fold in (1, 2):
        folder = ROOT / f"docs/weibo21_evolve_fold_{fold}"
        result = read(folder / "final-test.json")
        assert result["status"] == "complete" and result["failed_samples"] == 0
        records = {}
        with (folder / "final-test.samples.jsonl").open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if r.get("kind") == "sample" and r.get("status") == "complete":
                    records[r["sample_id"]] = r
        for domain, metrics in result["domain_metrics"].items():
            baseline = read(
                ROOT / f"outputs/static-100-before-pr-review/weibo21-test-{domain}-100.json"
            )
            for t in baseline["traces"]:
                r = records[t["sample_id"]]
                e = r["evaluation"]
                gold = e["gold"]
                assert e["domain"] == domain
                a = t["decision"]["label"] == gold
                b = r["trace"]["decision"]["label"] == gold
                pairs.append(
                    {
                        "fold": fold,
                        "domain": domain,
                        "sample_id": t["sample_id"],
                        "gold": gold,
                        "transition": "stable_correct"
                        if a and b
                        else "regression"
                        if a
                        else "recovery"
                        if b
                        else "stable_wrong",
                        "text": t["sample_public"]["text"],
                        "baseline": features(t),
                        "evolved": features(r["trace"]),
                    }
                )
            actual = pairs[-100:]
            assert len(baseline["traces"]) == 100
            assert (
                abs(
                    sum(p["baseline"]["label"] == p["gold"] for p in actual) / 100
                    - baseline["metrics"]["accuracy_all"]
                )
                < 1e-9
            )
            assert (
                abs(
                    sum(p["evolved"]["label"] == p["gold"] for p in actual) / 100
                    - metrics["accuracy_all"]
                )
                < 1e-9
            )
    summary = {
        "overall": summarize(pairs),
        "domains": {
            d: summarize([p for p in pairs if p["domain"] == d])
            for d in dict.fromkeys(p["domain"] for p in pairs)
        },
    }
    (ROOT / "docs/evolve-trace-paired-analysis.json").write_text(
        json.dumps({"summary": summary, "pairs": pairs}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
