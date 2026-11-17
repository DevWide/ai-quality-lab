"""Aggregate evals_python/output/scores.jsonl into per-metric stats.

    uv run python -m evals_python.summarize
"""

import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

SCORES_FILE = Path(__file__).parent / "output" / "scores.jsonl"


def main() -> None:
    rows = [json.loads(line) for line in SCORES_FILE.read_text(encoding="utf-8").splitlines() if line]
    runs = sorted({r["run_id"] for r in rows})
    by_metric = defaultdict(list)
    for r in rows:
        by_metric[r["metric"]].append(r)

    print(f"{len(runs)} run(s): {', '.join(runs)}\n")
    print(f"{'Metric':<28}{'mean':>6}{'min':>6}{'max':>6}{'pass':>9}")
    for metric, items in by_metric.items():
        scores = [i["score"] for i in items]
        passed = sum(i["passed"] for i in items)
        print(f"{metric:<28}{mean(scores):>6.2f}{min(scores):>6.2f}{max(scores):>6.2f}{passed:>5}/{len(items)}")


if __name__ == "__main__":
    main()
