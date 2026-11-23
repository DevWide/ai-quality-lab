"""Shared score log: every metric of every run goes to output/scores.jsonl.

A single run can't be trusted (the judge is not deterministic), so scores are
appended per run and aggregated later with `uv run python -m evals_python.summarize`.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

from app import config

SCORES_FILE = Path(__file__).parent / "output" / "scores.jsonl"
RUN_ID = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def record_score(*, case: str, metric: str, score: float | None, passed: bool, reason: str | None, framework: str) -> None:
    if score is not None and math.isnan(score):
        score = None  # the judge returned nothing usable; logged as a failure
    SCORES_FILE.parent.mkdir(exist_ok=True)
    row = {
        "run_id": RUN_ID,
        "framework": framework,
        "case": case,
        "metric": metric,
        "score": score,
        "passed": passed,
        "reason": reason,
        "judge": config.JUDGE_MODEL,
    }
    with SCORES_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
