from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List


DEFAULT_FILES = [
    "inference.py",
    "env/grader.py",
    "env/environment.py",
    "env/tasks.py",
]


HIGH_RISK_PATTERNS = {
    "constant_score_assignment": re.compile(r"score\s*=\s*(0(\.\d+)?|1(\.0+)?)"),
    "task_to_fixed_score_map": re.compile(r"\{\s*['\"]easy['\"]\s*:\s*[0-9.]+\s*,\s*['\"]medium['\"]\s*:\s*[0-9.]+"),
    "hardcoded_task_action": re.compile(r"if\s+task_id\s*==\s*['\"](easy|medium|hard)['\"]\s*:\s*return\s+['\"]"),
}

HEURISTIC_PATTERNS = {
    "always_hold_return": re.compile(r"return\s+['\"]hold['\"]"),
    "disable_llm_env": re.compile(r"DISABLE_LLM"),
    "fallback_action_constant": re.compile(r"SAFE_FALLBACK\s*=\s*['\"]hold['\"]"),
}


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _line_number(content: str, index: int) -> int:
    return content.count("\n", 0, index) + 1


def _scan_patterns(content: str, patterns: Dict[str, re.Pattern[str]]) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []
    for name, pattern in patterns.items():
        for match in pattern.finditer(content):
            findings.append(
                {
                    "name": name,
                    "line": _line_number(content, match.start()),
                    "snippet": match.group(0)[:180],
                }
            )
    return findings


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit baseline integrity for hardcoded scoring/actions")
    parser.add_argument("--strict", action="store_true", help="fail on high-risk findings")
    parser.add_argument("--report", default="/tmp/audit_baseline_integrity.json")
    parser.add_argument("--files", nargs="*", default=DEFAULT_FILES)
    args = parser.parse_args()

    file_reports: List[Dict[str, object]] = []
    high_risk_total = 0

    for rel_path in args.files:
        path = Path(rel_path)
        if not path.exists():
            file_reports.append(
                {
                    "file": rel_path,
                    "missing": True,
                    "high_risk_findings": [],
                    "heuristic_findings": [],
                }
            )
            high_risk_total += 1
            continue

        content = _read(path)
        high_risk_findings = _scan_patterns(content, HIGH_RISK_PATTERNS)
        heuristic_findings = _scan_patterns(content, HEURISTIC_PATTERNS)

        file_reports.append(
            {
                "file": rel_path,
                "missing": False,
                "high_risk_findings": high_risk_findings,
                "heuristic_findings": heuristic_findings,
            }
        )
        high_risk_total += len(high_risk_findings)

    report = {
        "ok": high_risk_total == 0,
        "strict": args.strict,
        "high_risk_total": high_risk_total,
        "files": file_reports,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"[REPORT] {report_path}")
    print(f"[SUMMARY] high_risk_total={high_risk_total}")

    if args.strict and high_risk_total > 0:
        print("[FAIL] strict integrity audit failed")
        return 1

    print("[PASS] integrity audit completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
