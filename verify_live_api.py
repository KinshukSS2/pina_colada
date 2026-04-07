from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import requests
from dotenv import load_dotenv


load_dotenv()


PROOF_PATTERN = re.compile(
    r"\[PROOF\]\s+run_id=(?P<run_id>\S+)\s+llm_attempted=(?P<llm_attempted>\d+)\s+"
    r"llm_succeeded=(?P<llm_succeeded>\d+)\s+nonce_verified=(?P<nonce_verified>\d+)\s+"
    r"fallback_steps=(?P<fallback_steps>\d+)\s+sanitized_steps=(?P<sanitized_steps>\d+)"
)
END_PATTERN = re.compile(r"\[END\]\s+session=(?P<session>\S+)\s+score=(?P<score>[0-9.]+)\s+steps=(?P<steps>\d+)")


@dataclass
class RunResult:
    mode: str
    return_code: int
    proof: Dict[str, Any]
    end: Dict[str, Any]
    stdout: str
    stderr: str


def _failed_run(mode: str, message: str, return_code: int = 99) -> RunResult:
    return RunResult(
        mode=mode,
        return_code=return_code,
        proof={},
        end={},
        stdout="",
        stderr=message,
    )


def _parse_proof(output: str) -> Dict[str, Any]:
    matches = list(PROOF_PATTERN.finditer(output))
    if not matches:
        return {}
    data = matches[-1].groupdict()
    return {
        "run_id": data["run_id"],
        "llm_attempted": int(data["llm_attempted"]),
        "llm_succeeded": int(data["llm_succeeded"]),
        "nonce_verified": int(data["nonce_verified"]),
        "fallback_steps": int(data["fallback_steps"]),
        "sanitized_steps": int(data["sanitized_steps"]),
    }


def _parse_end(output: str) -> Dict[str, Any]:
    matches = list(END_PATTERN.finditer(output))
    if not matches:
        return {}
    data = matches[-1].groupdict()
    return {
        "session": data["session"],
        "score": float(data["score"]),
        "steps": int(data["steps"]),
    }


def _ensure_server(env_base_url: str) -> None:
    response = requests.get(env_base_url.rstrip("/") + "/", timeout=5)
    response.raise_for_status()


def _run_inference(mode: str, task: str, env_base_url: str, trace_dir: str, model_name: str) -> RunResult:
    env = os.environ.copy()
    env["ENV_BASE_URL"] = env_base_url
    env["TRACE_API"] = "1"
    env["TRACE_DIR"] = trace_dir
    env["TRACE_BASENAME"] = f"verify_{mode}"
    env["VERIFY_STRICT"] = "1"
    env["VERIFY_MIN_SUCCESSFUL_LLM_CALLS"] = "1"
    env["MODEL_NAME"] = model_name

    if mode == "broken":
        env["OPENAI_API_KEY"] = "sk-or-v1-invalid-key-for-verification"
        env["API_BASE_URL"] = os.getenv("BROKEN_API_BASE_URL", "https://openrouter.ai/api/v1")
    else:
        if not env.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for live verification mode")
        env["API_BASE_URL"] = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")

    command = [sys.executable, "inference.py", task]
    completed = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    stdout = completed.stdout or ""
    stderr = completed.stderr or ""

    return RunResult(
        mode=mode,
        return_code=completed.returncode,
        proof=_parse_proof(stdout),
        end=_parse_end(stdout),
        stdout=stdout,
        stderr=stderr,
    )


def _evaluate_live(result: RunResult) -> list[str]:
    errors: list[str] = []
    if result.return_code != 0:
        errors.append(f"live run returned non-zero exit code: {result.return_code}")
    if not result.proof:
        errors.append("live run missing [PROOF] summary")
        return errors
    if result.proof.get("llm_succeeded", 0) < 1:
        errors.append("live run had zero successful LLM calls")
    if result.proof.get("nonce_verified", 0) < 1:
        errors.append("live run had zero nonce-verified responses")
    return errors


def _evaluate_broken(result: RunResult) -> list[str]:
    errors: list[str] = []
    if result.return_code == 0:
        errors.append("broken run unexpectedly exited with success")
    if not result.proof:
        errors.append("broken run missing [PROOF] summary")
        return errors
    if result.proof.get("llm_succeeded", 0) > 0:
        errors.append("broken run unexpectedly succeeded in LLM calls")
    return errors


def _print_result(result: RunResult) -> None:
    proof = result.proof
    end = result.end
    print(
        f"[{result.mode.upper()}] rc={result.return_code} "
        f"llm_succeeded={proof.get('llm_succeeded', 'NA')} "
        f"nonce_verified={proof.get('nonce_verified', 'NA')} "
        f"fallback_steps={proof.get('fallback_steps', 'NA')} "
        f"score={end.get('score', 'NA')} steps={end.get('steps', 'NA')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify that inference uses live external API calls")
    parser.add_argument("--mode", choices=["live", "broken", "compare"], default="compare")
    parser.add_argument("--task", default="easy")
    parser.add_argument("--env-base-url", default=os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860"))
    parser.add_argument("--trace-dir", default="/tmp")
    parser.add_argument("--model", default=os.getenv("MODEL_NAME", "openai/gpt-4o-mini"))
    parser.add_argument("--report", default="")
    args = parser.parse_args()

    _ensure_server(args.env_base_url)

    modes = [args.mode] if args.mode in {"live", "broken"} else ["live", "broken"]
    results: list[RunResult] = []
    evaluations: Dict[str, list[str]] = {}

    for mode in modes:
        try:
            result = _run_inference(mode, args.task, args.env_base_url, args.trace_dir, args.model)
        except Exception as exc:
            result = _failed_run(mode, str(exc))
        results.append(result)
        _print_result(result)
        if mode == "live":
            evaluations[mode] = _evaluate_live(result)
        else:
            evaluations[mode] = _evaluate_broken(result)

    all_errors: list[str] = []
    for mode, errors in evaluations.items():
        for error in errors:
            all_errors.append(f"{mode}: {error}")

    report = {
        "timestamp": int(time.time()),
        "mode": args.mode,
        "task": args.task,
        "env_base_url": args.env_base_url,
        "model": args.model,
        "results": [
            {
                "mode": r.mode,
                "return_code": r.return_code,
                "proof": r.proof,
                "end": r.end,
                "stderr": r.stderr,
            }
            for r in results
        ],
        "errors": all_errors,
        "ok": len(all_errors) == 0,
    }

    report_path = args.report.strip() or str(Path(args.trace_dir) / f"verify_live_api_{int(time.time())}.json")
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[REPORT] {report_path}")

    if all_errors:
        print("[FAIL] verification failed")
        for error in all_errors:
            print(f" - {error}")
        return 1

    print("[PASS] verification checks succeeded")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
