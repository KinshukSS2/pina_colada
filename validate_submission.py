#!/usr/bin/env python3
from __future__ import annotations

import os
import sys
from typing import Dict

import requests


BASE_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:7860")


def post(path: str, payload: Dict) -> Dict:
    response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def main() -> int:
    try:
        root = requests.get(f"{BASE_URL}/", timeout=10)
        root.raise_for_status()

        reset_1 = post("/reset", {"task_id": "easy", "session_id": "determinism-check"})
        step_1 = post("/step", {"session_id": "determinism-check", "action": "hold"})

        reset_2 = post("/reset", {"task_id": "easy", "session_id": "determinism-check"})
        step_2 = post("/step", {"session_id": "determinism-check", "action": "hold"})

        if step_1["observation"]["queue_ns"] != step_2["observation"]["queue_ns"]:
            print("[FAIL] Determinism check failed for queue_ns")
            return 1

        if step_1["observation"]["queue_ew"] != step_2["observation"]["queue_ew"]:
            print("[FAIL] Determinism check failed for queue_ew")
            return 1

        invalid = post("/step", {"session_id": "determinism-check", "action": "bad_action"})
        if invalid["info"]["action_valid"]:
            print("[FAIL] Invalid action should be rejected")
            return 1

        print("[PASS] Runtime checks passed")
        return 0
    except Exception as exc:
        print(f"[FAIL] Runtime checks failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
