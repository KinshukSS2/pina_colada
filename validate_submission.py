#!/usr/bin/env python3
from __future__ import annotations

import os
from typing import Dict

import requests
from dotenv import load_dotenv  # pylint: disable=import-error

load_dotenv()

BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")


def post(path: str, payload: Dict) -> Dict:
    response = requests.post(f"{BASE_URL}{path}", json=payload, timeout=15)
    response.raise_for_status()
    return response.json()


def main() -> int:
    try:
        # Health check
        health = requests.get(f"{BASE_URL}/health", timeout=10)
        health.raise_for_status()
        print(f"[PASS] Health check: {health.json()}")

        # Legacy endpoint tests
        post("/legacy/reset", {"task_id": "easy", "session_id": "determinism-check"})
        step_1 = post("/legacy/step", {"session_id": "determinism-check", "action": "assign:medium:billing"})

        post("/legacy/reset", {"task_id": "easy", "session_id": "determinism-check"})
        step_2 = post("/legacy/step", {"session_id": "determinism-check", "action": "assign:medium:billing"})

        obs1 = step_1["observation"]
        obs2 = step_2["observation"]

        if obs1["tickets_assigned"] != obs2["tickets_assigned"]:
            print("[FAIL] Determinism check failed for tickets_assigned")
            return 1

        if obs1["correct_assignments"] != obs2["correct_assignments"]:
            print("[FAIL] Determinism check failed for correct_assignments")
            return 1

        print("[PASS] Determinism check passed")

        # Invalid action test
        invalid = post("/legacy/step", {"session_id": "determinism-check", "action": "bad_action"})
        if invalid["info"]["action_valid"]:
            print("[FAIL] Invalid action should be rejected")
            return 1
        print("[PASS] Invalid action correctly rejected")

        # OpenEnv standard endpoints
        std_reset = post("/reset", {"task_id": "easy"})
        assert "observation" in std_reset, f"Missing observation in reset: {std_reset}"
        assert "done" in std_reset, f"Missing done in reset: {std_reset}"
        print("[PASS] Standard /reset works")

        std_step = post("/step", {"action": {"action": "assign:medium:general", "session_id": std_reset["observation"].get("session_id", "test")}})
        assert "observation" in std_step, f"Missing observation in step: {std_step}"
        assert "reward" in std_step, f"Missing reward in step: {std_step}"
        assert "done" in std_step, f"Missing done in step: {std_step}"
        print("[PASS] Standard /step works")

        # /tasks endpoint
        tasks = requests.get(f"{BASE_URL}/tasks", timeout=10).json()
        assert isinstance(tasks, list), "/tasks should return a list"
        task_ids = {t["task_id"] for t in tasks}
        for tid in ["easy", "medium", "hard"]:
            assert tid in task_ids, f"Missing task: {tid}"
        print("[PASS] /tasks endpoint works")

        # /schema endpoint
        schema = requests.get(f"{BASE_URL}/schema", timeout=10).json()
        assert "action" in schema, "Missing action schema"
        assert "observation" in schema, "Missing observation schema"
        print("[PASS] /schema endpoint works")

        print("[PASS] All runtime checks passed")
        return 0
    except (OSError, requests.RequestException, AssertionError, KeyError) as exc:
        print(f"[FAIL] Runtime checks failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
