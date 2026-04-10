"""Inference script for TicketTriageEnv.

Drives the environment via HTTP (legacy endpoints) and uses an LLM or
rule-based fallback to triage support tickets.

Usage:
    python inference.py [easy|medium|hard]
"""
from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from typing import Any, Dict

import requests
from openai import OpenAI  # pylint: disable=import-error
from dotenv import load_dotenv  # pylint: disable=import-error

load_dotenv()

ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
MODEL_CANDIDATES = os.getenv(
    "MODEL_CANDIDATES",
    "openai/gpt-4o-mini,openai/gpt-4.1-mini,openai/gpt-4o",
)
HF_TOKEN = os.getenv("HF_TOKEN", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or HF_TOKEN
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
DISABLE_LLM = os.getenv("DISABLE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_STEPS = int(os.getenv("MAX_STEPS", "100"))
ENV_NAME = "ticket-triage-openenv"

EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "ticket-triage-openenv",
}

# Valid action pattern
ACTION_PATTERN = re.compile(
    r"^(assign:(low|medium|high|critical):\w+|escalate|defer|skip|resolve:\d+)$"
)


# ---------------------------------------------------------------------------
# Environment API helpers
# ---------------------------------------------------------------------------

def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _env_reset(task_id: str, session_id: str = None) -> Dict[str, Any]:
    payload = {"task_id": task_id}
    if session_id:
        payload["session_id"] = session_id
    return _post("/legacy/reset", payload)


def _env_step(session_id: str, action: str) -> Dict[str, Any]:
    return _post("/legacy/step", {"session_id": session_id, "action": action})


def _env_state(session_id: str) -> Dict[str, Any]:
    return _post("/legacy/state", {"session_id": session_id})


def _build_model_candidates() -> list[str]:
    models = [MODEL_NAME]
    extras = [item.strip() for item in MODEL_CANDIDATES.split(",") if item.strip()]
    for model in extras:
        if model not in models:
            models.append(model)
    return models


def _to_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# LLM action
# ---------------------------------------------------------------------------

def _llm_action(
    client: OpenAI,
    observation: Dict[str, Any],
    model_name: str,
    run_id: str,
    step_index: int,
) -> Dict[str, Any]:
    nonce = f"{run_id}:step:{step_index}:{uuid.uuid4().hex[:12]}"

    departments = observation.get("valid_actions", [])
    dept_str = ", ".join(set(
        a.split(":")[-1] for a in departments if a.startswith("assign:")
    ))

    prompt = (
        "You are an expert Customer Support Ticket Triage Agent. Your goal is to maximize the grader score (0.0 to 1.0) "
        "by routing tickets to the correct department with the right priority.\n\n"
        "SCORING WEIGHTS:\n"
        "- Accuracy (correct dept+priority): 40%\n"
        "- Speed (low wait times, SLA compliance): 25%\n"
        "- Completeness (resolve all tickets): 15%\n"
        "- Consistency & fairness: 20%\n\n"
        "AVAILABLE ACTIONS:\n"
        "- assign:<priority>:<department>  — Assign ticket with priority (low/medium/high/critical) to a department\n"
        "- escalate — Escalate to supervisor (use for VIP with urgent SLA or ambiguous tickets)\n"
        "- defer — Send to back of queue (use sparingly)\n"
        "- skip — Skip current ticket (BAD — only when no tickets available)\n"
        "- resolve:<ticket_id> — Directly resolve a pending ticket\n\n"
        f"Available departments: {dept_str}\n\n"
        "DECISION RULES:\n"
        "1. Read the ticket subject and hints carefully to determine department and priority\n"
        "2. VIP tickets need HIGH or CRITICAL priority\n"
        "3. If SLA remaining <= 2, act immediately (assign or escalate)\n"
        "4. If info is hidden, guess department from the subject keywords\n"
        "5. Check department capacity before assigning\n"
        "6. NEVER skip when tickets are pending\n\n"
        "DEPARTMENT MAPPING:\n"
        "- billing: invoices, charges, refunds, payments, subscriptions\n"
        "- technical: crashes, errors, APIs, SSL, performance, sync\n"
        "- general: info requests, feature requests, feedback, docs\n"
        "- account: passwords, 2FA, suspensions, merges, ownership\n"
        "- security: suspicious activity, breaches, unauthorized access, phishing\n\n"
        f"FORMAT: Output EXACTLY ONE line. Start with [NONCE:{nonce}] followed by a single space and the action string. No explanation."
    )

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
        extra_headers=EXTRA_HEADERS,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": json.dumps(observation)},
        ],
    )
    content = completion.choices[0].message.content if completion.choices else ""
    print(f"[DEBUG LLM RAW] {content}", file=sys.stderr)
    parsed_action = _extract_action(content or "")

    return {
        "action": parsed_action,
        "nonce": nonce,
        "raw_response": content or "",
        "reason": "llm_action_selected",
    }


def _extract_action(text: str) -> str:
    """Extract a valid action from LLM output."""
    if not text:
        return "skip"

    stripped = text.strip().lower()
    stripped = stripped.replace("```", " ").replace("`", " ")
    stripped = stripped.replace("\n", " ")
    stripped = re.sub(r"\s+", " ", stripped).strip()
    stripped = stripped.strip(" \t\r\n`'\".,;:!?()[]{}")

    if ACTION_PATTERN.match(stripped):
        return stripped

    # Try to find action patterns
    matches = re.findall(
        r"(assign:(?:low|medium|high|critical):\w+|escalate|defer|skip|resolve:\d+)",
        stripped,
    )
    if matches:
        return matches[-1]

    return "skip"


# ---------------------------------------------------------------------------
# Rule-based fallback
# ---------------------------------------------------------------------------

def _get_smart_fallback(observation: Dict[str, Any]) -> str:
    """Rule-based fallback when LLM is unavailable."""
    tid = observation.get("current_ticket_id", -1)
    if tid < 0:
        return "skip"

    subject = str(observation.get("current_ticket_subject", "")).lower()
    hint = str(observation.get("current_ticket_category_hint", "general")).lower()
    urgency = str(observation.get("current_ticket_urgency_hint", "")).lower()
    is_vip = observation.get("current_ticket_is_vip", False)
    sla_rem = observation.get("current_ticket_sla_remaining", 99)
    info_hidden = observation.get("current_ticket_info_hidden", False)

    dept_load = observation.get("department_load", {})
    dept_cap = observation.get("department_capacity", {})

    # Priority heuristic
    if "critical" in urgency or "immediate" in urgency:
        priority = "critical"
    elif "urgent" in urgency or "quick" in urgency:
        priority = "high"
    elif "moderate" in urgency or "business" in urgency:
        priority = "medium"
    else:
        priority = "low"

    if is_vip and priority in ("low", "medium"):
        priority = "high"

    # Department heuristic
    if info_hidden or hint == "unknown":
        department = _guess_department(subject)
    else:
        department = hint if hint in dept_load else _guess_department(subject)

    # Capacity check
    load = dept_load.get(department, 0)
    cap = dept_cap.get(department, 999)
    if load >= cap:
        for d in dept_load:
            if dept_load.get(d, 0) < dept_cap.get(d, 999):
                department = d
                break
        else:
            if is_vip or sla_rem <= 2:
                return "escalate"
            return "defer"

    if sla_rem <= 1:
        if is_vip:
            return "escalate"
        return f"assign:{priority}:{department}"

    return f"assign:{priority}:{department}"


def _guess_department(subject: str) -> str:
    """Guess department from subject keywords."""
    subject = subject.lower()
    if any(w in subject for w in ("invoice", "charge", "refund", "billing", "payment", "subscription")):
        return "billing"
    if any(w in subject for w in ("crash", "error", "api", "ssl", "sync", "performance", "bug")):
        return "technical"
    if any(w in subject for w in ("password", "account", "auth", "locked", "suspend", "merge", "ownership")):
        return "account"
    if any(w in subject for w in ("suspicious", "breach", "unauthorized", "phishing", "compromised", "security")):
        return "security"
    return "general"


def _sanitize_action(raw_text: str, observation: Dict[str, Any]) -> str:
    """Sanitize LLM output into a valid action."""
    extracted = _extract_action(str(raw_text or ""))
    if not extracted or extracted == "skip":
        return _get_smart_fallback(observation)

    action_mask = observation.get("action_mask", {})
    if isinstance(action_mask, dict) and extracted in action_mask:
        if action_mask[extracted]:
            return extracted

    # If action not in mask, fall back
    if ACTION_PATTERN.match(extracted):
        return extracted
    return _get_smart_fallback(observation)


# ---------------------------------------------------------------------------
# Main episode loop
# ---------------------------------------------------------------------------

def run_episode(task_id: str = "easy") -> int:
    if not OPENAI_API_KEY and not DISABLE_LLM:
        raise RuntimeError("OPENAI_API_KEY must be set")

    client = None
    if not DISABLE_LLM:
        client = OpenAI(
            api_key=OPENAI_API_KEY,
            base_url=API_BASE_URL,
            max_retries=OPENAI_MAX_RETRIES,
        )

    reset_result = _env_reset(task_id)
    observation = reset_result["observation"]
    session_id = observation["session_id"]

    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:10]}"
    model_candidates = _build_model_candidates()
    active_model_index = 0
    llm_enabled = not DISABLE_LLM
    forced_termination_reason = None

    done = False
    steps_taken = 0
    all_rewards: list[float] = []
    last_actions: list[str] = []
    final_info: Dict[str, Any] = reset_result.get("info", {})

    required_steps = {"easy": 20, "medium": 20, "hard": 20}
    effective_max_steps = required_steps.get(task_id, MAX_STEPS)

    try:
        for step_index in range(effective_max_steps):
            if done:
                break

            action = "skip"
            if llm_enabled and client is not None:
                try:
                    current_model = model_candidates[active_model_index]
                    llm_result = _llm_action(client, observation, current_model, run_id, step_index)
                    action = llm_result["action"]
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    msg = str(exc).lower()
                    if "404" in msg or "429" in msg:
                        if active_model_index + 1 < len(model_candidates):
                            active_model_index += 1
                        else:
                            llm_enabled = False
                    action = _get_smart_fallback(observation)
            else:
                action = _get_smart_fallback(observation)

            action = _sanitize_action(action, observation)

            try:
                step_result = _env_step(session_id, action)
            except (OSError, ValueError) as exc:
                print(f"[WARN] /step failed: {exc}", file=sys.stderr)
                try:
                    step_result = _env_step(session_id, _get_smart_fallback(observation))
                except (OSError, ValueError):
                    forced_termination_reason = "step_api_failure"
                    break

            observation = step_result["observation"]
            reward = float(step_result.get("reward", 0.0))
            done = bool(step_result.get("done", False))
            final_info = step_result.get("info", {})
            error = final_info.get("action_reason") if not final_info.get("action_valid", True) else None

            last_actions.append(action)
            all_rewards.append(reward)
            steps_taken += 1

            done_str = "true" if done else "false"
            error_str = error if error else "null"
            print(f"[STEP] step={step_index} action={action} reward={reward:.2f} done={done_str} error={error_str}")

        if not done and forced_termination_reason is None:
            forced_termination_reason = "max_steps_reached"

        if forced_termination_reason:
            try:
                state_result = _env_state(session_id)
                final_info = state_result.get("info", final_info)
            except (OSError, ValueError):
                pass

    finally:
        score = float(observation.get("score_estimate", 0.0))
        success = score > 0.0
        success_str = "true" if success else "false"
        rewards_str = ",".join(f"{r:.2f}" for r in all_rewards)
        print(f"[END] success={success_str} steps={steps_taken} score={score:.2f} rewards={rewards_str}")

    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1:
        chosen_task = sys.argv[1].strip().lower()
        raise SystemExit(run_episode(task_id=chosen_task))
    else:
        all_tasks = ["easy", "medium", "hard"]
        exit_code = 0
        for task in all_tasks:
            result = run_episode(task_id=task)
            if result != 0:
                exit_code = result
        raise SystemExit(exit_code)
