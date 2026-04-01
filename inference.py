from __future__ import annotations

import json
import os
import re
import sys
from typing import Any, Dict

import requests
from openai import OpenAI


ENV_BASE_URL = os.getenv("ENV_BASE_URL", "http://127.0.0.1:7860")
API_BASE_URL = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "openai/gpt-4o-mini")
MODEL_CANDIDATES = os.getenv(
    "MODEL_CANDIDATES",
    "openai/gpt-4o-mini,openai/gpt-4.1-mini,openai/gpt-4o",
)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_TIMEOUT_SECONDS = float(os.getenv("LLM_TIMEOUT_SECONDS", "20"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "1"))
DISABLE_LLM = os.getenv("DISABLE_LLM", "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_STEPS = int(os.getenv("MAX_STEPS", "100"))
REPEAT_ACTION_LIMIT = int(os.getenv("REPEAT_ACTION_LIMIT", "12"))
NO_PROGRESS_WINDOW = int(os.getenv("NO_PROGRESS_WINDOW", "6"))
REWARD_FLAT_EPSILON = float(os.getenv("REWARD_FLAT_EPSILON", "0.0001"))
REPEAT_WARN_LIMIT = int(os.getenv("REPEAT_WARN_LIMIT", "8"))

EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "traffic-openenv",
}

SAFE_FALLBACK = "hold"
ACTION_PATTERN = re.compile(
    r"^(hold|switch|prioritize_emergency|set_ns_green:\d+|set_ew_green:\d+)$"
)


def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _build_model_candidates() -> list[str]:
    models = [MODEL_NAME]
    extras = [item.strip() for item in MODEL_CANDIDATES.split(",") if item.strip()]
    for model in extras:
        if model not in models:
            models.append(model)
    return models


def _llm_action(client: OpenAI, observation: Dict[str, Any], model_name: str) -> str:
    prompt = (
        "You are controlling one traffic intersection. "
        "Return exactly one action string from: hold, switch, prioritize_emergency, "
        "set_ns_green:<n>, set_ew_green:<n>. "
        "Use integer n between 2 and 14. Output only action text with no explanation. "
        "Prefer prioritize_emergency when emergency queues are non-zero. "
        "Prefer setting green on the direction with larger queue when imbalance exists. "
        "Avoid repeating the same action many times unless it is clearly improving queues. "
        "Use hold only when keeping current phase is clearly better than switching."
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
    return _extract_action(content or "")


def _extract_action(text: str) -> str:
    if not text:
        return SAFE_FALLBACK

    stripped = text.strip().lower()
    if ACTION_PATTERN.match(stripped):
        return stripped

    if "prioritize" in stripped and "emergency" in stripped:
        return "prioritize_emergency"
    if "set_ns_green" in stripped and not re.search(r"set_ns_green:\d+", stripped):
        return "set_ns_green:10"
    if "set_ew_green" in stripped and not re.search(r"set_ew_green:\d+", stripped):
        return "set_ew_green:10"
    if "switch" in stripped and "ns" in stripped:
        return "set_ns_green:10"
    if "switch" in stripped and "ew" in stripped:
        return "set_ew_green:10"

    for token in re.findall(r"(hold|switch|prioritize_emergency|set_ns_green:\d+|set_ew_green:\d+)", stripped):
        if ACTION_PATTERN.match(token):
            return token
    return SAFE_FALLBACK


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

    reset_payload = {"task_id": task_id}
    reset_result = _post("/reset", reset_payload)

    observation = reset_result["observation"]
    session_id = observation["session_id"]
    print(f"[START] session={session_id} task={observation['task_id']}")

    done = False
    steps_taken = 0
    model_candidates = _build_model_candidates()
    active_model_index = 0
    llm_enabled = not DISABLE_LLM
    forced_termination_reason: str | None = None
    last_actions: list[str] = []
    recent_rewards: list[float] = []
    recent_queue_totals: list[int] = []
    repeated_warning_action: str | None = None
    final_info: Dict[str, Any] = reset_result.get("info", {})
    env_max_steps = int(observation.get("max_steps", 60))
    effective_max_steps = min(env_max_steps, MAX_STEPS)

    for step_index in range(effective_max_steps):
        if done:
            break

        if llm_enabled:
            try:
                current_model = model_candidates[active_model_index]
                action = _llm_action(client, observation, current_model)
            except Exception as exc:
                print(f"[WARN] LLM action failed at step={step_index}: {exc}")
                message = str(exc).lower()
                if "404" in message or "429" in message:
                    if active_model_index + 1 < len(model_candidates):
                        active_model_index += 1
                        print(
                            f"[WARN] Switching model to {model_candidates[active_model_index]} and continuing"
                        )
                    else:
                        llm_enabled = False
                        print("[WARN] Disabling LLM calls for remaining steps; using safe fallback action")
                action = SAFE_FALLBACK
        else:
            action = SAFE_FALLBACK

        try:
            result = _post("/step", {"session_id": session_id, "action": action})
        except Exception as exc:
            print(f"[WARN] /step failed for action={action}: {exc}; retrying with fallback")
            try:
                result = _post("/step", {"session_id": session_id, "action": SAFE_FALLBACK})
                action = SAFE_FALLBACK
            except Exception as fallback_exc:
                forced_termination_reason = f"step_api_failure: {fallback_exc}"
                print(f"[WARN] {forced_termination_reason}")
                break

        observation = result["observation"]
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        final_info = result.get("info", {})

        last_actions.append(action)
        if len(last_actions) > REPEAT_ACTION_LIMIT:
            last_actions.pop(0)

        recent_rewards.append(reward)
        if len(recent_rewards) > NO_PROGRESS_WINDOW:
            recent_rewards.pop(0)

        queue_total = int(observation.get("queue_ns", 0)) + int(observation.get("queue_ew", 0))
        recent_queue_totals.append(queue_total)
        if len(recent_queue_totals) > NO_PROGRESS_WINDOW:
            recent_queue_totals.pop(0)

        print(f"[STEP] t={step_index} action={action} reward={reward:.4f} done={done}")
        steps_taken += 1

        repeated_action_loop = (
            len(last_actions) == REPEAT_ACTION_LIMIT and len(set(last_actions)) == 1
        )
        repeated_action_warning = (
            len(last_actions) >= REPEAT_WARN_LIMIT and len(set(last_actions[-REPEAT_WARN_LIMIT:])) == 1
        )
        reward_flat = (
            len(recent_rewards) == NO_PROGRESS_WINDOW
            and (max(recent_rewards) - min(recent_rewards)) <= REWARD_FLAT_EPSILON
        )
        queue_stalled = (
            len(recent_queue_totals) == NO_PROGRESS_WINDOW
            and len(set(recent_queue_totals)) == 1
        )

        moved_total = int(observation.get("moved_ns", 0)) + int(observation.get("moved_ew", 0))
        if step_index == 0:
            previous_moved_total = moved_total
        moved_progress = moved_total - previous_moved_total
        previous_moved_total = moved_total
        moved_stalled = moved_progress <= 0

        if repeated_action_warning:
            if repeated_warning_action != last_actions[-1]:
                print(f"[WARN] repeated_action_warning:{last_actions[-1]} streak={REPEAT_WARN_LIMIT}")
                repeated_warning_action = last_actions[-1]
        else:
            repeated_warning_action = None

        if repeated_action_loop and (queue_stalled or reward_flat or moved_stalled):
            forced_termination_reason = f"repeated_action_loop:{last_actions[-1]}"
            print(f"[WARN] {forced_termination_reason}; forcing termination")
            break

        if reward_flat and queue_stalled:
            forced_termination_reason = "no_progress_detected"
            print("[WARN] no_progress_detected; forcing termination")
            break

    if not done and forced_termination_reason is None and steps_taken >= effective_max_steps:
        forced_termination_reason = "max_steps_reached"
        print("[WARN] Reached max steps. Forcing termination.")

    if forced_termination_reason is not None:
        try:
            state_result = _post("/state", {"session_id": session_id})
            final_info = state_result.get("info", final_info)
            done = bool(state_result.get("done", done))
        except Exception as exc:
            print(f"[WARN] Failed to fetch final state after forced termination: {exc}")

    score = float(final_info.get("score_estimate", 0.0))
    print(f"[END] session={session_id} score={score:.6f} steps={steps_taken}")
    return 0


if __name__ == "__main__":
    chosen_task = "easy"
    if len(sys.argv) > 1:
        chosen_task = sys.argv[1].strip().lower()
    raise SystemExit(run_episode(task_id=chosen_task))
