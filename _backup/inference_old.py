from __future__ import annotations

import json
import os
import re
import sys
import time
import uuid
from typing import Any, Dict

import requests
from openai import OpenAI
from dotenv import load_dotenv


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
REPEAT_ACTION_LIMIT = int(os.getenv("REPEAT_ACTION_LIMIT", "50"))
NO_PROGRESS_WINDOW = int(os.getenv("NO_PROGRESS_WINDOW", "50"))
REWARD_FLAT_EPSILON = float(os.getenv("REWARD_FLAT_EPSILON", "0.0001"))
REPEAT_WARN_LIMIT = int(os.getenv("REPEAT_WARN_LIMIT", "8"))
TRACE_API = os.getenv("TRACE_API", "0").strip().lower() in {"1", "true", "yes", "on"}
TRACE_DIR = os.getenv("TRACE_DIR", "/tmp")
TRACE_BASENAME = os.getenv("TRACE_BASENAME", "inference_trace")
ENV_NAME = "traffic-control-openenv"

EXTRA_HEADERS = {
    "HTTP-Referer": "http://localhost",
    "X-Title": "traffic-openenv",
}

SAFE_FALLBACK = "hold"
ACTION_PATTERN = re.compile(
    r"^(hold|switch|prioritize_emergency|set_ns_green:\d+|set_ew_green:\d+)$"
)


# ---------------------------------------------------------------------------
# Environment API helpers — uses legacy endpoints for session persistence
# ---------------------------------------------------------------------------

def _post(path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{ENV_BASE_URL}{path}", json=payload, timeout=20)
    response.raise_for_status()
    return response.json()


def _env_reset(task_id: str, session_id: str = None) -> Dict[str, Any]:
    """Call /legacy/reset → {observation, reward, done, info}."""
    payload = {"task_id": task_id}
    if session_id:
        payload["session_id"] = session_id
    return _post("/legacy/reset", payload)


def _env_step(session_id: str, action: str) -> Dict[str, Any]:
    """Call /legacy/step → {observation, reward, done, info}."""
    return _post("/legacy/step", {"session_id": session_id, "action": action})


def _env_state(session_id: str) -> Dict[str, Any]:
    """Call /legacy/state → {observation, reward, done, info}."""
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


def _append_trace(trace_path: str, payload: Dict[str, Any]) -> None:
    with open(trace_path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _llm_action(
    client: OpenAI,
    observation: Dict[str, Any],
    model_name: str,
    run_id: str,
    step_index: int,
) -> Dict[str, Any]:
    nonce = f"{run_id}:step:{step_index}:{uuid.uuid4().hex[:12]}"
    prompt = (
        "You are a High-Performance Traffic Systems Engineer. Your absolute goal is to maximize the Grader Score (0.0 to 1.0). "
        "Weights: Throughput (45%, ^1.5 bonus), Emergency (30%, ^2.5 penalty), Fairness (15%), Discipline (10%).\n\n"
        "CRITICAL PHYSICS & REWARD CONSTRAINTS:\n"
        "1. PEDESTRIAN OVERRIDE: If 'pedestrian_waiting' is True, the ONLY legal moves are 'switch' or 'hold'. Anything else triggers a CATASTROPHIC SAFETY KILL.\n"
        "2. EMERGENCY EXPONENTIALS: Emergency wait is penalized ^2.5. If emergency_ns > 0 or emergency_ew > 0, use 'prioritize_emergency' immediately.\n"
        "3. THE SWITCHING TAX: Switching lights costs a -0.3 reward penalty. DO NOT switch for small queues (< 10 cars) unless a starvation event is imminent.\n"
        "4. KINEMATIC STARTUP: The first 2 seconds of green have 50% capacity. When clearing a large queue (> 15 cars), you MUST use 'set_ns_green:14' or 'set_ew_green:14'.\n"
        "5. TIMER RE-LOCK: Using 'set_xx' resets the timer. Use 'hold' on the turns following a 'set' command to let traffic flow.\n\n"
        "STRICT DECISION HIERARCHY (Evaluate in exact order):\n"
        "1. IF yellow_active: RETURN hold\n"
        "2. IF pedestrian_waiting:\n"
        "    - IF action_mask['switch']: RETURN switch\n"
        "    - ELSE: RETURN hold\n"
        "3. THROUGHPUT LOCK: IF phase_remaining > 2 AND ((current_phase == 'ns' AND queue_ns > 0) OR (current_phase == 'ew' AND queue_ew > 0)): RETURN hold\n"
        "4. IF (emergency_ns > 0 OR emergency_ew > 0) AND action_mask['prioritize_emergency']: RETURN prioritize_emergency\n"
        "5. IF current_phase == 'ns':\n"
        "    - IF queue_ns == 0 AND queue_ew > 0 AND action_mask['switch']: RETURN switch\n"
        "    - IF queue_ew > 40 AND action_mask['switch']: RETURN switch\n"
        "    - IF action_mask['set_ns_green']: RETURN set_ns_green:14\n"
        "    - RETURN hold\n"
        "6. IF current_phase == 'ew':\n"
        "    - IF queue_ew == 0 AND queue_ns > 0 AND action_mask['switch']: RETURN switch\n"
        "    - IF queue_ns > 40 AND action_mask['switch']: RETURN switch\n"
        "    - IF action_mask['set_ew_green']: RETURN set_ew_green:14\n"
        "    - RETURN hold\n\n"
        f"FORMAT: Output EXACTLY ONE line. Start with [NONCE:{nonce}] followed by a single space and the action. Do not include the word 'RETURN'. Do not include markdown."
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
    raw_text = content or ""
    parsed_action = _extract_action(raw_text)
    usage = getattr(completion, "usage", None)

    return {
        "action": parsed_action,
        "nonce": nonce,
        "raw_response": raw_text,
        "nonce_seen": f"[NONCE:{nonce}]" in raw_text,
        "response_id": getattr(completion, "id", None),
        "response_model": getattr(completion, "model", model_name),
        "fallback_status": "no",
        "reason": "llm_action_selected",
    }


def _extract_action(text: str) -> str:
    if not text:
        return SAFE_FALLBACK

    stripped = text.strip().lower()
    stripped = stripped.replace("```", " ").replace("`", " ")
    stripped = stripped.replace("\n", " ")
    stripped = re.sub(r"\s+", " ", stripped).strip()
    stripped = stripped.strip(" \t\r\n`'\".,;:!?()[]{}")

    if ACTION_PATTERN.match(stripped):
        return stripped

    if "prioritize" in stripped and "emergency" in stripped:
        return "prioritize_emergency"
    if "set_ns_green" in stripped and not re.search(r"set_ns_green:\d+", stripped):
        return "set_ns_green:10"
    if "set_ew_green" in stripped and not re.search(r"set_ew_green:\d+", stripped):
        return "set_ew_green:10"

    tokens = re.findall(r"(hold|switch|prioritize_emergency|set_ns_green:\d+|set_ew_green:\d+)", stripped)
    if tokens:
        for token in reversed(tokens):
            if ACTION_PATTERN.match(token):
                return token
    return SAFE_FALLBACK


def _get_smart_fallback(observation: Dict[str, Any]) -> str:
    action_mask = observation.get("action_mask", {})
    if not isinstance(action_mask, dict):
        return "hold"

    def _allowed(*keys: str) -> bool:
        return any(bool(action_mask.get(key, False)) for key in keys)

    if bool(observation.get("pedestrian_waiting", False)) and _allowed("switch"):
        return "switch"

    emergency_total = _to_int(observation.get("emergency_ns", 0)) + _to_int(observation.get("emergency_ew", 0))
    if emergency_total > 0 and _allowed("prioritize_emergency"):
        return "prioritize_emergency"

    queue_ns = max(0, _to_int(observation.get("queue_ns", 0)))
    queue_ew = max(0, _to_int(observation.get("queue_ew", 0)))

    if _allowed("set_ns_green") and _allowed("set_ew_green"):
        return "set_ns_green:10" if queue_ns >= queue_ew else "set_ew_green:10"
    if _allowed("set_ns_green"):
        return "set_ns_green:10"
    if _allowed("set_ew_green"):
        return "set_ew_green:10"

    current_phase = str(observation.get("current_phase", "ns"))
    if _allowed("switch"):
        if current_phase == "ns" and queue_ew > queue_ns:
            return "switch"
        if current_phase == "ew" and queue_ns > queue_ew:
            return "switch"

    return "hold"


def _sanitize_action(raw_text: str, observation: Dict[str, Any]) -> str:
    extracted = _extract_action(str(raw_text or ""))
    if not extracted:
        return _get_smart_fallback(observation)
    if extracted == "hold":
        return "hold"

    action_base = extracted.split(":", 1)[0]
    action_mask = observation.get("action_mask", {})
    if not isinstance(action_mask, dict):
        return extracted if ACTION_PATTERN.match(extracted) else _get_smart_fallback(observation)

    if action_base == "prioritize_emergency":
        emergency_total = _to_int(observation.get("emergency_ns", 0)) + _to_int(observation.get("emergency_ew", 0))
        if emergency_total <= 0:
            return "hold"
        if action_mask.get("prioritize_emergency", False):
            return extracted
        return "hold"

    if action_base in {"set_ns_green", "set_ew_green"}:
        if action_mask.get(action_base, False):
            return extracted
        return _get_smart_fallback(observation)

    if action_mask.get(action_base, False):
        return extracted
    return _get_smart_fallback(observation)


def _policy_override_action(observation: Dict[str, Any]) -> tuple[str | None, str | None]:
    action_mask = observation.get("action_mask", {})
    if not isinstance(action_mask, dict):
        return None, None

    if bool(observation.get("yellow_active", False)):
        return "hold", "policy_yellow_safety_hold"

    if bool(observation.get("pedestrian_waiting", False)):
        if bool(action_mask.get("switch", False)):
            return "switch", "policy_pedestrian_override"
        return "hold", "policy_pedestrian_hold"

    emergency_ns = _to_int(observation.get("emergency_ns", 0))
    emergency_ew = _to_int(observation.get("emergency_ew", 0))
    if (emergency_ns > 0 or emergency_ew > 0) and bool(action_mask.get("prioritize_emergency", False)):
        return "prioritize_emergency", "policy_emergency_priority"

    return None, None


def _anti_stall_action(observation: Dict[str, Any], current_action: str) -> tuple[str, str | None]:
    if current_action != "hold":
        return current_action, None

    action_mask = observation.get("action_mask", {})
    if not isinstance(action_mask, dict):
        return current_action, None

    if bool(observation.get("yellow_active", False)):
        return current_action, None
    if bool(observation.get("pedestrian_waiting", False)):
        return current_action, None
    if not bool(action_mask.get("switch", False)):
        return current_action, None

    queue_ns = max(0, _to_int(observation.get("queue_ns", 0)))
    queue_ew = max(0, _to_int(observation.get("queue_ew", 0)))
    current_phase = str(observation.get("current_phase", "ns"))

    should_switch = False
    if current_phase == "ns":
        should_switch = (queue_ns == 0 and queue_ew > 0) or (queue_ew > queue_ns + 4)
    elif current_phase == "ew":
        should_switch = (queue_ew == 0 and queue_ns > 0) or (queue_ns > queue_ew + 4)

    if should_switch:
        return "switch", "anti_stall_queue_rebalance"
    return current_action, None


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

    # STDOUT: [START] line
    print(f"[START] task={task_id} env={ENV_NAME} model={MODEL_NAME}")

    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:10]}"
    trace_path = os.path.join(TRACE_DIR, f"{TRACE_BASENAME}_{run_id}.jsonl")

    done = False
    steps_taken = 0
    model_candidates = _build_model_candidates()
    active_model_index = 0
    llm_enabled = not DISABLE_LLM
    forced_termination_reason: str | None = None
    last_actions: list[str] = []
    all_rewards: list[float] = []
    recent_rewards: list[float] = []
    recent_queue_totals: list[int] = []
    last_step_error: str | None = None
    final_info: Dict[str, Any] = reset_result.get("info", {})
    required_steps_by_task = {"easy": 50, "medium": 60, "hard": 70}
    required_min_steps = required_steps_by_task.get(task_id, MAX_STEPS)
    effective_max_steps = required_min_steps

    try:
        for step_index in range(effective_max_steps):
            if done:
                catastrophic_event = bool(observation.get("catastrophic_event", False))
                if catastrophic_event or steps_taken >= required_min_steps:
                    break
                done = False

            used_fallback_action = False
            llm_trace: Dict[str, Any] = {"action": SAFE_FALLBACK, "reason": None}

            if llm_enabled and client is not None:
                try:
                    current_model = model_candidates[active_model_index]
                    llm_trace = _llm_action(client, observation, current_model, run_id, step_index)
                    action = llm_trace["action"]
                except Exception as exc:
                    print(f"[WARN] LLM failed step={step_index}: {exc}", file=sys.stderr)
                    message = str(exc).lower()
                    if "404" in message or "429" in message:
                        if active_model_index + 1 < len(model_candidates):
                            active_model_index += 1
                        else:
                            llm_enabled = False
                    action = _get_smart_fallback(observation)
                    used_fallback_action = True
            else:
                action = _get_smart_fallback(observation)
                used_fallback_action = True

            model_action_text = str(llm_trace.get("action") or action)
            action = _sanitize_action(model_action_text, observation)
            policy_action, _ = _policy_override_action(observation)
            if policy_action is not None:
                action = _sanitize_action(policy_action, observation)
            action, _ = _anti_stall_action(observation, action)

            try:
                result = _env_step(session_id, action)
            except Exception as exc:
                print(f"[WARN] /step failed: {exc}", file=sys.stderr)
                try:
                    result = _env_step(session_id, _get_smart_fallback(observation))
                except Exception:
                    forced_termination_reason = "step_api_failure"
                    break

            observation = result["observation"]
            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            final_info = result.get("info", {})
            last_step_error = final_info.get("reason") if not final_info.get("action_valid", True) else None

            last_actions.append(action)
            if len(last_actions) > REPEAT_ACTION_LIMIT:
                last_actions.pop(0)

            all_rewards.append(reward)
            recent_rewards.append(reward)
            if len(recent_rewards) > NO_PROGRESS_WINDOW:
                recent_rewards.pop(0)

            queue_total = _to_int(observation.get("queue_ns", 0)) + _to_int(observation.get("queue_ew", 0))
            recent_queue_totals.append(queue_total)
            if len(recent_queue_totals) > NO_PROGRESS_WINDOW:
                recent_queue_totals.pop(0)

            # STDOUT: [STEP] line
            done_str = "true" if done else "false"
            error_str = last_step_error if last_step_error else "null"
            print(f"[STEP] step={step_index} action={action} reward={reward:.2f} done={done_str} error={error_str}")
            steps_taken += 1

            if done:
                catastrophic_event = bool(observation.get("catastrophic_event", False))
                if catastrophic_event:
                    break
                if steps_taken < required_min_steps:
                    done = False

            if steps_taken >= required_min_steps:
                repeated_action_loop = (
                    len(last_actions) == REPEAT_ACTION_LIMIT and len(set(last_actions)) == 1
                )
                reward_flat = (
                    len(recent_rewards) == NO_PROGRESS_WINDOW
                    and (max(recent_rewards) - min(recent_rewards)) <= REWARD_FLAT_EPSILON
                )
                queue_stalled = (
                    len(recent_queue_totals) == NO_PROGRESS_WINDOW
                    and len(set(recent_queue_totals)) == 1
                )
                if repeated_action_loop and (queue_stalled or reward_flat):
                    forced_termination_reason = "repeated_action_loop"
                    break
                if reward_flat and queue_stalled:
                    forced_termination_reason = "no_progress_detected"
                    break

        if not done and forced_termination_reason is None and steps_taken >= effective_max_steps:
            forced_termination_reason = "max_steps_reached"

        if forced_termination_reason is not None:
            try:
                state_result = _env_state(session_id)
                final_info = state_result.get("info", final_info)
            except Exception:
                pass

    finally:
        # STDOUT: [END] line — always emitted
        score = float(final_info.get("score_estimate", 0.0))
        success = (score > 0.0) and not bool(observation.get("catastrophic_event", False))
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
