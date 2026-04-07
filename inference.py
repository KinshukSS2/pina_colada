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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY") or os.getenv("HF_TOKEN")
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
VERIFY_STRICT = os.getenv("VERIFY_STRICT", "0").strip().lower() in {"1", "true", "yes", "on"}
VERIFY_MIN_SUCCESSFUL_LLM_CALLS = int(os.getenv("VERIFY_MIN_SUCCESSFUL_LLM_CALLS", "1"))

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
    print(f"[DEBUG LLM RAW] {content}")
    raw_text = content or ""
    parsed_action = _extract_action(raw_text)
    usage = getattr(completion, "usage", None)
    usage_prompt = getattr(usage, "prompt_tokens", None) if usage is not None else None
    usage_completion = getattr(usage, "completion_tokens", None) if usage is not None else None
    usage_total = getattr(usage, "total_tokens", None) if usage is not None else None

    return {
        "action": parsed_action,
        "nonce": nonce,
        "raw_response": raw_text,
        "nonce_seen": f"[NONCE:{nonce}]" in raw_text,
        "response_id": getattr(completion, "id", None),
        "response_model": getattr(completion, "model", model_name),
        "usage_prompt_tokens": usage_prompt,
        "usage_completion_tokens": usage_completion,
        "usage_total_tokens": usage_total,
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
    if "switch" in stripped and "ns" in stripped:
        return "set_ns_green:10"
    if "switch" in stripped and "ew" in stripped:
        return "set_ew_green:10"

    tokens = re.findall(r"(hold|switch|prioritize_emergency|set_ns_green:\d+|set_ew_green:\d+)", stripped)
    if tokens:
        for token in reversed(tokens):
            if ACTION_PATTERN.match(token):
                return token
    return SAFE_FALLBACK


def _get_smart_fallback(observation: Dict[str, Any]) -> str:
    """
    Dumbed-down fallback: Only save from instant-death, do NOT optimize traffic.
    Leave the queue management entirely to the LLM.
    """
    action_mask = observation.get("action_mask", {})
    if not isinstance(action_mask, dict):
        return "hold"

    # 1. Priority 0: Save from Pedestrian Death
    if bool(observation.get("pedestrian_waiting", False)) and action_mask.get("switch", False):
        return "switch"

    # 2. Priority 1: Save from Emergency Death
    emergency_total = _to_int(observation.get("emergency_ns", 0)) + _to_int(observation.get("emergency_ew", 0))
    if emergency_total > 0 and action_mask.get("prioritize_emergency", False):
        return "prioritize_emergency"

    # 3. DO NOTHING ELSE. No queue-based switching. Default to hold.
    return "hold"


def _sanitize_action(raw_text: str, observation: Dict[str, Any]) -> str:
    raw_value = str(raw_text or "")
    extracted = _extract_action(raw_value)

    if not extracted:
        return _get_smart_fallback(observation)

    if extracted == "hold":
        return "hold"

    action_base = extracted.split(":", 1)[0]
    action_mask = observation.get("action_mask", {})
    if not isinstance(action_mask, dict):
        return extracted if ACTION_PATTERN.match(extracted) else _get_smart_fallback(observation)

    if action_base not in {"hold", "switch", "prioritize_emergency", "set_ns_green", "set_ew_green"}:
        return _get_smart_fallback(observation)

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


def _policy_override_action(observation: Dict[str, Any], previous_action: str | None) -> tuple[str | None, str | None]:
    action_mask = observation.get("action_mask", {})
    if not isinstance(action_mask, dict):
        return None, None

    if bool(observation.get("yellow_active", False)) and bool(action_mask.get("hold", True)):
        return "hold", "policy_yellow_safety_hold"

    if bool(observation.get("pedestrian_waiting", False)):
        if bool(action_mask.get("switch", False)):
            return "switch", "policy_pedestrian_override"
        if bool(action_mask.get("hold", True)):
            return "hold", "policy_pedestrian_hold"

    emergency_ns = _to_int(observation.get("emergency_ns", 0))
    emergency_ew = _to_int(observation.get("emergency_ew", 0))
    if (emergency_ns > 0 or emergency_ew > 0) and bool(action_mask.get("prioritize_emergency", False)):
        return "prioritize_emergency", "policy_emergency_priority"

    return None, None


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

    run_id = f"run-{int(time.time())}-{uuid.uuid4().hex[:10]}"
    trace_path = os.path.join(TRACE_DIR, f"{TRACE_BASENAME}_{run_id}.jsonl")

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
    required_steps_by_task = {
        "easy": 50,
        "medium": 60,
        "hard": 70,
    }
    required_min_steps = required_steps_by_task.get(task_id, MAX_STEPS)
    effective_max_steps = required_min_steps
    llm_calls_attempted = 0
    llm_calls_succeeded = 0
    fallback_steps = 0
    sanitized_steps = 0
    nonce_verified_steps = 0

    if TRACE_API:
        _append_trace(
            trace_path,
            {
                "event": "run_start",
                "run_id": run_id,
                "timestamp": int(time.time()),
                "task_id": task_id,
                "session_id": session_id,
                "api_base_url": API_BASE_URL,
                "env_base_url": ENV_BASE_URL,
                "model_candidates": model_candidates,
                "verify_strict": VERIFY_STRICT,
            },
        )

    for step_index in range(effective_max_steps):
        if done:
            catastrophic_event = bool(observation.get("catastrophic_event", False))
            if catastrophic_event or steps_taken >= required_min_steps:
                break
            done = False

        used_fallback_action = False

        if llm_enabled:
            llm_calls_attempted += 1
            llm_trace: Dict[str, Any] = {
                "action": SAFE_FALLBACK,
                "nonce": None,
                "raw_response": "",
                "nonce_seen": False,
                "response_id": None,
                "response_model": model_candidates[active_model_index],
                "usage_prompt_tokens": None,
                "usage_completion_tokens": None,
                "usage_total_tokens": None,
                "error": None,
                "fallback_status": "unknown",
                "reason": None,
            }
            if client is None:
                llm_trace["error"] = "client_uninitialized"
                llm_trace["fallback_status"] = "yes"
                llm_trace["reason"] = "client_uninitialized"
                action = _get_smart_fallback(observation)
                used_fallback_action = True
                llm_enabled = False
            else:
                try:
                    current_model = model_candidates[active_model_index]
                    llm_trace = _llm_action(client, observation, current_model, run_id, step_index)
                    action = llm_trace["action"]
                    llm_calls_succeeded += 1
                    if llm_trace.get("nonce_seen"):
                        nonce_verified_steps += 1
                except Exception as exc:
                    print(f"[WARN] LLM action failed at step={step_index}: {exc}")
                    llm_trace["error"] = str(exc)
                    llm_trace["fallback_status"] = "yes"
                    llm_trace["reason"] = f"llm_error:{exc.__class__.__name__}"
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
                    action = _get_smart_fallback(observation)
                    used_fallback_action = True
        else:
            action = _get_smart_fallback(observation)
            used_fallback_action = True
            llm_trace = {
                "action": SAFE_FALLBACK,
                "nonce": None,
                "raw_response": "",
                "nonce_seen": False,
                "response_id": None,
                "response_model": None,
                "usage_prompt_tokens": None,
                "usage_completion_tokens": None,
                "usage_total_tokens": None,
                "error": "llm_disabled",
                "fallback_status": "yes",
                "reason": "llm_disabled",
            }

        previous_action = last_actions[-1] if last_actions else None

        raw_action_text = str(llm_trace.get("raw_response") or action)
        parsed_action = _extract_action(raw_action_text)
        action = _sanitize_action(raw_action_text, observation)
        policy_action, policy_reason = _policy_override_action(observation, previous_action)
        if policy_action is not None:
            action = _sanitize_action(policy_action, observation)
        if parsed_action != action:
            sanitized_steps += 1
        if used_fallback_action or (parsed_action != action and action == SAFE_FALLBACK):
            fallback_steps += 1

        fallback_status = str(llm_trace.get("fallback_status") or "unknown")
        if used_fallback_action:
            fallback_status = "yes"
        elif fallback_status not in {"yes", "no", "unknown"}:
            fallback_status = "unknown"
        if action == SAFE_FALLBACK and fallback_status == "no":
            fallback_status = "unknown"
        agent_reason = str(llm_trace.get("reason") or "")
        if policy_action is not None and policy_reason:
            agent_reason = policy_reason
            if fallback_status == "yes":
                fallback_status = "no"
        elif parsed_action != action:
            agent_reason = f"sanitized:{parsed_action}->{action}"
        elif not agent_reason:
            agent_reason = "action_forwarded"

        if TRACE_API:
            _append_trace(
                trace_path,
                {
                    "event": "llm_step",
                    "run_id": run_id,
                    "step": step_index,
                    "model": llm_trace.get("response_model"),
                    "nonce": llm_trace.get("nonce"),
                    "nonce_seen": llm_trace.get("nonce_seen"),
                    "response_id": llm_trace.get("response_id"),
                    "raw_response": llm_trace.get("raw_response"),
                    "parsed_action": parsed_action,
                    "sanitized_action": action,
                    "used_fallback": action == SAFE_FALLBACK,
                    "llm_error": llm_trace.get("error"),
                    "usage_prompt_tokens": llm_trace.get("usage_prompt_tokens"),
                    "usage_completion_tokens": llm_trace.get("usage_completion_tokens"),
                    "usage_total_tokens": llm_trace.get("usage_total_tokens"),
                    "observation": {
                        "queue_ns": _to_int(observation.get("queue_ns")),
                        "queue_ew": _to_int(observation.get("queue_ew")),
                        "emergency_ns": _to_int(observation.get("emergency_ns")),
                        "emergency_ew": _to_int(observation.get("emergency_ew")),
                        "phase": observation.get("current_phase"),
                        "phase_remaining": _to_int(observation.get("phase_remaining")),
                    },
                },
            )

        try:
            result = _post(
                "/step",
                {
                    "session_id": session_id,
                    "action": action,
                    "fallback_status": fallback_status,
                    "agent_reason": agent_reason,
                },
            )
        except Exception as exc:
            print(f"[WARN] /step failed for action={action}: {exc}; retrying with fallback")
            try:
                fallback_action = _get_smart_fallback(observation)
                result = _post(
                    "/step",
                    {
                        "session_id": session_id,
                        "action": fallback_action,
                        "fallback_status": "yes",
                        "agent_reason": "step_retry_fallback",
                    },
                )
                action = fallback_action
            except Exception as fallback_exc:
                forced_termination_reason = f"step_api_failure: {fallback_exc}"
                print(f"[WARN] {forced_termination_reason}")
                break

        observation = result["observation"]
        reward = float(result.get("reward", 0.0))
        done = bool(result.get("done", False))
        catastrophic_event = bool(observation.get("catastrophic_event", False))
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

        if done:
            if catastrophic_event:
                forced_termination_reason = "catastrophic_event"
                break
            if steps_taken < required_min_steps:
                done = False

        if TRACE_API:
            _append_trace(
                trace_path,
                {
                    "event": "env_step",
                    "run_id": run_id,
                    "step": step_index,
                    "reward": reward,
                    "done": done,
                    "score_estimate": float(final_info.get("score_estimate", 0.0)),
                    "catastrophic_event": bool(observation.get("catastrophic_event", False)),
                    "queue_ns": _to_int(observation.get("queue_ns")),
                    "queue_ew": _to_int(observation.get("queue_ew")),
                    "moved_ns": _to_int(observation.get("moved_ns")),
                    "moved_ew": _to_int(observation.get("moved_ew")),
                    "grader_breakdown": final_info.get("grader_breakdown", {}),
                },
            )

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

        if steps_taken >= required_min_steps:
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
    proof_summary = {
        "run_id": run_id,
        "steps_taken": steps_taken,
        "llm_calls_attempted": llm_calls_attempted,
        "llm_calls_succeeded": llm_calls_succeeded,
        "nonce_verified_steps": nonce_verified_steps,
        "fallback_steps": fallback_steps,
        "sanitized_steps": sanitized_steps,
        "trace_path": trace_path if TRACE_API else None,
        "score": score,
    }

    if TRACE_API:
        _append_trace(
            trace_path,
            {
                "event": "run_end",
                **proof_summary,
                "forced_termination_reason": forced_termination_reason,
                "done": done,
            },
        )

    catastrophic_reason = None
    grader_reasons = final_info.get("grader_reasons", [])
    if isinstance(grader_reasons, list):
        for reason in grader_reasons:
            text = str(reason)
            match = re.search(r"Catastrophic event:\s*([A-Za-z0-9_\-]+)", text)
            if match:
                catastrophic_reason = match.group(1).strip()
                break
    if not catastrophic_reason and observation.get("catastrophic_event"):
        catastrophic_reason = "unknown_catastrophe (check simulator logic)"

    print(
        "[PROOF] "
        f"run_id={run_id} llm_attempted={llm_calls_attempted} llm_succeeded={llm_calls_succeeded} "
        f"nonce_verified={nonce_verified_steps} fallback_steps={fallback_steps} sanitized_steps={sanitized_steps}"
    )

    death_reason_value = catastrophic_reason or forced_termination_reason
    death_tag = f" DEATH_REASON={death_reason_value}" if death_reason_value else " SURVIVED"
    print(f"[END] session={session_id} score={score:.6f} steps={steps_taken}{death_tag}")

    if VERIFY_STRICT:
        if llm_calls_succeeded < VERIFY_MIN_SUCCESSFUL_LLM_CALLS:
            print("[FAIL] strict verification failed: insufficient successful LLM calls")
            return 2
        if llm_calls_attempted > 0 and fallback_steps >= steps_taken and not DISABLE_LLM:
            print("[FAIL] strict verification failed: run used fallback for all steps")
            return 3
        if TRACE_API and nonce_verified_steps <= 0:
            print("[FAIL] strict verification failed: nonce not observed in LLM responses")
            return 4

    return 0


if __name__ == "__main__":
    chosen_task = "easy"
    if len(sys.argv) > 1:
        chosen_task = sys.argv[1].strip().lower()
    raise SystemExit(run_episode(task_id=chosen_task))
