from __future__ import annotations

import uvicorn

from openenv.core.env_server.http_server import create_app

from env.environment import TrafficControlEnvironment
from env.models import TrafficAction, TrafficObservation

HOST = "0.0.0.0"
PORT = 7860

app = create_app(
    TrafficControlEnvironment,
    TrafficAction,
    TrafficObservation,
    env_name="traffic_control",
    max_concurrent_envs=10,
)


# ------------------------------------------------------------------
# Custom endpoints for backward-compatible session-based HTTP access
# and grading. These supplement the standard OpenEnv endpoints.
# ------------------------------------------------------------------
from typing import Any, Dict, Optional
from pydantic import BaseModel
from env.grader import compute_grade
from env.models import EpisodeSummary
from env.tasks import get_task, task_catalog

_legacy_env = TrafficControlEnvironment()


def _task_summaries() -> list[dict]:
    return [
        {
            "id": task.task_id,
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "description": task.description,
            "grader": "env.grader:compute_grade",
            "has_grader": True,
        }
        for task in task_catalog().values()
    ]


@app.get("/tasks")
def get_tasks() -> list[dict]:
    return _task_summaries()


class LegacyResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: str = "easy"
    seed: Optional[int] = None
    sensor_noise: bool = False
    ood_start: bool = False


class LegacyStepRequest(BaseModel):
    session_id: str
    action: str
    fallback_status: Optional[str] = "unknown"
    agent_reason: Optional[str] = None


class LegacyStateRequest(BaseModel):
    session_id: str


class GradeRequest(BaseModel):
    task_id: str
    summary: Dict[str, Any]


@app.post("/legacy/reset")
def legacy_reset(request: Optional[LegacyResetRequest] = None) -> dict:
    if request is None:
        return _legacy_env.legacy_reset()
    return _legacy_env.legacy_reset(
        task_id=request.task_id,
        session_id=request.session_id,
        seed=request.seed,
        sensor_noise=request.sensor_noise,
        ood_start=request.ood_start,
    )


@app.post("/legacy/step")
def legacy_step(request: LegacyStepRequest) -> dict:
    return _legacy_env.legacy_step(
        session_id=request.session_id,
        action_text=request.action,
    )


@app.post("/legacy/state")
def legacy_state(request: LegacyStateRequest) -> dict:
    return _legacy_env.legacy_state(session_id=request.session_id)


@app.post("/grade")
def grade(request: GradeRequest) -> dict:
    task = get_task(request.task_id)
    summary = EpisodeSummary(**request.summary)
    result = compute_grade(summary, task)
    return {
        "score": result.score,
        "breakdown": result.breakdown,
        "reasons": result.reasons,
    }


def main() -> None:
    uvicorn.run("server.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()