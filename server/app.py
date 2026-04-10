from __future__ import annotations

import uvicorn  # pylint: disable=import-error

from openenv.core.env_server.http_server import create_app  # pylint: disable=import-error

from env.environment import TicketTriageEnvironment, task_catalog, _grade_trajectory, _SESSIONS
from env.schemas import TriageAction, TriageObservation, SimState

HOST = "0.0.0.0"
PORT = 7860

app = create_app(
    TicketTriageEnvironment,
    TriageAction,
    TriageObservation,
    env_name="ticket_triage",
    max_concurrent_envs=10,
)


@app.get("/")
def root() -> dict:
    return {
        "service": "ticket_triage_openenv",
        "status": "ok",
        "docs": "/docs",
    }


# ------------------------------------------------------------------
# Custom endpoints for backward-compatible session-based HTTP access
# and grading. These supplement the standard OpenEnv endpoints.
# ------------------------------------------------------------------
from typing import Optional
from pydantic import BaseModel  # pylint: disable=import-error

_legacy_env = TicketTriageEnvironment()


def _task_summaries() -> list[dict]:
    return [
        {
            "id": task.task_id,
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "description": task.description,
            "grader": f"graders.{task.difficulty}_grader",
            "has_grader": True,
        }
        for task in task_catalog().values()
    ]


@app.get("/tasks")
def get_tasks() -> list[dict]:
    return _task_summaries()


@app.get("/schema")
def get_schema() -> dict:
    return {
        "action": TriageAction.model_json_schema(),
        "observation": TriageObservation.model_json_schema(),
    }


class LegacyResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_id: str = "easy"
    seed: Optional[int] = None


class LegacyStepRequest(BaseModel):
    session_id: str
    action: str
    fallback_status: Optional[str] = "unknown"
    agent_reason: Optional[str] = None


class LegacyStateRequest(BaseModel):
    session_id: str


class GradeRequest(BaseModel):
    task_id: str
    session_id: str


@app.post("/legacy/reset")
def legacy_reset(request: Optional[LegacyResetRequest] = None) -> dict:
    if request is None:
        return _legacy_env.legacy_reset()
    return _legacy_env.legacy_reset(
        task_id=request.task_id,
        session_id=request.session_id,
        seed=request.seed,
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
    session = _SESSIONS.get(request.session_id)
    if session is None:
        return {"score": 0.0, "breakdown": {}, "reasons": ["session_not_found"]}
    state: SimState = session["state"]
    score, breakdown, reasons = _grade_trajectory(request.task_id, state.trajectory)
    return {
        "score": score,
        "breakdown": breakdown,
        "reasons": reasons,
    }


def main() -> None:
    uvicorn.run("server.app:app", host=HOST, port=PORT, reload=False)


if __name__ == "__main__":
    main()