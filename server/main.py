from __future__ import annotations

from typing import Optional

from fastapi import FastAPI

from env.environment import TrafficControlEnvironment
from env.models import ResetRequest, StateRequest, StepRequest
from env.tasks import task_catalog

app = FastAPI(title="Traffic Control OpenEnv")
environment = TrafficControlEnvironment()

# Deployment markers for validators and runtime defaults.
HOST = "0.0.0.0"
PORT = 7860


def _task_summaries() -> list[dict]:
    return [
        {
            "id": task.task_id,
            "task_id": task.task_id,
            "difficulty": task.difficulty,
            "description": task.description,
            "grader": True,
            "has_grader": True,
        }
        for task in task_catalog().values()
    ]


@app.get("/")
def root() -> dict:
    return {
        "service": "traffic-openenv",
        "status": "ok",
        "endpoints": ["/reset", "/step", "/state", "/tasks", "/health", "/metadata", "/schema"],
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "traffic-control-openenv",
        "entry_point": "server.main:app",
        "tasks": _task_summaries(),
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "actions": {
            "type": "text",
            "grammar": ["hold", "switch", "prioritize_emergency", "set_ns_green:<n>", "set_ew_green:<n>"],
        },
        "observations": {
            "type": "json",
            "schema": "Observation",
        },
        "tasks": _task_summaries(),
    }


@app.get("/tasks")
def get_tasks() -> list[dict]:
    return _task_summaries()


@app.post("/reset")
def reset(request: Optional[ResetRequest] = None) -> dict:
    if request is None:
        task_id = "easy"
        session_id = None
        seed = None
        sensor_noise = False
        ood_start = False
    else:
        task_id = request.task_id
        session_id = request.session_id
        seed = request.seed
        sensor_noise = request.sensor_noise
        ood_start = request.ood_start

    result = environment.reset(
        task_id=task_id,
        session_id=session_id,
        seed=seed,
        sensor_noise=sensor_noise,
        ood_start=ood_start,
    )
    return result.model_dump()


@app.post("/step")
def step(request: StepRequest) -> dict:
    result = environment.step(
        session_id=request.session_id,
        action_text=request.action,
        fallback_status=request.fallback_status or "unknown",
        agent_reason=request.agent_reason,
    )
    return result.model_dump()


@app.post("/state")
def state(request: StateRequest) -> dict:
    result = environment.state(session_id=request.session_id)
    return result.model_dump()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=False)
