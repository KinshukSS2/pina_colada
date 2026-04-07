from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from env.environment import TrafficControlEnvironment
from env.grader import compute_grade
from env.models import EpisodeSummary, ResetRequest, StateRequest, StepRequest
from env.tasks import get_task, task_catalog

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
            "grader": "env.grader:compute_grade",
            "has_grader": True,
        }
        for task in task_catalog().values()
    ]





@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": "traffic-control-openenv",
        "description": "Advanced deterministic traffic benchmark testing POMDP resilience, spatial reasoning, and dynamic constraints.",
        "entry_point": "server.main:app",
        "tasks": _task_summaries(),
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": {
            "type": "text",
            "grammar": ["hold", "switch", "prioritize_emergency", "set_ns_green:<n>", "set_ew_green:<n>"],
        },
        "observation": {
            "type": "json",
            "schema": "Observation",
        },
        "state": {
            "type": "json",
            "schema": "IntersectionState",
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


class GradeRequest(BaseModel):
    task_id: str
    summary: Dict[str, Any]


@app.post("/grade")
def grade(request: GradeRequest) -> dict:
    """Run the grader for a given task and episode summary."""
    task = get_task(request.task_id)
    summary = EpisodeSummary(**request.summary)
    result = compute_grade(summary, task)
    return {
        "score": result.score,
        "breakdown": result.breakdown,
        "reasons": result.reasons,
    }


@app.post("/mcp")
async def mcp_endpoint(request: Request) -> JSONResponse:
    """Minimal MCP JSON-RPC 2.0 endpoint for openenv runtime validation."""
    try:
        body = await request.json()
    except Exception:
        body = {}

    request_id = body.get("id", 1) if isinstance(body, dict) else 1
    method = body.get("method", "") if isinstance(body, dict) else ""

    if method == "tools/list":
        return JSONResponse({
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {"tools": []},
        })

    # Default: return a valid JSON-RPC 2.0 response for any request
    # (including empty body from the validator ping)
    return JSONResponse({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"tools": []},
    })


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server.main:app", host=HOST, port=PORT, reload=False)

