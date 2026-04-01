---
title: traffic_openenv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Traffic Control OpenEnv Environment

## Environment Description

This project provides a deterministic traffic control environment for OpenEnv Round-1. The environment simulates a single urban intersection where an agent controls signal phases under congestion, emergency priority pressure, and fairness constraints.

The goal is to maximize throughput while minimizing delay, emergency waiting time, and directional starvation.

## Action Space

The action space is a compact text grammar:

- `hold`
- `switch`
- `prioritize_emergency`
- `set_ns_green:<n>`
- `set_ew_green:<n>`

Actions are parser-friendly and deterministic. Invalid or malformed actions are handled safely and penalized.

## Observation Space

The observation space is structured JSON with stable keys, including:

- `session_id`, `task_id`, `timestep`, `max_steps`
- `current_phase`, `phase_remaining`
- `queue_ns`, `queue_ew`
- `emergency_ns`, `emergency_ew`
- `moved_ns`, `moved_ew`
- `total_wait_time`, `emergency_wait_time`
- `invalid_actions`, `fairness_gap`, `last_action_valid`
- `valid_actions`

## Tasks

Three deterministic tasks are included with increasing difficulty:

- `easy`: balanced traffic with low emergency frequency
- `medium`: rush-hour directional imbalance and periodic emergency arrivals
- `hard`: high congestion, emergency contention, and stricter fairness requirements

## Reward

Dense reward shaping is used for partial progress each step:

- positive throughput component
- delay and emergency-wait penalties
- fairness penalty for directional starvation
- invalid-action penalty
- terminal completion bonus

## Grader

The final score is deterministic and normalized to `[0.0, 1.0]`, using weighted metrics:

- throughput achieved vs target
- average delay
- emergency wait handling
- fairness gap
- invalid action rate

## Setup

### Local Python setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run server

```bash
uvicorn server.main:app --host 0.0.0.0 --port 7860
```

### API smoke checks

```bash
curl -X POST http://127.0.0.1:7860/reset -H "Content-Type: application/json" -d '{"task_id":"easy"}'
curl -X POST http://127.0.0.1:7860/state -H "Content-Type: application/json" -d '{"session_id":"<SESSION_ID>"}'
curl -X POST http://127.0.0.1:7860/step -H "Content-Type: application/json" -d '{"session_id":"<SESSION_ID>","action":"hold"}'
```

## Baseline

The baseline inference script is `inference.py` at repository root.

Environment variables:

- `API_BASE_URL` (default `http://127.0.0.1:7860`)
- `MODEL_NAME` (default `gemini-2.0-flash`)
- `GEMINI_API_KEY` (fallback to `HF_TOKEN`)

<!--
- OpenAI compatibility (future use):
	- `MODEL_NAME` (e.g., gpt-4o-mini)
	- `HF_TOKEN` (fallback to `OPENAI_API_KEY`)
-->

Run baseline:

```bash
export API_BASE_URL=http://127.0.0.1:7860
export MODEL_NAME=gemini-2.0-flash
export GEMINI_API_KEY=<YOUR_GEMINI_KEY>
python inference.py easy
```

## Docker

Build and run:

```bash
docker build -t traffic-env .
docker run --rm -p 7860:7860 traffic-env
```

## Hugging Face Space

This repository is compatible with Docker-based Hugging Face Spaces. Configure required secrets and deploy the repo root.

## OpenEnv

Validate packaging:

```bash
openenv validate
```

## Validation

Run project validator:

```bash
python validate_project.py
```
