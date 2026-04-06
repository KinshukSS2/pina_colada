---
title: traffic_openenv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# OpenEnv Traffic Control: System Optimization & State-Space Balancing

**Author:** Kinshuk Sanand  
**Event:** Meta PyTorch OpenEnv Hackathon x SST | India AI Hackathon'26

This project provides a deterministic traffic control environment and an optimized LLM-driven baseline policy for intelligent traffic light management. The objective is to maximize throughput, minimize starvation, and prioritize emergencies while preserving safety and deterministic behavior.

The core contribution is mathematical profiling and rebalancing of the baseline state-space, transforming an early-collapse setup into a stable environment that supports long-horizon control.

## Architecture Overview

The system simulates a single urban intersection under congestion, emergency pressure, and fairness constraints. It combines:

- deterministic environment dynamics and task builders (`easy`, `medium`, `hard`, `chaos`)
- dense reward shaping aligned to final grader targets
- strict safety constraints (pedestrian and transition discipline)
- a policy-driven baseline in `inference.py` using a live OpenRouter-compatible API path

## Environment Tuning & State-Space Balancing

Early profiling showed deterministic failure in high-load tasks due to a structural capacity and buffer mismatch:

- inflow could outpace effective outflow, causing unavoidable backlog growth
- catastrophic backlog limits were too tight for normal red/yellow queue breathing

Fixes applied in `env/tasks.py`:

- increased `base_capacity` across tasks to handle platoon inflow
- expanded `catastrophic_backlog` thresholds to prevent artificial early termination

These changes allow full-cycle phase control without immediate catastrophic clipping.

## Agent Policy Strategy

The baseline (`inference.py`) follows a strict deterministic hierarchy tuned for throughput efficiency and safety:

- safety-only fallback/override (no queue-routing interference in fallback)
- long green extension actions (`set_ns_green:14`, `set_ew_green:14`) to reduce switching tax
- throughput lock behavior to avoid emergency-induced oscillation

This policy reduces yellow/startup churn and improves sustained movement under heavy load.

## Benchmark Snapshot

After applying task rebalancing and throughput-lock policy, benchmark runs reached full task horizons:

| Task | Steps Survived | Final Score | Death Reason |
| :--- | :---: | :---: | :--- |
| **Easy** | 50 / 50 | 0.632426 | SURVIVED |
| **Medium** | 60 / 60 | 0.711217 | SURVIVED |
| **Hard** | 70 / 70 | 0.857898 | SURVIVED |

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
- `current_phase`, `phase_remaining`, `yellow_active`
- `queue_ns`, `queue_ew`
- `emergency_ns`, `emergency_ew`
- `moved_ns`, `moved_ew`
- `total_wait_time`, `emergency_wait_time`
- `invalid_actions`, `fairness_gap`, `last_action_valid`
- `valid_actions`

## Tasks

Deterministic tasks are included with increasing difficulty:

- `easy`: balanced traffic with low emergency frequency
- `medium`: rush-hour directional imbalance and periodic emergency arrivals
- `hard`: high congestion, emergency contention, and stricter fairness requirements
- `chaos`: extreme stress scenario with bursty platoons and simultaneous emergency pressure

## Reward

Dense step reward is bounded to `[-1.0, 1.0]` and intentionally aligned with the final grader:

- throughput bonus
- average-delay and emergency-delay penalties (emergency weighted more strongly)
- fairness/starvation penalties
- flicker/stability penalties for unstable phase control
- invalid-action and catastrophic penalties

This keeps RL shaping directional while preserving deterministic final evaluation.

## Grader

The episode grader returns:

- `score` in `[0.0, 1.0]`
- `breakdown` with 10 normalized metrics
- deterministic `reasons` for penalties/bonuses

Score equation:

```text
score =
	0.18 * throughput
+ 0.14 * (1 - avg_wait)
+ 0.08 * (1 - max_wait)
+ 0.10 * (1 - backlog_end)
+ 0.18 * (1 - emergency_delay)
+ 0.12 * emergency_priority
+ 0.10 * fairness
+ 0.08 * (1 - starvation)
+ 0.06 * (1 - flicker)
+ 0.04 * (1 - stability)
- invalid_action_penalty
```

All terms are normalized, then clamped to `[0, 1]`. If a catastrophic safety event is flagged (for example unsafe phase bypass under mandatory transition constraints), the final score is forced to a near-zero band.

### Metric rationale

- Emergency delay carries higher weight so emergency neglect cannot be masked by throughput.
- Fairness and starvation terms prevent one-lane exploitation.
- Flicker and stability terms discourage rapid oscillatory switching.
- Determinism is guaranteed for identical `task_id`, `seed`, and action trajectory.

### Grader tests

`test_grader.py` validates:

- perfect trajectory → high score
- selfish trajectory → moderate score
- catastrophic trajectory → near-zero score
- same seed + same trajectory → identical score/breakdown/reasons

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

- `ENV_BASE_URL` (runtime environment URL, default `http://127.0.0.1:7860`)
- `API_BASE_URL` (OpenRouter endpoint, default `https://openrouter.ai/api/v1`)
- `MODEL_NAME` (for example `openai/gpt-4o-mini`)
- `OPENAI_API_KEY` (OpenRouter key in OpenAI-compatible mode)

Run baseline:

```bash
export ENV_BASE_URL=http://127.0.0.1:7860
export API_BASE_URL=https://openrouter.ai/api/v1
export MODEL_NAME=openai/gpt-4o-mini
export OPENAI_API_KEY=<YOUR_OPENROUTER_KEY>
python inference.py easy
```

## Live API Verification

Use these checks to prove decisions came from a live external API call path (and not from hardcoded outputs):

```bash
python audit_baseline_integrity.py --strict
python verify_live_api.py --mode live --task easy --env-base-url http://127.0.0.1:7860
python verify_live_api.py --mode broken --task easy --env-base-url http://127.0.0.1:7860
```

Notes:

- `verify_live_api.py --mode live` requires `OPENAI_API_KEY` and expects successful LLM calls plus nonce echoes.
- `verify_live_api.py --mode broken` intentionally uses an invalid key and expects strict failure.
- Trace artifacts are written to `/tmp` by default as JSON/JSONL files.
- `inference.py` supports `TRACE_API=1` and `VERIFY_STRICT=1` for proof-focused runs.

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
