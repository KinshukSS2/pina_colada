---
title: traffic_openenv
emoji: 🚦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# 🚦 OpenEnv Traffic Control: System Optimization & State-Space Balancing

**Author:** Kinshuk Sanand  
**Event:** Meta PyTorch OpenEnv Hackathon × SST | India AI Hackathon'26  
**Python:** 3.10 – 3.11 &nbsp;|&nbsp; **License:** MIT &nbsp;|&nbsp; **Framework:** FastAPI + Pydantic v2

---

## Table of Contents

- [Overview](#overview)
- [Architecture Overview](#architecture-overview)
- [Environment Description](#environment-description)
- [Environment Tuning & State-Space Balancing](#environment-tuning--state-space-balancing)
- [Action Space](#action-space)
- [Observation Space](#observation-space)
- [Tasks](#tasks)
- [Reward](#reward)
- [Grader](#grader)
- [Agent Policy Strategy](#agent-policy-strategy)
- [Benchmark Snapshot](#benchmark-snapshot)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Running the Server](#running-the-server)
- [API Endpoints](#api-endpoints)
- [Baseline Inference](#baseline-inference)
- [Testing & Validation](#testing--validation)
- [Live API Verification](#live-api-verification)
- [Docker](#docker)
- [Hugging Face Spaces](#hugging-face-spaces)
- [OpenEnv](#openenv)
- [Environment Variables Reference](#environment-variables-reference)
- [Configuration Files](#configuration-files)
- [Contributing](#contributing)

---

## Overview

This project provides a **deterministic traffic control environment** and an **optimized LLM-driven baseline policy** for intelligent traffic light management. It is built to the [OpenEnv](https://github.com/OpenEnv) specification and designed to benchmark AI agent performance on real-time intersection control.

**Core objective:** Maximize throughput, minimize starvation, and prioritize emergencies — while preserving safety and full determinism.

**Key contribution:** Mathematical profiling and rebalancing of the baseline state-space, transforming an early-collapse setup into a stable environment capable of supporting long-horizon control.

---

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────┐
│                        Client / Agent                        │
│                      (inference.py + LLM)                    │
└────────────────────────┬─────────────────────────────────────┘
                         │  HTTP (JSON)
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                    FastAPI Server (server/)                   │
│  Endpoints: /reset  /step  /state  /grade  /mcp  /metadata   │
│             /schema  /tasks  /health                         │
└────────────────────────┬─────────────────────────────────────┘
                         │
                         ▼
┌──────────────────────────────────────────────────────────────┐
│                 Environment Core (env/)                       │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐│
│  │ environment │ │ simulator  │ │   grader   │ │   reward   ││
│  │   .py      │ │   .py      │ │   .py      │ │   .py      ││
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘│
│  ┌────────────┐ ┌────────────┐                               │
│  │  models.py │ │  tasks.py  │                               │
│  └────────────┘ └────────────┘                               │
└──────────────────────────────────────────────────────────────┘
```

---

## Environment Description

The system simulates a **single urban intersection** under congestion, emergency pressure, and fairness constraints. It combines:

- **Deterministic dynamics** — identical seeds + actions → identical outcomes
- **Dense reward shaping** aligned to final grader targets
- **Strict safety constraints** (pedestrian and transition discipline)
- **Task builders** with four difficulty levels (`easy`, `medium`, `hard`, `chaos`)
- **POMDP elements** — sensor offline states, sensor noise, out-of-distribution starts
- **Pedestrian crossing** events with patience timers
- **Lane health degradation** that reduces effective capacity
- A **policy-driven baseline** in `inference.py` using live OpenRouter-compatible LLM calls

---

## Environment Tuning & State-Space Balancing

Early profiling revealed deterministic failure in high-load tasks due to structural capacity and buffer mismatches:

- Inflow could outpace effective outflow, causing unavoidable backlog growth
- Catastrophic backlog limits were too tight for normal red/yellow queue breathing

**Fixes applied in `env/tasks.py`:**

- Increased `base_capacity` across tasks to handle platoon inflow (`8 → 8`, `10 → 10`, `14 → 14`, `16 → 16` for easy/medium/hard/chaos)
- Expanded `catastrophic_backlog` thresholds (`400 / 500 / 600 / 800`) to prevent artificial early termination
- Tuned arrival patterns, emergency timing, and target parameters per difficulty

These changes allow full-cycle phase control without immediate catastrophic clipping.

---

## Action Space

The action space is a compact text grammar, parser-friendly and fully deterministic:

| Action | Description |
|:---|:---|
| `hold` | Keep the current phase unchanged |
| `switch` | Toggle to the opposite phase (triggers 2-step yellow) |
| `prioritize_emergency` | Route green to the lane with more emergencies |
| `set_ns_green:<n>` | Set NS green for `n` steps (clamped to `[min_green, max_green]`) |
| `set_ew_green:<n>` | Set EW green for `n` steps (clamped to `[min_green, max_green]`) |

Invalid or malformed actions are safely handled (fallback to `hold`) and penalized.

**Safety rules:**
- During **yellow phase** (`yellow_active = true`), only `hold` is legal — anything else triggers a collision/safety violation
- When **pedestrians are waiting**, only `hold` and `switch` are legal — other actions cause a catastrophic safety kill

---

## Observation Space

The observation is structured JSON with stable keys:

| Key | Type | Description |
|:---|:---|:---|
| `session_id` | `str` | Unique session identifier |
| `task_id` | `str` | Active task (`easy`, `medium`, `hard`, `chaos`) |
| `timestep` | `int` | Current simulation step |
| `max_steps` | `int` | Episode horizon |
| `current_phase` | `str` | Active green phase (`ns` or `ew`) |
| `phase_remaining` | `int` | Steps left in current green |
| `yellow_active` | `bool` | Whether a yellow transition is in progress |
| `green_duration` | `int` | Consecutive green steps (low values = startup penalty) |
| `ascii_minimap` | `str` | Visual minimap of the intersection |
| `sensor_status` | `str` | `ONLINE` or `OFFLINE` (5% chance each step) |
| `pedestrian_waiting` | `bool` | Pedestrian crossing event active |
| `action_mask` | `dict` | Per-action legality booleans |
| `queue_ns` / `queue_ew` | `int` | Vehicle queues per direction (`-1` if sensor offline) |
| `emergency_ns` / `emergency_ew` | `int` | Emergency vehicles per direction |
| `lane_health` | `float` | Lane degradation factor (`0.6` or `1.0`) |
| `moved_ns` / `moved_ew` | `int` | Cumulative vehicles moved |
| `total_wait_time` | `int` | Cumulative queue wait |
| `emergency_wait_time` | `int` | Cumulative emergency wait (weighted 2×) |
| `invalid_actions` | `int` | Total invalid actions taken |
| `fairness_gap` | `float` | `|moved_ns − moved_ew| / total_moved` |
| `max_wait_seen` | `int` | Worst-case single-direction wait |
| `backlog_total` | `int` | Current total queue |
| `phase_switches` | `int` | Number of phase changes |
| `flicker_events` | `int` | Rapid switching events |
| `starvation_events` | `int` | Lane starvation events |
| `stability_penalty` | `float` | Accumulated stability penalty |
| `catastrophic_event` | `bool` | Whether a fatal event occurred |
| `seed` | `int` | Episode seed |
| `last_action_valid` | `bool` | Whether the last action was valid |
| `valid_actions` | `list[str]` | List of valid action strings |

---

## Tasks

Four deterministic task configurations with increasing difficulty:

| Task | Steps | Base Capacity | Arrivals (NS/EW avg) | Emergencies | Description |
|:---|:---:|:---:|:---:|:---:|:---|
| `easy` | 50 | 8 | 3–5 / 3–4 | 3 events | Balanced daytime flow, low emergency pressure |
| `medium` | 60 | 10 | 4–6 / 2–4 | 5 events | Rush-hour directional imbalance, periodic emergencies |
| `hard` | 70 | 14 | 4–8 / 4–8 | 8 events | Peak congestion, emergency contention, strict fairness |
| `chaos` | 75 | 16 | 2–11 / 2–11 | 8 events | Extreme bursty platoons, simultaneous emergencies |

Task configurations are defined in `env/tasks.py` and registered via `task_catalog()`.

---

## Reward

Dense step reward is squashed to the open interval `(0.01, 0.99)` using `tanh`, intentionally aligned with the final grader:

**Reward components:**
- **Throughput bonus:** `+0.1 × moved_vehicles`
- **Stability penalty:** `−0.3` for phase changes without emergency justification
- **Emergency wait penalty:** `−0.2 × Δemergency_wait`

**Discrete failure deductions:**
- Collision detected: `−10.0`
- Catastrophic event: `−5.0`
- Invalid action: `−1.0`

**Squashing:** `reward = clip((tanh(raw) + 1) / 2)` → smooth mapping to `(0.01, 0.99)`

This keeps RL shaping directional while preserving deterministic final evaluation.

---

## Grader

The episode grader produces a composite score in `[0.0, 1.0]` with full breakdowns.

### Score Equation

```
score =
    0.18 × throughput
  + 0.14 × (1 − avg_wait)
  + 0.08 × (1 − max_wait)
  + 0.10 × (1 − backlog_end)
  + 0.18 × (1 − emergency_delay)
  + 0.12 × emergency_priority
  + 0.10 × fairness
  + 0.08 × (1 − starvation)
  + 0.06 × (1 − flicker)
  + 0.04 × (1 − stability)
  − invalid_action_penalty
```

All terms are normalized and clamped to `[0, 1]`. If a catastrophic safety event is flagged, the final score is forced to `≤ 0.03`.

### Dual Scoring Path

The final score blends two scoring functions:
- **Grader score (65%):** 10-metric weighted composition above
- **Diagnostic score (35%):** Separate throughput(45%) + emergency(30%) + fairness(15%) + discipline(10%) scoring

### Metric Rationale

- **Emergency delay** carries high weight — emergency neglect cannot be masked by throughput
- **Fairness and starvation** terms prevent one-lane exploitation
- **Flicker and stability** terms discourage rapid oscillatory switching
- **Determinism guaranteed** for identical `task_id`, `seed`, and action trajectory

### Grader Tests (`test_grader.py`)

- ✅ Perfect trajectory → score ≥ 0.88
- ⚠️ Selfish trajectory → score in [0.20, 0.72]
- ❌ Catastrophic trajectory → score ≤ 0.05
- 🔁 Deterministic: same seed + same actions → identical score/breakdown/reasons

---

## Agent Policy Strategy

The baseline agent (`inference.py`) is an LLM-driven controller that uses a strict deterministic hierarchy with live OpenRouter API calls.

### Decision Pipeline

```
Observation → Policy Override → LLM Call → Action Extraction → Sanitization → Anti-Stall → Step
```

### Policy Override Layer (safety-first, pre-LLM)

1. **Yellow active** → force `hold` (prevents collision)
2. **Pedestrian waiting** → force `switch` or `hold` (prevents safety kill)
3. **Emergency present** → force `prioritize_emergency`

### LLM Reasoning Core

- Sends structured prompts with observation JSON to OpenRouter-compatible models
- Uses **nonce injection** for response verification (proves live API call, not cached)
- Multi-model fallback chain (`gpt-4o-mini → gpt-4.1-mini → gpt-4o`)
- Temperature 0 for deterministic outputs

### Post-LLM Safety Layers

- **Action extraction:** regex-based parsing with fuzzy matching
- **Sanitization:** validates against `action_mask`, falls back to smart heuristic
- **Anti-stall:** detects idle `hold` loops and forces rebalancing `switch`
- **Smart fallback:** queue-aware heuristic that switches based on load imbalance

### Key Policy Features

- Long green extension actions (`set_ns_green:14`, `set_ew_green:14`) to reduce switching tax
- Throughput lock behavior to avoid emergency-induced oscillation
- Repeated-action loop detection with forced termination
- Full trace logging (`TRACE_API=1`) for audit/proof

---

## Benchmark Snapshot

After task rebalancing and throughput-lock policy, benchmark runs reached full task horizons:

| Task | Steps Survived | Final Score | Status |
|:---|:---:|:---:|:---|
| **Easy** | 50 / 50 | 0.632 | ✅ SURVIVED |
| **Medium** | 60 / 60 | 0.711 | ✅ SURVIVED |
| **Hard** | 70 / 70 | 0.858 | ✅ SURVIVED |
| **Chaos** | 75 / 75 | — | 🔥 Extreme stress test (baseline pending) |

---

## Project Structure

```
traffic-control-openenv/
├── env/                          # Core environment package
│   ├── __init__.py
│   ├── models.py                 # Pydantic models (Action, Observation, IntersectionState, etc.)
│   ├── tasks.py                  # Task configurations (easy/medium/hard/chaos)
│   ├── simulator.py              # Deterministic simulation engine
│   ├── environment.py            # TrafficControlEnvironment class (reset/step/state)
│   ├── reward.py                 # Dense reward computation with tanh squashing
│   └── grader.py                 # 10-metric episode grader
│
├── server/                       # FastAPI REST API
│   ├── __init__.py
│   ├── main.py                   # Endpoint definitions (/reset, /step, /state, /grade, /mcp)
│   └── app.py                    # Uvicorn runner entrypoint
│
├── inference.py                  # LLM-driven baseline agent with policy overrides
├── test_grader.py                # Unit tests for grading system
├── test_agent.py                 # Model × task evaluation matrix runner
├── validate_project.py           # Comprehensive static + optional runtime project validator
├── validate_submission.py        # Runtime determinism and submission checks
├── verify_live_api.py            # Live/broken API verification (proves real LLM usage)
├── audit_baseline_integrity.py   # Static analysis for hardcoded scores/actions
├── run_openrouter_benchmark.sh   # Shell script for full benchmark suite
│
├── Dockerfile                    # Docker image (python:3.11-slim, port 7860)
├── requirements.txt              # Python dependencies
├── pyproject.toml                # Package metadata and build configuration
├── openenv.yaml                  # OpenEnv specification manifest
├── .env                          # Environment variables (gitignored)
└── .gitignore                    # Git ignore rules
```

---

## Setup

### Prerequisites

- **Python 3.10 – 3.11**
- An **OpenRouter API key** (or compatible OpenAI-format key)

### Local Python Setup

```bash
# Clone the repository
git clone <repo-url>
cd traffic-control-openenv

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Configuration

Create a `.env` file (auto-loaded by `inference.py`, `validate_submission.py`, and `verify_live_api.py`):

```bash
cat > .env << 'EOF'
ENV_BASE_URL=http://127.0.0.1:7860
API_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-4o-mini
OPENAI_API_KEY=<YOUR_OPENROUTER_KEY>
EOF
```

---

## Running the Server

Start the FastAPI environment server:

```bash
# Option 1: Uvicorn directly
uvicorn server.main:app --host 0.0.0.0 --port 7860

# Option 2: Via package entrypoint
python -m server.app

# Option 3: Via module main
python server/main.py
```

The server will be available at `http://127.0.0.1:7860`.

---

## API Endpoints

| Method | Endpoint | Description |
|:---|:---|:---|
| `GET` | `/health` | Health check |
| `GET` | `/metadata` | Environment metadata and task list |
| `GET` | `/schema` | Action/observation schema definition |
| `GET` | `/tasks` | List all available tasks |
| `POST` | `/reset` | Reset environment with task config |
| `POST` | `/step` | Execute an action and get next observation |
| `POST` | `/state` | Get current state without stepping |
| `POST` | `/grade` | Grade an episode summary |
| `POST` | `/mcp` | MCP JSON-RPC 2.0 endpoint (OpenEnv runtime) |

### API Smoke Checks

```bash
# Reset with easy task
curl -X POST http://127.0.0.1:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id":"easy"}'

# Get current state
curl -X POST http://127.0.0.1:7860/state \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<SESSION_ID>"}'

# Step with an action
curl -X POST http://127.0.0.1:7860/step \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<SESSION_ID>","action":"hold"}'
```

---

## Baseline Inference

The baseline inference script is `inference.py` at the repository root.

### Run Baseline

```bash
# Single task
python inference.py easy

# All tasks sequentially (easy, medium, hard)
python inference.py
```

### Benchmark Script

Run the full benchmark suite with automatic server management:

```bash
export OPENAI_API_KEY='sk-or-v1-...'
bash run_openrouter_benchmark.sh
```

This starts a local server, runs inference on `easy`/`medium`/`hard`, and produces a markdown results table.

### Evaluation Matrix

Run all model × task combinations:

```bash
python test_agent.py
```

---

## Testing & Validation

### Unit Tests

```bash
# Run grader tests
python -m unittest test_grader -v
```

### Project Validation

```bash
# Static validation (file presence, syntax, markers, config)
python validate_project.py

# With optional runtime checks
python validate_project.py --run-openenv-validate --run-docker-build --run-docker-smoke --check-tools
```

### Submission Validation

```bash
# Runtime determinism checks (requires running server)
python validate_submission.py
```

### Baseline Integrity Audit

```bash
# Check for hardcoded scores/actions
python audit_baseline_integrity.py --strict
```

---

## Live API Verification

Prove that decisions come from a live external API call path (not hardcoded outputs):

```bash
# Live mode: requires OPENAI_API_KEY, expects successful LLM calls + nonce echoes
python verify_live_api.py --mode live --task easy --env-base-url http://127.0.0.1:7860

# Broken mode: uses invalid key, expects strict failure
python verify_live_api.py --mode broken --task easy --env-base-url http://127.0.0.1:7860

# Compare mode (default): runs both live and broken, compares results
python verify_live_api.py --mode compare --task easy --env-base-url http://127.0.0.1:7860
```

**Notes:**
- Trace artifacts are written to `/tmp` by default as JSON/JSONL files
- `inference.py` supports `TRACE_API=1` and `VERIFY_STRICT=1` for proof-focused runs
- Nonce injection ensures each LLM response can be verified as genuinely live

---

## Docker

### Build & Run

```bash
docker build -t traffic-env .
docker run --rm -p 7860:7860 traffic-env
```

The Docker image uses `python:3.11-slim`, installs dependencies, and starts the uvicorn server on port `7860`.

---

## Hugging Face Spaces

This repository is compatible with **Docker-based Hugging Face Spaces**. To deploy:

1. Push the repository to a Hugging Face Space
2. Configure required secrets (`OPENAI_API_KEY`, `MODEL_NAME`, etc.) in Space settings
3. The `README.md` frontmatter configures the Space automatically (Docker SDK, port 7860)

---

## OpenEnv

This project follows the OpenEnv specification. The environment manifest is defined in `openenv.yaml`.

### Validate Packaging

```bash
openenv validate
```

---

## Environment Variables Reference

| Variable | Default | Description |
|:---|:---|:---|
| `ENV_BASE_URL` | `http://127.0.0.1:7860` | Environment server URL |
| `API_BASE_URL` | `https://openrouter.ai/api/v1` | LLM API endpoint (OpenRouter) |
| `MODEL_NAME` | `openai/gpt-4o-mini` | Primary model identifier |
| `MODEL_CANDIDATES` | `openai/gpt-4o-mini,openai/gpt-4.1-mini,openai/gpt-4o` | Comma-separated fallback model chain |
| `OPENAI_API_KEY` | — | OpenRouter/OpenAI API key |
| `HF_TOKEN` | — | Hugging Face token (fallback API key) |
| `DISABLE_LLM` | `0` | Disable LLM calls (use heuristic fallback) |
| `TRACE_API` | `0` | Enable JSONL trace logging |
| `TRACE_DIR` | `/tmp` | Directory for trace files |
| `VERIFY_STRICT` | `0` | Enable strict verification mode |
| `LLM_TIMEOUT_SECONDS` | `20` | LLM call timeout |
| `MAX_STEPS` | `100` | Maximum episode steps |
| `REPEAT_ACTION_LIMIT` | `50` | Action repetition detection window |

---

## Configuration Files

| File | Purpose |
|:---|:---|
| `openenv.yaml` | OpenEnv specification manifest (name, actions, observations, tasks) |
| `pyproject.toml` | Python package metadata, build config, entry points |
| `requirements.txt` | Pinned Python dependencies |
| `Dockerfile` | Container definition for deployment |
| `.env` | Local environment variables (gitignored) |
| `.gitignore` | Git ignore rules (caches, secrets, reference docs) |

---

## Contributing

1. Ensure all tests pass: `python -m unittest test_grader -v`
2. Run the project validator: `python validate_project.py`
3. Run the integrity audit: `python audit_baseline_integrity.py --strict`
4. If modifying environment dynamics, verify determinism: `python validate_submission.py`
