---
title: scaler
emoji: robot
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Adaptive Multi-Step Customer Support Operations Simulator (CSOS++)

## 🚀 OVERVIEW

Adaptive Multi-Step CSOS++ is a production-grade OpenEnv designed to simulate a real-world customer support environment. The system challenges AI agents to extract complex intents from customer queries, assign dynamic priorities, route to appropriate departments, and provide professional responses.

This project is intentionally designed for OpenEnv-style benchmark submission:
- Real-world support ticket workflow (not a toy game)
- Typed observation/action/reward models
- Deterministic grading and bounded reward in `[0.0, 1.0]`
- Multi-task evaluation (`EASY`, `MEDIUM`, `HARD`, `EXTREME`)
- Dockerized deployment

## 🏗️ ARCHITECTURE

The system is built on a robust, multi-layered architecture:
1. **Core Environment**: Handles state transitions, multi-step conversation logic, and partial observability.
2. **Deterministic Grader**: Evaluates agent actions based on Jaccard similarity, keyword matching, and exact category matching.
3. **Reward Engine**: Implements a complex reward function with action costs, time penalties, and consistency checks.
4. **FastAPI Interface**: Provides high-performance RESTful endpoints for agent interaction.

## ⚙️ REBELS OF REWARD

The reward function evaluates every action:
`reward = total_score - (0.03 × step_count) - (0.05 × repeated_errors) - action_cost - delay_penalty`

- **Action Costs**:
  - `classify`: 0.01 (Low cost)
  - `respond`: 0.05 (Medium cost)
  - `escalate/resolve`: 0.1 (High cost)
- **Consistency**: Changing correct decisions results in a penalty.

## 🧪 TASK DESIGN

The environment includes 4 tasks ranging from EASY to EXTREME:
- **EASY**: Single clear intent (e.g., password reset).
- **MEDIUM**: Ambiguous query (e.g., login issue + double charge).
- **HARD**: Multi-intent with emotional tone (e.g., 10-year customer complaining/billing/manager request).
- **EXTREME**: 3+ complex intents, legal/security threats requiring escalation and multi-step reasoning.

## 🌐 API USAGE

### POST `/reset?task_id=<id>`
Resets the environment with the specified task ID.

### POST `/step?task_id=<id>`
Submit an agent action.
```json
{
  "intents": ["intent1", "intent2"],
  "priority": "high",
  "departments": ["dept1"],
  "response_message": "...",
  "mark_resolved": false
}
```

### Action Space
- `intents`: `List[str]`
- `priority`: `str` in `{low, medium, high, critical}`
- `departments`: `List[str]`
- `response_message`: `str`
- `mark_resolved`: `bool`
- `ask_clarification`: `bool` (optional)

### Observation Space
- `ticket_id`: `str`
- `customer_query`: `str`
- `extracted_intents`: `List[str]`
- `priority`: `str`
- `assigned_departments`: `List[str]`
- `conversation_history`: `List[Dict[str, str]]`
- `status`: `str`
- `step_count`: `int`

### Reward Range
- Reward is always clipped to `[0.0, 1.0]`
- Endpoint returns full reward breakdown and textual feedback

### GET `/state?task_id=<id>`
Get current state observation.

### GET `/debug?task_id=<id>`
View debugging information (ground truth vs predictions).

## 🐳 DEPLOYMENT

1. Build the Docker image:
   `docker build -t csos-pp .`
2. Run the container:
   `docker run -p 7860:7860 csos-pp`

## 🛠️ LOCAL SETUP

1. Create virtual environment:
  `python -m venv .venv`
2. Activate env:
  `source .venv/bin/activate`
3. Install dependencies:
  `pip install -r requirements.txt`
4. Run API:
  `uvicorn app.api:app --host 0.0.0.0 --port 7860`
5. Open UI:
  `http://127.0.0.1:7860`

## 🤖 BASELINE INFERENCE (`inference.py`)

`inference.py` is in the repo root and prints structured logs:
- `[START] ...`
- `[STEP] ...`
- `[END] ...`

Environment variables supported by the baseline:
- `API_BASE_URL`: API endpoint to evaluate against (default `http://127.0.0.1:7860`)
- `MODEL_NAME`: model identifier used by OpenAI client for LLM mode
- `HF_TOKEN`: Hugging Face token used as OpenAI-compatible API key
- `OPENAI_BASE_URL`: OpenAI-compatible base URL (default `https://router.huggingface.co/v1`)

If `MODEL_NAME`/`HF_TOKEN` are not set, inference falls back to deterministic actions.

Run baseline:
`python inference.py`

## 🔁 REPRODUCIBILITY GUARANTEE

The system is fully deterministic. Given the same task and sequence of actions, the grader will always return the exact same score and feedback. No randomness is used in the evaluation process.

## ✅ Submission Checklist Mapping

- `Dockerfile` exists at repo root and starts FastAPI app on port `7860`
- `inference.py` exists at repo root and runs full task sweep
- `openenv.yaml` exists at repo root
- API includes `reset`, `step`, `state` endpoints (+ `debug`, `tasks` helper)
- 4 graded tasks are defined (`>=3` required)
