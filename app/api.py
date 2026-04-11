from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging
from statistics import mean

from .env import CustomerSupportEnv
from .models import Action, Observation, Reward, Task
from .parser import parse_action_json
from .dataset import TASKS

app = FastAPI(title="Adaptive Multi-Step CSOS++")

# State Management
env_store: Dict[str, CustomerSupportEnv] = {} # task_id -> env
last_action_store: Dict[str, Action] = {}
event_log: List[Dict[str, Any]] = []

# Ensure static directory exists
import os
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/reset")
async def reset_env(task_id: Optional[str] = None):
    task_id = task_id or TASKS[0].id
    env = CustomerSupportEnv(task_id=task_id)
    env_store[task_id] = env
    obs = env.reset()
    return obs

@app.post("/step")
async def step_env(task_id: str, action_data: Dict[str, Any]):
    if task_id not in env_store:
        env_store[task_id] = CustomerSupportEnv(task_id=task_id)
    
    env = env_store[task_id]
    action = parse_action_json(action_data)
    last_action_store[task_id] = action
    
    observation, reward, done = env.step(action)

    event_log.append(
        {
            "task_id": task_id,
            "score": reward.score,
            "done": done,
            "difficulty": env.task.difficulty,
            "domain": env.task.domain,
            "customer_segment": env.task.customer_segment,
            "step_count": observation.step_count,
        }
    )
    
    return {
        "observation": observation,
        "reward": reward,
        "done": done
    }

@app.get("/state")
async def get_state(task_id: str):
    if task_id not in env_store:
        return {"error": "Environment not initialized"}
    env = env_store[task_id]
    return env.state()

@app.get("/debug")
async def get_debug(task_id: str):
    if task_id not in env_store:
        return {"error": "Environment not initialized"}
        
    env = env_store[task_id]
    last_action = last_action_store.get(task_id)
    
    return {
        "last_action": last_action,
        "task_ground_truth": {
            "intents": env.task.expected_intents,
            "priority": env.task.expected_priority,
            "departments": env.task.expected_departments,
            "must_escalate": env.task.must_escalate,
            "max_steps": env.task.max_steps
        },
        "repeated_errors": env.repeated_errors,
        "step_count": env.observation.step_count
    }

@app.get("/")
async def get_ui():
    with open("static/index.html", "r") as f:
        return HTMLResponse(content=f.read(), status_code=200)

@app.get("/tasks")
async def get_tasks():
    return [task.id for task in TASKS]


@app.get("/tasks/details")
async def get_task_details():
    return [
        {
            "id": task.id,
            "difficulty": task.difficulty,
            "domain": task.domain,
            "customer_segment": task.customer_segment,
            "ambiguity_level": task.ambiguity_level,
            "must_escalate": task.must_escalate,
            "max_steps": task.max_steps,
        }
        for task in TASKS
    ]


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "active_envs": len(env_store),
        "task_catalog_size": len(TASKS),
    }


@app.get("/metrics")
async def metrics():
    if not event_log:
        return {
            "events": 0,
            "avg_score": 0,
            "completion_rate": 0,
            "by_difficulty": {},
            "by_domain": {},
            "by_customer_segment": {},
        }

    avg_score = mean(event["score"] for event in event_log)
    completion_rate = sum(1 for event in event_log if event["done"]) / len(event_log)

    by_difficulty: Dict[str, List[float]] = {}
    by_domain: Dict[str, List[float]] = {}
    by_segment: Dict[str, List[float]] = {}

    for event in event_log:
        by_difficulty.setdefault(event["difficulty"], []).append(event["score"])
        by_domain.setdefault(event["domain"], []).append(event["score"])
        by_segment.setdefault(event["customer_segment"], []).append(event["score"])

    return {
        "events": len(event_log),
        "avg_score": round(avg_score, 4),
        "completion_rate": round(completion_rate, 4),
        "by_difficulty": {k: round(mean(v), 4) for k, v in by_difficulty.items()},
        "by_domain": {k: round(mean(v), 4) for k, v in by_domain.items()},
        "by_customer_segment": {k: round(mean(v), 4) for k, v in by_segment.items()},
    }
