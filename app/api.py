from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import logging

from .env import CustomerSupportEnv
from .models import Action, Observation, Reward, Task
from .parser import parse_action_json
from .dataset import TASKS

app = FastAPI(title="Adaptive Multi-Step CSOS++")

# State Management
env_store: Dict[str, CustomerSupportEnv] = {} # task_id -> env
last_action_store: Dict[str, Action] = {}

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
