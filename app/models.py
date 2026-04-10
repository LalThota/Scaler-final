from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any

class Observation(BaseModel):
    ticket_id: str
    customer_query: str
    task_domain: str = "general"
    customer_segment: str = "standard"
    extracted_intents: List[str] = Field(default_factory=list)
    priority: str = "medium"
    assigned_departments: List[str] = Field(default_factory=list)
    conversation_history: List[Dict[str, str]] = Field(default_factory=list)
    status: str = "open"
    step_count: int = 0
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    suggested_next_action: str = "analyze"

class Action(BaseModel):
    intents: List[str] = Field(default_factory=list)
    priority: str = "medium"
    departments: List[str] = Field(default_factory=list)
    response_message: str = ""
    mark_resolved: bool = False
    ask_clarification: Optional[bool] = False

class Reward(BaseModel):
    score: float = Field(gt=0.0, lt=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)
    feedback: str = ""

class Task(BaseModel):
    id: str
    customer_query: str
    grader: str = "app.grader:grade_action_score"
    expected_intents: List[str]
    expected_priority: str
    expected_departments: List[str]
    difficulty: str
    domain: str = "general_support"
    customer_segment: str = "standard"
    ambiguity_level: str = "low"
    ground_truth_response_keywords: List[str]
    must_escalate: bool = False
    max_steps: int = 5
