from typing import List, Set, Dict, Any
from .models import Action, Task

def get_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))

def calculate_base_score(action: Action, task: Task) -> Dict[str, float]:
    # 0.3 intent score (Jaccard similarity)
    intent_score = get_jaccard_similarity(set(action.intents), set(task.expected_intents))
    
    # 0.2 priority score (exact match)
    priority_score = 1.0 if action.priority.lower() == task.expected_priority.lower() else 0.0
    
    # 0.2 department score (exact match)
    department_score = get_jaccard_similarity(set(action.departments), set(task.expected_departments))
    
    # 0.2 response score (keywords)
    response_score = 0.0
    if action.response_message:
        keywords_matched = 0
        for kw in task.ground_truth_response_keywords:
            if kw.lower() in action.response_message.lower():
                keywords_matched += 1
        response_score = keywords_matched / len(task.ground_truth_response_keywords) if task.ground_truth_response_keywords else 1.0
    
    # 0.1 resolution score
    resolution_score = 0.0
    if action.mark_resolved:
        if not task.must_escalate:
            resolution_score = 1.0
        else:
            # Task must be escalated, not resolved by the agent directly
            resolution_score = 0.0
    elif task.must_escalate and action.ask_clarification is None:
         # Need resolution logic for escalation.
         # For simplicity: if it's escalated or clarification is needed.
         pass

    return {
        "intent": 0.3 * intent_score,
        "priority": 0.2 * priority_score,
        "department": 0.2 * department_score,
        "response": 0.2 * response_score,
        "resolution": 0.1 * resolution_score
    }
