from typing import List, Set, Dict, Any
from .models import Action, Task

EPSILON = 1e-3


def _clamp_open_interval(score: float) -> float:
    return max(EPSILON, min(1.0 - EPSILON, score))

def get_jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1.intersection(set2)) / len(set1.union(set2))


def is_ambiguous_query(text: str) -> bool:
    lowered = text.lower()
    ambiguous_markers = ["also", "and", "?", "maybe", "not sure", "or", "but"]
    hit_count = sum(1 for marker in ambiguous_markers if marker in lowered)
    return hit_count >= 2


def calculate_action_confidence(base_score_dict: Dict[str, float], action: Action, task: Task) -> float:
    base_score = sum(base_score_dict.values())
    action_completeness = 0.0
    if action.intents:
        action_completeness += 0.3
    if action.departments:
        action_completeness += 0.3
    if action.response_message:
        action_completeness += 0.4

    ambiguity_penalty = 0.1 if (task.ambiguity_level == "high" and action.ask_clarification) else 0.0
    confidence = (0.75 * base_score) + (0.25 * action_completeness) + ambiguity_penalty
    return max(0.0, min(1.0, confidence))

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
    
    # 0.1 resolution and escalation score
    resolution_score = 0.0
    query_is_ambiguous = is_ambiguous_query(task.customer_query) or task.ambiguity_level in {"medium", "high"}

    if action.mark_resolved:
        if not task.must_escalate:
            resolution_score = 1.0
        else:
            resolution_score = 0.0
    else:
        if task.must_escalate:
            escalation_signals = ["escalat", "manager", "supervisor", "legal", "incident"]
            response_has_signal = any(sig in action.response_message.lower() for sig in escalation_signals)
            touched_critical_dept = bool(set(action.departments).intersection({"security", "legal"}))
            if response_has_signal or touched_critical_dept:
                resolution_score = 1.0
            elif action.ask_clarification and query_is_ambiguous:
                resolution_score = 0.7
        elif action.ask_clarification and query_is_ambiguous:
            # Clarification is useful for ambiguous tickets, even if not mandatory.
            resolution_score = 0.7

    return {
        "intent": 0.3 * intent_score,
        "priority": 0.2 * priority_score,
        "department": 0.2 * department_score,
        "response": 0.2 * response_score,
        "resolution": 0.1 * resolution_score
    }


def grade_action_score(action: Action, task: Task) -> float:
    """Return a strict open-interval scalar score for validator-facing graders."""
    total = float(sum(calculate_base_score(action, task).values()))
    return _clamp_open_interval(total)


def grade_easy_task_score(action: Action, task: Task) -> float:
    return grade_action_score(action, task)


def grade_medium_task_score(action: Action, task: Task) -> float:
    return grade_action_score(action, task)


def grade_hard_task_score(action: Action, task: Task) -> float:
    return grade_action_score(action, task)


def grade_extreme_task_score(action: Action, task: Task) -> float:
    return grade_action_score(action, task)
