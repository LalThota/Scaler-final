from typing import Dict, List, Any
from .models import Reward, Action, Task

# Action costs
COSTS = {
    "classify": 0.01,
    "respond": 0.05,
    "escalate": 0.1,
    "invalid": 0.2
}

def calculate_reward(base_score_dict: Dict[str, float], step_count: int, repeated_errors: int, last_action: Action) -> Reward:
    total_score = sum(base_score_dict.values())
    
    # Penalties
    step_penalty = 0.03 * step_count
    error_penalty = 0.05 * repeated_errors
    
    # Action cost
    action_cost = COSTS.get("classify", 0.01)
    if last_action.response_message:
        action_cost = COSTS.get("respond", 0.05)
    if last_action.mark_resolved:
        action_cost = COSTS.get("escalate", 0.1) # Treat resolution as a major action.
    
    # Combined reward
    reward_value = total_score - step_penalty - error_penalty - action_cost
    reward_value = max(0.0, min(1.0, reward_value))
    
    # Feedback
    feedback_msgs = []
    if base_score_dict["intent"] < 0.3:
        feedback_msgs.append("Incorrect intent extraction.")
    if base_score_dict["priority"] < 0.2:
        feedback_msgs.append("Incorrect priority assignment.")
    if base_score_dict["department"] < 0.2:
        feedback_msgs.append("Incorrect department routing.")
    if base_score_dict["response"] < 0.2:
        feedback_msgs.append("Response lacks required keywords.")
    
    if not feedback_msgs:
        feedback_msgs.append("Action is accurate and well-structured.")
    
    return Reward(
        score=reward_value,
        breakdown=base_score_dict,
        feedback=" ".join(feedback_msgs)
    )
