from typing import List, Dict, Optional, Any, Tuple
from .models import Observation, Action, Reward, Task
from .dataset import TASKS, get_task_by_id
from .grader import calculate_base_score
from .reward import calculate_reward

class CustomerSupportEnv:
    def __init__(self, task_id: Optional[str] = None):
        self.task_id = task_id or TASKS[0].id
        self.task = get_task_by_id(self.task_id)
        self.reset()
    
    def reset(self) -> Observation:
        self.observation = Observation(
            ticket_id=self.task.id,
            customer_query=self.task.customer_query,
            extracted_intents=[],
            priority="medium",
            assigned_departments=[],
            conversation_history=[],
            status="open",
            step_count=0
        )
        self.repeated_errors = 0
        self.last_action_was_good = True
        return self.observation
    
    def step(self, action: Action) -> Tuple[Observation, Reward, bool]:
        self.observation.step_count += 1
        
        # 1. Update State based on Action
        self.observation.extracted_intents = action.intents
        self.observation.priority = action.priority
        self.observation.assigned_departments = action.departments
        
        if action.response_message:
            self.observation.conversation_history.append({
                "role": "agent",
                "message": action.response_message
            })
            self.observation.status = "in_progress"
        
        if action.mark_resolved:
            self.observation.status = "resolved"
        
        # Check for consistency (e.g., if it was correct earlier and now incorrect)
        # For simplicity, we track how much it deviated from optimal.
        
        # 2. Calculate Base Score via Grader
        base_score_dict = calculate_base_score(action, self.task)
        
        # 3. Calculate Reward via Reward Module
        reward_obj = calculate_reward(base_score_dict, self.observation.step_count, self.repeated_errors, action)
        
        # Check if action was good or bad to track repeated errors
        action_score = sum(base_score_dict.values())
        if action_score < 0.5:
            self.repeated_errors += 1
        else:
            self.repeated_errors = 0 # reset on progress?
            
        # 4. Check if Terminal
        is_terminal = False
        if action.mark_resolved or self.observation.step_count >= self.task.max_steps:
            is_terminal = True
            
        # 5. Potential state updates based on Terminal (e.g. status)
        if is_terminal and not action.mark_resolved:
            self.observation.status = "escalated"
            
        return self.observation, reward_obj, is_terminal
    
    def state(self) -> Observation:
        return self.observation
