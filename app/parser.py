import json
import logging
from typing import Dict, Any, Optional
from .models import Action

logger = logging.getLogger(__name__)

def parse_action_json(json_input: str) -> Action:
    try:
        # 1. Parse JSON
        if isinstance(json_input, dict):
            data = json_input
        else:
            data = json.loads(json_input)
        
        # 2. Extract Fields safely
        action = Action(
            intents=data.get("intents", []),
            priority=data.get("priority", "medium"),
            departments=data.get("departments", []),
            response_message=data.get("response_message", ""),
            mark_resolved=data.get("mark_resolved", False),
            ask_clarification=data.get("ask_clarification", False)
        )
        
        # Ensure correct types
        if not isinstance(action.intents, list): action.intents = []
        if not isinstance(action.priority, str): action.priority = "medium"
        if not isinstance(action.departments, list): action.departments = []
        if not isinstance(action.response_message, str): action.response_message = ""
        if not isinstance(action.mark_resolved, bool): action.mark_resolved = False
        if not isinstance(action.ask_clarification, bool): action.ask_clarification = False
        
        return action
        
    except (json.JSONDecodeError, TypeError, ValueError) as e:
        logger.error(f"Error parsing action JSON: {e}")
        # Default empty action to avoid crash
        return Action(
            intents=[],
            priority="medium",
            departments=[],
            response_message="[FALLBACK: Invalid response]",
            mark_resolved=False,
            ask_clarification=False
        )
