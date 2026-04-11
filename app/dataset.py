from typing import List, Dict
from .models import Task

TASKS: List[Task] = [
    Task(
        id="EASY-001",
        customer_query="I would like to change my password for my account.",
        expected_intents=["password_reset"],
        expected_priority="low",
        expected_departments=["security"],
        difficulty="EASY",
        domain="account_security",
        customer_segment="consumer",
        ambiguity_level="low",
        ground_truth_response_keywords=["password", "reset", "link"],
        must_escalate=False,
        max_steps=3
    ),
    Task(
        id="MEDIUM-001",
        customer_query="I can't log in and also my last payment was double charged. Help!",
        expected_intents=["login_issue", "billing_error"],
        expected_priority="high",
        expected_departments=["technical_support", "billing"],
        difficulty="MEDIUM",
        domain="saas_support",
        customer_segment="consumer",
        ambiguity_level="medium",
        ground_truth_response_keywords=["login", "refund", "investigate", "double", "charge"],
        must_escalate=False,
        max_steps=5
    ),
    Task(
        id="HARD-001",
        customer_query="Listen, I've been a customer for 10 years and this service is a disgrace. I need a refund immediately and I want to speak to the manager about your security breach last week.",
        expected_intents=["complaint", "refund_request", "security_breach_info"],
        expected_priority="critical",
        expected_departments=["billing", "customer_relations", "security"],
        difficulty="HARD",
        domain="enterprise_support",
        customer_segment="loyal_customer",
        ambiguity_level="high",
        ground_truth_response_keywords=["apologize", "loyal", "manager", "security", "refund"],
        must_escalate=True,
        max_steps=5
    ),
    Task(
        id="EXTREME-001",
        customer_query="Everything is broken. System down. Data leaked? My credit card was used twice. I am suing you guys. Where is my data? Who is responsible?",
        expected_intents=["system_down", "security_breach_report", "billing_fraud", "legal_threat"],
        expected_priority="critical",
        expected_departments=["technical_support", "security", "billing", "legal"],
        difficulty="EXTREME",
        domain="incident_response",
        customer_segment="enterprise",
        ambiguity_level="high",
        ground_truth_response_keywords=["breach", "emergency", "legal", "investigation", "supervisor"],
        must_escalate=True,
        max_steps=8
    )
]

def get_task_by_id(task_id: str) -> Task:
    for task in TASKS:
        if task.id == task_id:
            return task
    return TASKS[0]
