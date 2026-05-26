# orchestrator.py
# ─────────────────────────────────────────────────────────
# RPA Orchestrator — LangGraph multi-agent workflow
# Supervisor → Invoice / HR / Escalation agents
# Uses model_factory for pluggable LLM support
# ─────────────────────────────────────────────────────────

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage
from typing import TypedDict, Optional
import time
import os
from dotenv import load_dotenv

# ── model_factory (one line change!) ────────────────────
from model_factory import get_llm

# ── Specialist agents ────────────────────────────────────
from agents.invoice_agent import run_invoice_agent
from agents.hr_agent import run_hr_agent

load_dotenv()

# ── Supervisor LLM via factory ──────────────────────────
llm = get_llm()


# ── STATE ────────────────────────────────────────────────
class OrchestratorState(TypedDict):
    request:        str
    next_agent:     str
    result:         str
    agent_used:     str
    invoice_amount: float
    needs_approval: bool
    approved:       Optional[bool]
    approver:       Optional[str]
    status:         str


# ── NODES ────────────────────────────────────────────────
def supervisor_node(state: OrchestratorState) -> dict:
    """Classify request and route to correct specialist"""
    print("\n→ Supervisor: Analysing and routing request...")

    prompt = f"""You are a supervisor routing business requests to specialist agents.

AGENTS:
- invoice_agent: invoices, payments, vendor bills, purchase orders, payment terms
- hr_agent: employees, onboarding, leave, HR policies, payroll queries, offboarding
- escalation_agent: unclear requests, missing information, out-of-scope queries

EXAMPLES (use these to guide routing):
"Process IBM invoice for £8,000"           → invoice_agent
"Pay Accenture £85,000 for consulting"     → invoice_agent
"What is the CFO approval threshold?"      → invoice_agent
"New employee Sara joining on Monday"      → hr_agent
"How many sick days am I entitled to?"     → hr_agent
"What is our maternity leave policy?"      → hr_agent
"What is our work from home policy?"       → hr_agent
"Lost my invoice, not sure what to do"     → escalation_agent
"Something went wrong with a payment"      → escalation_agent
"Urgent help needed"                       → escalation_agent

IMPORTANT:
- When uncertain → ALWAYS route to escalation_agent (safer than wrong routing)
- "Invoice" in message ≠ always invoice_agent (check full context)

Request: "{state['request']}"

Respond with EXACTLY ONE WORD: invoice_agent, hr_agent, or escalation_agent"""

    response = llm.invoke([HumanMessage(content=prompt)])
    agent = response.content.strip().lower()
    print(f"  Routed to: {agent}")

    return {"next_agent": agent}


def invoice_node(state: OrchestratorState) -> dict:
    """Handle invoice request via RAG agent"""
    result = run_invoice_agent(state["request"])
    return result


def hr_node(state: OrchestratorState) -> dict:
    """Handle HR request via RAG agent"""
    result = run_hr_agent(state["request"])
    return result


def escalation_node(state: OrchestratorState) -> dict:
    """Handle unclear or out-of-scope requests"""
    print("\n→ Escalation Agent: Handling escalation...")
    ref = f"ESC-{int(time.time())}"

    result = (
        f"Your request has been escalated to the relevant team. "
        f"Reference number: {ref}. "
        f"A team member will contact you within 2 business hours. "
        f"For urgent matters contact: support@acmecorp.com"
    )

    return {
        "result":         result,
        "agent_used":     "Escalation Agent",
        "invoice_amount": 0.0,
        "needs_approval": False,
        "status":         "escalated"
    }


def cfo_approval_node(state: OrchestratorState) -> dict:
    """Process CFO approval decision — runs AFTER update_state injects decision"""
    if state.get("approved"):
        return {
            "result": (
                f"Invoice for £{state['invoice_amount']:,.2f} — "
                f"APPROVED by {state.get('approver', 'CFO')} ✅. "
                f"Payment will be processed within standard payment terms."
            ),
            "status": "completed"
        }
    return {
        "result": (
            f"Invoice for £{state['invoice_amount']:,.2f} — "
            f"REJECTED by {state.get('approver', 'CFO')} ❌. "
            f"Please contact finance@acmecorp.com for further information."
        ),
        "status": "rejected"
    }


# ── ROUTING FUNCTIONS ────────────────────────────────────
def route_after_supervisor(state: OrchestratorState) -> str:
    agent = state.get("next_agent", "escalation_agent").lower()
    if "invoice" in agent: return "invoice"
    if "hr"      in agent: return "hr"
    return "escalation"


def route_after_invoice(state: OrchestratorState) -> str:
    return "needs_cfo" if state.get("needs_approval") else "done"


# ── BUILD GRAPH ──────────────────────────────────────────
workflow = StateGraph(OrchestratorState)

workflow.add_node("supervisor",    supervisor_node)
workflow.add_node("invoice",       invoice_node)
workflow.add_node("hr",            hr_node)
workflow.add_node("escalation",    escalation_node)
workflow.add_node("cfo_approval",  cfo_approval_node)

workflow.set_entry_point("supervisor")

workflow.add_conditional_edges(
    "supervisor", route_after_supervisor,
    {"invoice": "invoice", "hr": "hr", "escalation": "escalation"}
)

workflow.add_conditional_edges(
    "invoice", route_after_invoice,
    {"needs_cfo": "cfo_approval", "done": END}
)

workflow.add_edge("hr",           END)
workflow.add_edge("escalation",   END)
workflow.add_edge("cfo_approval", END)

# ── COMPILE with memory + interrupt ─────────────────────
memory = MemorySaver()
app = workflow.compile(
    checkpointer=memory,
    interrupt_before=["cfo_approval"]   # pause for CFO approval
)


# ── PUBLIC FUNCTIONS ─────────────────────────────────────
def run_orchestrator(request: str, thread_id: str = "default") -> dict:
    """
    Main entry point — run orchestrator for a user request.

    Args:
        request:   natural language business request
        thread_id: unique ID for this workflow instance

    Returns:
        dict with result, agent_used, needs_human, invoice_amount, status
    """
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "request":        request,
        "next_agent":     "",
        "result":         "",
        "agent_used":     "",
        "invoice_amount": 0.0,
        "needs_approval": False,
        "approved":       None,
        "approver":       None,
        "status":         "processing"
    }

    result = app.invoke(initial_state, config=config)

    return {
        "result":         result.get("result", ""),
        "agent_used":     result.get("agent_used", ""),
        "needs_human":    result.get("needs_approval", False),
        "invoice_amount": result.get("invoice_amount", 0.0),
        "status":         result.get("status", "")
    }


def handle_approval(thread_id: str, approved: bool, approver: str = "CFO") -> dict:
    """
    Handle CFO approval decision — resumes paused workflow.

    Args:
        thread_id: the workflow instance to resume
        approved:  True = approve, False = reject
        approver:  name of approver (default: CFO)

    Returns:
        dict with final result
    """
    config = {"configurable": {"thread_id": thread_id}}
    app.update_state(config, {"approved": approved, "approver": approver})
    result = app.invoke(None, config=config)   # None = resume, not restart

    return {"result": result.get("result", "")}
