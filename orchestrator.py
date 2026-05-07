from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from agents.invoice_agent import run_invoice_agent
from agents.hr_agent import run_hr_agent
from agents.escalation_agent import run_escalation_agent
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=os.environ.get("OPENAI_API_KEY")
)

# STATE
class OrchestratorState(TypedDict):
    request: str
    next_agent: str
    result: str
    needs_human: bool
    amount: float
    agent_used: str

# NODES
def supervisor_node(state: OrchestratorState) -> OrchestratorState:
    prompt = f"""You are a supervisor routing business requests to specialists.

ROUTING RULES:
- invoice_agent: clear invoice requests with vendor and amount
- hr_agent: employee, onboarding, HR related requests
- escalation_agent: unclear, ambiguous, missing info, exceptions

EXAMPLES:
"Process invoice from IBM for £25,000" → invoice_agent
"New employee Sara joining Monday" → hr_agent
"Something seems wrong with payments" → escalation_agent
"Lost my invoice not sure where" → escalation_agent

Request: "{state['request']}"
Respond with ONLY one word: invoice_agent, hr_agent, or escalation_agent"""

    response = llm.invoke([HumanMessage(content=prompt)])
    next_agent = response.content.strip().lower()

    amount = 0.0
    if "invoice" in state["request"].lower():
        amount_prompt = f"""Extract invoice amount as number only from:
"{state['request']}"
If no amount return 0. Return ONLY the number."""
        amount_response = llm.invoke([HumanMessage(content=amount_prompt)])
        try:
            amount = float(amount_response.content.strip())
        except:
            amount = 0.0

    state["next_agent"] = next_agent
    state["amount"] = amount
    state["needs_human"] = amount > 50000
    return state

def invoice_node(state: OrchestratorState) -> OrchestratorState:
    if state["needs_human"]:
        state["result"] = f"⚠️ HIGH VALUE INVOICE: £{state['amount']:,.0f} requires CFO approval. Please contact your CFO to proceed."
        state["agent_used"] = "Invoice Agent (Pending Approval)"
    else:
        state["result"] = run_invoice_agent(state["request"])
        state["agent_used"] = "Invoice Agent"
    return state

def hr_node(state: OrchestratorState) -> OrchestratorState:
    state["result"] = run_hr_agent(state["request"])
    state["agent_used"] = "HR Agent"
    return state

def escalation_node(state: OrchestratorState) -> OrchestratorState:
    state["result"] = run_escalation_agent(state["request"])
    state["agent_used"] = "Escalation Agent"
    return state

def route_by_agent(state: OrchestratorState) -> str:
    agent = state.get("next_agent", "escalation_agent")
    if "invoice" in agent:
        return "invoice"
    elif "hr" in agent:
        return "hr"
    return "escalation"

# BUILD GRAPH
def build_graph():
    workflow = StateGraph(OrchestratorState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("invoice", invoice_node)
    workflow.add_node("hr", hr_node)
    workflow.add_node("escalation", escalation_node)
    workflow.set_entry_point("supervisor")

    workflow.add_conditional_edges(
        "supervisor",
        route_by_agent,
        {
            "invoice": "invoice",
            "hr": "hr",
            "escalation": "escalation"
        }
    )

    workflow.add_edge("invoice", END)
    workflow.add_edge("hr", END)
    workflow.add_edge("escalation", END)

    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# PUBLIC FUNCTION — called by app.py
def run_orchestrator(request: str, thread_id: str) -> dict:
    """
    Main entry point for the orchestrator.
    Called by app.py and any other interface.
    """
    app = build_graph()
    config = {"configurable": {"thread_id": thread_id}}
    
    result = app.invoke({
        "request": request,
        "next_agent": "",
        "result": "",
        "needs_human": False,
        "amount": 0.0,
        "agent_used": ""
    }, config=config)
    
    return result


# Terminal interface — for testing without UI
if __name__ == "__main__":
    print("\n" + "="*50)
    print("  INTELLIGENT RPA ORCHESTRATOR")
    print("  Powered by LangGraph + RAG + Multi-Agent")
    print("="*50)
    print("Type your business request. Type 'quit' to exit.\n")

    thread_count = 0

    while True:
        user_input = input("You: ").strip()

        if user_input.lower() == "quit":
            break

        thread_count += 1
        result = run_orchestrator(user_input, f"request-{thread_count}")
        print(f"\nAgent: {result['agent_used']}")
        print(f"Result: {result['result']}\n")
        print("-"*50)