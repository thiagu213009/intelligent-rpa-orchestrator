from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, List
from agents.invoice_agent import run_invoice_agent
from agents.hr_agent import run_hr_agent
from agents.escalation_agent import run_escalation_agent
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",
                 api_key=os.environ.get("OPENAI_API_KEY"))

# STATE
class OrchestratorState(TypedDict):
    request: str
    next_agent: str
    result: str
    needs_human: bool
    amount: float

# NODE 1 — Supervisor
def supervisor_node(state: OrchestratorState) -> OrchestratorState:
    print("\n→ Supervisor: Analysing request...")

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

    # Extract amount if invoice request
    amount = 0.0
    if "invoice" in state["request"].lower():
        amount_prompt = f"""Extract the invoice amount as a number only from:
"{state['request']}"
If no amount found return 0.
Return ONLY the number, no symbols or text."""
        amount_response = llm.invoke([HumanMessage(content=amount_prompt)])
        try:
            amount = float(amount_response.content.strip())
        except:
            amount = 0.0

    if amount > 0:
        print(f"  Decision: {next_agent} | Amount: £{amount}")
    else:
        print(f"  Decision: {next_agent}")
    state["next_agent"] = next_agent
    state["amount"] = amount
    state["needs_human"] = amount > 50000
    return state

# NODE 2 — Invoice node
def invoice_node(state: OrchestratorState) -> OrchestratorState:
    if state["needs_human"]:
        print(f"\n⏸  HIGH VALUE INVOICE — £{state['amount']} needs approval")
        state["result"] = f"PENDING APPROVAL: Invoice for £{state['amount']} requires CFO approval."
    else:
        state["result"] = run_invoice_agent(state["request"])
    return state

# NODE 3 — HR node
def hr_node(state: OrchestratorState) -> OrchestratorState:
    state["result"] = run_hr_agent(state["request"])
    return state

# NODE 4 — Escalation node
def escalation_node(state: OrchestratorState) -> OrchestratorState:
    state["result"] = run_escalation_agent(state["request"])
    return state

# NODE 5 — Human approval node
def human_approval_node(state: OrchestratorState) -> OrchestratorState:
    print("\n" + "="*50)
    print(f"CFO APPROVAL REQUIRED")
    print(f"Invoice Amount: £{state['amount']}")
    print("="*50)
    decision = input("Approve or reject? ").strip().lower()
    
    if decision == "approve":
        state["result"] = run_invoice_agent(
            f"CFO has approved this invoice. Please confirm processing: {state['request']}"
)
    else:
        state["result"] = f"Invoice rejected by CFO."
    return state

# ROUTING FUNCTIONS
def route_by_agent(state: OrchestratorState) -> str:
    agent = state.get("next_agent", "escalation_agent")
    if "invoice" in agent:
        return "invoice"
    elif "hr" in agent:
        return "hr"
    return "escalation"

def route_invoice(state: OrchestratorState) -> str:
    if state["needs_human"]:
        return "needs_approval"
    return "done"

# BUILD GRAPH
workflow = StateGraph(OrchestratorState)

workflow.add_node("supervisor", supervisor_node)
workflow.add_node("invoice", invoice_node)
workflow.add_node("hr", hr_node)
workflow.add_node("escalation", escalation_node)
workflow.add_node("human_approval", human_approval_node)

workflow.set_entry_point("supervisor")

# Supervisor routes to agents
workflow.add_conditional_edges(
    "supervisor",
    route_by_agent,
    {
        "invoice": "invoice",
        "hr": "hr",
        "escalation": "escalation"
    }
)

# Invoice routes to human if high value
workflow.add_conditional_edges(
    "invoice",
    route_invoice,
    {
        "needs_approval": "human_approval",
        "done": END
    }
)

workflow.add_edge("hr", END)
workflow.add_edge("escalation", END)
workflow.add_edge("human_approval", END)

# Compile
memory = MemorySaver()
app = workflow.compile(checkpointer=memory)

# Interactive orchestrator
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
    config = {"configurable": {"thread_id": f"request-{thread_count}"}}

    result = app.invoke({
        "request": user_input,
        "next_agent": "",
        "result": "",
        "needs_human": False,
        "amount": 0.0
    }, config=config)

    print(f"\nRESULT:\n{result['result']}\n")
    print("-"*50)