import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict
from agents.invoice_agent import run_invoice_agent
from agents.hr_agent import run_hr_agent
from agents.escalation_agent import run_escalation_agent
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="RPA Orchestrator",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 Intelligent RPA Orchestrator")
st.caption("Powered by LangGraph + RAG + Multi-Agent AI")

# Sidebar
with st.sidebar:
    st.header("About")
    st.info("""
    This orchestrator routes business requests to specialist AI agents:
    
    📄 **Invoice Agent** — Processes invoices using company policy
    
    👥 **HR Agent** — Handles employee and onboarding requests
    
    🚨 **Escalation Agent** — Flags unclear requests for human review
    """)
    
    st.header("Policy Thresholds")
    st.warning("""
    - Under £10,000 → Auto approved
    - £10k - £50k → Manager approval
    - £50k - £100k → CFO approval
    - Above £100k → Board approval
    """)
    
    if st.button("Clear Conversation"):
        st.session_state.messages = []
        st.session_state.thread_count = 0
        st.rerun()

# Initialise session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "thread_count" not in st.session_state:
    st.session_state.thread_count = 0

# LLM
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

app = workflow.compile()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant":
            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(message["content"])
            with col2:
                if "agent" in message:
                    st.info(f"🤖 {message['agent']}")
        else:
            st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Enter your business request..."):
    
    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": prompt
    })
    
    with st.chat_message("user"):
        st.markdown(prompt)

    # Process request
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            
            st.session_state.thread_count += 1
            
            result = app.invoke({
                "request": prompt,
                "next_agent": "",
                "result": "",
                "needs_human": False,
                "amount": 0.0,
                "agent_used": ""
            })

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(result["result"])
            with col2:
                agent_used = result.get("agent_used", "Unknown")
                st.info(f"🤖 {agent_used}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["result"],
        "agent": result.get("agent_used", "Unknown")
    })