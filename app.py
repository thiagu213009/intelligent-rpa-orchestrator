import streamlit as st
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from orchestrator import run_orchestrator
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
            
            # CLEAN — just calls orchestrator, no logic here
            result = run_orchestrator(
                prompt,
                f"request-{st.session_state.thread_count}"
            )

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