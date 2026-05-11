import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Get API URL from environment (defaults to localhost)
API_URL = os.environ.get("API_URL", "http://localhost:8000")

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
    
    st.header("Architecture")
    st.success(f"""
    Streamlit UI → FastAPI → Orchestrator
    
    API: `{API_URL}`
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

    # Process request via FastAPI
    with st.chat_message("assistant"):
        with st.spinner("Processing your request..."):
            
            st.session_state.thread_count += 1
            
            try:
                # Call FastAPI instead of orchestrator directly
                response = requests.post(
                    f"{API_URL}/orchestrate",
                    json={
                        "request": prompt,
                        "thread_id": f"request-{st.session_state.thread_count}"
                    },
                    timeout=60
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    col1, col2 = st.columns([3, 1])
                    with col1:
                        st.markdown(result["result"])
                    with col2:
                        agent_used = result.get("agent_used", "Unknown")
                        st.info(f"🤖 {agent_used}")
                    
                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["result"],
                        "agent": agent_used
                    })
                else:
                    st.error(f"API Error: {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Cannot connect to API. Make sure FastAPI is running on port 8000.")
            except Exception as e:
                st.error(f"Error: {str(e)}")