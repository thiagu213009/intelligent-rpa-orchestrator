from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",
                 api_key=os.environ.get("OPENAI_API_KEY"))

def run_escalation_agent(request: str) -> str:
    """Handle unclear or exception requests."""
    print("\n→ Escalation Agent: Flagging for human review...")
    
    prompt = f"""You are an escalation specialist.
    
This request needs human attention: {request}

Provide:
1. Summary of what was requested
2. Why this needs human review
3. What information is needed to resolve it
4. Urgency level: low/medium/high"""
    
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content