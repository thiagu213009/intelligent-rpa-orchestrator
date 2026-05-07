# Intelligent RPA Orchestrator

An AI-powered business process orchestrator that combines 
Multi-Agent Architecture, RAG, and LangGraph workflows.

## What it does
- Routes business requests to specialist AI agents
- Invoice Agent — processes invoices using company policy (RAG)
- HR Agent — handles employee requests using HR policy (RAG)
- Escalation Agent — flags unclear requests for human review
- Human in the loop — CFO approval for high value invoices

## Tech Stack
- LangGraph — multi-agent workflow orchestration
- LangChain — agent and RAG framework
- ChromaDB — vector database for policy documents
- OpenAI GPT-4o-mini — LLM brain
- Streamlit — chat interface
- LangSmith — tracing and observability

## Architecture
User Request → Supervisor Agent → Routes to:
├── Invoice Agent (RAG from invoice_policy.pdf)
├── HR Agent (RAG from hr_policy.pdf)
└── Escalation Agent (flags for human review)

## Setup
1. Clone the repo
2. pip install -r requirements.txt
3. Create .env file with your API keys
4. streamlit run app.py

## Author
Built as part of AI Engineer learning journey