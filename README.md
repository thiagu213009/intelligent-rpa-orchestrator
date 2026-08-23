---
title: Intelligent RPA Orchestrator
emoji: 🤖
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

# 🤖 Intelligent RPA Orchestrator

> An AI-powered business-process orchestrator that routes incoming requests to specialist agents, grounds every decision in company policy via **RAG**, and enforces a **human-in-the-loop approval gate** for high-value actions — built on a **LangGraph** multi-agent workflow.

**🚀 Live demo:** [thiagu213009-intelligent-rpa-orchestrator.hf.space](https://thiagu213009-intelligent-rpa-orchestrator.hf.space)

---

## Why this project

Rule-based automation breaks the moment a request doesn't fit the script. This project explores the next step: a **supervisor agent** that reads an incoming business request, decides which specialist should handle it, and lets each specialist reason over the actual company policy documents instead of hard-coded rules.

It combines three ideas that matter in production agentic systems:

- **Intelligent routing** — one supervisor classifies and dispatches, rather than a rigid `if/else` flow.
- **Grounded decisions (RAG)** — agents answer from policy PDFs, so responses are traceable to a source, not hallucinated.
- **Human oversight** — high-value invoices are routed through a CFO approval gate rather than being auto-actioned.

Coming from an enterprise automation background, the goal here was to show what the shift from deterministic RPA to **agentic orchestration** actually looks like in code.

---

## What it does

The orchestrator accepts a natural-language business request and routes it through a supervisor to the right specialist agent:

| Agent | Responsibility | Grounding |
|---|---|---|
| **Supervisor** | Classifies each request and routes to the correct specialist | — |
| **Invoice Agent** | Processes invoices, applies approval thresholds | RAG over `invoice_policy.pdf` |
| **HR Agent** | Answers employee/HR queries | RAG over `hr_policy.pdf` |
| **Escalation Agent** | Flags unclear or out-of-scope requests for human review | — |

**Human-in-the-loop approval:** invoices above the CFO threshold are routed into an approval gate. The LangGraph workflow implements this as a genuine interrupt — when driven through the graph directly, execution pauses and waits for an approve/reject decision before continuing. *(Note: the Streamlit chat UI currently surfaces the escalation but does not hard-block further input while an item is pending — the pause is enforced at the graph level, not yet in the UI. See [Roadmap](#roadmap).)*

---

## Architecture

```
                   User request (natural language)
                              │
                    ┌─────────▼──────────┐
                    │  Supervisor Agent   │
                    │  classify + route   │
                    └─────────┬──────────┘
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
      ┌──────────────┐ ┌──────────────┐ ┌──────────────┐
      │ Invoice Agent│ │   HR Agent   │ │  Escalation  │
      │              │ │              │ │    Agent     │
      │ RAG:         │ │ RAG:         │ │ flags for    │
      │ invoice_     │ │ hr_policy.pdf│ │ human review │
      │ policy.pdf   │ │              │ │              │
      └──────┬───────┘ └──────────────┘ └──────────────┘
             │
             ▼
     ┌───────────────────┐
     │ Value > threshold? │
     └─────────┬─────────┘
          yes  │  no
       ┌───────┴────────┐
       ▼                ▼
┌──────────────┐   auto-process
│ CFO approval  │
│ gate (HITL    │
│ interrupt)    │
└──────────────┘

        All runs traced in LangSmith
```

- **Routing** is a conditional edge from the supervisor node — the classification decides which specialist node executes.
- **Each RAG agent** retrieves relevant chunks from its policy PDF (embedded in **ChromaDB**) and answers grounded in that context.
- **The approval gate** is a conditional branch on invoice value; above the threshold, the graph enters the human-in-the-loop path.

---

## Tech stack

| Layer | Technology |
|---|---|
| Orchestration | **LangGraph** — stateful multi-agent workflow |
| Agent & RAG framework | **LangChain** |
| Vector store | **ChromaDB** — policy-document embeddings |
| LLM | **OpenAI GPT-4o-mini** |
| Interface | **Streamlit** chat UI |
| Observability | **LangSmith** — end-to-end run tracing |
| Packaging & deploy | **Docker** on **Hugging Face Spaces** |

---

## Try it — guided demo

Run these in order in the live app to see each capability:

| # | Ask this | What it demonstrates |
|---|---|---|
| 1 | `What is the CFO approval limit?` | RAG retrieval from the policy PDF |
| 2 | `Process IBM invoice for £8,000` | Auto-approval below threshold |
| 3 | `How many sick days am I entitled to?` | HR agent + RAG grounding |
| 4 | `Process Accenture invoice for £85,000` | Human-in-the-loop CFO approval gate |
| 5 | `Lost my invoice, not sure what to do` | Intelligent escalation routing |

---

## Running locally

```bash
# 1. Clone
git clone https://github.com/thiagu213009/intelligent-rpa-orchestrator.git
cd intelligent-rpa-orchestrator

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create a .env file with your API keys
#    OPENAI_API_KEY=sk-...
#    LANGSMITH_API_KEY=ls-...   (optional, enables tracing)

# 4. Launch
streamlit run app.py
```

### Configuration

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | ✅ | LLM inference + embeddings |
| `LANGSMITH_API_KEY` | ⬜ | Enables LangSmith run tracing |

On Hugging Face Spaces these live under **Settings → Variables and secrets** as **Secrets**, never in the repo.

---

## What this project demonstrates

- **Supervisor / multi-agent routing** with LangGraph conditional edges.
- **RAG grounding** — specialist agents answering from real policy documents via ChromaDB, not from model memory.
- **Human-in-the-loop control** — a genuine graph-level interrupt for high-value approvals.
- **Observability** — every run traceable in LangSmith.
- **Containerized deployment** to a live, public endpoint.
- Applying an enterprise-automation mindset (routing, approvals, policy compliance) to an **agentic AI** architecture.

---

## Roadmap

- Enforce the human-in-the-loop pause in the Streamlit UI so pending approvals block further input, matching the graph-level interrupt behavior.
- Add an approve/reject control in the UI for the CFO gate.
- Expand the policy corpus and add citation display (show which policy clause backed each answer).

---

## Author

**Thiagaraj Karthikeyan (Thiagu)** — Solution Architect moving into Agentic AI / AI Engineering, with a background in enterprise process automation.

- GitHub: [github.com/thiagu213009](https://github.com/thiagu213009)
