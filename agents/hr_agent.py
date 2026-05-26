# agents/hr_agent.py
# ─────────────────────────────────────────────────────────
# HR Agent — RAG-powered HR policy specialist
# Reads from documents/hr_policy.pdf
# Uses model_factory for pluggable LLM support
# ─────────────────────────────────────────────────────────

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

# ── Import from model_factory ────────────────────────────
from model_factory import get_llm, get_embeddings

load_dotenv()

# ── LLM and Embeddings via factory ──────────────────────
llm        = get_llm()
embeddings = get_embeddings()


# ── Build RAG pipeline from HR policy PDF ───────────────
def build_hr_retriever():
    """Load hr_policy.pdf and build ChromaDB retriever"""
    print("[hr_agent] Loading hr_policy.pdf...")

    loader = PyPDFLoader("documents/hr_policy.pdf")
    pages  = loader.load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    ).split_documents(pages)

    print(f"[hr_agent] Created {len(chunks)} chunks from {len(pages)} pages")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hr_policy"   # separate collection from invoice!
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Initialise retriever once at module load
hr_retriever = build_hr_retriever()


# ── RAG Prompt ───────────────────────────────────────────
hr_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior HR Business Partner at Acme Corporation.
You have comprehensive knowledge of our HR policies, employee procedures and compliance requirements.

STRICT RULES:
- Answer ONLY from the HR policy context provided below
- Always cite the relevant HR policy section
- For sensitive matters (dismissal, grievance, disciplinary): advise employee to contact HR Director directly
- Never disclose other employees' personal information
- Maintain strict confidentiality at all times
- If answer is NOT in the context, say exactly:
  "This query falls outside the loaded HR policy scope. Please contact hr@acmecorp.com or call our confidential HR helpline: 0800-ACME-HR"
- NEVER make up policy rules, entitlements or procedures

HR Policy Context:
{context}"""),
    ("human", "{request}")
])


def format_docs(docs) -> str:
    """Join retrieved chunks into context string"""
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')+1}] {doc.page_content}"
        for doc in docs
    )


# ── RAG Chain with lambda fix ────────────────────────────
hr_chain = (
    {
        "context": (lambda x: x["request"]) | hr_retriever | format_docs,
        "request": (lambda x: x["request"])
    }
    | hr_prompt
    | llm
)


def run_hr_agent(request: str) -> dict:
    """
    Run the HR RAG agent.

    Args:
        request: Natural language HR request

    Returns:
        dict with keys: result, agent_used, invoice_amount, needs_approval, status
    """
    print("\n→ HR Agent: Processing request...")
    print(f"  Request: {request[:80]}...")

    response = hr_chain.invoke({"request": request})

    return {
        "result":         response.content,
        "agent_used":     "HR Agent",
        "invoice_amount": 0.0,
        "needs_approval": False,
        "status":         "completed"
    }
