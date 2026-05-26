# agents/invoice_agent.py
# ─────────────────────────────────────────────────────────
# Invoice Agent — RAG-powered invoice processing specialist
# Reads from documents/invoice_policy.pdf
# Uses model_factory for pluggable LLM support
# ─────────────────────────────────────────────────────────

from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import re
import os

# ── Import from model_factory (replaces direct ChatOpenAI) ─
from model_factory import get_llm, get_embeddings

load_dotenv()

# ── LLM and Embeddings via factory ──────────────────────
llm        = get_llm()          # reads MODEL_PROVIDER from .env
embeddings = get_embeddings()   # always OpenAI embeddings for now


# ── Build RAG pipeline from invoice policy PDF ──────────
def build_invoice_retriever():
    """Load invoice_policy.pdf and build ChromaDB retriever"""
    print("[invoice_agent] Loading invoice_policy.pdf...")

    loader = PyPDFLoader("documents/invoice_policy.pdf")
    pages  = loader.load()

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    ).split_documents(pages)

    print(f"[invoice_agent] Created {len(chunks)} chunks from {len(pages)} pages")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="invoice_policy"
    )
    return vectorstore.as_retriever(search_kwargs={"k": 4})


# Initialise retriever once at module load
invoice_retriever = build_invoice_retriever()


# ── RAG Prompt ───────────────────────────────────────────
invoice_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a senior invoice processing specialist at Acme Corporation.
You have deep expertise in our company's invoice policy, vendor management and payment procedures.

STRICT RULES:
- Answer ONLY from the policy context provided below
- Always cite the specific policy rule or section you are applying
- For amounts above £50,000: always flag CFO approval required
- For amounts above £100,000: flag CFO + Board approval required
- If answer is NOT in the context, say exactly:
  "This query falls outside the loaded policy scope. Please contact finance@acmecorp.com or raise a ticket with the Finance team."
- NEVER make up policy rules or thresholds
- Be precise with amounts and approval levels

Policy Context:
{context}"""),
    ("human", "{request}")
])


def format_docs(docs) -> str:
    """Join retrieved chunks into context string"""
    return "\n\n".join(
        f"[Page {doc.metadata.get('page', '?')+1}] {doc.page_content}"
        for doc in docs
    )


# ── RAG Chain with lambda fix (dict → string) ───────────
invoice_chain = (
    {
        "context": (lambda x: x["request"]) | invoice_retriever | format_docs,
        "request": (lambda x: x["request"])
    }
    | invoice_prompt
    | llm
)


def extract_amount(text: str) -> float:
    """Extract invoice amount from request text"""
    # Match: £8,000 or £85000 or £8.5k etc.
    patterns = [
        r'£([\d,]+(?:\.\d{1,2})?)',    # £8,000 or £8000
        r'GBP\s*([\d,]+)',              # GBP 8000
        r'([\d,]+(?:\.\d{1,2})?)\s*(?:pounds|gbp)',  # 8000 pounds
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1).replace(",", ""))
            except:
                pass
    return 0.0


def run_invoice_agent(request: str) -> dict:
    """
    Run the invoice RAG agent.

    Args:
        request: Natural language invoice request

    Returns:
        dict with keys: result, agent_used, invoice_amount, needs_approval, status
    """
    print("\n→ Invoice Agent: Processing request...")
    print(f"  Request: {request[:80]}...")

    # Run RAG chain
    response = invoice_chain.invoke({"request": request})

    # Extract amount for approval routing
    amount = extract_amount(request)
    needs_approval = amount > 50000

    print(f"  Amount detected: £{amount:,.2f}")
    print(f"  Needs approval:  {needs_approval}")

    return {
        "result":          response.content,
        "agent_used":      "Invoice Agent",
        "invoice_amount":  amount,
        "needs_approval":  needs_approval,
        "status":          "pending_approval" if needs_approval else "completed"
    }
