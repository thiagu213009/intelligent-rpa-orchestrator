from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",
                 api_key=os.environ.get("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(
                 api_key=os.environ.get("OPENAI_API_KEY"))

# Load invoice policy PDF
loader = PyPDFLoader("documents/invoice_policy.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_documents(pages)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="invoice_policy"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# RAG prompt for invoice agent
invoice_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert invoice processing specialist 
    with deep knowledge of company invoice policy.
    
    Use the policy context below to handle invoice requests accurately.
    Always mention the specific policy rule you are applying.
    If request violates policy or needs escalation — say so clearly.
    
    Policy Context: {context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Invoice RAG chain
invoice_rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": (lambda x: x["question"])
    }
    | invoice_prompt
    | llm
)

def run_invoice_agent(request: str) -> str:
    """Run invoice agent with RAG on policy document."""
    print("\n→ Invoice Agent: Processing with policy knowledge...")
    response = invoice_rag_chain.invoke({"question": request})
    return response.content