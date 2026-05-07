from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini",
                 api_key=os.environ.get("OPENAI_API_KEY"))

embeddings = OpenAIEmbeddings(
                 api_key=os.environ.get("OPENAI_API_KEY"))

# Load HR policy PDF
loader = PyPDFLoader("documents/hr_policy.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks = splitter.split_documents(pages)

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    collection_name="hr_policy"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

hr_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert HR specialist
    with deep knowledge of company HR policy.
    
    Use the policy context below to handle HR requests accurately.
    Always mention the specific policy rule you are applying.
    If request needs escalation — say so clearly.
    
    Policy Context: {context}"""),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

hr_rag_chain = (
    {
        "context": (lambda x: x["question"]) | retriever | format_docs,
        "question": (lambda x: x["question"])
    }
    | hr_prompt
    | llm
)

def run_hr_agent(request: str) -> str:
    """Run HR agent with RAG on policy document."""
    print("\n→ HR Agent: Processing with policy knowledge...")
    response = hr_rag_chain.invoke({"question": request})
    return response.content