from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from orchestrator import run_orchestrator
from dotenv import load_dotenv
import uvicorn
import os

load_dotenv()

app = FastAPI(
    title="Intelligent RPA Orchestrator API",
    description="AI-powered business process orchestrator",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class OrchestratorRequest(BaseModel):
    request: str
    thread_id: str = "default"

class OrchestratorResponse(BaseModel):
    request: str
    agent_used: str
    result: str
    needs_human: bool
    amount: float

@app.get("/")
def health_check():
    return {
        "status": "running",
        "service": "RPA Orchestrator API",
        "version": "1.0.0"
    }

@app.post("/orchestrate", response_model=OrchestratorResponse)
def orchestrate(body: OrchestratorRequest):
    result = run_orchestrator(
        request=body.request,
        thread_id=body.thread_id
    )
    
    return OrchestratorResponse(
        request=body.request,
        agent_used=result.get("agent_used", "Unknown"),
        result=result.get("result", ""),
        needs_human=result.get("needs_human", False),
        amount=result.get("amount", 0.0)
    )

if __name__ == "__main__":
    uvicorn.run(
    app,
    host=os.getenv("HOST", "0.0.0.0"),
    port=int(os.getenv("PORT", 8000))
)