# model_factory.py
# ─────────────────────────────────────────────────────────
# LLM Provider Abstraction Layer
# Controls which LLM backend is used across ALL agents.
#
# Usage — set MODEL_PROVIDER in .env:
#
#   MODEL_PROVIDER=openai   → OpenAI API (gpt-4o-mini)   [default]
#   MODEL_PROVIDER=azure    → Azure OpenAI Service        [enterprise / GDPR]
#   MODEL_PROVIDER=local    → Ollama + Llama 3.2          [air-gapped / sensitive data]
#
# Required .env variables per provider:
#
#   openai:
#     OPENAI_API_KEY=sk-proj-...
#
#   azure:
#     AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
#     AZURE_OPENAI_KEY=your-azure-api-key
#     AZURE_OPENAI_DEPLOYMENT=gpt-4o-mini
#
#   local:
#     OLLAMA_MODEL=llama3.2
#     (run: ollama pull llama3.2 && ollama serve)
# ─────────────────────────────────────────────────────────

from dotenv import load_dotenv
import os

load_dotenv()


def get_llm():
    """
    Returns the configured LLM based on MODEL_PROVIDER env variable.

    Returns:
        ChatOpenAI | AzureChatOpenAI | ChatOllama

    Raises:
        ValueError: if MODEL_PROVIDER is not one of: openai, azure, local
        ImportError: if required package not installed for chosen provider
    """
    provider = os.getenv("MODEL_PROVIDER", "openai").lower().strip()

    # ── OpenAI API ───────────────────────────────────────
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY not set in .env — "
                "add: OPENAI_API_KEY=sk-proj-..."
            )

        model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        print(f"[model_factory] Using OpenAI → {model}")

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )

    # ── Azure OpenAI ─────────────────────────────────────
    elif provider == "azure":
        from langchain_openai import AzureChatOpenAI

        endpoint   = os.environ.get("AZURE_OPENAI_ENDPOINT")
        api_key    = os.environ.get("AZURE_OPENAI_KEY")
        deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        api_ver    = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")

        if not endpoint:
            raise ValueError(
                "AZURE_OPENAI_ENDPOINT not set — "
                "add: AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com"
            )
        if not api_key:
            raise ValueError(
                "AZURE_OPENAI_KEY not set — "
                "add: AZURE_OPENAI_KEY=your-azure-key"
            )

        print(f"[model_factory] Using Azure OpenAI → deployment: {deployment}")

        return AzureChatOpenAI(
            azure_deployment=deployment,
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_ver,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )

    # ── Local Ollama ─────────────────────────────────────
    elif provider == "local":
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError(
                "langchain-ollama not installed.\n"
                "Run: pip3 install langchain-ollama\n"
                "Then: ollama pull llama3.2 && ollama serve"
            )

        model = os.getenv("OLLAMA_MODEL", "llama3.2")
        base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        print(f"[model_factory] Using Local Ollama → {model} at {base_url}")
        print("[model_factory] Data stays on YOUR machine — no external API calls")

        return ChatOllama(
            model=model,
            base_url=base_url,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0.7"))
        )

    # ── Unknown provider ─────────────────────────────────
    else:
        raise ValueError(
            f"Unknown MODEL_PROVIDER: '{provider}'\n"
            f"Valid options: openai, azure, local\n"
            f"Check your .env file: MODEL_PROVIDER=openai"
        )


def get_embeddings():
    """
    Returns the configured Embeddings model.
    Currently always uses OpenAI Embeddings.
    For local embeddings, Ollama embedding support can be added.
    """
    from langchain_openai import OpenAIEmbeddings

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY required for embeddings")

    print("[model_factory] Using OpenAI Embeddings → text-embedding-3-small")
    return OpenAIEmbeddings(
        api_key=api_key,
        model="text-embedding-3-small"
    )
