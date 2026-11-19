import os

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str = ""
    gemini_api_key: str = ""
    llm_provider: str = "gemini"
    llm_model_answer: str = "gpt-4o"
    llm_model_grader: str = "gpt-4o-mini"
    embedding_model: str = "gemini-embedding-001"

    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "documents"
    qdrant_api_key: str = ""

    database_url: str = "postgresql://postgres:postgres@localhost:5432/adv_rag"

    upstash_redis_url: str = ""
    upstash_redis_token: str = ""
    cache_ttl_embeddings: int = 604_800
    cache_ttl_rag: int = 3_600
    cache_ttl_sql_gen: int = 86_400
    cache_ttl_sql_result: int = 900
    cache_ttl_intent: int = 86_400

    storage_backend: str = "local"

    tavily_api_key: str = ""


    jwt_secret: str = ""
    jwt_expiration_minutes: int = 60

    rate_limit_requests: int = 20
    rate_limit_window_seconds: int = 60
    max_tokens_per_user_daily: int = 100_000
    auth_login_rate_limit_per_min: int = 5
    auth_register_rate_limit_per_hour: int = 3

    max_input_tokens: int = 3_000
    reserved_context_tokens: int = 1_000
    reserved_output_tokens: int = 1_000

    prompt_injection_threshold: float = 0.75
    toxicity_threshold: float = 0.75
    output_toxicity_threshold: float = 0.5
    max_validation_retries: int = 2

    hyde_num_hypotheses: int = 3
    hyde_enabled_by_default: bool = False
    hybrid_search_enabled: bool = True
    rrf_k: int = 60
    reranker_backend: str = "local"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    voyage_api_key: str = ""
    voyage_model: str = "rerank-2.5"
    reranker_initial_top_k: int = 20
    reranking_enabled_by_default: bool = True
    crag_relevance_threshold: float = 0.7
    crag_ambiguous_threshold: float = 0.5
    crag_enabled_by_default: bool = True
    reflection_min_score: float = 0.85
    max_reflection_retries: int = 2
    self_reflective_enabled_by_default: bool = False


    vanna_model: str = "gpt-4o"
    vanna_temperature: float = 0.0
    vanna_seed: int = 42

    log_json: bool = False
    log_level: str = "INFO"

    langsmith_tracing: bool = True
    langsmith_api_key: str = ""
    langsmith_project: str = "Agentic-RAG-System"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()

# Export LangSmith env vars to os.environ so LangGraph and @traceable pick them up automatically
if settings.langsmith_api_key:
    tracing_val = "true" if settings.langsmith_tracing else "false"
    os.environ.setdefault("LANGSMITH_TRACING", tracing_val)
    os.environ.setdefault("LANGCHAIN_TRACING_V2", tracing_val)
    os.environ.setdefault("LANGSMITH_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGCHAIN_API_KEY", settings.langsmith_api_key)
    os.environ.setdefault("LANGSMITH_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGCHAIN_PROJECT", settings.langsmith_project)
    os.environ.setdefault("LANGSMITH_ENDPOINT", settings.langsmith_endpoint)




    



    


