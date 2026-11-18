import datetime
import decimal
import json
import logging
import uuid
from typing import Any, cast

import psycopg
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from psycopg.rows import dict_row

from app.config import settings
from app.core.state import GraphState
from app.security.spotlighting import build_spotlighted_context
from app.services.llm_service import generate
from app.services.query_cache_service import query_cache
from app.services.rag_service import (
    generate_rag_response,
    get_cache_context,
    retrieve_documents,
)
from app.services.router_service import classify_intent
from app.services.sql_service import SQLService, is_select_only

logger = logging.getLogger(__name__)

sql_service = SQLService()


def _safe_json_default(obj: Any) -> Any:
    """Fallback serializer for non-JSON-native types."""
    if isinstance(obj, datetime.datetime):
        return obj.isoformat()
    if isinstance(obj, datetime.date):
        return obj.isoformat()
    if isinstance(obj, datetime.time):
        return obj.isoformat()
    if isinstance(obj, datetime.timedelta):
        return str(obj)
    if isinstance(obj, decimal.Decimal):
        return float(obj)
    if isinstance(obj, uuid.UUID):
        return str(obj)
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _safe_json_dumps(obj: Any, **kwargs: Any) -> str:
    """json.dumps with a safe fallback for exotic types."""
    return json.dumps(obj, default=_safe_json_default, **kwargs)


def route_intent(state: GraphState) -> dict:
    """LLM-based intent router for sql/rag/hybrid. Classifies intent once."""
    intent = classify_intent(state["question"])
    return {"intent": intent}


def check_rag_cache(state: GraphState) -> dict:
    """Check the RAG answer cache before performing expensive document retrieval."""
    flags = state.get("flags", {})
    cache_ctx = get_cache_context(flags)
    try:
        cached = query_cache.get_rag_answer(state["question"], cache_ctx)
    except Exception as exc:
        logger.warning("Cache check failed, continuing without cache: %s", exc)
        cached = None

    if cached is not None:
        chunk_previews = [
            cp for cp in cached.get("metadata", {}).get("retrieved_chunks", [])
        ]
        cached_meta = cached.get("metadata", {}) or {}
        cached_meta["flags"] = flags
        cached_meta["cache_hit"] = True
        return {
            "rag_cache_hit": True,
            "cache_hit": True,
            "final_answer": cached.get("answer", ""),
            "sources": cached.get("sources", []),
            "confidence": cached.get("confidence", 0.7),
            "chunk_previews": chunk_previews,
            "metadata": cached_meta,
            "reflection_iterations": cached_meta.get("reflection_iterations", 0),
            "refined_question": cached_meta.get("refined_question"),
            "cost_saved_usd": 0.0025,
        }
    return {
        "rag_cache_hit": False,
        "cache_hit": False,
    }


def retrieve_rag(state: GraphState) -> dict:
    """Retrieve document chunks and build spotlighted XML context without extra classification or SQL."""
    flags = state.get("flags", {})
    try:
        chunks = retrieve_documents(state["question"], flags=flags)
        status = "empty" if len(chunks) == 0 else "success"
        return {
            "retrieved_chunks": chunks,
            "spotlighted_context": build_spotlighted_context(chunks) if chunks else "",
            "retrieval_status": status,
            "retrieval_error": None,
        }
    except Exception as exc:
        logger.exception("Document retrieval failed: %s", exc)
        return {
            "retrieved_chunks": [],
            "spotlighted_context": "",
            "retrieval_status": "failed",
            "retrieval_error": "Document retrieval service was unavailable.",
        }


def generate_sql_node(state: GraphState) -> dict:
    """Generate SQL query and explanation for sql and hybrid intents."""
    try:
        result = sql_service.generate_sql(state["question"])
        sql = result.get("sql", "").strip()
        explanation = result.get("explanation", "")
        return {
            "generated_sql": sql,
            "sql_explanation": explanation,
            "sql_error": None,
        }
    except Exception as exc:
        logger.exception("SQL generation failed: %s", exc)
        return {
            "generated_sql": "",
            "sql_explanation": "",
            "sql_valid": False,
            "sql_execution_status": "invalid_sql",
            "sql_error": "Failed to generate SQL query.",
        }


def validate_sql_node(state: GraphState) -> dict:
    """Validate generated SQL before requesting user approval."""
    sql = state.get("generated_sql", "")
    if not sql or not sql.strip():
        return {
            "sql_valid": False,
            "sql_execution_status": "invalid_sql",
            "sql_error": "No SQL query was generated.",
        }
    if not is_select_only(sql):
        return {
            "sql_valid": False,
            "sql_execution_status": "invalid_sql",
            "sql_error": "Generated SQL query is not a permitted SELECT statement.",
        }
    return {
        "sql_valid": True,
        "sql_execution_status": "validated",
        "sql_error": None,
    }


def request_sql_approval(state: GraphState) -> dict:
    """Pause graph execution with interrupt() waiting for user approval."""
    approval = interrupt({
        "type": "sql_approval_required",
        "sql": state.get("generated_sql", ""),
        "explanation": state.get("sql_explanation", ""),
        "user_id": state.get("user_id"),
    })
    approved = bool(approval.get("approved", False))
    return {
        "sql_approved": approved,
        "sql_execution_status": "approved" if approved else "rejected",
    }


def execute_sql(state: GraphState) -> dict:
    """Execute approved SQL and store results."""
    if not state.get("sql_approved"):
        return {
            "sql_rows": [],
            "sql_execution_status": "rejected",
            "sql_error": "SQL query was not approved by user.",
            "final_answer": "SQL query was not approved.",
        }

    sql = state.get("generated_sql", "")
    try:
        rows = sql_service.execute_sql(sql)
        status = "zero_rows" if len(rows) == 0 else "success"
        return {
            "sql_rows": rows,
            "sql_execution_status": status,
            "sql_error": None,
            "final_answer": None,
        }
    except Exception as exc:
        logger.exception("SQL execution failed: %s", exc)
        return {
            "sql_rows": [],
            "sql_execution_status": "failed",
            "sql_error": "Database query execution failed.",
            "final_answer": "Database query execution failed.",
        }


def generate_answer(state: GraphState) -> dict:
    """Generate the single final answer based on intent and collected state."""
    intent = state.get("intent", "rag")

    if intent == "sql":
        status = state.get("sql_execution_status")
        flags = state.get("flags", {})

        if status == "invalid_sql":
            return {
                "final_answer": "The generated SQL query was invalid and could not be executed.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "route": "sql",
                    "retrieved_chunks": [],
                    "sql_execution_status": status,
                    "sql_error": state.get("sql_error"),
                    "generated_sql": state.get("generated_sql"),
                    "sql_explanation": state.get("sql_explanation"),
                    "sql_approved": False,
                    "flags": flags,
                },
            }

        if status == "rejected" or state.get("sql_approved") is False:
            return {
                "final_answer": "SQL query was not approved.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "route": "sql",
                    "retrieved_chunks": [],
                    "sql_execution_status": "rejected",
                    "generated_sql": state.get("generated_sql"),
                    "sql_explanation": state.get("sql_explanation"),
                    "sql_approved": False,
                    "flags": flags,
                },
            }

        if status == "failed":
            return {
                "final_answer": "Database query execution failed.",
                "sources": [],
                "confidence": 0.0,
                "metadata": {
                    "route": "sql",
                    "retrieved_chunks": [],
                    "sql_execution_status": "failed",
                    "sql_error": state.get("sql_error"),
                    "generated_sql": state.get("generated_sql"),
                    "sql_explanation": state.get("sql_explanation"),
                    "sql_approved": state.get("sql_approved", True),
                    "flags": flags,
                },
            }

        if status == "zero_rows" or not state.get("sql_rows"):
            return {
                "final_answer": "Query executed successfully, but returned no rows.",
                "sources": ["database query"],
                "confidence": 0.9,
                "metadata": {
                    "route": "sql",
                    "retrieved_chunks": [],
                    "sql_execution_status": "zero_rows",
                    "generated_sql": state.get("generated_sql"),
                    "sql_explanation": state.get("sql_explanation"),
                    "sql_rows": [],
                    "sql_approved": True,
                    "flags": flags,
                },
            }

        rows = state.get("sql_rows", [])
        answer = f"Query results:\n```\n{_safe_json_dumps(rows, indent=2)}\n```"
        return {
            "final_answer": answer,
            "sources": ["database query"],
            "confidence": 0.9,
            "metadata": {
                "route": "sql",
                "retrieved_chunks": [],
                "sql_execution_status": "success",
                "generated_sql": state.get("generated_sql"),
                "sql_explanation": state.get("sql_explanation"),
                "sql_rows": rows,
                "sql_approved": True,
                "flags": flags,
            },
        }

    if intent == "hybrid":
        return _generate_hybrid_answer(state)

    # Pure RAG path (cache miss): generate answer and cache it
    flags = state.get("flags", {})
    cache_ctx = get_cache_context(flags)
    chunks = state.get("retrieved_chunks", [])
    response = generate_rag_response(
        state["question"],
        chunks,
        flags=flags,
    )
    query_cache.set_rag_answer(state["question"], response.model_dump(), cache_ctx)
    chunk_previews = [
        chunk.model_dump() for chunk in response.metadata.retrieved_chunks
    ]
    meta_dict = response.metadata.model_dump()
    meta_dict["flags"] = flags
    meta_dict["retrieval_status"] = state.get("retrieval_status", "success")

    return {
        "final_answer": response.answer,
        "sources": response.sources,
        "confidence": response.confidence,
        "cache_hit": False,
        "chunk_previews": chunk_previews,
        "metadata": meta_dict,
        "reflection_iterations": response.metadata.reflection_iterations,
        "refined_question": response.metadata.refined_question,
    }


def _generate_hybrid_answer(state: GraphState) -> dict:
    rows = state.get("sql_rows", [])
    sql_approved = state.get("sql_approved", False)
    sql_status = state.get("sql_execution_status")
    retrieval_status = state.get("retrieval_status")
    rag_context = state.get("spotlighted_context", "")
    retrieved_chunks = state.get("retrieved_chunks", [])
    flags = state.get("flags", {})

    sql_section = ""
    sql_effective = False

    if sql_status == "invalid_sql":
        sql_section = "=== Database Query Results ===\n(SQL generation produced an invalid query. No database records retrieved.)\n\n"
    elif not sql_approved or sql_status == "rejected":
        sql_section = "=== Database Query Results ===\n(SQL query was not approved by user. No database records retrieved.)\n\n"
    elif sql_status == "failed":
        sql_section = "=== Database Query Results ===\n(Database query execution failed. No database records retrieved.)\n\n"
    elif sql_status == "zero_rows" or (sql_approved and len(rows) == 0):
        sql_section = "=== Database Query Results ===\n(Query executed successfully but returned 0 rows.)\n\n"
        sql_effective = True
    elif rows:
        sql_section = f"=== Database Query Results ===\n```\n{_safe_json_dumps(rows, indent=2)}\n```\n\n"
        sql_effective = True

    rag_section = ""
    if retrieval_status == "failed":
        rag_section = "=== Retrieved Documents ===\n(Document retrieval service was unavailable.)\n\n"
    elif retrieval_status == "empty" or not rag_context:
        rag_section = "=== Retrieved Documents ===\n(No matching documents found in knowledge base.)\n\n"
    else:
        rag_section = f"=== Retrieved Documents ===\n{rag_context}\n\n"

    system = (
        "You are an AI assistant. Synthesize database query results and "
        "retrieved documents into a single coherent answer. Cite sources using "
        "[database query] for SQL results and [source_name] for documents."
    )
    user_msg = f"{sql_section}{rag_section}Question: {state['question']}"

    try:
        result = generate(system, user_msg)
        answer_text = result["text"]
    except Exception as exc:
        logger.exception("LLM generation failed in hybrid answer: %s", exc)
        answer_text = "Failed to generate answer from combined sources."

    doc_sources = sorted(list({c.source for c in retrieved_chunks if hasattr(c, "source")}))
    has_sql_source = bool(sql_approved and rows and sql_effective and sql_status == "success")
    sources = (["database query"] if has_sql_source else []) + doc_sources
    chunk_previews = [
        {"text": c.text, "source": c.source, "score": c.score}
        for c in retrieved_chunks
        if hasattr(c, "text")
    ]
    return {
        "final_answer": answer_text,
        "sources": sources,
        "confidence": 0.85 if (has_sql_source and doc_sources) else 0.7,
        "chunk_previews": chunk_previews,
        "metadata": {
            "route": "hybrid",
            "retrieved_chunks": chunk_previews,
            "sql_approved": sql_approved,
            "sql_execution_status": sql_status,
            "retrieval_status": retrieval_status,
            "generated_sql": state.get("generated_sql"),
            "sql_explanation": state.get("sql_explanation"),
            "sql_rows": rows,
            "flags": flags,
        },
    }


def finalize(state: GraphState) -> dict:
    """Final node ensuring graph output structure is uniform."""
    return {}


def _get_checkpointer():
    try:
        conn = psycopg.connect(settings.database_url, autocommit=True, row_factory=dict_row)
        saver = PostgresSaver(conn=cast(Any, conn))
        saver.setup()
        return saver
    except Exception as exc:
        logger.warning("Could not initialize PostgresSaver (%s). Falling back to MemorySaver.", exc)
        from langgraph.checkpoint.memory import MemorySaver
        return MemorySaver()


def build_graph(checkpointer: Any | None = None):
    builder = StateGraph(state_schema=cast(Any, GraphState))
    builder.add_node("route_intent", route_intent)
    builder.add_node("check_rag_cache", check_rag_cache)
    builder.add_node("retrieve_rag", retrieve_rag)
    builder.add_node("generate_sql_node", generate_sql_node)
    builder.add_node("validate_sql_node", validate_sql_node)
    builder.add_node("request_sql_approval", request_sql_approval)
    builder.add_node("execute_sql", execute_sql)
    builder.add_node("generate_answer", generate_answer)
    builder.add_node("finalize", finalize)

    builder.add_edge(START, "route_intent")
    builder.add_conditional_edges(
        "route_intent",
        lambda s: s.get("intent", "rag"),
        {
            "sql": "generate_sql_node",
            "rag": "check_rag_cache",
            "hybrid": "retrieve_rag",
        },
    )
    builder.add_conditional_edges(
        "check_rag_cache",
        lambda s: "hit" if s.get("rag_cache_hit") else "miss",
        {
            "hit": "finalize",
            "miss": "retrieve_rag",
        },
    )
    builder.add_conditional_edges(
        "retrieve_rag",
        lambda s: s.get("intent", "rag"),
        {
            "rag": "generate_answer",
            "hybrid": "generate_sql_node",
        },
    )
    builder.add_edge("generate_sql_node", "validate_sql_node")
    builder.add_conditional_edges(
        "validate_sql_node",
        lambda s: "valid" if s.get("sql_valid") else "invalid",
        {
            "valid": "request_sql_approval",
            "invalid": "generate_answer",
        },
    )
    builder.add_edge("request_sql_approval", "execute_sql")
    builder.add_edge("execute_sql", "generate_answer")
    builder.add_edge("generate_answer", "finalize")
    builder.add_edge("finalize", END)

    if checkpointer is None:
        checkpointer = _get_checkpointer()
    return builder.compile(checkpointer=checkpointer)


graph = build_graph()
