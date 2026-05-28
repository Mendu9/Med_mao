"""
mao/agents/sql_agent.py
------------------------
SQL agent — natural language → SQL → execute → synthesize answer.

Route trigger: intent == "sql"

When to route here:
  - "How many articles were published in 2023?"
  - "What is the average score for questions about science?"
  - "List the top 10 Wikipedia topics by word count"
  - "Show me all records where category = 'physics'"
  - Any query about tabular/structured data

Design:
  - Two-step LLM pattern:
      1. Text-to-SQL: mistral converts NL question → SQL (schema injected)
      2. Synthesis: mistral converts SQL results → natural language answer
  - Schema introspection: fetches live table schema from Postgres at startup
  - Guard rail: only SELECT statements are executed (no DDL/DML)
  - Parameterized queries via SQLAlchemy (no raw f-strings with user input)

Tables available (matches docker-compose Postgres setup):
  - wikipedia_articles(id, title, content, category, word_count, created_at)
  - squad_questions(id, question, answer, context_id, score)

Libraries:
  - sqlalchemy   (pip install sqlalchemy)
  - psycopg2     (pip install psycopg2-binary)

Integration points:
  - memory/mem0_handler search before / save after
  - core/state.py       MAOState contract
  - core/config.py      postgres_url
"""

from __future__ import annotations

import logging
import re
from typing import Any

import requests
from mao.core import llm as groq_llm
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.exc import SQLAlchemyError

from mao.core.config import cfg
from mao.core.state import MAOState
from mao.memory.mem0_handler import build_system_prompt, save_memory, search_memories

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database engine — lazy singleton
# ---------------------------------------------------------------------------

_engine = None


def _get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(cfg.postgres_url, pool_pre_ping=True)
        logger.info("SQLAlchemy engine created for %s", cfg.postgres_url)
    return _engine


# ---------------------------------------------------------------------------
# Schema introspection — called once and cached
# ---------------------------------------------------------------------------

_schema_cache: str | None = None


def _get_schema_description() -> str:
    """
    Introspect Postgres schema and return a DDL-style description.

    Caches the result — schema doesn't change at runtime.
    """
    global _schema_cache
    if _schema_cache is not None:
        return _schema_cache

    try:
        engine = _get_engine()
        inspector = inspect(engine)
        lines = []
        for table_name in inspector.get_table_names():
            cols = inspector.get_columns(table_name)
            col_defs = ", ".join(
                f"{c['name']} {str(c['type'])}" for c in cols
            )
            lines.append(f"  {table_name}({col_defs})")
        _schema_cache = "Database schema:\n" + "\n".join(lines)
        logger.info("Schema introspected:\n%s", _schema_cache)
        return _schema_cache
    except Exception as exc:  # noqa: BLE001
        logger.error("Schema introspection failed: %s", exc)
        _schema_cache = """\
Database schema:
  wikipedia_articles(id INTEGER, title TEXT, content TEXT, category TEXT, word_count INTEGER, created_at TIMESTAMP)
  squad_questions(id INTEGER, question TEXT, answer TEXT, context_id INTEGER, score FLOAT)"""
        return _schema_cache


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

_SQL_GEN_SYSTEM = """\
You are an expert SQL query generator.

{schema}

Rules:
  - Generate ONLY a single SELECT statement — never INSERT, UPDATE, DELETE, DROP, etc.
  - Use standard PostgreSQL syntax.
  - Limit results to 50 rows unless the user requests more.
  - Use aliases for clarity (e.g. COUNT(*) AS count).
  - Respond with ONLY the SQL query, no explanation, no markdown fences.
"""

_SYNTHESIS_SYSTEM = """\
You are a data analyst presenting SQL query results to a non-technical user.

Explain the results in plain English:
  - Directly answer the user's question
  - Reference specific numbers/values from the results
  - Keep it concise (2-5 sentences unless the data demands more)
  - If there are no results, say so clearly
"""


def sql_node(state: MAOState) -> MAOState:
    """
    LangGraph node: text-to-SQL pipeline.
    """
    user_query: str = state["user_query"]
    user_id: str    = state["user_id"]
    memory_context: str = state.get("memory_context", "")

    if not memory_context:
        memory_context = search_memories(user_query, user_id)
        state["memory_context"] = memory_context

    schema = _get_schema_description()

    # --- Step 1: Text → SQL ---
    sql_system = build_system_prompt(
        _SQL_GEN_SYSTEM.format(schema=schema),
        memory_context,
    )
    sql_query = _generate_sql(sql_system, user_query)
    logger.info("Generated SQL: %s", sql_query)

    # --- Step 2: Guard rail ---
    if not _is_safe_sql(sql_query):
        response = (
            "I can only execute SELECT queries for safety reasons. "
            "Your question appears to require a data modification query which I cannot run."
        )
        save_memory(user_query, response, user_id)
        state.update({
            "response": response,
            "agent_used": "sql",
            "metadata": {"sql": sql_query, "error": "unsafe_query"},
        })
        return state

    # --- Step 3: Execute ---
    rows, columns, error = _execute_sql(sql_query)

    if error:
        response = f"SQL execution failed: {error}"
        save_memory(user_query, response, user_id)
        state.update({
            "response": response,
            "agent_used": "sql",
            "metadata": {"sql": sql_query, "error": error},
        })
        return state

    # --- Step 4: Synthesize natural language answer ---
    results_text = _format_results(rows, columns)
    synthesis_prompt = (
        f"User question: {user_query}\n\n"
        f"SQL query executed:\n{sql_query}\n\n"
        f"Query results:\n{results_text}\n\n"
        f"Answer the user's question based on these results."
    )
    response = _synthesize(
        build_system_prompt(_SYNTHESIS_SYSTEM, memory_context),
        synthesis_prompt,
    )

    save_memory(user_query, response, user_id)

    state["response"]   = response
    state["agent_used"] = "sql"
    state["metadata"]   = {
        "sql": sql_query,
        "row_count": len(rows),
        "columns": columns,
    }
    return state


# ---------------------------------------------------------------------------
# SQL helpers
# ---------------------------------------------------------------------------

def _generate_sql(system_prompt: str, user_query: str) -> str:
    try:
        return groq_llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_query},
            ],
            temperature=0.0,
            max_tokens=256,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("SQL generation failed: %s", exc)
        return "SELECT 1"


def _is_safe_sql(sql: str) -> bool:
    """Allow only SELECT statements — block DDL/DML."""
    clean = sql.strip().lstrip(";").upper()
    return clean.startswith("SELECT")


def _execute_sql(sql: str) -> tuple[list, list, str | None]:
    """Execute SQL and return (rows, column_names, error_or_None)."""
    try:
        engine = _get_engine()
        with engine.connect() as conn:
            result = conn.execute(text(sql))
            columns = list(result.keys())
            rows = [list(row) for row in result.fetchmany(50)]
        return rows, columns, None
    except SQLAlchemyError as exc:
        logger.error("SQL execution error: %s", exc)
        return [], [], str(exc)


def _format_results(rows: list, columns: list) -> str:
    if not rows:
        return "No results returned."
    header = " | ".join(columns)
    lines = [header, "-" * len(header)]
    for row in rows[:20]:
        lines.append(" | ".join(str(v) for v in row))
    if len(rows) > 20:
        lines.append(f"... ({len(rows)} rows total, showing first 20)")
    return "\n".join(lines)


def _synthesize(system_prompt: str, user_prompt: str) -> str:
    try:
        return groq_llm.chat(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=512,
        ).strip()
    except Exception as exc:  # noqa: BLE001
        logger.error("SQL synthesis failed: %s", exc)
        return "Could not synthesize SQL results."


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    from mao.core.state import make_initial_state
    state = make_initial_state(
        "How many Wikipedia articles do we have per category?",
        "user-test",
    )
    state = sql_node(state)
    print(state["response"])
    print("SQL used:", state["metadata"].get("sql"))
