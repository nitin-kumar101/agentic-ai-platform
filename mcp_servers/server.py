"""Single MCP server exposing math, web, and file tools."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from urllib.parse import quote_plus

import httpx
from mcp.server import MCPServer

from rag.ingest import create_embedding
from rag.vector_store import retrieve_chunks as retrieve_from_store
from mcp_servers.tools.timeseries import register_timeseries_tools



logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("ai-platform-mcp")

WORKSPACE = Path(__file__).resolve().parent.parent / "workspace"
WORKSPACE.mkdir(exist_ok=True)

logger.info("Workspace initialized at %s", WORKSPACE)

mcp = MCPServer(
    "ai-platform-server",
    instructions=(
        "General-purpose AI platform tools: arithmetic, web search/fetch, "
        "and sandboxed file operations in the workspace."
    ),
)
register_timeseries_tools(mcp)


def _resolve_safe_path(relative_path: str) -> Path:
    logger.debug("Resolving workspace path: %s", relative_path)

    target = (WORKSPACE / relative_path).resolve()

    if not str(target).startswith(str(WORKSPACE.resolve())):
        logger.warning("Attempted workspace escape: %s", relative_path)
        raise ValueError("Path escapes the workspace sandbox")

    logger.debug("Resolved path: %s", target)
    return target


# --- Math tools ---


@mcp.tool(title="Add numbers")
def add(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b


@mcp.tool(title="Subtract numbers")
def subtract(a: float, b: float) -> float:
    """Subtract b from a."""
    return a - b


@mcp.tool(title="Multiply numbers")
def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b


@mcp.tool(title="Divide numbers")
def divide(a: float, b: float) -> float:
    """Divide a by b."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b


@mcp.tool(title="Power")
def power(base: float, exponent: float) -> float:
    """Raise base to the power of exponent."""
    return base**exponent


# --- Web tools ---


@mcp.tool(title="Search the web")
async def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search the web using DuckDuckGo instant answer API."""
    logger.info("Searching the web for: %s", query)
    url = f"https://api.duckduckgo.com/?q={quote_plus(query)}&format=json&no_redirect=1"
    async with httpx.AsyncClient(timeout=15.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        data = response.json()

    results: list[dict[str, str]] = []
    if data.get("AbstractText"):
        results.append(
            {
                "title": data.get("Heading") or query,
                "snippet": data["AbstractText"],
                "url": data.get("AbstractURL") or "",
            }
        )

    for topic in data.get("RelatedTopics", [])[:max_results]:
        if isinstance(topic, dict) and "Text" in topic:
            results.append(
                {
                    "title": topic["Text"][:80],
                    "snippet": topic["Text"],
                    "url": topic.get("FirstURL", ""),
                }
            )
    logger.info("Found %d results", len(results))
    if not results:
        results.append(
            {
                "title": "No instant results",
                "snippet": f"No DuckDuckGo instant answer for '{query}'. Try a more specific query.",
                "url": "",
            }
        )
    return results


@mcp.tool(title="Fetch URL")
async def fetch_url(url: str) -> dict[str, str]:
    """Fetch text content from a public HTTP URL."""
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        response = await client.get(url, headers={"User-Agent": "AI-Platform-MCP/1.0"})
        response.raise_for_status()
        text = response.text[:8000]
    logger.info("Fetched URL: %s", url)
    return {
        "url": str(response.url),
        "status_code": str(response.status_code),
        "content_type": response.headers.get("content-type", "unknown"),
        "content": text,
    }


@mcp.tool(title="Summarize JSON")
def summarize_json(json_text: str) -> str:
    """Pretty-print and summarize a JSON string for the model."""
    parsed = json.loads(json_text)
    if isinstance(parsed, list):
        return f"JSON array with {len(parsed)} items: {json.dumps(parsed[:3], indent=2)}"
    if isinstance(parsed, dict):
        keys = list(parsed.keys())[:10]
        return f"JSON object with keys: {keys}\nSample: {json.dumps({k: parsed[k] for k in keys[:3]}, indent=2)}"
    return str(parsed)


# --- File tools ---

@mcp.tool(title="List files")
def list_files(subdirectory: str = ".") -> list[str]:
    """List files under a workspace subdirectory."""
    base = _resolve_safe_path(subdirectory)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {subdirectory}")
    return sorted(
        str(item.relative_to(WORKSPACE)).replace("\\", "/")
        for item in base.rglob("*")
        if item.is_file()
    )


@mcp.tool(title="Read file")
def read_file(path: str) -> str:
    """Read a text file from the workspace."""
    target = _resolve_safe_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return target.read_text(encoding="utf-8")


@mcp.tool(title="Write file")
def write_file(path: str, content: str) -> dict[str, str]:
    """Write text content to a file in the workspace."""
    target = _resolve_safe_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return {"path": path, "bytes_written": str(len(content.encode("utf-8")))}


@mcp.tool(title="Delete file")
def delete_file(path: str) -> dict[str, str]:
    """Delete a file from the workspace."""
    target = _resolve_safe_path(path)
    if not target.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    target.unlink()
    return {"deleted": path}


# --- RAG retrieval tool (Chroma by default, Postgres + pgvector optional) ---
#
# VECTOR_STORE=chroma (default): persistent local Chroma collection.
# VECTOR_STORE=postgres: hybrid search on a pgvector table.
#
# Postgres table schema:
#   id, content, embedding, source, metadata
# Required for embeddings: OPENAI_API_KEY
# Required for postgres: DATABASE_URL

@mcp.tool(title="Retrieve chunks (hybrid search)")
async def retrieve_chunks(
    query: str,
    table: str = "document_chunks",
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict[str, str]]:
    """Retrieve the most relevant text chunks for a query.

    Uses VECTOR_STORE (chroma by default, or postgres). Combines semantic
    similarity with keyword matching via Reciprocal Rank Fusion (RRF).

    Args:
        query: Natural-language search query.
        table: Chroma collection name, or Postgres table name.
        top_k: Number of chunks to return.
        rrf_k: RRF smoothing constant (60 is the usual default).
    """
    if not table.replace("_", "").isalnum():
        raise ValueError("Invalid table name")

    logger.info("Hybrid retrieval on table=%s query=%r top_k=%d", table, query, top_k)
    embedding = await create_embedding(query)
    results = await retrieve_from_store(
        query=query,
        embedding=embedding,
        table=table,
        top_k=top_k,
        rrf_k=rrf_k,
    )
    logger.info("Retrieved %d chunks", len(results))
    return results


def parse_datetime_column(series):
    import pandas as pd

    try:
        parsed = pd.to_datetime(
            series,
            format="mixed",
            dayfirst=True,
            errors="coerce",
        )
    except Exception:
        parsed = pd.to_datetime(series, dayfirst=True, errors="coerce")

    invalid = parsed.isna().sum()
    if invalid:
        print(f"Skipped {invalid} rows due to invalid datetime values.")
    return parsed


@mcp.tool(title="Forecast timeseries")
def forecast_timeseries(
    csv_path: str,
    forecast_periods: int = 30,
    output_csv: str = "forecast.csv",
    plot: bool = True,
):
    """Forecast future values from a CSV file."""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        from prophet import Prophet
    except ImportError as exc:
        raise ImportError(
            "Forecast tool requires: pip install pandas matplotlib prophet"
        ) from exc

    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    df = pd.read_csv(csv_path)

    if len(df.columns) != 2:
        raise ValueError(
            "CSV must contain exactly two columns: time,value"
        )

    time_col = df.columns[0]
    value_col = df.columns[1]

    df = df.rename(
        columns={
            time_col: "ds",
            value_col: "y"
        }
    )

    df["ds"] = parse_datetime_column(df["ds"])
    df["y"] = pd.to_numeric(df["y"])

    df = df.sort_values("ds")

    df = df.dropna()

    # Detect frequency automatically
    freq = pd.infer_freq(df["ds"])

    if freq is None:

        diff = df["ds"].diff().median()

        if diff.days >= 365:
            freq = "Y"

        elif diff.days >= 30:
            freq = "M"

        elif diff.days >= 7:
            freq = "W"

        elif diff.days >= 1:
            freq = "D"

        elif diff.seconds >= 3600:
            freq = "H"

        elif diff.seconds >= 60:
            freq = "min"

        else:
            freq = "S"

    model = Prophet(
        yearly_seasonality="auto",
        weekly_seasonality="auto",
        daily_seasonality="auto"
    )

    model.fit(df)

    future = model.make_future_dataframe(
        periods=forecast_periods,
        freq=freq
    )

    forecast = model.predict(future)

    result = forecast[
        [
            "ds",
            "yhat",
            "yhat_lower",
            "yhat_upper",
        ]
    ]

    result.to_csv(output_csv, index=False)

    print(f"\nForecast saved to {output_csv}")

    if plot:

        plt.figure(figsize=(14, 6))

        plt.plot(
            df["ds"],
            df["y"],
            label="Historical",
            linewidth=2,
        )

        plt.plot(
            result["ds"],
            result["yhat"],
            label="Forecast",
            linewidth=2,
        )

        plt.fill_between(
            result["ds"],
            result["yhat_lower"],
            result["yhat_upper"],
            alpha=0.25,
            label="Confidence Interval",
        )

        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.title("Time Series Forecast")

        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Save the plot
        output_dir = "plots"
        os.makedirs(output_dir, exist_ok=True)

        output_path = os.path.join(output_dir, "forecast_plot.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")

        print(f"Plot saved to: {output_path}")

        plt.show()
        plt.close()
    return result


if __name__ == "__main__":
    import os

    host = os.getenv("MCP_HOST", "127.0.0.1")
    port = int(os.getenv("MCP_PORT", "8001"))
    mcp.run(transport="streamable-http", host=host, port=port, stateless_http=True)