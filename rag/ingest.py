from __future__ import annotations

import os
from pathlib import Path

from rag.vector_store import ChunkRecord, DEFAULT_COLLECTION, get_vector_store

EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "text-embedding-3-small")

_openai_client = None


def _get_openai_client():
    global _openai_client
    if _openai_client is None:
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        _openai_client = AsyncOpenAI(api_key=api_key)
    return _openai_client


def extract_pdf_text(pdf_path: str) -> list[dict]:
    import pdfplumber

    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, start=1):
            text = page.extract_text()
            if text:
                pages.append({"page": page_number, "text": text})
    return pages


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
        if overlap >= chunk_size:
            start = end
    return chunks


async def create_embedding(text: str) -> list[float]:
    response = await _get_openai_client().embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


async def ingest_pdf(
    pdf_path: str,
    table: str = DEFAULT_COLLECTION,
) -> dict:
    store = get_vector_store()
    pages = extract_pdf_text(pdf_path)
    source = Path(pdf_path).name
    records: list[ChunkRecord] = []

    for page in pages:
        chunks = chunk_text(page["text"])
        for index, chunk in enumerate(chunks):
            embedding = await create_embedding(chunk)
            records.append(
                ChunkRecord(
                    content=chunk,
                    embedding=embedding,
                    source=source,
                    metadata={"page": page["page"], "chunk_number": index},
                )
            )

    written = await store.upsert(records, table=table)
    return {
        "backend": store.name,
        "table": table,
        "total_chunks": written,
    }


if __name__ == "__main__":
    import asyncio

    asyncio.run(ingest_pdf("sample.pdf"))
