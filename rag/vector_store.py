"""Vector store backends for RAG: Chroma (default) or PostgreSQL + pgvector."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from pathlib import Path
from typing import Any

from core.config import ROOT, get_vector_store_name

logger = logging.getLogger("ai-platform-rag")

DEFAULT_COLLECTION = "document_chunks"


class ChunkRecord:
    def __init__(
        self,
        content: str,
        embedding: list[float],
        source: str,
        metadata: dict[str, Any],
        chunk_id: str | None = None,
    ) -> None:
        self.content = content
        self.embedding = embedding
        self.source = source
        self.metadata = metadata
        self.chunk_id = chunk_id or str(uuid.uuid4())


class VectorStore(ABC):
    name: str

    @abstractmethod
    async def upsert(self, chunks: list[ChunkRecord], table: str = DEFAULT_COLLECTION) -> int:
        """Insert chunks. Returns the number of chunks written."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        embedding: list[float],
        table: str = DEFAULT_COLLECTION,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict[str, str]]:
        """Return ranked chunks for a query embedding."""


def _chroma_metadata(chunk: ChunkRecord) -> dict[str, Any]:
    meta: dict[str, Any] = {"source": chunk.source}
    for key, value in chunk.metadata.items():
        if isinstance(value, (str, int, float, bool)):
            meta[key] = value
        else:
            meta[key] = json.dumps(value)
    return meta


def _rrf_merge(
    ranked_lists: list[list[str]],
    docs: dict[str, str],
    metas: dict[str, dict[str, Any]],
    rrf_k: int,
    top_k: int,
) -> list[dict[str, str]]:
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, item_id in enumerate(ranked, start=1):
            scores[item_id] += 1.0 / (rrf_k + rank)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    results: list[dict[str, str]] = []
    for item_id, score in ordered:
        meta = metas.get(item_id) or {}
        source = str(meta.get("source") or "")
        public_meta = {k: v for k, v in meta.items() if k != "source"}
        results.append(
            {
                "id": item_id,
                "content": docs.get(item_id, ""),
                "source": source,
                "metadata": json.dumps(public_meta),
                "score": f"{score:.6f}",
            }
        )
    return results


class ChromaVectorStore(VectorStore):
    name = "chroma"

    def __init__(self) -> None:
        persist_dir = Path(
            os.getenv("CHROMA_PERSIST_DIR", str(ROOT / "rag" / "chroma_data"))
        )
        if not persist_dir.is_absolute():
            persist_dir = ROOT / persist_dir
        self._persist_dir = str(persist_dir)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
            except ImportError as exc:
                raise ImportError("Chroma backend requires: pip install chromadb") from exc

            os.makedirs(self._persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
            logger.info("Chroma persistent client at %s", self._persist_dir)
        return self._client

    def _collection(self, table: str):
        return self._get_client().get_or_create_collection(
            name=table,
            metadata={"hnsw:space": "cosine"},
        )

    def _upsert_sync(self, chunks: list[ChunkRecord], table: str) -> int:
        if not chunks:
            return 0
        collection = self._collection(table)
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            embeddings=[chunk.embedding for chunk in chunks],
            metadatas=[_chroma_metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def _retrieve_sync(
        self,
        query: str,
        embedding: list[float],
        table: str,
        top_k: int,
        rrf_k: int,
    ) -> list[dict[str, str]]:
        collection = self._collection(table)
        if collection.count() == 0:
            return []

        limit = max(top_k * 4, top_k)
        docs: dict[str, str] = {}
        metas: dict[str, dict[str, Any]] = {}

        semantic = collection.query(
            query_embeddings=[embedding],
            n_results=min(limit, collection.count()),
            include=["documents", "metadatas"],
        )
        semantic_ids = list(semantic.get("ids", [[]])[0] or [])
        semantic_docs = list(semantic.get("documents", [[]])[0] or [])
        semantic_metas = list(semantic.get("metadatas", [[]])[0] or [])
        for item_id, doc, meta in zip(semantic_ids, semantic_docs, semantic_metas):
            docs[item_id] = doc or ""
            metas[item_id] = meta or {}

        keyword_ids: list[str] = []
        terms = [term for term in query.lower().split() if len(term) > 2][:5]
        seen: set[str] = set()
        for term in terms:
            try:
                got = collection.get(
                    where_document={"$contains": term},
                    include=["documents", "metadatas"],
                    limit=limit,
                )
            except Exception:
                logger.debug("Chroma keyword filter unavailable; using semantic ranks only")
                break
            for item_id, doc, meta in zip(
                got.get("ids") or [],
                got.get("documents") or [],
                got.get("metadatas") or [],
            ):
                docs[item_id] = doc or ""
                metas[item_id] = meta or {}
                if item_id not in seen:
                    seen.add(item_id)
                    keyword_ids.append(item_id)

        ranked_lists = [semantic_ids]
        if keyword_ids:
            ranked_lists.append(keyword_ids)
        return _rrf_merge(ranked_lists, docs, metas, rrf_k, top_k)

    async def upsert(self, chunks: list[ChunkRecord], table: str = DEFAULT_COLLECTION) -> int:
        return await asyncio.to_thread(self._upsert_sync, chunks, table)

    async def retrieve(
        self,
        query: str,
        embedding: list[float],
        table: str = DEFAULT_COLLECTION,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict[str, str]]:
        return await asyncio.to_thread(
            self._retrieve_sync, query, embedding, table, top_k, rrf_k
        )


class PostgresVectorStore(VectorStore):
    name = "postgres"

    def __init__(self) -> None:
        self._pool = None

    async def _get_pool(self):
        if self._pool is None:
            try:
                import asyncpg
                from pgvector.asyncpg import register_vector
            except ImportError as exc:
                raise ImportError(
                    "Postgres backend requires: pip install asyncpg pgvector"
                ) from exc

            dsn = os.getenv("DATABASE_URL")
            if not dsn:
                raise RuntimeError(
                    "DATABASE_URL is required when VECTOR_STORE=postgres"
                )

            async def _init_connection(conn) -> None:
                await register_vector(conn)

            self._pool = await asyncpg.create_pool(
                dsn=dsn, min_size=1, max_size=5, init=_init_connection
            )
            logger.info("Postgres connection pool created")
        return self._pool

    async def upsert(self, chunks: list[ChunkRecord], table: str = DEFAULT_COLLECTION) -> int:
        if not table.replace("_", "").isalnum():
            raise ValueError("Invalid table name")
        pool = await self._get_pool()
        async with pool.acquire() as conn:
            for chunk in chunks:
                await conn.execute(
                    f"""
                    INSERT INTO {table}
                    (content, source, metadata, embedding)
                    VALUES ($1, $2, $3, $4)
                    """,
                    chunk.content,
                    chunk.source,
                    json.dumps(chunk.metadata),
                    chunk.embedding,
                )
        return len(chunks)

    async def retrieve(
        self,
        query: str,
        embedding: list[float],
        table: str = DEFAULT_COLLECTION,
        top_k: int = 5,
        rrf_k: int = 60,
    ) -> list[dict[str, str]]:
        if not table.replace("_", "").isalnum():
            raise ValueError("Invalid table name")

        pool = await self._get_pool()
        sql = f"""
            WITH semantic_search AS (
                SELECT id, content, source, metadata,
                       RANK() OVER (ORDER BY embedding <=> $1) AS rank
                FROM {table}
                ORDER BY embedding <=> $1
                LIMIT $2
            ),
            keyword_search AS (
                SELECT id, content, source, metadata,
                       RANK() OVER (
                           ORDER BY ts_rank_cd(to_tsvector('english', content),
                                               plainto_tsquery('english', $3)) DESC
                       ) AS rank
                FROM {table}
                WHERE to_tsvector('english', content) @@ plainto_tsquery('english', $3)
                LIMIT $2
            )
            SELECT
                COALESCE(s.id, k.id) AS id,
                COALESCE(s.content, k.content) AS content,
                COALESCE(s.source, k.source) AS source,
                COALESCE(s.metadata, k.metadata) AS metadata,
                (COALESCE(1.0 / ($4 + s.rank), 0.0)
                 + COALESCE(1.0 / ($4 + k.rank), 0.0)) AS score
            FROM semantic_search s
            FULL OUTER JOIN keyword_search k ON s.id = k.id
            ORDER BY score DESC
            LIMIT $2;
        """
        async with pool.acquire() as conn:
            rows = await conn.fetch(sql, embedding, top_k, query, rrf_k)

        results: list[dict[str, str]] = []
        for row in rows:
            metadata = row["metadata"]
            if metadata is not None and not isinstance(metadata, str):
                metadata = json.dumps(metadata)
            results.append(
                {
                    "id": str(row["id"]),
                    "content": row["content"],
                    "source": row["source"] or "",
                    "metadata": metadata or "{}",
                    "score": f"{row['score']:.6f}",
                }
            )
        return results


_store: VectorStore | None = None


def get_vector_store() -> VectorStore:
    global _store
    if _store is None:
        backend = get_vector_store_name()
        if backend in {"postgres", "postgresql", "pgvector"}:
            _store = PostgresVectorStore()
        elif backend in {"chroma", "chromadb"}:
            _store = ChromaVectorStore()
        else:
            raise ValueError(
                f"Unknown VECTOR_STORE={backend!r}. Use 'chroma' or 'postgres'."
            )
        logger.info("Using vector store backend: %s", _store.name)
    return _store


async def retrieve_chunks(
    query: str,
    embedding: list[float],
    table: str = DEFAULT_COLLECTION,
    top_k: int = 5,
    rrf_k: int = 60,
) -> list[dict[str, str]]:
    return await get_vector_store().retrieve(
        query=query,
        embedding=embedding,
        table=table,
        top_k=top_k,
        rrf_k=rrf_k,
    )
