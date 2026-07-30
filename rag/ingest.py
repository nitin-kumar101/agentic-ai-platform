import os
import json
import pdfplumber
import asyncpg

from openai import AsyncOpenAI
from pgvector.asyncpg import register_vector


EMBEDDING_MODEL = os.getenv(
    "RAG_EMBEDDING_MODEL",
    "text-embedding-3-small"
)


DATABASE_URL = os.environ["DATABASE_URL"]

_openai_client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"]
)


# -------------------------------
# PostgreSQL Connection
# -------------------------------

async def create_pool():

    async def init_connection(conn):
        await register_vector(conn)

    pool = await asyncpg.create_pool(
        DATABASE_URL,
        min_size=1,
        max_size=5,
        init=init_connection
    )

    return pool



# -------------------------------
# PDF Extraction
# -------------------------------

def extract_pdf_text(pdf_path: str):

    pages = []

    with pdfplumber.open(pdf_path) as pdf:

        for page_number, page in enumerate(pdf.pages, start=1):

            text = page.extract_text()

            if text:

                pages.append(
                    {
                        "page": page_number,
                        "text": text
                    }
                )

    return pages



# -------------------------------
# Text Chunking
# -------------------------------

def chunk_text(
    text,
    chunk_size=1000,
    overlap=200
):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start = end - overlap


    return chunks



# -------------------------------
# Embedding Generation
# -------------------------------

async def create_embedding(text):

    response = await _openai_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response.data[0].embedding



# -------------------------------
# Insert into PostgreSQL
# -------------------------------

async def insert_chunk(
    pool,
    content,
    embedding,
    source,
    metadata
):

    async with pool.acquire() as conn:

        await conn.execute(
            """
            INSERT INTO document_chunks
            (
                content,
                source,
                metadata,
                embedding
            )
            VALUES
            ($1,$2,$3,$4)
            """,

            content,
            source,
            json.dumps(metadata),
            embedding
        )



# -------------------------------
# Main ingestion pipeline
# -------------------------------

async def ingest_pdf(pdf_path):

    pool = await create_pool()


    pages = extract_pdf_text(pdf_path)


    total_chunks = 0


    for page in pages:


        chunks = chunk_text(
            page["text"]
        )


        for index, chunk in enumerate(chunks):

            embedding = await create_embedding(
                chunk
            )


            metadata = {

                "page": page["page"],

                "chunk_number": index

            }


            await insert_chunk(

                pool,

                content=chunk,

                embedding=embedding,

                source=pdf_path,

                metadata=metadata
            )


            total_chunks += 1


            print(
                f"Inserted chunk {total_chunks}"
            )


    await pool.close()


    print(
        f"Completed ingestion. Total chunks={total_chunks}"
    )



# -------------------------------
# Run
# -------------------------------

if __name__ == "__main__":

    import asyncio

    asyncio.run(
        ingest_pdf(
            "sample.pdf"
        )
    )