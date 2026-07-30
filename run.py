"""CLI entrypoint for the AI platform API."""

import os

import uvicorn
from dotenv import load_dotenv

from core.config import ROOT

load_dotenv(ROOT / ".env")

if __name__ == "__main__":
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", "8000"))
    uvicorn.run("api.main:app", host=host, port=port, reload=True)