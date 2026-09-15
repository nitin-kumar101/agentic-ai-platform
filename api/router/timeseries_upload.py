"""
Upload endpoint for time-series Excel/CSV files.

Calls timeseries_core directly (not over MCP) so the upload response is
synchronous and doesn't add a network hop. The same core module is used
by the MCP tools, so the agent can act on files uploaded here too.

Integration (in api/main.py):

    from api.routes.timeseries_upload import router as timeseries_router
    app.include_router(timeseries_router)
"""

from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from core.config import ROOT
from mcp_servers.tools import timeseries_core as core

router = APIRouter(prefix="/api/timeseries", tags=["timeseries"])

UPLOAD_DIR = ROOT / "uploads" / "timeseries"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB


class ForecastRequest(BaseModel):
    file_id: str
    periods: int = 30
    target_column: str | None = None
    datetime_column: str | None = None
    freq: str | None = None
    use_detected_regressors: bool = False


def _save_upload(file: UploadFile) -> tuple[str, Path]:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in core.SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Use .csv or .xlsx.",
        )

    file_id = uuid.uuid4().hex
    dest = UPLOAD_DIR / f"{file_id}{suffix}"

    size = 0
    with dest.open("wb") as out:
        while chunk := file.file.read(1024 * 1024):
            size += len(chunk)
            if size > MAX_UPLOAD_BYTES:
                dest.unlink(missing_ok=True)
                raise HTTPException(status_code=413, detail="File too large (max 25MB).")
            out.write(chunk)

    return file_id, dest


def _resolve_path(file_id: str) -> Path:
    matches = list(UPLOAD_DIR.glob(f"{file_id}.*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"Unknown file_id '{file_id}'.")
    return matches[0]


@router.post("/upload")
async def upload_timeseries_file(file: UploadFile = File(...)) -> dict:
    """
    Upload a CSV/XLSX file and get back the auto-detected datetime,
    dependent (target), and independent (feature) columns.

    Returns a file_id to pass to /forecast, /decompose, or /stationarity.
    """
    file_id, dest = _save_upload(file)
    try:
        inspection = core.inspect_file(dest)
    except core.TimeSeriesError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return {"file_id": file_id, **inspection}


@router.post("/forecast")
async def forecast_timeseries(req: ForecastRequest) -> dict:
    """Run a Prophet forecast on a previously uploaded file."""
    path = _resolve_path(req.file_id)
    try:
        return core.forecast(
            path,
            periods=req.periods,
            target_column=req.target_column,
            datetime_column=req.datetime_column,
            freq=req.freq,
            use_detected_regressors=req.use_detected_regressors,
        )
    except core.TimeSeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/decompose")
async def decompose_timeseries(
    file_id: str, target_column: str | None = None, period: int | None = None
) -> dict:
    """Trend/seasonality/residual decomposition for a previously uploaded file."""
    path = _resolve_path(file_id)
    try:
        return core.decompose(path, target_column=target_column, period=period)
    except core.TimeSeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/stationarity")
async def stationarity_timeseries(file_id: str, target_column: str | None = None) -> dict:
    """ADF stationarity test for a previously uploaded file."""
    path = _resolve_path(file_id)
    try:
        return core.stationarity_test(path, target_column=target_column)
    except core.TimeSeriesError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
