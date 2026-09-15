"""
Core time-series logic shared by:
  - mcp_servers/tools/timeseries.py   (MCP tool wrappers, called by the agent)
  - api/routes/timeseries_upload.py   (FastAPI upload endpoint, called directly)

Kept dependency-light and framework-free so both callers can import it
without pulling in MCP or FastAPI.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Column names strongly hinting "this is the thing we forecast"
_TARGET_NAME_HINTS = (
    "target", "y", "sales", "revenue", "demand", "price", "volume",
    "value", "amount", "quantity", "qty", "count", "load", "usage",
    "consumption", "close", "traffic",
)

# Column names strongly hinting "this is a timestamp"
_DATETIME_NAME_HINTS = ("date", "ds", "time", "timestamp", "datetime", "period")

SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class TimeSeriesError(ValueError):
    """Raised for user-facing data problems (bad file, no numeric columns, etc.)."""


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_dataframe(file_path: str | Path) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame with light cleanup."""
    path = Path(file_path)
    if not path.exists():
        raise TimeSeriesError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise TimeSeriesError(
            f"Unsupported file type '{suffix}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    try:
        if suffix == ".csv":
            df = pd.read_csv(path)
        else:
            df = pd.read_excel(path, engine="openpyxl" if suffix == ".xlsx" else None)
    except Exception as exc:  # noqa: BLE001
        raise TimeSeriesError(f"Could not parse '{path.name}': {exc}") from exc

    if df.empty:
        raise TimeSeriesError(f"'{path.name}' has no rows.")

    df.columns = [str(c).strip() for c in df.columns]
    return df


# --------------------------------------------------------------------------
# Variable detection
# --------------------------------------------------------------------------

def detect_datetime_column(df: pd.DataFrame) -> str | None:
    """Find the most likely datetime column by name, then by parseability."""
    # 1) Name-based match
    for col in df.columns:
        if col.lower() in _DATETIME_NAME_HINTS or any(
            hint in col.lower() for hint in _DATETIME_NAME_HINTS
        ):
            return col

    # 2) Already a datetime dtype
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # 3) Try parsing each object column; pick the one with the highest
    #    successful-parse rate (must beat 90%).
    best_col, best_rate = None, 0.0
    for col in df.select_dtypes(include=["object"]).columns:
        parsed = pd.to_datetime(df[col], errors="coerce")
        rate = parsed.notna().mean()
        if rate > best_rate:
            best_col, best_rate = col, rate

    return best_col if best_rate >= 0.9 else None


def detect_target_and_features(
    df: pd.DataFrame, datetime_col: str | None
) -> dict[str, Any]:
    """
    Heuristically pick a dependent variable (target) and independent
    variables (features) from the remaining numeric columns.

    Priority for the target:
      1. A column whose name matches a common "target" keyword
         (sales, revenue, volume, price, ...)
      2. The numeric column with the highest coefficient of variation
         (most "interesting" signal to forecast)
    Everything else numeric becomes a candidate feature/regressor.
    """
    candidate_cols = [c for c in df.columns if c != datetime_col]
    numeric_cols = [
        c for c in candidate_cols if pd.api.types.is_numeric_dtype(df[c])
    ]

    if not numeric_cols:
        raise TimeSeriesError(
            "No numeric columns found to forecast. Found columns: "
            f"{list(df.columns)}"
        )

    # 1) Name-hint match
    target = next(
        (c for c in numeric_cols if c.lower() in _TARGET_NAME_HINTS)
        , None,
    )
    if target is None:
        target = next(
            (c for c in numeric_cols
             if any(hint in c.lower() for hint in _TARGET_NAME_HINTS)),
            None,
        )

    # 2) Fall back to highest coefficient of variation (std / mean)
    if target is None:
        best_col, best_cv = numeric_cols[0], -1.0
        for c in numeric_cols:
            mean = df[c].mean()
            std = df[c].std()
            cv = abs(std / mean) if mean not in (0, None) and pd.notna(mean) else 0.0
            if cv > best_cv:
                best_col, best_cv = c, cv
        target = best_col

    features = [c for c in numeric_cols if c != target]

    return {
        "datetime_column": datetime_col,
        "target_column": target,
        "feature_columns": features,
        "all_numeric_columns": numeric_cols,
        "row_count": len(df),
    }


def inspect_file(file_path: str | Path) -> dict[str, Any]:
    """Full auto-detection pass: columns, datetime col, target, features, date range, inferred frequency."""
    df = load_dataframe(file_path)
    datetime_col = detect_datetime_column(df)

    if datetime_col is None:
        raise TimeSeriesError(
            "Could not identify a datetime column. Rename one column to "
            "'date', 'ds', or 'timestamp', or pass datetime_column explicitly."
        )

    parsed_dates = pd.to_datetime(df[datetime_col], errors="coerce")
    if parsed_dates.isna().mean() > 0.1:
        raise TimeSeriesError(
            f"Column '{datetime_col}' does not look like a consistent datetime column."
        )

    detection = detect_target_and_features(df, datetime_col)

    sorted_dates = parsed_dates.dropna().sort_values()
    inferred_freq = pd.infer_freq(sorted_dates) if len(sorted_dates) > 3 else None

    return {
        "file": str(Path(file_path).name),
        "columns": list(df.columns),
        "date_range": {
            "start": str(sorted_dates.iloc[0]) if len(sorted_dates) else None,
            "end": str(sorted_dates.iloc[-1]) if len(sorted_dates) else None,
        },
        "inferred_frequency": inferred_freq,
        **detection,
    }


# --------------------------------------------------------------------------
# Forecasting (statsmodels — Exponential Smoothing / SARIMAX)
# --------------------------------------------------------------------------

# Seasonal period heuristics by inferred pandas frequency code
_SEASONAL_PERIOD_BY_FREQ = {
    "H": 24, "T": 60, "min": 60, "D": 7, "B": 5,
    "W": 52, "M": 12, "MS": 12, "Q": 4, "QS": 4, "A": 1, "Y": 1,
}


def _build_regular_series(
    df: pd.DataFrame, datetime_col: str, value_col: str, freq: str | None
) -> tuple[pd.Series, str]:
    """
    Return (series, freq_str): a value column indexed by a *regular*
    DatetimeIndex at the given/inferred frequency, with gaps
    interpolated. statsmodels forecasting models require even spacing.
    """
    work = df[[datetime_col, value_col]].dropna().copy()
    work[datetime_col] = pd.to_datetime(work[datetime_col], errors="coerce")
    work = work.dropna(subset=[datetime_col])
    work = work.sort_values(datetime_col).drop_duplicates(subset=[datetime_col], keep="last")
    series = work.set_index(datetime_col)[value_col]

    resolved_freq = freq or pd.infer_freq(series.index)
    if resolved_freq is None:
        # Fall back to the median gap between observations.
        deltas = series.index.to_series().diff().dropna()
        if deltas.empty:
            raise TimeSeriesError("Not enough distinct timestamps to infer a frequency.")
        median_delta = deltas.median()
        offset = pd.tseries.frequencies.to_offset(median_delta)
        resolved_freq = offset.freqstr

    series = series.asfreq(resolved_freq)
    series = series.interpolate(method="linear", limit_direction="both")
    return series, resolved_freq


def _seasonal_period_for(freq: str) -> int:
    base = freq.split("-")[0]  # e.g. "W-SUN" -> "W"
    for key, period in _SEASONAL_PERIOD_BY_FREQ.items():
        if base == key or base.startswith(key):
            return period
    return 1  # no meaningful seasonality guess


def _fit_ets_and_predict(series: pd.Series, periods: int, seasonal_period: int):
    """Fit Holt-Winters (ETS) and return (mean, lower, upper) for `periods` future steps at 80% CI."""
    from statsmodels.tsa.exponential_smoothing.ets import ETSModel

    use_seasonal = seasonal_period > 1 and len(series) >= 2 * seasonal_period
    use_trend = len(series) >= 4

    model = ETSModel(
        series,
        error="add",
        trend="add" if use_trend else None,
        damped_trend=use_trend,
        seasonal="add" if use_seasonal else None,
        seasonal_periods=seasonal_period if use_seasonal else None,
    )
    fit = model.fit(disp=False)
    pred = fit.get_prediction(start=len(series), end=len(series) + periods - 1)
    mean = pred.predicted_mean
    ci = pred.pred_int(alpha=0.2)  # ~80% interval, matching Prophet's default
    return mean, ci.iloc[:, 0], ci.iloc[:, 1]


def _fit_sarimax_and_predict(series: pd.Series, exog: pd.DataFrame, future_exog: pd.DataFrame, periods: int):
    """Fit a simple SARIMAX(1,1,1) with exogenous regressors and return (mean, lower, upper) at 80% CI."""
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    model = SARIMAX(
        series, exog=exog, order=(1, 1, 1),
        enforce_stationarity=False, enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    pred = fit.get_forecast(steps=periods, exog=future_exog)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.2)
    return mean, ci.iloc[:, 0], ci.iloc[:, 1]


def forecast(
    file_path: str | Path,
    periods: int = 30,
    target_column: str | None = None,
    datetime_column: str | None = None,
    freq: str | None = None,
    use_detected_regressors: bool = False,
) -> dict[str, Any]:
    """
    Forecast a time series with statsmodels: Exponential Smoothing (ETS /
    Holt-Winters) by default, or SARIMAX with exogenous regressors when
    use_detected_regressors=True.

    If target_column / datetime_column aren't supplied, they're
    auto-detected the same way `inspect_file` does.

    If use_detected_regressors=True, other numeric columns are used as
    exogenous regressors. Their *future* values are themselves forecast
    with independent ETS models first (a simple cascade), since SARIMAX
    needs future regressor values to project the target.

    Output shape matches Prophet-style forecasts: a list of
    {ds, yhat, yhat_lower, yhat_upper} at an ~80% confidence interval.
    """
    df = load_dataframe(file_path)

    if datetime_column is None:
        datetime_column = detect_datetime_column(df)
    if datetime_column is None:
        raise TimeSeriesError("Could not detect a datetime column; pass datetime_column explicitly.")

    detection = detect_target_and_features(df, datetime_column)
    if target_column is None:
        target_column = detection["target_column"]

    feature_cols = detection["feature_columns"] if use_detected_regressors else []

    target_series, resolved_freq = _build_regular_series(df, datetime_column, target_column, freq)
    if len(target_series) < 10:
        raise TimeSeriesError(
            f"Not enough rows ({len(target_series)}) with valid data in '{target_column}' to forecast. Need at least 10."
        )
    seasonal_period = _seasonal_period_for(resolved_freq)

    future_index = pd.date_range(
        start=target_series.index[-1], periods=periods + 1, freq=resolved_freq
    )[1:]

    if feature_cols:
        # Cascade: forecast each regressor's future values with its own ETS model.
        exog_train = {}
        exog_future = {}
        used_features = []
        for feat in feature_cols:
            try:
                feat_series, _ = _build_regular_series(df, datetime_column, feat, resolved_freq)
                feat_series = feat_series.reindex(target_series.index).interpolate(limit_direction="both")
                if feat_series.isna().any() or len(feat_series) < 10:
                    continue
                feat_mean, _, _ = _fit_ets_and_predict(feat_series, periods, seasonal_period)
                exog_train[feat] = feat_series
                exog_future[feat] = pd.Series(feat_mean.values, index=future_index)
                used_features.append(feat)
            except Exception:  # noqa: BLE001
                # Skip any feature that can't be modeled reliably; don't fail the whole forecast.
                logger.warning("Skipping regressor '%s' — could not model its future values.", feat)
                continue

        if used_features:
            exog_df = pd.DataFrame(exog_train)
            future_exog_df = pd.DataFrame(exog_future)
            mean, lower, upper = _fit_sarimax_and_predict(target_series, exog_df, future_exog_df, periods)
            feature_cols = used_features
        else:
            mean, lower, upper = _fit_ets_and_predict(target_series, periods, seasonal_period)
            feature_cols = []
    else:
        mean, lower, upper = _fit_ets_and_predict(target_series, periods, seasonal_period)

    forecast_records = [
        {
            "ds": str(ts),
            "yhat": float(m),
            "yhat_lower": float(lo),
            "yhat_upper": float(hi),
        }
        for ts, m, lo, hi in zip(future_index, mean, lower, upper)
    ]

    return {
        "target_column": target_column,
        "datetime_column": datetime_column,
        "regressors_used": feature_cols,
        "frequency": resolved_freq,
        "periods": periods,
        "history_rows": len(target_series),
        "forecast": forecast_records,
    }


# --------------------------------------------------------------------------
# Decomposition & stationarity
# --------------------------------------------------------------------------

def decompose(
    file_path: str | Path,
    target_column: str | None = None,
    datetime_column: str | None = None,
    period: int | None = None,
) -> dict[str, Any]:
    from statsmodels.tsa.seasonal import seasonal_decompose

    df = load_dataframe(file_path)
    if datetime_column is None:
        datetime_column = detect_datetime_column(df)
    if datetime_column is None:
        raise TimeSeriesError("Could not detect a datetime column; pass datetime_column explicitly.")

    detection = detect_target_and_features(df, datetime_column)
    target_column = target_column or detection["target_column"]

    series = df[[datetime_column, target_column]].dropna()
    series[datetime_column] = pd.to_datetime(series[datetime_column], errors="coerce")
    series = series.dropna().sort_values(datetime_column).set_index(datetime_column)[target_column]

    inferred_freq = pd.infer_freq(series.index)
    period = period or ({"D": 7, "H": 24, "M": 12, "MS": 12, "Q": 4}.get(inferred_freq, 7))

    if len(series) < period * 2:
        raise TimeSeriesError(
            f"Need at least {period * 2} rows for period={period}; have {len(series)}."
        )

    result = seasonal_decompose(series, model="additive", period=period, extrapolate_trend="freq")

    def _clean(s: pd.Series) -> list[float | None]:
        return [None if pd.isna(v) else float(v) for v in s]

    return {
        "target_column": target_column,
        "datetime_column": datetime_column,
        "period_used": period,
        "dates": [str(d) for d in series.index],
        "observed": _clean(result.observed),
        "trend": _clean(result.trend),
        "seasonal": _clean(result.seasonal),
        "residual": _clean(result.resid),
    }


def stationarity_test(
    file_path: str | Path,
    target_column: str | None = None,
    datetime_column: str | None = None,
) -> dict[str, Any]:
    from statsmodels.tsa.stattools import adfuller

    df = load_dataframe(file_path)
    if datetime_column is None:
        datetime_column = detect_datetime_column(df)
    detection = detect_target_and_features(df, datetime_column)
    target_column = target_column or detection["target_column"]

    series = df[target_column].dropna()
    if len(series) < 10:
        raise TimeSeriesError(f"Not enough data points ({len(series)}) for ADF test.")

    stat, p_value, used_lag, n_obs, crit_values, _ = adfuller(series)

    return {
        "target_column": target_column,
        "adf_statistic": float(stat),
        "p_value": float(p_value),
        "used_lag": int(used_lag),
        "n_obs": int(n_obs),
        "critical_values": {k: float(v) for k, v in crit_values.items()},
        "is_stationary_at_5pct": bool(p_value < 0.05),
    }
