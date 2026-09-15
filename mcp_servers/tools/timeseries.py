"""
Time-series MCP tools.

Registers 4 tools on a given FastMCP instance:
  - ts_inspect_file        auto-detect datetime/target/feature columns
  - ts_forecast             Prophet forecast (optionally with regressors)
  - ts_decompose             trend / seasonality / residual decomposition
  - ts_stationarity_test    ADF stationarity test

Integration (in mcp_servers/server.py):

    from mcp_servers.tools.timeseries import register_timeseries_tools
    register_timeseries_tools(mcp)

All file_path arguments are expected to be paths already on disk
(e.g. under uploads/) — the agent doesn't need to pass raw file bytes.
"""

from __future__ import annotations

from typing import Any

from mcp_servers.tools import timeseries_core as core


def register_timeseries_tools(mcp) -> None:
    """Attach the time-series tools to an existing FastMCP server instance."""

    @mcp.tool()
    def ts_inspect_file(file_path: str) -> dict[str, Any]:
        """
        Inspect a CSV/XLSX time series file: detect the datetime column,
        the likely dependent variable (target to forecast), and candidate
        independent variables (features), plus date range and frequency.

        Args:
            file_path: Path to a .csv or .xlsx file on disk.

        Returns:
            Dict with columns, datetime_column, target_column,
            feature_columns, date_range, inferred_frequency, row_count.
        """
        try:
            return core.inspect_file(file_path)
        except core.TimeSeriesError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def ts_forecast(
        file_path: str,
        periods: int = 30,
        target_column: str | None = None,
        datetime_column: str | None = None,
        freq: str | None = None,
        use_detected_regressors: bool = False,
    ) -> dict[str, Any]:
        """
        Forecast a time series with Prophet. If target_column /
        datetime_column are omitted, they're auto-detected.

        Args:
            file_path: Path to a .csv or .xlsx file on disk.
            periods: Number of future steps to forecast.
            target_column: Dependent variable to forecast. Auto-detected if omitted.
            datetime_column: Datetime column name. Auto-detected if omitted.
            freq: Pandas frequency string (e.g. "D", "H", "MS"). Auto-inferred if omitted.
            use_detected_regressors: If True, other numeric columns are
                used as Prophet regressors (their future values are
                forecast first via their own Prophet models).

        Returns:
            Dict with target_column, regressors_used, frequency, and
            forecast (list of {ds, yhat, yhat_lower, yhat_upper}).
        """
        try:
            return core.forecast(
                file_path,
                periods=periods,
                target_column=target_column,
                datetime_column=datetime_column,
                freq=freq,
                use_detected_regressors=use_detected_regressors,
            )
        except core.TimeSeriesError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def ts_decompose(
        file_path: str,
        target_column: str | None = None,
        datetime_column: str | None = None,
        period: int | None = None,
    ) -> dict[str, Any]:
        """
        Decompose a time series into trend, seasonal, and residual
        components (additive model).

        Args:
            file_path: Path to a .csv or .xlsx file on disk.
            target_column: Column to decompose. Auto-detected if omitted.
            datetime_column: Datetime column name. Auto-detected if omitted.
            period: Seasonal period (e.g. 7 for weekly seasonality on
                daily data). Inferred from frequency if omitted.
        """
        try:
            return core.decompose(
                file_path,
                target_column=target_column,
                datetime_column=datetime_column,
                period=period,
            )
        except core.TimeSeriesError as exc:
            return {"error": str(exc)}

    @mcp.tool()
    def ts_stationarity_test(
        file_path: str,
        target_column: str | None = None,
        datetime_column: str | None = None,
    ) -> dict[str, Any]:
        """
        Run an Augmented Dickey-Fuller test to check whether a series is
        stationary (useful before choosing an ARIMA-style model).

        Args:
            file_path: Path to a .csv or .xlsx file on disk.
            target_column: Column to test. Auto-detected if omitted.
            datetime_column: Datetime column name. Auto-detected if omitted.
        """
        try:
            return core.stationarity_test(
                file_path,
                target_column=target_column,
                datetime_column=datetime_column,
            )
        except core.TimeSeriesError as exc:
            return {"error": str(exc)}
