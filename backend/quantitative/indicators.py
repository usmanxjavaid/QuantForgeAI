import numpy as np
import pandas as pd


def calculate_returns(close: pd.Series) -> pd.Series:
    """Calculate percentage returns between consecutive closes."""
    return close.pct_change(fill_method=None)


def calculate_sma(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Calculate the simple moving average."""
    _validate_period(period)

    return close.rolling(
        window=period,
        min_periods=period,
    ).mean()


def calculate_ema(
    close: pd.Series,
    period: int = 20,
) -> pd.Series:
    """Calculate an exponential moving average."""
    _validate_period(period)

    return close.ewm(
        span=period,
        adjust=False,
        min_periods=period,
    ).mean()


def calculate_volatility(
    returns: pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Calculate rolling sample standard deviation of returns.

    This is rolling volatility, not annualized volatility.
    """
    _validate_period(period)

    return returns.rolling(
        window=period,
        min_periods=period,
    ).std(ddof=1)


def calculate_rsi(
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Calculate RSI using rolling average gains and losses.

    This uses a simple rolling-average RSI calculation.
    """
    _validate_period(period)

    changes = close.diff()

    gains = changes.clip(lower=0)
    losses = -changes.clip(upper=0)

    average_gain = gains.rolling(
        window=period,
        min_periods=period,
    ).mean()

    average_loss = losses.rolling(
        window=period,
        min_periods=period,
    ).mean()

    relative_strength = average_gain / average_loss.replace(0, np.nan)

    rsi = 100 - (100 / (1 + relative_strength))

    # No losses with positive gains means RSI is 100.
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain > 0),
        100.0,
    )

    # No gains or losses means RSI is neutral at 50.
    rsi = rsi.mask(
        (average_loss == 0) & (average_gain == 0),
        50.0,
    )

    return rsi


def calculate_indicators(
    candles: pd.DataFrame,
    sma_period: int = 20,
    ema_period: int = 20,
    volatility_period: int = 20,
    rsi_period: int = 14,
) -> pd.DataFrame:
    """
    Return a copy of candle data with calculated indicators.

    Expected input:
        A DataFrame containing a numeric 'close' column.

    Original input data is not modified.
    """
    required_columns = {"close"}

    missing = required_columns - set(candles.columns)

    if missing:
        raise ValueError(
            f"Missing required columns: {sorted(missing)}"
        )

    if candles.empty:
        raise ValueError("Cannot calculate indicators on empty data")

    periods = {
        "sma_period": sma_period,
        "ema_period": ema_period,
        "volatility_period": volatility_period,
        "rsi_period": rsi_period,
    }

    for name, period in periods.items():
        try:
            _validate_period(period)
        except ValueError as exc:
            raise ValueError(f"{name}: {exc}") from exc

    result = candles.copy()

    result["close"] = pd.to_numeric(
        result["close"],
        errors="raise",
    )

    if not np.isfinite(result["close"].to_numpy(dtype=float)).all():
        raise ValueError("Close prices must be finite numbers")

    if (result["close"] <= 0).any():
        raise ValueError("Close prices must be greater than zero")

    # Calculate in chronological order if timestamps are available.
    if "timestamp" in result.columns:
        result = (
            result.sort_values("timestamp")
            .reset_index(drop=True)
        )

    result["return"] = calculate_returns(result["close"])

    result[f"sma_{sma_period}"] = calculate_sma(
        result["close"],
        sma_period,
    )

    result[f"ema_{ema_period}"] = calculate_ema(
        result["close"],
        ema_period,
    )

    result[f"volatility_{volatility_period}"] = calculate_volatility(
        result["return"],
        volatility_period,
    )

    result[f"rsi_{rsi_period}"] = calculate_rsi(
        result["close"],
        rsi_period,
    )

    return result


def _validate_period(period: int) -> None:
    if isinstance(period, bool) or not isinstance(period, int):
        raise ValueError("period must be a positive integer")

    if period < 1:
        raise ValueError("period must be a positive integer")