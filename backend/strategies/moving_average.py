
import numpy as np
import pandas as pd


def moving_average_strategy(
    candles: pd.DataFrame,
    fast_period: int = 20,
    slow_period: int = 50,
) -> pd.DataFrame:
    """Generate long-or-cash targets from moving-average trend direction.

    The target is calculated using each candle's close. The backtester
    must delay execution so that this close is not used retroactively.
    """
    if isinstance(fast_period, bool) or not isinstance(fast_period, int):
        raise ValueError("fast_period must be a positive integer")
    if isinstance(slow_period, bool) or not isinstance(slow_period, int):
        raise ValueError("slow_period must be a positive integer")
    if fast_period < 1 or slow_period < 1:
        raise ValueError("Moving-average periods must be positive")
    if fast_period >= slow_period:
        raise ValueError("fast_period must be smaller than slow_period")
    if "close" not in candles.columns:
        raise ValueError("Input data must contain a close column")
    if candles.empty:
        raise ValueError("Cannot run a strategy on empty data")

    result = candles.copy()
    result["close"] = pd.to_numeric(result["close"], errors="raise")

    close_values = result["close"].to_numpy(dtype=float)
    if not np.isfinite(close_values).all() or (close_values <= 0).any():
        raise ValueError("Close prices must be finite and greater than zero")

    if "timestamp" in result.columns:
        result = result.sort_values("timestamp").reset_index(drop=True)

    result["fast_sma"] = result["close"].rolling(
        window=fast_period, min_periods=fast_period
    ).mean()
    result["slow_sma"] = result["close"].rolling(
        window=slow_period, min_periods=slow_period
    ).mean()

    result["target_position"] = (
        (result["fast_sma"] > result["slow_sma"])
        & result["slow_sma"].notna()
    ).astype(int)

    return result
