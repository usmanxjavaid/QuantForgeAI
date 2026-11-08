
import sys

import numpy as np
import pandas as pd


def main() -> None:
    parser_message = (
        "Validates stored Binance BTC/USDT hourly spot candles."
    )
    print(f"\nQuantForgeAI Market Data Validation\n{ '=' * 42}")
    print(parser_message)

    path = "data/market_data/binance/spot/BTC_USDT/1h.parquet"

    try:
        candles = pd.read_parquet(path)
    except FileNotFoundError:
        print(f"\nERROR: Stored dataset not found: {path}")
        sys.exit(1)

    if candles.empty:
        print("\nERROR: Dataset is empty.")
        sys.exit(1)

    candles["timestamp"] = pd.to_datetime(
        candles["timestamp"], utc=True, errors="coerce"
    )

    timestamp_invalid = int(candles["timestamp"].isna().sum())
    valid_timestamps = candles["timestamp"].dropna()

    duplicate_count = int(valid_timestamps.duplicated().sum())

    sorted_timestamps = valid_timestamps.sort_values()
    if not sorted_timestamps.empty:
        first_timestamp = sorted_timestamps.iloc[0]
        last_timestamp = sorted_timestamps.iloc[-1]

        expected_timestamps = pd.date_range(
            start=first_timestamp,
            end=last_timestamp,
            freq="1h",
        )
        actual_timestamps = pd.DatetimeIndex(sorted_timestamps.unique())
        missing_timestamps = expected_timestamps.difference(actual_timestamps)
    else:
        first_timestamp = None
        last_timestamp = None
        missing_timestamps = pd.DatetimeIndex([])

    price_columns = ["open", "high", "low", "close"]
    volume_column = "volume"

    for column in price_columns + [volume_column]:
        candles[column] = pd.to_numeric(candles[column], errors="coerce")

    prices = candles[price_columns].to_numpy(dtype=float)
    volumes = candles[volume_column].to_numpy(dtype=float)

    invalid_prices = (
        ~np.isfinite(prices).all(axis=1)
        | (prices <= 0).any(axis=1)
    )

    invalid_volumes = ~np.isfinite(volumes) | (volumes < 0)

    invalid_ohlc = (
        (candles["high"] < candles["low"])
        | (candles["high"] < candles["open"])
        | (candles["high"] < candles["close"])
        | (candles["low"] > candles["open"])
        | (candles["low"] > candles["close"])
    ).fillna(True)

    print(f"\nRows stored:              {len(candles):,}")
    print(f"First candle (UTC):       {first_timestamp}")
    print(f"Last candle (UTC):        {last_timestamp}")
    print(f"Invalid timestamps:       {timestamp_invalid}")
    print(f"Duplicate timestamps:     {duplicate_count}")
    print(f"Missing hourly candles:   {len(missing_timestamps)}")
    print(f"Invalid price rows:       {int(invalid_prices.sum())}")
    print(f"Invalid volume rows:      {int(invalid_volumes.sum())}")
    print(f"Invalid OHLC rows:        {int(invalid_ohlc.sum())}")

    if len(missing_timestamps) > 0:
        print("\nFirst missing timestamps (up to 10):")
        for timestamp in missing_timestamps[:10]:
            print(f"  {timestamp}")

    issues = (
        timestamp_invalid > 0
        or duplicate_count > 0
        or len(missing_timestamps) > 0
        or invalid_prices.any()
        or invalid_volumes.any()
        or invalid_ohlc.any()
    )

    if issues:
        print("\nVALIDATION RESULT: Issues detected. Investigate before backtesting.")
        sys.exit(1)

    print("\nVALIDATION RESULT: All checks passed.")


if __name__ == "__main__":
    main()
