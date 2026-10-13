import argparse

import pandas as pd

from backend.data.storage import ParquetCandleStore
from backend.quantitative import calculate_indicators


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calculate quantitative indicators for stored spot data."
    )

    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
    )

    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["5m", "15m", "1h", "4h", "1d"],
    )

    parser.add_argument(
        "--rows",
        type=int,
        default=10,
        help="Number of recent rows to display.",
    )

    args = parser.parse_args()

    if args.rows < 1:
        parser.error("--rows must be at least 1")

    store = ParquetCandleStore()

    candles = store.load(
        exchange="binance",
        market_type="spot",
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    dataframe = pd.DataFrame(
        [candle.model_dump() for candle in candles]
    )

    result = calculate_indicators(dataframe)

    columns = [
        "timestamp",
        "close",
        "return",
        "sma_20",
        "ema_20",
        "volatility_20",
        "rsi_14",
    ]

    print(f"\nSymbol: {args.symbol}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Stored candles: {len(result)}")
    print("\nRecent quantitative data:\n")

    print(
        result[columns]
        .tail(args.rows)
        .to_string(index=False, float_format=lambda value: f"{value:.6f}")
    )


if __name__ == "__main__":
    main()