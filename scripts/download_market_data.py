import argparse
from datetime import datetime, timezone

from backend.data.providers import CCXTProvider
from backend.data.service import MarketDataService
from backend.data.storage import ParquetCandleStore


def parse_datetime(value: str) -> datetime:
    """
    Parse an ISO-8601 datetime.

    Examples:
        2026-01-01
        2026-01-01T00:00:00+00:00
    """

    if len(value) == 10:
        value = f"{value}T00:00:00+00:00"

    parsed = datetime.fromisoformat(value)

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(timezone.utc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical Binance spot market data."
    )

    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair, e.g. BTC/USDT",
    )

    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["5m", "15m", "1h", "4h", "1d"],
    )

    parser.add_argument(
        "--since",
        type=parse_datetime,
        help="Start datetime, e.g. 2026-01-01",
    )

    parser.add_argument(
        "--until",
        type=parse_datetime,
        help="End datetime, e.g. 2026-02-01",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Number of candles for a single request.",
    )

    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Keep the currently forming candle.",
    )

    args = parser.parse_args()

    provider = CCXTProvider()
    store = ParquetCandleStore()
    service = MarketDataService(provider, store)

    try:
        if args.since is not None:
            candles = service.fetch_range_and_store(
                symbol=args.symbol,
                timeframe=args.timeframe,
                since=args.since,
                until=args.until,
                batch_size=args.limit,
                include_incomplete=args.include_incomplete,
            )
        else:
            candles = service.fetch_and_store(
                symbol=args.symbol,
                timeframe=args.timeframe,
                limit=args.limit,
                include_incomplete=args.include_incomplete,
            )

        print(f"Downloaded {len(candles)} candles.")

        if candles:
            print(f"First: {candles[0].timestamp}")
            print(f"Last:  {candles[-1].timestamp}")
            print(f"Close: {candles[-1].close}")

    finally:
        provider.close()


if __name__ == "__main__":
    main()