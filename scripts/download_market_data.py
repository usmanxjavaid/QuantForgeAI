import argparse
from datetime import UTC, datetime

from backend.data.providers import CCXTProvider
from backend.data.service import MarketDataService
from backend.data.storage import ParquetCandleStore
from backend.data.timeframes import timeframe_to_seconds


def parse_datetime(value: str) -> datetime:
    """Parse an ISO-8601 datetime; date-only values are interpreted as UTC."""
    if len(value) == 10:
        value = f"{value}T00:00:00+00:00"
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download historical cryptocurrency spot market data."
    )
    parser.add_argument("--exchange", default="binance", help="CCXT exchange id.")
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair, e.g. BTC/USDT.")
    parser.add_argument("--timeframe", default="1h", help="Fixed interval, e.g. 15m, 2h, 1d.")
    parser.add_argument("--since", type=parse_datetime, help="Start datetime, e.g. 2026-01-01.")
    parser.add_argument("--until", type=parse_datetime, help="End datetime, e.g. 2026-02-01.")
    parser.add_argument("--limit", type=int, default=500, help="Candles per request.")
    parser.add_argument(
        "--include-incomplete",
        action="store_true",
        help="Keep the currently forming candle.",
    )
    args = parser.parse_args()

    try:
        timeframe_to_seconds(args.timeframe)
    except ValueError as exc:
        parser.error(str(exc))
    if args.limit < 1:
        parser.error("--limit must be positive")

    provider = CCXTProvider(exchange_name=args.exchange)
    service = MarketDataService(provider, ParquetCandleStore())
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
            if args.until is not None:
                parser.error("--until requires --since")
            candles = service.fetch_and_store(
                symbol=args.symbol,
                timeframe=args.timeframe,
                limit=args.limit,
                include_incomplete=args.include_incomplete,
            )

        print(f"Exchange: {args.exchange}")
        print(f"Downloaded {len(candles)} candles.")
        if candles:
            print(f"First: {candles[0].timestamp}")
            print(f"Last:  {candles[-1].timestamp}")
            print(f"Close: {candles[-1].close}")
    finally:
        provider.close()


if __name__ == "__main__":
    main()