
import argparse

import pandas as pd

from backend.backtesting import run_backtest
from backend.data.storage import ParquetCandleStore
from backend.strategies import moving_average_strategy


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backtest a moving-average strategy on stored spot candles."
    )
    parser.add_argument("--symbol", default="BTC/USDT")
    parser.add_argument(
        "--timeframe",
        default="1h",
        choices=["5m", "15m", "1h", "4h", "1d"],
    )
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument(
        "--fee", type=float, default=0.001,
        help="Fee per position change; 0.001 means 0.1%%.",
    )
    args = parser.parse_args()

    if args.fast >= args.slow:
        parser.error("--fast must be smaller than --slow")

    store = ParquetCandleStore()
    candles = store.load(
        exchange="binance",
        market_type="spot",
        symbol=args.symbol,
        timeframe=args.timeframe,
    )
    dataframe = pd.DataFrame([candle.model_dump() for candle in candles])
    strategy_data = moving_average_strategy(
        dataframe,
        fast_period=args.fast,
        slow_period=args.slow,
    )
    result = run_backtest(
        strategy_data,
        initial_capital=args.capital,
        fee_rate=args.fee,
    )

    print("\nQuantForgeAI Backtest")
    print("=" * 40)
    print(f"Symbol:             {args.symbol}")
    print(f"Timeframe:          {args.timeframe}")
    print(f"Strategy:           SMA {args.fast}/{args.slow}")
    print(f"Candles:            {len(result.candles)}")
    print(f"Initial capital:    {result.initial_capital:,.2f}")
    print(f"Final equity:       {result.final_equity:,.2f}")
    print(f"Strategy return:    {result.total_return:.2%}")
    print(f"Buy-and-hold return:{result.buy_hold_return:>10.2%}")
    print(f"Maximum drawdown:   {result.max_drawdown:.2%}")
    print(f"Completed trades:   {result.trade_count}")
    print(f"Closed-trade wins:  {result.win_rate:.2%}")
    print("\nNote: Historical simulation is not a prediction of future results.")


if __name__ == "__main__":
    main()
