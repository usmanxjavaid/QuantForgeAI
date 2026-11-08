
import argparse
import math

import pandas as pd

from backend.backtesting import (
    calculate_performance_metrics,
    run_backtest,
)
from backend.data.storage import ParquetCandleStore
from backend.strategies import moving_average_strategy


PERIODS_PER_YEAR = {
    "5m": 365 * 24 * 12,
    "15m": 365 * 24 * 4,
    "1h": 365 * 24,
    "4h": 365 * 6,
    "1d": 365,
}


def format_metric(value: float, suffix: str = "") -> str:
    """Format finite numbers while handling infinity safely."""
    if math.isinf(value):
        return f"Infinity{suffix}"
    if math.isnan(value):
        return f"N/A{suffix}"
    return f"{value:.2f}{suffix}"


def print_metrics(
    label: str,
    result,
    data: pd.DataFrame,
    periods_per_year: int,
) -> None:
    """Print detailed performance metrics for one data segment."""
    first_timestamp = data["timestamp"].iloc[0]
    last_timestamp = data["timestamp"].iloc[-1]

    metrics = calculate_performance_metrics(
        candles=result.candles,
        initial_capital=result.initial_capital,
        periods_per_year=periods_per_year,
    )

    print(f"\n{label}")
    print("-" * 56)
    print(f"Period:                  {first_timestamp} to {last_timestamp}")
    print(f"Candles:                 {len(data):,}")
    print(f"Initial capital:         {result.initial_capital:,.2f}")
    print(f"Final equity:            {result.final_equity:,.2f}")

    print("\nReturns")
    print("-" * 56)
    print(f"Strategy return:         {metrics.total_return:.2%}")
    print(f"Annualized return:       {metrics.annualized_return:.2%}")
    print(f"Buy-and-hold return:     {metrics.buy_hold_return:.2%}")

    print("\nRisk")
    print("-" * 56)
    print(f"Maximum drawdown:        {metrics.max_drawdown:.2%}")
    print(f"Drawdown duration:       {metrics.max_drawdown_duration:,} candles")
    print(
        "Annualized volatility:  "
        f"{metrics.annualized_volatility:.2%}"
    )
    print(f"Sharpe ratio:            {format_metric(metrics.sharpe_ratio)}")
    print(f"Sortino ratio:           {format_metric(metrics.sortino_ratio)}")
    print(f"Calmar ratio:            {format_metric(metrics.calmar_ratio)}")

    print("\nTrade Statistics")
    print("-" * 56)
    print(f"Completed trades:        {metrics.trade_count}")
    print(f"Win rate:                {metrics.win_rate:.2%}")
    print(
        "Average trade return:   "
        f"{metrics.average_trade_return:.2%}"
    )
    print(f"Profit factor:           {format_metric(metrics.profit_factor)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a moving-average strategy on chronological "
            "train/test periods."
        )
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
        "--fee",
        type=float,
        default=0.001,
        help="Fee per buy or sell; 0.001 means 0.1%.",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.70,
        help=(
            "Fraction of earliest candles used for training; "
            "must be between 0 and 1."
        ),
    )
    args = parser.parse_args()

    if args.fast < 1 or args.slow < 1 or args.fast >= args.slow:
        parser.error(
            "--fast and --slow must be positive, "
            "with --fast smaller than --slow"
        )

    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be greater than 0 and less than 1")

    if args.capital <= 0 or not math.isfinite(args.capital):
        parser.error("--capital must be finite and greater than zero")

    if not 0 <= args.fee < 1 or not math.isfinite(args.fee):
        parser.error("--fee must be finite and between 0 inclusive and 1 exclusive")

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
    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        utc=True,
    )
    dataframe = (
        dataframe.sort_values("timestamp")
        .reset_index(drop=True)
    )

    if len(dataframe) < args.slow * 2:
        parser.error(
            f"Need at least {args.slow * 2} candles for this evaluation; "
            f"found {len(dataframe)}."
        )

    # Calculate indicators chronologically. Rolling indicators use past
    # observations only, so the test period does not inform earlier signals.
    strategy_data = moving_average_strategy(
        dataframe,
        fast_period=args.fast,
        slow_period=args.slow,
    )

    split_index = int(len(strategy_data) * args.train_ratio)

    train_data = strategy_data.iloc[:split_index].copy()
    test_data = strategy_data.iloc[split_index:].copy()

    train_result = run_backtest(
        train_data,
        initial_capital=args.capital,
        fee_rate=args.fee,
    )
    test_result = run_backtest(
        test_data,
        initial_capital=args.capital,
        fee_rate=args.fee,
    )

    periods_per_year = PERIODS_PER_YEAR[args.timeframe]

    print("\nQuantForgeAI Out-of-Sample Evaluation")
    print("=" * 56)
    print(f"Symbol:                  {args.symbol}")
    print(f"Timeframe:               {args.timeframe}")
    print(f"Strategy:                SMA {args.fast}/{args.slow}")
    print(f"Total candles:           {len(strategy_data):,}")
    print(f"Training share:          {args.train_ratio:.0%}")
    print(f"Fee per side:            {args.fee:.2%}")
    print(f"Periods per year:        {periods_per_year:,}")

    print_metrics(
        "TRAINING PERIOD",
        train_result,
        train_data,
        periods_per_year,
    )
    print_metrics(
        "OUT-OF-SAMPLE TEST PERIOD",
        test_result,
        test_data,
        periods_per_year,
    )

    print("\nInterpretation")
    print("-" * 56)

    if test_result.total_return > test_result.buy_hold_return:
        print("The strategy outperformed buy-and-hold in the test period.")
    else:
        print("The strategy did not outperform buy-and-hold in the test period.")

    print(
        "This is one historical split, not proof of a persistent edge. "
        "Do not select parameters based on test-period results."
    )


if __name__ == "__main__":
    main()
