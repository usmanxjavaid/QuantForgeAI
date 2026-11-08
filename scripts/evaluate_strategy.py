
import argparse
import math

import pandas as pd

from backend.backtesting import calculate_performance_metrics, run_backtest
from backend.data.storage import ParquetCandleStore
from backend.data.timeframes import (
    periods_per_year as timeframe_periods_per_year,
)
from backend.data.timeframes import (
    timeframe_to_seconds,
)
from backend.strategies import moving_average_strategy


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
    annualization_periods: int,
) -> None:
    """Print detailed performance metrics for one data segment."""
    first_timestamp = data["timestamp"].iloc[0]
    last_timestamp = data["timestamp"].iloc[-1]

    metrics = calculate_performance_metrics(
        candles=result.candles,
        initial_capital=result.initial_capital,
        periods_per_year=annualization_periods,
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
    print(
        f"Drawdown duration:       "
        f"{metrics.max_drawdown_duration:,} candles"
    )
    print(f"Annualized volatility:   {metrics.annualized_volatility:.2%}")
    print(f"Sharpe ratio:            {format_metric(metrics.sharpe_ratio)}")
    print(f"Sortino ratio:           {format_metric(metrics.sortino_ratio)}")
    print(f"Calmar ratio:            {format_metric(metrics.calmar_ratio)}")

    print("\nTrade Statistics")
    print("-" * 56)
    print(f"Completed trades:        {metrics.trade_count}")
    print(f"Win rate:                {metrics.win_rate:.2%}")
    print(f"Average trade return:    {metrics.average_trade_return:.2%}")
    print(f"Profit factor:           {format_metric(metrics.profit_factor)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a moving-average strategy on chronological "
            "training and out-of-sample test periods."
        )
    )

    parser.add_argument(
        "--exchange",
        default="binance",
        help="Exchange ID used in stored market data.",
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Trading pair, for example BTC/USDT.",
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Candle interval, for example 15m, 2h or 1d.",
    )
    parser.add_argument("--fast", type=int, default=20)
    parser.add_argument("--slow", type=int, default=50)
    parser.add_argument("--capital", type=float, default=10_000.0)
    parser.add_argument("--fee", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.70)

    args = parser.parse_args()

    try:
        timeframe_to_seconds(args.timeframe)
    except ValueError as exc:
        parser.error(str(exc))

    if args.fast < 1 or args.slow < 1 or args.fast >= args.slow:
        parser.error(
            "--fast and --slow must be positive, "
            "with --fast smaller than --slow."
        )

    if not 0 < args.train_ratio < 1:
        parser.error("--train-ratio must be greater than 0 and less than 1.")

    if not math.isfinite(args.capital) or args.capital <= 0:
        parser.error("--capital must be finite and greater than zero.")

    if not math.isfinite(args.fee) or not 0 <= args.fee < 1:
        parser.error("--fee must be finite and between 0 inclusive and 1 exclusive.")

    candles = ParquetCandleStore().load(
        exchange=args.exchange,
        market_type="spot",
        symbol=args.symbol,
        timeframe=args.timeframe,
    )

    if not candles:
        parser.error(
            f"No stored candles found for {args.exchange} "
            f"{args.symbol} {args.timeframe}."
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

    minimum_candles = args.slow * 2
    if len(dataframe) < minimum_candles:
        parser.error(
            f"Need at least {minimum_candles} candles for this evaluation; "
            f"found {len(dataframe)}."
        )

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

    annualization_periods = timeframe_periods_per_year(args.timeframe)

    print("\nQuantForgeAI Out-of-Sample Evaluation")
    print("=" * 56)
    print(f"Exchange:                {args.exchange}")
    print(f"Symbol:                  {args.symbol}")
    print(f"Timeframe:               {args.timeframe}")
    print(f"Strategy:                SMA {args.fast}/{args.slow}")
    print(f"Total candles:           {len(strategy_data):,}")
    print(f"Training share:          {args.train_ratio:.0%}")
    print(f"Fee per side:            {args.fee:.2%}")
    print(f"Periods per year:        {annualization_periods:,}")

    print_metrics(
        "TRAINING PERIOD",
        train_result,
        train_data,
        annualization_periods,
    )
    print_metrics(
        "OUT-OF-SAMPLE TEST PERIOD",
        test_result,
        test_data,
        annualization_periods,
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
