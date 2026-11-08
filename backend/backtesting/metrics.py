
"""Performance metrics for QuantForgeAI backtests."""

import math
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PerformanceMetrics:
    """Summary of strategy performance and risk."""

    total_return: float
    annualized_return: float
    buy_hold_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    calmar_ratio: float
    trade_count: int
    win_rate: float
    average_trade_return: float
    profit_factor: float


def _annualize_return(
    final_equity: float,
    initial_capital: float,
    periods: int,
    periods_per_year: int,
) -> float:
    """Annualize a return without causing floating-point overflow warnings."""
    if periods <= 0:
        return 0.0

    growth = final_equity / initial_capital

    if growth <= 0:
        return -1.0

    log_annualized_growth = (
        math.log(growth) * periods_per_year / periods
    )

    if log_annualized_growth > math.log(sys.float_info.max):
        return math.inf

    return float(math.expm1(log_annualized_growth))


def calculate_performance_metrics(
    candles: pd.DataFrame,
    initial_capital: float,
    periods_per_year: int = 24 * 365,
    risk_free_rate: float = 0.0,
) -> PerformanceMetrics:
    """Calculate performance metrics from a backtest's candle results.

    Args:
        candles: BacktestResult.candles DataFrame.
        initial_capital: Starting portfolio value.
        periods_per_year: Number of candles in a year. For hourly crypto
            data, use 24 * 365; for daily data, use 365.
        risk_free_rate: Annual risk-free rate as a decimal.

    Returns:
        A PerformanceMetrics instance. Ratios that cannot be meaningfully
        calculated are returned as 0.0. Profit factor is infinity when
        there are winning trades but no losing trades, and 0.0 when
        there are no completed trades.
    """
    required = {
        "equity",
        "benchmark_equity",
        "closed_trade_return",
    }
    missing = required - set(candles.columns)

    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if candles.empty:
        raise ValueError("Cannot calculate metrics from empty data")

    if (
        isinstance(initial_capital, bool)
        or not np.isscalar(initial_capital)
        or not np.isfinite(initial_capital)
        or initial_capital <= 0
    ):
        raise ValueError(
            "initial_capital must be finite and greater than zero"
        )

    if (
        isinstance(periods_per_year, bool)
        or not isinstance(periods_per_year, int)
        or periods_per_year <= 0
    ):
        raise ValueError(
            "periods_per_year must be a positive integer"
        )

    if (
        isinstance(risk_free_rate, bool)
        or not np.isscalar(risk_free_rate)
        or not np.isfinite(risk_free_rate)
        or risk_free_rate <= -1
    ):
        raise ValueError(
            "risk_free_rate must be finite and greater than -1"
        )

    equity = pd.to_numeric(
        candles["equity"], errors="raise"
    ).to_numpy(dtype=float)

    benchmark_equity = pd.to_numeric(
        candles["benchmark_equity"], errors="raise"
    ).to_numpy(dtype=float)

    if (
        not np.isfinite(equity).all()
        or not np.isfinite(benchmark_equity).all()
        or (equity <= 0).any()
        or (benchmark_equity <= 0).any()
    ):
        raise ValueError(
            "Equity values must be finite and greater than zero"
        )

    # Returns relative to the initial capital and the benchmark.
    total_return = float(
        equity[-1] / initial_capital - 1.0
    )

    buy_hold_return = float(
        benchmark_equity[-1] / initial_capital - 1.0
    )

    # Include initial capital in the equity curve. This ensures that
    # the first candle's return and drawdown are not omitted.
    equity_with_initial = np.concatenate(
        ([float(initial_capital)], equity)
    )

    peaks = np.maximum.accumulate(equity_with_initial)
    drawdowns = equity_with_initial / peaks - 1.0

    max_drawdown = float(drawdowns.min())

    # Drawdown duration counts candles from the peak through the
    # recovery candle. An unrecovered drawdown extends to the last
    # available candle.
    current_duration = 0
    longest_duration = 0

    for value, peak in zip(equity, peaks[1:]):
        if value < peak:
            current_duration += 1
            longest_duration = max(
                longest_duration,
                current_duration,
            )
        elif current_duration > 0:
            recovery_duration = current_duration + 1
            longest_duration = max(
                longest_duration,
                recovery_duration,
            )
            current_duration = 0

    number_of_periods = len(equity)

    annualized_return = _annualize_return(
        final_equity=float(equity[-1]),
        initial_capital=float(initial_capital),
        periods=number_of_periods,
        periods_per_year=periods_per_year,
    )

    # Calculate candle returns including the initial-capital-to-first-
    # candle interval. The first return is zero when the first candle
    # closes at initial capital, as in the current backtest engine.
    period_returns = (
        pd.Series(equity_with_initial)
        .pct_change(fill_method=None)
        .dropna()
        .to_numpy(dtype=float)
    )

    if not np.isfinite(period_returns).all():
        raise ValueError("Candle returns must be finite")

    if len(period_returns) > 1:
        period_volatility = float(
            np.std(period_returns, ddof=1)
        )
    else:
        period_volatility = 0.0

    annualized_volatility = (
        period_volatility * math.sqrt(periods_per_year)
    )

    # Convert the annual risk-free rate into a per-period rate.
    risk_free_per_period = (
        (1.0 + risk_free_rate)
        ** (1.0 / periods_per_year)
        - 1.0
    )

    excess_returns = period_returns - risk_free_per_period

    sharpe_ratio = 0.0
    sortino_ratio = 0.0

    if period_volatility > 0:
        sharpe_ratio = float(
            np.mean(excess_returns)
            / period_volatility
            * math.sqrt(periods_per_year)
        )

    # Downside deviation uses all periods, with positive excess returns
    # contributing zero downside.
    downside_returns = np.minimum(excess_returns, 0.0)

    downside_deviation = float(
        np.sqrt(np.mean(downside_returns**2))
    )

    if downside_deviation > 0:
        sortino_ratio = float(
            np.mean(excess_returns)
            / downside_deviation
            * math.sqrt(periods_per_year)
        )

    calmar_ratio = (
        float(annualized_return / abs(max_drawdown))
        if max_drawdown < 0
        else 0.0
    )

    # Completed trades only. Open positions are not included.
    trade_returns = pd.to_numeric(
        candles["closed_trade_return"],
        errors="coerce",
    ).dropna().to_numpy(dtype=float)

    if not np.isfinite(trade_returns).all():
        raise ValueError(
            "Completed trade returns must be finite"
        )

    trade_count = len(trade_returns)

    win_rate = (
        float(np.mean(trade_returns > 0))
        if trade_count
        else 0.0
    )

    average_trade_return = (
        float(np.mean(trade_returns))
        if trade_count
        else 0.0
    )

    gross_profit = float(
        trade_returns[trade_returns > 0].sum()
    )

    gross_loss = float(
        -trade_returns[trade_returns < 0].sum()
    )

    if gross_loss > 0:
        profit_factor = gross_profit / gross_loss
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = 0.0

    return PerformanceMetrics(
        total_return=total_return,
        annualized_return=annualized_return,
        buy_hold_return=buy_hold_return,
        annualized_volatility=float(annualized_volatility),
        sharpe_ratio=float(sharpe_ratio),
        sortino_ratio=float(sortino_ratio),
        max_drawdown=max_drawdown,
        max_drawdown_duration=longest_duration,
        calmar_ratio=float(calmar_ratio),
        trade_count=trade_count,
        win_rate=win_rate,
        average_trade_return=average_trade_return,
        profit_factor=float(profit_factor),
    )
