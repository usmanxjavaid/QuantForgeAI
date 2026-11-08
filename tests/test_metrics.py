
import math

import numpy as np
import pandas as pd
import pytest

from backend.backtesting.metrics import (
    PerformanceMetrics,
    calculate_performance_metrics,
)


def make_candles(
    equity,
    benchmark_equity=None,
    trade_returns=None,
):
    if benchmark_equity is None:
        benchmark_equity = equity

    if trade_returns is None:
        trade_returns = [None] * len(equity)

    return pd.DataFrame(
        {
            "equity": equity,
            "benchmark_equity": benchmark_equity,
            "closed_trade_return": trade_returns,
        }
    )


def test_returns_performance_metrics_dataclass():
    metrics = PerformanceMetrics(
        total_return=0.1,
        annualized_return=0.1,
        buy_hold_return=0.05,
        annualized_volatility=0.2,
        sharpe_ratio=1.0,
        sortino_ratio=1.5,
        max_drawdown=-0.1,
        max_drawdown_duration=3,
        calmar_ratio=1.0,
        trade_count=2,
        win_rate=0.5,
        average_trade_return=0.02,
        profit_factor=1.5,
    )

    assert metrics.total_return == 0.1
    assert metrics.trade_count == 2


def test_calculates_total_and_buy_hold_return():
    candles = make_candles(
        equity=[10_000, 11_000],
        benchmark_equity=[10_000, 9_000],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
    )

    assert metrics.total_return == pytest.approx(0.10)
    assert metrics.buy_hold_return == pytest.approx(-0.10)


def test_calculates_maximum_drawdown_including_initial_capital():
    candles = make_candles(
        equity=[9_000, 8_000, 9_500, 10_500],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
    )

    assert metrics.max_drawdown == pytest.approx(-0.20)


def test_calculates_maximum_drawdown_duration():
    candles = make_candles(
        equity=[10_000, 9_000, 8_000, 9_000, 10_000, 10_500],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
    )

    assert metrics.max_drawdown_duration == 4


def test_calculates_annualized_return():
    candles = make_candles(
        equity=[10_100, 10_200, 10_300],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
        periods_per_year=3,
    )

    expected = (10_300 / 10_000) ** (3 / 3) - 1

    assert metrics.annualized_return == pytest.approx(expected)


def test_calculates_annualized_volatility():
    candles = make_candles(
        equity=[10_000, 10_100, 10_000],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
        periods_per_year=252,
    )

    returns = np.array([0.0, 0.01, 10_000 / 10_100 - 1])
    expected = np.std(returns, ddof=1) * math.sqrt(252)

    assert metrics.annualized_volatility == pytest.approx(expected)


def test_calculates_sharpe_ratio():
    candles = make_candles(
        equity=[10_000, 10_100, 10_050, 10_200],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
        periods_per_year=252,
    )

    returns = np.array(
        [
            0.0,
            10_100 / 10_000 - 1,
            10_050 / 10_100 - 1,
            10_200 / 10_050 - 1,
        ]
    )
    expected = (
        np.mean(returns)
        / np.std(returns, ddof=1)
        * math.sqrt(252)
    )

    assert metrics.sharpe_ratio == pytest.approx(expected)


def test_handles_zero_volatility_without_division_error():
    candles = make_candles(
        equity=[10_000, 10_000, 10_000],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
    )

    assert metrics.annualized_volatility == 0.0
    assert metrics.sharpe_ratio == 0.0
    assert metrics.sortino_ratio == 0.0


def test_calculates_completed_trade_statistics():
    candles = make_candles(
        equity=[10_000, 10_500, 10_200],
        trade_returns=[None, 0.05, -0.03],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
    )

    assert metrics.trade_count == 2
    assert metrics.win_rate == pytest.approx(0.5)
    assert metrics.average_trade_return == pytest.approx(0.01)
    assert metrics.profit_factor == pytest.approx(0.05 / 0.03)


def test_profit_factor_is_infinite_when_there_are_no_losses():
    candles = make_candles(
        equity=[10_000, 10_500, 11_000],
        trade_returns=[None, 0.05, 0.04],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
    )

    assert math.isinf(metrics.profit_factor)


def test_no_completed_trades_returns_zero_trade_statistics():
    candles = make_candles(
        equity=[10_000, 10_100, 10_200],
    )

    metrics = calculate_performance_metrics(
        candles,
        initial_capital=10_000,
    )

    assert metrics.trade_count == 0
    assert metrics.win_rate == 0.0
    assert metrics.average_trade_return == 0.0
    assert metrics.profit_factor == 0.0


@pytest.mark.parametrize(
    "candles, initial_capital, periods_per_year",
    [
        (pd.DataFrame(), 10_000, 365),
        (pd.DataFrame({"equity": [10_000]}), 10_000, 365),
        (make_candles([10_000]), 0, 365),
        (make_candles([10_000]), 10_000, 0),
    ],
)
def test_rejects_invalid_inputs(candles, initial_capital, periods_per_year):
    with pytest.raises(ValueError):
        calculate_performance_metrics(
            candles,
            initial_capital=initial_capital,
            periods_per_year=periods_per_year,
        )
