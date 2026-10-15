
import pandas as pd
import pytest

from backend.backtesting import run_backtest


def test_backtest_stays_in_cash_when_target_is_zero():
    data = pd.DataFrame({
        "open": [100.0, 110.0, 90.0],
        "close": [100.0, 110.0, 90.0],
        "target_position": [0, 0, 0],
    })

    result = run_backtest(data, initial_capital=1_000.0)

    assert result.final_equity == pytest.approx(1_000.0)
    assert result.total_return == pytest.approx(0.0)
    assert result.trade_count == 0
    assert result.win_rate == 0.0


def test_signal_executes_at_next_candle_open():
    data = pd.DataFrame({
        "open": [100.0, 100.0, 110.0],
        "close": [100.0, 110.0, 121.0],
        "target_position": [1, 1, 1],
    })

    result = run_backtest(
        data,
        initial_capital=1_000.0,
        fee_rate=0.0,
    )

    # The first signal executes at the second candle's open (100).
    # The portfolio then grows from 1,000 to 1,100 and then 1,210.
    assert result.final_equity == pytest.approx(1_210.0)
    assert result.candles["trade_event"].tolist() == ["HOLD", "BUY", "HOLD"]


def test_entry_and_exit_fees_are_charged():
    data = pd.DataFrame({
        "open": [100.0, 100.0, 110.0, 110.0],
        "close": [100.0, 100.0, 110.0, 110.0],
        "target_position": [0, 1, 0, 0],
    })

    result = run_backtest(
        data,
        initial_capital=1_000.0,
        fee_rate=0.01,
    )

    expected_fee = (1_000.0 / 111.1) * 110.0 * 0.01

    assert result.candles["trade_event"].tolist() == [
        "HOLD", "HOLD", "BUY", "SELL"
    ]
    assert result.candles["fees"].tolist() == pytest.approx(
        [0.0, 0.0, expected_fee, expected_fee]
    )
    assert result.trade_count == 1
    assert result.win_rate == 0.0
    assert result.final_equity < 1_000.0


def test_completed_trade_statistics_match_actual_fees():
    data = pd.DataFrame({
        "open": [100.0, 100.0, 110.0, 120.0],
        "close": [100.0, 100.0, 110.0, 120.0],
        "target_position": [0, 1, 0, 0],
    })

    result = run_backtest(
        data,
        initial_capital=1_000.0,
        fee_rate=0.001,
    )

    # Buy at 110, sell at 120, with fees charged at both prices.
    expected_return = (
        (120.0 * (1.0 - 0.001))
        / (110.0 * (1.0 + 0.001))
        - 1.0
    )

    assert result.trade_count == 1
    assert result.win_rate == 1.0
    assert result.final_equity == pytest.approx(
        1_000.0 * (1.0 + expected_return)
    )


def test_open_trade_is_marked_to_market_but_not_counted_as_closed():
    data = pd.DataFrame({
        "open": [100.0, 100.0, 105.0],
        "close": [100.0, 105.0, 110.0],
        "target_position": [1, 1, 1],
    })

    result = run_backtest(data, initial_capital=1_000.0, fee_rate=0.0)

    assert result.final_equity == pytest.approx(1_100.0)
    assert result.trade_count == 0
    assert result.win_rate == 0.0


def test_backtest_sorts_timestamps_before_calculating_returns():
    data = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01T02:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
        ], utc=True),
        "open": [110.0, 100.0, 100.0],
        "close": [121.0, 100.0, 110.0],
        "target_position": [1, 1, 1],
    })

    result = run_backtest(data, initial_capital=1_000.0, fee_rate=0.0)

    assert result.candles["timestamp"].is_monotonic_increasing
    assert result.final_equity == pytest.approx(1_210.0)


def test_backtest_rejects_invalid_positions():
    data = pd.DataFrame({
        "open": [100.0, 101.0],
        "close": [100.0, 101.0],
        "target_position": [0, 2],
    })

    with pytest.raises(ValueError, match="target_position"):
        run_backtest(data)


def test_backtest_rejects_invalid_fee():
    data = pd.DataFrame({
        "open": [100.0, 101.0],
        "close": [100.0, 101.0],
        "target_position": [0, 1],
    })

    with pytest.raises(ValueError, match="fee_rate"):
        run_backtest(data, fee_rate=1.0)


def test_backtest_requires_open_prices():
    data = pd.DataFrame({
        "close": [100.0, 101.0],
        "target_position": [0, 1],
    })

    with pytest.raises(ValueError, match="open"):
        run_backtest(data)


def test_backtest_rejects_invalid_prices():
    data = pd.DataFrame({
        "open": [100.0, 0.0],
        "close": [100.0, 101.0],
        "target_position": [0, 1],
    })

    with pytest.raises(ValueError, match="greater than zero"):
        run_backtest(data)
