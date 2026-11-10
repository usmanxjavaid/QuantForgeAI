
import pandas as pd
import pytest

from backend.strategies import (
    MovingAverageStrategy,
    Strategy,
    moving_average_strategy,
)


def test_strategy_stays_in_cash_until_slow_average_is_available():
    candles = pd.DataFrame({
        "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })

    result = moving_average_strategy(
        candles,
        fast_period=2,
        slow_period=3,
    )

    assert result["target_position"].tolist() == [0, 0, 1, 1, 1, 1]
    assert result["fast_sma"].iloc[2] > result["slow_sma"].iloc[2]


def test_strategy_does_not_modify_input_data():
    candles = pd.DataFrame({
        "close": [10.0, 11.0, 12.0, 13.0, 14.0],
    })
    original = candles.copy(deep=True)

    moving_average_strategy(
        candles,
        fast_period=2,
        slow_period=3,
    )

    pd.testing.assert_frame_equal(candles, original)


def test_strategy_sorts_candles_by_timestamp():
    candles = pd.DataFrame({
        "timestamp": pd.to_datetime([
            "2026-01-01T02:00:00Z",
            "2026-01-01T00:00:00Z",
            "2026-01-01T01:00:00Z",
        ], utc=True),
        "close": [12.0, 10.0, 11.0],
    })

    result = moving_average_strategy(
        candles,
        fast_period=1,
        slow_period=2,
    )

    assert result["timestamp"].is_monotonic_increasing
    assert result["close"].tolist() == [10.0, 11.0, 12.0]


@pytest.mark.parametrize(
    ("fast_period", "slow_period", "expected_exception"),
    [
        (0, 5, ValueError),
        (2, 0, ValueError),
        (-1, 5, ValueError),
        (5, 5, ValueError),
        (6, 5, ValueError),
        (True, 5, TypeError),
        ("2", 5, TypeError),
        (2.5, 5, TypeError),
        (2, False, TypeError),
    ],
)
def test_strategy_rejects_invalid_periods(
    fast_period,
    slow_period,
    expected_exception,
):
    with pytest.raises(expected_exception):
        MovingAverageStrategy(
            fast_period=fast_period,
            slow_period=slow_period,
        )


def test_strategy_rejects_missing_close_column():
    candles = pd.DataFrame({"open": [10.0, 11.0, 12.0]})

    with pytest.raises(ValueError, match="close"):
        moving_average_strategy(candles)


def test_strategy_rejects_invalid_close_prices():
    candles = pd.DataFrame({"close": [10.0, 0.0, 12.0]})

    with pytest.raises(ValueError, match="greater than zero"):
        moving_average_strategy(
            candles,
            fast_period=1,
            slow_period=2,
        )


def test_moving_average_strategy_implements_strategy_protocol():
    strategy = MovingAverageStrategy(
        fast_period=2,
        slow_period=3,
    )

    assert isinstance(strategy, Strategy)


def test_class_generates_expected_signals():
    candles = pd.DataFrame({
        "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })

    strategy = MovingAverageStrategy(
        fast_period=2,
        slow_period=3,
    )

    result = strategy.generate_signals(candles)

    assert result["target_position"].tolist() == [0, 0, 1, 1, 1, 1]


def test_class_and_legacy_function_generate_identical_results():
    candles = pd.DataFrame({
        "close": [10.0, 11.0, 12.0, 11.0, 13.0, 15.0, 14.0],
    })

    strategy = MovingAverageStrategy(
        fast_period=2,
        slow_period=3,
    )

    class_result = strategy.generate_signals(candles)

    function_result = moving_average_strategy(
        candles,
        fast_period=2,
        slow_period=3,
    )

    pd.testing.assert_frame_equal(class_result, function_result)


def test_strategy_configuration_is_immutable():
    strategy = MovingAverageStrategy(
        fast_period=2,
        slow_period=3,
    )

    with pytest.raises(AttributeError):
        strategy.fast_period = 5


def test_strategy_class_does_not_modify_input_data():
    candles = pd.DataFrame({
        "close": [10.0, 11.0, 12.0, 13.0, 14.0],
    })
    original = candles.copy(deep=True)

    strategy = MovingAverageStrategy(
        fast_period=2,
        slow_period=3,
    )

    strategy.generate_signals(candles)

    pd.testing.assert_frame_equal(candles, original)


def test_strategy_uses_default_configuration():
    strategy = MovingAverageStrategy()

    assert strategy.fast_period == 20
    assert strategy.slow_period == 50
