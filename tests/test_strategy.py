
import pandas as pd
import pytest

from backend.strategies import moving_average_strategy


def test_strategy_stays_in_cash_until_slow_average_is_available():
    candles = pd.DataFrame({
        "close": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
    })

    result = moving_average_strategy(
        candles,
        fast_period=2,
        slow_period=3,
    )

    # The slow average needs three candles before it can be calculated.
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
    ("fast_period", "slow_period"),
    [
        (0, 5),
        (2, 0),
        (-1, 5),
        (5, 5),
        (6, 5),
        (True, 5),
    ],
)
def test_strategy_rejects_invalid_periods(fast_period, slow_period):
    candles = pd.DataFrame({"close": [10.0, 11.0, 12.0]})

    with pytest.raises(ValueError):
        moving_average_strategy(
            candles,
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
