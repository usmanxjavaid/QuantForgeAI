from datetime import UTC, datetime

import pytest

from backend.data.models import Candle


def make_candle(timeframe: str) -> Candle:
    return Candle(
        timestamp=datetime(2026, 1, 1, tzinfo=UTC),
        exchange="BINANCE",
        market_type="spot",
        symbol="BTC/USDT",
        timeframe=timeframe,
        open=100,
        high=110,
        low=95,
        close=105,
        volume=10,
    )


@pytest.mark.parametrize("timeframe", ["2h", "6h", "1w", "30m"])
def test_candle_accepts_valid_fixed_timeframes(timeframe):
    candle = make_candle(timeframe)
    assert candle.timeframe == timeframe
    assert candle.exchange == "binance"


@pytest.mark.parametrize("timeframe", ["2H", "1M", "0h", "bad"])
def test_candle_rejects_invalid_timeframes(timeframe):
    with pytest.raises(ValueError):
        make_candle(timeframe)