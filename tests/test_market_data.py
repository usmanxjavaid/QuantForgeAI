from datetime import datetime, timezone

import pandas as pd
import pytest

from backend.data.models import Candle
from backend.data.storage import ParquetCandleStore


def make_candles() -> list[Candle]:
    return [
        Candle(
            timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            exchange="binance",
            market_type="spot",
            symbol="BTC/USDT",
            timeframe="1h",
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0,
            volume=123.45,
        ),
        Candle(
            timestamp=datetime(
                2026,
                1,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            exchange="binance",
            market_type="spot",
            symbol="BTC/USDT",
            timeframe="1h",
            open=105.0,
            high=115.0,
            low=100.0,
            close=112.0,
            volume=150.0,
        ),
    ]


def test_candle_requires_timezone_aware_timestamp() -> None:
    with pytest.raises(ValueError):
        Candle(
            timestamp=datetime(2026, 1, 1),
            exchange="binance",
            market_type="spot",
            symbol="BTC/USDT",
            timeframe="1h",
            open=100,
            high=110,
            low=95,
            close=105,
            volume=10,
        )


def test_candle_rejects_invalid_timeframe() -> None:
    with pytest.raises(ValueError):
        Candle(
            timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            exchange="binance",
            market_type="spot",
            symbol="BTC/USDT",
            timeframe="2h",
            open=100,
            high=110,
            low=95,
            close=105,
            volume=10,
        )


def test_candle_rejects_invalid_ohlc_relationship() -> None:
    with pytest.raises(ValueError):
        Candle(
            timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            exchange="binance",
            market_type="spot",
            symbol="BTC/USDT",
            timeframe="1h",
            open=100,
            high=90,
            low=95,
            close=98,
            volume=10,
        )


def test_candle_rejects_futures_style_symbol() -> None:
    with pytest.raises(ValueError):
        Candle(
            timestamp=datetime(
                2026,
                1,
                1,
                tzinfo=timezone.utc,
            ),
            exchange="binance",
            market_type="spot",
            symbol="BTC/USDT:USDT",
            timeframe="1h",
            open=100,
            high=110,
            low=95,
            close=105,
            volume=10,
        )


def test_parquet_round_trip(tmp_path) -> None:
    store = ParquetCandleStore(tmp_path)

    original = make_candles()

    path = store.save(original)

    assert path.exists()

    loaded = store.load(
        exchange="binance",
        market_type="spot",
        symbol="BTC/USDT",
        timeframe="1h",
    )

    assert len(loaded) == 2
    assert loaded[0].close == 105.0
    assert loaded[1].close == 112.0


def test_parquet_storage_deduplicates_candles(tmp_path) -> None:
    store = ParquetCandleStore(tmp_path)

    candles = make_candles()

    store.save(candles)
    store.save(candles)

    dataframe = pd.read_parquet(
        tmp_path
        / "binance"
        / "spot"
        / "BTC_USDT"
        / "1h.parquet"
    )

    assert len(dataframe) == 2