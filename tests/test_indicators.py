import numpy as np
import pandas as pd
import pytest

from backend.quantitative import (
    calculate_ema,
    calculate_indicators,
    calculate_returns,
    calculate_rsi,
    calculate_sma,
    calculate_volatility,
)


def test_returns_calculation() -> None:
    close = pd.Series([100.0, 110.0, 99.0])

    returns = calculate_returns(close)

    assert np.isnan(returns.iloc[0])
    assert returns.iloc[1] == pytest.approx(0.10)
    assert returns.iloc[2] == pytest.approx(-0.10)


def test_sma_waits_for_full_window() -> None:
    close = pd.Series([10.0, 20.0, 30.0, 40.0])

    sma = calculate_sma(close, period=3)

    assert sma.iloc[:2].isna().all()
    assert sma.iloc[2] == pytest.approx(20.0)
    assert sma.iloc[3] == pytest.approx(30.0)


def test_ema_waits_for_full_window() -> None:
    close = pd.Series([10.0, 20.0, 30.0, 40.0])

    ema = calculate_ema(close, period=3)

    assert ema.iloc[:2].isna().all()
    assert np.isfinite(ema.iloc[2])


def test_volatility_calculation() -> None:
    returns = pd.Series([0.01, 0.03, 0.02])

    volatility = calculate_volatility(returns, period=2)

    assert np.isnan(volatility.iloc[0])
    assert volatility.iloc[1] == pytest.approx(
        np.std([0.01, 0.03], ddof=1)
    )


def test_rsi_for_rising_prices() -> None:
    close = pd.Series([100.0, 101.0, 102.0, 103.0])

    rsi = calculate_rsi(close, period=2)

    assert rsi.iloc[2] == pytest.approx(100.0)
    assert rsi.iloc[3] == pytest.approx(100.0)


def test_rsi_for_flat_prices() -> None:
    close = pd.Series([100.0, 100.0, 100.0, 100.0])

    rsi = calculate_rsi(close, period=2)

    assert rsi.iloc[2] == pytest.approx(50.0)


def test_calculate_indicators_preserves_original_data() -> None:
    candles = pd.DataFrame(
        {
            "close": [100.0, 101.0, 102.0, 103.0],
        }
    )

    original = candles.copy()

    result = calculate_indicators(
        candles,
        sma_period=2,
        ema_period=2,
        volatility_period=2,
        rsi_period=2,
    )

    pd.testing.assert_frame_equal(candles, original)

    assert "return" in result.columns
    assert "sma_2" in result.columns
    assert "ema_2" in result.columns
    assert "volatility_2" in result.columns
    assert "rsi_2" in result.columns


def test_calculate_indicators_sorts_timestamps() -> None:
    candles = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2026-01-01T02:00:00Z",
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                ],
                utc=True,
            ),
            "close": [102.0, 100.0, 101.0],
        }
    )

    result = calculate_indicators(
        candles,
        sma_period=2,
        ema_period=2,
        volatility_period=2,
        rsi_period=2,
    )

    assert result["close"].tolist() == [100.0, 101.0, 102.0]


def test_calculate_indicators_rejects_invalid_close() -> None:
    candles = pd.DataFrame({"close": [100.0, 0.0, 102.0]})

    with pytest.raises(ValueError, match="greater than zero"):
        calculate_indicators(candles)


def test_calculate_indicators_rejects_missing_close() -> None:
    candles = pd.DataFrame({"open": [100.0, 101.0]})

    with pytest.raises(ValueError, match="close"):
        calculate_indicators(candles)


def test_indicators_reject_invalid_period() -> None:
    with pytest.raises(ValueError, match="positive integer"):
        calculate_sma(pd.Series([1.0, 2.0]), period=0)