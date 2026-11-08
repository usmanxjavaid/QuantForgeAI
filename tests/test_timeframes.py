import pytest

from backend.data.timeframes import periods_per_year, timeframe_to_seconds


@pytest.mark.parametrize(
    ("timeframe", "expected"),
    [
        ("1s", 1),
        ("5m", 300),
        ("15m", 900),
        ("1h", 3600),
        ("2h", 7200),
        ("4h", 14400),
        ("1d", 86400),
        ("1w", 604800),
    ],
)
def test_timeframe_to_seconds(timeframe, expected):
    assert timeframe_to_seconds(timeframe) == expected


@pytest.mark.parametrize("timeframe", ["", "0h", "-1h", "1M", "1y", "hour", "1.5h", "2H"])
def test_timeframe_to_seconds_rejects_invalid_or_variable_intervals(timeframe):
    with pytest.raises(ValueError):
        timeframe_to_seconds(timeframe)


def test_periods_per_year_is_derived_from_timeframe():
    assert periods_per_year("1h") == 365 * 24
    assert periods_per_year("2h") == 365 * 12
    assert periods_per_year("1d") == 365