
from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class Strategy(Protocol):
    """Interface that all trading strategies must implement."""

    def generate_signals(
        self,
        candles: pd.DataFrame,
    ) -> pd.DataFrame:
        """Return candle data with a target_position column."""
        ...
