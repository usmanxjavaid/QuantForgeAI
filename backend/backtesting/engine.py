
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestResult:
    initial_capital: float
    final_equity: float
    total_return: float
    buy_hold_return: float
    max_drawdown: float
    trade_count: int
    win_rate: float
    candles: pd.DataFrame


def run_backtest(
    data: pd.DataFrame,
    initial_capital: float = 10_000.0,
    fee_rate: float = 0.001,
) -> BacktestResult:
    """Backtest a long-or-cash spot strategy with proportional fees.

    A signal calculated at candle t's close is executed at candle t+1's
    open. Positions are marked to market at each candle's close.

    The portfolio uses fractional BTC units and invests all available cash
    when entering a position. Buys and sells both incur fees.

    The buy-and-hold benchmark buys at the first candle's open and values
    the resulting position at the final candle's close.
    """
    required = {"open", "close", "target_position"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if data.empty:
        raise ValueError("Cannot backtest empty data")

    if (
        isinstance(initial_capital, bool)
        or not np.isscalar(initial_capital)
        or not np.isfinite(initial_capital)
        or initial_capital <= 0
    ):
        raise ValueError("initial_capital must be finite and greater than zero")

    if (
        isinstance(fee_rate, bool)
        or not np.isscalar(fee_rate)
        or not np.isfinite(fee_rate)
        or not 0 <= fee_rate < 1
    ):
        raise ValueError("fee_rate must be between zero inclusive and one exclusive")

    result = data.copy()

    if "timestamp" in result.columns:
        result = result.sort_values("timestamp").reset_index(drop=True)
    else:
        result = result.reset_index(drop=True)

    for column in ("open", "close", "target_position"):
        result[column] = pd.to_numeric(result[column], errors="raise")

    opens = result["open"].to_numpy(dtype=float)
    closes = result["close"].to_numpy(dtype=float)
    targets = result["target_position"].to_numpy(dtype=float)

    if (
        not np.isfinite(opens).all()
        or not np.isfinite(closes).all()
        or (opens <= 0).any()
        or (closes <= 0).any()
    ):
        raise ValueError("Open and close prices must be finite and greater than zero")

    if not np.isfinite(targets).all() or not np.isin(targets, [0, 1]).all():
        raise ValueError("target_position must contain only 0 (cash) or 1 (long)")

    count = len(result)
    cash = float(initial_capital)
    units = 0.0
    in_trade = False
    entry_cost = 0.0
    closed_trade_returns: list[float] = []

    equity_values: list[float] = []
    position_values: list[int] = []
    fee_values: list[float] = []
    trade_events: list[str] = []

    for i in range(count):
        # A signal from the previous candle is actionable at this open.
        desired_position = int(targets[i - 1]) if i > 0 else 0
        execution_price = opens[i]
        fees_paid = 0.0
        event = "HOLD"

        currently_long = units > 0

        if desired_position == 1 and not currently_long:
            # Spend all available cash, including the entry fee.
            units = cash / (execution_price * (1.0 + fee_rate))
            fees_paid = units * execution_price * fee_rate
            entry_cost = units * execution_price + fees_paid
            cash = 0.0
            in_trade = True
            event = "BUY"

        elif desired_position == 0 and currently_long:
            proceeds_before_fee = units * execution_price
            fees_paid = proceeds_before_fee * fee_rate
            cash = proceeds_before_fee - fees_paid

            trade_return = cash / entry_cost - 1.0
            closed_trade_returns.append(float(trade_return))

            units = 0.0
            entry_cost = 0.0
            in_trade = False
            event = "SELL"

        # Mark the portfolio to market at this candle's close.
        equity = cash + units * closes[i]

        equity_values.append(float(equity))
        position_values.append(int(units > 0))
        fee_values.append(float(fees_paid))
        trade_events.append(event)

    result["position"] = position_values
    result["fees"] = fee_values
    result["trade_event"] = trade_events
    result["equity"] = equity_values

    previous_equity = result["equity"].shift(1)
    previous_equity.iloc[0] = initial_capital
    result["strategy_return"] = result["equity"] / previous_equity - 1.0

    result["market_return"] = result["close"].pct_change(fill_method=None).fillna(0.0)
    result["equity_peak"] = result["equity"].cummax()
    result["drawdown"] = result["equity"] / result["equity_peak"] - 1.0

    final_equity = float(result["equity"].iloc[-1])
    total_return = final_equity / initial_capital - 1.0
    max_drawdown = float(result["drawdown"].min())

    # Buy-and-hold benchmark: buy at the first open, pay the entry fee,
    # and value the remaining BTC at the last close.
    benchmark_units = initial_capital / (opens[0] * (1.0 + fee_rate))
    benchmark_final_equity = benchmark_units * closes[-1]
    buy_hold_return = benchmark_final_equity / initial_capital - 1.0

    trade_count = len(closed_trade_returns)
    win_rate = (
        sum(trade_return > 0 for trade_return in closed_trade_returns)
        / trade_count
        if trade_count
        else 0.0
    )

    # An open position at the end remains in final equity but is not
    # counted as a completed trade for win-rate purposes.
    _ = in_trade

    return BacktestResult(
        initial_capital=float(initial_capital),
        final_equity=final_equity,
        total_return=float(total_return),
        buy_hold_return=float(buy_hold_return),
        max_drawdown=max_drawdown,
        trade_count=trade_count,
        win_rate=float(win_rate),
        candles=result,
    )
