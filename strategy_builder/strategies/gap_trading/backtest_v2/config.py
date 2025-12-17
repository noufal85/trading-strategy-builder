"""Backtest Configuration Module.

Defines all configurable parameters for gap trading backtests.
"""

from dataclasses import dataclass, field
from datetime import date, time
from typing import List, Optional
from enum import Enum


class StopLossType(str, Enum):
    """Type of stop-loss calculation."""
    ATR = 'atr'
    FIXED_PCT = 'fixed_pct'


class ExitStrategy(str, Enum):
    """Exit strategy type."""
    EOD_ONLY = 'eod_only'  # Only exit at EOD
    STOP_OR_EOD = 'stop_or_eod'  # Exit on stop-loss or EOD
    TRAILING = 'trailing'  # Trailing stop


@dataclass
class BacktestConfig:
    """Configuration for gap trading backtest.

    Attributes:
        start_date: Backtest start date
        end_date: Backtest end date
        symbols: List of symbols to backtest (None = use all from universe)

        initial_capital: Starting capital
        risk_per_trade_pct: Risk per trade as percentage of capital
        max_position_pct: Maximum position size as percentage of capital

        min_gap_pct: Minimum gap threshold to consider
        max_gap_pct: Maximum gap threshold (filter extreme gaps)
        confirmation_minutes: Minutes after open to confirm gap (default 10)

        stop_loss_type: Type of stop-loss calculation
        stop_atr_multiplier: ATR multiplier for stops (if using ATR stops)
        stop_fixed_pct: Fixed percentage for stops (if using fixed stops)

        use_risk_tiers: Whether to apply risk tier position sizing

        max_positions: Maximum concurrent positions per day
        max_long_positions: Maximum long positions
        max_short_positions: Maximum short positions

        slippage_pct: Slippage percentage for execution simulation
        commission: Commission per trade

        eod_close_time: Time to close all positions (ET)

        use_minute_data: Whether to use minute data for accurate pricing
    """
    # Date range
    start_date: date
    end_date: date
    symbols: Optional[List[str]] = None

    # Capital settings
    initial_capital: float = 100000.0
    risk_per_trade_pct: float = 1.0
    max_position_pct: float = 5.0

    # Gap detection parameters
    min_gap_pct: float = 1.5
    max_gap_pct: float = 10.0
    confirmation_minutes: int = 10  # Minutes after market open

    # Stop-loss parameters
    stop_loss_type: StopLossType = StopLossType.ATR
    stop_atr_multiplier: float = 1.5
    stop_fixed_pct: float = 2.0

    # Risk tier settings
    use_risk_tiers: bool = True

    # Position limits
    max_positions: int = 5
    max_long_positions: int = 5
    max_short_positions: int = 3
    max_daily_trades: int = 10

    # Execution simulation
    slippage_pct: float = 0.05
    commission: float = 0.0

    # Timing
    market_open: time = field(default_factory=lambda: time(9, 30))
    eod_close_time: time = field(default_factory=lambda: time(15, 55))

    # Data options
    use_minute_data: bool = True

    def __post_init__(self):
        """Validate configuration."""
        if self.start_date >= self.end_date:
            raise ValueError("start_date must be before end_date")

        if self.initial_capital <= 0:
            raise ValueError("initial_capital must be positive")

        if not 0 < self.risk_per_trade_pct <= 10:
            raise ValueError("risk_per_trade_pct should be between 0 and 10")

        if self.min_gap_pct >= self.max_gap_pct:
            raise ValueError("min_gap_pct must be less than max_gap_pct")

        if self.confirmation_minutes < 0 or self.confirmation_minutes > 60:
            raise ValueError("confirmation_minutes should be between 0 and 60")

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'symbols': self.symbols,
            'initial_capital': self.initial_capital,
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'max_position_pct': self.max_position_pct,
            'min_gap_pct': self.min_gap_pct,
            'max_gap_pct': self.max_gap_pct,
            'confirmation_minutes': self.confirmation_minutes,
            'stop_loss_type': self.stop_loss_type.value,
            'stop_atr_multiplier': self.stop_atr_multiplier,
            'stop_fixed_pct': self.stop_fixed_pct,
            'use_risk_tiers': self.use_risk_tiers,
            'max_positions': self.max_positions,
            'slippage_pct': self.slippage_pct,
            'commission': self.commission,
            'use_minute_data': self.use_minute_data,
        }
