"""Trade Simulator for Gap Trading Backtest.

Simulates trade execution with realistic slippage, commissions,
and intraday stop-loss monitoring using minute data.
"""

import logging
from dataclasses import dataclass
from datetime import date, time, datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

from .config import BacktestConfig, StopLossType
from .data_loader import GapBacktestDataLoader
from .signal_engine import GapSignal, SignalType, RiskTier

logger = logging.getLogger(__name__)


class ExitReason(str, Enum):
    """Reason for trade exit."""
    STOP_LOSS = 'STOP_LOSS'
    EOD_CLOSE = 'EOD_CLOSE'
    TARGET = 'TARGET'  # Future: profit target
    MANUAL = 'MANUAL'


@dataclass
class SimulatedTrade:
    """Record of a simulated trade.

    Attributes:
        symbol: Stock ticker
        trade_date: Date of trade
        direction: LONG or SHORT
        entry_price: Entry price (including slippage)
        entry_time: Time of entry
        exit_price: Exit price (including slippage)
        exit_time: Time of exit
        shares: Number of shares
        stop_price: Stop-loss price
        exit_reason: Why the trade was closed
        gap_pct: Gap percentage that triggered entry
        atr: ATR at entry
        risk_tier: Risk tier classification
        pnl: Profit/loss in dollars
        pnl_pct: Return percentage
        commission: Total commission (entry + exit)
        hold_duration_minutes: How long position was held
        signal: Original signal that generated this trade
    """
    symbol: str
    trade_date: date
    direction: str  # 'LONG' or 'SHORT'
    entry_price: float
    entry_time: time
    exit_price: float
    exit_time: time
    shares: int
    stop_price: float
    exit_reason: ExitReason
    gap_pct: float
    atr: float
    risk_tier: str
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    hold_duration_minutes: int = 0
    signal: Optional[GapSignal] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'trade_date': self.trade_date.isoformat(),
            'direction': self.direction,
            'entry_price': round(self.entry_price, 2),
            'entry_time': self.entry_time.isoformat(),
            'exit_price': round(self.exit_price, 2),
            'exit_time': self.exit_time.isoformat(),
            'shares': self.shares,
            'stop_price': round(self.stop_price, 2),
            'exit_reason': self.exit_reason.value,
            'gap_pct': round(self.gap_pct, 2),
            'atr': round(self.atr, 4),
            'risk_tier': self.risk_tier,
            'pnl': round(self.pnl, 2),
            'pnl_pct': round(self.pnl_pct, 2),
            'commission': round(self.commission, 2),
            'hold_duration_minutes': self.hold_duration_minutes,
        }


class PositionSizer:
    """Calculate position sizes for trades.

    Uses ATR-based sizing with risk tier adjustments.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize position sizer.

        Args:
            config: Backtest configuration
        """
        self.config = config

        # Risk tier position multipliers
        self.tier_multipliers = {
            RiskTier.LOW: 1.0,
            RiskTier.MEDIUM: 0.75,
            RiskTier.HIGH: 0.5
        }

    def calculate_shares(
        self,
        signal: GapSignal,
        account_value: float,
        entry_price: float
    ) -> int:
        """Calculate number of shares to trade.

        Args:
            signal: Gap signal with ATR and risk tier
            account_value: Current account value
            entry_price: Entry price

        Returns:
            Number of shares to trade
        """
        # Calculate risk amount
        risk_amount = account_value * (self.config.risk_per_trade_pct / 100)

        # Calculate risk per share (distance to stop)
        if not signal.stop_price or not signal.atr:
            return 0

        risk_per_share = abs(entry_price - signal.stop_price)
        if risk_per_share <= 0:
            return 0

        # Base shares from risk calculation
        base_shares = risk_amount / risk_per_share

        # Apply risk tier multiplier
        if self.config.use_risk_tiers and signal.risk_tier:
            tier_mult = self.tier_multipliers.get(signal.risk_tier, 0.75)
            base_shares *= tier_mult

        # Apply maximum position size constraint
        max_position_value = account_value * (self.config.max_position_pct / 100)
        max_shares = max_position_value / entry_price

        # Final shares
        shares = int(min(base_shares, max_shares))

        return max(shares, 0)


class TradeSimulator:
    """Simulate trade execution with realistic conditions.

    Handles:
    - Entry with slippage
    - Intraday stop-loss monitoring using minute data
    - EOD close simulation
    - Commission calculation

    Attributes:
        config: Backtest configuration
        data_loader: Data loader for price data
        position_sizer: Position sizing calculator
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_loader: GapBacktestDataLoader
    ):
        """Initialize trade simulator.

        Args:
            config: Backtest configuration
            data_loader: Data loader instance
        """
        self.config = config
        self.data_loader = data_loader
        self.position_sizer = PositionSizer(config)

        logger.info(
            f"TradeSimulator initialized: slippage={config.slippage_pct}%, "
            f"commission=${config.commission}"
        )

    def simulate_trade(
        self,
        signal: GapSignal,
        account_value: float
    ) -> Optional[SimulatedTrade]:
        """Simulate a complete trade from entry to exit.

        Args:
            signal: Gap signal to trade
            account_value: Current account value for position sizing

        Returns:
            SimulatedTrade with full execution details, or None if trade failed
        """
        if signal.signal_type == SignalType.NO_TRADE:
            return None

        # Determine entry price (confirmation price with slippage)
        entry_price = signal.confirmation_price or signal.open_price
        if signal.signal_type == SignalType.BUY:
            entry_price *= (1 + self.config.slippage_pct / 100)  # Pay more for longs
        else:
            entry_price *= (1 - self.config.slippage_pct / 100)  # Receive less for shorts

        # Calculate position size
        shares = self.position_sizer.calculate_shares(signal, account_value, entry_price)
        if shares <= 0:
            logger.debug(f"Position size 0 for {signal.symbol}, skipping")
            return None

        # Entry time
        entry_time = signal.confirmation_time or self.config.market_open

        # Simulate exit
        exit_result = self._simulate_exit(signal, entry_price, entry_time)
        exit_price, exit_time, exit_reason = exit_result

        # Apply exit slippage
        if signal.signal_type == SignalType.BUY:
            exit_price *= (1 - self.config.slippage_pct / 100)  # Receive less
        else:
            exit_price *= (1 + self.config.slippage_pct / 100)  # Pay more to cover

        # Calculate P&L
        if signal.signal_type == SignalType.BUY:
            pnl = (exit_price - entry_price) * shares
        else:
            pnl = (entry_price - exit_price) * shares

        # Subtract commission
        commission = self.config.commission * 2  # Entry + exit
        pnl -= commission

        # Calculate percentage return
        position_value = entry_price * shares
        pnl_pct = (pnl / position_value) * 100 if position_value > 0 else 0

        # Calculate hold duration
        entry_dt = datetime.combine(signal.trade_date, entry_time)
        exit_dt = datetime.combine(signal.trade_date, exit_time)
        hold_duration = int((exit_dt - entry_dt).total_seconds() / 60)

        direction = 'LONG' if signal.signal_type == SignalType.BUY else 'SHORT'

        return SimulatedTrade(
            symbol=signal.symbol,
            trade_date=signal.trade_date,
            direction=direction,
            entry_price=entry_price,
            entry_time=entry_time,
            exit_price=exit_price,
            exit_time=exit_time,
            shares=shares,
            stop_price=signal.stop_price,
            exit_reason=exit_reason,
            gap_pct=signal.gap_pct,
            atr=signal.atr,
            risk_tier=signal.risk_tier.value if signal.risk_tier else 'MEDIUM',
            pnl=pnl,
            pnl_pct=pnl_pct,
            commission=commission,
            hold_duration_minutes=hold_duration,
            signal=signal
        )

    def _simulate_exit(
        self,
        signal: GapSignal,
        entry_price: float,
        entry_time: time
    ) -> tuple:
        """Simulate trade exit using intraday data.

        Checks if stop-loss was hit during the trading day,
        otherwise exits at EOD.

        Args:
            signal: Trade signal
            entry_price: Entry price
            entry_time: Entry time

        Returns:
            Tuple of (exit_price, exit_time, exit_reason)
        """
        is_long = signal.signal_type == SignalType.BUY
        stop_price = signal.stop_price

        # Try to use minute data for accurate stop simulation
        if self.config.use_minute_data:
            minute_df = self.data_loader.get_minute_bars(signal.symbol, signal.trade_date)

            if not minute_df.empty:
                # Check each minute bar after entry
                entry_dt = datetime.combine(signal.trade_date, entry_time)
                eod_dt = datetime.combine(signal.trade_date, self.config.eod_close_time)

                for _, bar in minute_df.iterrows():
                    bar_time = bar['timestamp']

                    # Skip bars before entry
                    if bar_time <= entry_dt:
                        continue

                    # Stop checking at EOD
                    if bar_time >= eod_dt:
                        break

                    # Check if stop was hit
                    if is_long and bar['low'] <= stop_price:
                        return (stop_price, bar_time.time(), ExitReason.STOP_LOSS)
                    elif not is_long and bar['high'] >= stop_price:
                        return (stop_price, bar_time.time(), ExitReason.STOP_LOSS)

                # No stop hit, get EOD price
                eod_bars = minute_df[minute_df['timestamp'].dt.time <= self.config.eod_close_time]
                if not eod_bars.empty:
                    eod_price = eod_bars.iloc[-1]['close']
                    return (eod_price, self.config.eod_close_time, ExitReason.EOD_CLOSE)

        # Fallback: Use daily high/low to check stop
        intraday_high, intraday_low = self.data_loader.get_intraday_range(
            signal.symbol,
            signal.trade_date,
            entry_time,
            self.config.eod_close_time
        )

        if intraday_high is not None and intraday_low is not None:
            if is_long and intraday_low <= stop_price:
                # Stop was hit - assume worst case, stopped at stop price
                return (stop_price, time(12, 0), ExitReason.STOP_LOSS)  # Assume midday
            elif not is_long and intraday_high >= stop_price:
                return (stop_price, time(12, 0), ExitReason.STOP_LOSS)

        # No stop hit, exit at close
        close_price = self.data_loader.get_close_price(
            signal.symbol,
            signal.trade_date,
            self.config.eod_close_time
        )

        if close_price:
            return (close_price, self.config.eod_close_time, ExitReason.EOD_CLOSE)

        # Last resort: use entry price (no change)
        logger.warning(f"No exit price found for {signal.symbol} on {signal.trade_date}")
        return (entry_price, self.config.eod_close_time, ExitReason.EOD_CLOSE)

    def simulate_multiple_trades(
        self,
        signals: List[GapSignal],
        account_value: float
    ) -> List[SimulatedTrade]:
        """Simulate multiple trades.

        Args:
            signals: List of signals to trade
            account_value: Starting account value

        Returns:
            List of simulated trades
        """
        trades = []

        for signal in signals:
            trade = self.simulate_trade(signal, account_value)
            if trade:
                trades.append(trade)
                # Update account value for next trade's position sizing
                account_value += trade.pnl

        return trades


def compare_stop_strategies(
    data_loader: GapBacktestDataLoader,
    signals: List[GapSignal],
    account_value: float,
    config: BacktestConfig
) -> Dict[str, Any]:
    """Compare different stop-loss strategies.

    Args:
        data_loader: Data loader instance
        signals: Signals to test
        account_value: Account value for sizing
        config: Base config

    Returns:
        Comparison results
    """
    results = {}

    # Test different strategies
    strategies = [
        ('ATR_1.0', StopLossType.ATR, 1.0),
        ('ATR_1.5', StopLossType.ATR, 1.5),
        ('ATR_2.0', StopLossType.ATR, 2.0),
        ('Fixed_1%', StopLossType.FIXED_PCT, 1.0),
        ('Fixed_2%', StopLossType.FIXED_PCT, 2.0),
        ('Fixed_3%', StopLossType.FIXED_PCT, 3.0),
    ]

    for name, stop_type, multiplier in strategies:
        # Create config variant
        test_config = BacktestConfig(
            start_date=config.start_date,
            end_date=config.end_date,
            initial_capital=config.initial_capital,
            risk_per_trade_pct=config.risk_per_trade_pct,
            min_gap_pct=config.min_gap_pct,
            max_gap_pct=config.max_gap_pct,
            stop_loss_type=stop_type,
            stop_atr_multiplier=multiplier if stop_type == StopLossType.ATR else 1.5,
            stop_fixed_pct=multiplier if stop_type == StopLossType.FIXED_PCT else 2.0,
            use_risk_tiers=config.use_risk_tiers,
            slippage_pct=config.slippage_pct,
            commission=config.commission
        )

        simulator = TradeSimulator(test_config, data_loader)

        trades = []
        for signal in signals:
            trade = simulator.simulate_trade(signal, account_value)
            if trade:
                trades.append(trade)

        # Calculate metrics
        if trades:
            total_pnl = sum(t.pnl for t in trades)
            winners = [t for t in trades if t.pnl > 0]
            losers = [t for t in trades if t.pnl < 0]
            stop_exits = [t for t in trades if t.exit_reason == ExitReason.STOP_LOSS]

            results[name] = {
                'total_trades': len(trades),
                'total_pnl': total_pnl,
                'win_rate': len(winners) / len(trades) * 100,
                'avg_trade': total_pnl / len(trades),
                'stop_rate': len(stop_exits) / len(trades) * 100,
                'avg_winner': sum(t.pnl for t in winners) / len(winners) if winners else 0,
                'avg_loser': sum(t.pnl for t in losers) / len(losers) if losers else 0,
            }
        else:
            results[name] = {'total_trades': 0}

    return results
