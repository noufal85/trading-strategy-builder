"""Signal Engine for Gap Trading Backtest.

Handles gap detection, confirmation, and signal generation.
"""

import logging
from dataclasses import dataclass
from datetime import date, time, datetime, timedelta
from typing import Optional, List, Dict, Any
from enum import Enum

from .config import BacktestConfig
from .data_loader import GapBacktestDataLoader

logger = logging.getLogger(__name__)


class SignalType(str, Enum):
    """Trade signal type."""
    BUY = 'BUY'
    SELL_SHORT = 'SELL_SHORT'
    NO_TRADE = 'NO_TRADE'


class GapDirection(str, Enum):
    """Gap direction."""
    UP = 'UP'
    DOWN = 'DOWN'


class RiskTier(str, Enum):
    """Risk tier classification."""
    LOW = 'LOW'
    MEDIUM = 'MEDIUM'
    HIGH = 'HIGH'


class RejectionReason(str, Enum):
    """Reason for rejecting a trade signal."""
    GAP_TOO_SMALL = 'GAP_TOO_SMALL'
    GAP_TOO_LARGE = 'GAP_TOO_LARGE'
    NOT_CONFIRMED = 'NOT_CONFIRMED'
    NO_DATA = 'NO_DATA'
    VOLATILITY_TOO_LOW = 'VOLATILITY_TOO_LOW'
    VOLATILITY_TOO_HIGH = 'VOLATILITY_TOO_HIGH'
    GAP_ATR_RATIO_HIGH = 'GAP_ATR_RATIO_HIGH'
    INSUFFICIENT_ATR = 'INSUFFICIENT_ATR'


@dataclass
class GapSignal:
    """Gap trading signal with all relevant data.

    Attributes:
        symbol: Stock ticker
        trade_date: Date of the signal
        gap_pct: Gap percentage from previous close to open
        gap_direction: UP or DOWN
        open_price: Market open price
        prev_close: Previous day's close
        confirmation_price: Price at confirmation time (e.g., 9:40)
        confirmation_time: Time gap was confirmed
        is_confirmed: Whether gap was confirmed
        signal_type: BUY, SELL_SHORT, or NO_TRADE
        rejection_reason: Why signal was rejected (if NO_TRADE)
        atr: ATR value for position sizing
        atr_pct: ATR as percentage of price
        risk_tier: LOW, MEDIUM, or HIGH
        stop_price: Calculated stop-loss price
    """
    symbol: str
    trade_date: date
    gap_pct: float
    gap_direction: GapDirection
    open_price: float
    prev_close: float
    confirmation_price: Optional[float]
    confirmation_time: Optional[time]
    is_confirmed: bool
    signal_type: SignalType
    rejection_reason: Optional[RejectionReason] = None
    atr: Optional[float] = None
    atr_pct: Optional[float] = None
    risk_tier: Optional[RiskTier] = None
    stop_price: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'symbol': self.symbol,
            'trade_date': self.trade_date.isoformat(),
            'gap_pct': round(self.gap_pct, 2),
            'gap_direction': self.gap_direction.value,
            'open_price': self.open_price,
            'prev_close': self.prev_close,
            'confirmation_price': self.confirmation_price,
            'is_confirmed': self.is_confirmed,
            'signal_type': self.signal_type.value,
            'rejection_reason': self.rejection_reason.value if self.rejection_reason else None,
            'atr': self.atr,
            'atr_pct': round(self.atr_pct, 2) if self.atr_pct else None,
            'risk_tier': self.risk_tier.value if self.risk_tier else None,
            'stop_price': self.stop_price,
        }


class SignalEngine:
    """Engine for generating gap trading signals.

    Handles:
    - Gap detection from previous close to open
    - Gap confirmation at specified time
    - Risk filtering based on volatility
    - Stop-loss calculation

    Attributes:
        config: Backtest configuration
        data_loader: Data loader instance
    """

    def __init__(self, config: BacktestConfig, data_loader: GapBacktestDataLoader):
        """Initialize signal engine.

        Args:
            config: Backtest configuration
            data_loader: Data loader for price data
        """
        self.config = config
        self.data_loader = data_loader

        # Calculate confirmation time
        open_minutes = config.market_open.hour * 60 + config.market_open.minute
        confirm_minutes = open_minutes + config.confirmation_minutes
        self.confirmation_time = time(confirm_minutes // 60, confirm_minutes % 60)

        logger.info(
            f"SignalEngine initialized: gap_range={config.min_gap_pct}-{config.max_gap_pct}%, "
            f"confirmation_time={self.confirmation_time}"
        )

    def detect_gaps(
        self,
        symbols: List[str],
        trade_date: date
    ) -> List[GapSignal]:
        """Detect gaps for all symbols on a given date.

        Args:
            symbols: List of symbols to scan
            trade_date: Date to check for gaps

        Returns:
            List of GapSignal objects (including NO_TRADE signals)
        """
        signals = []

        for symbol in symbols:
            signal = self.detect_gap(symbol, trade_date)
            if signal:
                signals.append(signal)

        # Log summary
        trade_signals = [s for s in signals if s.signal_type != SignalType.NO_TRADE]
        logger.debug(
            f"Detected {len(signals)} gaps on {trade_date}, "
            f"{len(trade_signals)} tradeable"
        )

        return signals

    def detect_gap(self, symbol: str, trade_date: date) -> Optional[GapSignal]:
        """Detect gap for a single symbol.

        Args:
            symbol: Stock ticker
            trade_date: Date to check

        Returns:
            GapSignal or None if no data
        """
        # Get previous close
        prev_close = self.data_loader.get_previous_close(symbol, trade_date)
        if not prev_close:
            return None

        # Get today's open
        open_price = self.data_loader.get_price_at_time(
            symbol, trade_date, self.config.market_open
        )
        if not open_price:
            return self._create_no_trade_signal(
                symbol, trade_date, 0, prev_close, 0,
                RejectionReason.NO_DATA
            )

        # Calculate gap percentage
        gap_pct = ((open_price - prev_close) / prev_close) * 100
        gap_direction = GapDirection.UP if gap_pct > 0 else GapDirection.DOWN

        # Check gap size thresholds
        abs_gap = abs(gap_pct)
        if abs_gap < self.config.min_gap_pct:
            return self._create_no_trade_signal(
                symbol, trade_date, gap_pct, prev_close, open_price,
                RejectionReason.GAP_TOO_SMALL
            )

        if abs_gap > self.config.max_gap_pct:
            return self._create_no_trade_signal(
                symbol, trade_date, gap_pct, prev_close, open_price,
                RejectionReason.GAP_TOO_LARGE
            )

        # Get ATR for volatility filtering and position sizing
        atr = self.data_loader.calculate_atr(symbol, trade_date)
        if not atr or atr <= 0:
            return self._create_no_trade_signal(
                symbol, trade_date, gap_pct, prev_close, open_price,
                RejectionReason.INSUFFICIENT_ATR
            )

        atr_pct = (atr / open_price) * 100

        # Apply volatility filters
        if atr_pct < 0.5:
            return self._create_no_trade_signal(
                symbol, trade_date, gap_pct, prev_close, open_price,
                RejectionReason.VOLATILITY_TOO_LOW, atr, atr_pct
            )

        if atr_pct > 8.0:
            return self._create_no_trade_signal(
                symbol, trade_date, gap_pct, prev_close, open_price,
                RejectionReason.VOLATILITY_TOO_HIGH, atr, atr_pct
            )

        # Check gap/ATR ratio
        gap_atr_ratio = abs_gap / atr_pct
        if gap_atr_ratio > 3.0:
            return self._create_no_trade_signal(
                symbol, trade_date, gap_pct, prev_close, open_price,
                RejectionReason.GAP_ATR_RATIO_HIGH, atr, atr_pct
            )

        # Get confirmation price
        confirmation_price = self.data_loader.get_price_at_time(
            symbol, trade_date, self.confirmation_time
        )

        # Check gap confirmation
        is_confirmed = self._check_confirmation(
            gap_direction, open_price, confirmation_price
        )

        if not is_confirmed:
            return self._create_no_trade_signal(
                symbol, trade_date, gap_pct, prev_close, open_price,
                RejectionReason.NOT_CONFIRMED, atr, atr_pct,
                confirmation_price=confirmation_price
            )

        # Calculate risk tier
        risk_tier = self._calculate_risk_tier(atr_pct)

        # Calculate stop price
        stop_price = self._calculate_stop_price(
            gap_direction, confirmation_price or open_price, atr, risk_tier
        )

        # Determine signal type
        signal_type = SignalType.BUY if gap_direction == GapDirection.UP else SignalType.SELL_SHORT

        return GapSignal(
            symbol=symbol,
            trade_date=trade_date,
            gap_pct=gap_pct,
            gap_direction=gap_direction,
            open_price=open_price,
            prev_close=prev_close,
            confirmation_price=confirmation_price,
            confirmation_time=self.confirmation_time,
            is_confirmed=True,
            signal_type=signal_type,
            rejection_reason=None,
            atr=atr,
            atr_pct=atr_pct,
            risk_tier=risk_tier,
            stop_price=stop_price
        )

    def _check_confirmation(
        self,
        gap_direction: GapDirection,
        open_price: float,
        confirmation_price: Optional[float]
    ) -> bool:
        """Check if gap is confirmed by price action.

        Gap UP confirmed: price at confirmation > open (momentum continues)
        Gap DOWN confirmed: price at confirmation < open (momentum continues)

        Args:
            gap_direction: UP or DOWN
            open_price: Market open price
            confirmation_price: Price at confirmation time

        Returns:
            True if gap is confirmed
        """
        if confirmation_price is None:
            # If no minute data, assume confirmed (fallback behavior)
            return True

        if gap_direction == GapDirection.UP:
            return confirmation_price > open_price
        else:
            return confirmation_price < open_price

    def _calculate_risk_tier(self, atr_pct: float) -> RiskTier:
        """Classify risk tier based on ATR percentage.

        Args:
            atr_pct: ATR as percentage of price

        Returns:
            Risk tier classification
        """
        if atr_pct < 2.0:
            return RiskTier.LOW
        elif atr_pct < 4.0:
            return RiskTier.MEDIUM
        else:
            return RiskTier.HIGH

    def _calculate_stop_price(
        self,
        gap_direction: GapDirection,
        entry_price: float,
        atr: float,
        risk_tier: RiskTier
    ) -> float:
        """Calculate stop-loss price based on configuration.

        Args:
            gap_direction: UP or DOWN
            entry_price: Entry price
            atr: ATR value
            risk_tier: Risk tier for multiplier adjustment

        Returns:
            Stop-loss price
        """
        from .config import StopLossType

        if self.config.stop_loss_type == StopLossType.FIXED_PCT:
            stop_distance = entry_price * (self.config.stop_fixed_pct / 100)
        else:
            # ATR-based stop with tier adjustment
            tier_multipliers = {
                RiskTier.LOW: 1.0,
                RiskTier.MEDIUM: 1.5,
                RiskTier.HIGH: 2.0
            }
            multiplier = self.config.stop_atr_multiplier
            if self.config.use_risk_tiers:
                multiplier = tier_multipliers.get(risk_tier, 1.5)
            stop_distance = atr * multiplier

        if gap_direction == GapDirection.UP:
            # Long position: stop below entry
            return entry_price - stop_distance
        else:
            # Short position: stop above entry
            return entry_price + stop_distance

    def _create_no_trade_signal(
        self,
        symbol: str,
        trade_date: date,
        gap_pct: float,
        prev_close: float,
        open_price: float,
        reason: RejectionReason,
        atr: Optional[float] = None,
        atr_pct: Optional[float] = None,
        confirmation_price: Optional[float] = None
    ) -> GapSignal:
        """Create a NO_TRADE signal with rejection reason.

        Args:
            symbol: Stock ticker
            trade_date: Date
            gap_pct: Gap percentage
            prev_close: Previous close
            open_price: Open price
            reason: Why signal was rejected
            atr: ATR value (if available)
            atr_pct: ATR percentage (if available)
            confirmation_price: Confirmation price (if available)

        Returns:
            GapSignal with NO_TRADE type
        """
        gap_direction = GapDirection.UP if gap_pct > 0 else GapDirection.DOWN

        return GapSignal(
            symbol=symbol,
            trade_date=trade_date,
            gap_pct=gap_pct,
            gap_direction=gap_direction,
            open_price=open_price,
            prev_close=prev_close,
            confirmation_price=confirmation_price,
            confirmation_time=self.confirmation_time if confirmation_price else None,
            is_confirmed=False,
            signal_type=SignalType.NO_TRADE,
            rejection_reason=reason,
            atr=atr,
            atr_pct=atr_pct,
            risk_tier=self._calculate_risk_tier(atr_pct) if atr_pct else None,
            stop_price=None
        )

    def filter_signals(
        self,
        signals: List[GapSignal],
        max_positions: int,
        max_long: int,
        max_short: int
    ) -> List[GapSignal]:
        """Filter and prioritize signals based on position limits.

        Args:
            signals: List of all signals
            max_positions: Maximum total positions
            max_long: Maximum long positions
            max_short: Maximum short positions

        Returns:
            Filtered list of signals to trade
        """
        # Get only tradeable signals
        tradeable = [s for s in signals if s.signal_type != SignalType.NO_TRADE]

        if not tradeable:
            return []

        # Sort by absolute gap size (larger gaps first)
        tradeable.sort(key=lambda s: abs(s.gap_pct), reverse=True)

        # Apply position limits
        selected = []
        long_count = 0
        short_count = 0

        for signal in tradeable:
            if len(selected) >= max_positions:
                break

            if signal.signal_type == SignalType.BUY:
                if long_count < max_long:
                    selected.append(signal)
                    long_count += 1
            else:  # SELL_SHORT
                if short_count < max_short:
                    selected.append(signal)
                    short_count += 1

        return selected


def analyze_gap_continuation(
    data_loader: GapBacktestDataLoader,
    symbols: List[str],
    start_date: date,
    end_date: date,
    min_gap_pct: float = 1.5,
    check_times: List[int] = None
) -> Dict[str, Any]:
    """Analyze gap continuation rates.

    Args:
        data_loader: Data loader instance
        symbols: Symbols to analyze
        start_date: Start date
        end_date: End date
        min_gap_pct: Minimum gap to consider
        check_times: Minutes after open to check (default: [5, 10, 15, 20, 30])

    Returns:
        Analysis results with continuation rates by time
    """
    if check_times is None:
        check_times = [5, 10, 15, 20, 30]

    results = {
        'total_gaps': 0,
        'by_time': {t: {'continued': 0, 'reversed': 0} for t in check_times},
        'by_gap_size': {},
        'by_direction': {'UP': {'continued': 0, 'total': 0}, 'DOWN': {'continued': 0, 'total': 0}}
    }

    trading_days = data_loader.get_trading_days(start_date, end_date)

    for trade_date in trading_days:
        for symbol in symbols:
            # Get prices
            prev_close = data_loader.get_previous_close(symbol, trade_date)
            open_price = data_loader.get_price_at_time(symbol, trade_date, time(9, 30))

            if not prev_close or not open_price:
                continue

            gap_pct = ((open_price - prev_close) / prev_close) * 100
            if abs(gap_pct) < min_gap_pct:
                continue

            results['total_gaps'] += 1
            direction = 'UP' if gap_pct > 0 else 'DOWN'
            results['by_direction'][direction]['total'] += 1

            # Check continuation at each time
            for minutes in check_times:
                check_time = time(9, 30 + minutes)
                price = data_loader.get_price_at_time(symbol, trade_date, check_time)

                if price:
                    if direction == 'UP':
                        continued = price > open_price
                    else:
                        continued = price < open_price

                    if continued:
                        results['by_time'][minutes]['continued'] += 1
                        if minutes == 10:  # Track direction continuation at 10 min
                            results['by_direction'][direction]['continued'] += 1
                    else:
                        results['by_time'][minutes]['reversed'] += 1

    # Calculate rates
    for t in check_times:
        total = results['by_time'][t]['continued'] + results['by_time'][t]['reversed']
        if total > 0:
            results['by_time'][t]['rate'] = results['by_time'][t]['continued'] / total * 100
        else:
            results['by_time'][t]['rate'] = 0

    for direction in ['UP', 'DOWN']:
        total = results['by_direction'][direction]['total']
        if total > 0:
            results['by_direction'][direction]['rate'] = (
                results['by_direction'][direction]['continued'] / total * 100
            )
        else:
            results['by_direction'][direction]['rate'] = 0

    return results
