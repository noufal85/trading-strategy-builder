"""Gap Trading Signal Generation Module.

Implements gap detection and trade signal generation:
- GapDetector: Identifies and confirms price gaps
- SignalGenerator: Generates BUY/SELL_SHORT/NO_TRADE signals
- Risk filters and ATR-based stop-loss calculations

Gap Logic:
- Gap % = (open - prev_close) / prev_close * 100
- Significant gap: >= 1.5% (configurable)
- Gap UP confirmed: price_at_940 > open_price
- Gap DOWN confirmed: price_at_940 < open_price
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, NamedTuple
from dataclasses import dataclass, field
from datetime import date, datetime, time
import logging

from .risk_tiers import RiskTier, RiskTierClassifier

logger = logging.getLogger(__name__)


class GapDirection(Enum):
    """Gap direction enumeration."""
    UP = "UP"
    DOWN = "DOWN"
    NONE = "NONE"


class SignalType(Enum):
    """Trade signal type enumeration."""
    BUY = "BUY"
    SELL_SHORT = "SELL_SHORT"
    NO_TRADE = "NO_TRADE"


class SignalReason(Enum):
    """Reason for signal generation or rejection."""
    GAP_UP_CONFIRMED = "Gap up confirmed - momentum continuing"
    GAP_DOWN_CONFIRMED = "Gap down confirmed - momentum continuing"
    GAP_NOT_CONFIRMED = "Gap not confirmed - price reversed"
    GAP_TOO_SMALL = "Gap below minimum threshold"
    GAP_TOO_LARGE = "Gap exceeds maximum threshold"
    VOLATILITY_TOO_HIGH = "Stock volatility exceeds limit"
    VOLATILITY_TOO_LOW = "Stock volatility below minimum"
    GAP_ATR_RATIO_HIGH = "Gap/ATR ratio exceeds limit"
    REFERENCE_ONLY = "Reference symbol - no trading"
    MARKET_CLOSED = "Market is closed"
    INSUFFICIENT_DATA = "Insufficient data for analysis"


@dataclass
class GapInfo:
    """Information about a detected gap.

    Attributes:
        symbol: Stock ticker symbol
        prev_close: Previous day's closing price
        open_price: Today's opening price
        gap_pct: Gap percentage
        gap_direction: UP, DOWN, or NONE
        is_significant: Whether gap meets minimum threshold
        detection_time: When gap was detected
    """
    symbol: str
    prev_close: float
    open_price: float
    gap_pct: float
    gap_direction: GapDirection
    is_significant: bool
    detection_time: Optional[datetime] = None

    @property
    def gap_amount(self) -> float:
        """Absolute gap amount in dollars."""
        return self.open_price - self.prev_close


@dataclass
class GapConfirmation:
    """Gap confirmation result at 9:40 AM.

    Attributes:
        gap_info: Original gap information
        price_at_confirmation: Price at confirmation time (9:40 AM)
        is_confirmed: Whether gap direction is confirmed
        confirmation_time: When confirmation was checked
    """
    gap_info: GapInfo
    price_at_confirmation: float
    is_confirmed: bool
    confirmation_time: Optional[datetime] = None

    @property
    def price_change_from_open(self) -> float:
        """Price change from open to confirmation."""
        return self.price_at_confirmation - self.gap_info.open_price

    @property
    def price_change_pct(self) -> float:
        """Percentage change from open to confirmation."""
        if self.gap_info.open_price <= 0:
            return 0.0
        return (self.price_change_from_open / self.gap_info.open_price) * 100


@dataclass
class TradeSignal:
    """Trade signal with entry and risk parameters.

    Attributes:
        symbol: Stock ticker symbol
        signal_type: BUY, SELL_SHORT, or NO_TRADE
        reason: Reason for signal
        entry_price: Suggested entry price
        stop_loss: Stop-loss price
        atr: Current ATR value
        atr_pct: ATR as percentage of price
        risk_tier: Risk tier classification
        gap_info: Gap information
        confirmation: Gap confirmation result
        position_multiplier: Position size multiplier from risk tier
        stop_multiplier: Stop distance multiplier from risk tier
        signal_time: When signal was generated
    """
    symbol: str
    signal_type: SignalType
    reason: SignalReason
    entry_price: float
    stop_loss: float
    atr: float
    atr_pct: float
    risk_tier: RiskTier
    gap_info: Optional[GapInfo] = None
    confirmation: Optional[GapConfirmation] = None
    position_multiplier: float = 1.0
    stop_multiplier: float = 1.0
    signal_time: Optional[datetime] = None

    @property
    def stop_distance(self) -> float:
        """Distance from entry to stop-loss."""
        return abs(self.entry_price - self.stop_loss)

    @property
    def stop_distance_pct(self) -> float:
        """Stop distance as percentage of entry."""
        if self.entry_price <= 0:
            return 0.0
        return (self.stop_distance / self.entry_price) * 100

    @property
    def is_tradeable(self) -> bool:
        """Whether this signal should be traded."""
        return self.signal_type != SignalType.NO_TRADE


class GapDetector:
    """Detects and analyzes price gaps.

    Gap Calculation:
    gap_pct = (open_price - prev_close) / prev_close * 100

    Significant gap: abs(gap_pct) >= min_gap_pct (default 1.5%)

    Gap Confirmation (at 9:40 AM ET):
    - Gap UP confirmed if price_at_940 > open_price
    - Gap DOWN confirmed if price_at_940 < open_price
    """

    def __init__(
        self,
        min_gap_pct: float = 1.5,
        max_gap_pct: float = 10.0,
        confirmation_buffer: float = 0.0
    ):
        """Initialize GapDetector.

        Args:
            min_gap_pct: Minimum gap percentage to consider (default 1.5%)
            max_gap_pct: Maximum gap percentage allowed (default 10%)
            confirmation_buffer: Buffer for confirmation (default 0%)
        """
        self.min_gap_pct = min_gap_pct
        self.max_gap_pct = max_gap_pct
        self.confirmation_buffer = confirmation_buffer

    def calculate_gap(
        self,
        prev_close: float,
        open_price: float
    ) -> Tuple[float, GapDirection]:
        """Calculate gap percentage and direction.

        Args:
            prev_close: Previous day's closing price
            open_price: Today's opening price

        Returns:
            Tuple of (gap_pct, GapDirection)
        """
        if prev_close <= 0:
            return 0.0, GapDirection.NONE

        gap_pct = ((open_price - prev_close) / prev_close) * 100

        if gap_pct > 0:
            direction = GapDirection.UP
        elif gap_pct < 0:
            direction = GapDirection.DOWN
        else:
            direction = GapDirection.NONE

        return gap_pct, direction

    def detect_gap(
        self,
        symbol: str,
        prev_close: float,
        open_price: float,
        detection_time: Optional[datetime] = None
    ) -> GapInfo:
        """Detect a gap for a symbol.

        Args:
            symbol: Stock ticker symbol
            prev_close: Previous day's closing price
            open_price: Today's opening price
            detection_time: When gap was detected

        Returns:
            GapInfo with gap details
        """
        gap_pct, direction = self.calculate_gap(prev_close, open_price)
        is_significant = abs(gap_pct) >= self.min_gap_pct

        return GapInfo(
            symbol=symbol,
            prev_close=prev_close,
            open_price=open_price,
            gap_pct=gap_pct,
            gap_direction=direction,
            is_significant=is_significant,
            detection_time=detection_time or datetime.now()
        )

    def is_significant_gap(self, gap_pct: float) -> bool:
        """Check if gap percentage is significant.

        Args:
            gap_pct: Gap percentage

        Returns:
            True if gap meets minimum threshold
        """
        return abs(gap_pct) >= self.min_gap_pct

    def is_gap_too_large(self, gap_pct: float) -> bool:
        """Check if gap exceeds maximum threshold.

        Args:
            gap_pct: Gap percentage

        Returns:
            True if gap exceeds maximum
        """
        return abs(gap_pct) > self.max_gap_pct

    def confirm_gap(
        self,
        gap_info: GapInfo,
        price_at_confirmation: float,
        confirmation_time: Optional[datetime] = None
    ) -> GapConfirmation:
        """Confirm if gap direction continues after open.

        Gap UP confirmed: price_at_940 > open_price
        Gap DOWN confirmed: price_at_940 < open_price

        Args:
            gap_info: Original gap information
            price_at_confirmation: Price at 9:40 AM (or confirmation time)
            confirmation_time: When confirmation was checked

        Returns:
            GapConfirmation with confirmation result
        """
        is_confirmed = False

        if gap_info.gap_direction == GapDirection.UP:
            # Gap up confirmed if price is still above open
            is_confirmed = price_at_confirmation > gap_info.open_price
        elif gap_info.gap_direction == GapDirection.DOWN:
            # Gap down confirmed if price is still below open
            is_confirmed = price_at_confirmation < gap_info.open_price

        return GapConfirmation(
            gap_info=gap_info,
            price_at_confirmation=price_at_confirmation,
            is_confirmed=is_confirmed,
            confirmation_time=confirmation_time or datetime.now()
        )

    def detect_gaps_batch(
        self,
        quotes: List[Dict],
        detection_time: Optional[datetime] = None
    ) -> List[GapInfo]:
        """Detect gaps for multiple symbols.

        Args:
            quotes: List of dicts with 'symbol', 'prev_close', 'open' keys
            detection_time: When gaps were detected

        Returns:
            List of GapInfo for significant gaps
        """
        gaps = []

        for quote in quotes:
            symbol = quote.get('symbol')
            prev_close = quote.get('prev_close', 0)
            open_price = quote.get('open', 0)

            if not symbol or prev_close <= 0 or open_price <= 0:
                continue

            gap_info = self.detect_gap(symbol, prev_close, open_price, detection_time)

            if gap_info.is_significant:
                gaps.append(gap_info)

        return gaps


class SignalGenerator:
    """Generates trade signals from confirmed gaps.

    Signal Logic:
    - Gap UP confirmed -> BUY signal
    - Gap DOWN confirmed -> SELL_SHORT signal
    - Not confirmed -> NO_TRADE

    Stop-Loss (ATR-based):
    - LONG: entry - (ATR * stop_multiplier)
    - SHORT: entry + (ATR * stop_multiplier)
    """

    def __init__(
        self,
        gap_detector: Optional[GapDetector] = None,
        risk_classifier: Optional[RiskTierClassifier] = None,
        min_atr_pct: float = 0.5,
        max_atr_pct: float = 8.0,
        max_gap_atr_ratio: float = 3.0,
        reference_symbols: Optional[List[str]] = None
    ):
        """Initialize SignalGenerator.

        Args:
            gap_detector: GapDetector instance (creates default if None)
            risk_classifier: RiskTierClassifier instance (creates default if None)
            min_atr_pct: Minimum ATR% to trade (default 0.5%)
            max_atr_pct: Maximum ATR% to trade (default 8%)
            max_gap_atr_ratio: Maximum gap/ATR ratio (default 3.0)
            reference_symbols: Symbols that are reference-only (no trading)
        """
        self.gap_detector = gap_detector or GapDetector()
        self.risk_classifier = risk_classifier or RiskTierClassifier()
        self.min_atr_pct = min_atr_pct
        self.max_atr_pct = max_atr_pct
        self.max_gap_atr_ratio = max_gap_atr_ratio
        self.reference_symbols = set(reference_symbols or ['VIX'])

    def generate_signal(
        self,
        confirmation: GapConfirmation,
        atr: float,
        atr_pct: float,
        current_price: Optional[float] = None
    ) -> TradeSignal:
        """Generate a trade signal from confirmed gap.

        Args:
            confirmation: Gap confirmation result
            atr: Current ATR value
            atr_pct: ATR as percentage of price
            current_price: Current price (defaults to confirmation price)

        Returns:
            TradeSignal with entry and risk parameters
        """
        gap_info = confirmation.gap_info
        symbol = gap_info.symbol
        entry_price = current_price or confirmation.price_at_confirmation

        # Get risk parameters
        risk_params = self.risk_classifier.get_risk_parameters(atr_pct)
        risk_tier = risk_params.tier
        position_mult = risk_params.position_multiplier
        stop_mult = risk_params.stop_atr_multiplier

        # Check reference symbols
        if symbol in self.reference_symbols:
            return self._no_trade_signal(
                symbol, entry_price, atr, atr_pct, risk_tier,
                SignalReason.REFERENCE_ONLY, gap_info, confirmation,
                position_mult, stop_mult
            )

        # Check if gap is confirmed
        if not confirmation.is_confirmed:
            return self._no_trade_signal(
                symbol, entry_price, atr, atr_pct, risk_tier,
                SignalReason.GAP_NOT_CONFIRMED, gap_info, confirmation,
                position_mult, stop_mult
            )

        # Apply risk filters
        filter_result = self._apply_risk_filters(gap_info, atr, atr_pct)
        if filter_result is not None:
            return self._no_trade_signal(
                symbol, entry_price, atr, atr_pct, risk_tier,
                filter_result, gap_info, confirmation,
                position_mult, stop_mult
            )

        # Generate signal based on gap direction
        if gap_info.gap_direction == GapDirection.UP:
            signal_type = SignalType.BUY
            reason = SignalReason.GAP_UP_CONFIRMED
            stop_loss = entry_price - (atr * stop_mult)
        else:  # GapDirection.DOWN
            signal_type = SignalType.SELL_SHORT
            reason = SignalReason.GAP_DOWN_CONFIRMED
            stop_loss = entry_price + (atr * stop_mult)

        return TradeSignal(
            symbol=symbol,
            signal_type=signal_type,
            reason=reason,
            entry_price=entry_price,
            stop_loss=stop_loss,
            atr=atr,
            atr_pct=atr_pct,
            risk_tier=risk_tier,
            gap_info=gap_info,
            confirmation=confirmation,
            position_multiplier=position_mult,
            stop_multiplier=stop_mult,
            signal_time=datetime.now()
        )

    def _apply_risk_filters(
        self,
        gap_info: GapInfo,
        atr: float,
        atr_pct: float
    ) -> Optional[SignalReason]:
        """Apply risk filters to potential signal.

        Returns SignalReason if filtered out, None if passes all filters.
        """
        # Check gap size
        if not gap_info.is_significant:
            return SignalReason.GAP_TOO_SMALL

        if self.gap_detector.is_gap_too_large(gap_info.gap_pct):
            return SignalReason.GAP_TOO_LARGE

        # Check volatility bounds
        if atr_pct < self.min_atr_pct:
            return SignalReason.VOLATILITY_TOO_LOW

        if atr_pct > self.max_atr_pct:
            return SignalReason.VOLATILITY_TOO_HIGH

        # Check gap/ATR ratio
        if atr > 0:
            gap_atr_ratio = abs(gap_info.gap_pct) / atr_pct
            if gap_atr_ratio > self.max_gap_atr_ratio:
                return SignalReason.GAP_ATR_RATIO_HIGH

        return None

    def _no_trade_signal(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        atr_pct: float,
        risk_tier: RiskTier,
        reason: SignalReason,
        gap_info: Optional[GapInfo],
        confirmation: Optional[GapConfirmation],
        position_mult: float,
        stop_mult: float
    ) -> TradeSignal:
        """Create a NO_TRADE signal."""
        return TradeSignal(
            symbol=symbol,
            signal_type=SignalType.NO_TRADE,
            reason=reason,
            entry_price=entry_price,
            stop_loss=entry_price,  # No stop for no-trade
            atr=atr,
            atr_pct=atr_pct,
            risk_tier=risk_tier,
            gap_info=gap_info,
            confirmation=confirmation,
            position_multiplier=position_mult,
            stop_multiplier=stop_mult,
            signal_time=datetime.now()
        )

    def calculate_stop_loss(
        self,
        entry_price: float,
        atr: float,
        signal_type: SignalType,
        stop_multiplier: float = 1.5
    ) -> float:
        """Calculate stop-loss price.

        Args:
            entry_price: Entry price
            atr: Current ATR value
            signal_type: BUY or SELL_SHORT
            stop_multiplier: ATR multiplier for stop distance

        Returns:
            Stop-loss price
        """
        stop_distance = atr * stop_multiplier

        if signal_type == SignalType.BUY:
            return entry_price - stop_distance
        elif signal_type == SignalType.SELL_SHORT:
            return entry_price + stop_distance
        else:
            return entry_price

    def generate_signals_batch(
        self,
        confirmations: List[GapConfirmation],
        stock_data: Dict[str, Dict]
    ) -> List[TradeSignal]:
        """Generate signals for multiple confirmed gaps.

        Args:
            confirmations: List of gap confirmations
            stock_data: Dict mapping symbol to {'atr': float, 'atr_pct': float}

        Returns:
            List of TradeSignal objects
        """
        signals = []

        for confirmation in confirmations:
            symbol = confirmation.gap_info.symbol
            data = stock_data.get(symbol, {})

            atr = data.get('atr', 0)
            atr_pct = data.get('atr_pct', 0)

            if atr <= 0 or atr_pct <= 0:
                logger.warning(f"Missing ATR data for {symbol}")
                continue

            signal = self.generate_signal(confirmation, atr, atr_pct)
            signals.append(signal)

        return signals
