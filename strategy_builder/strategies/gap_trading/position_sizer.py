"""Gap Trading Position Sizing Module.

Implements ATR-based position sizing with risk tier adjustments:
- Calculate shares based on risk amount and ATR
- Apply tier multipliers for position size
- Enforce maximum position constraints
- Calculate dollar risk per trade

Position Sizing Formula:
shares = risk_amount / (ATR * stop_multiplier)
shares = shares * tier_position_multiplier
"""

from typing import Dict, List, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
import math

from .risk_tiers import RiskTier, RiskTierClassifier, RiskParameters
from .signals import TradeSignal, SignalType

logger = logging.getLogger(__name__)


class SizingMethod(Enum):
    """Position sizing method enumeration."""
    ATR_BASED = "atr_based"
    FIXED_DOLLAR = "fixed_dollar"
    FIXED_SHARES = "fixed_shares"
    PERCENT_EQUITY = "percent_equity"


@dataclass
class PositionSize:
    """Calculated position size with details.

    Attributes:
        symbol: Stock ticker symbol
        shares: Number of shares to trade (supports fractional)
        entry_price: Expected entry price
        position_value: Total position value (shares * price)
        risk_amount: Dollar amount at risk
        stop_loss: Stop-loss price
        stop_distance: Distance from entry to stop
        atr: ATR value used in calculation
        risk_tier: Risk tier classification
        sizing_method: Method used for sizing
        adjustments: Dict of adjustments applied
    """
    symbol: str
    shares: float  # Supports fractional shares
    entry_price: float
    position_value: float
    risk_amount: float
    stop_loss: float
    stop_distance: float
    atr: float
    risk_tier: RiskTier
    sizing_method: SizingMethod
    adjustments: Dict[str, float] = None

    def __post_init__(self):
        if self.adjustments is None:
            self.adjustments = {}

    @property
    def risk_per_share(self) -> float:
        """Risk per share (stop distance)."""
        return self.stop_distance

    @property
    def risk_percent(self) -> float:
        """Risk as percentage of position value."""
        if self.position_value <= 0:
            return 0.0
        return (self.risk_amount / self.position_value) * 100


class PositionSizer:
    """Calculates position sizes for gap trading.

    Position Sizing Formula (ATR-based):
    1. base_shares = risk_amount / (ATR * stop_multiplier)
    2. adjusted_shares = base_shares * tier_position_multiplier
    3. final_shares = min(adjusted_shares, max_position_shares)

    Risk Management:
    - Default risk per trade: 1% of account
    - Maximum single position: 10% of account
    - Minimum shares: 0.0001 (fractional)
    - Supports fractional shares for small accounts
    """

    def __init__(
        self,
        account_value: float,
        risk_per_trade_pct: float = 1.0,
        max_position_pct: float = 10.0,
        min_shares: float = 0.0001,
        max_shares: Optional[float] = None,
        risk_classifier: Optional[RiskTierClassifier] = None,
        sizing_method: SizingMethod = SizingMethod.ATR_BASED
    ):
        """Initialize PositionSizer.

        Args:
            account_value: Total account value
            risk_per_trade_pct: Risk per trade as % of account (default 1%)
            max_position_pct: Max position as % of account (default 10%)
            min_shares: Minimum shares per trade (default 0.0001 for fractional)
            max_shares: Maximum shares per trade (optional)
            risk_classifier: RiskTierClassifier instance
            sizing_method: Position sizing method
        """
        self.account_value = account_value
        self.risk_per_trade_pct = risk_per_trade_pct
        self.max_position_pct = max_position_pct
        self.min_shares = min_shares
        self.max_shares = max_shares
        self.risk_classifier = risk_classifier or RiskTierClassifier()
        self.sizing_method = sizing_method

    @property
    def risk_amount(self) -> float:
        """Dollar risk amount per trade."""
        return self.account_value * (self.risk_per_trade_pct / 100)

    @property
    def max_position_value(self) -> float:
        """Maximum position value."""
        return self.account_value * (self.max_position_pct / 100)

    def calculate_shares(
        self,
        entry_price: float,
        atr: float,
        stop_multiplier: float = 1.5,
        risk_amount: Optional[float] = None
    ) -> float:
        """Calculate base shares from ATR.

        shares = risk_amount / (ATR * stop_multiplier)

        Args:
            entry_price: Entry price
            atr: ATR value
            stop_multiplier: ATR multiplier for stop distance
            risk_amount: Dollar risk (defaults to self.risk_amount)

        Returns:
            Number of shares (fractional, at least min_shares)
        """
        risk = risk_amount or self.risk_amount

        if atr <= 0 or stop_multiplier <= 0:
            logger.warning("Invalid ATR or stop multiplier")
            return self.min_shares

        stop_distance = atr * stop_multiplier
        if stop_distance <= 0:
            return self.min_shares

        shares = risk / stop_distance

        # Round to 4 decimal places for fractional shares (Alpaca precision)
        shares = round(shares, 4)

        return max(shares, self.min_shares)

    def apply_tier_adjustment(
        self,
        shares: float,
        risk_tier: RiskTier
    ) -> float:
        """Apply risk tier position multiplier.

        Args:
            shares: Base number of shares (fractional)
            risk_tier: Risk tier for adjustment

        Returns:
            Adjusted number of shares (fractional)
        """
        multiplier = self.risk_classifier.get_position_multiplier(risk_tier)
        adjusted = round(shares * multiplier, 4)

        return max(adjusted, self.min_shares)

    def check_max_position(
        self,
        shares: float,
        price: float,
        max_position_value: Optional[float] = None
    ) -> float:
        """Check and limit position to maximum.

        Args:
            shares: Proposed number of shares (fractional)
            price: Entry price
            max_position_value: Maximum position value (optional)

        Returns:
            Shares limited to maximum position (fractional)
        """
        max_value = max_position_value or self.max_position_value
        position_value = shares * price

        if position_value > max_value:
            shares = round(max_value / price, 4)

        # Also apply max_shares limit if set
        if self.max_shares is not None:
            shares = min(shares, self.max_shares)

        return max(shares, self.min_shares)

    def calculate_position(
        self,
        signal: TradeSignal,
        risk_amount: Optional[float] = None,
        buying_power: Optional[float] = None
    ) -> PositionSize:
        """Calculate complete position size from trade signal.

        Args:
            signal: TradeSignal with entry/stop/ATR info
            risk_amount: Override risk amount (optional)
            buying_power: Available buying power (optional)

        Returns:
            PositionSize with all details
        """
        risk = risk_amount or self.risk_amount
        entry_price = signal.entry_price
        stop_loss = signal.stop_loss
        atr = signal.atr
        risk_tier = signal.risk_tier
        stop_mult = signal.stop_multiplier

        # Track adjustments
        adjustments = {}

        # Step 1: Calculate base shares
        base_shares = self.calculate_shares(entry_price, atr, stop_mult, risk)
        adjustments['base_shares'] = base_shares

        # Step 2: Apply tier adjustment
        tier_adjusted = self.apply_tier_adjustment(base_shares, risk_tier)
        adjustments['tier_multiplier'] = self.risk_classifier.get_position_multiplier(risk_tier)
        adjustments['after_tier'] = tier_adjusted

        # Step 3: Check max position
        final_shares = self.check_max_position(tier_adjusted, entry_price)
        adjustments['max_position_cap'] = final_shares != tier_adjusted

        # Step 4: Check buying power if provided
        if buying_power is not None:
            max_affordable = round(buying_power / entry_price, 4)
            if final_shares > max_affordable:
                final_shares = max(max_affordable, self.min_shares)
                adjustments['buying_power_limit'] = True

        # Step 5: Round up to whole shares for shorts (Alpaca doesn't support fractional shorts)
        if signal.signal_type == SignalType.SELL_SHORT:
            final_shares = max(math.ceil(final_shares), 1)
            adjustments['rounded_for_short'] = True

        # Calculate final values
        position_value = final_shares * entry_price
        stop_distance = abs(entry_price - stop_loss)
        actual_risk = final_shares * stop_distance

        return PositionSize(
            symbol=signal.symbol,
            shares=final_shares,
            entry_price=entry_price,
            position_value=position_value,
            risk_amount=actual_risk,
            stop_loss=stop_loss,
            stop_distance=stop_distance,
            atr=atr,
            risk_tier=risk_tier,
            sizing_method=self.sizing_method,
            adjustments=adjustments
        )

    def calculate_from_params(
        self,
        symbol: str,
        entry_price: float,
        atr: float,
        atr_pct: float,
        signal_type: SignalType,
        risk_amount: Optional[float] = None
    ) -> PositionSize:
        """Calculate position size from raw parameters.

        Args:
            symbol: Stock ticker symbol
            entry_price: Entry price
            atr: ATR value
            atr_pct: ATR as percentage of price
            signal_type: BUY or SELL_SHORT
            risk_amount: Override risk amount

        Returns:
            PositionSize with all details
        """
        risk = risk_amount or self.risk_amount

        # Get risk parameters
        risk_params = self.risk_classifier.get_risk_parameters(atr_pct)
        stop_mult = risk_params.stop_atr_multiplier
        position_mult = risk_params.position_multiplier

        # Calculate stop-loss
        stop_distance = atr * stop_mult
        if signal_type == SignalType.BUY:
            stop_loss = entry_price - stop_distance
        else:
            stop_loss = entry_price + stop_distance

        # Calculate shares
        base_shares = self.calculate_shares(entry_price, atr, stop_mult, risk)
        tier_adjusted = round(base_shares * position_mult, 4)
        final_shares = self.check_max_position(tier_adjusted, entry_price)

        # Round up to whole shares for shorts (Alpaca doesn't support fractional shorts)
        if signal_type == SignalType.SELL_SHORT:
            final_shares = max(math.ceil(final_shares), 1)

        # Final values
        position_value = final_shares * entry_price
        actual_risk = final_shares * stop_distance

        return PositionSize(
            symbol=symbol,
            shares=final_shares,
            entry_price=entry_price,
            position_value=position_value,
            risk_amount=actual_risk,
            stop_loss=stop_loss,
            stop_distance=stop_distance,
            atr=atr,
            risk_tier=risk_params.tier,
            sizing_method=self.sizing_method,
            adjustments={
                'base_shares': base_shares,
                'tier_multiplier': position_mult,
                'stop_multiplier': stop_mult
            }
        )

    def update_account_value(self, new_value: float) -> None:
        """Update account value for position sizing.

        Args:
            new_value: New account value
        """
        self.account_value = new_value
        logger.info(f"Account value updated to ${new_value:,.2f}")

    def get_sizing_summary(self) -> Dict:
        """Get summary of current sizing parameters.

        Returns:
            Dict with sizing configuration
        """
        return {
            'account_value': self.account_value,
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'risk_amount': self.risk_amount,
            'max_position_pct': self.max_position_pct,
            'max_position_value': self.max_position_value,
            'min_shares': self.min_shares,
            'max_shares': self.max_shares,
            'sizing_method': self.sizing_method.value
        }


def calculate_shares_simple(
    risk_amount: float,
    atr: float,
    stop_multiplier: float = 1.5,
    position_multiplier: float = 1.0
) -> float:
    """Simple function to calculate shares (supports fractional).

    Args:
        risk_amount: Dollar amount to risk
        atr: ATR value
        stop_multiplier: ATR multiplier for stop
        position_multiplier: Tier-based position multiplier

    Returns:
        Number of shares (fractional)
    """
    if atr <= 0 or stop_multiplier <= 0:
        return 0.0001

    stop_distance = atr * stop_multiplier
    base_shares = risk_amount / stop_distance
    adjusted = base_shares * position_multiplier

    return max(round(adjusted, 4), 0.0001)


# Priority tier multipliers for capital deployment
PRIORITY_TIER_MULTIPLIERS = {
    'top_tier': 1.20,      # Top 2 signals: 120% of base
    'mid_tier': 1.00,      # Mid 2 signals: 100% of base
    'bottom_tier': 0.80,   # Bottom 2 signals: 80% of base
    'excluded': 0.00       # Excluded: no position
}


class TieredPositionSizer:
    """Position sizer with priority tier-based capital deployment.

    Divides capital across max_trades positions with tier multipliers:
    - top_tier (rank 1-2): 120% of base position
    - mid_tier (rank 3-4): 100% of base position
    - bottom_tier (rank 5-6): 80% of base position

    This ensures higher-conviction trades get more capital while
    maintaining approximately full capital deployment.
    """

    def __init__(
        self,
        total_capital: float,
        max_trades: int = 6,
        tier_multipliers: Optional[Dict[str, float]] = None,
        min_shares: float = 0.0001,
    ):
        """Initialize TieredPositionSizer.

        Args:
            total_capital: Total capital to deploy
            max_trades: Maximum trades per day (default 6)
            tier_multipliers: Custom tier multipliers (optional)
            min_shares: Minimum shares per trade
        """
        self.total_capital = total_capital
        self.max_trades = max_trades
        self.tier_multipliers = tier_multipliers or PRIORITY_TIER_MULTIPLIERS
        self.min_shares = min_shares

    @property
    def base_position_value(self) -> float:
        """Base position value = total capital / max trades."""
        return self.total_capital / self.max_trades

    def get_tier_multiplier(self, tier: str) -> float:
        """Get multiplier for a position tier.

        Args:
            tier: Position tier (top_tier, mid_tier, bottom_tier, excluded)

        Returns:
            Position size multiplier
        """
        return self.tier_multipliers.get(tier, 1.0)

    def calculate_position_value(self, tier: str) -> float:
        """Calculate position value for a tier.

        Args:
            tier: Position tier

        Returns:
            Dollar value for position
        """
        multiplier = self.get_tier_multiplier(tier)
        return round(self.base_position_value * multiplier, 2)

    def calculate_shares(
        self,
        entry_price: float,
        tier: str,
        is_short: bool = False
    ) -> float:
        """Calculate shares for a position.

        Args:
            entry_price: Entry price
            tier: Position tier
            is_short: Whether this is a short position

        Returns:
            Number of shares (rounded appropriately)
        """
        position_value = self.calculate_position_value(tier)

        if entry_price <= 0:
            return self.min_shares

        shares = position_value / entry_price

        # Round up for shorts (Alpaca doesn't support fractional shorts)
        if is_short:
            shares = max(math.ceil(shares), 1)
        else:
            shares = round(shares, 4)

        return max(shares, self.min_shares)

    def calculate_tiered_positions(
        self,
        signals: List[Dict],
        entry_price_key: str = 'entry_price',
        tier_key: str = 'position_tier',
        signal_type_key: str = 'signal_type'
    ) -> List[Dict]:
        """Calculate position sizes for all signals based on their tiers.

        Args:
            signals: List of signal dicts with entry_price and position_tier
            entry_price_key: Key for entry price in signal dict
            tier_key: Key for position tier in signal dict
            signal_type_key: Key for signal type in signal dict

        Returns:
            List of signals with updated shares and position_value
        """
        updated_signals = []

        for sig in signals:
            tier = sig.get(tier_key, 'mid_tier')
            entry_price = sig.get(entry_price_key, 0)
            signal_type = sig.get(signal_type_key, 'BUY')
            is_short = signal_type in ('SELL_SHORT', 'SHORT', 'SELL')

            if tier == 'excluded' or entry_price <= 0:
                # Keep original values for excluded signals
                updated_signals.append(sig)
                continue

            # Calculate new position values
            new_position_value = self.calculate_position_value(tier)
            new_shares = self.calculate_shares(entry_price, tier, is_short)

            # Update signal with new position sizing
            updated_sig = sig.copy()
            updated_sig['shares'] = new_shares
            updated_sig['position_value'] = new_position_value
            updated_sig['tier_multiplier'] = self.get_tier_multiplier(tier)
            updated_sig['sizing_method'] = 'tiered_capital_deployment'

            updated_signals.append(updated_sig)

        return updated_signals

    def get_deployment_summary(self, signals: List[Dict]) -> Dict:
        """Get summary of capital deployment across tiers.

        Args:
            signals: List of signals with position_tier

        Returns:
            Dict with deployment statistics
        """
        tier_counts = {'top_tier': 0, 'mid_tier': 0, 'bottom_tier': 0, 'excluded': 0}
        tier_values = {'top_tier': 0.0, 'mid_tier': 0.0, 'bottom_tier': 0.0}

        for sig in signals:
            tier = sig.get('position_tier', 'excluded')
            if tier in tier_counts:
                tier_counts[tier] += 1
            if tier in tier_values:
                tier_values[tier] += sig.get('position_value', 0)

        total_deployed = sum(tier_values.values())
        deployment_pct = (total_deployed / self.total_capital * 100) if self.total_capital > 0 else 0

        return {
            'total_capital': self.total_capital,
            'base_position_value': self.base_position_value,
            'tier_counts': tier_counts,
            'tier_values': tier_values,
            'total_deployed': total_deployed,
            'deployment_pct': round(deployment_pct, 2),
            'max_trades': self.max_trades
        }
