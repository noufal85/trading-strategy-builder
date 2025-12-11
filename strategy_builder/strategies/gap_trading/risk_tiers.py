"""Gap Trading Risk Tier Classification.

Classifies stocks into risk tiers based on volatility metrics
and provides position sizing and stop-loss multipliers.

Risk Tiers:
- LOW: ATR < 2%, Position 100%, Stop 1.0 ATR
- MEDIUM: ATR 2-4%, Position 75%, Stop 1.5 ATR
- HIGH: ATR > 4%, Position 50%, Stop 2.0 ATR
"""

from enum import Enum
from typing import Dict, Optional, Tuple, NamedTuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class RiskTier(Enum):
    """Risk tier classification for position sizing."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    UNKNOWN = "UNKNOWN"

    def __str__(self) -> str:
        return self.value


@dataclass
class RiskTierConfig:
    """Configuration for a risk tier.

    Attributes:
        tier: Risk tier enum
        atr_min: Minimum ATR% for this tier (inclusive)
        atr_max: Maximum ATR% for this tier (exclusive)
        position_multiplier: Position size multiplier (1.0 = 100%)
        stop_atr_multiplier: Stop-loss distance in ATR multiples
        description: Human-readable description
    """
    tier: RiskTier
    atr_min: float
    atr_max: float
    position_multiplier: float
    stop_atr_multiplier: float
    description: str


class RiskParameters(NamedTuple):
    """Risk parameters for a classified stock.

    Attributes:
        tier: Risk tier classification
        position_multiplier: Position size multiplier (0.5-1.0)
        stop_atr_multiplier: Stop distance in ATR multiples
        max_loss_pct: Maximum expected loss percentage
    """
    tier: RiskTier
    position_multiplier: float
    stop_atr_multiplier: float
    max_loss_pct: float


class RiskTierClassifier:
    """Classifies stocks into risk tiers for position sizing.

    Risk Tier Definitions:
    ┌──────────┬──────────┬──────────┬──────────┐
    │   TIER   │  ATR %   │ Position │   Stop   │
    ├──────────┼──────────┼──────────┼──────────┤
    │   LOW    │   < 2%   │   100%   │  1.0 ATR │
    │  MEDIUM  │   2-4%   │    75%   │  1.5 ATR │
    │   HIGH   │   > 4%   │    50%   │  2.0 ATR │
    └──────────┴──────────┴──────────┴──────────┘

    The logic:
    - Low volatility stocks: Full position size, tight stops
    - Medium volatility: Reduced position, moderate stops
    - High volatility: Half position, wide stops to avoid whipsaws

    Attributes:
        tiers: Dict of tier configurations
    """

    # Default tier configurations
    DEFAULT_TIERS: Dict[RiskTier, RiskTierConfig] = {
        RiskTier.LOW: RiskTierConfig(
            tier=RiskTier.LOW,
            atr_min=0.0,
            atr_max=2.0,
            position_multiplier=1.0,
            stop_atr_multiplier=1.0,
            description="Low volatility: Full position, tight stop"
        ),
        RiskTier.MEDIUM: RiskTierConfig(
            tier=RiskTier.MEDIUM,
            atr_min=2.0,
            atr_max=4.0,
            position_multiplier=0.75,
            stop_atr_multiplier=1.5,
            description="Medium volatility: 75% position, moderate stop"
        ),
        RiskTier.HIGH: RiskTierConfig(
            tier=RiskTier.HIGH,
            atr_min=4.0,
            atr_max=float('inf'),
            position_multiplier=0.5,
            stop_atr_multiplier=2.0,
            description="High volatility: 50% position, wide stop"
        ),
        RiskTier.UNKNOWN: RiskTierConfig(
            tier=RiskTier.UNKNOWN,
            atr_min=float('-inf'),
            atr_max=float('inf'),
            position_multiplier=0.25,  # Conservative default
            stop_atr_multiplier=2.0,
            description="Unknown volatility: Conservative position"
        )
    }

    def __init__(
        self,
        low_threshold: float = 2.0,
        high_threshold: float = 4.0,
        custom_tiers: Optional[Dict[RiskTier, RiskTierConfig]] = None
    ):
        """Initialize RiskTierClassifier.

        Args:
            low_threshold: ATR% threshold between LOW and MEDIUM (default 2.0)
            high_threshold: ATR% threshold between MEDIUM and HIGH (default 4.0)
            custom_tiers: Optional custom tier configurations
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

        if custom_tiers:
            self.tiers = custom_tiers
        else:
            # Update default tiers with custom thresholds
            self.tiers = self.DEFAULT_TIERS.copy()
            self.tiers[RiskTier.LOW] = RiskTierConfig(
                tier=RiskTier.LOW,
                atr_min=0.0,
                atr_max=low_threshold,
                position_multiplier=1.0,
                stop_atr_multiplier=1.0,
                description=f"Low volatility (<{low_threshold}%): Full position, tight stop"
            )
            self.tiers[RiskTier.MEDIUM] = RiskTierConfig(
                tier=RiskTier.MEDIUM,
                atr_min=low_threshold,
                atr_max=high_threshold,
                position_multiplier=0.75,
                stop_atr_multiplier=1.5,
                description=f"Medium volatility ({low_threshold}-{high_threshold}%): 75% position"
            )
            self.tiers[RiskTier.HIGH] = RiskTierConfig(
                tier=RiskTier.HIGH,
                atr_min=high_threshold,
                atr_max=float('inf'),
                position_multiplier=0.5,
                stop_atr_multiplier=2.0,
                description=f"High volatility (>{high_threshold}%): 50% position, wide stop"
            )

    def classify(
        self,
        atr_pct: Optional[float],
        volatility: Optional[float] = None
    ) -> RiskTier:
        """Classify a stock into a risk tier.

        Primary classification is based on ATR%. Volatility can be used
        as a secondary factor for edge cases.

        Args:
            atr_pct: ATR as percentage of price
            volatility: Optional annualized volatility (not currently used)

        Returns:
            RiskTier enum value
        """
        if atr_pct is None:
            return RiskTier.UNKNOWN

        if atr_pct < self.low_threshold:
            return RiskTier.LOW
        elif atr_pct < self.high_threshold:
            return RiskTier.MEDIUM
        else:
            return RiskTier.HIGH

    def get_position_multiplier(self, tier: RiskTier) -> float:
        """Get position size multiplier for a risk tier.

        Args:
            tier: Risk tier

        Returns:
            Position multiplier (0.25-1.0)
        """
        config = self.tiers.get(tier, self.tiers[RiskTier.UNKNOWN])
        return config.position_multiplier

    def get_stop_atr_multiplier(self, tier: RiskTier) -> float:
        """Get stop-loss ATR multiplier for a risk tier.

        Args:
            tier: Risk tier

        Returns:
            Stop-loss multiplier (1.0-2.0)
        """
        config = self.tiers.get(tier, self.tiers[RiskTier.UNKNOWN])
        return config.stop_atr_multiplier

    def get_risk_parameters(
        self,
        atr_pct: Optional[float],
        volatility: Optional[float] = None
    ) -> RiskParameters:
        """Get complete risk parameters for a stock.

        Args:
            atr_pct: ATR as percentage of price
            volatility: Optional annualized volatility

        Returns:
            RiskParameters with tier, multipliers, and max loss
        """
        tier = self.classify(atr_pct, volatility)
        config = self.tiers.get(tier, self.tiers[RiskTier.UNKNOWN])

        # Calculate maximum expected loss
        # Max loss = ATR% * stop multiplier * position multiplier
        atr = atr_pct if atr_pct is not None else 3.0  # Default assumption
        max_loss_pct = atr * config.stop_atr_multiplier * config.position_multiplier

        return RiskParameters(
            tier=tier,
            position_multiplier=config.position_multiplier,
            stop_atr_multiplier=config.stop_atr_multiplier,
            max_loss_pct=max_loss_pct
        )

    def get_tier_config(self, tier: RiskTier) -> RiskTierConfig:
        """Get full configuration for a risk tier.

        Args:
            tier: Risk tier

        Returns:
            RiskTierConfig for the tier
        """
        return self.tiers.get(tier, self.tiers[RiskTier.UNKNOWN])

    def calculate_position_size(
        self,
        tier: RiskTier,
        base_position_value: float,
        max_position_value: Optional[float] = None
    ) -> float:
        """Calculate position size based on risk tier.

        Args:
            tier: Risk tier
            base_position_value: Base position value (e.g., from equal weighting)
            max_position_value: Optional maximum position cap

        Returns:
            Adjusted position value
        """
        multiplier = self.get_position_multiplier(tier)
        position_value = base_position_value * multiplier

        if max_position_value is not None:
            position_value = min(position_value, max_position_value)

        return position_value

    def calculate_stop_distance(
        self,
        tier: RiskTier,
        atr: float,
        entry_price: float
    ) -> Tuple[float, float]:
        """Calculate stop-loss distance and price.

        Args:
            tier: Risk tier
            atr: Current ATR value
            entry_price: Entry price for the position

        Returns:
            Tuple of (stop_distance, stop_price)
        """
        multiplier = self.get_stop_atr_multiplier(tier)
        stop_distance = atr * multiplier
        stop_price = entry_price - stop_distance

        return stop_distance, stop_price

    def get_all_tier_info(self) -> Dict[str, Dict]:
        """Get summary information for all risk tiers.

        Returns:
            Dict with tier information
        """
        info = {}
        for tier, config in self.tiers.items():
            if tier == RiskTier.UNKNOWN:
                continue
            info[tier.value] = {
                'atr_range': f"{config.atr_min}% - {config.atr_max}%",
                'position_multiplier': config.position_multiplier,
                'stop_atr_multiplier': config.stop_atr_multiplier,
                'description': config.description
            }
        return info

    def __repr__(self) -> str:
        return (
            f"RiskTierClassifier("
            f"low_threshold={self.low_threshold}, "
            f"high_threshold={self.high_threshold})"
        )


# Convenience function
def classify_risk_tier(atr_pct: Optional[float]) -> str:
    """Quick function to classify a single ATR% value.

    Args:
        atr_pct: ATR as percentage of price

    Returns:
        Risk tier as string ('LOW', 'MEDIUM', 'HIGH', 'UNKNOWN')
    """
    classifier = RiskTierClassifier()
    return str(classifier.classify(atr_pct))


def get_risk_multipliers(atr_pct: Optional[float]) -> Tuple[float, float]:
    """Quick function to get position and stop multipliers.

    Args:
        atr_pct: ATR as percentage of price

    Returns:
        Tuple of (position_multiplier, stop_atr_multiplier)
    """
    classifier = RiskTierClassifier()
    params = classifier.get_risk_parameters(atr_pct)
    return params.position_multiplier, params.stop_atr_multiplier
