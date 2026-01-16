"""Direction Balancer for Gap Trading Strategy.

This module provides market bias calculation for gap trading signals.
Market direction is used to boost/penalize signal priority scores,
with final selection being purely merit-based (no allocation quotas).

Features (Updated 2026-01-15):
- SPY-based market bias calculation (BULLISH, BEARISH, NEUTRAL)
- Market bias passed to priority scoring for +15/-10 adjustments
- Merit-based selection: top N signals by priority score
- NO forced allocation splitting (removed 30-70% quotas)
- NO minimum guarantees per direction (removed)

Created: 2026-01-08
Updated: 2026-01-15 - Removed allocation splitting, switched to merit-based selection
"""

import logging
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class MarketBias(Enum):
    """Market bias based on SPY opening behavior."""
    STRONG_BULLISH = "strong_bullish"
    BULLISH = "bullish"
    NEUTRAL = "neutral"
    BEARISH = "bearish"
    STRONG_BEARISH = "strong_bearish"


@dataclass
class BalanceResult:
    """Result of direction balancing operation.

    Updated 2026-01-15: Simplified for merit-based selection.
    Removed: target counts, percentages, minimum_applied, shortfall_filled.
    """
    signals: List[Dict]
    long_count: int
    short_count: int
    market_bias: MarketBias
    spy_gap_pct: float
    spy_atr_pct: float
    spy_gap_atr_ratio: float
    # Legacy fields for backwards compatibility (always set to defaults)
    target_long_count: int = 0
    target_short_count: int = 0
    long_pct: float = 0.0
    short_pct: float = 0.0
    minimum_applied: bool = False
    shortfall_filled: bool = False


# Default configuration (simplified for merit-based selection)
DEFAULT_CONFIG = {
    "enabled": True,
    "spy_thresholds": {
        "strong_bullish": 1.0,
        "bullish": 0.5,
        "bearish": -0.5,
        "strong_bearish": -1.0
    }
    # Removed: allocations, min_per_direction_pct (no longer used)
}


class DirectionBalancer:
    """Provides market bias calculation for gap trading signals.

    Updated 2026-01-15: Simplified to merit-based selection.
    - Calculates market bias from SPY gap/ATR ratio
    - Market bias is used for priority score adjustments (done in indicators.py)
    - Selection is purely merit-based: top N signals by priority score
    - NO allocation quotas or minimum guarantees

    Example:
        >>> balancer = DirectionBalancer()
        >>> result = balancer.balance_signals(
        ...     signals=all_signals,
        ...     max_positions=8,
        ...     spy_gap_pct=1.5,
        ...     spy_atr_pct=2.0
        ... )
        >>> print(f"Selected: {result.long_count} LONG, {result.short_count} SHORT")
        >>> print(f"Market bias: {result.market_bias.value}")
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize DirectionBalancer.

        Args:
            config: Configuration dictionary. If None, uses defaults.
                - enabled: Whether to calculate market bias (default: True)
                - spy_thresholds: Thresholds for market bias classification
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.enabled = self.config.get("enabled", True)
        self.thresholds = self.config.get("spy_thresholds", DEFAULT_CONFIG["spy_thresholds"])

        logger.debug(f"DirectionBalancer initialized: enabled={self.enabled}, "
                    f"merit-based selection (no allocation quotas)")

    def calculate_spy_bias(self, spy_gap_pct: float, spy_atr_pct: float) -> Tuple[MarketBias, float]:
        """Determine market bias from SPY opening.

        Args:
            spy_gap_pct: SPY gap percentage (positive = gap up, negative = gap down)
            spy_atr_pct: SPY ATR as percentage of price

        Returns:
            Tuple of (MarketBias enum, gap/ATR ratio)
        """
        if spy_atr_pct <= 0:
            logger.warning(f"Invalid SPY ATR: {spy_atr_pct}, defaulting to neutral")
            return MarketBias.NEUTRAL, 0.0

        ratio = spy_gap_pct / spy_atr_pct

        if ratio >= self.thresholds["strong_bullish"]:
            bias = MarketBias.STRONG_BULLISH
        elif ratio >= self.thresholds["bullish"]:
            bias = MarketBias.BULLISH
        elif ratio <= self.thresholds["strong_bearish"]:
            bias = MarketBias.STRONG_BEARISH
        elif ratio <= self.thresholds["bearish"]:
            bias = MarketBias.BEARISH
        else:
            bias = MarketBias.NEUTRAL

        logger.info(f"SPY bias: gap={spy_gap_pct:.2f}%, ATR={spy_atr_pct:.2f}%, "
                   f"ratio={ratio:.2f}, bias={bias.value}")

        return bias, ratio

    def balance_signals(
        self,
        signals: List[Dict],
        max_positions: int,
        spy_gap_pct: float = 0.0,
        spy_atr_pct: float = 2.0,
        signal_type_key: str = "signal_type",
        priority_key: str = "priority_score"
    ) -> BalanceResult:
        """Select top signals by priority score (merit-based).

        Updated 2026-01-15: Simplified to pure merit-based selection.
        Market bias is calculated but only used for metadata/reporting.
        Priority scores should already include market direction boost.

        Args:
            signals: List of signal dictionaries
            max_positions: Maximum positions to take
            spy_gap_pct: SPY gap percentage
            spy_atr_pct: SPY ATR percentage
            signal_type_key: Key in signal dict for signal type
            priority_key: Key in signal dict for priority score

        Returns:
            BalanceResult with selected signals and metadata
        """
        # Calculate market bias (for metadata and reporting)
        market_bias, gap_atr_ratio = self.calculate_spy_bias(spy_gap_pct, spy_atr_pct)

        # Count available signals by direction
        longs = [s for s in signals if s.get(signal_type_key) == "BUY"]
        shorts = [s for s in signals if s.get(signal_type_key) == "SELL_SHORT"]

        logger.info(f"Signal pool: {len(longs)} LONG, {len(shorts)} SHORT (total: {len(signals)})")
        logger.info(f"Market bias: {market_bias.value} (SPY gap/ATR ratio: {gap_atr_ratio:.2f})")

        # Sort ALL signals by priority score (already includes market direction boost)
        sorted_signals = sorted(signals, key=lambda x: x.get(priority_key, 0), reverse=True)

        # Select top N signals - pure merit-based, no quotas
        selected = sorted_signals[:max_positions]

        # Count final selection by direction
        final_long_count = sum(1 for s in selected if s.get(signal_type_key) == "BUY")
        final_short_count = len(selected) - final_long_count

        logger.info(f"Merit-based selection: {final_long_count} LONG, {final_short_count} SHORT")

        return BalanceResult(
            signals=selected,
            long_count=final_long_count,
            short_count=final_short_count,
            market_bias=market_bias,
            spy_gap_pct=spy_gap_pct,
            spy_atr_pct=spy_atr_pct,
            spy_gap_atr_ratio=gap_atr_ratio
        )

    def get_balance_summary(self, result: BalanceResult) -> str:
        """Generate human-readable summary of balancing result.

        Args:
            result: BalanceResult from balance_signals()

        Returns:
            Formatted summary string
        """
        lines = [
            f"Selection Summary (Merit-Based):",
            f"  SPY Gap: {result.spy_gap_pct:+.2f}% (ATR ratio: {result.spy_gap_atr_ratio:+.2f})",
            f"  Market Bias: {result.market_bias.value.replace('_', ' ').title()}",
            f"  Selected: {result.long_count} LONG, {result.short_count} SHORT",
            f"  Note: Selection by priority score (no allocation quotas)",
        ]

        return "\n".join(lines)


def create_balancer_from_config(config: Dict) -> DirectionBalancer:
    """Factory function to create DirectionBalancer from Airflow config.

    Args:
        config: gap_trading_config from Airflow Variable

    Returns:
        Configured DirectionBalancer instance
    """
    direction_config = config.get("direction_balancing", {})
    return DirectionBalancer(direction_config)
