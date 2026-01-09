"""Direction Balancer for Gap Trading Strategy.

This module provides intelligent position direction balancing that ensures
a mix of LONG and SHORT positions, with dynamic weighting based on SPY's
opening behavior relative to ATR.

Features:
- Default 50/50 split between LONG and SHORT positions
- SPY-based dynamic weighting (30-70% range based on SPY gap/ATR ratio)
- Minimum guarantee - never 0 in either direction if signals exist
- Priority score preservation within each direction's allocation

Created: 2026-01-08
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
    """Result of direction balancing operation."""
    signals: List[Dict]
    long_count: int
    short_count: int
    target_long_count: int
    target_short_count: int
    market_bias: MarketBias
    spy_gap_pct: float
    spy_atr_pct: float
    spy_gap_atr_ratio: float
    long_pct: float
    short_pct: float
    minimum_applied: bool
    shortfall_filled: bool


# Default configuration
DEFAULT_CONFIG = {
    "enabled": True,
    "default_long_pct": 50,
    "spy_thresholds": {
        "strong_bullish": 1.0,
        "bullish": 0.5,
        "bearish": -0.5,
        "strong_bearish": -1.0
    },
    "allocations": {
        "strong_bullish": {"long": 70, "short": 30},
        "bullish": {"long": 60, "short": 40},
        "neutral": {"long": 50, "short": 50},
        "bearish": {"long": 40, "short": 60},
        "strong_bearish": {"long": 30, "short": 70}
    },
    "min_per_direction_pct": 12.5  # Minimum 12.5% in each direction
}


class DirectionBalancer:
    """Balances LONG/SHORT signal selection based on SPY behavior.

    This class ensures portfolio diversification by maintaining a mix of
    long and short positions, with dynamic allocation based on market
    conditions indicated by SPY's opening gap relative to its ATR.

    Example:
        >>> balancer = DirectionBalancer()
        >>> result = balancer.balance_signals(
        ...     signals=all_signals,
        ...     max_positions=8,
        ...     spy_gap_pct=1.5,
        ...     spy_atr_pct=2.0
        ... )
        >>> print(f"Selected: {result.long_count} LONG, {result.short_count} SHORT")
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize DirectionBalancer.

        Args:
            config: Configuration dictionary. If None, uses defaults.
                - enabled: Whether balancing is active (default: True)
                - default_long_pct: Default long allocation % (default: 50)
                - spy_thresholds: Thresholds for market bias classification
                - allocations: Long/short percentages for each bias level
                - min_per_direction_pct: Minimum % in each direction
        """
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.enabled = self.config.get("enabled", True)
        self.default_long_pct = self.config.get("default_long_pct", 50)
        self.thresholds = self.config.get("spy_thresholds", DEFAULT_CONFIG["spy_thresholds"])
        self.allocations = self.config.get("allocations", DEFAULT_CONFIG["allocations"])
        self.min_per_direction_pct = self.config.get("min_per_direction_pct", 12.5)

        logger.debug(f"DirectionBalancer initialized: enabled={self.enabled}, "
                    f"default_long_pct={self.default_long_pct}")

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

    def get_allocation(self, bias: MarketBias, max_positions: int) -> Tuple[int, int, float, float]:
        """Get (long_count, short_count) based on bias.

        Args:
            bias: Market bias from SPY analysis
            max_positions: Maximum number of positions to take

        Returns:
            Tuple of (long_count, short_count, long_pct, short_pct)
        """
        alloc = self.allocations.get(bias.value, {"long": 50, "short": 50})
        long_pct = alloc["long"]
        short_pct = alloc["short"]

        # Calculate raw counts
        long_count = round(max_positions * long_pct / 100)
        short_count = max_positions - long_count

        # Apply minimum guarantee
        min_per_dir = max(1, int(max_positions * self.min_per_direction_pct / 100))

        if long_count < min_per_dir and short_count > min_per_dir:
            long_count = min_per_dir
            short_count = max_positions - long_count
            logger.info(f"Applied minimum guarantee: adjusted to {long_count} LONG, {short_count} SHORT")
        elif short_count < min_per_dir and long_count > min_per_dir:
            short_count = min_per_dir
            long_count = max_positions - short_count
            logger.info(f"Applied minimum guarantee: adjusted to {long_count} LONG, {short_count} SHORT")

        return long_count, short_count, long_pct, short_pct

    def balance_signals(
        self,
        signals: List[Dict],
        max_positions: int,
        spy_gap_pct: float = 0.0,
        spy_atr_pct: float = 2.0,
        signal_type_key: str = "signal_type",
        priority_key: str = "priority_score"
    ) -> BalanceResult:
        """Select balanced signals from pool.

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
        if not self.enabled:
            # Fall back to pure priority sort
            logger.info("Direction balancing disabled, using pure priority sort")
            sorted_signals = sorted(signals, key=lambda x: x.get(priority_key, 0), reverse=True)
            selected = sorted_signals[:max_positions]

            long_count = sum(1 for s in selected if s.get(signal_type_key) == "BUY")
            short_count = len(selected) - long_count

            return BalanceResult(
                signals=selected,
                long_count=long_count,
                short_count=short_count,
                target_long_count=0,
                target_short_count=0,
                market_bias=MarketBias.NEUTRAL,
                spy_gap_pct=spy_gap_pct,
                spy_atr_pct=spy_atr_pct,
                spy_gap_atr_ratio=0.0,
                long_pct=0.0,
                short_pct=0.0,
                minimum_applied=False,
                shortfall_filled=False
            )

        # 1. Separate signals by direction
        longs = [s for s in signals if s.get(signal_type_key) == "BUY"]
        shorts = [s for s in signals if s.get(signal_type_key) == "SELL_SHORT"]

        logger.info(f"Signal pool: {len(longs)} LONG, {len(shorts)} SHORT (total: {len(signals)})")

        # 2. Sort each pool by priority score
        longs.sort(key=lambda x: x.get(priority_key, 0), reverse=True)
        shorts.sort(key=lambda x: x.get(priority_key, 0), reverse=True)

        # 3. Calculate market bias and allocation
        market_bias, gap_atr_ratio = self.calculate_spy_bias(spy_gap_pct, spy_atr_pct)
        target_long, target_short, long_pct, short_pct = self.get_allocation(market_bias, max_positions)

        logger.info(f"Target allocation: {target_long} LONG ({long_pct}%), {target_short} SHORT ({short_pct}%)")

        # 4. Select from each pool
        selected_longs = longs[:target_long]
        selected_shorts = shorts[:target_short]

        minimum_applied = False
        shortfall_filled = False

        # 5. Handle shortfall - if not enough signals in one direction
        actual_long = len(selected_longs)
        actual_short = len(selected_shorts)

        if actual_long < target_long:
            # Not enough longs, fill with more shorts
            shortfall = target_long - actual_long
            additional_shorts = shorts[target_short:target_short + shortfall]
            selected_shorts.extend(additional_shorts)
            shortfall_filled = True
            logger.info(f"Long shortfall: {shortfall}, filled with additional shorts")

        if actual_short < target_short:
            # Not enough shorts, fill with more longs
            shortfall = target_short - actual_short
            additional_longs = longs[target_long:target_long + shortfall]
            selected_longs.extend(additional_longs)
            shortfall_filled = True
            logger.info(f"Short shortfall: {shortfall}, filled with additional longs")

        # 6. Apply minimum guarantee (if both directions have signals)
        min_per_dir = max(1, int(max_positions * self.min_per_direction_pct / 100))

        if len(longs) > 0 and len(selected_longs) == 0 and len(shorts) >= max_positions:
            # We have longs but didn't select any - force at least minimum
            selected_longs = longs[:min_per_dir]
            selected_shorts = selected_shorts[:max_positions - min_per_dir]
            minimum_applied = True
            logger.warning(f"Forced minimum {min_per_dir} LONG positions")

        if len(shorts) > 0 and len(selected_shorts) == 0 and len(longs) >= max_positions:
            # We have shorts but didn't select any - force at least minimum
            selected_shorts = shorts[:min_per_dir]
            selected_longs = selected_longs[:max_positions - min_per_dir]
            minimum_applied = True
            logger.warning(f"Forced minimum {min_per_dir} SHORT positions")

        # 7. Combine and re-sort by priority for tier assignment
        combined = selected_longs + selected_shorts
        combined.sort(key=lambda x: x.get(priority_key, 0), reverse=True)

        # Ensure we don't exceed max positions
        combined = combined[:max_positions]

        final_long_count = sum(1 for s in combined if s.get(signal_type_key) == "BUY")
        final_short_count = len(combined) - final_long_count

        logger.info(f"Final selection: {final_long_count} LONG, {final_short_count} SHORT")

        return BalanceResult(
            signals=combined,
            long_count=final_long_count,
            short_count=final_short_count,
            target_long_count=target_long,
            target_short_count=target_short,
            market_bias=market_bias,
            spy_gap_pct=spy_gap_pct,
            spy_atr_pct=spy_atr_pct,
            spy_gap_atr_ratio=gap_atr_ratio,
            long_pct=long_pct,
            short_pct=short_pct,
            minimum_applied=minimum_applied,
            shortfall_filled=shortfall_filled
        )

    def get_balance_summary(self, result: BalanceResult) -> str:
        """Generate human-readable summary of balancing result.

        Args:
            result: BalanceResult from balance_signals()

        Returns:
            Formatted summary string
        """
        lines = [
            f"Direction Balance Summary:",
            f"  SPY Gap: {result.spy_gap_pct:+.2f}% (ATR ratio: {result.spy_gap_atr_ratio:+.2f})",
            f"  Market Bias: {result.market_bias.value.replace('_', ' ').title()}",
            f"  Target Allocation: {result.long_pct:.0f}% LONG / {result.short_pct:.0f}% SHORT",
            f"  Actual Selection: {result.long_count} LONG, {result.short_count} SHORT",
        ]

        if result.minimum_applied:
            lines.append("  Note: Minimum direction guarantee applied")
        if result.shortfall_filled:
            lines.append("  Note: Shortfall filled from opposite direction")

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
