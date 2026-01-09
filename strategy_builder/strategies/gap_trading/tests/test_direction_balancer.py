"""Unit tests for DirectionBalancer.

Tests the direction balancing functionality for gap trading signals.
"""

import pytest
from typing import List, Dict
from strategy_builder.strategies.gap_trading.direction_balancer import (
    DirectionBalancer,
    MarketBias,
    BalanceResult,
    DEFAULT_CONFIG,
    create_balancer_from_config,
)


def create_test_signals(long_count: int, short_count: int) -> List[Dict]:
    """Create test signals with given counts."""
    signals = []

    # Create LONG (BUY) signals with descending priority
    for i in range(long_count):
        signals.append({
            'symbol': f'LONG_{i+1}',
            'signal_type': 'BUY',
            'priority_score': 100 - i,
            'gap_pct': 2.0 + i * 0.1,
        })

    # Create SHORT (SELL_SHORT) signals with descending priority
    for i in range(short_count):
        signals.append({
            'symbol': f'SHORT_{i+1}',
            'signal_type': 'SELL_SHORT',
            'priority_score': 95 - i,
            'gap_pct': -2.0 - i * 0.1,
        })

    return signals


class TestMarketBiasCalculation:
    """Tests for SPY-based market bias calculation."""

    def test_neutral_market_near_zero_ratio(self):
        """SPY gap near zero should give neutral bias."""
        balancer = DirectionBalancer()

        # Small gap relative to ATR
        bias, ratio = balancer.calculate_spy_bias(spy_gap_pct=0.2, spy_atr_pct=2.0)

        assert bias == MarketBias.NEUTRAL
        assert ratio == pytest.approx(0.1, abs=0.01)

    def test_bullish_market_positive_ratio(self):
        """SPY gap > 0.5 ATR should give bullish bias."""
        balancer = DirectionBalancer()

        # Gap = 1.5%, ATR = 2% => ratio = 0.75
        bias, ratio = balancer.calculate_spy_bias(spy_gap_pct=1.5, spy_atr_pct=2.0)

        assert bias == MarketBias.BULLISH
        assert ratio == pytest.approx(0.75, abs=0.01)

    def test_strong_bullish_market(self):
        """SPY gap > 1.0 ATR should give strong bullish bias."""
        balancer = DirectionBalancer()

        # Gap = 2.5%, ATR = 2% => ratio = 1.25
        bias, ratio = balancer.calculate_spy_bias(spy_gap_pct=2.5, spy_atr_pct=2.0)

        assert bias == MarketBias.STRONG_BULLISH
        assert ratio == pytest.approx(1.25, abs=0.01)

    def test_bearish_market_negative_ratio(self):
        """SPY gap < -0.5 ATR should give bearish bias."""
        balancer = DirectionBalancer()

        # Gap = -1.5%, ATR = 2% => ratio = -0.75
        bias, ratio = balancer.calculate_spy_bias(spy_gap_pct=-1.5, spy_atr_pct=2.0)

        assert bias == MarketBias.BEARISH
        assert ratio == pytest.approx(-0.75, abs=0.01)

    def test_strong_bearish_market(self):
        """SPY gap < -1.0 ATR should give strong bearish bias."""
        balancer = DirectionBalancer()

        # Gap = -2.5%, ATR = 2% => ratio = -1.25
        bias, ratio = balancer.calculate_spy_bias(spy_gap_pct=-2.5, spy_atr_pct=2.0)

        assert bias == MarketBias.STRONG_BEARISH
        assert ratio == pytest.approx(-1.25, abs=0.01)

    def test_invalid_atr_defaults_to_neutral(self):
        """Invalid ATR (0 or negative) should default to neutral."""
        balancer = DirectionBalancer()

        bias, ratio = balancer.calculate_spy_bias(spy_gap_pct=1.5, spy_atr_pct=0.0)

        assert bias == MarketBias.NEUTRAL
        assert ratio == 0.0


class TestAllocation:
    """Tests for position allocation calculation."""

    def test_neutral_50_50_split(self):
        """Neutral market should give 50/50 split."""
        balancer = DirectionBalancer()

        long_count, short_count, long_pct, short_pct = balancer.get_allocation(
            MarketBias.NEUTRAL, max_positions=8
        )

        assert long_count == 4
        assert short_count == 4
        assert long_pct == 50
        assert short_pct == 50

    def test_bullish_60_40_split(self):
        """Bullish market should give 60/40 split."""
        balancer = DirectionBalancer()

        long_count, short_count, long_pct, short_pct = balancer.get_allocation(
            MarketBias.BULLISH, max_positions=8
        )

        assert long_count == 5  # 60% of 8 = 4.8, rounded to 5
        assert short_count == 3
        assert long_pct == 60
        assert short_pct == 40

    def test_strong_bullish_70_30_split(self):
        """Strong bullish market should give 70/30 split."""
        balancer = DirectionBalancer()

        long_count, short_count, long_pct, short_pct = balancer.get_allocation(
            MarketBias.STRONG_BULLISH, max_positions=8
        )

        assert long_count == 6  # 70% of 8 = 5.6, rounded to 6
        assert short_count == 2
        assert long_pct == 70
        assert short_pct == 30

    def test_bearish_40_60_split(self):
        """Bearish market should give 40/60 split (more shorts)."""
        balancer = DirectionBalancer()

        long_count, short_count, long_pct, short_pct = balancer.get_allocation(
            MarketBias.BEARISH, max_positions=8
        )

        assert long_count == 3
        assert short_count == 5
        assert long_pct == 40
        assert short_pct == 60

    def test_minimum_guarantee_applied(self):
        """Minimum guarantee should ensure at least 1 per direction."""
        balancer = DirectionBalancer()

        # With 8 positions and 70/30 split, short would get 2
        # But minimum is 12.5% = 1, so it stays at 2
        long_count, short_count, _, _ = balancer.get_allocation(
            MarketBias.STRONG_BULLISH, max_positions=8
        )

        assert short_count >= 1  # Minimum guarantee


class TestBalanceSignals:
    """Tests for the main balance_signals() method."""

    def test_neutral_market_50_50_split(self):
        """With neutral SPY gap, should select 50/50 mix."""
        balancer = DirectionBalancer()
        signals = create_test_signals(long_count=10, short_count=10)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=0.0,
            spy_atr_pct=2.0
        )

        assert result.long_count == 4
        assert result.short_count == 4
        assert len(result.signals) == 8
        assert result.market_bias == MarketBias.NEUTRAL

    def test_bullish_market_favors_longs(self):
        """SPY gap > 0.5 ATR should favor long positions."""
        balancer = DirectionBalancer()
        signals = create_test_signals(long_count=10, short_count=10)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=1.5,  # 0.75 ratio (bullish)
            spy_atr_pct=2.0
        )

        assert result.long_count == 5  # 60% = 5 longs
        assert result.short_count == 3  # 40% = 3 shorts
        assert result.market_bias == MarketBias.BULLISH

    def test_bearish_market_favors_shorts(self):
        """SPY gap < -0.5 ATR should favor short positions."""
        balancer = DirectionBalancer()
        signals = create_test_signals(long_count=10, short_count=10)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=-1.5,  # -0.75 ratio (bearish)
            spy_atr_pct=2.0
        )

        assert result.long_count == 3  # 40% = 3 longs
        assert result.short_count == 5  # 60% = 5 shorts
        assert result.market_bias == MarketBias.BEARISH

    def test_shortfall_handling_not_enough_longs(self):
        """If not enough longs, fill with additional shorts."""
        balancer = DirectionBalancer()
        # Only 2 longs, but target is 4 (neutral market)
        signals = create_test_signals(long_count=2, short_count=10)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=0.0,
            spy_atr_pct=2.0
        )

        # Should get 2 longs (all available) + 6 shorts (to fill 8)
        assert result.long_count == 2
        assert result.short_count == 6
        assert len(result.signals) == 8
        assert result.shortfall_filled == True

    def test_shortfall_handling_not_enough_shorts(self):
        """If not enough shorts, fill with additional longs."""
        balancer = DirectionBalancer()
        # Only 2 shorts, but target is 4 (neutral market)
        signals = create_test_signals(long_count=10, short_count=2)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=0.0,
            spy_atr_pct=2.0
        )

        # Should get 6 longs (to fill 8) + 2 shorts (all available)
        assert result.long_count == 6
        assert result.short_count == 2
        assert len(result.signals) == 8
        assert result.shortfall_filled == True

    def test_minimum_guarantee_when_all_one_direction(self):
        """Should force minimum positions when one direction dominates."""
        balancer = DirectionBalancer()
        # 10 longs with high priority, 10 shorts with lower priority
        signals = create_test_signals(long_count=10, short_count=10)
        # Make all longs have very high priority
        for s in signals:
            if s['signal_type'] == 'BUY':
                s['priority_score'] = 200 + s['priority_score']

        # Even with strong bullish bias, should not exclude all shorts
        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=2.5,  # Strong bullish
            spy_atr_pct=2.0
        )

        # Strong bullish: 70% long (6), 30% short (2)
        assert result.short_count >= 1  # Minimum guarantee

    def test_only_long_signals_available(self):
        """Should take all longs when no shorts available."""
        balancer = DirectionBalancer()
        signals = create_test_signals(long_count=10, short_count=0)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=0.0,
            spy_atr_pct=2.0
        )

        assert result.long_count == 8
        assert result.short_count == 0

    def test_only_short_signals_available(self):
        """Should take all shorts when no longs available."""
        balancer = DirectionBalancer()
        signals = create_test_signals(long_count=0, short_count=10)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=0.0,
            spy_atr_pct=2.0
        )

        assert result.long_count == 0
        assert result.short_count == 8

    def test_priority_preserved_within_direction(self):
        """Highest priority signals selected within each direction."""
        balancer = DirectionBalancer()
        signals = create_test_signals(long_count=5, short_count=5)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=4,
            spy_gap_pct=0.0,
            spy_atr_pct=2.0
        )

        # Should get 2 longs and 2 shorts (50/50 neutral)
        longs_selected = [s for s in result.signals if s['signal_type'] == 'BUY']
        shorts_selected = [s for s in result.signals if s['signal_type'] == 'SELL_SHORT']

        # Check that highest priority longs were selected
        assert longs_selected[0]['symbol'] == 'LONG_1'  # Highest priority
        assert longs_selected[1]['symbol'] == 'LONG_2'

        # Check that highest priority shorts were selected
        assert shorts_selected[0]['symbol'] == 'SHORT_1'
        assert shorts_selected[1]['symbol'] == 'SHORT_2'


class TestBalancerConfiguration:
    """Tests for configuration options."""

    def test_disabled_balancer_uses_pure_priority(self):
        """When disabled, should use pure priority sort without direction balancing."""
        config = {'enabled': False}
        balancer = DirectionBalancer(config)

        signals = create_test_signals(long_count=3, short_count=7)
        # Make shorts have higher priority than longs
        for i, s in enumerate(signals):
            if s['signal_type'] == 'SELL_SHORT':
                s['priority_score'] = 200 - i

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=0.0,
            spy_atr_pct=2.0
        )

        # All 8 positions should be shorts (they have higher priority)
        assert result.short_count == 7  # All 7 shorts
        assert result.long_count == 1  # Only 1 long to fill remaining

    def test_custom_thresholds(self):
        """Should respect custom SPY thresholds."""
        config = {
            'enabled': True,
            'spy_thresholds': {
                'strong_bullish': 2.0,  # Higher threshold
                'bullish': 1.0,
                'bearish': -1.0,
                'strong_bearish': -2.0
            },
            'allocations': DEFAULT_CONFIG['allocations']
        }
        balancer = DirectionBalancer(config)

        # With default thresholds, 1.5/2.0 = 0.75 would be bullish
        # With custom thresholds, 0.75 < 1.0 so it's neutral
        bias, ratio = balancer.calculate_spy_bias(spy_gap_pct=1.5, spy_atr_pct=2.0)

        assert bias == MarketBias.NEUTRAL

    def test_custom_allocations(self):
        """Should respect custom allocation percentages."""
        config = {
            'enabled': True,
            'spy_thresholds': DEFAULT_CONFIG['spy_thresholds'],
            'allocations': {
                'strong_bullish': {'long': 80, 'short': 20},
                'bullish': {'long': 70, 'short': 30},
                'neutral': {'long': 50, 'short': 50},
                'bearish': {'long': 30, 'short': 70},
                'strong_bearish': {'long': 20, 'short': 80}
            }
        }
        balancer = DirectionBalancer(config)

        # With custom 80/20 allocation for strong bullish
        long_count, short_count, _, _ = balancer.get_allocation(
            MarketBias.STRONG_BULLISH, max_positions=10
        )

        assert long_count == 8  # 80% of 10
        assert short_count == 2


class TestFactoryFunction:
    """Tests for create_balancer_from_config factory."""

    def test_create_from_airflow_config(self):
        """Should create balancer from Airflow-style config."""
        config = {
            'account_value': 100000,
            'max_trades_per_day': 8,
            'direction_balancing': {
                'enabled': True,
                'default_long_pct': 50,
                'min_per_direction_pct': 15
            }
        }

        balancer = create_balancer_from_config(config)

        assert balancer.enabled == True
        assert balancer.default_long_pct == 50
        assert balancer.min_per_direction_pct == 15

    def test_create_with_missing_direction_config(self):
        """Should use defaults when direction_balancing config is missing."""
        config = {
            'account_value': 100000,
            'max_trades_per_day': 8
            # No direction_balancing key
        }

        balancer = create_balancer_from_config(config)

        assert balancer.enabled == True  # Default
        assert balancer.default_long_pct == 50  # Default


class TestBalanceSummary:
    """Tests for the get_balance_summary() method."""

    def test_summary_format(self):
        """Should generate readable summary."""
        balancer = DirectionBalancer()
        signals = create_test_signals(long_count=10, short_count=10)

        result = balancer.balance_signals(
            signals=signals,
            max_positions=8,
            spy_gap_pct=1.5,
            spy_atr_pct=2.0
        )

        summary = balancer.get_balance_summary(result)

        assert 'Direction Balance Summary' in summary
        assert 'SPY Gap' in summary
        assert 'Market Bias' in summary
        assert 'LONG' in summary
        assert 'SHORT' in summary


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
