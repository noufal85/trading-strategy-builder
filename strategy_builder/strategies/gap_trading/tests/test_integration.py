"""Integration Tests for Gap Trading Strategy.

These tests verify that all components work together correctly.
Run with: pytest test_integration.py -v
"""

import pytest
from datetime import date, datetime, timedelta
from decimal import Decimal
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))


class TestUniverseModule:
    """Tests for universe management."""

    def test_universe_stock_creation(self):
        """Test UniverseStock dataclass creation."""
        from strategy_builder.strategies.gap_trading.universe import UniverseStock

        stock = UniverseStock(
            symbol='AAPL',
            name='Apple Inc.',
            sector='Technology',
            market_cap=3000000000000,
            avg_volume=50000000,
            is_permanent=False,
            is_reference_only=False,
        )

        assert stock.symbol == 'AAPL'
        assert stock.is_tradeable is True

    def test_permanent_symbols(self):
        """Test permanent symbols list."""
        from strategy_builder.strategies.gap_trading.universe import (
            PERMANENT_SYMBOLS,
            REFERENCE_SYMBOLS,
            get_permanent_symbols,
            is_permanent,
            is_reference_only,
        )

        # Check permanent symbols include ETFs
        assert 'SPY' in PERMANENT_SYMBOLS
        assert 'QQQ' in PERMANENT_SYMBOLS
        assert 'GLD' in PERMANENT_SYMBOLS

        # Check reference symbols
        assert 'VIX' in REFERENCE_SYMBOLS

        # Test helper functions
        assert is_permanent('SPY') is True
        assert is_permanent('AAPL') is False
        assert is_reference_only('VIX') is True

        perms = get_permanent_symbols()
        assert len(perms) >= 7


class TestVolatilityModule:
    """Tests for volatility calculations."""

    def test_volatility_metrics_creation(self):
        """Test VolatilityMetrics dataclass."""
        from strategy_builder.strategies.gap_trading.volatility import VolatilityMetrics

        metrics = VolatilityMetrics(
            symbol='AAPL',
            atr_14=2.5,
            atr_percent=1.5,
            adr_percent=2.0,
            volatility_20d=25.0,
            volatility_60d=22.0,
            beta=1.2,
            calculated_at=datetime.now(),
        )

        assert metrics.symbol == 'AAPL'
        assert metrics.atr_14 == 2.5
        assert metrics.atr_percent == 1.5

    def test_volatility_calculator_atr(self):
        """Test ATR calculation."""
        from strategy_builder.strategies.gap_trading.volatility import VolatilityCalculator
        import pandas as pd
        import numpy as np

        # Create sample OHLCV data
        dates = pd.date_range(start='2025-01-01', periods=30, freq='D')
        np.random.seed(42)

        data = pd.DataFrame({
            'date': dates,
            'open': 150 + np.random.randn(30) * 2,
            'high': 152 + np.random.randn(30) * 2,
            'low': 148 + np.random.randn(30) * 2,
            'close': 150 + np.random.randn(30) * 2,
            'volume': 1000000 + np.random.randint(-100000, 100000, 30),
        })

        # Ensure high >= open, close, low
        data['high'] = data[['open', 'high', 'close']].max(axis=1) + 0.5
        data['low'] = data[['open', 'low', 'close']].min(axis=1) - 0.5

        calculator = VolatilityCalculator()
        atr = calculator.calculate_atr(data, period=14)

        assert atr is not None
        assert atr > 0


class TestRiskTiersModule:
    """Tests for risk tier classification."""

    def test_risk_tier_enum(self):
        """Test RiskTier enum."""
        from strategy_builder.strategies.gap_trading.risk_tiers import RiskTier

        assert RiskTier.LOW.value == 'LOW'
        assert RiskTier.MEDIUM.value == 'MEDIUM'
        assert RiskTier.HIGH.value == 'HIGH'

    def test_classify_risk_tier(self):
        """Test risk tier classification."""
        from strategy_builder.strategies.gap_trading.risk_tiers import (
            classify_risk_tier,
            RiskTier,
        )

        # Low volatility
        assert classify_risk_tier(atr_pct=1.5) == RiskTier.LOW

        # Medium volatility
        assert classify_risk_tier(atr_pct=3.0) == RiskTier.MEDIUM

        # High volatility
        assert classify_risk_tier(atr_pct=5.0) == RiskTier.HIGH

    def test_risk_multipliers(self):
        """Test risk tier multipliers."""
        from strategy_builder.strategies.gap_trading.risk_tiers import (
            get_risk_multipliers,
            RiskTier,
        )

        params = get_risk_multipliers(RiskTier.LOW)
        assert params.position_multiplier == 1.0
        assert params.stop_atr_multiplier == 1.0

        params = get_risk_multipliers(RiskTier.MEDIUM)
        assert params.position_multiplier == 0.75
        assert params.stop_atr_multiplier == 1.5

        params = get_risk_multipliers(RiskTier.HIGH)
        assert params.position_multiplier == 0.5
        assert params.stop_atr_multiplier == 2.0


class TestSignalsModule:
    """Tests for gap detection and signal generation."""

    def test_gap_direction_enum(self):
        """Test GapDirection enum."""
        from strategy_builder.strategies.gap_trading.signals import GapDirection

        assert GapDirection.UP.value == 'UP'
        assert GapDirection.DOWN.value == 'DOWN'

    def test_gap_info_creation(self):
        """Test GapInfo dataclass."""
        from strategy_builder.strategies.gap_trading.signals import GapInfo, GapDirection

        gap = GapInfo(
            symbol='AAPL',
            prev_close=150.0,
            open_price=153.0,
            gap_pct=2.0,
            gap_direction=GapDirection.UP,
            gap_dollars=3.0,
        )

        assert gap.symbol == 'AAPL'
        assert gap.gap_pct == 2.0
        assert gap.gap_direction == GapDirection.UP
        assert gap.is_significant(min_gap_pct=1.5) is True
        assert gap.is_significant(min_gap_pct=3.0) is False

    def test_gap_detector(self):
        """Test gap detection logic."""
        from strategy_builder.strategies.gap_trading.signals import (
            GapDetector,
            GapDirection,
        )

        detector = GapDetector(min_gap_pct=1.5, max_gap_pct=10.0)

        # Test gap up detection
        gap = detector.detect_gap(
            symbol='AAPL',
            prev_close=100.0,
            open_price=103.0,
        )

        assert gap is not None
        assert gap.gap_direction == GapDirection.UP
        assert abs(gap.gap_pct - 3.0) < 0.01

        # Test gap down detection
        gap = detector.detect_gap(
            symbol='AAPL',
            prev_close=100.0,
            open_price=97.0,
        )

        assert gap is not None
        assert gap.gap_direction == GapDirection.DOWN
        assert abs(gap.gap_pct - (-3.0)) < 0.01

        # Test no significant gap
        gap = detector.detect_gap(
            symbol='AAPL',
            prev_close=100.0,
            open_price=100.5,
        )

        assert gap is None  # Below threshold

    def test_signal_type_enum(self):
        """Test SignalType enum."""
        from strategy_builder.strategies.gap_trading.signals import SignalType

        assert SignalType.LONG.value == 'LONG'
        assert SignalType.SHORT.value == 'SHORT'
        assert SignalType.NO_TRADE.value == 'NO_TRADE'


class TestPositionSizerModule:
    """Tests for position sizing."""

    def test_position_size_creation(self):
        """Test PositionSize dataclass."""
        from strategy_builder.strategies.gap_trading.position_sizer import PositionSize

        size = PositionSize(
            shares=100,
            position_value=15000.0,
            risk_amount=150.0,
            stop_distance=1.5,
        )

        assert size.shares == 100
        assert size.position_value == 15000.0

    def test_simple_share_calculation(self):
        """Test simple share calculation."""
        from strategy_builder.strategies.gap_trading.position_sizer import (
            calculate_shares_simple,
        )

        # Risk $100, stop $2 away from entry
        shares = calculate_shares_simple(
            account_value=10000,
            risk_pct=1.0,
            entry_price=50.0,
            stop_price=48.0,
        )

        # Risk amount = $100, stop distance = $2
        # Shares = 100 / 2 = 50
        assert shares == 50

    def test_position_sizer_atr_based(self):
        """Test ATR-based position sizing."""
        from strategy_builder.strategies.gap_trading.position_sizer import (
            PositionSizer,
            SizingMethod,
        )

        sizer = PositionSizer(
            account_value=100000,
            risk_per_trade_pct=1.0,
            sizing_method=SizingMethod.ATR_BASED,
        )

        size = sizer.calculate(
            entry_price=100.0,
            atr=2.0,
            stop_atr_multiplier=1.5,
            is_long=True,
        )

        assert size is not None
        assert size.shares > 0
        assert size.stop_distance == 3.0  # 2.0 ATR * 1.5 multiplier


class TestOrderManagerModule:
    """Tests for order management."""

    def test_execution_status_enum(self):
        """Test ExecutionStatus enum."""
        from strategy_builder.strategies.gap_trading.order_manager import ExecutionStatus

        assert ExecutionStatus.PENDING.value == 'PENDING'
        assert ExecutionStatus.FILLED.value == 'FILLED'
        assert ExecutionStatus.REJECTED.value == 'REJECTED'

    def test_order_response_creation(self):
        """Test OrderResponse dataclass."""
        from strategy_builder.strategies.gap_trading.order_manager import (
            OrderResponse,
            ExecutionStatus,
        )

        response = OrderResponse(
            success=True,
            order_id='12345',
            status=ExecutionStatus.FILLED,
            filled_price=150.0,
            filled_quantity=100,
        )

        assert response.success is True
        assert response.order_id == '12345'
        assert response.status == ExecutionStatus.FILLED


class TestPositionManagerModule:
    """Tests for position management."""

    def test_position_status_enum(self):
        """Test PositionStatus enum."""
        from strategy_builder.strategies.gap_trading.position_manager import PositionStatus

        assert PositionStatus.OPEN.value == 'OPEN'
        assert PositionStatus.STOPPED.value == 'STOPPED'
        assert PositionStatus.EOD_CLOSED.value == 'EOD_CLOSED'

    def test_position_creation(self):
        """Test Position dataclass."""
        from strategy_builder.strategies.gap_trading.position_manager import (
            Position,
            PositionStatus,
            PositionSide,
        )

        position = Position(
            id=1,
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            stop_price=147.0,
            status=PositionStatus.OPEN,
            trade_date=date.today(),
        )

        assert position.symbol == 'AAPL'
        assert position.side == PositionSide.LONG
        assert position.quantity == 100

    def test_position_unrealized_pnl(self):
        """Test unrealized P&L calculation."""
        from strategy_builder.strategies.gap_trading.position_manager import (
            Position,
            PositionStatus,
            PositionSide,
        )

        # Long position
        long_pos = Position(
            id=1,
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            stop_price=147.0,
            status=PositionStatus.OPEN,
            trade_date=date.today(),
        )

        # Price went up
        pnl = long_pos.calculate_unrealized_pnl(155.0)
        assert pnl == 500.0  # (155 - 150) * 100

        # Price went down
        pnl = long_pos.calculate_unrealized_pnl(148.0)
        assert pnl == -200.0  # (148 - 150) * 100

        # Short position
        short_pos = Position(
            id=2,
            symbol='AAPL',
            side=PositionSide.SHORT,
            quantity=100,
            entry_price=150.0,
            stop_price=153.0,
            status=PositionStatus.OPEN,
            trade_date=date.today(),
        )

        # Price went down (profit for short)
        pnl = short_pos.calculate_unrealized_pnl(145.0)
        assert pnl == 500.0  # (150 - 145) * 100

    def test_position_stop_hit(self):
        """Test stop-loss detection."""
        from strategy_builder.strategies.gap_trading.position_manager import (
            Position,
            PositionStatus,
            PositionSide,
        )

        # Long position with stop at 147
        long_pos = Position(
            id=1,
            symbol='AAPL',
            side=PositionSide.LONG,
            quantity=100,
            entry_price=150.0,
            stop_price=147.0,
            status=PositionStatus.OPEN,
            trade_date=date.today(),
        )

        assert long_pos.is_stop_hit(146.0) is True
        assert long_pos.is_stop_hit(148.0) is False

        # Short position with stop at 153
        short_pos = Position(
            id=2,
            symbol='AAPL',
            side=PositionSide.SHORT,
            quantity=100,
            entry_price=150.0,
            stop_price=153.0,
            status=PositionStatus.OPEN,
            trade_date=date.today(),
        )

        assert short_pos.is_stop_hit(154.0) is True
        assert short_pos.is_stop_hit(152.0) is False


class TestBacktestModule:
    """Tests for backtesting."""

    def test_backtest_config_creation(self):
        """Test BacktestConfig dataclass."""
        from strategy_builder.strategies.gap_trading.backtest import BacktestConfig

        config = BacktestConfig(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            initial_capital=100000,
            risk_per_trade_pct=1.0,
        )

        assert config.start_date == date(2025, 1, 1)
        assert config.initial_capital == 100000
        assert config.risk_per_trade_pct == 1.0

    def test_backtest_results_summary(self):
        """Test BacktestResults summary generation."""
        from strategy_builder.strategies.gap_trading.backtest import (
            BacktestConfig,
            BacktestResults,
        )

        config = BacktestConfig(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            initial_capital=100000,
        )

        results = BacktestResults(
            config=config,
            total_return=5.0,
            win_rate=55.0,
            total_trades=20,
            profit_factor=1.5,
        )

        summary = results.summary()

        assert 'Gap Trading Backtest Results' in summary
        assert '5.00%' in summary
        assert '55.0%' in summary

    def test_backtest_run_with_mock_data(self):
        """Test backtest run with mocked data."""
        from strategy_builder.strategies.gap_trading.backtest import (
            BacktestConfig,
            GapTradingBacktester,
        )
        import pandas as pd
        import numpy as np

        config = BacktestConfig(
            start_date=date(2025, 11, 1),
            end_date=date(2025, 11, 10),
            initial_capital=100000,
            min_gap_pct=1.5,
        )

        backtester = GapTradingBacktester(config)

        # Run with a few symbols (will use yfinance)
        results = backtester.run(symbols=['SPY'])

        assert results is not None
        assert results.config == config


class TestReportingModule:
    """Tests for reporting."""

    def test_trade_metrics_creation(self):
        """Test TradeMetrics dataclass."""
        from strategy_builder.strategies.gap_trading.reporting import TradeMetrics

        metrics = TradeMetrics(
            total_trades=20,
            winning_trades=12,
            losing_trades=8,
            win_rate=60.0,
            gross_profit=1500.0,
            gross_loss=800.0,
            net_pnl=700.0,
            profit_factor=1.875,
            avg_winner=125.0,
            avg_loser=-100.0,
            largest_winner=300.0,
            largest_loser=-200.0,
            avg_trade=35.0,
        )

        assert metrics.total_trades == 20
        assert metrics.win_rate == 60.0
        assert metrics.profit_factor == 1.875


class TestConfigModule:
    """Tests for configuration."""

    def test_config_loading(self):
        """Test GapTradingConfig loading."""
        from strategy_builder.strategies.gap_trading.config import GapTradingConfig

        # Create config with defaults
        config = GapTradingConfig()

        assert config is not None
        assert hasattr(config, 'gap_detection')
        assert hasattr(config, 'risk_management')


class TestModuleImports:
    """Test that all modules can be imported."""

    def test_import_universe(self):
        """Test universe module import."""
        from strategy_builder.strategies.gap_trading import (
            UniverseManager,
            UniverseStock,
            PERMANENT_SYMBOLS,
        )
        assert UniverseManager is not None

    def test_import_volatility(self):
        """Test volatility module import."""
        from strategy_builder.strategies.gap_trading import (
            VolatilityCalculator,
            VolatilityMetrics,
        )
        assert VolatilityCalculator is not None

    def test_import_risk_tiers(self):
        """Test risk tiers module import."""
        from strategy_builder.strategies.gap_trading import (
            RiskTier,
            RiskTierClassifier,
            classify_risk_tier,
        )
        assert RiskTier is not None

    def test_import_signals(self):
        """Test signals module import."""
        from strategy_builder.strategies.gap_trading import (
            GapDirection,
            SignalType,
            GapInfo,
            TradeSignal,
            GapDetector,
            SignalGenerator,
        )
        assert GapDetector is not None

    def test_import_position_sizer(self):
        """Test position sizer module import."""
        from strategy_builder.strategies.gap_trading import (
            SizingMethod,
            PositionSize,
            PositionSizer,
        )
        assert PositionSizer is not None

    def test_import_order_manager(self):
        """Test order manager module import."""
        from strategy_builder.strategies.gap_trading import (
            ExecutionStatus,
            OrderPurpose,
            OrderResponse,
            ExecutionResult,
            OrderManager,
        )
        assert OrderManager is not None

    def test_import_position_manager(self):
        """Test position manager module import."""
        from strategy_builder.strategies.gap_trading import (
            PositionStatus,
            PositionSide,
            Position,
            PositionManager,
        )
        assert PositionManager is not None

    def test_import_backtest(self):
        """Test backtest module import."""
        from strategy_builder.strategies.gap_trading import (
            BacktestMode,
            BacktestConfig,
            BacktestTrade,
            BacktestResults,
            GapTradingBacktester,
            run_backtest,
        )
        assert GapTradingBacktester is not None

    def test_import_reporting(self):
        """Test reporting module import."""
        from strategy_builder.strategies.gap_trading import (
            ReportPeriod,
            TradeMetrics,
            RiskMetrics,
            DailyReport,
            ReportGenerator,
        )
        assert ReportGenerator is not None

    def test_import_realtime_monitor(self):
        """Test realtime monitor module import."""
        from strategy_builder.strategies.gap_trading import (
            MonitorStatus,
            CloseReason,
            PriceCheck,
            MonitorConfig,
            RealtimeStopLossMonitor,
        )
        assert RealtimeStopLossMonitor is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
