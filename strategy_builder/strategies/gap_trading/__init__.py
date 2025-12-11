"""
Gap Trading Strategy Package

A gap trading strategy that identifies and trades significant price gaps
at market open, using ATR-based position sizing and stop-loss management.

Key Features:
- Permanent watchlist (SPY, QQQ, DIA, IWM, GLD, USO, TLT)
- Dynamic stock screening based on liquidity and volatility
- ATR-based position sizing and risk tier classification
- Automated order execution via Tradier
- Realtime stop-loss monitoring
- Telegram notifications for trade execution
"""

from .config import GapTradingConfig
from .universe import (
    UniverseManager,
    UniverseStock,
    PERMANENT_SYMBOLS,
    REFERENCE_SYMBOLS,
    get_permanent_symbols,
    get_reference_symbols,
    is_permanent,
    is_reference_only,
)
from .volatility import (
    VolatilityCalculator,
    VolatilityMetrics,
    GapStats,
)
from .risk_tiers import (
    RiskTier,
    RiskTierClassifier,
    RiskTierConfig,
    RiskParameters,
    classify_risk_tier,
    get_risk_multipliers,
)
from .signals import (
    GapDirection,
    SignalType,
    SignalReason,
    GapInfo,
    GapConfirmation,
    TradeSignal,
    GapDetector,
    SignalGenerator,
)
from .position_sizer import (
    SizingMethod,
    PositionSize,
    PositionSizer,
    calculate_shares_simple,
)
from .order_manager import (
    ExecutionStatus,
    OrderPurpose,
    OrderResponse,
    ExecutionResult,
    SyncResult,
    OrderManager,
)
from .position_manager import (
    PositionStatus,
    PositionSide,
    Position,
    ExitInfo,
    PositionManager,
)
from .realtime_monitor import (
    MonitorStatus,
    CloseReason,
    PriceCheck,
    MonitorConfig,
    MonitorState,
    RealtimeStopLossMonitor,
    run_monitor,
)
from .reporting import (
    ReportPeriod,
    TradeMetrics,
    RiskMetrics,
    DailyReport,
    ReportGenerator,
)
from .backtest import (
    BacktestMode,
    BacktestConfig,
    BacktestTrade,
    BacktestResults,
    GapTradingBacktester,
    run_backtest,
)

__all__ = [
    # Config
    'GapTradingConfig',
    # Universe
    'UniverseManager',
    'UniverseStock',
    'PERMANENT_SYMBOLS',
    'REFERENCE_SYMBOLS',
    'get_permanent_symbols',
    'get_reference_symbols',
    'is_permanent',
    'is_reference_only',
    # Volatility
    'VolatilityCalculator',
    'VolatilityMetrics',
    'GapStats',
    # Risk Tiers
    'RiskTier',
    'RiskTierClassifier',
    'RiskTierConfig',
    'RiskParameters',
    'classify_risk_tier',
    'get_risk_multipliers',
    # Signals
    'GapDirection',
    'SignalType',
    'SignalReason',
    'GapInfo',
    'GapConfirmation',
    'TradeSignal',
    'GapDetector',
    'SignalGenerator',
    # Position Sizing
    'SizingMethod',
    'PositionSize',
    'PositionSizer',
    'calculate_shares_simple',
    # Order Management
    'ExecutionStatus',
    'OrderPurpose',
    'OrderResponse',
    'ExecutionResult',
    'SyncResult',
    'OrderManager',
    # Position Management
    'PositionStatus',
    'PositionSide',
    'Position',
    'ExitInfo',
    'PositionManager',
]
