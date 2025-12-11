"""
Gap Trading Configuration Loader

Provides typed configuration loading and validation for the gap trading strategy.
"""

import os
import yaml
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from pathlib import Path


@dataclass
class RiskTierConfig:
    """Configuration for a single risk tier."""
    min_atr_pct: float = 0.0
    max_atr_pct: float = 100.0
    position_multiplier: float = 1.0
    stop_atr_multiplier: float = 1.0


@dataclass
class ScreeningConfig:
    """Stock screening criteria."""
    min_market_cap: int = 1_000_000_000
    min_avg_volume: int = 100_000
    min_dollar_volume: int = 20_000_000
    min_price: float = 10.0
    max_price: float = 500.0
    max_stocks: int = 100


@dataclass
class UniverseConfig:
    """Universe management configuration."""
    permanent_symbols: List[str] = field(default_factory=lambda: ['SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'USO', 'TLT'])
    reference_symbols: List[str] = field(default_factory=lambda: ['VIX'])
    screening: ScreeningConfig = field(default_factory=ScreeningConfig)
    update_day_of_week: str = 'sunday'
    update_time_et: str = '18:00'


@dataclass
class GapConfirmationConfig:
    """Gap confirmation settings."""
    enabled: bool = True
    time_et: str = '09:40'


@dataclass
class GapDetectionConfig:
    """Gap detection configuration."""
    min_gap_pct: float = 1.5
    max_gap_pct: float = 10.0
    confirmation: GapConfirmationConfig = field(default_factory=GapConfirmationConfig)
    trade_gap_up: bool = True
    trade_gap_down: bool = True


@dataclass
class VolatilityConfig:
    """Volatility calculation configuration."""
    atr_period: int = 14
    adr_period: int = 20
    volatility_period_short: int = 20
    volatility_period_long: int = 60
    beta_period: int = 60
    beta_benchmark: str = 'SPY'
    risk_tiers: Dict[str, RiskTierConfig] = field(default_factory=dict)

    def get_risk_tier(self, atr_pct: float) -> str:
        """Classify risk tier based on ATR percentage."""
        if atr_pct < self.risk_tiers.get('low', RiskTierConfig()).max_atr_pct:
            return 'LOW'
        elif atr_pct < self.risk_tiers.get('medium', RiskTierConfig()).max_atr_pct:
            return 'MEDIUM'
        else:
            return 'HIGH'

    def get_tier_config(self, tier: str) -> RiskTierConfig:
        """Get configuration for a specific risk tier."""
        tier_key = tier.lower()
        return self.risk_tiers.get(tier_key, RiskTierConfig())


@dataclass
class StopLossConfig:
    """Stop-loss configuration."""
    type: str = 'atr_based'
    fixed_pct: float = 2.0


@dataclass
class TargetConfig:
    """Target price configuration."""
    enabled: bool = False
    type: str = 'atr_based'
    atr_multiplier: float = 2.0
    fixed_pct: float = 4.0


@dataclass
class PositionsConfig:
    """Position sizing configuration."""
    sizing_method: str = 'atr_based'
    risk_per_trade: float = 100.0
    risk_per_trade_pct: float = 0.5
    max_position_pct: float = 5.0
    max_position_value: float = 10000.0
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    target: TargetConfig = field(default_factory=TargetConfig)


@dataclass
class RiskConfig:
    """Risk management configuration."""
    max_daily_trades: int = 10
    max_daily_loss: float = 500.0
    max_daily_loss_pct: float = 2.0
    max_open_positions: int = 5
    max_long_positions: int = 5
    max_short_positions: int = 3
    max_sector_exposure_pct: float = 30.0
    max_correlated_positions: int = 3


@dataclass
class MarketHoursConfig:
    """Market hours configuration."""
    market_open: str = '09:30'
    market_close: str = '16:00'
    trading_start: str = '09:40'
    trading_end: str = '15:55'
    eod_close_time: str = '15:55'
    pre_market_start: str = '04:00'
    pre_market_end: str = '09:30'
    timezone: str = 'America/New_York'


@dataclass
class TradierConfig:
    """Tradier-specific broker configuration."""
    sandbox: bool = True


@dataclass
class BrokerConfig:
    """Broker configuration."""
    provider: str = 'tradier'
    mode: str = 'paper'
    tradier: TradierConfig = field(default_factory=TradierConfig)
    order_timeout: int = 30
    retry_attempts: int = 3
    retry_delay: int = 5


@dataclass
class HealthCheckConfig:
    """Health check configuration."""
    enabled: bool = True
    interval_minutes: int = 15
    stale_threshold_minutes: int = 5


@dataclass
class MonitoringConfig:
    """Monitoring configuration."""
    price_check_interval: int = 60
    health_check: HealthCheckConfig = field(default_factory=HealthCheckConfig)
    log_level: str = 'INFO'
    log_price_checks: bool = True


@dataclass
class TelegramConfig:
    """Telegram notification configuration."""
    enabled: bool = True
    notify_on_entry: bool = True
    notify_on_exit: bool = True
    notify_on_stop: bool = True
    notify_on_eod: bool = True
    notify_on_rejection: bool = True
    notify_daily_report: bool = True


@dataclass
class NotificationsConfig:
    """Notifications configuration."""
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    daily_report_time: str = '16:00'


@dataclass
class RetentionConfig:
    """Data retention configuration."""
    price_checks_days: int = 30
    daily_gaps_days: int = 365
    volatility_history_days: int = 365


@dataclass
class DatabaseConfig:
    """Database configuration."""
    schema: str = 'gap_trading'
    retention: RetentionConfig = field(default_factory=RetentionConfig)


@dataclass
class BacktestingConfig:
    """Backtesting configuration."""
    default_days: int = 30
    initial_capital: float = 100000.0
    slippage_pct: float = 0.05
    commission_per_trade: float = 0.0
    save_trades: bool = True
    save_equity_curve: bool = True
    generate_report: bool = True


@dataclass
class GapTradingConfig:
    """
    Main configuration class for the gap trading strategy.

    Usage:
        config = GapTradingConfig.load()
        # or
        config = GapTradingConfig.load('/path/to/custom_config.yaml')
    """
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    gap_detection: GapDetectionConfig = field(default_factory=GapDetectionConfig)
    volatility: VolatilityConfig = field(default_factory=VolatilityConfig)
    positions: PositionsConfig = field(default_factory=PositionsConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    market_hours: MarketHoursConfig = field(default_factory=MarketHoursConfig)
    broker: BrokerConfig = field(default_factory=BrokerConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    notifications: NotificationsConfig = field(default_factory=NotificationsConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    backtesting: BacktestingConfig = field(default_factory=BacktestingConfig)

    @classmethod
    def get_default_config_path(cls) -> Path:
        """Get the default configuration file path."""
        return Path(__file__).parent / 'gap_trading_config.yaml'

    @classmethod
    def load(cls, config_path: Optional[str] = None) -> 'GapTradingConfig':
        """
        Load configuration from YAML file.

        Args:
            config_path: Optional path to config file. If not provided,
                        uses the default config in the package.

        Returns:
            GapTradingConfig instance with loaded values.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If config file has invalid YAML syntax.
            ValueError: If config values fail validation.
        """
        if config_path is None:
            config_path = cls.get_default_config_path()
        else:
            config_path = Path(config_path)

        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path, 'r') as f:
            raw_config = yaml.safe_load(f)

        config = cls._parse_config(raw_config)
        config.validate()
        return config

    @classmethod
    def _parse_config(cls, raw: dict) -> 'GapTradingConfig':
        """Parse raw YAML dict into typed configuration."""
        config = cls()

        # Universe
        if 'universe' in raw:
            u = raw['universe']
            config.universe = UniverseConfig(
                permanent_symbols=u.get('permanent_symbols', config.universe.permanent_symbols),
                reference_symbols=u.get('reference_symbols', config.universe.reference_symbols),
                screening=ScreeningConfig(**u.get('screening', {})) if 'screening' in u else config.universe.screening,
                update_day_of_week=u.get('update_schedule', {}).get('day_of_week', config.universe.update_day_of_week),
                update_time_et=u.get('update_schedule', {}).get('time_et', config.universe.update_time_et),
            )

        # Gap Detection
        if 'gap_detection' in raw:
            g = raw['gap_detection']
            config.gap_detection = GapDetectionConfig(
                min_gap_pct=g.get('min_gap_pct', config.gap_detection.min_gap_pct),
                max_gap_pct=g.get('max_gap_pct', config.gap_detection.max_gap_pct),
                confirmation=GapConfirmationConfig(**g.get('confirmation', {})) if 'confirmation' in g else config.gap_detection.confirmation,
                trade_gap_up=g.get('trade_gap_up', config.gap_detection.trade_gap_up),
                trade_gap_down=g.get('trade_gap_down', config.gap_detection.trade_gap_down),
            )

        # Volatility
        if 'volatility' in raw:
            v = raw['volatility']
            risk_tiers = {}
            if 'risk_tiers' in v:
                for tier_name, tier_data in v['risk_tiers'].items():
                    risk_tiers[tier_name] = RiskTierConfig(
                        min_atr_pct=tier_data.get('min_atr_pct', 0.0),
                        max_atr_pct=tier_data.get('max_atr_pct', 100.0),
                        position_multiplier=tier_data.get('position_multiplier', 1.0),
                        stop_atr_multiplier=tier_data.get('stop_atr_multiplier', 1.0),
                    )
            config.volatility = VolatilityConfig(
                atr_period=v.get('atr_period', config.volatility.atr_period),
                adr_period=v.get('adr_period', config.volatility.adr_period),
                volatility_period_short=v.get('volatility_period_short', config.volatility.volatility_period_short),
                volatility_period_long=v.get('volatility_period_long', config.volatility.volatility_period_long),
                beta_period=v.get('beta_period', config.volatility.beta_period),
                beta_benchmark=v.get('beta_benchmark', config.volatility.beta_benchmark),
                risk_tiers=risk_tiers,
            )

        # Positions
        if 'positions' in raw:
            p = raw['positions']
            config.positions = PositionsConfig(
                sizing_method=p.get('sizing_method', config.positions.sizing_method),
                risk_per_trade=p.get('risk_per_trade', config.positions.risk_per_trade),
                risk_per_trade_pct=p.get('risk_per_trade_pct', config.positions.risk_per_trade_pct),
                max_position_pct=p.get('max_position_pct', config.positions.max_position_pct),
                max_position_value=p.get('max_position_value', config.positions.max_position_value),
                stop_loss=StopLossConfig(**p.get('stop_loss', {})) if 'stop_loss' in p else config.positions.stop_loss,
                target=TargetConfig(**p.get('target', {})) if 'target' in p else config.positions.target,
            )

        # Risk
        if 'risk' in raw:
            config.risk = RiskConfig(**raw['risk'])

        # Market Hours
        if 'market_hours' in raw:
            config.market_hours = MarketHoursConfig(**raw['market_hours'])

        # Broker
        if 'broker' in raw:
            b = raw['broker']
            config.broker = BrokerConfig(
                provider=b.get('provider', config.broker.provider),
                mode=b.get('mode', config.broker.mode),
                tradier=TradierConfig(**b.get('tradier', {})) if 'tradier' in b else config.broker.tradier,
                order_timeout=b.get('order_timeout', config.broker.order_timeout),
                retry_attempts=b.get('retry_attempts', config.broker.retry_attempts),
                retry_delay=b.get('retry_delay', config.broker.retry_delay),
            )

        # Monitoring
        if 'monitoring' in raw:
            m = raw['monitoring']
            config.monitoring = MonitoringConfig(
                price_check_interval=m.get('price_check_interval', config.monitoring.price_check_interval),
                health_check=HealthCheckConfig(**m.get('health_check', {})) if 'health_check' in m else config.monitoring.health_check,
                log_level=m.get('log_level', config.monitoring.log_level),
                log_price_checks=m.get('log_price_checks', config.monitoring.log_price_checks),
            )

        # Notifications
        if 'notifications' in raw:
            n = raw['notifications']
            config.notifications = NotificationsConfig(
                telegram=TelegramConfig(**n.get('telegram', {})) if 'telegram' in n else config.notifications.telegram,
                daily_report_time=n.get('daily_report_time', config.notifications.daily_report_time),
            )

        # Database
        if 'database' in raw:
            d = raw['database']
            config.database = DatabaseConfig(
                schema=d.get('schema', config.database.schema),
                retention=RetentionConfig(**d.get('retention', {})) if 'retention' in d else config.database.retention,
            )

        # Backtesting
        if 'backtesting' in raw:
            config.backtesting = BacktestingConfig(**raw['backtesting'])

        return config

    def validate(self) -> bool:
        """
        Validate configuration values.

        Returns:
            True if all validations pass.

        Raises:
            ValueError: If any validation fails.
        """
        errors = []

        # Universe validation
        if not self.universe.permanent_symbols:
            errors.append("At least one permanent symbol must be defined")
        if self.universe.screening.min_price >= self.universe.screening.max_price:
            errors.append("min_price must be less than max_price")
        if self.universe.screening.max_stocks < 1:
            errors.append("max_stocks must be at least 1")

        # Gap detection validation
        if self.gap_detection.min_gap_pct <= 0:
            errors.append("min_gap_pct must be positive")
        if self.gap_detection.max_gap_pct <= self.gap_detection.min_gap_pct:
            errors.append("max_gap_pct must be greater than min_gap_pct")

        # Volatility validation
        if self.volatility.atr_period < 1:
            errors.append("atr_period must be at least 1")

        # Positions validation
        if self.positions.risk_per_trade <= 0:
            errors.append("risk_per_trade must be positive")
        if self.positions.max_position_pct <= 0 or self.positions.max_position_pct > 100:
            errors.append("max_position_pct must be between 0 and 100")

        # Risk validation
        if self.risk.max_daily_trades < 1:
            errors.append("max_daily_trades must be at least 1")
        if self.risk.max_open_positions < 1:
            errors.append("max_open_positions must be at least 1")

        # Broker validation
        if self.broker.provider not in ['tradier']:
            errors.append(f"Unsupported broker provider: {self.broker.provider}")
        if self.broker.mode not in ['paper', 'live']:
            errors.append(f"Invalid broker mode: {self.broker.mode}")

        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors))

        return True

    def get_all_symbols(self) -> List[str]:
        """Get all symbols (permanent + reference)."""
        return self.universe.permanent_symbols + self.universe.reference_symbols

    def get_tradeable_symbols(self) -> List[str]:
        """Get symbols that can be traded (permanent only, no reference)."""
        return self.universe.permanent_symbols.copy()

    def is_permanent_symbol(self, symbol: str) -> bool:
        """Check if a symbol is in the permanent watchlist."""
        return symbol.upper() in [s.upper() for s in self.universe.permanent_symbols]

    def is_reference_symbol(self, symbol: str) -> bool:
        """Check if a symbol is reference-only (no trading)."""
        return symbol.upper() in [s.upper() for s in self.universe.reference_symbols]
