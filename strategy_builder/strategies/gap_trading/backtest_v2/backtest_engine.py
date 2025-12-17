"""Backtest Engine for Gap Trading Strategy.

Main orchestrator that coordinates data loading, signal generation,
trade simulation, and metrics calculation.
"""

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List, Dict, Any, Optional
import json
from pathlib import Path

import pandas as pd

from .config import BacktestConfig
from .data_loader import GapBacktestDataLoader
from .signal_engine import SignalEngine, GapSignal, SignalType
from .trade_simulator import TradeSimulator, SimulatedTrade
from .metrics_engine import MetricsEngine, BacktestMetrics

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Complete backtest results.

    Attributes:
        config: Configuration used for backtest
        metrics: Performance metrics
        trades: List of all executed trades
        signals: List of all generated signals
        equity_curve: Daily equity values
        rejection_analysis: Analysis of rejected signals
    """
    config: BacktestConfig
    metrics: BacktestMetrics
    trades: List[SimulatedTrade] = field(default_factory=list)
    signals: List[GapSignal] = field(default_factory=list)
    equity_curve: List[Dict] = field(default_factory=list)
    rejection_analysis: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'config': self.config.to_dict(),
            'metrics': self.metrics.to_dict(),
            'trades': [t.to_dict() for t in self.trades],
            'equity_curve': self.equity_curve,
            'rejection_analysis': self.rejection_analysis,
            'total_signals': len(self.signals),
            'traded_signals': len(self.trades),
        }

    def save_to_json(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Results saved to {filepath}")

    def save_trades_csv(self, filepath: str):
        """Save trades to CSV file."""
        if not self.trades:
            logger.warning("No trades to save")
            return

        df = pd.DataFrame([t.to_dict() for t in self.trades])
        df.to_csv(filepath, index=False)
        logger.info(f"Trades saved to {filepath}")

    def summary(self) -> str:
        """Generate text summary."""
        return self.metrics.summary()


class BacktestEngine:
    """Main backtest orchestrator.

    Coordinates:
    - Data loading and caching
    - Signal generation for each trading day
    - Trade simulation with position sizing
    - Metrics calculation

    Attributes:
        config: Backtest configuration
        data_loader: Data loader instance
        signal_engine: Signal generation engine
        trade_simulator: Trade simulation engine
        metrics_engine: Metrics calculation engine
    """

    def __init__(
        self,
        config: BacktestConfig,
        fmp_client: Any = None,
        api_key: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """Initialize backtest engine.

        Args:
            config: Backtest configuration
            fmp_client: Optional FMP client instance
            api_key: FMP API key (if not using client)
            cache_dir: Directory for data caching
        """
        self.config = config

        # Initialize components
        self.data_loader = GapBacktestDataLoader(
            fmp_client=fmp_client,
            api_key=api_key,
            cache_dir=cache_dir
        )
        self.signal_engine = SignalEngine(config, self.data_loader)
        self.trade_simulator = TradeSimulator(config, self.data_loader)
        self.metrics_engine = MetricsEngine(config)

        logger.info(
            f"BacktestEngine initialized: {config.start_date} to {config.end_date}"
        )

    def run(
        self,
        symbols: Optional[List[str]] = None,
        preload_data: bool = True,
        progress_callback: Optional[callable] = None
    ) -> BacktestResult:
        """Run the complete backtest.

        Args:
            symbols: List of symbols to backtest (uses config.symbols if None)
            preload_data: Whether to preload all data upfront
            progress_callback: Optional callback(date, day_num, total_days)

        Returns:
            BacktestResult with complete analysis
        """
        # Get symbols
        symbols = symbols or self.config.symbols or self._get_default_symbols()
        logger.info(f"Starting backtest with {len(symbols)} symbols")

        # Preload data if requested
        if preload_data:
            self.data_loader.preload_data(
                symbols,
                self.config.start_date,
                self.config.end_date,
                include_minute_data=self.config.use_minute_data
            )

        # Get trading days
        trading_days = self.data_loader.get_trading_days(
            self.config.start_date,
            self.config.end_date
        )
        logger.info(f"Processing {len(trading_days)} trading days")

        # Initialize tracking
        all_signals: List[GapSignal] = []
        all_trades: List[SimulatedTrade] = []
        equity_curve: List[Dict] = []
        rejection_counts: Dict[str, int] = {}
        account_value = self.config.initial_capital

        # Process each trading day
        for day_num, trade_date in enumerate(trading_days):
            # Progress callback
            if progress_callback:
                progress_callback(trade_date, day_num + 1, len(trading_days))

            # Generate signals for all symbols
            signals = self.signal_engine.detect_gaps(symbols, trade_date)
            all_signals.extend(signals)

            # Track rejections
            for signal in signals:
                if signal.rejection_reason:
                    reason = signal.rejection_reason.value
                    rejection_counts[reason] = rejection_counts.get(reason, 0) + 1

            # Filter to tradeable signals
            tradeable = self.signal_engine.filter_signals(
                signals,
                self.config.max_positions,
                self.config.max_long_positions,
                self.config.max_short_positions
            )

            # Simulate trades
            daily_trades = self.trade_simulator.simulate_multiple_trades(
                tradeable,
                account_value
            )
            all_trades.extend(daily_trades)

            # Update account value
            daily_pnl = sum(t.pnl for t in daily_trades)
            account_value += daily_pnl

            # Record equity curve
            equity_curve.append({
                'date': trade_date.isoformat(),
                'equity': account_value,
                'daily_pnl': daily_pnl,
                'trades': len(daily_trades),
                'signals': len(signals),
                'tradeable_signals': len(tradeable)
            })

            # Log progress periodically
            if (day_num + 1) % 20 == 0:
                logger.info(
                    f"Day {day_num + 1}/{len(trading_days)}: "
                    f"{len(all_trades)} trades, equity=${account_value:,.2f}"
                )

        # Calculate metrics
        logger.info("Calculating performance metrics...")
        metrics = self.metrics_engine.calculate_all(all_trades, equity_curve)

        # Build result
        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=all_trades,
            signals=all_signals,
            equity_curve=equity_curve,
            rejection_analysis=rejection_counts
        )

        logger.info(
            f"Backtest complete: {len(all_trades)} trades, "
            f"Return={metrics.total_return_pct:+.1f}%, "
            f"Win Rate={metrics.win_rate:.1f}%"
        )

        return result

    def run_parameter_sweep(
        self,
        param_ranges: Dict[str, List[Any]],
        symbols: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Run multiple backtests with different parameters.

        Args:
            param_ranges: Dict of parameter names to list of values
                Example: {'min_gap_pct': [1.0, 1.5, 2.0], 'stop_atr_multiplier': [1.0, 1.5, 2.0]}
            symbols: List of symbols to test

        Returns:
            DataFrame with results for each parameter combination
        """
        from itertools import product

        # Generate all combinations
        param_names = list(param_ranges.keys())
        param_values = list(param_ranges.values())
        combinations = list(product(*param_values))

        logger.info(f"Running parameter sweep with {len(combinations)} combinations")

        results = []

        for i, combo in enumerate(combinations):
            # Create config with these parameters
            params = dict(zip(param_names, combo))

            config_dict = {
                'start_date': self.config.start_date,
                'end_date': self.config.end_date,
                'initial_capital': self.config.initial_capital,
                'symbols': symbols,
            }

            # Apply varied parameters
            for name, value in params.items():
                config_dict[name] = value

            try:
                test_config = BacktestConfig(**config_dict)
                engine = BacktestEngine(
                    test_config,
                    fmp_client=self.data_loader.fmp_client
                )

                result = engine.run(symbols, preload_data=False)

                # Record results
                row = params.copy()
                row.update({
                    'total_return_pct': result.metrics.total_return_pct,
                    'cagr': result.metrics.cagr,
                    'sharpe_ratio': result.metrics.sharpe_ratio,
                    'max_drawdown_pct': result.metrics.max_drawdown_pct,
                    'total_trades': result.metrics.total_trades,
                    'win_rate': result.metrics.win_rate,
                    'profit_factor': result.metrics.profit_factor,
                })
                results.append(row)

                logger.info(
                    f"Combo {i+1}/{len(combinations)}: "
                    f"{params} -> Return={result.metrics.total_return_pct:+.1f}%"
                )

            except Exception as e:
                logger.error(f"Failed for params {params}: {e}")
                row = params.copy()
                row['error'] = str(e)
                results.append(row)

        return pd.DataFrame(results)

    def _get_default_symbols(self) -> List[str]:
        """Get default symbols if none specified."""
        return [
            'SPY', 'QQQ', 'DIA', 'IWM',  # ETFs
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA',  # Tech
            'JPM', 'BAC', 'GS',  # Financials
            'XOM', 'CVX',  # Energy
            'JNJ', 'PFE',  # Healthcare
        ]


def run_quick_backtest(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    initial_capital: float = 100000,
    api_key: Optional[str] = None,
    **kwargs
) -> BacktestResult:
    """Convenience function for quick backtests.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: Optional list of symbols
        initial_capital: Starting capital
        api_key: FMP API key
        **kwargs: Additional config parameters

    Returns:
        BacktestResult
    """
    from datetime import datetime

    config = BacktestConfig(
        start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
        end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
        initial_capital=initial_capital,
        symbols=symbols,
        **kwargs
    )

    engine = BacktestEngine(config, api_key=api_key)
    return engine.run(symbols)


def compare_configurations(
    configs: List[BacktestConfig],
    symbols: Optional[List[str]] = None,
    api_key: Optional[str] = None
) -> pd.DataFrame:
    """Compare multiple backtest configurations.

    Args:
        configs: List of configurations to compare
        symbols: Symbols to test
        api_key: FMP API key

    Returns:
        DataFrame comparing results
    """
    results = []

    for i, config in enumerate(configs):
        logger.info(f"Running configuration {i+1}/{len(configs)}")

        engine = BacktestEngine(config, api_key=api_key)
        result = engine.run(symbols)

        row = {
            'config_id': i,
            'min_gap_pct': config.min_gap_pct,
            'max_gap_pct': config.max_gap_pct,
            'confirmation_minutes': config.confirmation_minutes,
            'stop_atr_multiplier': config.stop_atr_multiplier,
            'use_risk_tiers': config.use_risk_tiers,
            'total_return_pct': result.metrics.total_return_pct,
            'cagr': result.metrics.cagr,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown_pct': result.metrics.max_drawdown_pct,
            'total_trades': result.metrics.total_trades,
            'win_rate': result.metrics.win_rate,
            'profit_factor': result.metrics.profit_factor,
            'stop_loss_exit_pct': result.metrics.stop_loss_exit_pct,
        }
        results.append(row)

    return pd.DataFrame(results)
