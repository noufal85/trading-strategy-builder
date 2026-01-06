#!/usr/bin/env python3
"""RSI/ADX Enhancement Validation Script.

Validates the RSI/ADX approach using existing historical trade data
to demonstrate potential improvement before implementation.

Usage:
    python -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.validate_rsi_adx

Created Date: 2026-01-05
"""

import os
import sys
import logging
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import json

import pandas as pd
import numpy as np

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class HistoricalSignal:
    """Historical signal with indicator data."""
    signal_id: int
    trade_date: date
    symbol: str
    direction: str  # UP or DOWN
    gap_percent: float
    entry_price: float
    executed: bool
    # Indicator values (to be calculated)
    rsi_14: Optional[float] = None
    adx_14: Optional[float] = None
    volume_ratio: Optional[float] = None
    priority_score: Optional[float] = None
    priority_rank: Optional[int] = None
    position_tier: Optional[str] = None
    # Actual position data (if executed)
    actual_pnl: Optional[float] = None
    actual_pnl_pct: Optional[float] = None


@dataclass
class ValidationResult:
    """Results of validation analysis."""
    period_start: date
    period_end: date
    trading_days: int

    # Actual performance
    actual_trades: int
    actual_winners: int
    actual_losers: int
    actual_total_pnl: float
    actual_avg_pnl: float
    actual_win_rate: float

    # Simulated performance (RSI/ADX + Top 6)
    simulated_trades: int
    simulated_winners: int
    simulated_losers: int
    simulated_total_pnl: float
    simulated_avg_pnl: float
    simulated_win_rate: float

    # Improvement analysis
    pnl_change: float
    pnl_change_pct: float
    win_rate_change: float
    trades_filtered: int
    trades_added: int

    # Detailed data
    filtered_trades: List[Dict] = field(default_factory=list)
    added_trades: List[Dict] = field(default_factory=list)
    all_signals_with_indicators: List[Dict] = field(default_factory=list)


class RSIADXValidator:
    """Validates RSI/ADX approach using historical data."""

    def __init__(
        self,
        db_host: str = "localhost",
        db_port: int = 5432,
        db_name: str = "timescaledb",
        db_user: str = "postgres",
        db_password: str = "password",
        fmp_api_key: Optional[str] = None,
        max_trades_per_day: int = 6
    ):
        """Initialize validator.

        Args:
            db_host: Database host
            db_port: Database port
            db_name: Database name
            db_user: Database user
            db_password: Database password
            fmp_api_key: FMP API key (defaults to env var)
            max_trades_per_day: Maximum trades in new strategy
        """
        self.db_config = {
            'host': db_host,
            'port': db_port,
            'database': db_name,
            'user': db_user,
            'password': db_password
        }
        self.fmp_api_key = fmp_api_key or os.environ.get('FMP_API_KEY')
        self.max_trades_per_day = max_trades_per_day

        # Caches
        self._price_cache: Dict[str, pd.DataFrame] = {}
        self._fmp_client = None

    def _get_db_connection(self):
        """Get database connection."""
        try:
            import psycopg2
            return psycopg2.connect(**self.db_config)
        except ImportError:
            logger.error("psycopg2 not installed. Install with: pip install psycopg2-binary")
            raise

    def _get_fmp_client(self):
        """Get or create FMP client."""
        if self._fmp_client:
            return self._fmp_client

        try:
            from fmp import FMPClient
            self._fmp_client = FMPClient(api_key=self.fmp_api_key)
            return self._fmp_client
        except ImportError:
            logger.error("FMP package not installed")
            raise

    def load_historical_signals(self) -> List[HistoricalSignal]:
        """Load historical signals from database."""
        logger.info("Loading historical signals from database...")

        conn = self._get_db_connection()
        try:
            query = """
                SELECT
                    s.signal_id,
                    s.trade_date,
                    s.symbol,
                    s.direction,
                    s.gap_percent,
                    s.entry_price,
                    s.executed,
                    p.pnl as actual_pnl,
                    p.pnl_percent as actual_pnl_pct
                FROM gap_trading.trade_signals s
                LEFT JOIN gap_trading.positions p
                    ON s.signal_id = p.signal_id
                WHERE s.trade_date >= '2025-12-01'
                ORDER BY s.trade_date, s.symbol
            """

            df = pd.read_sql(query, conn)

            signals = []
            for _, row in df.iterrows():
                signals.append(HistoricalSignal(
                    signal_id=row['signal_id'],
                    trade_date=row['trade_date'],
                    symbol=row['symbol'],
                    direction=row['direction'] if row['direction'] else 'UP',
                    gap_percent=float(row['gap_percent']) if row['gap_percent'] else 0.0,
                    entry_price=float(row['entry_price']) if row['entry_price'] else 0.0,
                    executed=bool(row['executed']),
                    actual_pnl=float(row['actual_pnl']) if row['actual_pnl'] else None,
                    actual_pnl_pct=float(row['actual_pnl_pct']) if row['actual_pnl_pct'] else None
                ))

            logger.info(f"Loaded {len(signals)} historical signals")
            return signals

        finally:
            conn.close()

    def get_historical_prices(
        self,
        symbol: str,
        end_date: date,
        lookback_days: int = 30
    ) -> Optional[pd.DataFrame]:
        """Get historical prices for indicator calculation.

        Args:
            symbol: Stock symbol
            end_date: End date (typically the signal date)
            lookback_days: Days of history needed

        Returns:
            DataFrame with OHLCV data or None
        """
        cache_key = f"{symbol}_{end_date}_{lookback_days}"

        if cache_key in self._price_cache:
            return self._price_cache[cache_key]

        start_date = end_date - timedelta(days=lookback_days + 10)  # Extra buffer

        try:
            client = self._get_fmp_client()
            data = client.get(f'historical-price-full/{symbol}', {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            })

            if not data or 'historical' not in data:
                logger.warning(f"No historical data for {symbol}")
                return None

            historical = data['historical']
            if not historical:
                return None

            df = pd.DataFrame(historical)
            df['date'] = pd.to_datetime(df['date']).dt.date
            df = df.sort_values('date').reset_index(drop=True)

            # Select columns
            df = df[['date', 'open', 'high', 'low', 'close', 'volume']]

            self._price_cache[cache_key] = df
            return df

        except Exception as e:
            logger.error(f"Error fetching prices for {symbol}: {e}")
            return None

    def calculate_indicators_for_signal(
        self,
        signal: HistoricalSignal
    ) -> HistoricalSignal:
        """Calculate RSI and ADX for a signal.

        Args:
            signal: Historical signal to enrich

        Returns:
            Signal with indicators populated
        """
        from strategy_builder.strategies.gap_trading.indicators import calculate_rsi, calculate_adx

        # Get historical prices up to signal date
        df = self.get_historical_prices(signal.symbol, signal.trade_date)

        if df is None or len(df) < 30:
            logger.warning(f"Insufficient data for {signal.symbol} on {signal.trade_date}")
            return signal

        # Filter to data before/on signal date
        df = df[df['date'] <= signal.trade_date]

        if len(df) < 30:
            return signal

        # Calculate RSI
        prices = df['close'].tolist()
        signal.rsi_14 = calculate_rsi(prices, period=14)

        # Calculate ADX
        highs = df['high'].tolist()
        lows = df['low'].tolist()
        closes = df['close'].tolist()
        adx, plus_di, minus_di = calculate_adx(highs, lows, closes, period=14)
        signal.adx_14 = adx

        # Calculate volume ratio (today vs 20-day average)
        if len(df) >= 20:
            recent_df = df.tail(21)  # Last 21 days to get today + 20 day average
            if len(recent_df) >= 2:
                today_volume = recent_df.iloc[-1]['volume']
                avg_volume = recent_df.iloc[:-1]['volume'].mean()
                if avg_volume > 0:
                    signal.volume_ratio = today_volume / avg_volume

        return signal

    def calculate_priority_scores(
        self,
        signals: List[HistoricalSignal]
    ) -> List[HistoricalSignal]:
        """Calculate priority scores and rank signals by day.

        Args:
            signals: List of signals with indicators

        Returns:
            Signals with priority scores and ranks
        """
        from strategy_builder.strategies.gap_trading.indicators import calculate_priority_score, get_position_tier

        # Group by trade date
        signals_by_date: Dict[date, List[HistoricalSignal]] = {}
        for sig in signals:
            if sig.trade_date not in signals_by_date:
                signals_by_date[sig.trade_date] = []
            signals_by_date[sig.trade_date].append(sig)

        # Calculate scores and rank within each day
        for trade_date, day_signals in signals_by_date.items():
            # Calculate priority score for each signal
            for sig in day_signals:
                gap_direction = "UP" if sig.gap_percent > 0 else "DOWN"
                sig.priority_score = calculate_priority_score(
                    gap_pct=sig.gap_percent,
                    volume_ratio=sig.volume_ratio or 1.0,
                    adx=sig.adx_14,
                    rsi=sig.rsi_14,
                    gap_direction=gap_direction
                )

            # Rank by priority score (descending)
            sorted_signals = sorted(
                day_signals,
                key=lambda s: s.priority_score or 0,
                reverse=True
            )

            for rank, sig in enumerate(sorted_signals, 1):
                sig.priority_rank = rank
                sig.position_tier = get_position_tier(
                    rank, len(sorted_signals), self.max_trades_per_day
                )

        return signals

    def run_validation(self) -> ValidationResult:
        """Run full validation analysis.

        Returns:
            ValidationResult with comparison data
        """
        logger.info("=" * 60)
        logger.info("RSI/ADX Enhancement Validation")
        logger.info("=" * 60)

        # Step 1: Load historical signals
        signals = self.load_historical_signals()

        if not signals:
            raise ValueError("No historical signals found")

        # Step 2: Calculate indicators for each signal
        logger.info("Calculating indicators for each signal...")
        total = len(signals)
        for i, sig in enumerate(signals):
            if (i + 1) % 10 == 0:
                logger.info(f"Processing {i + 1}/{total}...")
            self.calculate_indicators_for_signal(sig)

        # Step 3: Calculate priority scores
        logger.info("Calculating priority scores...")
        signals = self.calculate_priority_scores(signals)

        # Step 4: Analyze results
        logger.info("Analyzing results...")
        result = self._analyze_results(signals)

        return result

    def _analyze_results(
        self,
        signals: List[HistoricalSignal]
    ) -> ValidationResult:
        """Analyze actual vs simulated performance.

        Args:
            signals: Signals with indicators and priority scores

        Returns:
            ValidationResult
        """
        # Get date range
        dates = [s.trade_date for s in signals]
        period_start = min(dates)
        period_end = max(dates)
        trading_days = len(set(dates))

        # Separate executed vs not executed
        executed_signals = [s for s in signals if s.executed and s.actual_pnl is not None]

        # ACTUAL performance (what really happened)
        actual_trades = len(executed_signals)
        actual_winners = len([s for s in executed_signals if s.actual_pnl > 0])
        actual_losers = len([s for s in executed_signals if s.actual_pnl < 0])
        actual_total_pnl = sum(s.actual_pnl for s in executed_signals)
        actual_avg_pnl = actual_total_pnl / actual_trades if actual_trades > 0 else 0
        actual_win_rate = actual_winners / actual_trades * 100 if actual_trades > 0 else 0

        # SIMULATED performance (top 6 by priority)
        # Get signals that would have been selected (rank <= 6)
        simulated_signals = [s for s in signals if s.priority_rank and s.priority_rank <= self.max_trades_per_day]

        # For simulated P&L, we need to use actual P&L where available
        # and estimate for trades that weren't executed
        simulated_with_pnl = [s for s in simulated_signals if s.actual_pnl is not None]

        simulated_trades = len(simulated_with_pnl)
        simulated_winners = len([s for s in simulated_with_pnl if s.actual_pnl > 0])
        simulated_losers = len([s for s in simulated_with_pnl if s.actual_pnl < 0])
        simulated_total_pnl = sum(s.actual_pnl for s in simulated_with_pnl)
        simulated_avg_pnl = simulated_total_pnl / simulated_trades if simulated_trades > 0 else 0
        simulated_win_rate = simulated_winners / simulated_trades * 100 if simulated_trades > 0 else 0

        # Identify filtered and added trades
        # Filtered: was executed but wouldn't be in top 6
        filtered_trades = []
        for s in executed_signals:
            if s.priority_rank and s.priority_rank > self.max_trades_per_day:
                filtered_trades.append({
                    'date': str(s.trade_date),
                    'symbol': s.symbol,
                    'gap_pct': s.gap_percent,
                    'rsi': s.rsi_14,
                    'adx': s.adx_14,
                    'actual_pnl': s.actual_pnl,
                    'priority_rank': s.priority_rank,
                    'priority_score': s.priority_score
                })

        # Added: would be in top 6 but wasn't executed
        added_trades = []
        for s in signals:
            if s.priority_rank and s.priority_rank <= self.max_trades_per_day and not s.executed:
                added_trades.append({
                    'date': str(s.trade_date),
                    'symbol': s.symbol,
                    'gap_pct': s.gap_percent,
                    'rsi': s.rsi_14,
                    'adx': s.adx_14,
                    'priority_rank': s.priority_rank,
                    'priority_score': s.priority_score
                })

        # Calculate improvement
        pnl_change = simulated_total_pnl - actual_total_pnl
        pnl_change_pct = (pnl_change / abs(actual_total_pnl) * 100) if actual_total_pnl != 0 else 0
        win_rate_change = simulated_win_rate - actual_win_rate

        # Prepare all signals for detailed output
        all_signals_data = []
        for s in signals:
            all_signals_data.append({
                'signal_id': s.signal_id,
                'date': str(s.trade_date),
                'symbol': s.symbol,
                'direction': s.direction,
                'gap_pct': s.gap_percent,
                'rsi_14': s.rsi_14,
                'adx_14': s.adx_14,
                'volume_ratio': s.volume_ratio,
                'priority_score': s.priority_score,
                'priority_rank': s.priority_rank,
                'position_tier': s.position_tier,
                'executed': s.executed,
                'actual_pnl': s.actual_pnl,
                'actual_pnl_pct': s.actual_pnl_pct
            })

        return ValidationResult(
            period_start=period_start,
            period_end=period_end,
            trading_days=trading_days,
            actual_trades=actual_trades,
            actual_winners=actual_winners,
            actual_losers=actual_losers,
            actual_total_pnl=actual_total_pnl,
            actual_avg_pnl=actual_avg_pnl,
            actual_win_rate=actual_win_rate,
            simulated_trades=simulated_trades,
            simulated_winners=simulated_winners,
            simulated_losers=simulated_losers,
            simulated_total_pnl=simulated_total_pnl,
            simulated_avg_pnl=simulated_avg_pnl,
            simulated_win_rate=simulated_win_rate,
            pnl_change=pnl_change,
            pnl_change_pct=pnl_change_pct,
            win_rate_change=win_rate_change,
            trades_filtered=len(filtered_trades),
            trades_added=len(added_trades),
            filtered_trades=filtered_trades,
            added_trades=added_trades,
            all_signals_with_indicators=all_signals_data
        )

    def print_report(self, result: ValidationResult) -> str:
        """Generate human-readable report.

        Args:
            result: Validation result

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 70)
        report.append("RSI/ADX ENHANCEMENT VALIDATION REPORT")
        report.append("=" * 70)
        report.append(f"Period: {result.period_start} to {result.period_end} ({result.trading_days} trading days)")
        report.append("")

        report.append("-" * 70)
        report.append("ACTUAL PERFORMANCE (Current Logic - Top by Gap %)")
        report.append("-" * 70)
        report.append(f"  Total Trades:     {result.actual_trades}")
        report.append(f"  Winners:          {result.actual_winners}")
        report.append(f"  Losers:           {result.actual_losers}")
        report.append(f"  Win Rate:         {result.actual_win_rate:.1f}%")
        report.append(f"  Total P&L:        ${result.actual_total_pnl:,.2f}")
        report.append(f"  Avg P&L/Trade:    ${result.actual_avg_pnl:,.2f}")
        report.append("")

        report.append("-" * 70)
        report.append(f"SIMULATED PERFORMANCE (RSI/ADX + Top {self.max_trades_per_day} by Priority)")
        report.append("-" * 70)
        report.append(f"  Total Trades:     {result.simulated_trades}")
        report.append(f"  Winners:          {result.simulated_winners}")
        report.append(f"  Losers:           {result.simulated_losers}")
        report.append(f"  Win Rate:         {result.simulated_win_rate:.1f}%")
        report.append(f"  Total P&L:        ${result.simulated_total_pnl:,.2f}")
        report.append(f"  Avg P&L/Trade:    ${result.simulated_avg_pnl:,.2f}")
        report.append("")

        report.append("-" * 70)
        report.append("IMPROVEMENT ANALYSIS")
        report.append("-" * 70)
        pnl_emoji = "+" if result.pnl_change >= 0 else ""
        report.append(f"  P&L Change:       {pnl_emoji}${result.pnl_change:,.2f} ({pnl_emoji}{result.pnl_change_pct:.1f}%)")
        wr_emoji = "+" if result.win_rate_change >= 0 else ""
        report.append(f"  Win Rate Change:  {wr_emoji}{result.win_rate_change:.1f}%")
        report.append(f"  Trades Filtered:  {result.trades_filtered} (would have been excluded)")
        report.append(f"  Trades Added:     {result.trades_added} (would have been included)")
        report.append("")

        if result.filtered_trades:
            report.append("-" * 70)
            report.append("TOP 5 TRADES THAT WOULD HAVE BEEN FILTERED")
            report.append("-" * 70)
            report.append(f"{'Date':<12} {'Symbol':<8} {'Gap%':>7} {'RSI':>6} {'ADX':>6} {'P&L':>10} {'Rank':>6}")
            report.append("-" * 70)

            # Sort by actual P&L to show worst performers that would be filtered
            sorted_filtered = sorted(result.filtered_trades, key=lambda x: x.get('actual_pnl', 0))
            for trade in sorted_filtered[:5]:
                rsi = f"{trade['rsi']:.1f}" if trade['rsi'] else "N/A"
                adx = f"{trade['adx']:.1f}" if trade['adx'] else "N/A"
                pnl = f"${trade['actual_pnl']:,.2f}" if trade['actual_pnl'] else "N/A"
                report.append(
                    f"{trade['date']:<12} {trade['symbol']:<8} {trade['gap_pct']:>7.2f} "
                    f"{rsi:>6} {adx:>6} {pnl:>10} {trade['priority_rank']:>6}"
                )
            report.append("")

        if result.added_trades:
            report.append("-" * 70)
            report.append("TOP 5 TRADES THAT WOULD HAVE BEEN ADDED")
            report.append("-" * 70)
            report.append(f"{'Date':<12} {'Symbol':<8} {'Gap%':>7} {'RSI':>6} {'ADX':>6} {'Score':>8} {'Rank':>6}")
            report.append("-" * 70)

            sorted_added = sorted(result.added_trades, key=lambda x: x.get('priority_score', 0), reverse=True)
            for trade in sorted_added[:5]:
                rsi = f"{trade['rsi']:.1f}" if trade['rsi'] else "N/A"
                adx = f"{trade['adx']:.1f}" if trade['adx'] else "N/A"
                score = f"{trade['priority_score']:.1f}" if trade['priority_score'] else "N/A"
                report.append(
                    f"{trade['date']:<12} {trade['symbol']:<8} {trade['gap_pct']:>7.2f} "
                    f"{rsi:>6} {adx:>6} {score:>8} {trade['priority_rank']:>6}"
                )
            report.append("")

        report.append("=" * 70)

        # Recommendation
        report.append("RECOMMENDATION")
        report.append("=" * 70)
        if result.pnl_change > 0 and result.win_rate_change >= 0:
            report.append("  POSITIVE: RSI/ADX enhancement shows improvement.")
            report.append("  Recommend proceeding with implementation.")
        elif result.pnl_change > 0:
            report.append("  MIXED: P&L improved but win rate decreased.")
            report.append("  Consider adjusting parameters before implementation.")
        elif result.win_rate_change > 0:
            report.append("  MIXED: Win rate improved but total P&L decreased.")
            report.append("  Review position sizing impact.")
        else:
            report.append("  CAUTION: Enhancement shows no clear improvement.")
            report.append("  Recommend further analysis before implementation.")

        report.append("")
        report.append("NOTE: Small sample size - results should be validated with")
        report.append("extended backtesting using more historical data.")
        report.append("=" * 70)

        return "\n".join(report)

    def save_results(self, result: ValidationResult, output_dir: str = "."):
        """Save results to files.

        Args:
            result: Validation result
            output_dir: Output directory
        """
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save report
        report = self.print_report(result)
        report_path = os.path.join(output_dir, "rsi_adx_validation_report.txt")
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Report saved to: {report_path}")

        # Save detailed data as JSON
        data = {
            'summary': {
                'period_start': str(result.period_start),
                'period_end': str(result.period_end),
                'trading_days': result.trading_days,
                'actual': {
                    'trades': result.actual_trades,
                    'winners': result.actual_winners,
                    'losers': result.actual_losers,
                    'total_pnl': result.actual_total_pnl,
                    'avg_pnl': result.actual_avg_pnl,
                    'win_rate': result.actual_win_rate
                },
                'simulated': {
                    'trades': result.simulated_trades,
                    'winners': result.simulated_winners,
                    'losers': result.simulated_losers,
                    'total_pnl': result.simulated_total_pnl,
                    'avg_pnl': result.simulated_avg_pnl,
                    'win_rate': result.simulated_win_rate
                },
                'improvement': {
                    'pnl_change': result.pnl_change,
                    'pnl_change_pct': result.pnl_change_pct,
                    'win_rate_change': result.win_rate_change,
                    'trades_filtered': result.trades_filtered,
                    'trades_added': result.trades_added
                }
            },
            'filtered_trades': result.filtered_trades,
            'added_trades': result.added_trades,
            'all_signals': result.all_signals_with_indicators
        }

        json_path = os.path.join(output_dir, "rsi_adx_validation_data.json")
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Data saved to: {json_path}")

        # Save signals as CSV for easy analysis
        df = pd.DataFrame(result.all_signals_with_indicators)
        csv_path = os.path.join(output_dir, "rsi_adx_validation_signals.csv")
        df.to_csv(csv_path, index=False)
        logger.info(f"Signals CSV saved to: {csv_path}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate RSI/ADX enhancement approach')
    parser.add_argument('--db-host', default='localhost', help='Database host')
    parser.add_argument('--db-port', type=int, default=5432, help='Database port')
    parser.add_argument('--max-trades', type=int, default=6, help='Max trades per day')
    parser.add_argument('--output-dir', default='.', help='Output directory for reports')

    args = parser.parse_args()

    # Check for FMP API key
    if not os.environ.get('FMP_API_KEY'):
        logger.error("FMP_API_KEY environment variable not set")
        sys.exit(1)

    validator = RSIADXValidator(
        db_host=args.db_host,
        db_port=args.db_port,
        max_trades_per_day=args.max_trades
    )

    try:
        result = validator.run_validation()

        # Print report
        report = validator.print_report(result)
        print(report)

        # Save results
        validator.save_results(result, args.output_dir)

    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
