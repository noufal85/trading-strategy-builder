"""Backtesting Module for Gap Trading Strategy.

Provides historical backtesting capabilities to validate strategy
performance before live deployment.

Features:
- Historical gap detection simulation
- Position sizing with historical ATR
- Stop-loss and EOD close simulation
- Performance metrics calculation
- Trade-by-trade analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class BacktestMode(str, Enum):
    """Backtest simulation mode."""
    INTRADAY = 'intraday'  # Simulate intraday price action
    DAILY = 'daily'  # Use daily OHLC only


@dataclass
class BacktestConfig:
    """Configuration for backtesting.

    Attributes:
        start_date: Backtest start date
        end_date: Backtest end date
        initial_capital: Starting capital
        risk_per_trade_pct: Risk per trade as percentage
        min_gap_pct: Minimum gap threshold
        max_gap_pct: Maximum gap threshold
        stop_atr_multiplier: ATR multiplier for stops
        max_positions: Maximum concurrent positions
        commission: Commission per trade
        slippage_pct: Slippage percentage
        use_risk_tiers: Whether to use risk tier sizing
    """
    start_date: date
    end_date: date
    initial_capital: float = 100000.0
    risk_per_trade_pct: float = 1.0
    min_gap_pct: float = 1.5
    max_gap_pct: float = 10.0
    stop_atr_multiplier: float = 1.5
    max_positions: int = 5
    commission: float = 0.0
    slippage_pct: float = 0.05
    use_risk_tiers: bool = True


@dataclass
class BacktestTrade:
    """Record of a backtested trade.

    Attributes:
        trade_date: Date of trade
        symbol: Stock ticker
        direction: LONG or SHORT
        entry_price: Entry price
        exit_price: Exit price
        shares: Number of shares
        stop_price: Stop-loss price
        gap_pct: Gap percentage that triggered entry
        atr: ATR at entry
        risk_tier: Risk tier classification
        exit_reason: Reason for exit (stop_loss, eod)
        pnl: Profit/loss
        pnl_pct: Return percentage
        commission: Commission paid
    """
    trade_date: date
    symbol: str
    direction: str
    entry_price: float
    exit_price: float
    shares: int
    stop_price: float
    gap_pct: float
    atr: float
    risk_tier: str = 'MEDIUM'
    exit_reason: str = 'eod'
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'trade_date': self.trade_date.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'exit_price': self.exit_price,
            'shares': self.shares,
            'stop_price': self.stop_price,
            'gap_pct': self.gap_pct,
            'atr': self.atr,
            'risk_tier': self.risk_tier,
            'exit_reason': self.exit_reason,
            'pnl': round(self.pnl, 2),
            'pnl_pct': round(self.pnl_pct, 2),
            'commission': self.commission,
        }


@dataclass
class BacktestResults:
    """Results from a backtest run.

    Attributes:
        config: Backtest configuration
        trades: List of all trades
        equity_curve: Daily equity values
        total_return: Total return percentage
        cagr: Compound annual growth rate
        sharpe_ratio: Sharpe ratio
        sortino_ratio: Sortino ratio
        max_drawdown: Maximum drawdown percentage
        win_rate: Percentage of winning trades
        profit_factor: Gross profit / Gross loss
        total_trades: Number of trades
        avg_trade: Average P&L per trade
        best_trade: Best single trade P&L
        worst_trade: Worst single trade P&L
        avg_winner: Average winning trade
        avg_loser: Average losing trade
        max_consecutive_wins: Longest winning streak
        max_consecutive_losses: Longest losing streak
        trading_days: Number of days with activity
    """
    config: BacktestConfig
    trades: List[BacktestTrade] = field(default_factory=list)
    equity_curve: List[Dict[str, Any]] = field(default_factory=list)
    total_return: float = 0.0
    cagr: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    total_trades: int = 0
    avg_trade: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    trading_days: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_date': self.config.start_date.isoformat(),
            'end_date': self.config.end_date.isoformat(),
            'initial_capital': self.config.initial_capital,
            'total_return': round(self.total_return, 2),
            'cagr': round(self.cagr, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'win_rate': round(self.win_rate, 2),
            'profit_factor': round(self.profit_factor, 2),
            'total_trades': self.total_trades,
            'avg_trade': round(self.avg_trade, 2),
            'best_trade': round(self.best_trade, 2),
            'worst_trade': round(self.worst_trade, 2),
            'trading_days': self.trading_days,
        }

    def summary(self) -> str:
        """Generate text summary."""
        return f"""
Gap Trading Backtest Results
============================
Period: {self.config.start_date} to {self.config.end_date}
Initial Capital: ${self.config.initial_capital:,.2f}

Performance
-----------
Total Return: {self.total_return:+.2f}%
CAGR: {self.cagr:.2f}%
Sharpe Ratio: {self.sharpe_ratio:.2f}
Max Drawdown: {self.max_drawdown:.2f}%

Trading Statistics
------------------
Total Trades: {self.total_trades}
Win Rate: {self.win_rate:.1f}%
Profit Factor: {self.profit_factor:.2f}
Avg Trade: ${self.avg_trade:,.2f}
Best Trade: ${self.best_trade:,.2f}
Worst Trade: ${self.worst_trade:,.2f}

Streaks
-------
Max Consecutive Wins: {self.max_consecutive_wins}
Max Consecutive Losses: {self.max_consecutive_losses}
Trading Days: {self.trading_days}
"""


class GapTradingBacktester:
    """Backtester for gap trading strategy.

    Simulates historical trading performance using gap detection
    and ATR-based position sizing.

    Attributes:
        config: Backtest configuration
        data_provider: Data provider for historical prices
    """

    def __init__(
        self,
        config: BacktestConfig,
        data_provider: Any = None,
        db_conn: Any = None
    ):
        """Initialize backtester.

        Args:
            config: Backtest configuration
            data_provider: Optional FMP or other data provider
            db_conn: Optional database connection for universe data
        """
        self.config = config
        self.data_provider = data_provider
        self.db_conn = db_conn

        self._equity = config.initial_capital
        self._trades: List[BacktestTrade] = []
        self._equity_curve: List[Dict] = []

        logger.info(
            f"Backtester initialized: {config.start_date} to {config.end_date}, "
            f"capital=${config.initial_capital:,.0f}"
        )

    def run(self, symbols: Optional[List[str]] = None) -> BacktestResults:
        """Run the backtest.

        Args:
            symbols: List of symbols to backtest (or uses universe)

        Returns:
            BacktestResults with full analysis
        """
        logger.info("Starting backtest...")

        # Get symbols
        if symbols is None:
            symbols = self._get_universe_symbols()

        if not symbols:
            logger.warning("No symbols to backtest")
            return BacktestResults(config=self.config)

        logger.info(f"Backtesting {len(symbols)} symbols")

        # Get historical data
        historical_data = self._get_historical_data(symbols)

        # Get ATR data for position sizing
        atr_data = self._calculate_historical_atr(historical_data)

        # Run day-by-day simulation
        current_date = self.config.start_date
        while current_date <= self.config.end_date:
            # Skip weekends
            if current_date.weekday() < 5:
                self._simulate_day(current_date, historical_data, atr_data)

            current_date += timedelta(days=1)

        # Calculate final metrics
        results = self._calculate_results()

        logger.info(f"Backtest complete: {results.total_trades} trades, "
                    f"{results.total_return:.2f}% return")

        return results

    def _get_universe_symbols(self) -> List[str]:
        """Get symbols from universe database."""
        if self.db_conn:
            try:
                cursor = self.db_conn.cursor()
                cursor.execute("""
                    SELECT symbol FROM gap_trading.stock_universe
                    WHERE is_active = true AND is_reference_only = false
                """)
                return [row[0] for row in cursor.fetchall()]
            except Exception as e:
                logger.error(f"Failed to get universe: {e}")

        # Default symbols
        return ['SPY', 'QQQ', 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META']

    def _get_historical_data(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """Get historical OHLCV data for symbols.

        Returns:
            Dict mapping symbol to DataFrame with OHLCV data
        """
        data = {}

        if self.data_provider:
            for symbol in symbols:
                try:
                    df = self.data_provider.get_historical(
                        symbol,
                        start_date=self.config.start_date - timedelta(days=30),
                        end_date=self.config.end_date
                    )
                    if df is not None and not df.empty:
                        data[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to get data for {symbol}: {e}")
        else:
            # Use yfinance as fallback
            import yfinance as yf

            for symbol in symbols:
                try:
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(
                        start=self.config.start_date - timedelta(days=30),
                        end=self.config.end_date + timedelta(days=1)
                    )
                    if not df.empty:
                        df = df.reset_index()
                        df.columns = [c.lower() for c in df.columns]
                        data[symbol] = df
                except Exception as e:
                    logger.warning(f"Failed to get yfinance data for {symbol}: {e}")

        logger.info(f"Loaded data for {len(data)} symbols")
        return data

    def _calculate_historical_atr(
        self,
        data: Dict[str, pd.DataFrame],
        period: int = 14
    ) -> Dict[str, pd.Series]:
        """Calculate ATR for each symbol.

        Returns:
            Dict mapping symbol to ATR series
        """
        atr_data = {}

        for symbol, df in data.items():
            if len(df) < period + 1:
                continue

            high = df['high']
            low = df['low']
            close = df['close']

            # True Range
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

            # ATR
            atr = tr.rolling(window=period).mean()
            atr_data[symbol] = atr

        return atr_data

    def _simulate_day(
        self,
        trade_date: date,
        data: Dict[str, pd.DataFrame],
        atr_data: Dict[str, pd.Series]
    ):
        """Simulate one trading day.

        Args:
            trade_date: Date to simulate
            data: Historical data dict
            atr_data: ATR data dict
        """
        daily_trades = []

        for symbol, df in data.items():
            # Get today's data
            date_mask = df['date'].dt.date == trade_date if 'date' in df.columns else False
            if not date_mask.any():
                continue

            today = df[date_mask].iloc[0]
            prev_idx = df[date_mask].index[0] - 1

            if prev_idx < 0:
                continue

            prev_day = df.iloc[prev_idx]

            # Calculate gap
            prev_close = prev_day['close']
            open_price = today['open']

            if prev_close <= 0:
                continue

            gap_pct = ((open_price - prev_close) / prev_close) * 100

            # Check if gap is significant
            if abs(gap_pct) < self.config.min_gap_pct:
                continue
            if abs(gap_pct) > self.config.max_gap_pct:
                continue

            # Get ATR
            atr_series = atr_data.get(symbol)
            if atr_series is None or prev_idx >= len(atr_series):
                continue

            atr = atr_series.iloc[prev_idx]
            if pd.isna(atr) or atr <= 0:
                continue

            # Determine direction
            direction = 'LONG' if gap_pct > 0 else 'SHORT'

            # Simulate gap confirmation (simple: open price is entry)
            entry_price = open_price * (1 + self.config.slippage_pct / 100)

            # Calculate stop loss
            if direction == 'LONG':
                stop_price = entry_price - (atr * self.config.stop_atr_multiplier)
            else:
                stop_price = entry_price + (atr * self.config.stop_atr_multiplier)

            # Calculate position size
            risk_amount = self._equity * (self.config.risk_per_trade_pct / 100)
            risk_per_share = abs(entry_price - stop_price)
            shares = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0

            if shares <= 0:
                continue

            # Determine risk tier
            atr_pct = (atr / entry_price) * 100
            if atr_pct < 2:
                risk_tier = 'LOW'
            elif atr_pct < 4:
                risk_tier = 'MEDIUM'
            else:
                risk_tier = 'HIGH'

            # Apply risk tier multiplier if enabled
            if self.config.use_risk_tiers:
                tier_multipliers = {'LOW': 1.0, 'MEDIUM': 0.75, 'HIGH': 0.5}
                shares = int(shares * tier_multipliers.get(risk_tier, 0.75))

            if shares <= 0:
                continue

            # Simulate exit (check if stop hit intraday)
            high = today['high']
            low = today['low']
            close = today['close']

            stop_hit = False
            if direction == 'LONG' and low <= stop_price:
                exit_price = stop_price * (1 - self.config.slippage_pct / 100)
                exit_reason = 'stop_loss'
                stop_hit = True
            elif direction == 'SHORT' and high >= stop_price:
                exit_price = stop_price * (1 + self.config.slippage_pct / 100)
                exit_reason = 'stop_loss'
                stop_hit = True
            else:
                exit_price = close * (1 - self.config.slippage_pct / 100 if direction == 'LONG'
                                      else 1 + self.config.slippage_pct / 100)
                exit_reason = 'eod'

            # Calculate P&L
            if direction == 'LONG':
                pnl = (exit_price - entry_price) * shares
            else:
                pnl = (entry_price - exit_price) * shares

            pnl -= self.config.commission * 2  # Entry and exit
            pnl_pct = (pnl / (entry_price * shares)) * 100 if shares > 0 else 0

            # Record trade
            trade = BacktestTrade(
                trade_date=trade_date,
                symbol=symbol,
                direction=direction,
                entry_price=entry_price,
                exit_price=exit_price,
                shares=shares,
                stop_price=stop_price,
                gap_pct=gap_pct,
                atr=atr,
                risk_tier=risk_tier,
                exit_reason=exit_reason,
                pnl=pnl,
                pnl_pct=pnl_pct,
                commission=self.config.commission * 2
            )
            daily_trades.append(trade)

        # Limit to max positions
        if len(daily_trades) > self.config.max_positions:
            # Sort by gap size and take top N
            daily_trades.sort(key=lambda t: abs(t.gap_pct), reverse=True)
            daily_trades = daily_trades[:self.config.max_positions]

        # Update equity and record trades
        daily_pnl = 0
        for trade in daily_trades:
            self._trades.append(trade)
            daily_pnl += trade.pnl

        self._equity += daily_pnl

        # Record equity curve
        self._equity_curve.append({
            'date': trade_date.isoformat(),
            'equity': self._equity,
            'daily_pnl': daily_pnl,
            'trades': len(daily_trades)
        })

    def _calculate_results(self) -> BacktestResults:
        """Calculate final backtest metrics.

        Returns:
            BacktestResults with all metrics
        """
        if not self._trades:
            return BacktestResults(
                config=self.config,
                trades=self._trades,
                equity_curve=self._equity_curve
            )

        # Basic metrics
        pnls = [t.pnl for t in self._trades]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]

        total_return = ((self._equity - self.config.initial_capital) /
                        self.config.initial_capital * 100)

        # CAGR
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        if years > 0 and self._equity > 0:
            cagr = ((self._equity / self.config.initial_capital) ** (1 / years) - 1) * 100
        else:
            cagr = 0

        # Sharpe ratio (simplified, assuming 0% risk-free)
        if pnls:
            daily_returns = [e['daily_pnl'] / self.config.initial_capital * 100
                            for e in self._equity_curve if e['daily_pnl'] != 0]
            if daily_returns:
                sharpe = (np.mean(daily_returns) / np.std(daily_returns) *
                          np.sqrt(252)) if np.std(daily_returns) > 0 else 0
            else:
                sharpe = 0
        else:
            sharpe = 0

        # Max drawdown
        peak = self.config.initial_capital
        max_dd = 0
        for e in self._equity_curve:
            if e['equity'] > peak:
                peak = e['equity']
            dd = (peak - e['equity']) / peak * 100
            max_dd = max(max_dd, dd)

        # Win rate and profit factor
        win_rate = len(winners) / len(pnls) * 100 if pnls else 0
        gross_profit = sum(winners) if winners else 0
        gross_loss = abs(sum(losers)) if losers else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

        # Streaks
        max_wins, max_losses = self._calculate_streaks(pnls)

        return BacktestResults(
            config=self.config,
            trades=self._trades,
            equity_curve=self._equity_curve,
            total_return=total_return,
            cagr=cagr,
            sharpe_ratio=sharpe,
            sortino_ratio=0,  # TODO: Calculate Sortino
            max_drawdown=max_dd,
            win_rate=win_rate,
            profit_factor=profit_factor,
            total_trades=len(self._trades),
            avg_trade=sum(pnls) / len(pnls) if pnls else 0,
            best_trade=max(pnls) if pnls else 0,
            worst_trade=min(pnls) if pnls else 0,
            avg_winner=sum(winners) / len(winners) if winners else 0,
            avg_loser=sum(losers) / len(losers) if losers else 0,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
            trading_days=len([e for e in self._equity_curve if e['trades'] > 0])
        )

    def _calculate_streaks(self, pnls: List[float]) -> Tuple[int, int]:
        """Calculate max winning and losing streaks."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for pnl in pnls:
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses


def run_backtest(
    start_date: str,
    end_date: str,
    symbols: Optional[List[str]] = None,
    initial_capital: float = 100000,
    **kwargs
) -> BacktestResults:
    """Convenience function to run a backtest.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        symbols: Optional list of symbols
        initial_capital: Starting capital
        **kwargs: Additional config parameters

    Returns:
        BacktestResults
    """
    config = BacktestConfig(
        start_date=datetime.strptime(start_date, '%Y-%m-%d').date(),
        end_date=datetime.strptime(end_date, '%Y-%m-%d').date(),
        initial_capital=initial_capital,
        **kwargs
    )

    backtester = GapTradingBacktester(config)
    return backtester.run(symbols)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Gap Trading Backtest')
    parser.add_argument('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, default=100000, help='Initial capital')
    parser.add_argument('--symbols', type=str, nargs='+', help='Symbols to backtest')

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    results = run_backtest(
        start_date=args.start,
        end_date=args.end,
        symbols=args.symbols,
        initial_capital=args.capital
    )

    print(results.summary())
