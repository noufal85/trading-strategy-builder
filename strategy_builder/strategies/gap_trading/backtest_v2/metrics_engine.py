"""Metrics Engine for Gap Trading Backtest.

Calculates comprehensive performance metrics from backtest trades.
"""

import logging
from dataclasses import dataclass, field
from datetime import date
from typing import List, Dict, Any, Optional
import numpy as np

from .config import BacktestConfig
from .trade_simulator import SimulatedTrade, ExitReason

logger = logging.getLogger(__name__)


@dataclass
class BacktestMetrics:
    """Comprehensive backtest performance metrics.

    Attributes:
        # Return metrics
        total_return_pct: Total return percentage
        cagr: Compound annual growth rate
        final_equity: Final account equity

        # Risk metrics
        max_drawdown_pct: Maximum drawdown percentage
        max_drawdown_dollars: Maximum drawdown in dollars
        sharpe_ratio: Annualized Sharpe ratio
        sortino_ratio: Sortino ratio (downside deviation)
        calmar_ratio: CAGR / Max Drawdown

        # Trade statistics
        total_trades: Total number of trades
        winning_trades: Number of winning trades
        losing_trades: Number of losing trades
        win_rate: Percentage of winning trades
        profit_factor: Gross profit / Gross loss
        gross_profit: Total profit from winners
        gross_loss: Total loss from losers
        avg_trade: Average P&L per trade
        avg_winner: Average winning trade
        avg_loser: Average losing trade
        largest_win: Largest winning trade
        largest_loss: Largest losing trade
        avg_win_loss_ratio: Avg winner / Avg loser

        # Streak statistics
        max_consecutive_wins: Longest winning streak
        max_consecutive_losses: Longest losing streak

        # Exit analysis
        stop_loss_exits: Number of stop-loss exits
        eod_exits: Number of EOD exits
        stop_loss_exit_pct: Percentage of stop-loss exits

        # Time analysis
        avg_hold_time_minutes: Average hold time in minutes
        avg_hold_time_winners: Avg hold time for winners
        avg_hold_time_losers: Avg hold time for losers

        # Direction analysis
        long_trades: Number of long trades
        short_trades: Number of short trades
        long_win_rate: Win rate for longs
        short_win_rate: Win rate for shorts
        long_pnl: Total P&L from longs
        short_pnl: Total P&L from shorts

        # Risk tier analysis
        low_tier_trades: Trades in LOW risk tier
        medium_tier_trades: Trades in MEDIUM risk tier
        high_tier_trades: Trades in HIGH risk tier
        low_tier_win_rate: Win rate for LOW tier
        medium_tier_win_rate: Win rate for MEDIUM tier
        high_tier_win_rate: Win rate for HIGH tier

        # Daily statistics
        trading_days: Number of days with trades
        avg_trades_per_day: Average trades per trading day
        best_day_pnl: Best single day P&L
        worst_day_pnl: Worst single day P&L
    """
    # Return metrics
    total_return_pct: float = 0.0
    cagr: float = 0.0
    final_equity: float = 0.0

    # Risk metrics
    max_drawdown_pct: float = 0.0
    max_drawdown_dollars: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0

    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    avg_trade: float = 0.0
    avg_winner: float = 0.0
    avg_loser: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0

    # Streak statistics
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # Exit analysis
    stop_loss_exits: int = 0
    eod_exits: int = 0
    stop_loss_exit_pct: float = 0.0

    # Time analysis
    avg_hold_time_minutes: float = 0.0
    avg_hold_time_winners: float = 0.0
    avg_hold_time_losers: float = 0.0

    # Direction analysis
    long_trades: int = 0
    short_trades: int = 0
    long_win_rate: float = 0.0
    short_win_rate: float = 0.0
    long_pnl: float = 0.0
    short_pnl: float = 0.0

    # Risk tier analysis
    low_tier_trades: int = 0
    medium_tier_trades: int = 0
    high_tier_trades: int = 0
    low_tier_win_rate: float = 0.0
    medium_tier_win_rate: float = 0.0
    high_tier_win_rate: float = 0.0

    # Daily statistics
    trading_days: int = 0
    avg_trades_per_day: float = 0.0
    best_day_pnl: float = 0.0
    worst_day_pnl: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_return_pct': round(self.total_return_pct, 2),
            'cagr': round(self.cagr, 2),
            'final_equity': round(self.final_equity, 2),
            'max_drawdown_pct': round(self.max_drawdown_pct, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'sortino_ratio': round(self.sortino_ratio, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 1),
            'profit_factor': round(self.profit_factor, 2),
            'avg_trade': round(self.avg_trade, 2),
            'avg_winner': round(self.avg_winner, 2),
            'avg_loser': round(self.avg_loser, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
            'stop_loss_exits': self.stop_loss_exits,
            'stop_loss_exit_pct': round(self.stop_loss_exit_pct, 1),
            'long_trades': self.long_trades,
            'short_trades': self.short_trades,
            'long_win_rate': round(self.long_win_rate, 1),
            'short_win_rate': round(self.short_win_rate, 1),
            'trading_days': self.trading_days,
        }

    def summary(self) -> str:
        """Generate text summary."""
        return f"""
{'='*60}
          GAP TRADING BACKTEST RESULTS
{'='*60}

RETURNS
  Total Return:     {self.total_return_pct:+.1f}%
  CAGR:             {self.cagr:.1f}%
  Final Equity:     ${self.final_equity:,.2f}
  Max Drawdown:     {self.max_drawdown_pct:.1f}%
  Sharpe Ratio:     {self.sharpe_ratio:.2f}
  Sortino Ratio:    {self.sortino_ratio:.2f}

TRADING STATISTICS
  Total Trades:     {self.total_trades}
  Win Rate:         {self.win_rate:.1f}%
  Profit Factor:    {self.profit_factor:.2f}
  Avg Trade:        ${self.avg_trade:,.2f}
  Avg Winner:       ${self.avg_winner:,.2f}
  Avg Loser:        ${self.avg_loser:,.2f}
  Largest Win:      ${self.largest_win:,.2f}
  Largest Loss:     ${self.largest_loss:,.2f}

EXIT ANALYSIS
  Stop-Loss Exits:  {self.stop_loss_exits} ({self.stop_loss_exit_pct:.1f}%)
  EOD Exits:        {self.eod_exits} ({100-self.stop_loss_exit_pct:.1f}%)
  Avg Hold Time:    {self.avg_hold_time_minutes:.0f} minutes

DIRECTION ANALYSIS
  Long Trades:      {self.long_trades} (Win: {self.long_win_rate:.1f}%)
  Short Trades:     {self.short_trades} (Win: {self.short_win_rate:.1f}%)
  Long P&L:         ${self.long_pnl:,.2f}
  Short P&L:        ${self.short_pnl:,.2f}

STREAKS
  Max Wins:         {self.max_consecutive_wins}
  Max Losses:       {self.max_consecutive_losses}
  Trading Days:     {self.trading_days}
{'='*60}
"""


class MetricsEngine:
    """Engine for calculating backtest performance metrics.

    Calculates comprehensive trading statistics, risk metrics,
    and performance analysis from a list of simulated trades.
    """

    def __init__(self, config: BacktestConfig):
        """Initialize metrics engine.

        Args:
            config: Backtest configuration
        """
        self.config = config

    def calculate_all(
        self,
        trades: List[SimulatedTrade],
        equity_curve: Optional[List[Dict]] = None
    ) -> BacktestMetrics:
        """Calculate all performance metrics.

        Args:
            trades: List of simulated trades
            equity_curve: Optional daily equity values

        Returns:
            BacktestMetrics with all calculated values
        """
        if not trades:
            return BacktestMetrics(final_equity=self.config.initial_capital)

        metrics = BacktestMetrics()

        # Calculate trade statistics
        self._calculate_trade_stats(trades, metrics)

        # Calculate return metrics
        self._calculate_returns(trades, metrics)

        # Calculate risk metrics
        self._calculate_risk_metrics(trades, equity_curve, metrics)

        # Calculate exit analysis
        self._calculate_exit_analysis(trades, metrics)

        # Calculate direction analysis
        self._calculate_direction_analysis(trades, metrics)

        # Calculate risk tier analysis
        self._calculate_tier_analysis(trades, metrics)

        # Calculate daily statistics
        self._calculate_daily_stats(trades, metrics)

        # Calculate streaks
        self._calculate_streaks(trades, metrics)

        return metrics

    def _calculate_trade_stats(
        self,
        trades: List[SimulatedTrade],
        metrics: BacktestMetrics
    ):
        """Calculate basic trade statistics."""
        pnls = [t.pnl for t in trades]
        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]

        metrics.total_trades = len(trades)
        metrics.winning_trades = len(winners)
        metrics.losing_trades = len(losers)

        # Win rate
        metrics.win_rate = len(winners) / len(trades) * 100 if trades else 0

        # Gross profit/loss
        metrics.gross_profit = sum(t.pnl for t in winners) if winners else 0
        metrics.gross_loss = abs(sum(t.pnl for t in losers)) if losers else 0

        # Profit factor
        metrics.profit_factor = (
            metrics.gross_profit / metrics.gross_loss
            if metrics.gross_loss > 0 else 0
        )

        # Average trade
        metrics.avg_trade = sum(pnls) / len(pnls) if pnls else 0
        metrics.avg_winner = sum(t.pnl for t in winners) / len(winners) if winners else 0
        metrics.avg_loser = sum(t.pnl for t in losers) / len(losers) if losers else 0

        # Largest win/loss
        metrics.largest_win = max(pnls) if pnls else 0
        metrics.largest_loss = min(pnls) if pnls else 0

        # Win/loss ratio
        metrics.avg_win_loss_ratio = (
            abs(metrics.avg_winner / metrics.avg_loser)
            if metrics.avg_loser != 0 else 0
        )

    def _calculate_returns(
        self,
        trades: List[SimulatedTrade],
        metrics: BacktestMetrics
    ):
        """Calculate return metrics."""
        total_pnl = sum(t.pnl for t in trades)
        metrics.final_equity = self.config.initial_capital + total_pnl

        # Total return
        metrics.total_return_pct = (
            total_pnl / self.config.initial_capital * 100
        )

        # CAGR
        days = (self.config.end_date - self.config.start_date).days
        years = days / 365.25
        if years > 0 and metrics.final_equity > 0:
            metrics.cagr = (
                (metrics.final_equity / self.config.initial_capital) ** (1 / years) - 1
            ) * 100
        else:
            metrics.cagr = 0

    def _calculate_risk_metrics(
        self,
        trades: List[SimulatedTrade],
        equity_curve: Optional[List[Dict]],
        metrics: BacktestMetrics
    ):
        """Calculate risk-adjusted metrics."""
        # Build equity curve if not provided
        if equity_curve is None:
            equity_curve = self._build_equity_curve(trades)

        if not equity_curve:
            return

        # Extract equity values
        equity_values = [e['equity'] for e in equity_curve]

        # Max drawdown
        peak = equity_values[0]
        max_dd = 0
        max_dd_dollars = 0

        for equity in equity_values:
            if equity > peak:
                peak = equity
            dd_pct = (peak - equity) / peak * 100
            dd_dollars = peak - equity
            if dd_pct > max_dd:
                max_dd = dd_pct
                max_dd_dollars = dd_dollars

        metrics.max_drawdown_pct = max_dd
        metrics.max_drawdown_dollars = max_dd_dollars

        # Daily returns for Sharpe/Sortino
        daily_pnls = [e.get('daily_pnl', 0) for e in equity_curve]
        daily_returns = [
            pnl / self.config.initial_capital * 100
            for pnl in daily_pnls if pnl != 0
        ]

        if daily_returns and len(daily_returns) > 1:
            mean_return = np.mean(daily_returns)
            std_return = np.std(daily_returns)

            # Sharpe ratio (annualized, 0% risk-free)
            if std_return > 0:
                metrics.sharpe_ratio = mean_return / std_return * np.sqrt(252)
            else:
                metrics.sharpe_ratio = 0

            # Sortino ratio (downside deviation)
            negative_returns = [r for r in daily_returns if r < 0]
            if negative_returns:
                downside_std = np.std(negative_returns)
                if downside_std > 0:
                    metrics.sortino_ratio = mean_return / downside_std * np.sqrt(252)

        # Calmar ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown_pct

    def _calculate_exit_analysis(
        self,
        trades: List[SimulatedTrade],
        metrics: BacktestMetrics
    ):
        """Analyze exit reasons."""
        stop_exits = [t for t in trades if t.exit_reason == ExitReason.STOP_LOSS]
        eod_exits = [t for t in trades if t.exit_reason == ExitReason.EOD_CLOSE]

        metrics.stop_loss_exits = len(stop_exits)
        metrics.eod_exits = len(eod_exits)
        metrics.stop_loss_exit_pct = len(stop_exits) / len(trades) * 100 if trades else 0

        # Hold time analysis
        hold_times = [t.hold_duration_minutes for t in trades]
        metrics.avg_hold_time_minutes = np.mean(hold_times) if hold_times else 0

        winners = [t for t in trades if t.pnl > 0]
        losers = [t for t in trades if t.pnl < 0]

        if winners:
            metrics.avg_hold_time_winners = np.mean([t.hold_duration_minutes for t in winners])
        if losers:
            metrics.avg_hold_time_losers = np.mean([t.hold_duration_minutes for t in losers])

    def _calculate_direction_analysis(
        self,
        trades: List[SimulatedTrade],
        metrics: BacktestMetrics
    ):
        """Analyze long vs short performance."""
        longs = [t for t in trades if t.direction == 'LONG']
        shorts = [t for t in trades if t.direction == 'SHORT']

        metrics.long_trades = len(longs)
        metrics.short_trades = len(shorts)

        long_winners = [t for t in longs if t.pnl > 0]
        short_winners = [t for t in shorts if t.pnl > 0]

        metrics.long_win_rate = len(long_winners) / len(longs) * 100 if longs else 0
        metrics.short_win_rate = len(short_winners) / len(shorts) * 100 if shorts else 0

        metrics.long_pnl = sum(t.pnl for t in longs)
        metrics.short_pnl = sum(t.pnl for t in shorts)

    def _calculate_tier_analysis(
        self,
        trades: List[SimulatedTrade],
        metrics: BacktestMetrics
    ):
        """Analyze performance by risk tier."""
        low_tier = [t for t in trades if t.risk_tier == 'LOW']
        medium_tier = [t for t in trades if t.risk_tier == 'MEDIUM']
        high_tier = [t for t in trades if t.risk_tier == 'HIGH']

        metrics.low_tier_trades = len(low_tier)
        metrics.medium_tier_trades = len(medium_tier)
        metrics.high_tier_trades = len(high_tier)

        low_winners = [t for t in low_tier if t.pnl > 0]
        medium_winners = [t for t in medium_tier if t.pnl > 0]
        high_winners = [t for t in high_tier if t.pnl > 0]

        metrics.low_tier_win_rate = len(low_winners) / len(low_tier) * 100 if low_tier else 0
        metrics.medium_tier_win_rate = len(medium_winners) / len(medium_tier) * 100 if medium_tier else 0
        metrics.high_tier_win_rate = len(high_winners) / len(high_tier) * 100 if high_tier else 0

    def _calculate_daily_stats(
        self,
        trades: List[SimulatedTrade],
        metrics: BacktestMetrics
    ):
        """Calculate daily trading statistics."""
        # Group trades by date
        by_date: Dict[date, List[SimulatedTrade]] = {}
        for trade in trades:
            if trade.trade_date not in by_date:
                by_date[trade.trade_date] = []
            by_date[trade.trade_date].append(trade)

        metrics.trading_days = len(by_date)
        metrics.avg_trades_per_day = len(trades) / len(by_date) if by_date else 0

        # Best/worst day
        daily_pnls = [sum(t.pnl for t in trades) for trades in by_date.values()]
        metrics.best_day_pnl = max(daily_pnls) if daily_pnls else 0
        metrics.worst_day_pnl = min(daily_pnls) if daily_pnls else 0

    def _calculate_streaks(
        self,
        trades: List[SimulatedTrade],
        metrics: BacktestMetrics
    ):
        """Calculate winning and losing streaks."""
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for trade in trades:
            if trade.pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif trade.pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        metrics.max_consecutive_wins = max_wins
        metrics.max_consecutive_losses = max_losses

    def _build_equity_curve(
        self,
        trades: List[SimulatedTrade]
    ) -> List[Dict]:
        """Build equity curve from trades.

        Args:
            trades: List of trades

        Returns:
            List of daily equity snapshots
        """
        equity_curve = []
        equity = self.config.initial_capital

        # Group by date
        by_date: Dict[date, List[SimulatedTrade]] = {}
        for trade in trades:
            if trade.trade_date not in by_date:
                by_date[trade.trade_date] = []
            by_date[trade.trade_date].append(trade)

        # Build curve
        for trade_date in sorted(by_date.keys()):
            daily_trades = by_date[trade_date]
            daily_pnl = sum(t.pnl for t in daily_trades)
            equity += daily_pnl

            equity_curve.append({
                'date': trade_date.isoformat(),
                'equity': equity,
                'daily_pnl': daily_pnl,
                'trades': len(daily_trades)
            })

        return equity_curve
