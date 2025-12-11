"""Reporting Module for Gap Trading Strategy.

Generates daily reports, performance analytics, and trade summaries.

Features:
- Daily P&L summary
- Win rate and risk metrics
- Sector/tier breakdown
- Trade history export
- Performance visualization data
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional, List, Dict, Any, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


class ReportPeriod(str, Enum):
    """Report time period."""
    DAILY = 'daily'
    WEEKLY = 'weekly'
    MONTHLY = 'monthly'
    QUARTERLY = 'quarterly'
    YEARLY = 'yearly'
    ALL_TIME = 'all_time'


@dataclass
class TradeMetrics:
    """Metrics for a set of trades.

    Attributes:
        total_trades: Total number of trades
        winning_trades: Number of profitable trades
        losing_trades: Number of unprofitable trades
        win_rate: Percentage of winning trades
        gross_profit: Total profit from winners
        gross_loss: Total loss from losers
        net_pnl: Net profit/loss
        avg_win: Average profit per winning trade
        avg_loss: Average loss per losing trade
        profit_factor: Gross profit / Gross loss
        largest_win: Largest single win
        largest_loss: Largest single loss
        avg_trade: Average P&L per trade
        max_consecutive_wins: Max winning streak
        max_consecutive_losses: Max losing streak
    """
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(self.win_rate, 2),
            'gross_profit': round(self.gross_profit, 2),
            'gross_loss': round(self.gross_loss, 2),
            'net_pnl': round(self.net_pnl, 2),
            'avg_win': round(self.avg_win, 2),
            'avg_loss': round(self.avg_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'avg_trade': round(self.avg_trade, 2),
            'max_consecutive_wins': self.max_consecutive_wins,
            'max_consecutive_losses': self.max_consecutive_losses,
        }


@dataclass
class RiskMetrics:
    """Risk-related metrics.

    Attributes:
        max_drawdown: Maximum drawdown percentage
        max_drawdown_amount: Maximum drawdown in dollars
        sharpe_ratio: Sharpe ratio (if enough data)
        sortino_ratio: Sortino ratio (if enough data)
        risk_reward_ratio: Average risk/reward ratio
        avg_risk_per_trade: Average dollar risk per trade
        total_risk_taken: Cumulative risk taken
    """
    max_drawdown: float = 0.0
    max_drawdown_amount: float = 0.0
    sharpe_ratio: Optional[float] = None
    sortino_ratio: Optional[float] = None
    risk_reward_ratio: float = 0.0
    avg_risk_per_trade: float = 0.0
    total_risk_taken: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'max_drawdown_pct': round(self.max_drawdown, 2),
            'max_drawdown_amount': round(self.max_drawdown_amount, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2) if self.sharpe_ratio else None,
            'sortino_ratio': round(self.sortino_ratio, 2) if self.sortino_ratio else None,
            'risk_reward_ratio': round(self.risk_reward_ratio, 2),
            'avg_risk_per_trade': round(self.avg_risk_per_trade, 2),
            'total_risk_taken': round(self.total_risk_taken, 2),
        }


@dataclass
class DailyReport:
    """Daily trading report.

    Attributes:
        report_date: Date of report
        trade_metrics: Trade performance metrics
        risk_metrics: Risk metrics
        by_direction: Breakdown by LONG/SHORT
        by_risk_tier: Breakdown by risk tier
        by_symbol: Breakdown by symbol
        signals_generated: Number of signals generated
        signals_executed: Number of signals executed
        stops_triggered: Number of stop-losses triggered
        eod_closes: Number of EOD closes
        account_value_start: Starting account value
        account_value_end: Ending account value
    """
    report_date: date
    trade_metrics: TradeMetrics = field(default_factory=TradeMetrics)
    risk_metrics: RiskMetrics = field(default_factory=RiskMetrics)
    by_direction: Dict[str, TradeMetrics] = field(default_factory=dict)
    by_risk_tier: Dict[str, TradeMetrics] = field(default_factory=dict)
    by_symbol: Dict[str, TradeMetrics] = field(default_factory=dict)
    signals_generated: int = 0
    signals_executed: int = 0
    stops_triggered: int = 0
    eod_closes: int = 0
    account_value_start: float = 0.0
    account_value_end: float = 0.0
    generated_at: datetime = field(default_factory=datetime.now)

    @property
    def daily_return_pct(self) -> float:
        """Calculate daily return percentage."""
        if self.account_value_start == 0:
            return 0.0
        return ((self.account_value_end - self.account_value_start) /
                self.account_value_start * 100)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'report_date': self.report_date.isoformat(),
            'trade_metrics': self.trade_metrics.to_dict(),
            'risk_metrics': self.risk_metrics.to_dict(),
            'by_direction': {k: v.to_dict() for k, v in self.by_direction.items()},
            'by_risk_tier': {k: v.to_dict() for k, v in self.by_risk_tier.items()},
            'signals_generated': self.signals_generated,
            'signals_executed': self.signals_executed,
            'stops_triggered': self.stops_triggered,
            'eod_closes': self.eod_closes,
            'daily_return_pct': round(self.daily_return_pct, 2),
            'generated_at': self.generated_at.isoformat(),
        }


class ReportGenerator:
    """Generates trading reports and analytics.

    Attributes:
        db_conn: Database connection
    """

    def __init__(self, db_conn: Any):
        """Initialize ReportGenerator.

        Args:
            db_conn: Database connection
        """
        self.db_conn = db_conn
        logger.info("ReportGenerator initialized")

    def generate_daily_report(self, report_date: Optional[date] = None) -> DailyReport:
        """Generate daily report.

        Args:
            report_date: Date to report on (defaults to today)

        Returns:
            DailyReport object
        """
        if report_date is None:
            report_date = date.today()

        logger.info(f"Generating daily report for {report_date}")

        # Get trades for the day
        trades = self._get_trades_for_date(report_date)

        # Calculate metrics
        trade_metrics = self._calculate_trade_metrics(trades)
        risk_metrics = self._calculate_risk_metrics(trades)

        # Breakdowns
        by_direction = self._breakdown_by_direction(trades)
        by_risk_tier = self._breakdown_by_risk_tier(trades)
        by_symbol = self._breakdown_by_symbol(trades)

        # Get signal counts
        signals_generated, signals_executed = self._get_signal_counts(report_date)

        # Get close reasons
        stops_triggered, eod_closes = self._get_close_reasons(trades)

        # Get account values (if available)
        account_start, account_end = self._get_account_values(report_date)

        report = DailyReport(
            report_date=report_date,
            trade_metrics=trade_metrics,
            risk_metrics=risk_metrics,
            by_direction=by_direction,
            by_risk_tier=by_risk_tier,
            by_symbol=by_symbol,
            signals_generated=signals_generated,
            signals_executed=signals_executed,
            stops_triggered=stops_triggered,
            eod_closes=eod_closes,
            account_value_start=account_start,
            account_value_end=account_end,
        )

        # Store report in database
        self._store_report(report)

        logger.info(
            f"Daily report generated: {trade_metrics.total_trades} trades, "
            f"P&L=${trade_metrics.net_pnl:.2f}, Win rate={trade_metrics.win_rate:.1f}%"
        )

        return report

    def get_performance_summary(
        self,
        period: ReportPeriod = ReportPeriod.MONTHLY,
        end_date: Optional[date] = None
    ) -> Dict[str, Any]:
        """Get performance summary for a period.

        Args:
            period: Time period
            end_date: End date (defaults to today)

        Returns:
            Performance summary dict
        """
        if end_date is None:
            end_date = date.today()

        # Calculate start date
        if period == ReportPeriod.DAILY:
            start_date = end_date
        elif period == ReportPeriod.WEEKLY:
            start_date = end_date - timedelta(days=7)
        elif period == ReportPeriod.MONTHLY:
            start_date = end_date - timedelta(days=30)
        elif period == ReportPeriod.QUARTERLY:
            start_date = end_date - timedelta(days=90)
        elif period == ReportPeriod.YEARLY:
            start_date = end_date - timedelta(days=365)
        else:
            start_date = date(2020, 1, 1)  # All time

        trades = self._get_trades_for_period(start_date, end_date)
        metrics = self._calculate_trade_metrics(trades)
        risk = self._calculate_risk_metrics(trades)

        # Get daily P&L series for charting
        daily_pnl = self._get_daily_pnl_series(start_date, end_date)

        return {
            'period': period.value,
            'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(),
            'trade_metrics': metrics.to_dict(),
            'risk_metrics': risk.to_dict(),
            'daily_pnl': daily_pnl,
            'trading_days': len(daily_pnl),
        }

    def format_telegram_report(self, report: DailyReport) -> str:
        """Format report for Telegram message.

        Args:
            report: DailyReport to format

        Returns:
            Formatted string for Telegram
        """
        metrics = report.trade_metrics

        # Determine overall emoji
        if metrics.net_pnl > 0:
            pnl_emoji = '🟢'
        elif metrics.net_pnl < 0:
            pnl_emoji = '🔴'
        else:
            pnl_emoji = '⚪'

        message = f"""
{pnl_emoji} *Gap Trading Daily Report*
{report.report_date.strftime('%A, %B %d, %Y')}

*Performance*
• Net P&L: ${metrics.net_pnl:+,.2f}
• Daily Return: {report.daily_return_pct:+.2f}%
• Trades: {metrics.total_trades}
• Win Rate: {metrics.win_rate:.1f}%

*Trade Breakdown*
• Winners: {metrics.winning_trades} (${metrics.gross_profit:,.2f})
• Losers: {metrics.losing_trades} (${metrics.gross_loss:,.2f})
• Avg Win: ${metrics.avg_win:,.2f}
• Avg Loss: ${metrics.avg_loss:,.2f}
"""

        # Add direction breakdown if trades exist
        if report.by_direction:
            message += "\n*By Direction*\n"
            for direction, m in report.by_direction.items():
                emoji = '📈' if direction == 'LONG' else '📉'
                message += f"{emoji} {direction}: {m.total_trades} trades, ${m.net_pnl:+,.2f}\n"

        # Add risk tier breakdown
        if report.by_risk_tier:
            message += "\n*By Risk Tier*\n"
            for tier, m in report.by_risk_tier.items():
                message += f"• {tier}: {m.total_trades} trades, ${m.net_pnl:+,.2f}\n"

        # Add execution summary
        message += f"""
*Execution*
• Signals Generated: {report.signals_generated}
• Trades Executed: {report.signals_executed}
• Stops Triggered: {report.stops_triggered}
• EOD Closes: {report.eod_closes}
"""

        # Add risk metrics if significant
        if report.risk_metrics.max_drawdown > 0:
            message += f"\n*Risk*\n• Max Drawdown: {report.risk_metrics.max_drawdown:.2f}%\n"

        return message

    def _get_trades_for_date(self, trade_date: date) -> List[Dict[str, Any]]:
        """Get closed trades for a specific date."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT
                id, symbol, side, quantity, entry_price, exit_price,
                realized_pnl, risk_tier, exit_reason, atr_at_entry,
                entry_time, exit_time
            FROM gap_trading.positions
            WHERE trade_date = %s
              AND status IN ('STOPPED', 'EOD_CLOSED', 'MANUAL_CLOSED')
            ORDER BY exit_time
        """, (trade_date,))

        columns = ['id', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price',
                   'realized_pnl', 'risk_tier', 'exit_reason', 'atr_at_entry',
                   'entry_time', 'exit_time']
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _get_trades_for_period(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get closed trades for a date range."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT
                id, symbol, side, quantity, entry_price, exit_price,
                realized_pnl, risk_tier, exit_reason, atr_at_entry,
                entry_time, exit_time, trade_date
            FROM gap_trading.positions
            WHERE trade_date BETWEEN %s AND %s
              AND status IN ('STOPPED', 'EOD_CLOSED', 'MANUAL_CLOSED')
            ORDER BY exit_time
        """, (start_date, end_date))

        columns = ['id', 'symbol', 'side', 'quantity', 'entry_price', 'exit_price',
                   'realized_pnl', 'risk_tier', 'exit_reason', 'atr_at_entry',
                   'entry_time', 'exit_time', 'trade_date']
        return [dict(zip(columns, row)) for row in cursor.fetchall()]

    def _calculate_trade_metrics(self, trades: List[Dict[str, Any]]) -> TradeMetrics:
        """Calculate trade metrics from list of trades."""
        if not trades:
            return TradeMetrics()

        winners = [t for t in trades if (t.get('realized_pnl') or 0) > 0]
        losers = [t for t in trades if (t.get('realized_pnl') or 0) < 0]

        gross_profit = sum(t.get('realized_pnl', 0) for t in winners)
        gross_loss = abs(sum(t.get('realized_pnl', 0) for t in losers))
        net_pnl = gross_profit - gross_loss

        # Calculate streaks
        max_wins, max_losses = self._calculate_streaks(trades)

        return TradeMetrics(
            total_trades=len(trades),
            winning_trades=len(winners),
            losing_trades=len(losers),
            win_rate=(len(winners) / len(trades) * 100) if trades else 0,
            gross_profit=gross_profit,
            gross_loss=gross_loss,
            net_pnl=net_pnl,
            avg_win=gross_profit / len(winners) if winners else 0,
            avg_loss=gross_loss / len(losers) if losers else 0,
            profit_factor=gross_profit / gross_loss if gross_loss > 0 else 0,
            largest_win=max(t.get('realized_pnl', 0) for t in trades) if trades else 0,
            largest_loss=abs(min(t.get('realized_pnl', 0) for t in trades)) if trades else 0,
            avg_trade=net_pnl / len(trades) if trades else 0,
            max_consecutive_wins=max_wins,
            max_consecutive_losses=max_losses,
        )

    def _calculate_risk_metrics(self, trades: List[Dict[str, Any]]) -> RiskMetrics:
        """Calculate risk metrics from trades."""
        if not trades:
            return RiskMetrics()

        # Calculate drawdown
        cumulative_pnl = []
        running_pnl = 0
        for t in trades:
            running_pnl += t.get('realized_pnl', 0)
            cumulative_pnl.append(running_pnl)

        peak = 0
        max_dd = 0
        max_dd_amount = 0
        for pnl in cumulative_pnl:
            if pnl > peak:
                peak = pnl
            dd = peak - pnl
            if dd > max_dd_amount:
                max_dd_amount = dd
                max_dd = (dd / peak * 100) if peak > 0 else 0

        # Calculate risk per trade
        risks = []
        for t in trades:
            atr = t.get('atr_at_entry', 0)
            qty = abs(t.get('quantity', 0))
            if atr and qty:
                risks.append(atr * qty)

        avg_risk = sum(risks) / len(risks) if risks else 0
        total_risk = sum(risks)

        # Risk/reward
        avg_win = sum(t.get('realized_pnl', 0) for t in trades if t.get('realized_pnl', 0) > 0)
        avg_loss = abs(sum(t.get('realized_pnl', 0) for t in trades if t.get('realized_pnl', 0) < 0))
        winners = len([t for t in trades if t.get('realized_pnl', 0) > 0])
        losers = len([t for t in trades if t.get('realized_pnl', 0) < 0])

        if losers > 0 and winners > 0:
            rr = (avg_win / winners) / (avg_loss / losers)
        else:
            rr = 0

        return RiskMetrics(
            max_drawdown=max_dd,
            max_drawdown_amount=max_dd_amount,
            risk_reward_ratio=rr,
            avg_risk_per_trade=avg_risk,
            total_risk_taken=total_risk,
        )

    def _calculate_streaks(self, trades: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Calculate max winning and losing streaks."""
        if not trades:
            return 0, 0

        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0

        for t in trades:
            pnl = t.get('realized_pnl', 0)
            if pnl > 0:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            elif pnl < 0:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)

        return max_wins, max_losses

    def _breakdown_by_direction(self, trades: List[Dict[str, Any]]) -> Dict[str, TradeMetrics]:
        """Break down metrics by trade direction."""
        long_trades = [t for t in trades if t.get('side') == 'LONG']
        short_trades = [t for t in trades if t.get('side') == 'SHORT']

        return {
            'LONG': self._calculate_trade_metrics(long_trades),
            'SHORT': self._calculate_trade_metrics(short_trades),
        }

    def _breakdown_by_risk_tier(self, trades: List[Dict[str, Any]]) -> Dict[str, TradeMetrics]:
        """Break down metrics by risk tier."""
        tiers = {}
        for t in trades:
            tier = t.get('risk_tier', 'UNKNOWN')
            if tier not in tiers:
                tiers[tier] = []
            tiers[tier].append(t)

        return {tier: self._calculate_trade_metrics(tier_trades)
                for tier, tier_trades in tiers.items()}

    def _breakdown_by_symbol(self, trades: List[Dict[str, Any]]) -> Dict[str, TradeMetrics]:
        """Break down metrics by symbol (top 10)."""
        symbols = {}
        for t in trades:
            symbol = t.get('symbol', 'UNKNOWN')
            if symbol not in symbols:
                symbols[symbol] = []
            symbols[symbol].append(t)

        # Sort by trade count and take top 10
        sorted_symbols = sorted(symbols.items(), key=lambda x: len(x[1]), reverse=True)[:10]

        return {symbol: self._calculate_trade_metrics(symbol_trades)
                for symbol, symbol_trades in sorted_symbols}

    def _get_signal_counts(self, report_date: date) -> Tuple[int, int]:
        """Get signal counts for a date."""
        cursor = self.db_conn.cursor()

        cursor.execute("""
            SELECT COUNT(*) FROM gap_trading.trade_signals
            WHERE signal_date = %s
        """, (report_date,))
        generated = cursor.fetchone()[0] or 0

        cursor.execute("""
            SELECT COUNT(*) FROM gap_trading.positions
            WHERE trade_date = %s
        """, (report_date,))
        executed = cursor.fetchone()[0] or 0

        return generated, executed

    def _get_close_reasons(self, trades: List[Dict[str, Any]]) -> Tuple[int, int]:
        """Count trades by close reason."""
        stops = len([t for t in trades if t.get('exit_reason') == 'stop_loss'])
        eod = len([t for t in trades if t.get('exit_reason') in ('end_of_day', 'eod')])
        return stops, eod

    def _get_account_values(self, report_date: date) -> Tuple[float, float]:
        """Get starting and ending account values (placeholder)."""
        # TODO: Implement actual account value tracking
        return 100000.0, 100000.0

    def _get_daily_pnl_series(self, start_date: date, end_date: date) -> List[Dict[str, Any]]:
        """Get daily P&L series for charting."""
        cursor = self.db_conn.cursor()
        cursor.execute("""
            SELECT
                trade_date,
                SUM(realized_pnl) as daily_pnl,
                COUNT(*) as trade_count
            FROM gap_trading.positions
            WHERE trade_date BETWEEN %s AND %s
              AND status IN ('STOPPED', 'EOD_CLOSED', 'MANUAL_CLOSED')
            GROUP BY trade_date
            ORDER BY trade_date
        """, (start_date, end_date))

        return [
            {
                'date': row[0].isoformat(),
                'pnl': float(row[1] or 0),
                'trades': row[2]
            }
            for row in cursor.fetchall()
        ]

    def _store_report(self, report: DailyReport):
        """Store report in database."""
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO gap_trading.daily_reports
                (report_date, total_trades, winning_trades, losing_trades,
                 net_pnl, win_rate, gross_profit, gross_loss, report_data, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (report_date) DO UPDATE SET
                    total_trades = EXCLUDED.total_trades,
                    winning_trades = EXCLUDED.winning_trades,
                    losing_trades = EXCLUDED.losing_trades,
                    net_pnl = EXCLUDED.net_pnl,
                    win_rate = EXCLUDED.win_rate,
                    gross_profit = EXCLUDED.gross_profit,
                    gross_loss = EXCLUDED.gross_loss,
                    report_data = EXCLUDED.report_data,
                    created_at = EXCLUDED.created_at
            """, (
                report.report_date,
                report.trade_metrics.total_trades,
                report.trade_metrics.winning_trades,
                report.trade_metrics.losing_trades,
                report.trade_metrics.net_pnl,
                report.trade_metrics.win_rate,
                report.trade_metrics.gross_profit,
                report.trade_metrics.gross_loss,
                str(report.to_dict()),
                report.generated_at
            ))
            self.db_conn.commit()
        except Exception as e:
            logger.error(f"Failed to store report: {e}")
            self.db_conn.rollback()
