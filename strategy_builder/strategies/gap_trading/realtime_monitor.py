"""Realtime Stop-Loss Monitor for Gap Trading Strategy.

Continuous process that monitors open positions and triggers
stop-loss orders when price thresholds are breached.

Features:
- Price checks every 60 seconds (configurable)
- ATR-based stop-loss monitoring
- End-of-day position closing at 3:55 PM ET
- Audit trail in price_checks table
- Health check endpoint
"""

import logging
import time
import signal
import sys
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from enum import Enum
from typing import Optional, List, Dict, Any, Callable
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

# Eastern timezone for market hours
ET = ZoneInfo('America/New_York')


class MonitorStatus(str, Enum):
    """Monitor process status."""
    STARTING = 'starting'
    RUNNING = 'running'
    PAUSED = 'paused'
    STOPPING = 'stopping'
    STOPPED = 'stopped'
    ERROR = 'error'


class CloseReason(str, Enum):
    """Reason for position close."""
    STOP_LOSS = 'stop_loss'
    END_OF_DAY = 'end_of_day'
    MANUAL = 'manual'
    ERROR = 'error'


@dataclass
class PriceCheck:
    """Record of a price check.

    Attributes:
        id: Database ID
        symbol: Stock ticker
        check_time: Timestamp of check
        current_price: Price at check time
        stop_price: Current stop-loss level
        is_stop_hit: Whether stop was triggered
        position_id: Related position ID
    """
    symbol: str
    check_time: datetime
    current_price: float
    stop_price: float
    is_stop_hit: bool
    position_id: Optional[int] = None
    id: Optional[int] = None


@dataclass
class MonitorConfig:
    """Configuration for realtime monitor.

    Attributes:
        check_interval: Seconds between price checks
        eod_close_time: Time to close all positions (HH:MM ET)
        market_open_time: Market open time (HH:MM ET)
        market_close_time: Market close time (HH:MM ET)
        max_consecutive_errors: Errors before alerting
        health_check_port: Port for health endpoint (0 = disabled)
    """
    check_interval: int = 60
    eod_close_time: str = '15:55'  # 3:55 PM ET
    market_open_time: str = '09:30'
    market_close_time: str = '16:00'
    max_consecutive_errors: int = 5
    health_check_port: int = 8080

    @property
    def eod_close_datetime(self) -> datetime:
        """Get today's EOD close time as datetime."""
        today = datetime.now(ET).date()
        hour, minute = map(int, self.eod_close_time.split(':'))
        return datetime(today.year, today.month, today.day, hour, minute, tzinfo=ET)

    @property
    def market_open_datetime(self) -> datetime:
        """Get today's market open as datetime."""
        today = datetime.now(ET).date()
        hour, minute = map(int, self.market_open_time.split(':'))
        return datetime(today.year, today.month, today.day, hour, minute, tzinfo=ET)

    @property
    def market_close_datetime(self) -> datetime:
        """Get today's market close as datetime."""
        today = datetime.now(ET).date()
        hour, minute = map(int, self.market_close_time.split(':'))
        return datetime(today.year, today.month, today.day, hour, minute, tzinfo=ET)


@dataclass
class MonitorState:
    """Current state of the monitor.

    Attributes:
        status: Current status
        start_time: When monitor started
        last_check_time: Last successful price check
        checks_count: Total checks performed
        stops_triggered: Number of stops triggered today
        errors_count: Consecutive error count
        positions_monitored: Current position count
    """
    status: MonitorStatus = MonitorStatus.STOPPED
    start_time: Optional[datetime] = None
    last_check_time: Optional[datetime] = None
    checks_count: int = 0
    stops_triggered: int = 0
    errors_count: int = 0
    positions_monitored: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for health check."""
        return {
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'last_check_time': self.last_check_time.isoformat() if self.last_check_time else None,
            'checks_count': self.checks_count,
            'stops_triggered': self.stops_triggered,
            'errors_count': self.errors_count,
            'positions_monitored': self.positions_monitored,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }


class RealtimeStopLossMonitor:
    """Monitors open positions and triggers stop-loss orders.

    This is designed to run as a continuous process, checking prices
    at regular intervals and closing positions when stops are hit.

    Attributes:
        config: Monitor configuration
        state: Current monitor state
    """

    def __init__(
        self,
        tradier_client: Any,
        db_conn: Any,
        quote_provider: Any = None,
        config: Optional[MonitorConfig] = None,
        on_stop_triggered: Optional[Callable] = None,
        on_eod_close: Optional[Callable] = None,
    ):
        """Initialize the monitor.

        Args:
            tradier_client: TradierClient for order execution
            db_conn: Database connection
            quote_provider: FMP or other quote source (optional, uses Tradier if None)
            config: Monitor configuration
            on_stop_triggered: Callback when stop is triggered
            on_eod_close: Callback for EOD close
        """
        self.tradier_client = tradier_client
        self.db_conn = db_conn
        self.quote_provider = quote_provider
        self.config = config or MonitorConfig()
        self.state = MonitorState()

        self.on_stop_triggered = on_stop_triggered
        self.on_eod_close = on_eod_close

        self._running = False
        self._setup_signal_handlers()

        logger.info(
            f"RealtimeStopLossMonitor initialized "
            f"(interval={self.config.check_interval}s, "
            f"eod={self.config.eod_close_time})"
        )

    def _setup_signal_handlers(self):
        """Set up graceful shutdown handlers."""
        signal.signal(signal.SIGTERM, self._handle_shutdown)
        signal.signal(signal.SIGINT, self._handle_shutdown)

    def _handle_shutdown(self, signum, frame):
        """Handle shutdown signal."""
        logger.info(f"Received signal {signum}, initiating shutdown...")
        self.stop()

    def start(self):
        """Start the monitoring loop."""
        logger.info("Starting RealtimeStopLossMonitor...")

        self._running = True
        self.state.status = MonitorStatus.STARTING
        self.state.start_time = datetime.now()
        self.state.errors_count = 0

        # Wait for market to be open
        self._wait_for_market_open()

        self.state.status = MonitorStatus.RUNNING
        logger.info("Monitor is now running")

        # Main monitoring loop
        while self._running:
            try:
                self._run_check_cycle()
                self.state.errors_count = 0  # Reset on success
            except Exception as e:
                self.state.errors_count += 1
                logger.error(f"Error in check cycle: {e}")

                if self.state.errors_count >= self.config.max_consecutive_errors:
                    logger.critical(
                        f"Too many consecutive errors ({self.state.errors_count}), "
                        "alerting but continuing..."
                    )
                    self._send_error_alert(str(e))

            # Sleep until next check
            if self._running:
                time.sleep(self.config.check_interval)

        self.state.status = MonitorStatus.STOPPED
        logger.info("Monitor stopped")

    def stop(self):
        """Stop the monitoring loop."""
        logger.info("Stopping monitor...")
        self.state.status = MonitorStatus.STOPPING
        self._running = False

    def _wait_for_market_open(self):
        """Wait until market opens."""
        now = datetime.now(ET)
        market_open = self.config.market_open_datetime

        if now < market_open:
            wait_seconds = (market_open - now).total_seconds()
            logger.info(f"Waiting {wait_seconds:.0f}s for market open at {market_open}")

            # Wait in chunks to allow shutdown
            while wait_seconds > 0 and self._running:
                sleep_time = min(60, wait_seconds)
                time.sleep(sleep_time)
                wait_seconds -= sleep_time

    def _run_check_cycle(self):
        """Run one price check cycle."""
        now = datetime.now(ET)

        # Check if past EOD close time
        if now >= self.config.eod_close_datetime:
            logger.info("EOD close time reached, closing all positions")
            self._close_all_positions(CloseReason.END_OF_DAY)
            self._running = False
            return

        # Check if market is closed
        if now >= self.config.market_close_datetime:
            logger.info("Market closed, stopping monitor")
            self._running = False
            return

        # Get open positions
        positions = self._get_open_positions()
        self.state.positions_monitored = len(positions)

        if not positions:
            logger.debug("No open positions to monitor")
            return

        # Get current quotes
        symbols = [p['symbol'] for p in positions]
        quotes = self._get_quotes(symbols)

        # Check each position
        for position in positions:
            symbol = position['symbol']
            quote = quotes.get(symbol)

            if not quote:
                logger.warning(f"No quote for {symbol}")
                continue

            current_price = quote.get('price', 0)
            stop_price = position.get('stop_price')

            if not stop_price:
                logger.warning(f"No stop price for {symbol}")
                continue

            # Check if stop hit
            is_long = position.get('side') == 'LONG'
            is_stop_hit = self._is_stop_hit(current_price, stop_price, is_long)

            # Record price check
            self._record_price_check(
                symbol=symbol,
                current_price=current_price,
                stop_price=stop_price,
                is_stop_hit=is_stop_hit,
                position_id=position.get('id')
            )

            self.state.checks_count += 1
            self.state.last_check_time = datetime.now()

            if is_stop_hit:
                logger.warning(
                    f"STOP HIT for {symbol}: price={current_price}, "
                    f"stop={stop_price}, side={'LONG' if is_long else 'SHORT'}"
                )
                self._trigger_stop_close(position, current_price)
                self.state.stops_triggered += 1

    def _is_stop_hit(self, current_price: float, stop_price: float, is_long: bool) -> bool:
        """Check if stop-loss has been triggered.

        Args:
            current_price: Current market price
            stop_price: Stop-loss price
            is_long: True if long position

        Returns:
            True if stop is hit
        """
        if is_long:
            return current_price <= stop_price
        else:
            return current_price >= stop_price

    def _get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions from database."""
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT position_id, symbol, direction, shares, entry_price, stop_loss,
                       stop_order_id
                FROM gap_trading.positions
                WHERE status = 'OPEN'
                  AND trade_date = CURRENT_DATE
            """)

            columns = ['id', 'symbol', 'side', 'quantity', 'entry_price',
                       'stop_price', 'stop_order_id']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def _get_quotes(self, symbols: List[str]) -> Dict[str, Dict[str, Any]]:
        """Get current quotes for symbols.

        Args:
            symbols: List of ticker symbols

        Returns:
            Dict mapping symbol to quote data
        """
        if not symbols:
            return {}

        try:
            # Use quote provider if available, otherwise Tradier
            if self.quote_provider:
                quotes_list = self.quote_provider.get_batch_quotes(symbols)
                return {
                    q.symbol: {'price': q.price, 'bid': q.bid, 'ask': q.ask}
                    for q in quotes_list
                }
            else:
                quotes_list = self.tradier_client.get_quotes(symbols)
                return {
                    q.symbol: {'price': q.last, 'bid': q.bid, 'ask': q.ask}
                    for q in quotes_list
                }

        except Exception as e:
            logger.error(f"Failed to get quotes: {e}")
            return {}

    def _trigger_stop_close(self, position: Dict[str, Any], current_price: float):
        """Trigger position close due to stop-loss.

        Args:
            position: Position dict
            current_price: Current market price
        """
        # Import OrderManager - handle both package and standalone contexts
        try:
            from .order_manager import OrderManager
        except ImportError:
            import importlib.util
            import os
            module_dir = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location(
                "order_manager",
                os.path.join(module_dir, "order_manager.py")
            )
            order_manager_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(order_manager_module)
            OrderManager = order_manager_module.OrderManager

        symbol = position['symbol']
        quantity = abs(position['quantity'])
        is_long = position['side'] == 'LONG'
        stop_order_id = position.get('stop_order_id')

        logger.info(f"Triggering stop-loss close for {symbol}")

        try:
            order_manager = OrderManager(
                tradier_client=self.tradier_client,
                db_conn=self.db_conn
            )

            # Close position (cancels stop order and places market close)
            result = order_manager.close_position(
                symbol=symbol,
                shares=quantity,
                is_long=is_long,
                stop_order_id=stop_order_id
            )

            if result.success:
                # Update position in database
                self._update_position_closed(
                    position_id=position['id'],
                    exit_price=current_price,
                    exit_reason=CloseReason.STOP_LOSS.value
                )

                logger.info(f"Position {symbol} closed via stop-loss")

                # Callback
                if self.on_stop_triggered:
                    self.on_stop_triggered(position, current_price)

            else:
                logger.error(f"Failed to close {symbol}: {result.message}")

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")

    def _close_all_positions(self, reason: CloseReason):
        """Close all open positions (EOD or manual).

        Args:
            reason: Reason for closing
        """
        # Import OrderManager - handle both package and standalone contexts
        try:
            from .order_manager import OrderManager
        except ImportError:
            # Standalone script mode - use importlib
            import importlib.util
            import os
            module_dir = os.path.dirname(os.path.abspath(__file__))
            spec = importlib.util.spec_from_file_location(
                "order_manager",
                os.path.join(module_dir, "order_manager.py")
            )
            order_manager_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(order_manager_module)
            OrderManager = order_manager_module.OrderManager

        positions = self._get_open_positions()
        logger.info(f"Closing {len(positions)} positions ({reason.value})")

        order_manager = OrderManager(
            tradier_client=self.tradier_client,
            db_conn=self.db_conn
        )

        for position in positions:
            symbol = position['symbol']
            quantity = abs(position['quantity'])
            is_long = position['side'] == 'LONG'
            stop_order_id = position.get('stop_order_id')

            try:
                result = order_manager.close_position(
                    symbol=symbol,
                    shares=quantity,
                    is_long=is_long,
                    stop_order_id=stop_order_id
                )

                if result.success:
                    # Get current price for exit
                    quotes = self._get_quotes([symbol])
                    exit_price = quotes.get(symbol, {}).get('price', 0)

                    self._update_position_closed(
                        position_id=position['id'],
                        exit_price=exit_price,
                        exit_reason=reason.value
                    )
                    logger.info(f"Closed {symbol} ({reason.value})")
                else:
                    logger.error(f"Failed to close {symbol}: {result.message}")

            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")

        # Callback
        if self.on_eod_close and reason == CloseReason.END_OF_DAY:
            self.on_eod_close(positions)

    def _record_price_check(
        self,
        symbol: str,
        current_price: float,
        stop_price: float,
        is_stop_hit: bool,
        position_id: Optional[int] = None
    ):
        """Record price check in database.

        Args:
            symbol: Stock ticker
            current_price: Current price
            stop_price: Stop-loss price
            is_stop_hit: Whether stop was triggered
            position_id: Related position ID
        """
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO gap_trading.price_checks
                (symbol, check_time, current_price, stop_price, is_stop_hit, position_id)
                VALUES (%s, %s, %s, %s, %s, %s)
            """, (
                symbol,
                datetime.now(),
                current_price,
                stop_price,
                is_stop_hit,
                position_id
            ))
            self.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to record price check: {e}")
            if self.db_conn:
                self.db_conn.rollback()

    def _update_position_closed(
        self,
        position_id: int,
        exit_price: float,
        exit_reason: str
    ):
        """Update position as closed in database.

        Args:
            position_id: Position ID
            exit_price: Exit price
            exit_reason: Reason for close
        """
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()

            # Get entry price for P&L calculation
            cursor.execute(
                "SELECT entry_price, shares, direction FROM gap_trading.positions WHERE position_id = %s",
                (position_id,)
            )
            row = cursor.fetchone()
            if not row:
                return

            entry_price, quantity, side = row

            # Calculate P&L
            if side == 'LONG':
                realized_pnl = (exit_price - float(entry_price)) * quantity
            else:
                realized_pnl = (float(entry_price) - exit_price) * abs(quantity)

            # Map exit_reason to valid enum values
            db_exit_reason = {
                'stop_loss': 'STOP_LOSS',
                'end_of_day': 'EOD_CLOSE',
                'manual': 'MANUAL'
            }.get(exit_reason, exit_reason.upper())

            # Determine status - use 'CLOSED' which is valid in the schema
            status = 'CLOSED'

            cursor.execute("""
                UPDATE gap_trading.positions SET
                    status = %s,
                    exit_price = %s,
                    exit_time = %s,
                    exit_reason = %s,
                    pnl = %s
                WHERE position_id = %s
            """, (
                status,
                exit_price,
                datetime.now(),
                db_exit_reason,
                realized_pnl,
                position_id
            ))
            self.db_conn.commit()

        except Exception as e:
            logger.error(f"Failed to update position {position_id}: {e}")
            if self.db_conn:
                self.db_conn.rollback()

    def _send_error_alert(self, error_message: str):
        """Send alert about consecutive errors.

        Args:
            error_message: Error description
        """
        # TODO: Integrate with Telegram notifications
        logger.critical(f"ALERT: Monitor error - {error_message}")

    def get_health(self) -> Dict[str, Any]:
        """Get health check data.

        Returns:
            Dict with health information
        """
        now = datetime.now()
        last_check = self.state.last_check_time

        # Check if stale
        is_stale = False
        stale_seconds = 0
        if last_check:
            stale_seconds = (now - last_check).total_seconds()
            is_stale = stale_seconds > (self.config.check_interval * 2)

        return {
            **self.state.to_dict(),
            'is_healthy': self.state.status == MonitorStatus.RUNNING and not is_stale,
            'is_stale': is_stale,
            'stale_seconds': stale_seconds,
            'config': {
                'check_interval': self.config.check_interval,
                'eod_close_time': self.config.eod_close_time,
            }
        }


def run_monitor(
    tradier_api_key: Optional[str] = None,
    tradier_account_id: Optional[str] = None,
    db_connection_string: Optional[str] = None,
    check_interval: int = 60,
    sandbox: bool = True
):
    """Run the realtime monitor as a standalone process.

    Args:
        tradier_api_key: Tradier API key (or from env)
        tradier_account_id: Tradier account ID (or from env)
        db_connection_string: Database connection string
        check_interval: Seconds between checks
        sandbox: Use Tradier sandbox
    """
    import os
    import psycopg2

    # Initialize Tradier client
    from stock_data_web.tradier import TradierClient

    tradier = TradierClient(
        api_key=tradier_api_key,
        account_id=tradier_account_id,
        sandbox=sandbox
    )

    # Initialize database connection
    conn_string = db_connection_string or os.environ.get(
        'DATABASE_URL',
        'postgresql://postgres:password@localhost:5432/timescaledb'
    )
    db_conn = psycopg2.connect(conn_string)

    # Create and run monitor
    config = MonitorConfig(check_interval=check_interval)
    monitor = RealtimeStopLossMonitor(
        tradier_client=tradier,
        db_conn=db_conn,
        config=config
    )

    try:
        monitor.start()
    finally:
        tradier.close()
        db_conn.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Gap Trading Stop-Loss Monitor')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--sandbox', action='store_true', default=True, help='Use Tradier sandbox')
    parser.add_argument('--live', action='store_true', help='Use Tradier live (production)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_monitor(
        check_interval=args.interval,
        sandbox=not args.live
    )
