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
            logger.info("EOD close time reached, initiating EOD close sequence")
            self._execute_eod_close_sequence()
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
        """Get all open gap trading positions from database.

        Only returns positions where strategy = 'gap_trading' to avoid
        interfering with other trading strategies.
        """
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT position_id, symbol, direction, shares, entry_price, stop_loss,
                       stop_order_id, strategy
                FROM gap_trading.positions
                WHERE status = 'OPEN'
                  AND trade_date = CURRENT_DATE
                  AND strategy = 'gap_trading'
            """)

            columns = ['id', 'symbol', 'side', 'quantity', 'entry_price',
                       'stop_price', 'stop_order_id', 'strategy']
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

        IMPORTANT: Only updates DB after verifying broker execution.
        This ensures database and broker are always in sync.

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
        position_id = position['id']

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
                # VERIFY order execution before updating DB
                order_tag = getattr(result, 'tag', None)
                verified, fill_price, error = self._verify_order_execution(
                    order_id=result.order_id,
                    expected_tag=order_tag,
                    timeout=60
                )

                if verified:
                    # Use actual fill price from broker
                    actual_exit_price = fill_price if fill_price > 0 else current_price

                    # Now safe to update database
                    self._update_position_closed(
                        position_id=position_id,
                        exit_price=actual_exit_price,
                        exit_reason=CloseReason.STOP_LOSS.value
                    )

                    logger.info(
                        f"Position {symbol} closed via stop-loss @ ${actual_exit_price:.2f} (verified)"
                    )

                    # Callback
                    if self.on_stop_triggered:
                        self.on_stop_triggered(position, actual_exit_price)
                else:
                    # Order placed but not verified - DO NOT update DB
                    logger.error(
                        f"CRITICAL: Close order placed but NOT verified for {symbol}! "
                        f"Order {result.order_id} - {error}. "
                        f"Position remains OPEN in DB, check broker manually."
                    )
                    # Record the failure for tracking
                    self._record_close_failure(
                        position_id=position_id,
                        order_id=result.order_id,
                        error=error
                    )
            else:
                logger.error(f"Failed to close {symbol}: {result.message}")
                self._record_close_failure(
                    position_id=position_id,
                    order_id=None,
                    error=result.message
                )

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")
            self._record_close_failure(
                position_id=position_id,
                order_id=None,
                error=str(e)
            )

    def _execute_eod_close_sequence(self):
        """Execute EOD close with retry logic.

        Improved EOD close flow:
        1. 3:55 PM: Initial close attempt for all positions
        2. Wait up to 60s for verification
        3. Retry failed positions up to 2 more times
        4. Final alert for any remaining open positions
        5. Stop monitor after all attempts

        This ensures we don't exit immediately after placing orders,
        giving time for verification and retry.
        """
        max_retries = 3
        retry_delay = 30  # seconds between retries

        logger.info("=" * 50)
        logger.info("STARTING EOD CLOSE SEQUENCE")
        logger.info("=" * 50)

        for attempt in range(1, max_retries + 1):
            # Check how much time we have left
            now = datetime.now(ET)
            market_close = self.config.market_close_datetime
            time_until_close = (market_close - now).total_seconds()

            if time_until_close < 30:
                logger.warning(
                    f"Only {time_until_close:.0f}s until market close, "
                    "stopping retries"
                )
                break

            logger.info(f"EOD Close Attempt {attempt}/{max_retries}")

            # Attempt to close all positions
            result = self._close_all_positions(CloseReason.END_OF_DAY)

            if result['failed'] == 0:
                logger.info(
                    f"✅ All {result['closed']} positions closed successfully!"
                )
                break

            logger.warning(
                f"Attempt {attempt}: {result['closed']} closed, "
                f"{result['failed']} failed"
            )

            # If this wasn't the last attempt, wait and retry
            if attempt < max_retries:
                # Check for remaining open positions
                remaining = self._get_open_positions()
                if not remaining:
                    logger.info("No remaining open positions, closing sequence complete")
                    break

                logger.info(
                    f"Waiting {retry_delay}s before retry... "
                    f"({len(remaining)} positions remaining)"
                )
                time.sleep(retry_delay)

        # Final check for any remaining positions
        final_open = self._get_open_positions()
        if final_open:
            logger.critical(
                f"⚠️ EOD CLOSE INCOMPLETE: {len(final_open)} positions still open!"
            )
            for pos in final_open:
                logger.critical(
                    f"  - {pos['symbol']}: {pos['quantity']} shares ({pos['side']})"
                )
            # Final alert was already sent by _close_all_positions
        else:
            logger.info("✅ EOD close sequence complete - all positions closed")

        logger.info("=" * 50)
        logger.info("EOD CLOSE SEQUENCE FINISHED")
        logger.info("=" * 50)

        # Now stop the monitor
        self._running = False

    def _close_all_positions(self, reason: CloseReason):
        """Close all open positions (EOD or manual).

        IMPORTANT: Only updates DB after verifying broker execution.
        Tracks failed closes and sends alerts for unverified positions.

        Args:
            reason: Reason for closing

        Returns:
            Dict with success/failure counts
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

        if not positions:
            return {'closed': 0, 'failed': 0, 'positions': []}

        order_manager = OrderManager(
            tradier_client=self.tradier_client,
            db_conn=self.db_conn
        )

        # Track results
        closed_count = 0
        failed_positions = []

        for position in positions:
            symbol = position['symbol']
            quantity = abs(position['quantity'])
            is_long = position['side'] == 'LONG'
            stop_order_id = position.get('stop_order_id')
            position_id = position['id']

            try:
                result = order_manager.close_position(
                    symbol=symbol,
                    shares=quantity,
                    is_long=is_long,
                    stop_order_id=stop_order_id
                )

                if result.success:
                    # VERIFY order execution before updating DB
                    order_tag = getattr(result, 'tag', None)
                    verified, fill_price, error = self._verify_order_execution(
                        order_id=result.order_id,
                        expected_tag=order_tag,
                        timeout=60
                    )

                    if verified:
                        # Use actual fill price from broker
                        exit_price = fill_price if fill_price > 0 else 0
                        if exit_price == 0:
                            quotes = self._get_quotes([symbol])
                            exit_price = quotes.get(symbol, {}).get('price', 0)

                        self._update_position_closed(
                            position_id=position_id,
                            exit_price=exit_price,
                            exit_reason=reason.value
                        )
                        closed_count += 1
                        logger.info(f"Closed {symbol} @ ${exit_price:.2f} ({reason.value}) - VERIFIED")
                    else:
                        # Order placed but not verified - DO NOT update DB
                        logger.error(
                            f"CRITICAL: EOD close order NOT verified for {symbol}! "
                            f"Order {result.order_id} - {error}. "
                            f"Position remains OPEN in DB."
                        )
                        self._record_close_failure(
                            position_id=position_id,
                            order_id=result.order_id,
                            error=error
                        )
                        failed_positions.append({
                            'symbol': symbol,
                            'position_id': position_id,
                            'order_id': result.order_id,
                            'error': error,
                            'shares': quantity,
                            'direction': 'LONG' if is_long else 'SHORT'
                        })
                else:
                    logger.error(f"Failed to place close order for {symbol}: {result.message}")
                    self._record_close_failure(
                        position_id=position_id,
                        order_id=None,
                        error=result.message
                    )
                    failed_positions.append({
                        'symbol': symbol,
                        'position_id': position_id,
                        'order_id': None,
                        'error': result.message,
                        'shares': quantity,
                        'direction': 'LONG' if is_long else 'SHORT'
                    })

            except Exception as e:
                logger.error(f"Error closing {symbol}: {e}")
                self._record_close_failure(
                    position_id=position_id,
                    order_id=None,
                    error=str(e)
                )
                failed_positions.append({
                    'symbol': symbol,
                    'position_id': position_id,
                    'order_id': None,
                    'error': str(e),
                    'shares': quantity,
                    'direction': 'LONG' if is_long else 'SHORT'
                })

        # Log summary
        logger.info(
            f"EOD Close Summary: {closed_count}/{len(positions)} closed, "
            f"{len(failed_positions)} failed"
        )

        # Send alert for failed positions
        if failed_positions:
            self._send_eod_failure_alert(failed_positions, reason)

        # Callback
        if self.on_eod_close and reason == CloseReason.END_OF_DAY:
            self.on_eod_close(positions)

        return {
            'closed': closed_count,
            'failed': len(failed_positions),
            'failed_positions': failed_positions
        }

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

    def _record_close_failure(
        self,
        position_id: int,
        order_id: Optional[int],
        error: str
    ):
        """Record a failed close attempt in the database.

        Updates close_attempts and last_close_error columns.
        Does NOT change position status - position remains OPEN.

        Args:
            position_id: Position ID
            order_id: Order ID if order was placed
            error: Error message
        """
        if not self.db_conn:
            return

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE gap_trading.positions SET
                    close_attempts = close_attempts + 1,
                    last_close_error = %s
                WHERE position_id = %s
            """, (
                f"Order {order_id}: {error}" if order_id else error,
                position_id
            ))
            self.db_conn.commit()
            logger.info(f"Recorded close failure for position {position_id}")

        except Exception as e:
            logger.error(f"Failed to record close failure: {e}")
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

    def _send_eod_failure_alert(self, failed_positions: List[Dict], reason: CloseReason):
        """Send Telegram alert for failed EOD close positions.

        Args:
            failed_positions: List of positions that failed to close
            reason: Close reason (EOD, etc.)
        """
        if not failed_positions:
            return

        # Build alert message
        now = datetime.now(ET)
        message = f"""🚨 *Gap Trading EOD Close Failed*
Date: {now.strftime('%Y-%m-%d')}
Time: {now.strftime('%H:%M')} ET
Reason: {reason.value}

*{len(failed_positions)} Position(s) Failed to Close:*
"""
        for pos in failed_positions:
            message += f"""
• *{pos['symbol']}* ({pos['direction']})
  Shares: {pos['shares']}
  Order ID: {pos.get('order_id', 'N/A')}
  Error: {pos['error']}
"""

        message += """
⚠️ *Action Required*: Manual close may be needed.
Check Tradier positions and verify DB state.
"""

        logger.critical(f"EOD FAILURE ALERT:\n{message}")

        # Try to send via Telegram
        try:
            # Try importing from airflow_common if available
            try:
                from airflow_common.notifications.telegram import TelegramNotifier
                notifier = TelegramNotifier()
                notifier.send_message(message, parse_mode="Markdown")
                logger.info("EOD failure alert sent via Telegram")
            except ImportError:
                # Fallback: try direct telegram via environment
                import os
                import requests

                bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
                chat_id = os.environ.get('TELEGRAM_CHAT_ID')

                if bot_token and chat_id:
                    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
                    response = requests.post(url, json={
                        'chat_id': chat_id,
                        'text': message,
                        'parse_mode': 'Markdown'
                    }, timeout=10)
                    if response.ok:
                        logger.info("EOD failure alert sent via Telegram (direct)")
                    else:
                        logger.warning(f"Telegram API error: {response.text}")
                else:
                    logger.warning(
                        "Cannot send Telegram alert - no credentials. "
                        "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID environment variables."
                    )
        except Exception as e:
            logger.error(f"Failed to send Telegram alert: {e}")

    def _verify_order_execution(
        self,
        order_id: int,
        expected_tag: str,
        timeout: int = 60,
        poll_interval: int = 2
    ) -> tuple:
        """Verify order executed at broker before updating DB.

        Uses both order_id lookup AND tag verification for double-check.
        This ensures DB is only updated after broker confirms execution.

        Args:
            order_id: Tradier order ID
            expected_tag: Expected order tag for verification
            timeout: Max seconds to wait for fill
            poll_interval: Seconds between status checks

        Returns:
            Tuple of (success: bool, fill_price: float, error: str)
        """
        start_time = time.time()
        last_status = None

        while time.time() - start_time < timeout:
            try:
                order = self.tradier_client.get_order(order_id)

                if order is None:
                    return (False, 0.0, f"Order {order_id} not found at broker")

                # Verify tag matches (ensures we're checking the right order)
                order_tag = getattr(order, 'tag', None)
                if order_tag and expected_tag and order_tag != expected_tag:
                    logger.error(
                        f"Tag mismatch: expected '{expected_tag}', got '{order_tag}'"
                    )
                    return (False, 0.0, f"Tag mismatch - order may be incorrect")

                last_status = order.status.value if hasattr(order.status, 'value') else str(order.status)

                # Check if filled
                if order.is_filled:
                    fill_price = getattr(order, 'avg_fill_price', 0) or 0
                    logger.info(
                        f"Order {order_id} verified: FILLED @ ${fill_price:.2f}"
                    )
                    return (True, float(fill_price), None)

                # Check if rejected/cancelled
                if order.is_terminal and not order.is_filled:
                    logger.warning(f"Order {order_id} terminal but not filled: {last_status}")
                    return (False, 0.0, f"Order {last_status}")

                time.sleep(poll_interval)

            except Exception as e:
                logger.error(f"Error verifying order {order_id}: {e}")
                time.sleep(poll_interval)

        # Timeout reached
        logger.warning(f"Order {order_id} verification timeout (last status: {last_status})")
        return (False, 0.0, f"Timeout - last status: {last_status}")

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
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    db_connection_string: Optional[str] = None,
    check_interval: int = 60,
    paper_trading: bool = True
):
    """Run the realtime monitor as a standalone process.

    Args:
        api_key: Broker API key (or from env)
        api_secret: Broker API secret (or from env)
        db_connection_string: Database connection string
        check_interval: Seconds between checks
        paper_trading: Use paper trading mode
    """
    import os
    import psycopg2

    # Initialize broker client (Alpaca)
    from stock_data_web.alpaca import AlpacaClient

    broker_client = AlpacaClient(
        api_key=api_key,
        api_secret=api_secret,
        paper_trading=paper_trading
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
        tradier_client=broker_client,
        db_conn=db_conn,
        config=config
    )

    try:
        monitor.start()
    finally:
        broker_client.close()
        db_conn.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Gap Trading Stop-Loss Monitor')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    parser.add_argument('--paper', action='store_true', default=True, help='Use paper trading')
    parser.add_argument('--live', action='store_true', help='Use live trading (production)')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    run_monitor(
        check_interval=args.interval,
        paper_trading=not args.live
    )
