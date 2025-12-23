"""Order Manager for Gap Trading Strategy.

Handles order execution via broker API (Alpaca) including entry orders,
stop-loss orders, and order lifecycle management.

Features:
- Retry with exponential backoff for transient failures
- Circuit breaker pattern to prevent cascading failures
- Comprehensive error categorization and handling
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, List, Dict, Any, Tuple, TYPE_CHECKING

from .error_handling import (
    ErrorCategory,
    GapTradingError,
    NetworkError,
    AuthenticationError,
    BusinessError,
    InsufficientFundsError,
    RateLimitError,
    CircuitOpenError,
    retry_with_backoff,
    with_error_handling,
    get_broker_circuit_breaker,
    wrap_exception,
    CircuitBreaker
)

if TYPE_CHECKING:
    from stock_data_web.alpaca import AlpacaClient

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    """Order execution result status."""
    SUCCESS = 'success'
    PARTIAL = 'partial'
    FAILED = 'failed'
    REJECTED = 'rejected'
    TIMEOUT = 'timeout'
    INSUFFICIENT_FUNDS = 'insufficient_funds'


class OrderPurpose(str, Enum):
    """Purpose of the order."""
    ENTRY = 'entry'
    STOP_LOSS = 'stop_loss'
    TAKE_PROFIT = 'take_profit'
    CLOSE = 'close'


@dataclass
class OrderResponse:
    """Response from order placement.

    Attributes:
        success: Whether order was placed successfully
        order_id: Broker order ID (if successful)
        symbol: Stock symbol
        side: Order side (buy, sell, etc.)
        quantity: Number of shares (supports fractional)
        order_type: Order type (market, stop, etc.)
        price: Limit/stop price (if applicable)
        status: Order status
        message: Status message or error description
        fill_price: Actual fill price (if filled)
        fill_quantity: Number of shares filled (supports fractional)
        timestamp: Order timestamp
        tag: Order tag for tracking (format: gaptrading-{type}-{symbol}-{timestamp})
    """
    success: bool
    order_id: Optional[int] = None
    symbol: str = ''
    side: str = ''
    quantity: float = 0  # Supports fractional shares
    order_type: str = 'market'
    price: Optional[float] = None
    status: str = ''
    message: str = ''
    fill_price: Optional[float] = None
    fill_quantity: float = 0  # Supports fractional shares
    timestamp: datetime = field(default_factory=datetime.now)
    tag: Optional[str] = None


@dataclass
class ExecutionResult:
    """Result of signal execution.

    Attributes:
        status: Overall execution status
        entry_order: Entry order response
        stop_order: Stop-loss order response (if placed)
        position_id: Local position ID (if created)
        message: Status message
        error: Error details (if failed)
    """
    status: ExecutionStatus
    entry_order: Optional[OrderResponse] = None
    stop_order: Optional[OrderResponse] = None
    position_id: Optional[int] = None
    message: str = ''
    error: Optional[str] = None


@dataclass
class SyncResult:
    """Result of broker sync operation.

    Attributes:
        success: Whether sync completed successfully
        positions_synced: Number of positions synced
        orders_synced: Number of orders synced
        discrepancies: List of detected discrepancies
        timestamp: Sync timestamp
    """
    success: bool
    positions_synced: int = 0
    orders_synced: int = 0
    discrepancies: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class OrderManager:
    """Manages order execution for gap trading strategy.

    Integrates with broker API (Alpaca) to place entry orders, stop-loss orders,
    and manage order lifecycle.

    Features:
    - Retry with exponential backoff for transient failures
    - Circuit breaker to prevent cascading failures after repeated errors
    - Comprehensive error categorization (auth, network, business, etc.)

    Attributes:
        broker_client: Broker client instance for operations (AlpacaClient)
        db_conn: Database connection for order tracking
        max_fill_wait: Maximum seconds to wait for order fill
        poll_interval: Seconds between fill status checks
        tag_prefix: Prefix for order tags
        circuit_breaker: Circuit breaker for broker API protection
    """

    def __init__(
        self,
        broker_client: 'AlpacaClient',
        db_conn: Any = None,
        max_fill_wait: int = 60,
        poll_interval: int = 2,
        tag_prefix: str = 'gaptrading',
        circuit_breaker: Optional[CircuitBreaker] = None
    ):
        """Initialize OrderManager.

        Args:
            broker_client: Broker client instance (AlpacaClient)
            db_conn: Database connection (psycopg2 or SQLAlchemy)
            max_fill_wait: Max seconds to wait for order fill
            poll_interval: Seconds between fill status checks
            tag_prefix: Prefix for order tags
            circuit_breaker: Optional circuit breaker (uses global if None)
        """
        self.broker_client = broker_client
        self.db_conn = db_conn
        self.max_fill_wait = max_fill_wait
        self.poll_interval = poll_interval
        self.tag_prefix = tag_prefix
        self.circuit_breaker = circuit_breaker or get_broker_circuit_breaker()

        logger.info(
            f"OrderManager initialized (max_wait={max_fill_wait}s, "
            f"poll={poll_interval}s, circuit_breaker={self.circuit_breaker.name})"
        )

    def check_buying_power(self, required_amount: float) -> Tuple[bool, float]:
        """Check if sufficient buying power is available.

        Uses circuit breaker to prevent cascading failures if broker API
        is experiencing issues.

        Args:
            required_amount: Amount needed for trade

        Returns:
            Tuple of (has_sufficient, available_amount)

        Raises:
            CircuitOpenError: If circuit breaker is open
            GapTradingError: If broker API call fails
        """
        try:
            # Check circuit breaker first
            if self.circuit_breaker.is_open:
                raise CircuitOpenError(
                    f"Circuit breaker '{self.circuit_breaker.name}' is open - "
                    "broker API calls are blocked"
                )

            balance = self.broker_client.get_balance()
            available = balance.available_for_trading

            # Record success
            self.circuit_breaker.record_success()

            has_sufficient = available >= required_amount

            logger.info(
                f"Buying power check: required=${required_amount:.2f}, "
                f"available=${available:.2f}, sufficient={has_sufficient}"
            )

            return has_sufficient, available

        except CircuitOpenError:
            raise  # Re-raise circuit open errors
        except Exception as e:
            # Record failure and wrap the exception
            self.circuit_breaker.record_failure(e)
            wrapped = wrap_exception(e)
            logger.error(f"Failed to check buying power: {wrapped}")
            raise wrapped

    def execute_signal(
        self,
        signal: Any,
        position_size: Any,
        use_oto: bool = False,
        use_stop_orders: bool = True
    ) -> ExecutionResult:
        """Execute a trade signal with full lifecycle.

        Flow (Separate orders mode - default):
        1. Check buying power
        2. Place entry order (market)
        3. Wait for fill
        4. Place stop-loss order separately (if use_stop_orders=True)
        5. Record in database

        Flow (OTO mode - use_oto=True):
        1. Check buying power
        2. Place OTO order (entry + stop in single API call)
        3. Wait for entry fill
        4. Record in database

        Flow (Entry-only mode - use_stop_orders=False):
        1. Check buying power
        2. Place entry order (market)
        3. Wait for fill
        4. Record in database (stop_loss stored but no broker order)
        5. Software-based stop monitoring handles stop triggers

        Note: use_stop_orders=False is recommended for intraday strategies
        where all positions close at EOD. This avoids stop orders "holding"
        shares and blocking EOD close.

        Args:
            signal: TradeSignal from signal generator
            position_size: PositionSize from position sizer
            use_oto: Use OTO (One-Triggers-Other) orders (default False)
            use_stop_orders: Place broker stop orders (default True).
                            Set to False for software-based stop monitoring.

        Returns:
            ExecutionResult with order details
        """
        symbol = signal.symbol
        shares = position_size.shares
        signal_type = signal.signal_type.value  # BUY or SELL_SHORT
        stop_price = position_size.stop_loss

        logger.info(f"Executing signal: {signal_type} {shares} {symbol} (OTO={use_oto})")

        # 1. Check buying power
        required = shares * signal.entry_price
        has_funds, available = self.check_buying_power(required)

        if not has_funds:
            logger.warning(
                f"Insufficient buying power: need ${required:.2f}, "
                f"have ${available:.2f}"
            )
            return ExecutionResult(
                status=ExecutionStatus.INSUFFICIENT_FUNDS,
                message=f"Need ${required:.2f}, have ${available:.2f}",
                error="Insufficient buying power"
            )

        # Use OTO order (entry + stop in single API call)
        if use_oto:
            return self._execute_with_oto(signal, signal_type, symbol, shares, stop_price)

        # Entry with optional stop order (use_stop_orders=False for software monitoring)
        return self._execute_with_separate_orders(
            signal, signal_type, symbol, shares, stop_price,
            use_stop_orders=use_stop_orders
        )

    def _execute_with_oto(
        self,
        signal: Any,
        signal_type: str,
        symbol: str,
        shares: float,
        stop_price: float
    ) -> ExecutionResult:
        """Execute signal using OTO (One-Triggers-Other) order.

        Places entry and stop-loss in a single API call.
        """
        # Determine entry side
        if signal_type.upper() in ('BUY', 'LONG'):
            entry_side = 'buy'
        elif signal_type.upper() in ('SELL_SHORT', 'SHORT'):
            entry_side = 'sell_short'
        else:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"Invalid signal type: {signal_type}",
                error=f"Invalid signal type: {signal_type}"
            )

        # Use hyphens in tag for compatibility with broker APIs
        tag = f"{self.tag_prefix}-oto-{symbol}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        try:
            # Place OTO order (entry + stop in single call)
            oto_result = self.broker_client.place_oto_order(
                symbol=symbol,
                entry_side=entry_side,
                quantity=shares,
                stop_price=stop_price,
                entry_type='market',
                tag=tag
            )

            logger.info(f"OTO order placed for {symbol}: {oto_result}")

            # Wait for entry to fill
            entry_order_id = oto_result.get('entry_order_id') or oto_result.get('parent_order_id')
            filled_order = self._wait_for_fill(entry_order_id)

            if not filled_order or filled_order.status != 'filled':
                # Cancel unfilled OTO order
                parent_id = oto_result.get('parent_order_id')
                if parent_id:
                    self.cancel_order(parent_id)

                return ExecutionResult(
                    status=ExecutionStatus.TIMEOUT,
                    message="OTO entry order not filled within timeout",
                    error="Fill timeout"
                )

            # Create entry order response for DB recording
            entry_response = OrderResponse(
                success=True,
                order_id=entry_order_id,
                symbol=symbol,
                side=entry_side,
                quantity=shares,
                order_type='market',
                status='filled',
                fill_price=filled_order.avg_fill_price,
                fill_quantity=filled_order.filled_quantity,
                message="OTO entry filled"
            )

            # Create stop order response (stop is automatically placed by broker)
            stop_order_id = oto_result.get('stop_order_id')
            stop_response = OrderResponse(
                success=True,
                order_id=stop_order_id,
                symbol=symbol,
                side='sell' if entry_side == 'buy' else 'buy_to_cover',
                quantity=filled_order.filled_quantity,
                order_type='stop',
                price=stop_price,
                status='open',  # Stop is now active
                message="OTO stop order active"
            )

            # Record in database
            position_id = None
            if self.db_conn:
                position_id = self._record_orders(
                    signal=signal,
                    entry_order=entry_response,
                    stop_order=stop_response
                )

            return ExecutionResult(
                status=ExecutionStatus.SUCCESS,
                entry_order=entry_response,
                stop_order=stop_response,
                position_id=position_id,
                message=f"OTO executed {signal_type} {filled_order.filled_quantity} "
                        f"{symbol} @ ${filled_order.avg_fill_price:.2f}"
            )

        except Exception as e:
            logger.error(f"OTO order failed for {symbol}: {e}")
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                message=f"OTO order failed: {str(e)}",
                error=str(e)
            )

    def _execute_with_separate_orders(
        self,
        signal: Any,
        signal_type: str,
        symbol: str,
        shares: float,
        stop_price: float,
        use_stop_orders: bool = True
    ) -> ExecutionResult:
        """Execute signal with entry order and optional stop order.

        Args:
            signal: Trade signal
            signal_type: BUY or SELL_SHORT
            symbol: Stock ticker
            shares: Number of shares
            stop_price: Stop loss price (stored in DB even if no broker order)
            use_stop_orders: If True, place broker stop order. If False, skip
                            stop order and rely on software monitoring.
        """
        # 2. Place entry order
        entry_response = self.place_entry_order(
            symbol=symbol,
            signal_type=signal_type,
            shares=shares
        )

        if not entry_response.success:
            return ExecutionResult(
                status=ExecutionStatus.FAILED,
                entry_order=entry_response,
                message=f"Entry order failed: {entry_response.message}",
                error=entry_response.message
            )

        # 3. Wait for fill
        filled_order = self._wait_for_fill(entry_response.order_id)

        if not filled_order or filled_order.status != 'filled':
            # Cancel unfilled order
            if entry_response.order_id:
                self.cancel_order(entry_response.order_id)

            return ExecutionResult(
                status=ExecutionStatus.TIMEOUT,
                entry_order=entry_response,
                message="Entry order not filled within timeout",
                error="Fill timeout"
            )

        # Update entry response with fill info
        entry_response.fill_price = filled_order.avg_fill_price
        entry_response.fill_quantity = filled_order.filled_quantity
        entry_response.status = 'filled'

        # 4. Place stop-loss order (only if use_stop_orders=True)
        stop_response = None
        if use_stop_orders:
            stop_response = self.place_stop_order(
                symbol=symbol,
                signal_type=signal_type,
                shares=filled_order.filled_quantity,
                stop_price=stop_price
            )
        else:
            logger.info(
                f"Skipping broker stop order for {symbol} (software monitoring mode). "
                f"Stop price ${stop_price:.2f} recorded for monitoring."
            )
            # Create a mock response to record the stop price in DB
            stop_response = OrderResponse(
                success=True,
                symbol=symbol,
                side='sell' if signal_type.upper() in ('BUY', 'LONG') else 'buy_to_cover',
                quantity=filled_order.filled_quantity,
                price=stop_price,
                message="Software monitoring - no broker order"
            )
            # Mark it as software-managed (no broker order ID)
            stop_response.order_id = None

        # 5. Record in database - ALWAYS record if entry was filled
        # This is critical because the entry order executed and we need to track the position
        # Note: stop_price is recorded even if no broker stop order (for software monitoring)
        position_id = None
        if self.db_conn:
            position_id = self._record_orders(
                signal=signal,
                entry_order=entry_response,
                stop_order=stop_response if (stop_response and stop_response.success) else None,
                stop_price=stop_price  # Always pass stop_price for DB recording
            )

        # If broker stop order was placed but failed
        if use_stop_orders and stop_response and not stop_response.success:
            logger.error(
                f"Failed to place stop order for {symbol}: "
                f"{stop_response.message}"
            )
            # Entry filled but stop failed - position is still recorded
            return ExecutionResult(
                status=ExecutionStatus.PARTIAL,
                entry_order=entry_response,
                stop_order=stop_response,
                position_id=position_id,
                message="Entry filled but stop order failed - position recorded without stop",
                error=stop_response.message
            )

        # Success message varies based on stop order mode
        if use_stop_orders:
            msg = (f"Executed {signal_type} {filled_order.filled_quantity} "
                   f"{symbol} @ ${filled_order.avg_fill_price:.2f} "
                   f"with stop @ ${stop_price:.2f}")
        else:
            msg = (f"Executed {signal_type} {filled_order.filled_quantity} "
                   f"{symbol} @ ${filled_order.avg_fill_price:.2f} "
                   f"(software stop @ ${stop_price:.2f})")

        return ExecutionResult(
            status=ExecutionStatus.SUCCESS,
            entry_order=entry_response,
            stop_order=stop_response if use_stop_orders else None,
            position_id=position_id,
            message=msg
        )

    def place_entry_order(
        self,
        symbol: str,
        signal_type: str,
        shares: float,
        max_retries: int = 2
    ) -> OrderResponse:
        """Place entry market order with retry and circuit breaker protection.

        Args:
            symbol: Stock ticker
            signal_type: 'BUY' or 'SELL_SHORT'
            shares: Number of shares
            max_retries: Maximum retry attempts for transient failures

        Returns:
            OrderResponse with order details
        """
        # Determine order side
        if signal_type.upper() in ('BUY', 'LONG'):
            side = 'buy'
        elif signal_type.upper() in ('SELL_SHORT', 'SHORT'):
            side = 'sell_short'
        else:
            return OrderResponse(
                success=False,
                symbol=symbol,
                message=f"Invalid signal type: {signal_type}"
            )

        # Use hyphens in tag for compatibility with broker APIs
        tag = f"{self.tag_prefix}-entry-{symbol}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Check circuit breaker
        if self.circuit_breaker.is_open:
            logger.warning(f"Circuit breaker open - entry order blocked for {symbol}")
            return OrderResponse(
                success=False,
                symbol=symbol,
                side=side,
                quantity=shares,
                message=f"Circuit breaker open - broker API calls blocked",
                tag=tag
            )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                order = self.broker_client.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=shares,
                    tag=tag
                )

                # Record success
                self.circuit_breaker.record_success()

                logger.info(
                    f"Entry order placed: {side} {shares} {symbol}, "
                    f"order_id={order.id}, tag={tag}"
                )

                return OrderResponse(
                    success=True,
                    order_id=order.id,
                    symbol=symbol,
                    side=side,
                    quantity=shares,
                    order_type='market',
                    status=order.status.value,
                    message="Entry order placed successfully",
                    tag=tag
                )

            except Exception as e:
                last_error = e
                wrapped = wrap_exception(e)
                self.circuit_breaker.record_failure(e)

                # Check if retryable
                if not wrapped.retryable or attempt >= max_retries:
                    logger.error(f"Entry order failed for {symbol}: {wrapped}")
                    break

                # Calculate backoff delay
                delay = min(2 ** attempt, 30)
                logger.warning(
                    f"Entry order attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

        return OrderResponse(
            success=False,
            symbol=symbol,
            side=side,
            quantity=shares,
            message=str(last_error) if last_error else "Unknown error",
            tag=tag
        )

    def place_stop_order(
        self,
        symbol: str,
        signal_type: str,
        shares: float,
        stop_price: float,
        max_retries: int = 2
    ) -> OrderResponse:
        """Place stop-loss order with retry and circuit breaker protection.

        Args:
            symbol: Stock ticker
            signal_type: Original signal type (determines exit side)
            shares: Number of shares
            stop_price: Stop trigger price
            max_retries: Maximum retry attempts for transient failures

        Returns:
            OrderResponse with order details
        """
        # Determine exit side (opposite of entry)
        if signal_type.upper() in ('BUY', 'LONG'):
            side = 'sell'  # Exit long with sell
        elif signal_type.upper() in ('SELL_SHORT', 'SHORT'):
            side = 'buy_to_cover'  # Exit short with buy_to_cover
        else:
            return OrderResponse(
                success=False,
                symbol=symbol,
                message=f"Invalid signal type: {signal_type}"
            )

        # Use hyphens in tag for compatibility with broker APIs
        tag = f"{self.tag_prefix}-stop-{symbol}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Check circuit breaker
        if self.circuit_breaker.is_open:
            logger.warning(f"Circuit breaker open - stop order blocked for {symbol}")
            return OrderResponse(
                success=False,
                symbol=symbol,
                side=side,
                quantity=shares,
                price=stop_price,
                message="Circuit breaker open - broker API calls blocked",
                tag=tag
            )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                order = self.broker_client.place_stop_order(
                    symbol=symbol,
                    side=side,
                    quantity=shares,
                    stop_price=stop_price,
                    tag=tag
                )

                # Record success
                self.circuit_breaker.record_success()

                logger.info(
                    f"Stop order placed: {side} {shares} {symbol} @ ${stop_price:.2f}, "
                    f"order_id={order.id}, tag={tag}"
                )

                return OrderResponse(
                    success=True,
                    order_id=order.id,
                    symbol=symbol,
                    side=side,
                    quantity=shares,
                    order_type='stop',
                    price=stop_price,
                    status=order.status.value,
                    message="Stop order placed successfully",
                    tag=tag
                )

            except Exception as e:
                last_error = e
                wrapped = wrap_exception(e)
                self.circuit_breaker.record_failure(e)

                # Check if retryable
                if not wrapped.retryable or attempt >= max_retries:
                    logger.error(f"Stop order failed for {symbol}: {wrapped}")
                    break

                # Calculate backoff delay
                delay = min(2 ** attempt, 30)
                logger.warning(
                    f"Stop order attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

        return OrderResponse(
            success=False,
            symbol=symbol,
            side=side,
            quantity=shares,
            price=stop_price,
            message=str(last_error) if last_error else "Unknown error",
            tag=tag
        )

    def close_position(
        self,
        symbol: str,
        shares: float,
        is_long: bool,
        stop_order_id: Optional[int] = None,
        max_retries: int = 3
    ) -> OrderResponse:
        """Close an open position with retry and circuit breaker protection.

        Cancels existing stop order and places market order to close.
        Uses more retries than entry/stop orders because closing is critical.

        Args:
            symbol: Stock ticker
            shares: Number of shares to close
            is_long: True if long position, False if short
            stop_order_id: Stop order to cancel (if any)
            max_retries: Maximum retry attempts (default 3 for critical closes)

        Returns:
            OrderResponse with close order details
        """
        # 1. Cancel existing stop order if provided
        if stop_order_id:
            try:
                self.cancel_order(stop_order_id)
            except Exception as e:
                logger.warning(f"Failed to cancel stop order {stop_order_id}: {e}")

        # 2. Determine close side
        if is_long:
            side = 'sell'
        else:
            side = 'buy_to_cover'

        # Use hyphens in tag for compatibility with broker APIs
        tag = f"{self.tag_prefix}-close-{symbol}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        # Check circuit breaker - but for closes we may want to try anyway
        # as closing positions is critical. Log warning but don't block.
        if self.circuit_breaker.is_open:
            logger.warning(
                f"Circuit breaker open but attempting close for {symbol} "
                f"(closing is critical operation)"
            )

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                order = self.broker_client.place_market_order(
                    symbol=symbol,
                    side=side,
                    quantity=shares,
                    tag=tag
                )

                # Record success
                self.circuit_breaker.record_success()

                logger.info(
                    f"Close order placed: {side} {shares} {symbol}, "
                    f"order_id={order.id}, tag={tag}"
                )

                return OrderResponse(
                    success=True,
                    order_id=order.id,
                    symbol=symbol,
                    side=side,
                    quantity=shares,
                    order_type='market',
                    status=order.status.value,
                    message="Position close order placed",
                    tag=tag
                )

            except Exception as e:
                last_error = e
                wrapped = wrap_exception(e)
                self.circuit_breaker.record_failure(e)

                # For closes, we retry even non-retryable errors (except auth)
                # because closing positions is critical
                if wrapped.category == ErrorCategory.AUTH_ERROR:
                    logger.error(f"Authentication error closing {symbol}: {wrapped}")
                    break

                if attempt < max_retries:
                    # Calculate backoff delay
                    delay = min(2 ** attempt, 30)
                    logger.warning(
                        f"Close attempt {attempt}/{max_retries} failed: {e}. "
                        f"Retrying in {delay}s..."
                    )
                    time.sleep(delay)
                else:
                    logger.error(
                        f"All {max_retries} close attempts failed for {symbol}: {e}"
                    )

        return OrderResponse(
            success=False,
            symbol=symbol,
            side=side,
            quantity=shares,
            message=str(last_error) if last_error else "Unknown error",
            tag=tag
        )

    def cancel_order(self, order_id: int, max_retries: int = 2) -> bool:
        """Cancel an open order with retry support.

        Args:
            order_id: Broker order ID
            max_retries: Maximum retry attempts

        Returns:
            True if cancelled successfully
        """
        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                result = self.broker_client.cancel_order(order_id)
                self.circuit_breaker.record_success()
                logger.info(f"Order {order_id} cancelled: {result}")
                return result

            except Exception as e:
                last_error = e
                wrapped = wrap_exception(e)
                self.circuit_breaker.record_failure(e)

                if not wrapped.retryable or attempt >= max_retries:
                    logger.error(f"Failed to cancel order {order_id}: {e}")
                    break

                delay = min(2 ** attempt, 10)
                logger.warning(
                    f"Cancel attempt {attempt}/{max_retries} failed: {e}. "
                    f"Retrying in {delay}s..."
                )
                time.sleep(delay)

        return False

    def get_order_status(self, order_id: int) -> Optional[Any]:
        """Get current status of an order with circuit breaker protection.

        Args:
            order_id: Broker order ID

        Returns:
            Order object or None
        """
        try:
            if self.circuit_breaker.is_open:
                logger.warning(f"Circuit breaker open - cannot get order {order_id}")
                return None

            order = self.broker_client.get_order(order_id)
            self.circuit_breaker.record_success()
            return order

        except Exception as e:
            self.circuit_breaker.record_failure(e)
            logger.error(f"Failed to get order {order_id}: {e}")
            return None

    def sync_with_broker(self) -> SyncResult:
        """Sync local database with broker state with circuit breaker protection.

        Fetches current positions and orders from broker (Alpaca),
        compares with local database, and reports discrepancies.

        Returns:
            SyncResult with sync details
        """
        if not self.db_conn:
            return SyncResult(
                success=False,
                discrepancies=[{"type": "error", "message": "No database connection"}]
            )

        # Check circuit breaker
        if self.circuit_breaker.is_open:
            return SyncResult(
                success=False,
                discrepancies=[{
                    "type": "circuit_open",
                    "message": f"Circuit breaker '{self.circuit_breaker.name}' is open"
                }]
            )

        discrepancies = []

        try:
            # Get broker positions
            broker_positions = self.broker_client.get_positions()

            # Get broker orders
            broker_orders = self.broker_client.get_orders()

            # Record success for broker API calls
            self.circuit_breaker.record_success()

            # Get local positions from database
            local_positions = self._get_local_positions()

            # Compare positions
            broker_symbols = {p.symbol for p in broker_positions}
            local_symbols = {p['symbol'] for p in local_positions}

            # Positions in broker but not local
            for symbol in broker_symbols - local_symbols:
                discrepancies.append({
                    "type": "position_missing_local",
                    "symbol": symbol,
                    "message": f"Position in broker but not in local DB"
                })

            # Positions in local but not broker
            for symbol in local_symbols - broker_symbols:
                discrepancies.append({
                    "type": "position_missing_broker",
                    "symbol": symbol,
                    "message": f"Position in local DB but not in broker"
                })

            # Quantity mismatches
            for bp in broker_positions:
                local = next(
                    (lp for lp in local_positions if lp['symbol'] == bp.symbol),
                    None
                )
                if local and local.get('quantity') != bp.quantity:
                    discrepancies.append({
                        "type": "quantity_mismatch",
                        "symbol": bp.symbol,
                        "broker_qty": bp.quantity,
                        "local_qty": local.get('quantity'),
                        "message": f"Quantity mismatch: broker={bp.quantity}, "
                                   f"local={local.get('quantity')}"
                    })

            # Log discrepancies
            if discrepancies:
                logger.warning(f"Sync found {len(discrepancies)} discrepancies")
                for d in discrepancies:
                    logger.warning(f"  - {d['type']}: {d['message']}")

            return SyncResult(
                success=True,
                positions_synced=len(broker_positions),
                orders_synced=len(broker_orders),
                discrepancies=discrepancies
            )

        except Exception as e:
            self.circuit_breaker.record_failure(e)
            wrapped = wrap_exception(e)
            logger.error(f"Sync failed: {wrapped}")
            return SyncResult(
                success=False,
                discrepancies=[{
                    "type": "error",
                    "category": wrapped.category.value,
                    "message": str(e)
                }]
            )

    def _wait_for_fill(self, order_id: int) -> Optional[Any]:
        """Wait for order to be filled.

        Args:
            order_id: Order ID to monitor

        Returns:
            Filled order or None if timeout/error
        """
        start_time = time.time()

        while time.time() - start_time < self.max_fill_wait:
            try:
                order = self.broker_client.get_order(order_id)

                if order.is_filled:
                    logger.info(
                        f"Order {order_id} filled: {order.filled_quantity} @ "
                        f"${order.avg_fill_price:.2f}"
                    )
                    return order

                if order.is_terminal and not order.is_filled:
                    logger.warning(
                        f"Order {order_id} in terminal state: {order.status.value}"
                    )
                    return None

                time.sleep(self.poll_interval)

            except Exception as e:
                logger.error(f"Error checking order {order_id}: {e}")
                time.sleep(self.poll_interval)

        logger.warning(f"Order {order_id} not filled within {self.max_fill_wait}s")
        return None

    def _record_orders(
        self,
        signal: Any,
        entry_order: OrderResponse,
        stop_order: Optional[OrderResponse],
        stop_price: Optional[float] = None
    ) -> Optional[int]:
        """Record orders in database.

        Args:
            signal: Original trade signal
            entry_order: Entry order response
            stop_order: Stop order response (None if software monitoring mode)
            stop_price: Stop loss price (used when stop_order is None)

        Returns:
            Position ID if created, None otherwise
        """
        if not self.db_conn:
            return None

        try:
            cursor = self.db_conn.cursor()
            today = datetime.now().date()
            now = datetime.now()

            # First create the position record (orders reference it via FK)
            # IMPORTANT: Always set strategy='gap_trading' to identify positions
            # from this strategy vs other trading strategies
            insert_position_sql = """
                INSERT INTO gap_trading.positions (
                    trade_date, symbol, direction, shares, entry_price, entry_time,
                    stop_loss, status, signal_id, strategy
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING position_id
            """

            # Determine stop loss price: use stop_order.price if available, else use stop_price param
            effective_stop_price = stop_order.price if stop_order else stop_price

            cursor.execute(insert_position_sql, (
                today,
                entry_order.symbol,
                'LONG' if entry_order.side == 'buy' else 'SHORT',
                entry_order.fill_quantity,
                entry_order.fill_price,
                now,
                effective_stop_price,  # Always record stop price for monitoring
                'OPEN',
                getattr(signal, 'id', None),
                'gap_trading'  # Strategy identifier for filtering
            ))
            position_id = cursor.fetchone()[0]

            # Insert into orders table - entry order
            # Include tag for broker verification/reconciliation
            insert_order_sql = """
                INSERT INTO gap_trading.orders (
                    trade_date, symbol, broker_order_id, order_type, side,
                    order_class, quantity, limit_price, stop_price, status,
                    fill_price, filled_quantity, filled_at, related_position_id,
                    related_signal_id, tag
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """

            # Entry order
            cursor.execute(insert_order_sql, (
                today,
                entry_order.symbol,
                str(entry_order.order_id) if entry_order.order_id else None,
                entry_order.order_type,
                entry_order.side,
                'equity',
                entry_order.quantity,
                None,  # limit_price (market orders don't have one)
                None,  # stop_price (entry is market order)
                entry_order.status,
                entry_order.fill_price,
                entry_order.fill_quantity,
                now if entry_order.fill_quantity else None,
                position_id,
                getattr(signal, 'id', None),
                entry_order.tag  # Tag for broker reconciliation
            ))
            entry_db_id = cursor.fetchone()[0]

            # Update position with entry_order_id
            cursor.execute(
                "UPDATE gap_trading.positions SET entry_order_id = %s WHERE position_id = %s",
                (entry_db_id, position_id)
            )

            # Stop order
            stop_db_id = None
            if stop_order and stop_order.success:
                cursor.execute(insert_order_sql, (
                    today,
                    stop_order.symbol,
                    str(stop_order.order_id) if stop_order.order_id else None,
                    stop_order.order_type,
                    stop_order.side,
                    'equity',
                    stop_order.quantity,
                    None,  # limit_price
                    stop_order.price,  # stop_price
                    stop_order.status,
                    stop_order.fill_price,
                    stop_order.fill_quantity,
                    None,  # filled_at (not filled yet)
                    position_id,
                    getattr(signal, 'id', None),
                    stop_order.tag  # Tag for broker reconciliation
                ))
                stop_db_id = cursor.fetchone()[0]

                # Update position with stop_order_id
                cursor.execute(
                    "UPDATE gap_trading.positions SET stop_order_id = %s WHERE position_id = %s",
                    (stop_db_id, position_id)
                )

            self.db_conn.commit()
            logger.info(f"Recorded position {position_id} for {entry_order.symbol}")

            return position_id

        except Exception as e:
            logger.error(f"Failed to record orders: {e}")
            if self.db_conn:
                self.db_conn.rollback()
            return None

    def _get_local_positions(self) -> List[Dict[str, Any]]:
        """Get open positions from local database.

        Returns:
            List of position dictionaries
        """
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT symbol, shares, direction, entry_price, stop_loss
                FROM gap_trading.positions
                WHERE status = 'OPEN'
            """)

            # Map to expected keys for sync comparison
            columns = ['symbol', 'quantity', 'side', 'entry_price', 'stop_price']
            return [dict(zip(columns, row)) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get local positions: {e}")
            return []
