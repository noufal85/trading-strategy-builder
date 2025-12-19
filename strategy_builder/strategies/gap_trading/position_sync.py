"""Position Sync Module for Gap Trading Strategy.

Reconciles database state with actual broker (Alpaca) positions.
Detects discrepancies and auto-fixes them with proper logging.

Usage:
    from strategy_builder.strategies.gap_trading.position_sync import PositionSyncManager

    sync_manager = PositionSyncManager(db_conn, broker_client)
    result = sync_manager.run_full_sync()
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, TYPE_CHECKING
from enum import Enum

if TYPE_CHECKING:
    from stock_data_web.alpaca import AlpacaClient

logger = logging.getLogger(__name__)


class DiscrepancyType(str, Enum):
    """Types of position discrepancies."""
    POSITION_MISSING_BROKER = 'position_missing_broker'  # In DB but not in broker
    POSITION_MISSING_DB = 'position_missing_db'          # In broker but not in DB
    QUANTITY_MISMATCH = 'quantity_mismatch'              # Different quantities
    DIRECTION_MISMATCH = 'direction_mismatch'            # Long vs Short mismatch
    STALE_STOP_ORDER = 'stale_stop_order'                # Stop order for closed position


class SyncAction(str, Enum):
    """Actions taken during sync."""
    MARK_CLOSED = 'mark_closed'
    UPDATE_QUANTITY = 'update_quantity'
    CANCEL_STOP = 'cancel_stop'
    CREATE_POSITION = 'create_position'
    NO_ACTION = 'no_action'


@dataclass
class Discrepancy:
    """Represents a single discrepancy found during sync."""
    type: DiscrepancyType
    symbol: str
    message: str
    db_value: Any = None
    broker_value: Any = None
    position_id: Optional[int] = None
    action_taken: SyncAction = SyncAction.NO_ACTION
    action_success: bool = False
    error: Optional[str] = None


@dataclass
class SyncResult:
    """Result of a full sync operation."""
    success: bool
    sync_time: datetime = field(default_factory=datetime.now)
    db_positions_count: int = 0
    broker_positions_count: int = 0
    discrepancies_found: int = 0
    discrepancies_fixed: int = 0
    discrepancies: List[Discrepancy] = field(default_factory=list)
    duration_ms: int = 0
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'success': self.success,
            'sync_time': self.sync_time.isoformat(),
            'db_positions_count': self.db_positions_count,
            'broker_positions_count': self.broker_positions_count,
            'discrepancies_found': self.discrepancies_found,
            'discrepancies_fixed': self.discrepancies_fixed,
            'duration_ms': self.duration_ms,
            'errors': self.errors,
            'discrepancies': [
                {
                    'type': d.type.value,
                    'symbol': d.symbol,
                    'message': d.message,
                    'action_taken': d.action_taken.value,
                    'action_success': d.action_success,
                    'error': d.error
                }
                for d in self.discrepancies
            ]
        }


class PositionSyncManager:
    """Manages position synchronization between database and broker.

    This class handles:
    - Fetching positions from database and broker
    - Detecting discrepancies
    - Auto-fixing discrepancies (marking closed, cancelling stops)
    - Logging sync operations to database
    - Sending alerts on significant desyncs
    """

    def __init__(
        self,
        db_conn,
        broker_client: 'AlpacaClient',
        auto_fix: bool = True,
        dry_run: bool = False
    ):
        """Initialize the sync manager.

        Args:
            db_conn: Database connection (psycopg2 or PostgresHook connection)
            broker_client: AlpacaClient instance
            auto_fix: Whether to automatically fix discrepancies
            dry_run: If True, detect but don't apply fixes
        """
        self.db_conn = db_conn
        self.broker_client = broker_client
        self.auto_fix = auto_fix
        self.dry_run = dry_run

    def run_full_sync(self) -> SyncResult:
        """Run a full position synchronization.

        Returns:
            SyncResult with details of the sync operation
        """
        start_time = datetime.now()
        discrepancies: List[Discrepancy] = []
        errors: List[str] = []

        try:
            # Step 1: Fetch positions from both sources
            logger.info("Fetching positions from database and broker...")

            db_positions = self._get_db_positions()
            broker_positions = self._get_broker_positions()

            logger.info(f"Found {len(db_positions)} DB positions, {len(broker_positions)} broker positions")

            # Step 2: Compare and find discrepancies
            discrepancies = self._compare_positions(db_positions, broker_positions)

            if discrepancies:
                logger.warning(f"Found {len(discrepancies)} discrepancies")
                for d in discrepancies:
                    logger.warning(f"  - {d.type.value}: {d.symbol} - {d.message}")
            else:
                logger.info("No discrepancies found - positions in sync")

            # Step 3: Fix discrepancies if enabled
            fixed_count = 0
            if self.auto_fix and not self.dry_run and discrepancies:
                fixed_count = self._fix_discrepancies(discrepancies)
                logger.info(f"Fixed {fixed_count}/{len(discrepancies)} discrepancies")

            # Step 4: Check for stale stop orders
            stale_stops = self._find_stale_stop_orders(db_positions, broker_positions)
            discrepancies.extend(stale_stops)

            if stale_stops and self.auto_fix and not self.dry_run:
                fixed_count += self._cancel_stale_stops(stale_stops)

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            result = SyncResult(
                success=True,
                sync_time=start_time,
                db_positions_count=len(db_positions),
                broker_positions_count=len(broker_positions),
                discrepancies_found=len(discrepancies),
                discrepancies_fixed=fixed_count,
                discrepancies=discrepancies,
                duration_ms=duration_ms,
                errors=errors
            )

            # Step 5: Log sync to database
            self._log_sync_result(result)

            return result

        except Exception as e:
            logger.error(f"Sync failed with error: {e}")
            errors.append(str(e))

            duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)

            return SyncResult(
                success=False,
                sync_time=start_time,
                discrepancies_found=len(discrepancies),
                discrepancies=discrepancies,
                duration_ms=duration_ms,
                errors=errors
            )

    def _get_db_positions(self) -> List[Dict[str, Any]]:
        """Fetch open positions from database.

        Returns:
            List of position dictionaries
        """
        cursor = self.db_conn.cursor()

        cursor.execute("""
            SELECT
                position_id,
                symbol,
                direction,
                shares,
                entry_price,
                status,
                stop_order_id
            FROM gap_trading.positions
            WHERE status = 'OPEN'
            ORDER BY symbol
        """)

        columns = ['position_id', 'symbol', 'direction', 'shares', 'entry_price', 'status', 'stop_order_id']
        positions = [dict(zip(columns, row)) for row in cursor.fetchall()]

        return positions

    def _get_broker_positions(self) -> List[Dict[str, Any]]:
        """Fetch positions from broker (Alpaca).

        Returns:
            List of position dictionaries
        """
        broker_positions = self.broker_client.get_positions()

        positions = []
        for p in broker_positions:
            positions.append({
                'symbol': p.symbol,
                'quantity': abs(int(float(p.quantity))),
                'direction': 'LONG' if p.is_long else 'SHORT',
                'entry_price': float(p.avg_entry_price),
                'current_price': float(p.current_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl)
            })

        return positions

    def _compare_positions(
        self,
        db_positions: List[Dict[str, Any]],
        broker_positions: List[Dict[str, Any]]
    ) -> List[Discrepancy]:
        """Compare DB and broker positions to find discrepancies.

        Args:
            db_positions: Positions from database
            broker_positions: Positions from broker

        Returns:
            List of discrepancies found
        """
        discrepancies: List[Discrepancy] = []

        # Build lookup sets
        db_symbols: Set[str] = {p['symbol'] for p in db_positions}
        broker_symbols: Set[str] = {p['symbol'] for p in broker_positions}

        # Positions in DB but not in broker (should be marked closed)
        for symbol in db_symbols - broker_symbols:
            db_pos = next(p for p in db_positions if p['symbol'] == symbol)
            discrepancies.append(Discrepancy(
                type=DiscrepancyType.POSITION_MISSING_BROKER,
                symbol=symbol,
                message=f"Position marked OPEN in DB but not found in broker",
                db_value={'shares': db_pos['shares'], 'direction': db_pos['direction']},
                broker_value=None,
                position_id=db_pos['position_id']
            ))

        # Positions in broker but not in DB (unexpected - log but don't auto-fix)
        for symbol in broker_symbols - db_symbols:
            broker_pos = next(p for p in broker_positions if p['symbol'] == symbol)
            discrepancies.append(Discrepancy(
                type=DiscrepancyType.POSITION_MISSING_DB,
                symbol=symbol,
                message=f"Position found in broker but not in DB (qty: {broker_pos['quantity']})",
                db_value=None,
                broker_value={'quantity': broker_pos['quantity'], 'direction': broker_pos['direction']}
            ))

        # Check for mismatches in common positions
        common_symbols = db_symbols & broker_symbols
        for symbol in common_symbols:
            db_pos = next(p for p in db_positions if p['symbol'] == symbol)
            broker_pos = next(p for p in broker_positions if p['symbol'] == symbol)

            # Quantity mismatch
            if db_pos['shares'] != broker_pos['quantity']:
                discrepancies.append(Discrepancy(
                    type=DiscrepancyType.QUANTITY_MISMATCH,
                    symbol=symbol,
                    message=f"Quantity mismatch: DB={db_pos['shares']}, Broker={broker_pos['quantity']}",
                    db_value=db_pos['shares'],
                    broker_value=broker_pos['quantity'],
                    position_id=db_pos['position_id']
                ))

            # Direction mismatch
            if db_pos['direction'] != broker_pos['direction']:
                discrepancies.append(Discrepancy(
                    type=DiscrepancyType.DIRECTION_MISMATCH,
                    symbol=symbol,
                    message=f"Direction mismatch: DB={db_pos['direction']}, Broker={broker_pos['direction']}",
                    db_value=db_pos['direction'],
                    broker_value=broker_pos['direction'],
                    position_id=db_pos['position_id']
                ))

        return discrepancies

    def _find_stale_stop_orders(
        self,
        db_positions: List[Dict[str, Any]],
        broker_positions: List[Dict[str, Any]]
    ) -> List[Discrepancy]:
        """Find stop orders that reference closed positions.

        Args:
            db_positions: Positions from database
            broker_positions: Positions from broker

        Returns:
            List of stale stop order discrepancies
        """
        discrepancies: List[Discrepancy] = []

        cursor = self.db_conn.cursor()

        # Find stop orders that are still 'open' but position doesn't exist in broker
        broker_symbols = {p['symbol'] for p in broker_positions}

        cursor.execute("""
            SELECT o.id, o.symbol, o.tradier_order_id, p.position_id, p.status as pos_status
            FROM gap_trading.orders o
            LEFT JOIN gap_trading.positions p ON o.related_position_id = p.position_id
            WHERE o.order_type = 'stop'
            AND o.status = 'open'
        """)

        for row in cursor.fetchall():
            order_id, symbol, broker_order_id, position_id, pos_status = row

            # If position is closed or symbol not in broker, stop order is stale
            if pos_status == 'CLOSED' or symbol not in broker_symbols:
                discrepancies.append(Discrepancy(
                    type=DiscrepancyType.STALE_STOP_ORDER,
                    symbol=symbol,
                    message=f"Stop order {order_id} is stale (position {'closed' if pos_status == 'CLOSED' else 'not in broker'})",
                    db_value={'order_id': order_id, 'broker_order_id': broker_order_id},
                    position_id=position_id
                ))

        return discrepancies

    def _fix_discrepancies(self, discrepancies: List[Discrepancy]) -> int:
        """Fix detected discrepancies.

        Args:
            discrepancies: List of discrepancies to fix

        Returns:
            Number of discrepancies successfully fixed
        """
        fixed_count = 0
        cursor = self.db_conn.cursor()

        for d in discrepancies:
            try:
                if d.type == DiscrepancyType.POSITION_MISSING_BROKER:
                    # Mark position as closed - it was closed in broker
                    cursor.execute("""
                        UPDATE gap_trading.positions
                        SET status = 'CLOSED',
                            exit_reason = 'MANUAL',
                            exit_time = NOW(),
                            updated_at = NOW(),
                            last_close_error = 'Closed by sync - not found in broker'
                        WHERE position_id = %s
                    """, (d.position_id,))

                    d.action_taken = SyncAction.MARK_CLOSED
                    d.action_success = True
                    fixed_count += 1
                    logger.info(f"Marked position {d.position_id} ({d.symbol}) as CLOSED")

                elif d.type == DiscrepancyType.QUANTITY_MISMATCH:
                    # Update quantity to match broker
                    cursor.execute("""
                        UPDATE gap_trading.positions
                        SET shares = %s,
                            updated_at = NOW()
                        WHERE position_id = %s
                    """, (d.broker_value, d.position_id))

                    d.action_taken = SyncAction.UPDATE_QUANTITY
                    d.action_success = True
                    fixed_count += 1
                    logger.info(f"Updated {d.symbol} quantity from {d.db_value} to {d.broker_value}")

                elif d.type == DiscrepancyType.POSITION_MISSING_DB:
                    # Don't auto-create positions - just log
                    d.action_taken = SyncAction.NO_ACTION
                    d.action_success = False
                    d.error = "Manual intervention required - position exists in broker but not tracked in DB"
                    logger.warning(f"Position {d.symbol} in broker but not in DB - requires manual review")

                else:
                    d.action_taken = SyncAction.NO_ACTION

            except Exception as e:
                d.action_success = False
                d.error = str(e)
                logger.error(f"Failed to fix {d.type.value} for {d.symbol}: {e}")

        self.db_conn.commit()
        return fixed_count

    def _cancel_stale_stops(self, stale_stops: List[Discrepancy]) -> int:
        """Cancel stale stop orders in database.

        Args:
            stale_stops: List of stale stop order discrepancies

        Returns:
            Number of stops cancelled
        """
        cancelled = 0
        cursor = self.db_conn.cursor()

        for d in stale_stops:
            try:
                order_id = d.db_value.get('order_id')

                cursor.execute("""
                    UPDATE gap_trading.orders
                    SET status = 'cancelled',
                        updated_at = NOW()
                    WHERE id = %s
                """, (order_id,))

                d.action_taken = SyncAction.CANCEL_STOP
                d.action_success = True
                cancelled += 1
                logger.info(f"Cancelled stale stop order {order_id} for {d.symbol}")

            except Exception as e:
                d.action_success = False
                d.error = str(e)
                logger.error(f"Failed to cancel stop order for {d.symbol}: {e}")

        self.db_conn.commit()
        return cancelled

    def _log_sync_result(self, result: SyncResult) -> None:
        """Log sync result to database.

        Args:
            result: SyncResult to log
        """
        try:
            cursor = self.db_conn.cursor()

            # Check if sync_log table exists, create if not
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS gap_trading.sync_log (
                    id SERIAL PRIMARY KEY,
                    sync_time TIMESTAMP NOT NULL,
                    db_positions INTEGER,
                    broker_positions INTEGER,
                    discrepancies_found INTEGER,
                    discrepancies_fixed INTEGER,
                    duration_ms INTEGER,
                    success BOOLEAN,
                    errors JSONB,
                    details JSONB
                )
            """)

            import json

            cursor.execute("""
                INSERT INTO gap_trading.sync_log
                (sync_time, db_positions, broker_positions, discrepancies_found,
                 discrepancies_fixed, duration_ms, success, errors, details)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                result.sync_time,
                result.db_positions_count,
                result.broker_positions_count,
                result.discrepancies_found,
                result.discrepancies_fixed,
                result.duration_ms,
                result.success,
                json.dumps(result.errors),
                json.dumps(result.to_dict())
            ))

            self.db_conn.commit()
            logger.info("Sync result logged to database")

        except Exception as e:
            logger.error(f"Failed to log sync result: {e}")

    def generate_alert_message(self, result: SyncResult) -> Optional[str]:
        """Generate Telegram alert message if there were discrepancies.

        Args:
            result: SyncResult from sync operation

        Returns:
            Alert message string or None if no alert needed
        """
        if result.discrepancies_found == 0:
            return None

        # Determine severity
        if result.discrepancies_found > 5:
            emoji = "🔴"
            severity = "Critical"
        elif result.discrepancies_found > 2:
            emoji = "🟡"
            severity = "Warning"
        else:
            emoji = "🟠"
            severity = "Info"

        message = f"""
{emoji} *Position Sync Alert* - {severity}
{result.sync_time.strftime('%Y-%m-%d %H:%M')} ET

*Summary:*
• DB Positions: {result.db_positions_count}
• Broker Positions: {result.broker_positions_count}
• Discrepancies Found: {result.discrepancies_found}
• Discrepancies Fixed: {result.discrepancies_fixed}

*Details:*
"""

        # Group by type
        by_type: Dict[str, List[Discrepancy]] = {}
        for d in result.discrepancies:
            type_name = d.type.value
            if type_name not in by_type:
                by_type[type_name] = []
            by_type[type_name].append(d)

        for type_name, items in by_type.items():
            symbols = [d.symbol for d in items[:5]]
            message += f"• {type_name}: {', '.join(symbols)}"
            if len(items) > 5:
                message += f" (+{len(items) - 5} more)"
            message += "\n"

        if result.errors:
            message += f"\n*Errors:* {len(result.errors)}\n"
            for err in result.errors[:3]:
                message += f"• {err[:50]}...\n"

        message += f"\n_Duration: {result.duration_ms}ms_"

        return message


# Convenience function for DAG usage
def run_position_sync(
    db_conn,
    broker_client: 'AlpacaClient',
    auto_fix: bool = True,
    dry_run: bool = False,
    send_alert: bool = True
) -> Dict[str, Any]:
    """Run position sync and optionally send alerts.

    Args:
        db_conn: Database connection
        broker_client: AlpacaClient instance
        auto_fix: Whether to auto-fix discrepancies
        dry_run: If True, detect but don't fix
        send_alert: Whether to send Telegram alert on discrepancies

    Returns:
        Dictionary with sync results
    """
    sync_manager = PositionSyncManager(
        db_conn=db_conn,
        broker_client=broker_client,
        auto_fix=auto_fix,
        dry_run=dry_run
    )

    result = sync_manager.run_full_sync()

    # Send alert if needed
    alert_message = None
    if send_alert and result.discrepancies_found > 0:
        alert_message = sync_manager.generate_alert_message(result)

        if alert_message:
            try:
                from airflow_common.notifications.telegram import TelegramNotifier
                notifier = TelegramNotifier()
                notifier.send_message(alert_message, parse_mode="Markdown")
                logger.info("Desync alert sent to Telegram")
            except Exception as e:
                logger.warning(f"Failed to send Telegram alert: {e}")

    return {
        'success': result.success,
        'db_positions': result.db_positions_count,
        'broker_positions': result.broker_positions_count,
        'discrepancies_found': result.discrepancies_found,
        'discrepancies_fixed': result.discrepancies_fixed,
        'duration_ms': result.duration_ms,
        'errors': result.errors,
        'alert_sent': alert_message is not None
    }
