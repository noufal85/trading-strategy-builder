"""Position Manager for Gap Trading Strategy.

Manages position lifecycle from creation through close,
tracking P&L, stop-loss levels, and position status.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class PositionStatus(str, Enum):
    """Position lifecycle status."""
    PENDING = 'PENDING'          # Order placed, awaiting fill
    OPEN = 'OPEN'                # Position is open
    STOPPED = 'STOPPED'          # Closed by stop-loss
    EOD_CLOSED = 'EOD_CLOSED'    # Closed at end of day
    MANUAL_CLOSED = 'MANUAL_CLOSED'  # Manually closed
    ERROR = 'ERROR'              # Error state


class PositionSide(str, Enum):
    """Position direction."""
    LONG = 'LONG'
    SHORT = 'SHORT'


@dataclass
class Position:
    """Represents a trading position.

    Attributes:
        id: Database position ID
        symbol: Stock ticker symbol
        side: LONG or SHORT
        quantity: Number of shares
        entry_price: Average entry price
        entry_time: Position open timestamp
        stop_price: Current stop-loss price
        atr_at_entry: ATR value at entry (for stop calculations)
        risk_tier: Risk tier at entry (LOW, MEDIUM, HIGH)
        entry_order_id: Tradier entry order ID
        stop_order_id: Tradier stop order ID
        signal_id: Original signal ID
        status: Current position status
        exit_price: Exit price (if closed)
        exit_time: Exit timestamp (if closed)
        exit_reason: Reason for close
        realized_pnl: Realized P&L (if closed)
        trade_date: Trading date
    """
    id: Optional[int] = None
    symbol: str = ''
    side: PositionSide = PositionSide.LONG
    quantity: int = 0
    entry_price: float = 0.0
    entry_time: Optional[datetime] = None
    stop_price: Optional[float] = None
    atr_at_entry: Optional[float] = None
    risk_tier: Optional[str] = None
    entry_order_id: Optional[int] = None
    stop_order_id: Optional[int] = None
    signal_id: Optional[int] = None
    status: PositionStatus = PositionStatus.PENDING
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    exit_reason: Optional[str] = None
    realized_pnl: Optional[float] = None
    trade_date: Optional[date] = None

    @property
    def is_long(self) -> bool:
        """Check if position is long."""
        return self.side == PositionSide.LONG

    @property
    def is_short(self) -> bool:
        """Check if position is short."""
        return self.side == PositionSide.SHORT

    @property
    def is_open(self) -> bool:
        """Check if position is still open."""
        return self.status in (PositionStatus.PENDING, PositionStatus.OPEN)

    @property
    def notional_value(self) -> float:
        """Calculate notional value at entry."""
        return abs(self.quantity) * self.entry_price

    def calculate_unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L in dollars
        """
        if self.is_long:
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * abs(self.quantity)

    def calculate_unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L as percentage.

        Args:
            current_price: Current market price

        Returns:
            Unrealized P&L as percentage
        """
        if self.entry_price == 0:
            return 0.0

        pnl = self.calculate_unrealized_pnl(current_price)
        return (pnl / self.notional_value) * 100

    def is_stop_hit(self, current_price: float) -> bool:
        """Check if stop-loss has been hit.

        Args:
            current_price: Current market price

        Returns:
            True if stop-loss triggered
        """
        if self.stop_price is None:
            return False

        if self.is_long:
            return current_price <= self.stop_price
        else:
            return current_price >= self.stop_price

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'id': self.id,
            'symbol': self.symbol,
            'side': self.side.value,
            'quantity': self.quantity,
            'entry_price': self.entry_price,
            'entry_time': self.entry_time.isoformat() if self.entry_time else None,
            'stop_price': self.stop_price,
            'atr_at_entry': self.atr_at_entry,
            'risk_tier': self.risk_tier,
            'entry_order_id': self.entry_order_id,
            'stop_order_id': self.stop_order_id,
            'signal_id': self.signal_id,
            'status': self.status.value,
            'exit_price': self.exit_price,
            'exit_time': self.exit_time.isoformat() if self.exit_time else None,
            'exit_reason': self.exit_reason,
            'realized_pnl': self.realized_pnl,
            'trade_date': self.trade_date.isoformat() if self.trade_date else None,
        }


@dataclass
class ExitInfo:
    """Information about position exit.

    Attributes:
        exit_price: Exit price
        exit_time: Exit timestamp
        exit_reason: Reason for exit
        close_order_id: Tradier close order ID
    """
    exit_price: float
    exit_time: datetime = field(default_factory=datetime.now)
    exit_reason: str = 'manual'
    close_order_id: Optional[int] = None


class PositionManager:
    """Manages position lifecycle for gap trading strategy.

    Handles position creation, updates, closing, and P&L calculation.

    Attributes:
        db_conn: Database connection for position persistence
    """

    def __init__(self, db_conn: Any = None):
        """Initialize PositionManager.

        Args:
            db_conn: Database connection (psycopg2 or SQLAlchemy)
        """
        self.db_conn = db_conn
        logger.info("PositionManager initialized")

    def create_position(
        self,
        symbol: str,
        side: PositionSide,
        quantity: int,
        entry_price: float,
        stop_price: Optional[float] = None,
        atr_at_entry: Optional[float] = None,
        risk_tier: Optional[str] = None,
        entry_order_id: Optional[int] = None,
        stop_order_id: Optional[int] = None,
        signal_id: Optional[int] = None
    ) -> Position:
        """Create a new position.

        Args:
            symbol: Stock ticker
            side: LONG or SHORT
            quantity: Number of shares
            entry_price: Entry price
            stop_price: Stop-loss price
            atr_at_entry: ATR at entry
            risk_tier: Risk tier classification
            entry_order_id: Tradier entry order ID
            stop_order_id: Tradier stop order ID
            signal_id: Original signal ID

        Returns:
            Created Position object
        """
        position = Position(
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            entry_time=datetime.now(),
            stop_price=stop_price,
            atr_at_entry=atr_at_entry,
            risk_tier=risk_tier,
            entry_order_id=entry_order_id,
            stop_order_id=stop_order_id,
            signal_id=signal_id,
            status=PositionStatus.OPEN,
            trade_date=date.today()
        )

        # Persist to database
        if self.db_conn:
            position.id = self._insert_position(position)

        logger.info(
            f"Created position: {side.value} {quantity} {symbol} @ "
            f"${entry_price:.2f}, stop=${stop_price:.2f if stop_price else 'N/A'}"
        )

        return position

    def update_position(
        self,
        position_id: int,
        updates: Dict[str, Any]
    ) -> Optional[Position]:
        """Update an existing position.

        Args:
            position_id: Position ID to update
            updates: Dictionary of fields to update

        Returns:
            Updated Position or None if not found
        """
        position = self.get_position(position_id)
        if not position:
            logger.warning(f"Position {position_id} not found for update")
            return None

        # Apply updates
        for key, value in updates.items():
            if hasattr(position, key):
                setattr(position, key, value)

        # Persist updates
        if self.db_conn:
            self._update_position_db(position)

        logger.info(f"Updated position {position_id}: {list(updates.keys())}")
        return position

    def close_position(
        self,
        position_id: int,
        exit_info: ExitInfo
    ) -> Optional[Position]:
        """Close a position.

        Args:
            position_id: Position ID to close
            exit_info: Exit details

        Returns:
            Closed Position or None if not found
        """
        position = self.get_position(position_id)
        if not position:
            logger.warning(f"Position {position_id} not found for close")
            return None

        if not position.is_open:
            logger.warning(
                f"Position {position_id} already closed: {position.status.value}"
            )
            return position

        # Calculate realized P&L
        if position.is_long:
            realized_pnl = (exit_info.exit_price - position.entry_price) * position.quantity
        else:
            realized_pnl = (position.entry_price - exit_info.exit_price) * abs(position.quantity)

        # Determine status based on exit reason
        status_map = {
            'stop_loss': PositionStatus.STOPPED,
            'stopped': PositionStatus.STOPPED,
            'eod': PositionStatus.EOD_CLOSED,
            'end_of_day': PositionStatus.EOD_CLOSED,
            'manual': PositionStatus.MANUAL_CLOSED,
        }
        new_status = status_map.get(
            exit_info.exit_reason.lower(),
            PositionStatus.MANUAL_CLOSED
        )

        # Update position
        position.exit_price = exit_info.exit_price
        position.exit_time = exit_info.exit_time
        position.exit_reason = exit_info.exit_reason
        position.realized_pnl = realized_pnl
        position.status = new_status

        # Persist to database
        if self.db_conn:
            self._update_position_db(position)

        logger.info(
            f"Closed position {position_id}: {position.symbol} "
            f"@ ${exit_info.exit_price:.2f}, P&L=${realized_pnl:+.2f} "
            f"({exit_info.exit_reason})"
        )

        return position

    def get_position(self, position_id: int) -> Optional[Position]:
        """Get position by ID.

        Args:
            position_id: Position ID

        Returns:
            Position or None if not found
        """
        if not self.db_conn:
            return None

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT id, symbol, side, quantity, entry_price, entry_time,
                       stop_price, atr_at_entry, risk_tier, entry_order_id,
                       stop_order_id, signal_id, status, exit_price, exit_time,
                       exit_reason, realized_pnl, trade_date
                FROM gap_trading.positions
                WHERE id = %s
            """, (position_id,))

            row = cursor.fetchone()
            if not row:
                return None

            return self._row_to_position(row)

        except Exception as e:
            logger.error(f"Failed to get position {position_id}: {e}")
            return None

    def get_position_by_symbol(self, symbol: str) -> Optional[Position]:
        """Get open position for a symbol.

        Args:
            symbol: Stock ticker

        Returns:
            Position or None if not found
        """
        positions = self.get_open_positions()
        for pos in positions:
            if pos.symbol.upper() == symbol.upper():
                return pos
        return None

    def get_open_positions(self) -> List[Position]:
        """Get all open positions.

        Returns:
            List of open Position objects
        """
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT id, symbol, side, quantity, entry_price, entry_time,
                       stop_price, atr_at_entry, risk_tier, entry_order_id,
                       stop_order_id, signal_id, status, exit_price, exit_time,
                       exit_reason, realized_pnl, trade_date
                FROM gap_trading.positions
                WHERE status IN ('OPEN', 'PENDING')
                ORDER BY entry_time DESC
            """)

            return [self._row_to_position(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get open positions: {e}")
            return []

    def get_today_positions(self) -> List[Position]:
        """Get all positions opened today.

        Returns:
            List of today's Position objects
        """
        if not self.db_conn:
            return []

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                SELECT id, symbol, side, quantity, entry_price, entry_time,
                       stop_price, atr_at_entry, risk_tier, entry_order_id,
                       stop_order_id, signal_id, status, exit_price, exit_time,
                       exit_reason, realized_pnl, trade_date
                FROM gap_trading.positions
                WHERE trade_date = CURRENT_DATE
                ORDER BY entry_time DESC
            """)

            return [self._row_to_position(row) for row in cursor.fetchall()]

        except Exception as e:
            logger.error(f"Failed to get today's positions: {e}")
            return []

    def calculate_total_pnl(self, positions: Optional[List[Position]] = None) -> Dict[str, float]:
        """Calculate total P&L for positions.

        Args:
            positions: List of positions (defaults to today's positions)

        Returns:
            Dictionary with P&L breakdown
        """
        if positions is None:
            positions = self.get_today_positions()

        realized = 0.0
        win_count = 0
        loss_count = 0

        for pos in positions:
            if pos.realized_pnl is not None:
                realized += pos.realized_pnl
                if pos.realized_pnl > 0:
                    win_count += 1
                elif pos.realized_pnl < 0:
                    loss_count += 1

        total_trades = win_count + loss_count
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0.0

        return {
            'realized_pnl': realized,
            'win_count': win_count,
            'loss_count': loss_count,
            'total_trades': total_trades,
            'win_rate': win_rate
        }

    def get_positions_requiring_close(self) -> List[Position]:
        """Get positions that need to be closed (EOD).

        Returns:
            List of positions to close
        """
        return [p for p in self.get_open_positions() if p.status == PositionStatus.OPEN]

    def _insert_position(self, position: Position) -> Optional[int]:
        """Insert position into database.

        Args:
            position: Position to insert

        Returns:
            New position ID or None
        """
        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                INSERT INTO gap_trading.positions (
                    symbol, side, quantity, entry_price, entry_time,
                    stop_price, atr_at_entry, risk_tier, entry_order_id,
                    stop_order_id, signal_id, status, trade_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (
                position.symbol,
                position.side.value,
                position.quantity,
                position.entry_price,
                position.entry_time,
                position.stop_price,
                position.atr_at_entry,
                position.risk_tier,
                position.entry_order_id,
                position.stop_order_id,
                position.signal_id,
                position.status.value,
                position.trade_date
            ))

            position_id = cursor.fetchone()[0]
            self.db_conn.commit()
            return position_id

        except Exception as e:
            logger.error(f"Failed to insert position: {e}")
            if self.db_conn:
                self.db_conn.rollback()
            return None

    def _update_position_db(self, position: Position) -> bool:
        """Update position in database.

        Args:
            position: Position to update

        Returns:
            True if successful
        """
        if not position.id:
            return False

        try:
            cursor = self.db_conn.cursor()
            cursor.execute("""
                UPDATE gap_trading.positions SET
                    stop_price = %s,
                    stop_order_id = %s,
                    status = %s,
                    exit_price = %s,
                    exit_time = %s,
                    exit_reason = %s,
                    realized_pnl = %s
                WHERE id = %s
            """, (
                position.stop_price,
                position.stop_order_id,
                position.status.value,
                position.exit_price,
                position.exit_time,
                position.exit_reason,
                position.realized_pnl,
                position.id
            ))

            self.db_conn.commit()
            return True

        except Exception as e:
            logger.error(f"Failed to update position {position.id}: {e}")
            if self.db_conn:
                self.db_conn.rollback()
            return False

    def _row_to_position(self, row: tuple) -> Position:
        """Convert database row to Position object.

        Args:
            row: Database row tuple

        Returns:
            Position object
        """
        return Position(
            id=row[0],
            symbol=row[1],
            side=PositionSide(row[2]),
            quantity=row[3],
            entry_price=float(row[4]) if row[4] else 0.0,
            entry_time=row[5],
            stop_price=float(row[6]) if row[6] else None,
            atr_at_entry=float(row[7]) if row[7] else None,
            risk_tier=row[8],
            entry_order_id=row[9],
            stop_order_id=row[10],
            signal_id=row[11],
            status=PositionStatus(row[12]) if row[12] else PositionStatus.OPEN,
            exit_price=float(row[13]) if row[13] else None,
            exit_time=row[14],
            exit_reason=row[15],
            realized_pnl=float(row[16]) if row[16] else None,
            trade_date=row[17]
        )
