"""Unit tests for Gap Trading position sync module.

Tests cover:
- Position comparison logic
- Discrepancy detection
- Sync result generation
- Circuit breaker integration
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime
from dataclasses import dataclass

from strategy_builder.strategies.gap_trading.position_sync import (
    PositionSyncManager,
    SyncResult,
    Discrepancy,
    DiscrepancyType,
    SyncAction,
    run_position_sync,
)
from strategy_builder.strategies.gap_trading.error_handling import (
    CircuitBreaker,
    CircuitOpenError,
    NetworkError,
    reset_broker_circuit_breaker,
)


@dataclass
class MockPosition:
    """Mock broker position object."""
    symbol: str
    quantity: str
    is_long: bool
    avg_entry_price: str
    current_price: str
    market_value: str
    unrealized_pl: str


@pytest.fixture
def mock_db_conn():
    """Create mock database connection."""
    conn = Mock()
    cursor = Mock()
    conn.cursor.return_value = cursor
    cursor.fetchall.return_value = []
    return conn


@pytest.fixture
def mock_broker_client():
    """Create mock broker client."""
    client = Mock()
    client.get_positions.return_value = []
    return client


@pytest.fixture
def mock_circuit_breaker():
    """Create mock circuit breaker."""
    cb = CircuitBreaker(name="test", failure_threshold=5, timeout=300)
    return cb


class TestDiscrepancy:
    """Tests for Discrepancy dataclass."""

    def test_basic_creation(self):
        """Test basic discrepancy creation."""
        d = Discrepancy(
            type=DiscrepancyType.POSITION_MISSING_BROKER,
            symbol="AAPL",
            message="Position not found in broker"
        )
        assert d.type == DiscrepancyType.POSITION_MISSING_BROKER
        assert d.symbol == "AAPL"
        assert d.action_taken == SyncAction.NO_ACTION
        assert d.action_success is False

    def test_with_values(self):
        """Test discrepancy with db/broker values."""
        d = Discrepancy(
            type=DiscrepancyType.QUANTITY_MISMATCH,
            symbol="TSLA",
            message="Quantity mismatch",
            db_value=100,
            broker_value=50,
            position_id=123
        )
        assert d.db_value == 100
        assert d.broker_value == 50
        assert d.position_id == 123


class TestSyncResult:
    """Tests for SyncResult dataclass."""

    def test_basic_creation(self):
        """Test basic sync result creation."""
        result = SyncResult(success=True)
        assert result.success is True
        assert result.discrepancies_found == 0
        assert result.discrepancies_fixed == 0
        assert result.errors == []

    def test_to_dict(self):
        """Test conversion to dictionary."""
        result = SyncResult(
            success=True,
            db_positions_count=5,
            broker_positions_count=3,
            discrepancies_found=2,
            discrepancies_fixed=1,
            duration_ms=150
        )
        d = result.to_dict()

        assert d['success'] is True
        assert d['db_positions_count'] == 5
        assert d['broker_positions_count'] == 3
        assert d['discrepancies_found'] == 2
        assert d['discrepancies_fixed'] == 1
        assert d['duration_ms'] == 150
        assert 'sync_time' in d


class TestPositionSyncManager:
    """Tests for PositionSyncManager class."""

    def test_init_default_circuit_breaker(self, mock_db_conn, mock_broker_client):
        """Test initialization with default circuit breaker."""
        reset_broker_circuit_breaker()
        manager = PositionSyncManager(mock_db_conn, mock_broker_client)

        assert manager.db_conn == mock_db_conn
        assert manager.broker_client == mock_broker_client
        assert manager.auto_fix is True
        assert manager.dry_run is False
        assert manager.circuit_breaker is not None

    def test_init_custom_circuit_breaker(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test initialization with custom circuit breaker."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )
        assert manager.circuit_breaker == mock_circuit_breaker

    def test_run_full_sync_no_positions(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test sync with no positions in either source."""
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchall.return_value = []
        mock_broker_client.get_positions.return_value = []

        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )
        result = manager.run_full_sync()

        assert result.success is True
        assert result.db_positions_count == 0
        assert result.broker_positions_count == 0
        assert result.discrepancies_found == 0

    def test_run_full_sync_circuit_open(self, mock_db_conn, mock_broker_client):
        """Test sync fails when circuit breaker is open."""
        cb = CircuitBreaker(name="test", failure_threshold=2)
        # Force circuit open
        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))
        assert cb.is_open

        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=cb
        )
        result = manager.run_full_sync()

        assert result.success is False
        assert "Circuit breaker open" in result.errors[0]

    def test_compare_positions_in_sync(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test comparison when positions are in sync."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        db_positions = [
            {'position_id': 1, 'symbol': 'AAPL', 'direction': 'LONG', 'shares': 100,
             'entry_price': 150.0, 'status': 'OPEN', 'stop_order_id': None}
        ]
        broker_positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'direction': 'LONG',
             'entry_price': 150.0, 'current_price': 155.0, 'market_value': 15500.0, 'unrealized_pl': 500.0}
        ]

        discrepancies = manager._compare_positions(db_positions, broker_positions)
        assert len(discrepancies) == 0

    def test_compare_positions_missing_in_broker(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test detection of position missing in broker."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        db_positions = [
            {'position_id': 1, 'symbol': 'AAPL', 'direction': 'LONG', 'shares': 100,
             'entry_price': 150.0, 'status': 'OPEN', 'stop_order_id': None}
        ]
        broker_positions = []  # Empty - position not in broker

        discrepancies = manager._compare_positions(db_positions, broker_positions)

        assert len(discrepancies) == 1
        assert discrepancies[0].type == DiscrepancyType.POSITION_MISSING_BROKER
        assert discrepancies[0].symbol == 'AAPL'
        assert discrepancies[0].position_id == 1

    def test_compare_positions_missing_in_db(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test detection of position missing in database."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        db_positions = []  # Empty - not tracked in DB
        broker_positions = [
            {'symbol': 'TSLA', 'quantity': 50, 'direction': 'LONG',
             'entry_price': 200.0, 'current_price': 210.0, 'market_value': 10500.0, 'unrealized_pl': 500.0}
        ]

        discrepancies = manager._compare_positions(db_positions, broker_positions)

        assert len(discrepancies) == 1
        assert discrepancies[0].type == DiscrepancyType.POSITION_MISSING_DB
        assert discrepancies[0].symbol == 'TSLA'

    def test_compare_positions_quantity_mismatch(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test detection of quantity mismatch."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        db_positions = [
            {'position_id': 1, 'symbol': 'AAPL', 'direction': 'LONG', 'shares': 100,
             'entry_price': 150.0, 'status': 'OPEN', 'stop_order_id': None}
        ]
        broker_positions = [
            {'symbol': 'AAPL', 'quantity': 75, 'direction': 'LONG',  # Different quantity
             'entry_price': 150.0, 'current_price': 155.0, 'market_value': 11625.0, 'unrealized_pl': 375.0}
        ]

        discrepancies = manager._compare_positions(db_positions, broker_positions)

        assert len(discrepancies) == 1
        assert discrepancies[0].type == DiscrepancyType.QUANTITY_MISMATCH
        assert discrepancies[0].db_value == 100
        assert discrepancies[0].broker_value == 75

    def test_compare_positions_direction_mismatch(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test detection of direction mismatch."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        db_positions = [
            {'position_id': 1, 'symbol': 'AAPL', 'direction': 'LONG', 'shares': 100,
             'entry_price': 150.0, 'status': 'OPEN', 'stop_order_id': None}
        ]
        broker_positions = [
            {'symbol': 'AAPL', 'quantity': 100, 'direction': 'SHORT',  # Different direction
             'entry_price': 150.0, 'current_price': 145.0, 'market_value': 14500.0, 'unrealized_pl': 500.0}
        ]

        discrepancies = manager._compare_positions(db_positions, broker_positions)

        assert len(discrepancies) == 1
        assert discrepancies[0].type == DiscrepancyType.DIRECTION_MISMATCH
        assert discrepancies[0].db_value == 'LONG'
        assert discrepancies[0].broker_value == 'SHORT'

    def test_get_broker_positions_success(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test successful broker position fetch."""
        mock_broker_client.get_positions.return_value = [
            MockPosition(
                symbol='AAPL',
                quantity='100',
                is_long=True,
                avg_entry_price='150.00',
                current_price='155.00',
                market_value='15500.00',
                unrealized_pl='500.00'
            )
        ]

        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )
        positions = manager._get_broker_positions()

        assert len(positions) == 1
        assert positions[0]['symbol'] == 'AAPL'
        assert positions[0]['quantity'] == 100
        assert positions[0]['direction'] == 'LONG'

    def test_get_broker_positions_circuit_open(self, mock_db_conn, mock_broker_client):
        """Test broker fetch fails when circuit is open."""
        cb = CircuitBreaker(name="test", failure_threshold=2)
        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))

        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=cb
        )

        with pytest.raises(CircuitOpenError):
            manager._get_broker_positions()

    def test_get_broker_positions_retry_on_failure(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test retry logic on transient failure."""
        call_count = 0

        def failing_get_positions():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Connection timeout")
            return [MockPosition(
                symbol='AAPL', quantity='100', is_long=True,
                avg_entry_price='150.00', current_price='155.00',
                market_value='15500.00', unrealized_pl='500.00'
            )]

        mock_broker_client.get_positions = failing_get_positions

        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        # Should succeed after retries
        positions = manager._get_broker_positions(max_retries=3)
        assert len(positions) == 1
        assert call_count == 3

    def test_fix_discrepancies_mark_closed(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test fixing position missing in broker by marking closed."""
        cursor = mock_db_conn.cursor.return_value

        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            auto_fix=True,
            dry_run=False,
            circuit_breaker=mock_circuit_breaker
        )

        discrepancies = [
            Discrepancy(
                type=DiscrepancyType.POSITION_MISSING_BROKER,
                symbol='AAPL',
                message='Not in broker',
                position_id=1
            )
        ]

        fixed_count = manager._fix_discrepancies(discrepancies)

        assert fixed_count == 1
        assert discrepancies[0].action_taken == SyncAction.MARK_CLOSED
        assert discrepancies[0].action_success is True
        cursor.execute.assert_called()
        mock_db_conn.commit.assert_called()

    def test_fix_discrepancies_update_quantity(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test fixing quantity mismatch."""
        cursor = mock_db_conn.cursor.return_value

        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            auto_fix=True,
            dry_run=False,
            circuit_breaker=mock_circuit_breaker
        )

        discrepancies = [
            Discrepancy(
                type=DiscrepancyType.QUANTITY_MISMATCH,
                symbol='AAPL',
                message='Quantity mismatch',
                db_value=100,
                broker_value=75,
                position_id=1
            )
        ]

        fixed_count = manager._fix_discrepancies(discrepancies)

        assert fixed_count == 1
        assert discrepancies[0].action_taken == SyncAction.UPDATE_QUANTITY
        assert discrepancies[0].action_success is True

    def test_fix_discrepancies_no_action_missing_db(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test no action for position missing in DB."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            auto_fix=True,
            dry_run=False,
            circuit_breaker=mock_circuit_breaker
        )

        discrepancies = [
            Discrepancy(
                type=DiscrepancyType.POSITION_MISSING_DB,
                symbol='TSLA',
                message='Not tracked in DB'
            )
        ]

        fixed_count = manager._fix_discrepancies(discrepancies)

        assert fixed_count == 0
        assert discrepancies[0].action_taken == SyncAction.NO_ACTION
        assert discrepancies[0].action_success is False
        assert 'Manual intervention' in discrepancies[0].error

    def test_generate_alert_message_no_discrepancies(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test no alert when no discrepancies."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        result = SyncResult(success=True, discrepancies_found=0)
        message = manager.generate_alert_message(result)

        assert message is None

    def test_generate_alert_message_with_discrepancies(self, mock_db_conn, mock_broker_client, mock_circuit_breaker):
        """Test alert message generation with discrepancies."""
        manager = PositionSyncManager(
            mock_db_conn,
            mock_broker_client,
            circuit_breaker=mock_circuit_breaker
        )

        result = SyncResult(
            success=True,
            db_positions_count=5,
            broker_positions_count=3,
            discrepancies_found=2,
            discrepancies_fixed=1,
            discrepancies=[
                Discrepancy(
                    type=DiscrepancyType.POSITION_MISSING_BROKER,
                    symbol='AAPL',
                    message='Not in broker',
                    action_taken=SyncAction.MARK_CLOSED,
                    action_success=True
                ),
                Discrepancy(
                    type=DiscrepancyType.QUANTITY_MISMATCH,
                    symbol='TSLA',
                    message='Qty mismatch',
                    action_taken=SyncAction.UPDATE_QUANTITY,
                    action_success=True
                )
            ]
        )

        message = manager.generate_alert_message(result)

        assert message is not None
        assert 'Position Sync Alert' in message
        assert 'DB Positions: 5' in message
        assert 'Broker Positions: 3' in message
        assert 'Discrepancies Found: 2' in message
        assert 'AAPL' in message
        assert 'TSLA' in message


class TestRunPositionSync:
    """Tests for run_position_sync convenience function."""

    def test_basic_run(self, mock_db_conn, mock_broker_client):
        """Test basic sync run."""
        reset_broker_circuit_breaker()
        cursor = mock_db_conn.cursor.return_value
        cursor.fetchall.return_value = []
        mock_broker_client.get_positions.return_value = []

        result = run_position_sync(
            mock_db_conn,
            mock_broker_client,
            auto_fix=True,
            dry_run=False,
            send_alert=False
        )

        assert result['success'] is True
        assert result['discrepancies_found'] == 0
        assert result['alert_sent'] is False


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
