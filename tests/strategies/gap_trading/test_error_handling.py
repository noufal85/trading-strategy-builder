"""Unit tests for Gap Trading error handling module.

Tests cover:
- Error classification
- Custom exception hierarchy
- Retry decorator with exponential backoff
- Circuit breaker pattern
"""

import pytest
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime, timedelta

from strategy_builder.strategies.gap_trading.error_handling import (
    ErrorCategory,
    GapTradingError,
    AuthenticationError,
    NetworkError,
    RateLimitError,
    BusinessError,
    InsufficientFundsError,
    MarketClosedError,
    DataError,
    CircuitOpenError,
    classify_error,
    wrap_exception,
    retry_with_backoff,
    CircuitBreaker,
    CircuitState,
    get_broker_circuit_breaker,
    reset_broker_circuit_breaker,
)


class TestErrorCategory:
    """Tests for ErrorCategory enum."""

    def test_all_categories_exist(self):
        """Verify all expected error categories exist."""
        expected = ['AUTH_ERROR', 'NETWORK_ERROR', 'RATE_LIMIT',
                    'BUSINESS_ERROR', 'DATA_ERROR', 'UNKNOWN_ERROR']
        actual = [e.name for e in ErrorCategory]
        assert set(expected) == set(actual)

    def test_category_values(self):
        """Verify category string values."""
        assert ErrorCategory.AUTH_ERROR.value == 'auth_error'
        assert ErrorCategory.NETWORK_ERROR.value == 'network_error'
        assert ErrorCategory.RATE_LIMIT.value == 'rate_limit'


class TestGapTradingError:
    """Tests for GapTradingError base exception."""

    def test_basic_creation(self):
        """Test basic exception creation."""
        error = GapTradingError("Test error")
        assert str(error) == "Test error"
        assert error.category == ErrorCategory.UNKNOWN_ERROR
        assert error.retryable is False
        assert error.details == {}
        assert isinstance(error.timestamp, datetime)

    def test_with_category_and_retryable(self):
        """Test exception with category and retryable flag."""
        error = GapTradingError(
            "Network timeout",
            category=ErrorCategory.NETWORK_ERROR,
            retryable=True,
            details={'host': 'api.example.com'}
        )
        assert error.category == ErrorCategory.NETWORK_ERROR
        assert error.retryable is True
        assert error.details['host'] == 'api.example.com'

    def test_to_dict(self):
        """Test conversion to dictionary."""
        error = GapTradingError("Test", category=ErrorCategory.AUTH_ERROR)
        d = error.to_dict()

        assert d['error_type'] == 'GapTradingError'
        assert d['message'] == 'Test'
        assert d['category'] == 'auth_error'
        assert d['retryable'] is False
        assert 'timestamp' in d


class TestSpecificExceptions:
    """Tests for specific exception subclasses."""

    def test_authentication_error(self):
        """Test AuthenticationError defaults."""
        error = AuthenticationError("Invalid API key")
        assert error.category == ErrorCategory.AUTH_ERROR
        assert error.retryable is False

    def test_network_error(self):
        """Test NetworkError defaults."""
        error = NetworkError("Connection timeout")
        assert error.category == ErrorCategory.NETWORK_ERROR
        assert error.retryable is True

    def test_rate_limit_error(self):
        """Test RateLimitError with retry_after."""
        error = RateLimitError("Too many requests", retry_after=60)
        assert error.category == ErrorCategory.RATE_LIMIT
        assert error.retryable is True
        assert error.retry_after == 60
        assert error.details['retry_after'] == 60

    def test_business_error(self):
        """Test BusinessError defaults."""
        error = BusinessError("Order rejected")
        assert error.category == ErrorCategory.BUSINESS_ERROR
        assert error.retryable is False

    def test_insufficient_funds_error(self):
        """Test InsufficientFundsError formatting."""
        error = InsufficientFundsError(required=1000.0, available=500.0)
        assert "1000.00" in str(error)
        assert "500.00" in str(error)
        assert error.details['required'] == 1000.0
        assert error.details['available'] == 500.0

    def test_market_closed_error(self):
        """Test MarketClosedError default message."""
        error = MarketClosedError()
        assert "closed" in str(error).lower()
        assert error.category == ErrorCategory.BUSINESS_ERROR

    def test_data_error(self):
        """Test DataError defaults."""
        error = DataError("Invalid JSON response")
        assert error.category == ErrorCategory.DATA_ERROR
        assert error.retryable is False

    def test_circuit_open_error(self):
        """Test CircuitOpenError defaults."""
        error = CircuitOpenError("Circuit breaker is open")
        assert error.category == ErrorCategory.NETWORK_ERROR
        assert error.retryable is True  # Can retry after timeout


class TestClassifyError:
    """Tests for error classification function."""

    def test_classify_auth_errors(self):
        """Test classification of authentication errors."""
        test_cases = [
            "Authentication failed",
            "Unauthorized access",
            "Invalid API key",
            "403 Forbidden",
        ]
        for msg in test_cases:
            category, retryable = classify_error(Exception(msg))
            assert category == ErrorCategory.AUTH_ERROR, f"Failed for: {msg}"
            assert retryable is False

    def test_classify_network_errors(self):
        """Test classification of network errors."""
        test_cases = [
            "Connection timeout",
            "Network unreachable",
            "Connection refused",
            "DNS lookup failed",
        ]
        for msg in test_cases:
            category, retryable = classify_error(Exception(msg))
            assert category == ErrorCategory.NETWORK_ERROR, f"Failed for: {msg}"
            assert retryable is True

    def test_classify_rate_limit_errors(self):
        """Test classification of rate limit errors."""
        test_cases = [
            "Rate limit exceeded",
            "Too many requests",
            "429 Too Many Requests",
            "Request throttled",
        ]
        for msg in test_cases:
            category, retryable = classify_error(Exception(msg))
            assert category == ErrorCategory.RATE_LIMIT, f"Failed for: {msg}"
            assert retryable is True

    def test_classify_business_errors(self):
        """Test classification of business errors."""
        test_cases = [
            "Insufficient funds",
            "Order rejected",
            "Market closed",
        ]
        for msg in test_cases:
            category, retryable = classify_error(Exception(msg))
            assert category == ErrorCategory.BUSINESS_ERROR, f"Failed for: {msg}"
            assert retryable is False

    def test_classify_unknown_errors(self):
        """Test classification of unknown errors."""
        category, retryable = classify_error(Exception("Something unexpected"))
        assert category == ErrorCategory.UNKNOWN_ERROR
        assert retryable is False


class TestWrapException:
    """Tests for wrap_exception function."""

    def test_wrap_already_wrapped(self):
        """Test that already wrapped exceptions are returned as-is."""
        original = NetworkError("Already wrapped")
        wrapped = wrap_exception(original)
        assert wrapped is original

    def test_wrap_generic_auth_error(self):
        """Test wrapping generic auth exception."""
        original = Exception("Authentication failed")
        wrapped = wrap_exception(original)
        assert isinstance(wrapped, AuthenticationError)
        assert wrapped.details['original_type'] == 'Exception'

    def test_wrap_generic_network_error(self):
        """Test wrapping generic network exception."""
        original = Exception("Connection timeout")
        wrapped = wrap_exception(original)
        assert isinstance(wrapped, NetworkError)
        assert wrapped.retryable is True

    def test_wrap_preserves_message(self):
        """Test that original message is preserved."""
        original = Exception("Original message here")
        wrapped = wrap_exception(original)
        assert "Original message here" in str(wrapped)


class TestRetryDecorator:
    """Tests for retry_with_backoff decorator."""

    def test_success_no_retry(self):
        """Test successful call doesn't retry."""
        call_count = 0

        @retry_with_backoff(max_attempts=3)
        def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = successful_func()
        assert result == "success"
        assert call_count == 1

    def test_retry_on_exception(self):
        """Test retry on retryable exception."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def failing_then_success():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise NetworkError("Temporary failure")
            return "success"

        result = failing_then_success()
        assert result == "success"
        assert call_count == 3

    def test_max_retries_exhausted(self):
        """Test that max retries are respected."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise NetworkError("Always fails")

        with pytest.raises(NetworkError):
            always_fails()

        assert call_count == 3

    def test_non_retryable_no_retry(self):
        """Test non-retryable exceptions don't retry."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, initial_delay=0.01)
        def auth_failure():
            nonlocal call_count
            call_count += 1
            raise AuthenticationError("Bad credentials")

        with pytest.raises(AuthenticationError):
            auth_failure()

        assert call_count == 1  # No retries for auth errors

    def test_on_retry_callback(self):
        """Test on_retry callback is called."""
        retry_info = []

        def on_retry(exc, attempt, delay):
            retry_info.append((type(exc).__name__, attempt, delay))

        @retry_with_backoff(max_attempts=3, initial_delay=0.01, on_retry=on_retry)
        def fails_twice():
            if len(retry_info) < 2:
                raise NetworkError("Temporary")
            return "success"

        result = fails_twice()
        assert result == "success"
        assert len(retry_info) == 2
        assert retry_info[0][1] == 1  # First attempt
        assert retry_info[1][1] == 2  # Second attempt


class TestCircuitBreaker:
    """Tests for CircuitBreaker class."""

    def test_initial_state_closed(self):
        """Test circuit breaker starts in CLOSED state."""
        cb = CircuitBreaker(name="test")
        assert cb.state == CircuitState.CLOSED
        assert cb.is_closed is True
        assert cb.is_open is False

    def test_opens_after_failures(self):
        """Test circuit opens after threshold failures."""
        cb = CircuitBreaker(failure_threshold=3, name="test")

        for i in range(3):
            cb.record_failure(Exception(f"Failure {i}"))

        assert cb.state == CircuitState.OPEN
        assert cb.is_open is True

    def test_success_resets_failure_count(self):
        """Test success resets failure count."""
        cb = CircuitBreaker(failure_threshold=3, name="test")

        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))
        cb.record_success()  # Should reset count
        cb.record_failure(Exception("Failure 3"))

        # Should still be closed (only 1 failure after reset)
        assert cb.is_closed is True

    def test_half_open_after_timeout(self):
        """Test circuit goes to HALF_OPEN after timeout."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.1, name="test")

        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))
        assert cb.is_open is True

        # Wait for timeout
        time.sleep(0.15)

        # Accessing state should transition to HALF_OPEN
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_to_closed_on_success(self):
        """Test HALF_OPEN closes after success threshold."""
        cb = CircuitBreaker(
            failure_threshold=2,
            success_threshold=2,
            timeout=0.05,
            name="test"
        )

        # Open the circuit
        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))

        # Wait for timeout
        time.sleep(0.1)

        # Should be HALF_OPEN now
        assert cb.state == CircuitState.HALF_OPEN

        # Record successes
        cb.record_success()
        assert cb.state == CircuitState.HALF_OPEN  # Still half-open
        cb.record_success()
        assert cb.state == CircuitState.CLOSED  # Now closed

    def test_half_open_to_open_on_failure(self):
        """Test HALF_OPEN goes back to OPEN on failure."""
        cb = CircuitBreaker(failure_threshold=2, timeout=0.05, name="test")

        # Open the circuit
        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))

        # Wait for timeout
        time.sleep(0.1)

        # Should be HALF_OPEN now
        assert cb.state == CircuitState.HALF_OPEN

        # Any failure should go back to OPEN
        cb.record_failure(Exception("Another failure"))
        assert cb.state == CircuitState.OPEN

    def test_decorator_usage(self):
        """Test circuit breaker as decorator."""
        cb = CircuitBreaker(failure_threshold=2, name="test")
        call_count = 0

        @cb
        def protected_func():
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise Exception("Failing")
            return "success"

        # First two calls fail
        with pytest.raises(Exception):
            protected_func()
        with pytest.raises(Exception):
            protected_func()

        # Circuit should be open now
        assert cb.is_open

        # Third call should raise CircuitOpenError
        with pytest.raises(CircuitOpenError):
            protected_func()

    def test_manual_reset(self):
        """Test manual reset of circuit breaker."""
        cb = CircuitBreaker(failure_threshold=2, name="test")

        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))
        assert cb.is_open

        cb.reset()
        assert cb.is_closed

    def test_get_status(self):
        """Test status dictionary."""
        cb = CircuitBreaker(
            failure_threshold=5,
            timeout=300,
            name="test_breaker"
        )

        status = cb.get_status()
        assert status['name'] == 'test_breaker'
        assert status['state'] == 'closed'
        assert status['failure_count'] == 0
        assert status['failure_threshold'] == 5
        assert status['timeout'] == 300

    def test_state_change_callback(self):
        """Test on_state_change callback."""
        state_changes = []

        def on_change(old, new):
            state_changes.append((old, new))

        cb = CircuitBreaker(
            failure_threshold=2,
            name="test",
            on_state_change=on_change
        )

        cb.record_failure(Exception("Failure 1"))
        cb.record_failure(Exception("Failure 2"))

        assert len(state_changes) == 1
        assert state_changes[0] == (CircuitState.CLOSED, CircuitState.OPEN)


class TestGlobalCircuitBreaker:
    """Tests for global circuit breaker singleton."""

    def test_get_broker_circuit_breaker_singleton(self):
        """Test broker circuit breaker is a singleton."""
        reset_broker_circuit_breaker()

        cb1 = get_broker_circuit_breaker()
        cb2 = get_broker_circuit_breaker()

        assert cb1 is cb2
        assert cb1.name == "broker_api"

    def test_reset_broker_circuit_breaker(self):
        """Test resetting broker circuit breaker."""
        cb = get_broker_circuit_breaker()

        # Force it open
        for i in range(5):
            cb.record_failure(Exception(f"Failure {i}"))

        assert cb.is_open

        # Reset should close it
        reset_broker_circuit_breaker()
        assert cb.is_closed


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
