"""Error Handling Module for Gap Trading Strategy.

Provides:
- Custom exception hierarchy for error categorization
- Retry decorator with exponential backoff
- Circuit breaker pattern for preventing cascading failures
- Error logging and alerting utilities
"""

import functools
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable, Type, Tuple, List, Any, Dict
from threading import Lock

logger = logging.getLogger(__name__)


# =============================================================================
# Error Categories
# =============================================================================

class ErrorCategory(str, Enum):
    """Categorization of errors for handling decisions."""
    AUTH_ERROR = 'auth_error'           # API key issues, authentication failures
    NETWORK_ERROR = 'network_error'     # Timeout, connection refused, DNS errors
    RATE_LIMIT = 'rate_limit'           # API rate limiting
    BUSINESS_ERROR = 'business_error'   # Insufficient funds, invalid order, market closed
    DATA_ERROR = 'data_error'           # Invalid data, parsing errors
    UNKNOWN_ERROR = 'unknown_error'     # Catch-all for unexpected errors


# =============================================================================
# Custom Exceptions
# =============================================================================

class GapTradingError(Exception):
    """Base exception for all gap trading errors."""

    def __init__(self, message: str, category: ErrorCategory = ErrorCategory.UNKNOWN_ERROR,
                 retryable: bool = False, details: Optional[Dict] = None):
        super().__init__(message)
        self.category = category
        self.retryable = retryable
        self.details = details or {}
        self.timestamp = datetime.now()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging/storage."""
        return {
            'error_type': self.__class__.__name__,
            'message': str(self),
            'category': self.category.value,
            'retryable': self.retryable,
            'details': self.details,
            'timestamp': self.timestamp.isoformat()
        }


class AuthenticationError(GapTradingError):
    """Authentication or authorization failure."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message,
            category=ErrorCategory.AUTH_ERROR,
            retryable=False,  # Auth errors should not be retried
            details=details
        )


class NetworkError(GapTradingError):
    """Network-related errors (timeout, connection refused, etc.)."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK_ERROR,
            retryable=True,  # Network errors are usually transient
            details=details
        )


class RateLimitError(GapTradingError):
    """API rate limit exceeded."""

    def __init__(self, message: str, retry_after: Optional[int] = None,
                 details: Optional[Dict] = None):
        details = details or {}
        details['retry_after'] = retry_after
        super().__init__(
            message,
            category=ErrorCategory.RATE_LIMIT,
            retryable=True,
            details=details
        )
        self.retry_after = retry_after


class BusinessError(GapTradingError):
    """Business logic errors (insufficient funds, invalid order, etc.)."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message,
            category=ErrorCategory.BUSINESS_ERROR,
            retryable=False,  # Business errors require intervention
            details=details
        )


class InsufficientFundsError(BusinessError):
    """Insufficient buying power for trade."""

    def __init__(self, required: float, available: float,
                 details: Optional[Dict] = None):
        details = details or {}
        details.update({'required': required, 'available': available})
        super().__init__(
            f"Insufficient funds: required ${required:.2f}, available ${available:.2f}",
            details=details
        )


class MarketClosedError(BusinessError):
    """Market is closed."""

    def __init__(self, message: str = "Market is currently closed",
                 details: Optional[Dict] = None):
        super().__init__(message, details=details)


class DataError(GapTradingError):
    """Data-related errors (invalid data, parsing errors)."""

    def __init__(self, message: str, details: Optional[Dict] = None):
        super().__init__(
            message,
            category=ErrorCategory.DATA_ERROR,
            retryable=False,
            details=details
        )


# =============================================================================
# Error Classification
# =============================================================================

def classify_error(exception: Exception) -> Tuple[ErrorCategory, bool]:
    """Classify an exception into a category and determine if retryable.

    Args:
        exception: The exception to classify

    Returns:
        Tuple of (ErrorCategory, is_retryable)
    """
    error_str = str(exception).lower()
    error_type = type(exception).__name__.lower()

    # Check for authentication errors
    auth_patterns = ['authentication', 'unauthorized', 'forbidden', 'api key',
                     'invalid credentials', '401', '403']
    if any(p in error_str for p in auth_patterns):
        return ErrorCategory.AUTH_ERROR, False

    # Check for rate limiting
    rate_patterns = ['rate limit', 'too many requests', '429', 'throttl']
    if any(p in error_str for p in rate_patterns):
        return ErrorCategory.RATE_LIMIT, True

    # Check for network errors
    network_patterns = ['timeout', 'connection', 'network', 'dns', 'unreachable',
                       'connection refused', 'connection reset', 'eof', 'broken pipe']
    network_types = ['timeout', 'connection', 'socket', 'http']
    if any(p in error_str for p in network_patterns) or any(t in error_type for t in network_types):
        return ErrorCategory.NETWORK_ERROR, True

    # Check for business errors
    business_patterns = ['insufficient', 'invalid order', 'rejected', 'market closed',
                        'position not found', 'symbol not found']
    if any(p in error_str for p in business_patterns):
        return ErrorCategory.BUSINESS_ERROR, False

    # Check for data errors
    data_patterns = ['parse', 'invalid data', 'json', 'decode', 'encoding']
    if any(p in error_str for p in data_patterns):
        return ErrorCategory.DATA_ERROR, False

    # Default to unknown
    return ErrorCategory.UNKNOWN_ERROR, False


def wrap_exception(exception: Exception) -> GapTradingError:
    """Wrap a generic exception in an appropriate GapTradingError subclass.

    Args:
        exception: The exception to wrap

    Returns:
        Appropriate GapTradingError subclass
    """
    if isinstance(exception, GapTradingError):
        return exception

    category, retryable = classify_error(exception)
    details = {
        'original_type': type(exception).__name__,
        'original_message': str(exception)
    }

    if category == ErrorCategory.AUTH_ERROR:
        return AuthenticationError(str(exception), details=details)
    elif category == ErrorCategory.NETWORK_ERROR:
        return NetworkError(str(exception), details=details)
    elif category == ErrorCategory.RATE_LIMIT:
        return RateLimitError(str(exception), details=details)
    elif category == ErrorCategory.BUSINESS_ERROR:
        return BusinessError(str(exception), details=details)
    elif category == ErrorCategory.DATA_ERROR:
        return DataError(str(exception), details=details)
    else:
        return GapTradingError(str(exception), category=category,
                               retryable=retryable, details=details)


# =============================================================================
# Retry Decorator
# =============================================================================

@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    max_attempts: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0
    jitter: float = 0.1  # Random jitter factor (0-1)
    retryable_exceptions: Tuple[Type[Exception], ...] = (NetworkError, RateLimitError)
    on_retry: Optional[Callable[[Exception, int, float], None]] = None


def retry_with_backoff(
    max_attempts: int = 3,
    backoff_factor: float = 2.0,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    retryable_exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable[[Exception, int, float], None]] = None
):
    """Decorator for retrying functions with exponential backoff.

    Args:
        max_attempts: Maximum number of attempts (including initial)
        backoff_factor: Multiplier for delay between attempts
        initial_delay: Initial delay in seconds
        max_delay: Maximum delay between attempts
        retryable_exceptions: Tuple of exception types to retry
        on_retry: Optional callback(exception, attempt, delay) on each retry

    Example:
        @retry_with_backoff(max_attempts=3, backoff_factor=2.0)
        def call_api():
            return api.get_data()
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = initial_delay

            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)

                except retryable_exceptions as e:
                    last_exception = e

                    # Check if this is a GapTradingError and if it's retryable
                    if isinstance(e, GapTradingError) and not e.retryable:
                        logger.warning(f"Non-retryable error on attempt {attempt}: {e}")
                        raise

                    # Handle rate limit with specific delay
                    if isinstance(e, RateLimitError) and e.retry_after:
                        delay = e.retry_after

                    if attempt < max_attempts:
                        # Add some jitter to prevent thundering herd
                        import random
                        jittered_delay = delay * (1 + random.uniform(-0.1, 0.1))
                        jittered_delay = min(jittered_delay, max_delay)

                        logger.warning(
                            f"Attempt {attempt}/{max_attempts} failed: {e}. "
                            f"Retrying in {jittered_delay:.1f}s..."
                        )

                        if on_retry:
                            on_retry(e, attempt, jittered_delay)

                        time.sleep(jittered_delay)
                        delay = min(delay * backoff_factor, max_delay)
                    else:
                        logger.error(
                            f"All {max_attempts} attempts failed. Last error: {e}"
                        )

                except Exception as e:
                    # Non-retryable exception - raise immediately
                    logger.error(f"Non-retryable exception on attempt {attempt}: {e}")
                    raise

            # All retries exhausted
            if last_exception:
                raise last_exception

        return wrapper
    return decorator


# =============================================================================
# Circuit Breaker
# =============================================================================

class CircuitState(str, Enum):
    """Circuit breaker states."""
    CLOSED = 'closed'      # Normal operation
    OPEN = 'open'          # Failing, rejecting calls
    HALF_OPEN = 'half_open'  # Testing if service recovered


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker."""
    failure_threshold: int = 5       # Failures before opening
    success_threshold: int = 2       # Successes to close from half-open
    timeout: float = 300.0           # Seconds before trying half-open (5 min)
    excluded_exceptions: Tuple[Type[Exception], ...] = ()  # Don't count these as failures


class CircuitBreaker:
    """Circuit breaker pattern implementation.

    Prevents cascading failures by stopping calls to a failing service
    and allowing it time to recover.

    States:
    - CLOSED: Normal operation, failures are counted
    - OPEN: Service is failing, calls are rejected immediately
    - HALF_OPEN: Testing if service has recovered

    Example:
        breaker = CircuitBreaker(failure_threshold=5, timeout=300)

        @breaker
        def call_broker():
            return broker.get_positions()
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        timeout: float = 300.0,
        excluded_exceptions: Tuple[Type[Exception], ...] = (),
        name: str = "default",
        on_state_change: Optional[Callable[[CircuitState, CircuitState], None]] = None
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.excluded_exceptions = excluded_exceptions
        self.name = name
        self.on_state_change = on_state_change

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._last_failure_time: Optional[datetime] = None
        self._lock = Lock()

    @property
    def state(self) -> CircuitState:
        """Get current circuit state, handling timeout transitions."""
        with self._lock:
            if self._state == CircuitState.OPEN:
                if self._should_attempt_reset():
                    self._transition_to(CircuitState.HALF_OPEN)
            return self._state

    @property
    def is_closed(self) -> bool:
        return self.state == CircuitState.CLOSED

    @property
    def is_open(self) -> bool:
        return self.state == CircuitState.OPEN

    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to try half-open."""
        if self._last_failure_time is None:
            return False
        elapsed = (datetime.now() - self._last_failure_time).total_seconds()
        return elapsed >= self.timeout

    def _transition_to(self, new_state: CircuitState):
        """Transition to a new state."""
        old_state = self._state
        self._state = new_state

        if new_state == CircuitState.CLOSED:
            self._failure_count = 0
            self._success_count = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._success_count = 0

        if old_state != new_state:
            logger.info(f"Circuit breaker '{self.name}': {old_state.value} -> {new_state.value}")
            if self.on_state_change:
                self.on_state_change(old_state, new_state)

    def record_success(self):
        """Record a successful call."""
        with self._lock:
            if self._state == CircuitState.HALF_OPEN:
                self._success_count += 1
                if self._success_count >= self.success_threshold:
                    self._transition_to(CircuitState.CLOSED)
            elif self._state == CircuitState.CLOSED:
                # Reset failure count on success
                self._failure_count = 0

    def record_failure(self, exception: Exception):
        """Record a failed call."""
        # Don't count excluded exceptions
        if isinstance(exception, self.excluded_exceptions):
            return

        with self._lock:
            self._last_failure_time = datetime.now()

            if self._state == CircuitState.HALF_OPEN:
                # Any failure in half-open goes back to open
                self._transition_to(CircuitState.OPEN)
            elif self._state == CircuitState.CLOSED:
                self._failure_count += 1
                if self._failure_count >= self.failure_threshold:
                    self._transition_to(CircuitState.OPEN)

    def __call__(self, func: Callable) -> Callable:
        """Use as decorator."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker."""
        state = self.state

        if state == CircuitState.OPEN:
            raise CircuitOpenError(
                f"Circuit breaker '{self.name}' is OPEN. "
                f"Service calls are blocked for {self.timeout}s after "
                f"{self.failure_threshold} consecutive failures."
            )

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result

        except Exception as e:
            self.record_failure(e)
            raise

    def reset(self):
        """Manually reset the circuit breaker to closed state."""
        with self._lock:
            self._transition_to(CircuitState.CLOSED)

    def get_status(self) -> Dict[str, Any]:
        """Get current status for monitoring."""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self._failure_count,
            'success_count': self._success_count,
            'last_failure_time': self._last_failure_time.isoformat() if self._last_failure_time else None,
            'failure_threshold': self.failure_threshold,
            'timeout': self.timeout
        }


class CircuitOpenError(GapTradingError):
    """Raised when circuit breaker is open."""

    def __init__(self, message: str):
        super().__init__(
            message,
            category=ErrorCategory.NETWORK_ERROR,
            retryable=True  # Can retry after timeout
        )


# =============================================================================
# Combined Decorator
# =============================================================================

def with_error_handling(
    max_retries: int = 3,
    backoff_factor: float = 2.0,
    circuit_breaker: Optional[CircuitBreaker] = None,
    on_error: Optional[Callable[[Exception], None]] = None
):
    """Combined decorator for retry with backoff and optional circuit breaker.

    Args:
        max_retries: Maximum retry attempts
        backoff_factor: Exponential backoff multiplier
        circuit_breaker: Optional CircuitBreaker instance
        on_error: Optional callback for error logging/alerting

    Example:
        breaker = CircuitBreaker(name="broker_api")

        @with_error_handling(max_retries=3, circuit_breaker=breaker)
        def place_order(symbol, qty):
            return broker.place_market_order(symbol, qty)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # If circuit is open, fail fast
            if circuit_breaker and circuit_breaker.is_open:
                raise CircuitOpenError(
                    f"Circuit breaker '{circuit_breaker.name}' is open"
                )

            last_exception = None
            delay = 1.0

            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # Record success if using circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result

                except Exception as e:
                    last_exception = e
                    wrapped = wrap_exception(e)

                    # Record failure if using circuit breaker
                    if circuit_breaker:
                        circuit_breaker.record_failure(e)

                    # Call error handler if provided
                    if on_error:
                        on_error(wrapped)

                    # Check if retryable
                    if not wrapped.retryable:
                        logger.error(f"Non-retryable error: {wrapped}")
                        raise wrapped

                    # Check if circuit is now open
                    if circuit_breaker and circuit_breaker.is_open:
                        logger.warning(f"Circuit breaker opened after failure: {e}")
                        raise CircuitOpenError(
                            f"Circuit breaker '{circuit_breaker.name}' opened"
                        )

                    if attempt < max_retries:
                        logger.warning(
                            f"Attempt {attempt}/{max_retries} failed: {e}. "
                            f"Retrying in {delay:.1f}s..."
                        )
                        time.sleep(delay)
                        delay = min(delay * backoff_factor, 60.0)
                    else:
                        logger.error(f"All {max_retries} attempts failed: {e}")

            if last_exception:
                raise wrap_exception(last_exception)

        return wrapper
    return decorator


# =============================================================================
# Global Circuit Breakers (Singletons for common services)
# =============================================================================

# Circuit breaker for broker API calls
_broker_circuit_breaker: Optional[CircuitBreaker] = None

def get_broker_circuit_breaker() -> CircuitBreaker:
    """Get or create the broker API circuit breaker."""
    global _broker_circuit_breaker
    if _broker_circuit_breaker is None:
        _broker_circuit_breaker = CircuitBreaker(
            name="broker_api",
            failure_threshold=5,
            success_threshold=2,
            timeout=300.0  # 5 minutes
        )
    return _broker_circuit_breaker


def reset_broker_circuit_breaker():
    """Reset the broker circuit breaker (for testing/recovery)."""
    global _broker_circuit_breaker
    if _broker_circuit_breaker:
        _broker_circuit_breaker.reset()
