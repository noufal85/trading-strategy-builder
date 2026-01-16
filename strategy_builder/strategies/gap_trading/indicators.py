"""Technical Indicator Calculations for Gap Trading.

Provides RSI and ADX calculations for signal quality assessment
and priority ranking.

Created Date: 2026-01-05
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class IndicatorData:
    """Container for technical indicator values."""
    symbol: str
    rsi_14: Optional[float] = None
    adx_14: Optional[float] = None
    plus_di: Optional[float] = None  # +DI component
    minus_di: Optional[float] = None  # -DI component
    rsi_signal: str = "neutral"  # bullish, bearish, overbought, oversold
    adx_signal: str = "moderate"  # strong, moderate, weak
    error: Optional[str] = None


def calculate_rsi(prices: List[float], period: int = 14) -> Optional[float]:
    """Calculate Relative Strength Index (RSI).

    RSI = 100 - (100 / (1 + RS))
    where RS = Average Gain / Average Loss over period

    Args:
        prices: List of closing prices (oldest to newest)
        period: RSI period (default 14)

    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        logger.warning(f"Insufficient data for RSI: need {period + 1} prices, got {len(prices)}")
        return None

    # Convert to numpy array
    prices_arr = np.array(prices, dtype=float)

    # Calculate price changes
    deltas = np.diff(prices_arr)

    # Separate gains and losses
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    # Use simple moving average for initial calculation
    # Then use exponential moving average (Wilder's smoothing)
    if len(gains) < period:
        return None

    # Initial averages
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # Apply Wilder's smoothing for remaining periods
    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

    # Calculate RS and RSI
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return round(rsi, 2)


def calculate_adx(
    high: List[float],
    low: List[float],
    close: List[float],
    period: int = 14
) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate Average Directional Index (ADX) with +DI and -DI.

    ADX measures trend strength (not direction):
    - ADX > 25: Strong trend
    - ADX 20-25: Moderate trend
    - ADX < 20: Weak/no trend

    Args:
        high: List of high prices (oldest to newest)
        low: List of low prices (oldest to newest)
        close: List of closing prices (oldest to newest)
        period: ADX period (default 14)

    Returns:
        Tuple of (ADX, +DI, -DI) or (None, None, None) if insufficient data
    """
    min_length = period * 2 + 1  # Need enough data for smoothing

    if len(high) < min_length or len(low) < min_length or len(close) < min_length:
        logger.warning(f"Insufficient data for ADX: need {min_length} bars, got {len(high)}")
        return None, None, None

    # Convert to numpy arrays
    high_arr = np.array(high, dtype=float)
    low_arr = np.array(low, dtype=float)
    close_arr = np.array(close, dtype=float)

    # Calculate True Range (TR)
    tr1 = high_arr[1:] - low_arr[1:]
    tr2 = np.abs(high_arr[1:] - close_arr[:-1])
    tr3 = np.abs(low_arr[1:] - close_arr[:-1])
    true_range = np.maximum(np.maximum(tr1, tr2), tr3)

    # Calculate Directional Movement (+DM and -DM)
    up_move = high_arr[1:] - high_arr[:-1]
    down_move = low_arr[:-1] - low_arr[1:]

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Wilder's smoothing function
    def wilders_smooth(data: np.ndarray, period: int) -> np.ndarray:
        """Apply Wilder's exponential smoothing."""
        smoothed = np.zeros(len(data))
        smoothed[period - 1] = np.sum(data[:period])
        for i in range(period, len(data)):
            smoothed[i] = smoothed[i - 1] - (smoothed[i - 1] / period) + data[i]
        return smoothed

    # Smooth TR, +DM, -DM
    atr = wilders_smooth(true_range, period)
    smooth_plus_dm = wilders_smooth(plus_dm, period)
    smooth_minus_dm = wilders_smooth(minus_dm, period)

    # Calculate +DI and -DI
    plus_di = np.zeros(len(atr))
    minus_di = np.zeros(len(atr))

    # Avoid division by zero
    valid_atr = atr > 0
    plus_di[valid_atr] = 100 * smooth_plus_dm[valid_atr] / atr[valid_atr]
    minus_di[valid_atr] = 100 * smooth_minus_dm[valid_atr] / atr[valid_atr]

    # Calculate DX (Directional Index)
    di_sum = plus_di + minus_di
    di_diff = np.abs(plus_di - minus_di)

    dx = np.zeros(len(di_sum))
    valid_di = di_sum > 0
    dx[valid_di] = 100 * di_diff[valid_di] / di_sum[valid_di]

    # Calculate ADX (smoothed DX)
    # Start ADX calculation after we have enough DX values
    start_idx = period * 2 - 1
    if start_idx >= len(dx):
        return None, None, None

    # Initial ADX is simple average of first 'period' DX values
    adx = np.zeros(len(dx))
    adx[start_idx] = np.mean(dx[start_idx - period + 1:start_idx + 1])

    # Smooth subsequent ADX values
    for i in range(start_idx + 1, len(dx)):
        adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period

    # Return latest values
    latest_adx = round(adx[-1], 2) if adx[-1] > 0 else None
    latest_plus_di = round(plus_di[-1], 2) if plus_di[-1] > 0 else None
    latest_minus_di = round(minus_di[-1], 2) if minus_di[-1] > 0 else None

    return latest_adx, latest_plus_di, latest_minus_di


def get_rsi_signal(rsi: float, gap_direction: str) -> str:
    """Classify RSI condition based on value and gap direction.

    Args:
        rsi: RSI value (0-100)
        gap_direction: 'UP' or 'DOWN'

    Returns:
        Signal classification: bullish, bearish, overbought, oversold, neutral
    """
    if rsi is None:
        return "neutral"

    if gap_direction == "UP":
        # For gap up (buying), we prefer RSI not overbought
        if rsi >= 70:
            return "overbought"  # Warning - may be extended
        elif rsi < 30:
            return "oversold"  # Could be strong bounce
        elif rsi < 50:
            return "bullish"  # Room to run
        else:
            return "neutral"
    else:
        # For gap down (shorting), we prefer RSI not oversold
        if rsi <= 30:
            return "oversold"  # Warning - may bounce
        elif rsi > 70:
            return "overbought"  # Could be strong reversal
        elif rsi > 50:
            return "bearish"  # Room to fall
        else:
            return "neutral"


def get_adx_signal(adx: float) -> str:
    """Classify ADX trend strength.

    Args:
        adx: ADX value (0-100)

    Returns:
        Signal classification: strong, moderate, weak
    """
    if adx is None:
        return "moderate"

    if adx > 25:
        return "strong"
    elif adx >= 20:
        return "moderate"
    else:
        return "weak"


def calculate_rsi_score(rsi: float, gap_direction: str) -> float:
    """Calculate RSI score (0-100) based on alignment with gap direction.

    For GAP UP (BUY):
        RSI 30-50: 100 (ideal - room to run)
        RSI 50-60: 80
        RSI 60-70: 50
        RSI > 70: 20 (overbought)
        RSI < 30: 60 (oversold bounce)

    For GAP DOWN (SHORT):
        RSI 50-70: 100 (ideal - room to fall)
        RSI 40-50: 80
        RSI 30-40: 50
        RSI < 30: 20 (oversold)
        RSI > 70: 60 (overbought reversal)
    """
    if rsi is None:
        return 50.0  # Neutral score for missing data

    if gap_direction == "UP":
        if 30 <= rsi <= 50:
            return 100.0
        elif 50 < rsi <= 60:
            return 80.0
        elif 60 < rsi <= 70:
            return 50.0
        elif rsi > 70:
            return 20.0
        else:  # rsi < 30
            return 60.0
    else:  # DOWN
        if 50 <= rsi <= 70:
            return 100.0
        elif 40 <= rsi < 50:
            return 80.0
        elif 30 <= rsi < 40:
            return 50.0
        elif rsi < 30:
            return 20.0
        else:  # rsi > 70
            return 60.0


def calculate_adx_score(adx: float) -> float:
    """Calculate ADX score (0-100) based on trend strength.

    ADX 50+: 100 (very strong trend)
    ADX 25-50: 50-100 (linear scale)
    ADX 20-25: 40-50 (moderate)
    ADX < 20: 0-40 (weak)
    """
    if adx is None:
        return 50.0  # Neutral score for missing data

    # Normalize to 0-100 scale
    # ADX 50 = 100 score
    score = min(adx / 50 * 100, 100)
    return round(score, 2)


def get_market_cap_tier(market_cap: Optional[float]) -> str:
    """Classify stock by market capitalization.

    Args:
        market_cap: Market capitalization in dollars

    Returns:
        Tier classification: MEGA, LARGE, MID, SMALL, MICRO
    """
    if market_cap is None or market_cap <= 0:
        return "UNKNOWN"

    if market_cap >= 200_000_000_000:  # $200B+
        return "MEGA"
    elif market_cap >= 10_000_000_000:  # $10B - $200B
        return "LARGE"
    elif market_cap >= 2_000_000_000:  # $2B - $10B
        return "MID"
    elif market_cap >= 300_000_000:  # $300M - $2B
        return "SMALL"
    else:  # < $300M
        return "MICRO"


def calculate_market_cap_score(market_cap: Optional[float]) -> float:
    """Calculate market cap score (0-100) favoring larger caps.

    Larger market caps are considered more stable and liquid.

    Scores:
        MEGA (>$200B): 100
        LARGE ($10B-$200B): 85
        MID ($2B-$10B): 70
        SMALL ($300M-$2B): 55
        MICRO (<$300M): 40
        UNKNOWN: 50 (neutral)
    """
    tier = get_market_cap_tier(market_cap)

    tier_scores = {
        "MEGA": 100.0,
        "LARGE": 85.0,
        "MID": 70.0,
        "SMALL": 55.0,
        "MICRO": 40.0,
        "UNKNOWN": 50.0
    }

    return tier_scores.get(tier, 50.0)


def get_volatility_tier(volatility: Optional[float]) -> str:
    """Classify stock by annualized volatility (volatility_20d).

    Args:
        volatility: Annualized volatility percentage (e.g., 30.0 for 30%)

    Returns:
        Tier classification: LOW, MEDIUM, HIGH, VERY_HIGH, EXTREME
    """
    if volatility is None or volatility <= 0:
        return "UNKNOWN"

    if volatility < 30:
        return "LOW"
    elif volatility < 50:
        return "MEDIUM"
    elif volatility < 80:
        return "HIGH"
    elif volatility < 120:
        return "VERY_HIGH"
    else:
        return "EXTREME"


def calculate_volatility_score(volatility: Optional[float]) -> float:
    """Calculate volatility score (0-100) penalizing high volatility.

    Lower volatility is preferred for more predictable gap behavior.

    Scores (higher = better, lower volatility):
        LOW (<30%): 100 - No penalty
        MEDIUM (30-50%): 85 - Slight penalty
        HIGH (50-80%): 70 - Moderate penalty
        VERY_HIGH (80-120%): 50 - Heavy penalty
        EXTREME (>120%): 30 - Maximum penalty
        UNKNOWN: 70 (neutral-conservative)
    """
    tier = get_volatility_tier(volatility)

    tier_scores = {
        "LOW": 100.0,
        "MEDIUM": 85.0,
        "HIGH": 70.0,
        "VERY_HIGH": 50.0,
        "EXTREME": 30.0,
        "UNKNOWN": 70.0
    }

    return tier_scores.get(tier, 70.0)


def calculate_priority_score(
    gap_pct: float,
    volume_ratio: float,
    adx: Optional[float],
    rsi: Optional[float],
    gap_direction: str,
    market_cap: Optional[float] = None,
    volatility: Optional[float] = None,
    weights: Optional[dict] = None,
    market_bias: Optional[str] = None,
    signal_direction: Optional[str] = None
) -> float:
    """Calculate composite priority score for signal ranking.

    Formula (default weights - updated 2026-01-08 to favor larger, more liquid stocks):
    priority_score = (
        gap_score * 0.15 +
        volume_score * 0.20 +      # Increased for liquidity
        adx_score * 0.20 +
        rsi_score * 0.15 +
        market_cap_score * 0.20 +  # Increased to favor larger caps
        volatility_score * 0.10    # Increased penalty for high volatility
    ) + market_direction_boost     # +15 aligned, -10 counter-trend

    Args:
        gap_pct: Gap percentage (absolute value used)
        volume_ratio: Volume relative to average (1.0 = average)
        adx: ADX value (0-100)
        rsi: RSI value (0-100)
        gap_direction: 'UP' or 'DOWN'
        market_cap: Market capitalization in dollars (optional)
        volatility: Annualized volatility percentage (optional)
        weights: Optional custom weights dict
        market_bias: Market bias from SPY ("BULLISH", "BEARISH", "NEUTRAL", etc.)
        signal_direction: Signal direction ("LONG" or "SHORT")

    Returns:
        Priority score (0-100, clamped)
    """
    # Default weights (2026-01-08: Updated to favor larger, more liquid stocks)
    if weights is None:
        weights = {
            'gap': 0.15,         # Gap size importance
            'volume': 0.20,      # Liquidity emphasis (increased)
            'adx': 0.20,         # Trend strength
            'rsi': 0.15,         # Direction alignment
            'market_cap': 0.20,  # Larger caps strongly preferred (increased)
            'volatility': 0.10   # Penalty for high volatility (increased)
        }

    # Calculate component scores
    # Gap score: 5% gap = 100 score (capped at 100)
    gap_score = min(abs(gap_pct) / 5.0 * 100, 100)

    # Volume score: 2x volume = 100 score (capped at 100)
    volume_score = min(volume_ratio * 50, 100) if volume_ratio else 50.0

    # ADX score
    adx_score = calculate_adx_score(adx) if adx is not None else 50.0

    # RSI score
    rsi_score = calculate_rsi_score(rsi, gap_direction) if rsi is not None else 50.0

    # Market cap score
    mcap_score = calculate_market_cap_score(market_cap)

    # Volatility score (lower volatility = higher score)
    vol_score = calculate_volatility_score(volatility)

    # Weighted composite (base score)
    base_priority = (
        gap_score * weights['gap'] +
        volume_score * weights['volume'] +
        adx_score * weights['adx'] +
        rsi_score * weights['rsi'] +
        mcap_score * weights['market_cap'] +
        vol_score * weights['volatility']
    )

    # Add market direction boost/penalty (2026-01-15)
    direction_boost = 0.0
    if market_bias and signal_direction:
        direction_boost = calculate_market_direction_boost(market_bias, signal_direction)

    final_priority = base_priority + direction_boost

    # Clamp to 0-100 range
    return round(max(0, min(100, final_priority)), 2)


def get_position_tier(rank: int, total_signals: int, max_trades: int = 6) -> str:
    """Determine position size tier based on priority rank.

    Top 2: top_tier (120% of base)
    Mid 2: mid_tier (100% of base)
    Bottom 2: bottom_tier (80% of base)

    Args:
        rank: Signal rank (1 = highest priority)
        total_signals: Total signals being considered
        max_trades: Maximum trades per day

    Returns:
        Position tier: top_tier, mid_tier, bottom_tier, or excluded
    """
    if rank > max_trades:
        return "excluded"

    if rank <= 2:
        return "top_tier"
    elif rank <= 4:
        return "mid_tier"
    else:
        return "bottom_tier"


def get_tier_multiplier(tier: str) -> float:
    """Get position size multiplier for tier.

    Args:
        tier: Position tier

    Returns:
        Size multiplier
    """
    multipliers = {
        'top_tier': 1.20,
        'mid_tier': 1.00,
        'bottom_tier': 0.80,
        'excluded': 0.0
    }
    return multipliers.get(tier, 1.0)


def calculate_sma(prices: List[float], period: int = 20) -> Optional[float]:
    """Calculate Simple Moving Average.

    Args:
        prices: List of closing prices (oldest to newest)
        period: SMA period (default 20)

    Returns:
        SMA value or None if insufficient data
    """
    if not prices or len(prices) < period:
        logger.warning(f"Insufficient data for SMA: need {period} prices, got {len(prices) if prices else 0}")
        return None
    return sum(prices[-period:]) / period


def is_trending_in_direction(
    close: float,
    open_price: float,
    prev_close: float,
    direction: str
) -> bool:
    """Determine if price is trending in position direction.

    Used for overnight holding decision:
    - For LONG: Bullish if close > open (green candle) AND close > prev_close
    - For SHORT: Bearish if close < open (red candle) AND close < prev_close

    Args:
        close: Current day's closing price
        open_price: Current day's opening price
        prev_close: Previous day's closing price
        direction: Position direction ("LONG" or "SHORT")

    Returns:
        True if trending in the position's direction
    """
    if direction == "LONG":
        # Bullish continuation: green candle AND higher close
        return close > open_price and close > prev_close
    else:  # SHORT
        # Bearish continuation: red candle AND lower close
        return close < open_price and close < prev_close


def calculate_market_direction_boost(
    market_bias: str,
    signal_direction: str
) -> float:
    """Calculate score adjustment based on market direction alignment.

    Boosts signals aligned with market direction, penalizes counter-trend signals.

    Args:
        market_bias: Market bias from SPY analysis
                     ("STRONG_BULLISH", "BULLISH", "NEUTRAL", "BEARISH", "STRONG_BEARISH")
        signal_direction: Signal direction ("LONG" or "SHORT")

    Returns:
        Score adjustment (-10 to +15 points)
    """
    # Aligned with market - boost
    if market_bias in ["BULLISH", "STRONG_BULLISH"] and signal_direction == "LONG":
        return 15.0
    elif market_bias in ["BEARISH", "STRONG_BEARISH"] and signal_direction == "SHORT":
        return 15.0
    # Counter-trend - penalty
    elif market_bias in ["BULLISH", "STRONG_BULLISH"] and signal_direction == "SHORT":
        return -10.0
    elif market_bias in ["BEARISH", "STRONG_BEARISH"] and signal_direction == "LONG":
        return -10.0
    # Neutral market - no adjustment
    return 0.0
