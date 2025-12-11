"""Gap Trading Volatility Calculator.

Calculates volatility metrics for gap trading strategy:
- ATR (Average True Range) and ATR%
- ADR (Average Daily Range) percentage
- Annualized volatility (20-day and 60-day)
- Beta vs SPY
- Gap statistics (average gap size, fill rates)

Uses FMP gap_trading module for core calculations and adds
gap-specific statistics for trading decisions.
"""

from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class GapStats:
    """Statistics about historical gaps for a symbol.

    Attributes:
        avg_gap_up_pct: Average gap up percentage
        avg_gap_down_pct: Average gap down percentage
        gap_up_fill_rate: Percentage of gap ups that fill within day
        gap_down_fill_rate: Percentage of gap downs that fill within day
        large_gap_threshold: Threshold for "large" gaps (default 2%)
        large_gap_count: Number of large gaps in period
        avg_gap_fill_time_bars: Average bars to fill gap (if filled)
    """
    avg_gap_up_pct: float = 0.0
    avg_gap_down_pct: float = 0.0
    gap_up_fill_rate: float = 0.0
    gap_down_fill_rate: float = 0.0
    large_gap_threshold: float = 2.0
    large_gap_count: int = 0
    avg_gap_fill_time_bars: Optional[float] = None


@dataclass
class VolatilityMetrics:
    """Complete volatility metrics for a symbol.

    Attributes:
        symbol: Stock ticker symbol
        atr_14: 14-day Average True Range
        atr_pct: ATR as percentage of current price
        adr_pct: 20-day Average Daily Range as percentage
        volatility_20d: 20-day annualized volatility
        volatility_60d: 60-day annualized volatility
        beta: Beta vs SPY
        current_price: Latest closing price
        calculation_date: Date of calculation
        gap_stats: Gap statistics (optional)
    """
    symbol: str
    atr_14: float
    atr_pct: float
    adr_pct: float
    volatility_20d: float
    volatility_60d: Optional[float] = None
    beta: Optional[float] = None
    current_price: Optional[float] = None
    calculation_date: Optional[date] = None
    gap_stats: Optional[GapStats] = None


class VolatilityCalculator:
    """Calculator for gap trading volatility metrics.

    Provides methods to calculate ATR, ADR, volatility, beta,
    and gap statistics for position sizing and risk management.

    Can work standalone with price DataFrames or integrate with
    the FMP gap_trading module for data fetching.

    Formulas:
    - ATR = SMA(True Range, 14)
    - True Range = MAX(H-L, |H-PrevC|, |L-PrevC|)
    - ADR % = AVG((High-Low)/Close) * 100
    - Volatility = StdDev(ln returns) * sqrt(252) * 100
    - Beta = Cov(stock, SPY) / Var(SPY)
    """

    def __init__(
        self,
        atr_period: int = 14,
        adr_period: int = 20,
        vol_period_short: int = 20,
        vol_period_long: int = 60,
        beta_period: int = 60
    ):
        """Initialize VolatilityCalculator.

        Args:
            atr_period: Period for ATR calculation (default 14)
            adr_period: Period for ADR calculation (default 20)
            vol_period_short: Short volatility period (default 20)
            vol_period_long: Long volatility period (default 60)
            beta_period: Period for beta calculation (default 60)
        """
        self.atr_period = atr_period
        self.adr_period = adr_period
        self.vol_period_short = vol_period_short
        self.vol_period_long = vol_period_long
        self.beta_period = beta_period
        self._spy_returns: Optional[pd.Series] = None

    def calculate_atr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None
    ) -> pd.Series:
        """Calculate Average True Range.

        True Range = MAX(H-L, |H-PrevC|, |L-PrevC|)
        ATR = SMA(True Range, period)

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ATR period (default: self.atr_period)

        Returns:
            Series of ATR values
        """
        period = period or self.atr_period
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)

        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()

        return atr

    def calculate_atr_percent(self, atr: float, price: float) -> float:
        """Calculate ATR as percentage of price.

        Args:
            atr: Average True Range value
            price: Current price

        Returns:
            ATR percentage
        """
        if price <= 0:
            return 0.0
        return (atr / price) * 100

    def calculate_adr(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        period: Optional[int] = None
    ) -> float:
        """Calculate Average Daily Range percentage.

        ADR% = AVG((High-Low)/Close) * 100

        Args:
            high: High prices
            low: Low prices
            close: Close prices
            period: ADR period (default: self.adr_period)

        Returns:
            ADR percentage
        """
        period = period or self.adr_period
        daily_range_pct = ((high - low) / close) * 100
        return float(daily_range_pct.tail(period).mean())

    def calculate_volatility(
        self,
        close: pd.Series,
        period: Optional[int] = None,
        annualize: bool = True
    ) -> float:
        """Calculate annualized volatility from price series.

        Volatility = StdDev(ln returns) * sqrt(252) * 100

        Args:
            close: Close prices
            period: Lookback period (default: self.vol_period_short)
            annualize: Whether to annualize (multiply by sqrt(252))

        Returns:
            Volatility as percentage
        """
        period = period or self.vol_period_short
        log_returns = np.log(close / close.shift(1))
        std = log_returns.tail(period).std()

        if annualize:
            return float(std * np.sqrt(252) * 100)
        return float(std * 100)

    def calculate_beta(
        self,
        stock_returns: pd.Series,
        market_returns: pd.Series,
        period: Optional[int] = None
    ) -> Optional[float]:
        """Calculate beta vs market (SPY).

        Beta = Cov(stock, market) / Var(market)

        Args:
            stock_returns: Stock log returns
            market_returns: Market (SPY) log returns
            period: Period for calculation (default: self.beta_period)

        Returns:
            Beta value or None if insufficient data
        """
        period = period or self.beta_period

        # Align and trim to period
        stock_tail = stock_returns.tail(period).dropna()
        market_tail = market_returns.tail(period).dropna()

        if len(stock_tail) < 20 or len(market_tail) < 20:
            return None

        # Align indices
        common_idx = stock_tail.index.intersection(market_tail.index)
        if len(common_idx) < 20:
            return None

        stock_aligned = stock_tail.loc[common_idx]
        market_aligned = market_tail.loc[common_idx]

        cov = stock_aligned.cov(market_aligned)
        var = market_aligned.var()

        if var > 0:
            return float(cov / var)
        return None

    def calculate_gap_stats(
        self,
        df: pd.DataFrame,
        gap_threshold: float = 0.5,
        large_gap_threshold: float = 2.0
    ) -> GapStats:
        """Calculate gap statistics from historical data.

        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns
            gap_threshold: Minimum gap percentage to consider (default 0.5%)
            large_gap_threshold: Threshold for "large" gaps (default 2%)

        Returns:
            GapStats with gap analysis
        """
        df = df.copy()

        # Calculate gap percentage: (today's open - yesterday's close) / yesterday's close
        df['prev_close'] = df['close'].shift(1)
        df['gap_pct'] = ((df['open'] - df['prev_close']) / df['prev_close']) * 100

        # Filter significant gaps
        gaps_up = df[df['gap_pct'] >= gap_threshold]
        gaps_down = df[df['gap_pct'] <= -gap_threshold]

        # Calculate average gaps
        avg_gap_up = float(gaps_up['gap_pct'].mean()) if len(gaps_up) > 0 else 0.0
        avg_gap_down = float(abs(gaps_down['gap_pct'].mean())) if len(gaps_down) > 0 else 0.0

        # Calculate fill rates (gap filled if price moves back through gap)
        # Gap up fills if low <= prev_close
        # Gap down fills if high >= prev_close
        gap_up_fills = len(gaps_up[gaps_up['low'] <= gaps_up['prev_close']])
        gap_down_fills = len(gaps_down[gaps_down['high'] >= gaps_down['prev_close']])

        gap_up_fill_rate = (gap_up_fills / len(gaps_up) * 100) if len(gaps_up) > 0 else 0.0
        gap_down_fill_rate = (gap_down_fills / len(gaps_down) * 100) if len(gaps_down) > 0 else 0.0

        # Count large gaps
        large_gaps = df[abs(df['gap_pct']) >= large_gap_threshold]

        return GapStats(
            avg_gap_up_pct=avg_gap_up,
            avg_gap_down_pct=avg_gap_down,
            gap_up_fill_rate=gap_up_fill_rate,
            gap_down_fill_rate=gap_down_fill_rate,
            large_gap_threshold=large_gap_threshold,
            large_gap_count=len(large_gaps)
        )

    def calculate_all_metrics(
        self,
        df: pd.DataFrame,
        symbol: str,
        spy_df: Optional[pd.DataFrame] = None,
        include_gap_stats: bool = True
    ) -> Optional[VolatilityMetrics]:
        """Calculate all volatility metrics for a symbol.

        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            symbol: Stock ticker symbol
            spy_df: SPY DataFrame for beta calculation (optional)
            include_gap_stats: Whether to calculate gap statistics

        Returns:
            VolatilityMetrics or None if insufficient data
        """
        required_cols = ['open', 'high', 'low', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.error(f"Missing required columns in DataFrame for {symbol}")
            return None

        if len(df) < self.atr_period:
            logger.warning(f"Insufficient data for {symbol}: {len(df)} bars")
            return None

        # Calculate ATR
        atr_series = self.calculate_atr(df['high'], df['low'], df['close'])
        atr_14 = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

        # Current price
        current_price = float(df['close'].iloc[-1])

        # ATR %
        atr_pct = self.calculate_atr_percent(atr_14, current_price)

        # ADR %
        adr_pct = self.calculate_adr(df['high'], df['low'], df['close'])

        # Volatility
        vol_20 = self.calculate_volatility(df['close'], self.vol_period_short)
        vol_60 = None
        if len(df) >= self.vol_period_long:
            vol_60 = self.calculate_volatility(df['close'], self.vol_period_long)

        # Beta
        beta = None
        if spy_df is not None and len(spy_df) >= self.beta_period:
            stock_returns = np.log(df['close'] / df['close'].shift(1))
            market_returns = np.log(spy_df['close'] / spy_df['close'].shift(1))

            # Align by date if date column exists
            if 'date' in df.columns and 'date' in spy_df.columns:
                stock_df_temp = df.set_index('date')['close']
                spy_df_temp = spy_df.set_index('date')['close']
                stock_returns = np.log(stock_df_temp / stock_df_temp.shift(1))
                market_returns = np.log(spy_df_temp / spy_df_temp.shift(1))

            beta = self.calculate_beta(stock_returns, market_returns)

        # Gap stats
        gap_stats = None
        if include_gap_stats:
            gap_stats = self.calculate_gap_stats(df)

        # Calculation date
        calc_date = date.today()
        if 'date' in df.columns:
            last_date = df['date'].iloc[-1]
            if isinstance(last_date, (date, datetime)):
                calc_date = last_date if isinstance(last_date, date) else last_date.date()

        return VolatilityMetrics(
            symbol=symbol,
            atr_14=atr_14,
            atr_pct=atr_pct,
            adr_pct=adr_pct,
            volatility_20d=vol_20,
            volatility_60d=vol_60,
            beta=beta,
            current_price=current_price,
            calculation_date=calc_date,
            gap_stats=gap_stats
        )

    def calculate_batch_metrics(
        self,
        data: Dict[str, pd.DataFrame],
        spy_df: Optional[pd.DataFrame] = None,
        include_gap_stats: bool = True
    ) -> Dict[str, VolatilityMetrics]:
        """Calculate volatility metrics for multiple symbols.

        Args:
            data: Dict mapping symbol to OHLCV DataFrame
            spy_df: SPY DataFrame for beta calculation (optional)
            include_gap_stats: Whether to calculate gap statistics

        Returns:
            Dict mapping symbol to VolatilityMetrics
        """
        results = {}

        for symbol, df in data.items():
            metrics = self.calculate_all_metrics(
                df, symbol, spy_df, include_gap_stats
            )
            if metrics:
                results[symbol] = metrics

        return results

    def metrics_to_dataframe(
        self,
        metrics: Dict[str, VolatilityMetrics]
    ) -> pd.DataFrame:
        """Convert volatility metrics dict to DataFrame.

        Args:
            metrics: Dict mapping symbol to VolatilityMetrics

        Returns:
            DataFrame with volatility data
        """
        if not metrics:
            return pd.DataFrame()

        data = []
        for symbol, m in metrics.items():
            row = {
                'symbol': symbol,
                'atr_14': m.atr_14,
                'atr_pct': m.atr_pct,
                'adr_pct': m.adr_pct,
                'volatility_20d': m.volatility_20d,
                'volatility_60d': m.volatility_60d,
                'beta': m.beta,
                'current_price': m.current_price,
                'calculation_date': m.calculation_date
            }

            # Add gap stats if available
            if m.gap_stats:
                row['avg_gap_up_pct'] = m.gap_stats.avg_gap_up_pct
                row['avg_gap_down_pct'] = m.gap_stats.avg_gap_down_pct
                row['gap_up_fill_rate'] = m.gap_stats.gap_up_fill_rate
                row['gap_down_fill_rate'] = m.gap_stats.gap_down_fill_rate
                row['large_gap_count'] = m.gap_stats.large_gap_count

            data.append(row)

        return pd.DataFrame(data)
