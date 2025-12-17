"""Gap Backtest Data Loader Module.

Provides data loading with FMP minute-level data support for accurate
backtesting of gap trading strategies.
"""

import logging
import os
import json
import hashlib
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class MinuteBar:
    """Single minute price bar."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class DailyBar:
    """Single daily price bar."""
    date: date
    open: float
    high: float
    low: float
    close: float
    volume: int
    adj_close: Optional[float] = None


class GapBacktestDataLoader:
    """Data loader for gap trading backtests.

    Supports both minute-level and daily data from FMP API with
    intelligent caching to minimize API calls.

    Attributes:
        fmp_client: FMP API client instance
        cache_dir: Directory for caching data
    """

    def __init__(
        self,
        fmp_client: Any = None,
        cache_dir: Optional[str] = None,
        api_key: Optional[str] = None
    ):
        """Initialize data loader.

        Args:
            fmp_client: FMP client instance (optional, will create one if not provided)
            cache_dir: Directory for data caching
            api_key: FMP API key (if not using existing client)
        """
        self.fmp_client = fmp_client
        self.api_key = api_key or os.environ.get('FMP_API_KEY')

        # Set up cache directory
        if cache_dir:
            self.cache_dir = Path(cache_dir)
        else:
            self.cache_dir = Path.home() / '.gap_trading_cache'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # In-memory caches
        self._daily_cache: Dict[str, pd.DataFrame] = {}
        self._minute_cache: Dict[str, pd.DataFrame] = {}
        self._atr_cache: Dict[str, pd.Series] = {}

        logger.info(f"GapBacktestDataLoader initialized, cache_dir={self.cache_dir}")

    def _get_fmp_client(self):
        """Get or create FMP client."""
        if self.fmp_client:
            return self.fmp_client

        try:
            from fmp import FMPClient
            self.fmp_client = FMPClient(api_key=self.api_key)
            return self.fmp_client
        except ImportError:
            logger.error("FMP package not installed. Install with: pip install fmp")
            raise

    def get_daily_bars(
        self,
        symbol: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get daily OHLCV data for a symbol.

        Args:
            symbol: Stock ticker symbol
            start_date: Start date
            end_date: End date
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: date, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{start_date}_{end_date}_daily"

        # Check memory cache
        if use_cache and cache_key in self._daily_cache:
            return self._daily_cache[cache_key].copy()

        # Check file cache
        cache_file = self.cache_dir / f"daily_{symbol}_{start_date}_{end_date}.parquet"
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._daily_cache[cache_key] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Failed to read cache for {symbol}: {e}")

        # Fetch from FMP
        logger.info(f"Fetching daily data for {symbol} from {start_date} to {end_date}")
        try:
            client = self._get_fmp_client()
            data = client.get(f'historical-price-full/{symbol}', {
                'from': start_date.strftime('%Y-%m-%d'),
                'to': end_date.strftime('%Y-%m-%d')
            })

            if not data or 'historical' not in data:
                logger.warning(f"No daily data returned for {symbol}")
                return pd.DataFrame()

            historical = data['historical']
            if not historical:
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(historical)
            df['date'] = pd.to_datetime(df['date']).dt.date
            df = df.sort_values('date').reset_index(drop=True)

            # Select and rename columns
            columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            if 'adjClose' in df.columns:
                columns.append('adjClose')
                df = df[columns]
                df = df.rename(columns={'adjClose': 'adj_close'})
            else:
                df = df[columns]

            # Cache results
            try:
                df.to_parquet(cache_file)
            except Exception as e:
                logger.warning(f"Failed to cache data for {symbol}: {e}")

            self._daily_cache[cache_key] = df
            return df.copy()

        except Exception as e:
            logger.error(f"Error fetching daily data for {symbol}: {e}")
            return pd.DataFrame()

    def get_minute_bars(
        self,
        symbol: str,
        target_date: date,
        use_cache: bool = True
    ) -> pd.DataFrame:
        """Get 1-minute bars for a specific date.

        Args:
            symbol: Stock ticker symbol
            target_date: Date to fetch minute data for
            use_cache: Whether to use cached data

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume

        Note:
            FMP provides last 30 days of minute data. For older dates,
            this will return empty DataFrame.
        """
        cache_key = f"{symbol}_{target_date}_minute"

        # Check memory cache
        if use_cache and cache_key in self._minute_cache:
            return self._minute_cache[cache_key].copy()

        # Check file cache
        cache_file = self.cache_dir / f"minute_{symbol}_{target_date}.parquet"
        if use_cache and cache_file.exists():
            try:
                df = pd.read_parquet(cache_file)
                self._minute_cache[cache_key] = df
                return df.copy()
            except Exception as e:
                logger.warning(f"Failed to read minute cache for {symbol}: {e}")

        # Fetch from FMP
        logger.debug(f"Fetching minute data for {symbol} on {target_date}")
        try:
            client = self._get_fmp_client()

            # FMP minute data endpoint
            # Note: FMP only provides ~30 days of minute data
            data = client.get(f'historical-chart/1min/{symbol}', {
                'from': target_date.strftime('%Y-%m-%d'),
                'to': target_date.strftime('%Y-%m-%d')
            })

            if not data:
                logger.debug(f"No minute data for {symbol} on {target_date}")
                return pd.DataFrame()

            # Convert to DataFrame
            df = pd.DataFrame(data)
            if df.empty:
                return pd.DataFrame()

            df['timestamp'] = pd.to_datetime(df['date'])
            df = df.sort_values('timestamp').reset_index(drop=True)

            # Filter to target date only
            df = df[df['timestamp'].dt.date == target_date]

            # Select columns
            df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

            # Cache results
            if not df.empty:
                try:
                    df.to_parquet(cache_file)
                except Exception as e:
                    logger.warning(f"Failed to cache minute data for {symbol}: {e}")

            self._minute_cache[cache_key] = df
            return df.copy()

        except Exception as e:
            logger.error(f"Error fetching minute data for {symbol} on {target_date}: {e}")
            return pd.DataFrame()

    def get_price_at_time(
        self,
        symbol: str,
        target_date: date,
        target_time: time,
        fallback_to_daily: bool = True
    ) -> Optional[float]:
        """Get price at a specific time.

        Args:
            symbol: Stock ticker symbol
            target_date: Date
            target_time: Time (Eastern Time)
            fallback_to_daily: If minute data unavailable, use daily open

        Returns:
            Price at the specified time, or None if unavailable
        """
        # Try minute data first
        minute_df = self.get_minute_bars(symbol, target_date)

        if not minute_df.empty:
            # Find the bar at or just before target time
            target_dt = datetime.combine(target_date, target_time)
            mask = minute_df['timestamp'] <= target_dt
            matching_bars = minute_df[mask]

            if not matching_bars.empty:
                return matching_bars.iloc[-1]['close']

        # Fallback to daily open
        if fallback_to_daily:
            daily_df = self.get_daily_bars(
                symbol,
                target_date - timedelta(days=1),
                target_date
            )
            if not daily_df.empty:
                today_data = daily_df[daily_df['date'] == target_date]
                if not today_data.empty:
                    return today_data.iloc[0]['open']

        return None

    def get_intraday_range(
        self,
        symbol: str,
        target_date: date,
        start_time: time,
        end_time: time,
        fallback_to_daily: bool = True
    ) -> Tuple[Optional[float], Optional[float]]:
        """Get high/low between two times for stop-loss simulation.

        Args:
            symbol: Stock ticker symbol
            target_date: Date
            start_time: Start time (ET)
            end_time: End time (ET)
            fallback_to_daily: If minute data unavailable, use daily high/low

        Returns:
            Tuple of (high, low) between the times
        """
        minute_df = self.get_minute_bars(symbol, target_date)

        if not minute_df.empty:
            start_dt = datetime.combine(target_date, start_time)
            end_dt = datetime.combine(target_date, end_time)

            mask = (minute_df['timestamp'] >= start_dt) & (minute_df['timestamp'] <= end_dt)
            period_bars = minute_df[mask]

            if not period_bars.empty:
                return period_bars['high'].max(), period_bars['low'].min()

        # Fallback to daily high/low
        if fallback_to_daily:
            daily_df = self.get_daily_bars(
                symbol,
                target_date - timedelta(days=1),
                target_date
            )
            if not daily_df.empty:
                today_data = daily_df[daily_df['date'] == target_date]
                if not today_data.empty:
                    row = today_data.iloc[0]
                    return row['high'], row['low']

        return None, None

    def get_close_price(
        self,
        symbol: str,
        target_date: date,
        eod_time: Optional[time] = None
    ) -> Optional[float]:
        """Get closing price for a date.

        Args:
            symbol: Stock ticker symbol
            target_date: Date
            eod_time: Optional specific time (default: use daily close)

        Returns:
            Closing price
        """
        if eod_time:
            # Use minute data for specific time
            price = self.get_price_at_time(symbol, target_date, eod_time)
            if price:
                return price

        # Use daily close
        daily_df = self.get_daily_bars(
            symbol,
            target_date - timedelta(days=1),
            target_date
        )
        if not daily_df.empty:
            today_data = daily_df[daily_df['date'] == target_date]
            if not today_data.empty:
                return today_data.iloc[0]['close']

        return None

    def calculate_atr(
        self,
        symbol: str,
        as_of_date: date,
        period: int = 14,
        lookback_days: int = 60
    ) -> Optional[float]:
        """Calculate ATR as of a specific date.

        Args:
            symbol: Stock ticker symbol
            as_of_date: Date to calculate ATR for
            period: ATR period (default 14)
            lookback_days: Days of data to fetch

        Returns:
            ATR value or None if insufficient data
        """
        cache_key = f"{symbol}_{as_of_date}_atr{period}"
        if cache_key in self._atr_cache:
            return self._atr_cache[cache_key]

        # Get historical data
        start_date = as_of_date - timedelta(days=lookback_days)
        df = self.get_daily_bars(symbol, start_date, as_of_date)

        if df.empty or len(df) < period + 1:
            return None

        # Calculate True Range
        df = df.copy()
        df['prev_close'] = df['close'].shift(1)
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['prev_close'])
        df['tr3'] = abs(df['low'] - df['prev_close'])
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)

        # Calculate ATR
        df['atr'] = df['true_range'].rolling(window=period).mean()

        # Get ATR as of the target date
        target_data = df[df['date'] <= as_of_date]
        if target_data.empty:
            return None

        atr = target_data.iloc[-1]['atr']
        if pd.isna(atr):
            return None

        self._atr_cache[cache_key] = atr
        return atr

    def get_previous_close(
        self,
        symbol: str,
        target_date: date
    ) -> Optional[float]:
        """Get previous trading day's close price.

        Args:
            symbol: Stock ticker symbol
            target_date: Date to get previous close for

        Returns:
            Previous day's closing price
        """
        # Fetch a few days of data to ensure we get previous trading day
        start_date = target_date - timedelta(days=10)
        df = self.get_daily_bars(symbol, start_date, target_date - timedelta(days=1))

        if df.empty:
            return None

        # Get the most recent trading day before target_date
        prev_data = df[df['date'] < target_date]
        if prev_data.empty:
            return None

        return prev_data.iloc[-1]['close']

    def preload_data(
        self,
        symbols: List[str],
        start_date: date,
        end_date: date,
        include_minute_data: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Preload data for multiple symbols.

        Args:
            symbols: List of symbols to preload
            start_date: Start date
            end_date: End date
            include_minute_data: Whether to preload minute data (slow)

        Returns:
            Dict mapping symbol to daily DataFrame
        """
        logger.info(f"Preloading data for {len(symbols)} symbols from {start_date} to {end_date}")

        results = {}
        for symbol in symbols:
            df = self.get_daily_bars(symbol, start_date - timedelta(days=30), end_date)
            if not df.empty:
                results[symbol] = df

        logger.info(f"Preloaded daily data for {len(results)} symbols")

        if include_minute_data:
            logger.info("Preloading minute data (this may take a while)...")
            current_date = start_date
            while current_date <= end_date:
                if current_date.weekday() < 5:  # Skip weekends
                    for symbol in symbols:
                        self.get_minute_bars(symbol, current_date)
                current_date += timedelta(days=1)
            logger.info("Minute data preloading complete")

        return results

    def clear_cache(self, cache_type: str = 'all'):
        """Clear data caches.

        Args:
            cache_type: 'memory', 'file', or 'all'
        """
        if cache_type in ('memory', 'all'):
            self._daily_cache.clear()
            self._minute_cache.clear()
            self._atr_cache.clear()
            logger.info("Memory cache cleared")

        if cache_type in ('file', 'all'):
            import shutil
            if self.cache_dir.exists():
                shutil.rmtree(self.cache_dir)
                self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("File cache cleared")

    def get_trading_days(
        self,
        start_date: date,
        end_date: date,
        reference_symbol: str = 'SPY'
    ) -> List[date]:
        """Get list of trading days in date range.

        Args:
            start_date: Start date
            end_date: End date
            reference_symbol: Symbol to use for determining trading days

        Returns:
            List of trading days
        """
        df = self.get_daily_bars(reference_symbol, start_date, end_date)
        if df.empty:
            # Fall back to simple weekday filter
            days = []
            current = start_date
            while current <= end_date:
                if current.weekday() < 5:
                    days.append(current)
                current += timedelta(days=1)
            return days

        return df['date'].tolist()
