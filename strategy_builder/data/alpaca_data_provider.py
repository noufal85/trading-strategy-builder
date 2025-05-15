"""
Alpaca data provider for the Strategy Builder framework
"""
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import pandas as pd
import alpaca_trade_api as tradeapi
from alpaca_trade_api.rest import TimeFrame

from .data_provider import DataProvider
from ..utils.types import Bar


class AlpacaDataProvider(DataProvider):
    """
    Data provider that fetches data from Alpaca Markets.
    
    This provider can be used for both historical data for backtesting
    and real-time data for live trading.
    """
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        base_url: str = "https://paper-api.alpaca.markets",
        data_feed: str = "iex"
    ):
        """
        Initialize the Alpaca data provider.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            base_url: Alpaca API base URL (paper or live)
            data_feed: Data feed to use (iex or sip)
        """
        self.api = tradeapi.REST(
            api_key,
            api_secret,
            base_url=base_url,
            api_version="v2"
        )
        self.data_feed = data_feed
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical market data from Alpaca.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date for the data
            end_date: End date for the data (defaults to current date)
            interval: Data interval (e.g., "1d" for daily, "1h" for hourly)
                Valid intervals: 1m, 5m, 15m, 30m, 1h, 1d
            
        Returns:
            pd.DataFrame: DataFrame containing the historical data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Map interval to Alpaca TimeFrame
        interval_mapping = {
            "1m": TimeFrame.Minute,
            "5m": TimeFrame.Minute,
            "15m": TimeFrame.Minute,
            "30m": TimeFrame.Minute,
            "1h": TimeFrame.Hour,
            "1d": TimeFrame.Day
        }
        
        # Get multiplier for time frames
        multiplier = 1
        if interval.startswith("5"):
            multiplier = 5
        elif interval.startswith("15"):
            multiplier = 15
        elif interval.startswith("30"):
            multiplier = 30
        
        timeframe = interval_mapping.get(interval, TimeFrame.Day)
        
        # Fetch data from Alpaca
        try:
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_date.isoformat(),
                end=end_date.isoformat(),
                adjustment='raw',
                feed=self.data_feed,
                limit=10000,
                timeframe_multiplier=multiplier
            ).df
            
            if bars.empty:
                raise ValueError(f"No data available for symbol: {symbol}")
            
            # Reset index to make timestamp a column
            bars = bars.reset_index()
            
            return self.normalize_data(bars)
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def get_latest_data(self, symbol: str) -> Union[Bar, Dict[str, Any]]:
        """
        Get the latest data point from Alpaca.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Union[Bar, Dict[str, Any]]: The latest data point
        """
        try:
            # Get the last trade
            last_trade = self.api.get_latest_trade(symbol)
            
            # Get the last quote
            last_quote = self.api.get_latest_quote(symbol)
            
            # Create a Bar object
            return Bar(
                symbol=symbol,
                timestamp=datetime.now(),
                open=last_quote.ap,  # Use ask price as open
                high=max(last_quote.ap, last_quote.bp),  # Use max of ask and bid as high
                low=min(last_quote.ap, last_quote.bp),  # Use min of ask and bid as low
                close=last_trade.p,  # Use last trade price as close
                volume=last_trade.s,  # Use last trade size as volume
                metadata={
                    'bid': last_quote.bp,
                    'ask': last_quote.ap,
                    'bid_size': last_quote.bs,
                    'ask_size': last_quote.as_
                }
            )
        except Exception as e:
            # If we can't get the latest trade/quote, fall back to getting the latest bar
            end_date = datetime.now()
            start_date = end_date - timedelta(days=1)
            
            data = self.get_historical_data(symbol, start_date, end_date, interval="1m")
            if data.empty:
                raise ValueError(f"No data available for symbol: {symbol}")
            
            latest = data.iloc[-1].to_dict()
            
            return Bar(
                symbol=symbol,
                timestamp=latest.get('timestamp', datetime.now()),
                open=float(latest['open']),
                high=float(latest['high']),
                low=float(latest['low']),
                close=float(latest['close']),
                volume=float(latest['volume']),
                metadata=None
            )
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Alpaca data to a standard format.
        
        Args:
            data: Raw data from Alpaca
            
        Returns:
            pd.DataFrame: Normalized data
        """
        # Rename columns to standard format
        column_mapping = {
            'timestamp': 'date',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'trade_count': 'trades',
            'vwap': 'vwap'
        }
        
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        return data
    
    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information from Alpaca.
        
        Returns:
            Dict[str, Any]: Account information
        """
        try:
            account = self.api.get_account()
            return {
                'id': account.id,
                'status': account.status,
                'equity': float(account.equity),
                'cash': float(account.cash),
                'buying_power': float(account.buying_power),
                'long_market_value': float(account.long_market_value),
                'short_market_value': float(account.short_market_value),
                'initial_margin': float(account.initial_margin),
                'maintenance_margin': float(account.maintenance_margin),
                'last_equity': float(account.last_equity),
                'day_trade_count': account.day_trade_count,
                'multiplier': account.multiplier
            }
        except Exception as e:
            raise ValueError(f"Failed to get account information: {str(e)}")
    
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Get current positions from Alpaca.
        
        Returns:
            List[Dict[str, Any]]: List of positions
        """
        try:
            positions = self.api.list_positions()
            return [
                {
                    'symbol': position.symbol,
                    'quantity': float(position.qty),
                    'entry_price': float(position.avg_entry_price),
                    'current_price': float(position.current_price),
                    'timestamp': datetime.now(),
                    'unrealized_pnl': float(position.unrealized_pl),
                    'market_value': float(position.market_value),
                    'cost_basis': float(position.cost_basis)
                }
                for position in positions
            ]
        except Exception as e:
            raise ValueError(f"Failed to get positions: {str(e)}")
