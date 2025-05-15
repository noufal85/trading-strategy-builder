"""
Yahoo Finance data provider for the Strategy Builder framework
"""
from typing import Dict, Any, Optional, Union
from datetime import datetime, timedelta
import pandas as pd
import yfinance as yf

from .data_provider import DataProvider
from ..utils.types import Bar


class YahooDataProvider(DataProvider):
    """
    Data provider that fetches data from Yahoo Finance.
    """
    
    def __init__(self):
        """
        Initialize the Yahoo Finance data provider.
        """
        pass
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical market data from Yahoo Finance.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date for the data
            end_date: End date for the data (defaults to current date)
            interval: Data interval (e.g., "1d" for daily, "1h" for hourly)
                Valid intervals: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
            
        Returns:
            pd.DataFrame: DataFrame containing the historical data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Map interval to yfinance format if needed
        interval_mapping = {
            "1m": "1m",
            "2m": "2m",
            "5m": "5m",
            "15m": "15m",
            "30m": "30m",
            "1h": "60m",
            "1d": "1d",
            "1wk": "1wk",
            "1mo": "1mo"
        }
        
        yf_interval = interval_mapping.get(interval, interval)
        
        # Fetch data from Yahoo Finance
        try:
            data = yf.download(
                symbol,
                start=start_date,
                end=end_date,
                interval=yf_interval,
                progress=False
            )
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
        
        if data.empty:
            raise ValueError(f"No data available for symbol: {symbol}")
        
        # Reset index to make Date a column
        data = data.reset_index()
        
        return self.normalize_data(data)
    
    def get_latest_data(self, symbol: str) -> Union[Bar, Dict[str, Any]]:
        """
        Get the latest data point from Yahoo Finance.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Union[Bar, Dict[str, Any]]: The latest data point
        """
        # Get the last 2 days of data to ensure we have the latest
        end_date = datetime.now()
        start_date = end_date - timedelta(days=2)
        
        data = self.get_historical_data(symbol, start_date, end_date)
        if data.empty:
            raise ValueError(f"No data available for symbol: {symbol}")
        
        latest = data.iloc[-1].to_dict()
        
        # Convert to Bar format
        return Bar(
            symbol=symbol,
            timestamp=latest.get('Date', latest.get('Datetime', datetime.now())),
            open=float(latest['Open']),
            high=float(latest['High']),
            low=float(latest['Low']),
            close=float(latest['Close']),
            volume=float(latest['Volume']),
            metadata=None
        )
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize Yahoo Finance data to a standard format.
        
        Args:
            data: Raw data from Yahoo Finance
            
        Returns:
            pd.DataFrame: Normalized data
        """
        # Rename columns to standard format
        column_mapping = {
            'Date': 'date',
            'Datetime': 'date',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Adj Close': 'adj_close',
            'Volume': 'volume'
        }
        
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        return data
    
    def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a ticker symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Dict[str, Any]: Information about the ticker
        """
        ticker = yf.Ticker(symbol)
        return ticker.info
