"""
FMP (Financial Modeling Prep) data provider for the Strategy Builder framework.

This module integrates the FMP package to provide market data for strategies.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Union
import pandas as pd

try:
    from fmp import FMPClient, StockPrice
except ImportError as e:
    raise ImportError(
        "FMP package not found. Please install dependencies using:\n"
        "  pip install -r requirements.txt\n"
        "\nIf you get authentication errors, configure your GitHub PAT token:\n"
        "  git config --global credential.helper store\n"
        "  echo 'https://YOUR_USERNAME:YOUR_PAT_TOKEN@github.com' >> ~/.git-credentials\n"
        f"\nOriginal error: {e}"
    )

from .data_provider import DataProvider
from ..utils.types import Bar


class FMPDataProvider(DataProvider):
    """
    Data provider that uses the FMP (Financial Modeling Prep) API to fetch market data.
    
    This provider integrates with the FMP package to retrieve historical and real-time
    stock market data for use in trading strategies.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the FMP data provider.
        
        Args:
            api_key: FMP API key. If None, will use FMP_API_KEY environment variable
        """
        self.client = FMPClient(api_key=api_key)
        self.stock_price = StockPrice(self.client)
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical market data for a symbol using FMP API.
        
        Args:
            symbol: The ticker symbol (e.g., 'AAPL')
            start_date: Start date for the data
            end_date: End date for the data (defaults to current date)
            interval: Data interval (ignored for now, FMP returns daily data)
            
        Returns:
            pd.DataFrame: DataFrame containing historical OHLCV data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Format dates for FMP API (YYYY-MM-DD)
        from_date = start_date.strftime('%Y-%m-%d')
        to_date = end_date.strftime('%Y-%m-%d')
        
        try:
            # Get historical data from FMP
            data = self.stock_price.get_historical_price(
                symbol=symbol,
                from_date=from_date,
                to_date=to_date,
                as_dataframe=True
            )
            
            return self.normalize_data(data)
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def get_latest_data(self, symbol: str) -> Union[Bar, Dict[str, Any]]:
        """
        Get the latest data point for a symbol using FMP API.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Union[Bar, Dict[str, Any]]: The latest data point
        """
        try:
            # Get real-time quote
            quote_data = self.stock_price.get_quote(symbol)
            
            # Convert to Bar format if we have OHLCV data
            if isinstance(quote_data, dict) and all(key in quote_data for key in ['price', 'previousClose']):
                return Bar(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    open=float(quote_data.get('previousClose', quote_data['price'])),
                    high=float(quote_data.get('dayHigh', quote_data['price'])),
                    low=float(quote_data.get('dayLow', quote_data['price'])),
                    close=float(quote_data['price']),
                    volume=float(quote_data.get('volume', 0)),
                    metadata=quote_data
                )
            
            return quote_data
            
        except Exception as e:
            raise ValueError(f"Failed to fetch latest data for {symbol}: {str(e)}")
    
    def get_bulk_historical_data(
        self,
        symbols: list,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Get historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date for the data
            end_date: End date for the data
            interval: Data interval
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary mapping symbols to their data
        """
        results = {}
        
        for symbol in symbols:
            try:
                data = self.get_historical_data(symbol, start_date, end_date, interval)
                results[symbol] = data
            except Exception as e:
                print(f"Warning: Failed to fetch data for {symbol}: {str(e)}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols
        
        return results
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize FMP data to a standard format.
        
        Args:
            data: Raw data from FMP
            
        Returns:
            pd.DataFrame: Normalized data with standard column names
        """
        if data.empty:
            return data
        
        # FMP typically returns data with these column names
        # Ensure we have the standard column names (lowercase)
        column_mapping = {
            'Date': 'date',
            'date': 'date',
            'Open': 'open',
            'open': 'open',
            'High': 'high',
            'high': 'high',
            'Low': 'low',
            'low': 'low',
            'Close': 'close',
            'close': 'close',
            'Volume': 'volume',
            'volume': 'volume',
            'Adj Close': 'adj_close',
            'adjClose': 'adj_close'
        }
        
        # Rename columns to standard format
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        # Ensure date column is datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
            data = data.set_index('date')
        
        # Sort by date (oldest first)
        data = data.sort_index()
        
        return data
