"""
MarketStack data provider for the Strategy Builder framework
"""
from typing import Dict, Any, Optional, Union, List
from datetime import datetime, timedelta
import os
import sys
import pandas as pd
from dotenv import load_dotenv

# Handle imports differently when run as a script vs. imported as a module
if __name__ == "__main__":
    # Add the parent directory to the path so we can import the strategy_builder package
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from strategy_builder.data.data_provider import DataProvider
    from strategy_builder.utils.types import Bar
else:
    # Use relative imports when imported as a module
    from .data_provider import DataProvider
    from ..utils.types import Bar


class MarketstackDataProvider(DataProvider):
    """
    Data provider that fetches data from MarketStack API.
    
    This provider can be used for historical data for backtesting.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        use_https: bool = True,
        use_cache: bool = True,
        cache_ttl: int = 3600,
        dotenv_path: Optional[str] = None
    ):
        """
        Initialize the MarketStack data provider.
        
        Args:
            api_key: MarketStack API key (optional, can be loaded from environment)
            use_https: Whether to use HTTPS for API requests
            use_cache: Whether to cache API requests
            cache_ttl: Time-to-live for cached requests in seconds
            dotenv_path: Path to .env file
        """
        # Try to import the marketstack package
        try:
            from marketstack import MarketstackClient
            from marketstack.exceptions import MarketstackError
        except ImportError:
            raise ImportError(
                "The marketstack package is required for MarketstackDataProvider. "
                "Install it with: pip install git+https://github.com/noufal85/marketstack.git"
            )
        
        self.MarketstackClient = MarketstackClient
        self.MarketstackError = MarketstackError
        
        # Load environment variables from .env file
        # If dotenv_path is None, load from default .env file
        # If dotenv_path is provided, load from that path
        # Only skip loading if dotenv_path is explicitly set to False
        if dotenv_path is not False:
            load_dotenv(dotenv_path)
        
        # Initialize the client
        if api_key:
            self.client = MarketstackClient(
                api_key=api_key,
                use_https=use_https,
                use_cache=use_cache,
                cache_ttl=cache_ttl
            )
        else:
            # Try to load from environment
            # MarketstackClient.from_env() will load from environment variables
            # which should now include any values from the .env file
            self.client = MarketstackClient.from_env()
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical market data from MarketStack.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date for the data
            end_date: End date for the data (defaults to current date)
            interval: Data interval (e.g., "1d" for daily, "1h" for hourly)
                Valid intervals: 1d (daily), 1h (hourly)
            
        Returns:
            pd.DataFrame: DataFrame containing the historical data
        """
        if end_date is None:
            end_date = datetime.now()
        
        # Format dates for MarketStack API
        date_from = start_date.strftime('%Y-%m-%d')
        date_to = end_date.strftime('%Y-%m-%d')
        
        try:
            # For daily data
            if interval == "1d":
                response = self.client.get_eod(
                    symbols=symbol,
                    date_from=date_from,
                    date_to=date_to,
                    limit=1000  # Maximum limit
                )
            # For intraday data
            else:
                # Map interval to MarketStack format
                interval_mapping = {
                    "1m": "1min",
                    "5m": "5min",
                    "15m": "15min",
                    "30m": "30min",
                    "1h": "1hour"
                }
                
                ms_interval = interval_mapping.get(interval, "1hour")
                
                response = self.client.get_intraday(
                    symbols=symbol,
                    date_from=date_from,
                    date_to=date_to,
                    interval=ms_interval,
                    limit=1000  # Maximum limit
                )
            
            # Check if we have data
            if not response or 'data' not in response or not response['data']:
                raise ValueError(f"No data available for symbol: {symbol}")
            
            # Convert to DataFrame
            data = pd.DataFrame(response['data'])
            
            return self.normalize_data(data)
            
        except self.MarketstackError as e:
            raise ValueError(f"Failed to fetch data for {symbol}: {str(e)}")
    
    def get_latest_data(self, symbol: str) -> Union[Bar, Dict[str, Any]]:
        """
        Get the latest data point from MarketStack.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Union[Bar, Dict[str, Any]]: The latest data point
        """
        try:
            # Get the latest EOD data
            response = self.client.get_eod_latest(symbols=symbol)
            
            # Check if we have data
            if not response or 'data' not in response or not response['data']:
                raise ValueError(f"No data available for symbol: {symbol}")
            
            # Get the first data point (should be the only one for a single symbol)
            latest = response['data'][0]
            
            # Convert to Bar format
            return Bar(
                symbol=symbol,
                timestamp=datetime.fromisoformat(latest['date'].replace('Z', '+00:00')),
                open=float(latest['open']),
                high=float(latest['high']),
                low=float(latest['low']),
                close=float(latest['close']),
                volume=float(latest['volume']),
                metadata={
                    'adj_close': latest.get('adj_close', latest['close']),
                    'split_factor': latest.get('split_factor', 1.0),
                    'dividend': latest.get('dividend', 0.0),
                    'exchange': latest.get('exchange', '')
                }
            )
            
        except self.MarketstackError as e:
            # If we can't get the latest data, try getting historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=5)  # Get the last 5 days
            
            data = self.get_historical_data(symbol, start_date, end_date)
            if data.empty:
                raise ValueError(f"No data available for symbol: {symbol}")
            
            latest = data.iloc[-1].to_dict()
            
            return Bar(
                symbol=symbol,
                timestamp=latest.get('date', datetime.now()),
                open=float(latest['open']),
                high=float(latest['high']),
                low=float(latest['low']),
                close=float(latest['close']),
                volume=float(latest['volume']),
                metadata=None
            )
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize MarketStack data to a standard format.
        
        Args:
            data: Raw data from MarketStack
            
        Returns:
            pd.DataFrame: Normalized data
        """
        # Rename columns to standard format if needed
        column_mapping = {
            'date': 'date',
            'symbol': 'symbol',
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
            'adj_close': 'adj_close',
            'split_factor': 'split_factor',
            'dividend': 'dividend',
            'exchange': 'exchange'
        }
        
        # Keep only the columns we need
        columns_to_keep = list(column_mapping.keys())
        data = data[[col for col in columns_to_keep if col in data.columns]]
        
        # Rename columns
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        # Convert date column to datetime if it's not already
        if 'date' in data.columns and not pd.api.types.is_datetime64_any_dtype(data['date']):
            data['date'] = pd.to_datetime(data['date'])
        
        return data
    
    def get_ticker_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about a ticker symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Dict[str, Any]: Information about the ticker
        """
        try:
            response = self.client.get_ticker(symbol)
            
            if not response or 'data' not in response:
                raise ValueError(f"No ticker information available for symbol: {symbol}")
            
            return response['data']
            
        except self.MarketstackError as e:
            raise ValueError(f"Failed to get ticker information for {symbol}: {str(e)}")


# Add a test main block to demonstrate usage when run directly
if __name__ == "__main__":
    print("Testing MarketStack Data Provider")
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    api_key = os.environ.get("MARKETSTACK_API_KEY")
    if not api_key:
        print("Error: MARKETSTACK_API_KEY not found in environment variables or .env file")
        print("Please set the MARKETSTACK_API_KEY environment variable or add it to your .env file")
        sys.exit(1)
    
    # Create a data provider
    provider = MarketstackDataProvider(api_key=api_key)
    
    # Test getting historical data
    try:
        symbol = "AAPL"
        start_date = datetime.now() - timedelta(days=30)  # Last 30 days
        end_date = datetime.now()
        
        print(f"\nGetting historical data for {symbol} from {start_date.date()} to {end_date.date()}...")
        data = provider.get_historical_data(symbol, start_date, end_date)
        
        print(f"Retrieved {len(data)} data points")
        print("\nFirst 5 data points:")
        print(data.head())
        
        # Test getting latest data
        print(f"\nGetting latest data for {symbol}...")
        latest = provider.get_latest_data(symbol)
        print(f"Latest data: {latest}")
        
        print("\nTest completed successfully!")
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)
