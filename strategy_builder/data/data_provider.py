"""
Data provider interface and implementations for the Strategy Builder framework
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd

from ..utils.types import Bar, Tick, MarketData


class DataProvider(ABC):
    """
    Abstract base class for data providers.
    
    Data providers are responsible for fetching and providing market data
    to the backtesting engine and strategies.
    """
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical market data for a symbol.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date for the data
            end_date: End date for the data (defaults to current date)
            interval: Data interval (e.g., "1d" for daily, "1h" for hourly)
            
        Returns:
            pd.DataFrame: DataFrame containing the historical data
        """
        pass
    
    @abstractmethod
    def get_latest_data(self, symbol: str) -> Union[Bar, Dict[str, Any]]:
        """
        Get the latest data point for a symbol.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Union[Bar, Dict[str, Any]]: The latest data point
        """
        pass
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data to a standard format.
        
        Args:
            data: Raw data from the provider
            
        Returns:
            pd.DataFrame: Normalized data
        """
        # Default implementation - can be overridden in subclasses
        return data


class CSVDataProvider(DataProvider):
    """
    Data provider that reads data from CSV files.
    """
    
    def __init__(self, data_dir: str):
        """
        Initialize the CSV data provider.
        
        Args:
            data_dir: Directory containing CSV files
        """
        self.data_dir = data_dir
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Get historical market data from a CSV file.
        
        Args:
            symbol: The ticker symbol
            start_date: Start date for the data
            end_date: End date for the data (defaults to current date)
            interval: Data interval (ignored for CSV data)
            
        Returns:
            pd.DataFrame: DataFrame containing the historical data
        """
        # Construct file path
        file_path = f"{self.data_dir}/{symbol}.csv"
        
        # Read CSV file
        try:
            data = pd.read_csv(file_path)
        except FileNotFoundError:
            raise ValueError(f"Data file not found for symbol: {symbol}")
        
        # Convert date column to datetime
        if 'date' in data.columns:
            data['date'] = pd.to_datetime(data['date'])
        elif 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
        else:
            raise ValueError("CSV file must contain a 'date' or 'timestamp' column")
        
        # Filter by date range
        date_col = 'date' if 'date' in data.columns else 'timestamp'
        if end_date is None:
            end_date = datetime.now()
        
        data = data[(data[date_col] >= start_date) & (data[date_col] <= end_date)]
        
        return self.normalize_data(data)
    
    def get_latest_data(self, symbol: str) -> Union[Bar, Dict[str, Any]]:
        """
        Get the latest data point from a CSV file.
        
        Args:
            symbol: The ticker symbol
            
        Returns:
            Union[Bar, Dict[str, Any]]: The latest data point
        """
        data = self.get_historical_data(symbol, datetime.now() - timedelta(days=30))
        if data.empty:
            raise ValueError(f"No data available for symbol: {symbol}")
        
        latest = data.iloc[-1].to_dict()
        
        # Convert to Bar format
        if all(col in latest for col in ['open', 'high', 'low', 'close', 'volume']):
            return Bar(
                symbol=symbol,
                timestamp=latest.get('date', latest.get('timestamp', datetime.now())),
                open=float(latest['open']),
                high=float(latest['high']),
                low=float(latest['low']),
                close=float(latest['close']),
                volume=float(latest['volume']),
                metadata=None
            )
        
        return latest
    
    def normalize_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize CSV data to a standard format.
        
        Args:
            data: Raw data from the CSV file
            
        Returns:
            pd.DataFrame: Normalized data
        """
        # Rename columns to standard format if needed
        column_mapping = {
            'Date': 'date',
            'Timestamp': 'timestamp',
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume'
        }
        
        data = data.rename(columns={k: v for k, v in column_mapping.items() if k in data.columns})
        
        return data
