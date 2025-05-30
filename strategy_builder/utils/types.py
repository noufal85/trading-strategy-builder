"""
Type definitions for the Strategy Builder framework
"""
from typing import Dict, Any, TypedDict, Literal, Union, Optional
from datetime import datetime


class Signal(TypedDict):
    """
    Trading signal generated by a strategy
    """
    type: Literal['BUY', 'SELL']
    symbol: str
    price: float
    timestamp: datetime
    quantity: Optional[float]
    metadata: Optional[Dict[str, Any]]


class Trade(TypedDict):
    """
    Executed trade
    """
    id: str
    symbol: str
    type: Literal['BUY', 'SELL']
    price: float
    quantity: float
    timestamp: datetime
    commission: float
    pnl: Optional[float]
    metadata: Optional[Dict[str, Any]]


class Position(TypedDict):
    """
    Current position
    """
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    timestamp: datetime
    unrealized_pnl: float
    metadata: Optional[Dict[str, Any]]


class Bar(TypedDict):
    """
    OHLCV bar data
    """
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Optional[Dict[str, Any]]


class Tick(TypedDict):
    """
    Tick data
    """
    symbol: str
    timestamp: datetime
    price: float
    volume: float
    metadata: Optional[Dict[str, Any]]


# Define MarketData as a union of possible data types
MarketData = Union[Bar, Tick, Dict[str, Any]]
