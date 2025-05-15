"""
Base strategy class for the Strategy Builder framework
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple

from ..utils.types import Signal, Trade, Position


class Strategy(ABC):
    """
    Base class for all trading strategies.
    
    This abstract class defines the interface that all strategies must implement.
    It provides the basic structure for processing market data, generating signals,
    and managing positions.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        """
        Initialize a new strategy.
        
        Args:
            name: A name for the strategy
        """
        self.name = name
        self.positions = []  # List of current positions
        self.trades = []     # List of historical trades
        self.signals = []    # List of generated signals
        
    @abstractmethod
    def on_data(self, data: Dict[str, Any]) -> Optional[Signal]:
        """
        Process new market data and potentially generate a trading signal.
        
        This method is called for each new data point (e.g., bar or tick).
        
        Args:
            data: Dictionary containing market data (e.g., OHLCV)
            
        Returns:
            Optional Signal: A trading signal if one is generated, None otherwise
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """
        Calculate the position size for a given signal.
        
        Args:
            signal: The trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            float: The position size (e.g., number of shares or contracts)
        """
        pass
    
    def on_signal(self, signal: Signal, portfolio: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle a generated signal and potentially execute a trade.
        
        Args:
            signal: The trading signal
            portfolio: Dictionary containing portfolio information
            
        Returns:
            Optional Dict: Trade details if a trade is executed, None otherwise
        """
        # Default implementation - override in subclasses for custom behavior
        self.signals.append(signal)
        return None
    
    def on_trade(self, trade: Trade) -> None:
        """
        Handle a trade execution event.
        
        Args:
            trade: The executed trade
        """
        # Default implementation - override in subclasses for custom behavior
        self.trades.append(trade)
    
    def on_position_update(self, position: Position) -> None:
        """
        Handle a position update event.
        
        Args:
            position: The updated position
        """
        # Default implementation - override in subclasses for custom behavior
        # Update or add the position
        
        # Check if self.positions is a list or a dictionary
        if isinstance(self.positions, list):
            # Handle as a list
            for i, pos in enumerate(self.positions):
                # Check if pos is a dictionary before trying to access it with dictionary-style indexing
                if isinstance(pos, dict) and pos.get('symbol') == position.get('symbol'):
                    self.positions[i] = position
                    return
            self.positions.append(position)
        elif isinstance(self.positions, dict):
            # Handle as a dictionary (using symbol as key)
            symbol = position.get('symbol')
            if symbol:
                self.positions[symbol] = position
        else:
            # Initialize as a list if it's neither
            self.positions = [position]
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Calculate and return performance metrics for the strategy.
        
        Returns:
            Dict: Dictionary containing performance metrics
        """
        # Default implementation with basic metrics
        # Can be overridden in subclasses for more advanced metrics
        
        if not self.trades:
            return {
                'total_trades': 0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'total_return': 0.0
            }
        
        winning_trades = [t for t in self.trades if t['pnl'] > 0]
        losing_trades = [t for t in self.trades if t['pnl'] < 0]
        
        total_profit = sum(t['pnl'] for t in winning_trades) if winning_trades else 0
        total_loss = abs(sum(t['pnl'] for t in losing_trades)) if losing_trades else 0
        
        metrics = {
            'total_trades': len(self.trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(self.trades) if self.trades else 0,
            'profit_factor': total_profit / total_loss if total_loss > 0 else float('inf'),
            'total_return': sum(t['pnl'] for t in self.trades),
            'average_profit': total_profit / len(winning_trades) if winning_trades else 0,
            'average_loss': total_loss / len(losing_trades) if losing_trades else 0
        }
        
        return metrics
