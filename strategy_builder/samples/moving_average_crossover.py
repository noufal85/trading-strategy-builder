"""
Sample Moving Average Crossover strategy implementation
"""
from typing import Dict, Any, Optional
from datetime import datetime
import pandas as pd
import numpy as np

from ..core.strategy import Strategy
from ..utils.types import Signal


class MovingAverageCrossover(Strategy):
    """
    Moving Average Crossover Strategy
    
    This strategy generates buy signals when a fast moving average crosses above
    a slow moving average, and sell signals when the fast moving average crosses
    below the slow moving average.
    """
    
    def __init__(
        self,
        fast_period: int = 20,
        slow_period: int = 50,
        risk_per_trade: float = 0.02,
        stop_loss: float = 0.03,
        take_profit: float = 0.06
    ):
        """
        Initialize the Moving Average Crossover strategy.
        
        Args:
            fast_period: Period for the fast moving average
            slow_period: Period for the slow moving average
            risk_per_trade: Percentage of portfolio to risk per trade
            stop_loss: Stop loss percentage
            take_profit: Take profit percentage
        """
        super().__init__(name="Moving Average Crossover")
        
        # Strategy parameters
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.risk_per_trade = risk_per_trade
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        
        # Data storage for moving averages
        self.price_history = {}  # Dict to store price history by symbol
        self.fast_ma = {}        # Dict to store fast MA values by symbol
        self.slow_ma = {}        # Dict to store slow MA values by symbol
        
        # Position tracking
        self.positions = {}      # Dict to store positions by symbol
        self.entry_prices = {}   # Dict to store entry prices by symbol
        self.stop_levels = {}    # Dict to store stop loss levels by symbol
        self.target_levels = {}  # Dict to store take profit levels by symbol
    
    def on_data(self, data: Dict[str, Any]) -> Optional[Signal]:
        """
        Process new market data and generate trading signals.
        
        Args:
            data: Dictionary containing market data
            
        Returns:
            Optional[Signal]: A trading signal if one is generated, None otherwise
        """
        try:
            # Validate input data
            required_fields = ['symbol', 'timestamp', 'close']
            for field in required_fields:
                if field not in data:
                    print(f"Warning: Missing required field '{field}' in data: {data}")
                    return None
            
            symbol = data['symbol']
            timestamp = data['timestamp']
            close_price = data['close']
            
            # Debug info
            print(f"Processing data for {symbol} at {timestamp}: close={close_price}")
            
            # Initialize price history for this symbol if it doesn't exist
            if symbol not in self.price_history:
                self.price_history[symbol] = []
                self.fast_ma[symbol] = None
                self.slow_ma[symbol] = None
                print(f"Initialized price history for {symbol}")
            
            # Add price to history
            self.price_history[symbol].append(close_price)
            
            # Keep only the necessary history
            max_period = max(self.fast_period, self.slow_period)
            if len(self.price_history[symbol]) > max_period + 10:  # Keep a few extra for buffer
                self.price_history[symbol] = self.price_history[symbol][-max_period-10:]
            
            # Calculate moving averages if we have enough data
            if len(self.price_history[symbol]) >= self.slow_period:
                # Calculate fast MA
                fast_ma_value = np.mean(self.price_history[symbol][-self.fast_period:])
                
                # Calculate slow MA
                slow_ma_value = np.mean(self.price_history[symbol][-self.slow_period:])
                
                # Store previous values for crossover detection
                prev_fast_ma = self.fast_ma[symbol]
                prev_slow_ma = self.slow_ma[symbol]
                
                # Update current values
                self.fast_ma[symbol] = fast_ma_value
                self.slow_ma[symbol] = slow_ma_value
                
                # Debug info
                print(f"{symbol} MAs: fast={fast_ma_value:.2f}, slow={slow_ma_value:.2f}")
                
                # Check for crossover if we have previous values
                if prev_fast_ma is not None and prev_slow_ma is not None:
                    # Buy signal: fast MA crosses above slow MA
                    if prev_fast_ma <= prev_slow_ma and fast_ma_value > slow_ma_value:
                        print(f"BUY SIGNAL for {symbol}: fast MA crossed above slow MA")
                        return {
                            'type': 'BUY',
                            'symbol': symbol,
                            'price': close_price,
                            'timestamp': timestamp,
                            'quantity': None,
                            'metadata': {
                                'fast_ma': fast_ma_value,
                                'slow_ma': slow_ma_value
                            }
                        }
                    
                    # Sell signal: fast MA crosses below slow MA
                    elif prev_fast_ma >= prev_slow_ma and fast_ma_value < slow_ma_value:
                        print(f"SELL SIGNAL for {symbol}: fast MA crossed below slow MA")
                        return {
                            'type': 'SELL',
                            'symbol': symbol,
                            'price': close_price,
                            'timestamp': timestamp,
                            'quantity': None,
                            'metadata': {
                                'fast_ma': fast_ma_value,
                                'slow_ma': slow_ma_value
                            }
                        }
            else:
                print(f"Not enough data for {symbol}: {len(self.price_history[symbol])}/{self.slow_period}")
                
        except Exception as e:
            import traceback
            print(f"Error in on_data for {data.get('symbol', 'unknown')}: {str(e)}")
            print(f"Data: {data}")
            print("\nFull error traceback:")
            traceback.print_exc()
            return None
        
        return None
    
    def calculate_position_size(self, signal: Signal, portfolio_value: float) -> float:
        """
        Calculate the position size based on risk management rules.
        
        Args:
            signal: The trading signal
            portfolio_value: Current portfolio value
            
        Returns:
            float: The position size (number of shares)
        """
        symbol = signal['symbol']
        price = signal['price']
        
        if signal['type'] == 'BUY':
            # Calculate stop loss price
            stop_price = price * (1 - self.stop_loss)
            
            # Calculate risk amount
            risk_amount = portfolio_value * self.risk_per_trade
            
            # Calculate position size based on risk
            price_risk = price - stop_price
            if price_risk <= 0:
                return 0
            
            position_size = risk_amount / price_risk
            
            # Store stop loss and take profit levels
            self.stop_levels[symbol] = stop_price
            self.target_levels[symbol] = price * (1 + self.take_profit)
            
            return position_size
        
        return 0
    
    def on_signal(self, signal: Signal, portfolio: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Handle a generated signal and potentially execute a trade.
        
        Args:
            signal: The trading signal
            portfolio: Dictionary containing portfolio information
            
        Returns:
            Optional[Dict[str, Any]]: Trade details if a trade is executed, None otherwise
        """
        super().on_signal(signal, portfolio)
        
        symbol = signal['symbol']
        signal_type = signal['type']
        price = signal['price']
        
        if signal_type == 'BUY':
            # Check if we already have a position
            has_position = False
            if symbol in self.positions:
                if isinstance(self.positions[symbol], dict):
                    has_position = self.positions[symbol].get('quantity', 0) > 0
                else:
                    has_position = self.positions[symbol] > 0
            
            if has_position:
                return None  # Already have a position, don't buy more
            
            # Calculate position size
            position_size = self.calculate_position_size(signal, portfolio['equity'])
            
            # Store entry price
            self.entry_prices[symbol] = price
            
            # Return trade details
            return {
                'action': 'BUY',
                'symbol': symbol,
                'price': price,
                'quantity': position_size,
                'stop_loss': self.stop_levels.get(symbol),
                'take_profit': self.target_levels.get(symbol)
            }
        
        elif signal_type == 'SELL':
            # Check if we have a position to sell
            has_position = False
            position_size = 0
            
            if symbol in self.positions:
                if isinstance(self.positions[symbol], dict):
                    position_size = self.positions[symbol].get('quantity', 0)
                    has_position = position_size > 0
                else:
                    position_size = self.positions[symbol]
                    has_position = position_size > 0
            
            if not has_position:
                return None  # No position to sell
            
            # Clear position tracking
            self.positions[symbol] = 0
            self.entry_prices.pop(symbol, None)
            self.stop_levels.pop(symbol, None)
            self.target_levels.pop(symbol, None)
            
            # Return trade details
            return {
                'action': 'SELL',
                'symbol': symbol,
                'price': price,
                'quantity': position_size
            }
        
        return None
    
    def on_trade(self, trade: Dict[str, Any]) -> None:
        """
        Handle a trade execution event.
        
        Args:
            trade: The executed trade
        """
        super().on_trade(trade)
        
        symbol = trade['symbol']
        trade_type = trade['type']
        quantity = trade['quantity']
        
        # Update position tracking
        if trade_type == 'BUY':
            # Make sure we're starting with a numeric value
            current_position = 0
            if symbol in self.positions:
                if isinstance(self.positions[symbol], dict):
                    current_position = self.positions[symbol].get('quantity', 0)
                elif isinstance(self.positions[symbol], (int, float)):
                    current_position = self.positions[symbol]
            self.positions[symbol] = current_position + quantity
        elif trade_type == 'SELL':
            # Make sure we're starting with a numeric value
            current_position = 0
            if symbol in self.positions:
                if isinstance(self.positions[symbol], dict):
                    current_position = self.positions[symbol].get('quantity', 0)
                elif isinstance(self.positions[symbol], (int, float)):
                    current_position = self.positions[symbol]
            self.positions[symbol] = max(0, current_position - quantity)
