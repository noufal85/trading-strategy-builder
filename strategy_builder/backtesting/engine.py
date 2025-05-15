"""
Backtesting engine for the Strategy Builder framework
"""
from typing import Dict, Any, List, Optional, Union, Callable
from datetime import datetime
import pandas as pd
import numpy as np
import uuid
import os

from ..core.strategy import Strategy
from ..data.data_provider import DataProvider
from ..utils.types import Signal, Trade, Position, Bar
from ..utils.logger import StrategyLogger


class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(
        self,
        strategy: Strategy,
        data_provider: DataProvider,
        initial_capital: float = 100000.0,
        commission: float = 0.0,
        slippage: float = 0.0,
        log_level: str = "INFO",
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_dir: str = "logs",
        verbose: bool = False
    ):
        """
        Initialize the backtest engine.
        
        Args:
            strategy: The strategy to backtest
            data_provider: The data provider to use
            initial_capital: Initial capital for the backtest
            commission: Commission rate (as a decimal, e.g., 0.001 for 0.1%)
            slippage: Slippage rate (as a decimal, e.g., 0.001 for 0.1%)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_dir: Directory to store log files
            verbose: Whether to log verbose messages
        """
        self.strategy = strategy
        self.data_provider = data_provider
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        
        # Backtest state
        self.portfolio = {
            'cash': initial_capital,
            'equity': initial_capital,
            'positions': {}
        }
        
        self.trades = []
        self.equity_curve = []
        self.performance_metrics = {}
        
        # Initialize logger
        self.logger = StrategyLogger(
            strategy_name=f"Backtest_{strategy.name}",
            level=log_level,
            log_to_console=log_to_console,
            log_to_file=log_to_file,
            log_dir=log_dir,
            verbose=verbose
        )
    
    def run(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: Optional[datetime] = None,
        interval: str = "1d"
    ) -> Dict[str, Any]:
        """
        Run the backtest.
        
        Args:
            symbols: List of symbols to include in the backtest
            start_date: Start date for the backtest
            end_date: End date for the backtest (defaults to current date)
            interval: Data interval (e.g., "1d" for daily, "1h" for hourly)
            
        Returns:
            Dict[str, Any]: Backtest results
        """
        if end_date is None:
            end_date = datetime.now()
        
        self.logger.info(f"Starting backtest for {symbols} from {start_date.date()} to {end_date.date()}")
        self.logger.info(f"Initial capital: ${self.initial_capital:,.2f}, Commission: {self.commission*100:.2f}%, Slippage: {self.slippage*100:.2f}%")
        
        # Get historical data for all symbols
        data = {}
        for symbol in symbols:
            try:
                self.logger.debug(f"Fetching data for {symbol}...")
                symbol_data = self.data_provider.get_historical_data(
                    symbol, start_date, end_date, interval
                )
                data[symbol] = symbol_data
                self.logger.debug(f"Got {len(symbol_data)} data points for {symbol}")
            except Exception as e:
                self.logger.error(f"Error fetching data for {symbol}: {str(e)}")
                continue
        
        if not data:
            error_msg = "No data available for any of the provided symbols"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Combine data and sort by date
        all_data = []
        for symbol, symbol_data in data.items():
            symbol_data['symbol'] = symbol
            all_data.append(symbol_data)
        
        combined_data = pd.concat(all_data).sort_values('date')
        self.logger.info(f"Combined data: {len(combined_data)} bars")
        
        # Initialize backtest state
        self.portfolio = {
            'cash': self.initial_capital,
            'equity': self.initial_capital,
            'positions': {}
        }
        self.trades = []
        self.equity_curve = []
        
        # Run the backtest
        self.logger.info("Running backtest...")
        bar_count = 0
        for _, row in combined_data.iterrows():
            try:
                self._process_bar(row)
                bar_count += 1
                if bar_count % 1000 == 0:
                    self.logger.debug(f"Processed {bar_count} bars")
            except Exception as e:
                import traceback
                self.logger.error(f"Error processing bar: {str(e)}")
                self.logger.error(f"Row type: {type(row)}")
                self.logger.error(f"Row contents: {row}")
                self.logger.error("Full error traceback:")
                traceback.print_exc()
                raise
        
        # Calculate performance metrics
        self.logger.info("Calculating performance metrics...")
        self._calculate_performance_metrics()
        
        # Log results
        result = {
            'initial_capital': self.initial_capital,
            'final_equity': self.portfolio['equity'],
            'total_return': (self.portfolio['equity'] - self.initial_capital) / self.initial_capital,
            'trades': self.trades,
            'equity_curve': self.equity_curve,
            'performance_metrics': self.performance_metrics
        }
        
        self.logger.log_backtest_result(result)
        
        return result
    
    def _process_bar(self, bar_data: pd.Series) -> None:
        """
        Process a single bar of data.
        
        Args:
            bar_data: Bar data as a pandas Series
        """
        # Handle MultiIndex Series (from Yahoo Finance)
        if isinstance(bar_data.index, pd.MultiIndex):
            # Extract symbol from the second level of the MultiIndex
            symbol_values = bar_data.index.get_level_values(1).unique()
            if len(symbol_values) > 0:
                symbol = symbol_values[0]
            else:
                self.logger.warning("Could not determine symbol from MultiIndex")
                return
            
            # Extract data values
            try:
                date_value = None
                open_value = None
                high_value = None
                low_value = None
                close_value = None
                volume_value = None
                
                # Find the date value
                for idx, value in bar_data.items():
                    if idx[0] == 'date':
                        date_value = value
                        break
                
                # Find values for each ticker
                for idx, value in bar_data.items():
                    if idx[1] == symbol:
                        if idx[0] == 'open':
                            open_value = value
                        elif idx[0] == 'high':
                            high_value = value
                        elif idx[0] == 'low':
                            low_value = value
                        elif idx[0] == 'close':
                            close_value = value
                        elif idx[0] == 'volume':
                            volume_value = value
                
                # Check if we have all required values
                if None in [date_value, open_value, high_value, low_value, close_value, volume_value]:
                    self.logger.warning(f"Missing data for {symbol}")
                    return
                
                bar = {
                    'symbol': symbol,
                    'timestamp': date_value,
                    'open': open_value,
                    'high': high_value,
                    'low': low_value,
                    'close': close_value,
                    'volume': volume_value
                }
            except Exception as e:
                self.logger.warning(f"Error extracting data for {symbol}: {str(e)}")
                return
        else:
            # Regular Series
            try:
                # Check if bar_data is a Series with named index or a dictionary-like structure
                if isinstance(bar_data, pd.Series):
                    # Handle Series with named index
                    if 'symbol' in bar_data:
                        symbol = bar_data['symbol']
                    else:
                        # If symbol is not in the data, try to get it from the index
                        symbol = bar_data.name if hasattr(bar_data, 'name') and bar_data.name else None
                        
                        # If we still don't have a symbol, check if it's in the index
                        if symbol is None and isinstance(bar_data.index, pd.Index):
                            for idx_name in bar_data.index.names:
                                if idx_name == 'symbol':
                                    symbol = bar_data.index.get_level_values('symbol')[0]
                                    break
                    
                    # If we still don't have a symbol, we can't process this bar
                    if symbol is None:
                        self.logger.warning("Could not determine symbol from data")
                        self.logger.warning(f"Data: {bar_data}")
                        return
                    
                    # Get timestamp/date
                    timestamp = None
                    if 'timestamp' in bar_data:
                        timestamp = bar_data['timestamp']
                    elif 'date' in bar_data:
                        timestamp = bar_data['date']
                    else:
                        self.logger.warning("No timestamp or date found in data")
                        return
                    
                    # Create bar dictionary with additional error handling
                    try:
                        bar = {
                            'symbol': symbol,
                            'timestamp': timestamp,
                            'open': float(bar_data['open']),
                            'high': float(bar_data['high']),
                            'low': float(bar_data['low']),
                            'close': float(bar_data['close']),
                            'volume': float(bar_data['volume'])
                        }
                    except (TypeError, ValueError) as e:
                        self.logger.warning(f"Error converting data values: {str(e)}")
                        self.logger.warning(f"Data types: open={type(bar_data['open'])}, high={type(bar_data['high'])}, low={type(bar_data['low'])}, close={type(bar_data['close'])}, volume={type(bar_data['volume'])}")
                        
                        # Try alternative access method for pandas Series
                        try:
                            bar = {
                                'symbol': symbol,
                                'timestamp': timestamp,
                                'open': float(bar_data.open),
                                'high': float(bar_data.high),
                                'low': float(bar_data.low),
                                'close': float(bar_data.close),
                                'volume': float(bar_data.volume)
                            }
                        except Exception as e2:
                            self.logger.warning(f"Alternative access method also failed: {str(e2)}")
                            raise ValueError(f"Could not process bar data: {str(e)} / {str(e2)}")
                else:
                    # Dictionary-like structure
                    symbol = bar_data['symbol']
                    
                    bar = {
                        'symbol': symbol,
                        'timestamp': bar_data['date'] if 'date' in bar_data else bar_data['timestamp'],
                        'open': bar_data['open'],
                        'high': bar_data['high'],
                        'low': bar_data['low'],
                        'close': bar_data['close'],
                        'volume': bar_data['volume']
                    }
            except KeyError as e:
                self.logger.warning(f"Missing data: {e}")
                self.logger.warning(f"Available columns: {bar_data.index.tolist() if isinstance(bar_data, pd.Series) else list(bar_data.keys())}")
                return
            except Exception as e:
                self.logger.warning(f"Error processing bar: {str(e)}")
                self.logger.warning(f"Data: {bar_data}")
                return
        
        # Update positions with current prices
        self._update_positions(bar)
        
        # Generate signal from strategy
        signal = self.strategy.on_data(bar)
        
        # Process signal if one was generated
        if signal:
            trade = self._execute_signal(signal, bar)
            if trade:
                self.trades.append(trade)
                self.strategy.on_trade(trade)
        
        # Update equity curve
        equity = self.portfolio['cash']
        for symbol, position in self.portfolio['positions'].items():
            equity += position['quantity'] * position['current_price']
        
        self.portfolio['equity'] = equity
        self.equity_curve.append({
            'timestamp': bar['timestamp'],
            'equity': equity
        })
    
    def _update_positions(self, bar: Dict[str, Any]) -> None:
        """
        Update positions with current prices.
        
        Args:
            bar: Current bar data
        """
        symbol = bar['symbol']
        
        if symbol in self.portfolio['positions']:
            position = self.portfolio['positions'][symbol]
            position['current_price'] = bar['close']
            position['timestamp'] = bar['timestamp']
            position['unrealized_pnl'] = (
                position['current_price'] - position['entry_price']
            ) * position['quantity']
            
            # Update strategy with position
            self.strategy.on_position_update(position)
    
    def _execute_signal(self, signal: Signal, bar: Dict[str, Any]) -> Optional[Trade]:
        """
        Execute a trading signal.
        
        Args:
            signal: The trading signal
            bar: Current bar data
            
        Returns:
            Optional[Trade]: The executed trade, if any
        """
        symbol = signal['symbol']
        signal_type = signal['type']
        
        # Calculate execution price with slippage
        if signal_type == 'BUY':
            execution_price = bar['close'] * (1 + self.slippage)
        else:  # SELL
            execution_price = bar['close'] * (1 - self.slippage)
        
        # Calculate position size
        if 'quantity' in signal and signal['quantity'] is not None:
            quantity = signal['quantity']
        else:
            quantity = self.strategy.calculate_position_size(signal, self.portfolio['equity'])
        
        # Check if we have enough cash for a BUY
        if signal_type == 'BUY':
            cost = quantity * execution_price
            commission_cost = cost * self.commission
            total_cost = cost + commission_cost
            
            if total_cost > self.portfolio['cash']:
                # Adjust quantity if not enough cash
                max_quantity = self.portfolio['cash'] / (execution_price * (1 + self.commission))
                quantity = max_quantity
                cost = quantity * execution_price
                commission_cost = cost * self.commission
                total_cost = cost + commission_cost
        
        # Execute the trade
        trade_id = str(uuid.uuid4())
        timestamp = bar['timestamp']
        
        if signal_type == 'BUY':
            # Check if we already have a position
            if symbol in self.portfolio['positions']:
                # Update existing position
                position = self.portfolio['positions'][symbol]
                old_quantity = position['quantity']
                old_cost = old_quantity * position['entry_price']
                new_cost = quantity * execution_price
                total_cost = old_cost + new_cost
                total_quantity = old_quantity + quantity
                
                # Calculate new average entry price
                position['entry_price'] = total_cost / total_quantity if total_quantity > 0 else 0
                position['quantity'] = total_quantity
                position['current_price'] = bar['close']
                position['timestamp'] = timestamp
                position['unrealized_pnl'] = (
                    position['current_price'] - position['entry_price']
                ) * position['quantity']
            else:
                # Create new position
                self.portfolio['positions'][symbol] = {
                    'symbol': symbol,
                    'quantity': quantity,
                    'entry_price': execution_price,
                    'current_price': bar['close'],
                    'timestamp': timestamp,
                    'unrealized_pnl': 0.0
                }
            
            # Update cash
            cost = quantity * execution_price
            commission_cost = cost * self.commission
            self.portfolio['cash'] -= (cost + commission_cost)
            
            # Create trade record
            trade = {
                'id': trade_id,
                'symbol': symbol,
                'type': 'BUY',
                'price': execution_price,
                'quantity': quantity,
                'timestamp': timestamp,
                'commission': commission_cost,
                'pnl': 0.0
            }
            
            return trade
            
        elif signal_type == 'SELL':
            # Check if we have a position to sell
            if symbol in self.portfolio['positions']:
                position = self.portfolio['positions'][symbol]
                
                # Limit quantity to what we have
                quantity = min(quantity, position['quantity'])
                
                if quantity <= 0:
                    return None
                
                # Calculate PnL
                entry_cost = quantity * position['entry_price']
                exit_value = quantity * execution_price
                pnl = exit_value - entry_cost
                
                # Update position
                position['quantity'] -= quantity
                position['timestamp'] = timestamp
                
                if position['quantity'] <= 0:
                    # Close position
                    del self.portfolio['positions'][symbol]
                else:
                    # Update unrealized PnL
                    position['unrealized_pnl'] = (
                        position['current_price'] - position['entry_price']
                    ) * position['quantity']
                
                # Update cash
                commission_cost = exit_value * self.commission
                self.portfolio['cash'] += (exit_value - commission_cost)
                
                # Create trade record
                trade = {
                    'id': trade_id,
                    'symbol': symbol,
                    'type': 'SELL',
                    'price': execution_price,
                    'quantity': quantity,
                    'timestamp': timestamp,
                    'commission': commission_cost,
                    'pnl': pnl
                }
                
                return trade
        
        return None
    
    def _calculate_performance_metrics(self) -> None:
        """
        Calculate performance metrics for the backtest.
        """
        # Initialize metrics with default values
        self.performance_metrics = {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0
        }
        
        if not self.equity_curve:
            return
        
        # Convert equity curve to DataFrame
        equity_df = pd.DataFrame(self.equity_curve)
        equity_df.set_index('timestamp', inplace=True)
        
        # Calculate returns
        equity_df['return'] = equity_df['equity'].pct_change()
        
        # Calculate metrics
        total_return = (self.portfolio['equity'] - self.initial_capital) / self.initial_capital
        
        # Annualized return
        days = (equity_df.index[-1] - equity_df.index[0]).days
        annualized_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
        # Drawdown
        equity_df['peak'] = equity_df['equity'].cummax()
        equity_df['drawdown'] = (equity_df['equity'] - equity_df['peak']) / equity_df['peak']
        max_drawdown = equity_df['drawdown'].min()
        
        # Sharpe ratio (assuming risk-free rate of 0)
        daily_returns = equity_df['return'].dropna()
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std() if len(daily_returns) > 1 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = daily_returns[daily_returns < 0]
        sortino_ratio = np.sqrt(252) * daily_returns.mean() / downside_returns.std() if len(downside_returns) > 1 else 0
        
        # Win rate
        if self.trades:
            winning_trades = [t for t in self.trades if t['pnl'] > 0]
            win_rate = len(winning_trades) / len(self.trades)
        else:
            win_rate = 0
        
        # Profit factor
        total_profit = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        total_loss = abs(sum(t['pnl'] for t in self.trades if t['pnl'] < 0))
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        
        # Store metrics
        self.performance_metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'total_trades': len(self.trades),
            'winning_trades': len([t for t in self.trades if t['pnl'] > 0]),
            'losing_trades': len([t for t in self.trades if t['pnl'] < 0])
        }
