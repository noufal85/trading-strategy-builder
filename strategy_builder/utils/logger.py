"""
Logger for the Strategy Builder framework
"""
import os
import csv
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Union


class StrategyLogger:
    """
    Logger for strategy builder framework.
    
    This class provides logging capabilities for the strategy builder framework,
    including console logging, file logging, and CSV trade logging.
    """
    
    LEVELS = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    def __init__(
        self,
        strategy_name: str,
        level: str = 'INFO',
        log_to_console: bool = True,
        log_to_file: bool = True,
        log_dir: str = 'logs',
        verbose: bool = False
    ):
        """
        Initialize the logger.
        
        Args:
            strategy_name: Name of the strategy
            level: Log level (DEBUG, INFO, WARNING, ERROR)
            log_to_console: Whether to log to console
            log_to_file: Whether to log to file
            log_dir: Directory to store log files
            verbose: Whether to log verbose messages
        """
        self.strategy_name = strategy_name
        self.level = self.LEVELS.get(level.upper(), logging.INFO)
        self.log_to_console = log_to_console
        self.log_to_file = log_to_file
        self.verbose = verbose
        
        # Create logger
        self.logger = logging.getLogger(f"strategy_builder.{strategy_name}")
        self.logger.setLevel(self.level)
        self.logger.handlers = []  # Clear any existing handlers
        
        # Create formatters
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_formatter = logging.Formatter('%(asctime)s,%(name)s,%(levelname)s,%(message)s')
        
        # Console handler
        if log_to_console:
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.level)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        self.log_file = None
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = os.path.join(log_dir, f"{strategy_name}_{timestamp}.log")
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(self.level)
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
            self.log_file = log_file
        
        # CSV trade log
        self.csv_file = None
        self.csv_writer = None
        self.csv_file_path = None
        if log_to_file:
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            csv_file_path = os.path.join(log_dir, f"{strategy_name}_trades_{timestamp}.csv")
            self.csv_file = open(csv_file_path, 'w', newline='')
            self.csv_writer = csv.writer(self.csv_file)
            self.csv_writer.writerow([
                'Timestamp', 'Symbol', 'Type', 'Price', 'Quantity', 
                'Commission', 'PnL', 'Portfolio Value'
            ])
            self.csv_file_path = csv_file_path
    
    def debug(self, message: str) -> None:
        """
        Log debug message.
        
        Args:
            message: Debug message
        """
        if self.verbose or self.level <= logging.DEBUG:
            self.logger.debug(message)
    
    def info(self, message: str) -> None:
        """
        Log info message.
        
        Args:
            message: Info message
        """
        self.logger.info(message)
    
    def warning(self, message: str) -> None:
        """
        Log warning message.
        
        Args:
            message: Warning message
        """
        self.logger.warning(message)
    
    def error(self, message: str) -> None:
        """
        Log error message.
        
        Args:
            message: Error message
        """
        self.logger.error(message)
    
    def log_trade(self, trade: Dict[str, Any], portfolio_value: float) -> None:
        """
        Log trade to CSV file.
        
        Args:
            trade: Trade details
            portfolio_value: Current portfolio value
        """
        if self.csv_writer:
            self.csv_writer.writerow([
                trade['timestamp'],
                trade['symbol'],
                trade['type'],
                trade['price'],
                trade['quantity'],
                trade['commission'],
                trade.get('pnl', 0),
                portfolio_value
            ])
            self.csv_file.flush()  # Ensure it's written immediately
        
        # Also log to regular log
        trade_type = trade['type']
        symbol = trade['symbol']
        price = trade['price']
        quantity = trade['quantity']
        pnl = trade.get('pnl', 0)
        
        if trade_type == 'BUY':
            self.info(f"OPENED POSITION: {symbol} - {quantity:.2f} shares at ${price:.2f}")
        else:  # SELL
            self.info(f"CLOSED POSITION: {symbol} - {quantity:.2f} shares at ${price:.2f} - PnL: ${pnl:.2f}")
    
    def log_position_update(self, position: Dict[str, Any]) -> None:
        """
        Log position update.
        
        Args:
            position: Position details
        """
        if self.verbose:
            symbol = position['symbol']
            quantity = position['quantity']
            current_price = position['current_price']
            unrealized_pnl = position.get('unrealized_pnl', 0)
            
            self.debug(f"Position update: {symbol} - {quantity:.2f} shares at ${current_price:.2f} - Unrealized PnL: ${unrealized_pnl:.2f}")
    
    def log_signal(self, signal: Dict[str, Any]) -> None:
        """
        Log trading signal.
        
        Args:
            signal: Signal details
        """
        signal_type = signal['type']
        symbol = signal['symbol']
        price = signal['price']
        
        self.info(f"SIGNAL: {signal_type} {symbol} at ${price:.2f}")
    
    def log_backtest_result(self, result: Dict[str, Any]) -> None:
        """
        Log backtest result.
        
        Args:
            result: Backtest result
        """
        initial_capital = result['initial_capital']
        final_equity = result['final_equity']
        total_return = result['total_return']
        metrics = result['performance_metrics']
        
        self.info(f"BACKTEST RESULT: Initial: ${initial_capital:.2f}, Final: ${final_equity:.2f}, Return: {total_return*100:.2f}%")
        self.info(f"Win Rate: {metrics['win_rate']*100:.2f}%, Profit Factor: {metrics['profit_factor']:.2f}")
        self.info(f"Sharpe: {metrics['sharpe_ratio']:.2f}, Sortino: {metrics['sortino_ratio']:.2f}, Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        # Log to CSV file
        if self.csv_file_path:
            result_csv_path = self.csv_file_path.replace('_trades_', '_results_')
            with open(result_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Initial Capital', initial_capital])
                writer.writerow(['Final Equity', final_equity])
                writer.writerow(['Total Return', f"{total_return*100:.2f}%"])
                writer.writerow(['Total Trades', metrics['total_trades']])
                writer.writerow(['Winning Trades', metrics['winning_trades']])
                writer.writerow(['Losing Trades', metrics['losing_trades']])
                writer.writerow(['Win Rate', f"{metrics['win_rate']*100:.2f}%"])
                writer.writerow(['Profit Factor', metrics['profit_factor']])
                writer.writerow(['Sharpe Ratio', metrics['sharpe_ratio']])
                writer.writerow(['Sortino Ratio', metrics['sortino_ratio']])
                writer.writerow(['Max Drawdown', f"{metrics['max_drawdown']*100:.2f}%"])
                writer.writerow(['Annualized Return', f"{metrics['annualized_return']*100:.2f}%"])
    
    def close(self) -> None:
        """
        Close the logger and any open files.
        """
        if self.csv_file:
            self.csv_file.close()
