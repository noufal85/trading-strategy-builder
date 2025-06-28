"""
Sample script to run a backtest using FMP data
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Add the parent directory to the path so we can import the strategy_builder package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_builder.data.fmp_data_provider import FMPDataProvider
from strategy_builder.strategies.moving_average_crossover import MovingAverageCrossover  # Changed 'samples' to 'strategies'
from strategy_builder.backtesting.engine import BacktestEngine


def run_fmp_backtest(api_key=None, verbose=False):
    """
    Run a backtest of the Moving Average Crossover strategy using FMP data
    
    Args:
        api_key: FMP API key (optional, can be loaded from environment)
        verbose: Whether to print verbose debug information
    """
    # Load environment variables from .env file if api_key is not provided
    if api_key is None:
        load_dotenv()
        api_key = os.environ.get("FMP_API_KEY")
    
    # Create a data provider
    data_provider = FMPDataProvider(api_key=api_key)
    
    # Create a strategy with logging
    strategy = MovingAverageCrossover(
        fast_period=20,
        slow_period=50,
        risk_per_trade=0.02,
        stop_loss=0.03,
        take_profit=0.06,
        log_level="DEBUG" if verbose else "INFO",
        verbose=verbose
    )
    
    # Create a backtest engine with logging
    engine = BacktestEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_capital=100000.0,
        commission=0.001,  # 0.1% commission
        slippage=0.001,    # 0.1% slippage
        log_level="DEBUG" if verbose else "INFO",
        verbose=verbose
    )
    
    # Define backtest parameters
    symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN']
    start_date = datetime.now() - timedelta(days=365)  # 1 year of data
    end_date = datetime.now()
    
    # Run the backtest
    print(f"Running backtest for {symbols} from {start_date.date()} to {end_date.date()}...")
    try:
        results = engine.run(symbols, start_date, end_date, interval="1d")
    except Exception as e:
        import traceback
        print(f"Error running backtest: {str(e)}")
        print("\nFull error traceback:")
        traceback.print_exc()
        return None
    
    # Print results if available
    if results:
        print("\nBacktest Results:")
        print(f"Initial Capital: ${results['initial_capital']:,.2f}")
        print(f"Final Equity: ${results['final_equity']:,.2f}")
        print(f"Total Return: {results['total_return']*100:.2f}%")
        
        # Print performance metrics
        metrics = results['performance_metrics']
        print("\nPerformance Metrics:")
        print(f"Total Trades: {metrics['total_trades']}")
        print(f"Win Rate: {metrics['win_rate']*100:.2f}%")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        print(f"Annualized Return: {metrics['annualized_return']*100:.2f}%")
        
        # Plot equity curve
        plot_equity_curve(results['equity_curve'])
    
    return results


def plot_equity_curve(equity_curve):
    """
    Plot the equity curve from backtest results
    
    Args:
        equity_curve: List of dictionaries with timestamp and equity values
    """
    # Convert to DataFrame
    df = pd.DataFrame(equity_curve)
    
    # Check if the DataFrame is empty
    if df.empty:
        print("No equity curve data to plot")
        return
    
    # Check if 'timestamp' column exists
    if 'timestamp' not in df.columns:
        print("Warning: 'timestamp' column not found in equity curve data")
        # Plot without timestamp
        plt.figure(figsize=(12, 6))
        plt.plot(df.index, df['equity'])
        plt.title('Equity Curve')
        plt.xlabel('Bar Number')
        plt.ylabel('Equity ($)')
        plt.grid(True)
        plt.tight_layout()
        
        # Save the plot
        plt.savefig('fmp_equity_curve.png')
        print("\nEquity curve saved as 'fmp_equity_curve.png'")
        
        return
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['equity'])
    plt.title('Equity Curve')
    plt.xlabel('Date')
    plt.ylabel('Equity ($)')
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot
    plt.savefig('fmp_equity_curve.png')
    print("\nEquity curve saved as 'fmp_equity_curve.png'")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    import argparse
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run a backtest using FMP data')
    parser.add_argument('--api-key', help='FMP API key')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    args = parser.parse_args()
    
    # Get API key from command line or environment
    api_key = args.api_key
    if not api_key:
        # Load environment variables from .env file
        load_dotenv()
        api_key = os.environ.get("FMP_API_KEY")
    
    # Check if API key is available
    if not api_key:
        print("Please provide an FMP API key using --api-key or set the FMP_API_KEY environment variable")
        print("Example:")
        print("  python -m strategy_builder.samples.run_fmp_backtest --api-key YOUR_API_KEY")
        print("  # OR")
        print("  Add FMP_API_KEY=your-api-key to your .env file")
        print("  # OR")
        print("  export FMP_API_KEY='your-api-key'")
        print("  python -m strategy_builder.samples.run_fmp_backtest")
        sys.exit(1)
    
    # Run the backtest
    run_fmp_backtest(api_key, verbose=args.verbose)