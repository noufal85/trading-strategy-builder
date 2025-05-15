"""
Sample script to run a backtest using Alpaca data
"""
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Add the parent directory to the path so we can import the strategy_builder package
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from strategy_builder.data.alpaca_data_provider import AlpacaDataProvider
from strategy_builder.samples.moving_average_crossover import MovingAverageCrossover
from strategy_builder.backtesting.engine import BacktestEngine


def run_alpaca_backtest(api_key, api_secret):
    """
    Run a backtest of the Moving Average Crossover strategy using Alpaca data
    
    Args:
        api_key: Alpaca API key
        api_secret: Alpaca API secret
    """
    # Create a data provider
    data_provider = AlpacaDataProvider(
        api_key=api_key,
        api_secret=api_secret,
        base_url="https://paper-api.alpaca.markets",  # Use paper trading API
        data_feed="iex"  # Use IEX data feed (free)
    )
    
    # Create a strategy
    strategy = MovingAverageCrossover(
        fast_period=20,
        slow_period=50,
        risk_per_trade=0.02,
        stop_loss=0.03,
        take_profit=0.06
    )
    
    # Create a backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        data_provider=data_provider,
        initial_capital=100000.0,
        commission=0.001,  # 0.1% commission
        slippage=0.001     # 0.1% slippage
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
        print(f"Error running backtest: {str(e)}")
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
        plt.savefig('alpaca_equity_curve.png')
        print("\nEquity curve saved as 'alpaca_equity_curve.png'")
        
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
    plt.savefig('alpaca_equity_curve.png')
    print("\nEquity curve saved as 'alpaca_equity_curve.png'")
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    # Check if API key and secret are provided as environment variables
    api_key = os.environ.get("ALPACA_API_KEY")
    api_secret = os.environ.get("ALPACA_API_SECRET")
    
    if not api_key or not api_secret:
        print("Please set ALPACA_API_KEY and ALPACA_API_SECRET environment variables")
        print("Example:")
        print("  export ALPACA_API_KEY='your-api-key'")
        print("  export ALPACA_API_SECRET='your-api-secret'")
        sys.exit(1)
    
    run_alpaca_backtest(api_key, api_secret)
