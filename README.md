# Trading Strategy Builder

A comprehensive framework for building and backtesting trading strategies.

## Overview

This repository contains a framework for building, testing, and backtesting trading strategies. It allows traders and developers to define custom trading strategies, backtest them against historical data, and evaluate their performance using various metrics.

## Repository Structure

- [Trading Strategy Features](trading_strategy_features.md): Comprehensive feature roadmap organized into stages with checkboxes for tracking implementation progress.
- [Strategy Documentation](strategy_documentation.md): Guidelines for implementing trading strategies, including a sample Moving Average Crossover strategy with pseudo-code implementation logic.

## Implementation Status

- **Current Stage**: Stage 1 - Core Framework
- **Status**: In Progress

## Getting Started

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/noufal85/trading-strategy-builder.git
   cd trading-strategy-builder
   ```

2. Create and activate virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install core dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install FMP package** (choose one option):
   ```bash
   # Option A: From GitHub using SSH (recommended for systems with SSH keys)
   git clone git@github.com:noufal85/fmp.git && cd fmp && pip install . && cd .. && rm -rf fmp
   
   # Option B: From GitHub using PAT token (if SSH not configured)
   pip install git+https://your_token@github.com/noufal85/fmp.git
   
   # Option C: From local directory
   pip install -e /path/to/local/fmp
   ```

5. Install the main package:
   ```bash
   pip install -e .
   ```

6. Setup environment variables:
   ```bash
   cp .env.sample .env
   # Edit .env file with your API keys (especially FMP_API_KEY)
   ```

#### GitHub PAT Token Setup (for private repository)
If you need to install FMP from GitHub private repository:
```bash
# Method 1: Using SSH clone (recommended for systems with SSH keys)
git clone git@github.com:noufal85/fmp.git && cd fmp && pip install . && cd .. && rm -rf fmp

# Method 2: Using git credentials (persistent)
git config --global credential.helper store
echo 'https://YOUR_USERNAME:YOUR_PAT_TOKEN@github.com' >> ~/.git-credentials
```

### Usage

The framework consists of several components:

1. **Strategy Definition**: Create custom trading strategies by inheriting from the `Strategy` base class.
2. **Data Management**: Fetch and manage market data using data providers like `YahooDataProvider` or `AlpacaDataProvider`.
3. **Backtesting Engine**: Backtest strategies against historical data and evaluate performance.

### Environment Variables

The framework supports loading configuration from environment variables using a `.env` file. A sample file `.env.sample` is provided that you can copy and customize:

```
cp .env.sample .env
```

Then edit the `.env` file to add your API keys and other configuration options.

### Running the Sample Strategy

A sample Moving Average Crossover strategy is included in the `strategy_builder/samples` directory. There are several sample scripts to run backtests with different data providers:

#### Using Yahoo Finance Data

```
python -m strategy_builder.samples.run_backtest
```

This will:
1. Fetch historical data for AAPL, MSFT, GOOGL, and AMZN from Yahoo Finance
2. Run a backtest of the Moving Average Crossover strategy
3. Display performance metrics
4. Generate an equity curve plot

#### Using Alpaca Markets Data

```
python -m strategy_builder.samples.run_alpaca_backtest
```

This requires Alpaca API credentials:
1. Sign up for an Alpaca account at https://alpaca.markets/
2. Get your API key and secret from the Alpaca dashboard
3. Set environment variables in your `.env` file:
   ```
   ALPACA_API_KEY=your-api-key
   ALPACA_API_SECRET=your-api-secret
   ```
4. Run the script to fetch data from Alpaca and run the backtest

#### Using FMP Data

```
python -m strategy_builder.samples.run_fmp_backtest
```

This requires an FMP API key:
1. Sign up for an FMP account at https://financialmodelingprep.com/
2. Get your API key from the FMP dashboard
3. Set environment variables in your `.env` file:
   ```
   FMP_API_KEY=your-api-key
   ```
4. Run the script to fetch data from FMP and run the backtest

## Creating Your Own Strategy

To create your own strategy:

1. Create a new Python file in the `strategy_builder/samples` directory
2. Import the necessary components:
   ```python
   from strategy_builder.core.strategy import Strategy
   from strategy_builder.utils.types import Signal
   ```

3. Create a class that inherits from `Strategy` and implement the required methods:
   ```python
   class MyStrategy(Strategy):
       def __init__(self, param1, param2):
           super().__init__(name="My Strategy")
           self.param1 = param1
           self.param2 = param2
           
       def on_data(self, data):
           # Process data and generate signals
           # ...
           return signal
           
       def calculate_position_size(self, signal, portfolio_value):
           # Calculate position size based on risk management rules
           # ...
           return position_size
   ```

4. Create a script to run your strategy:
   ```python
   from strategy_builder.data.yahoo_data_provider import YahooDataProvider
   from strategy_builder.backtesting.engine import BacktestEngine
   from your_strategy_file import MyStrategy
   
   # Create instances
   data_provider = YahooDataProvider()
   strategy = MyStrategy(param1=value1, param2=value2)
   engine = BacktestEngine(strategy, data_provider)
   
   # Run backtest
   results = engine.run(['AAPL', 'MSFT'], start_date, end_date)
   ```

## Periodic Update Process

This repository is updated periodically with the following process:

1. **Strategy Implementation**:
   - When a new strategy is implemented, add it to the `strategy_documentation.md` file
   - Update the status from "In Progress" to "Completed" once testing is successful

2. **Feature Implementation**:
   - Check off completed features in the `trading_strategy_features.md` file
   - Add implementation details and documentation as features are completed

3. **Version Control**:
   - Commit changes with descriptive messages
   - Tag major releases with version numbers
   - Document breaking changes in release notes

4. **Documentation Updates**:
   - Keep documentation in sync with code changes
   - Update examples as the API evolves
   - Document new features and improvements

## Requirements

- Python 3.8+
- NumPy, Pandas
- Matplotlib (for visualization)
- yfinance (for Yahoo Finance data)
- alpaca-trade-api (for Alpaca Markets data)
- python-dotenv (for environment variable management)
- FMP package (for Financial Modeling Prep data - installed separately)

See `requirements.txt` for specific version requirements.
