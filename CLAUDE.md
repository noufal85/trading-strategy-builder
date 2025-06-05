# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python trading strategy framework for building, backtesting, and evaluating trading strategies. The framework uses a modular architecture with separate components for strategies, data providers, backtesting engines, and utilities.

## Key Architecture Components

### Core Framework
- **Strategy Base Class** (`strategy_builder/core/strategy.py`): Abstract base class that all trading strategies inherit from. Implements logging, position management, and performance metrics calculation.
- **Backtesting Engine** (`strategy_builder/backtesting/engine.py`): Handles strategy execution, position tracking, trade execution, and performance calculation during backtests.
- **Data Providers** (`strategy_builder/data/`): Abstract interface with implementations for Yahoo Finance, Alpaca, MarketStack, and FMP data sources.

### Data Flow
1. Data providers fetch historical market data as pandas DataFrames
2. Backtesting engine processes data bar-by-bar, calling strategy.on_data() for each bar
3. Strategies generate signals which the engine executes as trades
4. Engine tracks positions, calculates PnL, and maintains equity curve
5. Results include trade history, performance metrics, and equity curve data

### Strategy Implementation Pattern
All strategies inherit from `Strategy` base class and implement:
- `on_data(data)`: Process market data and return trading signals
- `calculate_position_size(signal, portfolio_value)`: Determine trade size
- Optional: `on_signal()`, `on_trade()`, `on_position_update()` for custom behavior

## Common Development Commands

### Running Backtests
```bash
# Run sample Moving Average Crossover strategy with Yahoo Finance data
python -m strategy_builder.samples.run_backtest

# Run with Alpaca data (requires API credentials in .env)
python -m strategy_builder.strategies.moving_average_crossover.run_alpaca_backtest

# Run with MarketStack data (requires API credentials in .env)
python -m strategy_builder.strategies.moving_average_crossover.run_marketstack_backtest
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Copy environment template and configure API keys
cp .env.sample .env
# Edit .env file with your API credentials
```

### Package Installation
```bash
# Install package in development mode
pip install -e .
```

## Environment Variables
The framework uses `.env` file for configuration. Required variables depend on data provider:
- `ALPACA_API_KEY`, `ALPACA_API_SECRET` for Alpaca data
- `MARKETSTACK_API_KEY` for MarketStack data  
- Optional: `MARKETSTACK_USE_HTTPS`, `MARKETSTACK_USE_CACHE`, `MARKETSTACK_CACHE_TTL`

### Logging Configuration (for Docker deployments)
- `LOGS_DIR`: Directory for log files (default: `logs`)
- `LOG_LEVEL`: Logging level - DEBUG, INFO, WARNING, ERROR (default: `INFO`)
- `LOG_TO_CONSOLE`: Whether to log to console (default: `true`)
- `LOG_TO_FILE`: Whether to log to file (default: `true`)

### Reports Configuration (for Docker deployments)
- `REPORTS_DIR`: Directory for HTML reports (default: `reports`)
- `GENERATE_HTML_REPORTS`: Whether to generate HTML reports by default (default: `false`)

## Logging System
The framework includes comprehensive logging via `StrategyLogger`:
- Logs are written to configurable directory (default: `logs/`, override with `LOGS_DIR` env var)
- Configurable log levels, console/file output, verbosity (configurable via env vars)
- Automatically logs signals, trades, position updates, and backtest results
- Trade data is saved to CSV files for analysis

## File Organization
- `strategy_builder/core/`: Core framework classes (Strategy base class)
- `strategy_builder/data/`: Data provider implementations
- `strategy_builder/backtesting/`: Backtesting engine
- `strategy_builder/strategies/`: Strategy implementations organized by strategy type
- `strategy_builder/samples/`: Sample scripts and basic strategy examples
- `strategy_builder/utils/`: Utilities (logging, types)
- `logs/`: Generated log files and trade CSVs

## Development Notes
- All strategies must inherit from the `Strategy` base class
- Data providers must implement the `DataProvider` interface
- The backtesting engine expects specific data formats and handles position/trade management
- Comprehensive logging is built-in - use the strategy's `self.logger` for custom logging
- Performance metrics are automatically calculated (returns, Sharpe ratio, drawdown, etc.)