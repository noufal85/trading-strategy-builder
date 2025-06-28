# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Python trading strategy framework for building, backtesting, and evaluating trading strategies. The framework uses a modular architecture with separate components for strategies, data providers, backtesting engines, and utilities.

## Key Architecture Components

### Core Framework
- **Strategy Base Class** (`strategy_builder/core/strategy.py`): Abstract base class that all trading strategies inherit from. Implements logging, position management, and performance metrics calculation.
- **Backtesting Engine** (`strategy_builder/backtesting/engine.py`): Handles strategy execution, position tracking, trade execution, and performance calculation during backtests.
- **Data Providers** (`strategy_builder/data/`): Abstract interface with implementations for Yahoo Finance, Alpaca, FMP, and CSV data sources.

### Available Data Providers
- **YahooDataProvider**: Free, no API key required, uses yfinance library
- **AlpacaDataProvider**: Requires API key/secret, provides real-time and historical data
- **FMPDataProvider**: Financial Modeling Prep API (requires separate FMP package installation)
- **CSVDataProvider**: Load data from local CSV files for testing

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

# Run with specific data providers (from strategy implementations)
python -m strategy_builder.strategies.moving_average_crossover.run_alpaca_backtest
python -m strategy_builder.strategies.moving_average_crossover.run_fmp_backtest

# Run from samples directory with different providers
python -m strategy_builder.samples.run_alpaca_backtest
python -m strategy_builder.samples.run_fmp_backtest
```

### Environment Setup
```bash
# 1. Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 2. Install core dependencies
pip install -r requirements.txt

# 3. Install FMP package manually (choose one option):

# Option A: From GitHub using SSH (recommended for systems with SSH keys)
git clone git@github.com:noufal85/fmp.git && cd fmp && pip install . && cd .. && rm -rf fmp

# Option A2: From GitHub using PAT token (if SSH not configured)
pip install git+https://your_token@github.com/noufal85/fmp.git

# Option B: From local directory (for development environments)
pip install -e /home/noufal/automation/fmp

# Option C: From any local path
pip install -e /path/to/your/fmp

# 4. Install the main package in development mode
pip install -e .

# 5. Copy environment template and configure API keys
cp .env.sample .env
# Edit .env file with your API credentials (especially FMP_API_KEY)
```

### GitHub PAT Token Setup (for private repository installation)
If installing FMP from GitHub private repository, you have several authentication options:
```bash
# Method 1: Using SSH clone (recommended for systems with SSH keys)
git clone git@github.com:noufal85/fmp.git && cd fmp && pip install . && cd .. && rm -rf fmp

# Method 2: Using pip with PAT token directly
pip install git+https://your_token@github.com/noufal85/fmp.git

# Method 3: Using git credentials (persistent)
git config --global credential.helper store
echo 'https://YOUR_USERNAME:YOUR_PAT_TOKEN@github.com' >> ~/.git-credentials
```

### Package Installation
```bash
# Install package in development mode
pip install -e .
```

## Environment Variables
The framework uses `.env` file for configuration. Required variables depend on data provider:
- `FMP_API_KEY` for FMP data provider (Financial Modeling Prep)
- `ALPACA_API_KEY`, `ALPACA_API_SECRET` for Alpaca data

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

## Testing and Code Quality
Currently, this project does not have:
- Unit tests or test infrastructure (no pytest configuration)
- Linting or code formatting tools (no flake8, black, ruff configuration)
- CI/CD pipeline or GitHub Actions

When implementing tests or linting in the future, follow standard Python practices.

## Additional Project Resources
- **README.md**: Getting started guide with installation instructions
- **strategy_documentation.md**: Strategy implementation guidelines and examples
- **trading_strategy_features.md**: Feature roadmap with implementation status
- **strategy_builder/strategies/trend_following/README.md**: Trend following strategy documentation

## Import Issue Prevention & Resolution

### Common Import Problems in Containerized Environments

This project has complex dependencies that can cause import issues, especially in Docker/Airflow environments. Follow these guidelines to prevent and resolve import problems:

#### 1. Circular Import Detection and Resolution

**Problem**: Package `__init__.py` files importing from modules that don't exist or have their own import dependencies.

**Solution**:
```python
# ❌ Problematic: Direct import in __init__.py that can cause circular dependencies
from .module_with_complex_dependencies import ComplexClass

# ✅ Better: Comment out problematic imports and import directly when needed
# from .module_with_complex_dependencies import ComplexClass

# ✅ Alternative: Use conditional imports or lazy loading
try:
    from .module_with_complex_dependencies import ComplexClass
except ImportError:
    ComplexClass = None
```

#### 2. Missing Function/Class Resolution

**Problem**: `ImportError: cannot import name 'function_name' from 'module'`

**Diagnosis Steps**:
1. **Check if file exists**: Verify the target file exists and contains the expected function/class
2. **Check file contents**: Ensure the file isn't empty or missing the expected exports
3. **Verify import path**: Confirm the import path matches the actual file structure

**Example Fix**:
```bash
# Problem: from .run_backtest import run_ma_crossover_backtest
# Solution: Create the missing function in run_backtest.py

def run_ma_crossover_backtest():
    """Minimal function to satisfy import requirements"""
    return {"status": "placeholder"}
```

#### 3. Container vs Local Code Sync Issues

**Problem**: Local changes not reflected in Docker container due to caching or volume mount issues.

**Resolution Steps**:
1. **Restart containers**: `docker-compose restart`
2. **Rebuild with no cache**: `docker-compose build --no-cache`
3. **Verify volume mounts**: Check docker-compose.yml volume mappings
4. **Check file timestamps**: Ensure local changes are newer than container builds

#### 4. Dependency Chain Import Strategy

**When imports fail due to complex dependency chains**:

**Option A: Bypass Package Structure (Recommended for complex cases)**
```python
# Use importlib.util.spec_from_file_location for direct file imports
import importlib.util
spec = importlib.util.spec_from_file_location("module", "/path/to/file.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
TargetClass = module.TargetClass
```

**Option B: Minimal Package Imports**
```python
# Temporarily comment out problematic imports in __init__.py files
# from .complex_dependency import ComplexClass  # Commented out
from .simple_module import SimpleFunction  # Keep only working imports
```

**Option C: Create Import-Safe Alternatives**
```python
# Create minimal placeholder functions that satisfy import requirements
def complex_function():
    """Minimal placeholder to resolve import dependencies"""
    return {"status": "placeholder", "message": "Simplified for import compatibility"}
```

#### 5. External Package Dependency Issues

**Problem**: External packages (like FMP) with their own import issues affecting the project.

**Resolution**:
1. **Fix external package locally** first
2. **Test imports in isolation**:
   ```bash
   docker exec container python -c "from external_package import TargetClass; print('Success')"
   ```
3. **Push external package fixes** to version control
4. **Rebuild containers** to get updated external packages

#### 6. Best Practices for Import-Safe Code

**Package Structure Guidelines**:
- Keep `__init__.py` files minimal with only essential imports
- Use conditional imports for optional dependencies
- Create placeholder functions for complex imports during development
- Test imports in isolation before adding to complex dependency chains

**Development Workflow**:
1. **Test imports locally** before containerizing
2. **Use simple imports** during initial development
3. **Add complex dependencies** incrementally
4. **Document import dependencies** in CLAUDE.md
5. **Create import debug scripts** for troubleshooting

**Container Development**:
- Always test imports in fresh container environments
- Use `docker exec container python -c "import test"` for quick verification
- Rebuild containers after significant import structure changes
- Keep external dependencies updated and in sync between local/container environments

#### 7. Troubleshooting Import Issues

**Debugging Commands**:
```bash
# Test specific import in container
docker exec container python -c "from package.module import TargetClass; print('Import works')"

# Check Python path in container
docker exec container python -c "import sys; print('\n'.join(sys.path))"

# Verify file exists in container
docker exec container ls -la /path/to/file.py

# Check if function exists in file
docker exec container python -c "import ast; print([n.name for n in ast.parse(open('/path/file.py').read()).body if isinstance(n, ast.FunctionDef)])"
```

**Common Error Patterns and Fixes**:
- `cannot import name 'X' from 'Y'` → Check if X exists in Y, create placeholder if needed
- `No module named 'package'` → Verify package installation and Python path
- `FMP package not found` → External dependency issue, rebuild with updated packages
- `circular import detected` → Use conditional imports or importlib direct file loading

This systematic approach prevents most import issues and provides clear resolution paths when they occur.