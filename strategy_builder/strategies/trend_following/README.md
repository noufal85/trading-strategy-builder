# Trend Following Strategy

A comprehensive trend following strategy implementation that screens stocks for upward price trends combined with above-average volume patterns. This strategy is designed to run daily before market open to identify stocks that meet specific trending criteria.

## Features

- **Pluggable Trend Analysis**: Multiple trend detection methods (Simple and Linear Regression)
- **Volume Analysis**: Screens for above-average volume patterns
- **Configurable Parameters**: Fully customizable analysis parameters
- **FMP Integration**: Uses your personal FMP package for reliable market data
- **Multiple Output Formats**: Console and CSV output options
- **Comprehensive Logging**: Detailed logging and analysis results

## Installation

1. **Install the FMP package** (your personal repository):
```bash
pip install git+https://github.com/noufal85/fmp.git
```

2. **Install other dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set up your FMP API key**:
```bash
export FMP_API_KEY=your_api_key_here
```

Or create a `.env` file:
```
FMP_API_KEY=your_api_key_here
```

## Quick Start

### Basic Usage

```python
from strategy_builder.strategies.trend_following import TrendFollowingStrategy

# Initialize with default settings
strategy = TrendFollowingStrategy()

# Run daily screening
results = strategy.run_daily_screening()

# Process results
for result in results:
    if result['qualifies']:
        print(f"{result['symbol']}: ${result['current_price']:.2f}")
```

### Command Line Usage

```bash
# Basic screening with default settings
python run_daily_scan.py

# Screen specific stocks
python run_daily_scan.py --stocks AAPL,MSFT,GOOGL,AMZN,TSLA

# Use linear regression trend analysis
python run_daily_scan.py --trend-method linear_regression

# Custom parameters
python run_daily_scan.py --analysis-days 7 --volume-threshold 1.5 --output both

# Dry run to see configuration
python run_daily_scan.py --dry-run
```

## Configuration Options

### Default Configuration

```python
DEFAULT_CONFIG = {
    'analysis_days': 5,              # Days to analyze for trend
    'volume_avg_days': 20,           # Days for volume average baseline
    'volume_threshold': 1.2,         # Volume must be 120% above average
    'trend_method': 'simple',        # 'simple' or 'linear_regression'
    'min_slope': 0.01,              # Minimum price slope (1% per day)
    'min_positive_days': 3,         # Minimum positive days required
    'ma_period': 5,                 # Moving average period
    'output_format': 'console',      # 'console', 'csv', or 'both'
    'detailed_output': True,         # Include detailed metrics
}
```

### Custom Configuration

```python
custom_config = {
    'analysis_days': 7,
    'volume_threshold': 1.5,         # Higher volume requirement
    'trend_method': 'linear_regression',
    'output_format': 'both',
    'detailed_output': True
}

strategy = TrendFollowingStrategy(config=custom_config)
```

## Trend Analysis Methods

### 1. Simple Trend Analyzer (Default)

Uses basic momentum indicators:
- **Price Slope**: Linear regression slope over analysis period
- **Positive Days**: Count of positive price movement days
- **Moving Average**: Current price vs moving average

**Configuration**:
```python
{
    'trend_method': 'simple',
    'min_slope': 0.01,              # 1% minimum daily slope
    'min_positive_days': 3,         # 3 out of 5 days positive
    'ma_period': 5                  # 5-day moving average
}
```

### 2. Linear Regression Trend Analyzer

Uses statistical regression analysis:
- **Linear Regression**: Fit line to recent price data
- **R-squared**: Measure trend reliability
- **Slope Analysis**: Statistical slope significance

**Configuration**:
```python
{
    'trend_method': 'linear_regression',
    'linear_min_slope': 0.02,       # 2% minimum daily slope
    'min_r_squared': 0.7            # 70% minimum R-squared
}
```

## Volume Analysis

The strategy analyzes volume patterns to ensure institutional interest:

- **Average Volume**: Calculate baseline over 20 days (configurable)
- **Recent Volume**: Compare last N days vs baseline
- **Volume Threshold**: Require volume above threshold (default 120%)

## Stock Lists

### Popular Stocks (Default)
25 popular stocks across sectors:
- **Technology**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, etc.
- **Finance**: JPM, BAC, WFC, GS, MS
- **Healthcare**: JNJ, PFE, UNH, ABBV, MRK
- **Consumer**: KO, PEP, WMT, HD, MCD
- **Industrial**: BA, CAT, GE, MMM, UPS

### Extended Stocks
50 stocks including additional sectors:
- **Energy**: XOM, CVX, COP, EOG, SLB
- **Utilities**: NEE, DUK, SO, D, EXC

### Custom Stock Lists
```python
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
strategy = TrendFollowingStrategy(stock_list=tech_stocks)
```

## Command Line Options

```bash
# Configuration
--config CONFIG_FILE          # JSON configuration file
--stocks SYMBOL1,SYMBOL2      # Comma-separated stock list
--stock-list popular|extended # Predefined stock list

# Analysis Parameters
--analysis-days N             # Days to analyze (default: 5)
--volume-avg-days N          # Volume baseline days (default: 20)
--volume-threshold X.X       # Volume threshold multiplier (default: 1.2)

# Trend Detection
--trend-method simple|linear_regression  # Trend analysis method
--min-slope X.XX             # Minimum price slope
--min-positive-days N        # Minimum positive days

# Output Options
--output console|csv|both    # Output format
--output-file FILENAME       # CSV output filename
--detailed-output           # Include detailed metrics

# Other Options
--log-level DEBUG|INFO|WARNING|ERROR
--api-key API_KEY           # FMP API key
--dry-run                   # Show configuration only
```

## Example Outputs

### Console Output
```
================================================================================
TREND FOLLOWING SCREENING RESULTS - 2025-01-15 09:30:00
================================================================================
Found 3 qualifying stocks:

🔥 AAPL - $185.50
   Trend Strength: 0.85 | Volume Strength: 0.92
   Price Slope: 0.023 (2.3%/day)
   Positive Days: 4/4
   Volume Ratio: 1.84x average
   Current Volume: 89,234,567

🔥 MSFT - $420.75
   Trend Strength: 0.78 | Volume Strength: 0.88
   Price Slope: 0.018 (1.8%/day)
   Positive Days: 3/4
   Volume Ratio: 1.76x average
   Current Volume: 45,123,890

🔥 NVDA - $875.25
   Trend Strength: 0.92 | Volume Strength: 0.95
   Price Slope: 0.035 (3.5%/day)
   Positive Days: 5/4
   Volume Ratio: 1.90x average
   Current Volume: 78,456,123

================================================================================
```

### CSV Output
Results are saved with detailed metrics for further analysis:
- Symbol, price, volume data
- Trend analysis metrics
- Volume analysis details
- Date ranges and data points

## Integration with Schedulers

### Windows Task Scheduler
Create a batch file to run the scanner:
```batch
@echo off
cd /d "C:\path\to\strategy_builder\strategies\trend_following"
python run_daily_scan.py --output both
```

### Linux/macOS Cron
Add to crontab for daily execution:
```bash
# Run at 8:30 AM Eastern (before market open)
30 8 * * 1-5 cd /path/to/strategy_builder/strategies/trend_following && python run_daily_scan.py --output both
```

### Python Scheduler
```python
import schedule
import time

def run_daily_scan():
    os.system("python run_daily_scan.py --output both")

schedule.every().monday.at("08:30").do(run_daily_scan)
schedule.every().tuesday.at("08:30").do(run_daily_scan)
schedule.every().wednesday.at("08:30").do(run_daily_scan)
schedule.every().thursday.at("08:30").do(run_daily_scan)
schedule.every().friday.at("08:30").do(run_daily_scan)

while True:
    schedule.run_pending()
    time.sleep(60)
```

## Error Handling

The strategy includes comprehensive error handling:
- **Missing Data**: Gracefully handles stocks with insufficient data
- **API Errors**: Continues processing other stocks if one fails
- **Network Issues**: Retries and logs connection problems
- **Invalid Configurations**: Validates and corrects configuration parameters

## Logging

Detailed logging is provided at multiple levels:
- **INFO**: Overall progress and qualifying stocks
- **DEBUG**: Detailed analysis for each stock
- **WARNING**: Missing data or API issues
- **ERROR**: Critical failures

## File Structure

```
trend_following/
├── __init__.py                 # Package initialization
├── config.py                   # Configuration and stock lists
├── trend_analyzer.py           # Pluggable trend analysis methods
├── strategy.py                 # Main strategy implementation
├── run_daily_scan.py          # Command-line entry point
├── example.py                 # Usage examples
└── README.md                  # This documentation
```

## Troubleshooting

### Common Issues

1. **"FMP package not found"**
   - Install the FMP package: `pip install git+https://github.com/noufal85/fmp.git`

2. **"FMP API key is required"**
   - Set the environment variable: `export FMP_API_KEY=your_key`

3. **"No data available for symbol"**
   - Check if the symbol is valid and trading
   - Verify your FMP API key has sufficient quota

4. **"Insufficient data for analysis"**
   - Reduce `analysis_days` or `volume_avg_days` for newer stocks
   - Check if the stock has enough trading history

### Debug Mode
```bash
python run_daily_scan.py --log-level DEBUG --stocks AAPL
```

## Contributing

The trend analysis system is designed to be extensible. To add new trend analyzers:

1. Create a new class inheriting from `TrendAnalyzer`
2. Implement the `analyze_trend` method
3. Add configuration options in `config.py`
4. Update the strategy to use your new analyzer

## License

This strategy is part of the Strategy Builder framework and follows the same licensing terms.
