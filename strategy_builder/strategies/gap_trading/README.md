# Gap Trading Strategy

**Created Date**: 2025-12-09
**Updated Date**: 2026-01-08

A gap trading strategy that identifies and trades price gaps at market open, with confirmation-based entries and risk-managed position sizing.

## Strategy Overview

The gap trading strategy:
1. **Identifies gaps** - Detects when a stock opens significantly higher/lower than previous close
2. **Confirms direction** - Waits 10 minutes after open to confirm gap continuation
3. **Enters positions** - Goes long on confirmed gap-ups, short on confirmed gap-downs
4. **Manages risk** - Uses ATR-based stop-losses and risk tier position sizing
5. **Exits at EOD** - All positions closed before market close (intraday only)

## Directory Structure

```
gap_trading/
├── README.md                 # This file
├── __init__.py               # Package exports
├── backtest.py               # Legacy backtest (daily data only)
├── backtest_v2/              # Enhanced backtesting framework
│   ├── __init__.py
│   ├── config.py             # BacktestConfig dataclass
│   ├── data_loader.py        # FMP minute & daily data loader
│   ├── signal_engine.py      # Gap detection & confirmation
│   ├── trade_simulator.py    # Trade execution simulation
│   ├── metrics_engine.py     # Performance metrics calculation
│   ├── backtest_engine.py    # Main orchestrator
│   └── scripts/
│       └── run_backtest.py   # CLI script
├── config/                   # Strategy configuration
├── direction_balancer.py     # LONG/SHORT position balancing (NEW)
├── indicators.py             # Technical indicators (RSI, ADX, etc.)
├── order_manager.py          # Live order execution (Tradier/Alpaca)
├── position_manager.py       # Position tracking
├── position_sizer.py         # Position size calculation
├── realtime_monitor.py       # Live stop-loss monitoring
├── reporting.py              # P&L and performance reports
├── risk_tiers.py             # Symbol risk classification
├── signals.py                # Signal generation
├── universe.py               # Symbol universe management
├── volatility.py             # Volatility calculations
└── tests/
    ├── test_direction_balancer.py  # Direction balancer tests (26 tests)
    ├── test_integration.py
    └── ...
```

## Backtesting Framework (backtest_v2)

The enhanced backtesting framework uses FMP minute data for accurate intraday simulation.

### Features

- **Minute-level precision**: Uses 1-minute bars for confirmation prices and stop-loss simulation
- **Realistic execution**: Simulates slippage and commissions
- **Intraday stop-loss**: Checks stop triggers using minute bars throughout the day
- **Risk tier sizing**: Position sizing based on symbol volatility tiers
- **Parameter sweeps**: Grid search optimization for strategy parameters
- **Comprehensive metrics**: 40+ performance metrics including Sharpe, Sortino, drawdown analysis

### Quick Start

```bash
# Set environment variables
export FMP_API_KEY="your_api_key"

# Or source from .env file
set -a && source .env && set +a

# Run basic backtest
python3 -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
    --start 2025-11-01 --end 2025-12-15 \
    --symbols SPY QQQ AAPL MSFT

# With custom parameters
python3 -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
    --start 2025-11-01 --end 2025-12-15 \
    --symbols SPY QQQ AAPL MSFT \
    --min-gap 0.5 \
    --max-gap 10.0 \
    --confirmation-minutes 10 \
    --stop-multiplier 1.5

# Run parameter sweep
python3 -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
    --start 2025-11-01 --end 2025-12-15 \
    --symbols SPY QQQ AAPL MSFT \
    --sweep

# Analyze gap continuation patterns
python3 -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
    --start 2025-11-01 --end 2025-12-15 \
    --symbols SPY QQQ AAPL MSFT \
    --analyze-gaps

# With detailed file logging
python3 -m strategy_builder.strategies.gap_trading.backtest_v2.scripts.run_backtest \
    --start 2025-11-01 --end 2025-12-15 \
    --symbols SPY QQQ AAPL MSFT \
    --log-file backtest.log \
    --log-trades \
    --verbose
```

### CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--start` | Required | Start date (YYYY-MM-DD) |
| `--end` | Required | End date (YYYY-MM-DD) |
| `--capital` | 100000 | Initial capital |
| `--symbols` | Diversified list | Symbols to backtest |
| `--min-gap` | 1.5 | Minimum gap percentage |
| `--max-gap` | 10.0 | Maximum gap percentage |
| `--confirmation-minutes` | 10 | Minutes after open for confirmation |
| `--stop-multiplier` | 1.5 | ATR multiplier for stops |
| `--no-risk-tiers` | False | Disable risk tier position sizing |
| `--no-minute-data` | False | Use daily data only (faster) |
| `--output` | None | Save results to JSON file |
| `--trades-csv` | None | Save trades to CSV file |
| `--log-file` | None | Log file path for detailed logging |
| `--log-trades` | False | Log detailed information for each trade |
| `--sweep` | False | Run parameter sweep |
| `--analyze-gaps` | False | Run gap continuation analysis |
| `--verbose` | False | Verbose/debug output |

### Output Metrics

The backtest produces comprehensive metrics:

**Returns**
- Total Return, CAGR, Final Equity
- Max Drawdown, Sharpe Ratio, Sortino Ratio

**Trading Statistics**
- Total Trades, Win Rate, Profit Factor
- Average Trade, Average Winner, Average Loser
- Largest Win, Largest Loss

**Exit Analysis**
- Stop-Loss Exits (count and percentage)
- EOD Exits (count and percentage)
- Average Hold Time

**Direction Analysis**
- Long/Short trade counts and win rates
- Long/Short P&L breakdown

**Signal Rejection Analysis**
- GAP_TOO_SMALL: Gaps below minimum threshold
- GAP_TOO_LARGE: Gaps above maximum threshold
- NOT_CONFIRMED: Gaps that reversed during confirmation period
- INSUFFICIENT_DATA: Missing data for analysis

### Example Output

```
============================================================
          GAP TRADING BACKTEST RESULTS
============================================================

RETURNS
  Total Return:     -0.4%
  CAGR:             -3.0%
  Final Equity:     $99,630.01
  Max Drawdown:     0.4%
  Sharpe Ratio:     -9.02
  Sortino Ratio:    -8.29

TRADING STATISTICS
  Total Trades:     6
  Win Rate:         33.3%
  Profit Factor:    0.06
  Avg Trade:        $-61.66
  Avg Winner:       $11.59
  Avg Loser:        $-98.29

EXIT ANALYSIS
  Stop-Loss Exits:  2 (33.3%)
  EOD Exits:        4 (66.7%)
  Avg Hold Time:    299 minutes

Signal Rejection Analysis:
  GAP_TOO_SMALL: 111
  NOT_CONFIRMED: 3
```

### Programmatic Usage

```python
from datetime import date
from strategy_builder.strategies.gap_trading.backtest_v2 import (
    BacktestConfig,
    BacktestEngine,
)

# Create configuration
config = BacktestConfig(
    start_date=date(2025, 11, 1),
    end_date=date(2025, 12, 15),
    initial_capital=100000,
    min_gap_pct=1.0,
    max_gap_pct=10.0,
    confirmation_minutes=10,
    stop_atr_multiplier=1.5,
    use_risk_tiers=True,
    use_minute_data=True,
)

# Run backtest
engine = BacktestEngine(config, api_key="your_fmp_api_key")
result = engine.run(['SPY', 'QQQ', 'AAPL', 'MSFT'])

# Print results
print(result.summary())

# Access metrics
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Win Rate: {result.metrics.win_rate:.1f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")

# Access trades
for trade in result.trades:
    print(f"{trade.symbol}: {trade.direction} {trade.pnl:+.2f}")

# Save results
result.save_to_json("results.json")
result.save_trades_csv("trades.csv")
```

### Parameter Sweep

```python
# Define parameter ranges
param_ranges = {
    'min_gap_pct': [0.5, 1.0, 1.5, 2.0],
    'max_gap_pct': [7.5, 10.0, 15.0],
    'confirmation_minutes': [5, 10, 15, 20],
    'stop_atr_multiplier': [1.0, 1.5, 2.0],
}

# Run sweep
results_df = engine.run_parameter_sweep(param_ranges, symbols)

# Best configuration
best = results_df.sort_values('sharpe_ratio', ascending=False).iloc[0]
print(f"Best Sharpe: {best['sharpe_ratio']:.2f}")
print(f"Min Gap: {best['min_gap_pct']}%")
print(f"Confirmation: {best['confirmation_minutes']} min")
```

## Live Trading Components

### Order Manager

Executes orders through Tradier or Alpaca brokers:

```python
from strategy_builder.strategies.gap_trading.order_manager import OrderManager

order_mgr = OrderManager(broker='tradier')
order = order_mgr.execute_entry(signal, position_size)
```

### Realtime Monitor

Monitors positions for stop-loss triggers:

```python
from strategy_builder.strategies.gap_trading.realtime_monitor import RealtimeStopLossMonitor

monitor = RealtimeStopLossMonitor()
monitor.start()  # Runs until market close
```

### Risk Tiers

Symbols are classified into risk tiers affecting position size:

| Tier | Description | Risk Factor |
|------|-------------|-------------|
| MEGA_CAP | Largest, most stable | 1.0 |
| LARGE_CAP | Large cap stocks | 0.9 |
| MID_CAP | Mid cap stocks | 0.8 |
| SMALL_CAP | Smaller, volatile | 0.6 |
| HIGH_VOLATILITY | High beta stocks | 0.5 |

### Direction Balancer (NEW - 2026-01-08)

The `DirectionBalancer` ensures a diversified mix of LONG and SHORT positions, with dynamic weighting based on SPY's opening behavior relative to ATR.

#### Features

- **Default 50/50 Split**: By default, positions are equally distributed between LONG and SHORT
- **SPY-Based Dynamic Weighting**: Allocation adjusts based on SPY gap/ATR ratio
- **Minimum Direction Guarantee**: Never 0 positions in a direction if signals exist
- **Shortfall Handling**: If not enough signals in one direction, fills from the other
- **Priority Preservation**: Highest priority signals selected within each direction

#### Market Bias Allocations

| Market Bias | SPY Gap/ATR Ratio | Long % | Short % |
|-------------|-------------------|--------|---------|
| Strong Bullish | >= +1.0 | 70% | 30% |
| Bullish | >= +0.5 | 60% | 40% |
| Neutral | -0.5 to +0.5 | 50% | 50% |
| Bearish | <= -0.5 | 40% | 60% |
| Strong Bearish | <= -1.0 | 30% | 70% |

#### Usage

```python
from strategy_builder.strategies.gap_trading.direction_balancer import (
    DirectionBalancer,
    MarketBias,
    BalanceResult,
)

# Create balancer with default config
balancer = DirectionBalancer()

# Or with custom config
config = {
    'enabled': True,
    'default_long_pct': 50,
    'spy_thresholds': {
        'strong_bullish': 1.0,
        'bullish': 0.5,
        'bearish': -0.5,
        'strong_bearish': -1.0
    },
    'allocations': {
        'strong_bullish': {'long': 70, 'short': 30},
        'bullish': {'long': 60, 'short': 40},
        'neutral': {'long': 50, 'short': 50},
        'bearish': {'long': 40, 'short': 60},
        'strong_bearish': {'long': 30, 'short': 70}
    },
    'min_per_direction_pct': 12.5
}
balancer = DirectionBalancer(config)

# Balance signals
result = balancer.balance_signals(
    signals=all_signals,
    max_positions=8,
    spy_gap_pct=1.5,      # SPY gapped up 1.5%
    spy_atr_pct=2.0       # SPY ATR is 2%
)

# Access results
print(f"Market Bias: {result.market_bias}")  # MarketBias.BULLISH
print(f"Selected: {result.long_count} LONG, {result.short_count} SHORT")
print(balancer.get_balance_summary(result))
```

#### Configuration via Airflow Variable

The balancer integrates with the `gap_trading_config` Airflow Variable:

```json
{
  "direction_balancing": {
    "enabled": true,
    "default_long_pct": 50,
    "spy_thresholds": {
      "strong_bullish": 1.0,
      "bullish": 0.5,
      "bearish": -0.5,
      "strong_bearish": -1.0
    },
    "allocations": {
      "strong_bullish": {"long": 70, "short": 30},
      "bullish": {"long": 60, "short": 40},
      "neutral": {"long": 50, "short": 50},
      "bearish": {"long": 40, "short": 60},
      "strong_bearish": {"long": 30, "short": 70}
    },
    "min_per_direction_pct": 12.5
  }
}
```

To disable direction balancing (fall back to pure priority sort):
```json
{
  "direction_balancing": {
    "enabled": false
  }
}
```

## Configuration

Strategy parameters are configured in `config/`:

```yaml
# gap_trading_config.yaml
gap_detection:
  min_gap_pct: 1.5
  max_gap_pct: 10.0

confirmation:
  minutes_after_open: 10

risk_management:
  stop_atr_multiplier: 1.5
  max_position_pct: 0.05
  risk_per_trade: 0.01
```

## Data Requirements

- **FMP API Key**: Required for minute-level data
- **Daily Data**: Historical OHLCV for ATR calculation (20+ days)
- **Minute Data**: Intraday bars for confirmation and stop monitoring

## Dependencies

- `fmp` - Financial Modeling Prep API client
- `pandas` - Data manipulation
- `numpy` - Numerical calculations
- `pyarrow` (optional) - Parquet caching for faster reruns

## Notes

1. **Gap threshold tuning**: Large-cap stocks rarely have 1.5%+ gaps. Consider lowering `--min-gap` to 0.5-1.0% for liquid stocks.

2. **Caching**: Data is cached to `~/.gap_trading_cache/` to avoid repeated API calls. Install `pyarrow` for parquet caching.

3. **API rate limits**: FMP has rate limits. For long backtests, data is fetched progressively.

4. **Market hours**: Strategy only trades during regular market hours (9:30 AM - 4:00 PM ET).
