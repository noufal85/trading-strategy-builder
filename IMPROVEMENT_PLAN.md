# Trading Strategy Builder - Improvement Plan

**Created Date**: 2025-12-08
**Updated Date**: 2025-12-08

## Executive Summary

The trading-strategy-builder package contains **6,271 lines** across **33 files**. Critical issues include massive duplicate code in standalone_runner.py, embedded HTML templates, and overly long methods in the backtesting engine.

---

## Code Quality Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Total Lines | 6,271 | ~5,200 |
| Largest File | 751 lines | <400 lines |
| Duplicate Code | ~400 lines | 0 |
| Hardcoded Values | 15+ | 0 |
| Max Method Length | 207 lines | <100 lines |

---

## Critical Issues

### 1. Massive HTML Template in Code (751 lines)
**File**: `strategies/trend_following/html_report_generator.py`

**Problems**:
- Lines 99-317: 218 lines of embedded CSS
- Lines 319-374: Chart generation mixed with HTML
- Lines 562-669: 107 lines of f-string HTML for dashboard

**Solution**:
- Extract CSS to `assets/style.css`
- Create HTML templates in `templates/` directory
- Split into `DashboardGenerator`, `StockPageGenerator`, `ChartRenderer`

**Estimated savings**: ~250 lines

### 2. Duplicate Code in standalone_runner.py
**File**: `strategies/trend_following/standalone_runner.py` (430 lines)

**Duplicates from other files**:
- `SimpleTrendAnalyzer` (lines 64-130) - Exists in `trend_analyzer.py`
- `VolumeAnalyzer` (lines 133-179) - Exists in `trend_analyzer.py`
- `FMPDataProvider` (lines 182-215) - Exists in `data/fmp_data_provider.py`
- `YFinanceDataProvider` (lines 218-243) - Exists in `data/yahoo_data_provider.py`
- `analyze_stock()` (lines 259-300) - Duplicates `strategy.py`
- `print_results()` (lines 303-339) - Duplicates strategy.py output

**Root cause**: Created as standalone workaround for import issues

**Solution**: DELETE this file and fix original import issues

**Estimated savings**: ~300 lines

### 3. Failure Analysis Logic in 3 Places
**Locations**:
- `strategy.py` (lines 326-349): `_get_failure_reasons()`
- `html_report_generator.py` (lines 537-560): Identical code
- `standalone_runner.py` (lines 259-300): Inline in `analyze_stock()`

**Solution**: Extract to `analysis/failure_analyzer.py`

**Estimated savings**: ~50 lines

---

## High Priority Refactoring

### 1. Split backtesting/engine.py
**Current**: 582 lines with massive methods

**Problems**:
- `_process_bar()` (lines 175-322): 147 lines, 6 data format attempts
- `_execute_signal()` (lines 368-508): 140 lines

**Solution**: Extract to smaller classes:
- `BarDataNormalizer` - Handle all bar format conversions
- `PositionManager` - Position tracking
- `SignalExecutor` - Trade execution

### 2. Consolidate Configuration
**Current**: 4+ different config locations
- `trend_following/config.py` - DEFAULT_CONFIG (181 lines)
- `scanner_api.py` - API_DEFAULTS (30+ entries)
- `standalone_runner.py` - Another DEFAULT_CONFIG
- `run_daily_scan.py` - More hardcoded defaults

**Solution**: Single `config/base_config.py`:
```python
@dataclass
class TrendFollowingConfig:
    min_slope: float = 0.01
    volume_threshold: float = 1.2
    analysis_days: int = 5
    # ... all other values
```

### 3. Create Output Formatters
**Current duplicates**:
- Console output in `strategy.py` (lines 424-458)
- CSV export in `strategy.py` (lines 460-500)
- Print utilities in `scanner_api.py` (lines 336-387)
- Same logic in `standalone_runner.py` (lines 303-339)

**Solution**: Create `output/formatters.py`:
```python
class ConsoleFormatter:
    def format_results(self, results): ...

class CSVFormatter:
    def export(self, results, filename): ...
```

---

## Medium Priority Improvements

### 1. Remove Duplicate Backtest Runner Scripts
**Files with duplicates**:
- `strategies/moving_average_crossover/run_backtest.py`
- `strategies/moving_average_crossover/run_fmp_backtest.py`
- `strategies/moving_average_crossover/run_alpaca_backtest.py`
- Same files duplicated in `strategies/` root

**Solution**: Keep only `strategies/` versions, delete duplicates in subdirectory

### 2. Create Custom Exception Hierarchy
**File**: `errors/exceptions.py` (NEW)

```python
class StrategyError(Exception):
    pass

class DataFetchError(StrategyError):
    pass

class ConfigurationError(StrategyError):
    pass
```

### 3. Replace print() with logging
**Files affected**: html_report_generator.py, strategy.py, scanner_api.py, standalone_runner.py

**Current**: 30+ instances of `print()` with emoji mixed with `logger`

**Solution**: Standardize on logging with configurable verbosity

---

## Implementation Phases

### Phase 1: Critical (Week 1)
1. DELETE standalone_runner.py - fix imports in main modules
2. Extract failure analysis to shared utility
3. Create emoji/constants module

### Phase 2: High Priority (Week 2)
1. Break down html_report_generator.py
2. Split backtesting/engine.py
3. Consolidate configuration management

### Phase 3: Medium Priority (Week 3)
1. Remove duplicate backtest runners
2. Add comprehensive error handling
3. Create test suite

---

## File Structure After Refactoring

```
trading-strategy-builder/
├── strategy_builder/
│   ├── config/
│   │   └── base_config.py           # NEW - All configs
│   ├── output/
│   │   ├── formatters.py            # NEW - Console, CSV, JSON
│   │   └── templates/               # NEW - HTML templates
│   │       ├── dashboard.html
│   │       └── stock_page.html
│   ├── analysis/
│   │   └── failure_analyzer.py      # NEW - Extracted
│   ├── backtesting/
│   │   ├── engine.py                # SIMPLIFIED
│   │   ├── bar_normalizer.py        # NEW
│   │   └── position_manager.py      # NEW
│   ├── errors/
│   │   └── exceptions.py            # NEW
│   └── strategies/
│       └── trend_following/
│           ├── strategy.py          # SIMPLIFIED
│           ├── html_report_generator.py  # REDUCED to ~300 lines
│           └── standalone_runner.py # DELETED
```

---

## Expected Outcomes

| Improvement | Lines Saved | Benefit |
|-------------|-------------|---------|
| Delete standalone_runner.py | ~300 | Eliminate duplication |
| HTML template extraction | ~250 | Maintainability |
| Failure analysis extraction | ~50 | DRY |
| Backtest consolidation | ~150 | Clarity |
| Configuration consolidation | ~100 | Single source of truth |
| **Total** | **~850** | ~14% reduction |

---

## Anti-Patterns to Fix

1. **Duplicate classes** - SimpleTrendAnalyzer in 2 files
2. **Omnibus classes** - Classes doing 5+ things
3. **God methods** - 140+ line methods
4. **Circular dependencies** - standalone_runner reimplements to avoid imports
5. **Magic strings** - Signal types, exit reasons as strings
6. **Magic numbers** - 252 trading days, thresholds everywhere
7. **Mixed output** - print() and logger used together
8. **Catch-all exceptions** - `except Exception` everywhere
