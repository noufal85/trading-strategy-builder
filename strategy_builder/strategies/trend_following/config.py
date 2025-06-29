"""
Configuration settings for the trend following strategy.
"""

from typing import List, Dict, Any


# Default configuration for the trend following strategy
DEFAULT_CONFIG = {
    # Configuration identification
    'config_name': 'default',       # Human-readable name for this configuration
    
    # Data analysis parameters
    'analysis_days': 5,              # Number of recent days to analyze for trend
    'volume_avg_days': 20,           # Days to calculate average volume baseline
    'volume_threshold': 1.2,         # Volume must be 120% above average
    
    # Trend detection parameters
    'trend_method': 'simple',        # 'simple' or 'linear_regression'
    'min_slope': 0.01,              # Minimum price slope for uptrend (1% per day)
    'min_positive_days': 3,         # Minimum positive days out of analysis_days
    'ma_period': 5,                 # Moving average period for trend confirmation
    
    # Linear regression specific parameters (if using linear_regression method)
    'min_r_squared': 0.7,           # Minimum R-squared for trend reliability
    'linear_min_slope': 0.02,       # Higher slope requirement for linear regression
    
    # Output and logging
    'output_format': 'console',      # 'console', 'csv', or 'both'
    'output_file': 'trend_following_results.csv',
    'log_level': 'INFO',
    'detailed_output': True,         # Include detailed metrics in output
}

# Popular stocks to analyze (default list)
POPULAR_STOCKS = [
    # Technology
    'AAPL',    # Apple Inc.
    'MSFT',    # Microsoft Corporation
    'GOOGL',   # Alphabet Inc. (Class A)
    'AMZN',    # Amazon.com Inc.
    'TSLA',    # Tesla Inc.
    'META',    # Meta Platforms Inc.
    'NVDA',    # NVIDIA Corporation
    'NFLX',    # Netflix Inc.
    'AMD',     # Advanced Micro Devices Inc.
    'CRM',     # Salesforce Inc.
    
    # Finance
    'JPM',     # JPMorgan Chase & Co.
    'BAC',     # Bank of America Corporation
    'WFC',     # Wells Fargo & Company
    'GS',      # Goldman Sachs Group Inc.
    'MS',      # Morgan Stanley
    
    # Healthcare
    'JNJ',     # Johnson & Johnson
    'PFE',     # Pfizer Inc.
    'UNH',     # UnitedHealth Group Incorporated
    'ABBV',    # AbbVie Inc.
    'MRK',     # Merck & Co. Inc.
    
    # Consumer
    'KO',      # Coca-Cola Company
    'PEP',     # PepsiCo Inc.
    'WMT',     # Walmart Inc.
    'HD',      # Home Depot Inc.
    'MCD',     # McDonald's Corporation
    
    # Industrial
    'BA',      # Boeing Company
    'CAT',     # Caterpillar Inc.
    'GE',      # General Electric Company
    'MMM',     # 3M Company
    'UPS',     # United Parcel Service Inc.
]

# Extended stock list for comprehensive screening
EXTENDED_STOCKS = POPULAR_STOCKS + [
    # Additional Tech
    'ORCL',    # Oracle Corporation
    'ADBE',    # Adobe Inc.
    'CSCO',    # Cisco Systems Inc.
    'INTC',    # Intel Corporation
    'IBM',     # International Business Machines
    
    # Additional Finance
    'C',       # Citigroup Inc.
    'USB',     # U.S. Bancorp
    'PNC',     # PNC Financial Services Group
    'TFC',     # Truist Financial Corporation
    'AXP',     # American Express Company
    
    # Energy
    'XOM',     # Exxon Mobil Corporation
    'CVX',     # Chevron Corporation
    'COP',     # ConocoPhillips
    'EOG',     # EOG Resources Inc.
    'SLB',     # Schlumberger Limited
    
    # Utilities
    'NEE',     # NextEra Energy Inc.
    'DUK',     # Duke Energy Corporation
    'SO',      # Southern Company
    'D',       # Dominion Energy Inc.
    'EXC',     # Exelon Corporation
]


def get_config(custom_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Get configuration with optional custom overrides.
    
    Args:
        custom_config: Dictionary of custom configuration values to override defaults
        
    Returns:
        Dict containing the final configuration
    """
    config = DEFAULT_CONFIG.copy()
    
    if custom_config:
        config.update(custom_config)
    
    return config


def get_stock_list(list_type: str = 'popular') -> List[str]:
    """
    Get a list of stocks to analyze.
    
    Args:
        list_type: Type of stock list ('popular', 'extended', or 'custom')
        
    Returns:
        List of stock symbols
    """
    if list_type == 'popular':
        return POPULAR_STOCKS.copy()
    elif list_type == 'extended':
        return EXTENDED_STOCKS.copy()
    else:
        return POPULAR_STOCKS.copy()


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and sanitize configuration values.
    
    Args:
        config: Configuration dictionary to validate
        
    Returns:
        Dict containing validated configuration
    """
    validated_config = config.copy()
    
    # Ensure positive values for time periods
    validated_config['analysis_days'] = max(1, config.get('analysis_days', 5))
    validated_config['volume_avg_days'] = max(1, config.get('volume_avg_days', 20))
    validated_config['ma_period'] = max(1, config.get('ma_period', 5))
    
    # Ensure reasonable thresholds
    validated_config['volume_threshold'] = max(1.0, config.get('volume_threshold', 1.2))
    validated_config['min_slope'] = max(0.0, config.get('min_slope', 0.01))
    validated_config['min_positive_days'] = max(0, min(
        config.get('min_positive_days', 3),
        validated_config['analysis_days']
    ))
    
    # Validate trend method
    valid_methods = ['simple', 'linear_regression']
    if config.get('trend_method') not in valid_methods:
        validated_config['trend_method'] = 'simple'
    
    # Validate output format
    valid_formats = ['console', 'csv', 'both']
    if config.get('output_format') not in valid_formats:
        validated_config['output_format'] = 'console'
    
    return validated_config
