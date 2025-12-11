"""Gap Trading Universe Manager.

Manages the stock universe for gap trading strategy:
- Permanent symbols (ETFs that are always included)
- Dynamic screening for liquid stocks
- Universe updates and maintenance
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, date
import logging
import pandas as pd

from fmp import FMPClient
from fmp.gap_trading import GapTradingScreener, GapTradingHistorical
from fmp.gap_trading.screener import ScreeningCriteria, ScreenedStock
from fmp.gap_trading.historical import VolatilityMetrics


logger = logging.getLogger(__name__)


# Permanent symbols - always included regardless of screening
PERMANENT_SYMBOLS = ['SPY', 'QQQ', 'DIA', 'IWM', 'GLD', 'USO', 'TLT']

# Reference symbols - tracked but not traded
REFERENCE_SYMBOLS = ['VIX']

# Permanent symbol metadata
PERMANENT_SYMBOL_INFO = {
    'SPY': {'name': 'SPDR S&P 500 ETF Trust', 'sector': 'ETF - Index'},
    'QQQ': {'name': 'Invesco QQQ Trust', 'sector': 'ETF - Index'},
    'DIA': {'name': 'SPDR Dow Jones Industrial Average ETF', 'sector': 'ETF - Index'},
    'IWM': {'name': 'iShares Russell 2000 ETF', 'sector': 'ETF - Index'},
    'GLD': {'name': 'SPDR Gold Shares', 'sector': 'ETF - Commodity'},
    'USO': {'name': 'United States Oil Fund', 'sector': 'ETF - Commodity'},
    'TLT': {'name': 'iShares 20+ Year Treasury Bond ETF', 'sector': 'ETF - Bond'},
    'VIX': {'name': 'CBOE Volatility Index', 'sector': 'Reference - Index'},
}


def get_permanent_symbols() -> List[str]:
    """Get list of permanent symbols."""
    return PERMANENT_SYMBOLS.copy()


def get_reference_symbols() -> List[str]:
    """Get list of reference symbols (no trading)."""
    return REFERENCE_SYMBOLS.copy()


def is_permanent(symbol: str) -> bool:
    """Check if symbol is permanent."""
    return symbol.upper() in PERMANENT_SYMBOLS


def is_reference_only(symbol: str) -> bool:
    """Check if symbol is reference-only (no trading)."""
    return symbol.upper() in REFERENCE_SYMBOLS


def get_all_tracked_symbols() -> List[str]:
    """Get all symbols (permanent + reference)."""
    return PERMANENT_SYMBOLS + REFERENCE_SYMBOLS


@dataclass
class UniverseStock:
    """Stock in the trading universe.

    Attributes:
        symbol: Stock ticker symbol
        company_name: Company name
        sector: Sector classification
        exchange: Exchange (NYSE, NASDAQ)
        market_cap: Market capitalization
        avg_volume: Average daily volume
        current_price: Current/latest price
        is_permanent: Whether it's a permanent symbol
        is_active: Whether actively included in trading
        added_date: Date added to universe
        removed_date: Date removed (if applicable)
        removal_reason: Reason for removal
    """
    symbol: str
    company_name: str
    sector: Optional[str] = None
    exchange: Optional[str] = None
    market_cap: Optional[float] = None
    avg_volume: Optional[int] = None
    current_price: Optional[float] = None
    is_permanent: bool = False
    is_active: bool = True
    added_date: Optional[date] = None
    removed_date: Optional[date] = None
    removal_reason: Optional[str] = None


class UniverseManager:
    """Manages the gap trading stock universe.

    Handles:
    - Screening new stocks based on criteria
    - Adding/removing stocks from universe
    - Ensuring permanent symbols are always present
    - Calculating volatility for new stocks

    Attributes:
        db_conn: Database connection (psycopg2 or sqlalchemy)
        fmp_client: FMP API client
        config: Gap trading configuration
    """

    def __init__(
        self,
        db_conn: Any,
        fmp_client: Optional[FMPClient] = None,
        config: Optional[Any] = None
    ):
        """Initialize UniverseManager.

        Args:
            db_conn: Database connection
            fmp_client: FMP API client (creates one if not provided)
            config: GapTradingConfig instance
        """
        self.db_conn = db_conn
        self.fmp_client = fmp_client or FMPClient()
        self.config = config

        # Initialize FMP modules
        self.screener = GapTradingScreener(self.fmp_client)
        self.historical = GapTradingHistorical(self.fmp_client)

    def get_current_universe(self, active_only: bool = True) -> List[UniverseStock]:
        """Get current stocks in universe from database.

        Args:
            active_only: Only return active stocks

        Returns:
            List of UniverseStock objects
        """
        query = """
            SELECT symbol, company_name, sector, NULL as exchange, market_cap,
                   avg_volume_20d, current_price, is_permanent, is_active,
                   added_date, deactivated_date, deactivation_reason
            FROM gap_trading.stock_universe
        """
        if active_only:
            query += " WHERE is_active = true"
        query += " ORDER BY is_permanent DESC, symbol"

        cursor = self.db_conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()

        stocks = []
        for row in rows:
            stocks.append(UniverseStock(
                symbol=row[0],
                company_name=row[1],
                sector=row[2],
                exchange=row[3],
                market_cap=row[4],
                avg_volume=row[5],
                current_price=float(row[6]) if row[6] else None,
                is_permanent=row[7],
                is_active=row[8],
                added_date=row[9],
                removed_date=row[10],
                removal_reason=row[11]
            ))

        return stocks

    def get_active_symbols(self) -> List[str]:
        """Get list of active symbol tickers.

        Returns:
            List of ticker symbols
        """
        query = """
            SELECT symbol FROM gap_trading.stock_universe
            WHERE is_active = true
            ORDER BY symbol
        """
        cursor = self.db_conn.cursor()
        cursor.execute(query)
        return [row[0] for row in cursor.fetchall()]

    def get_tradeable_symbols(self) -> List[str]:
        """Get symbols that can be traded (excludes reference-only).

        Returns:
            List of tradeable ticker symbols
        """
        symbols = self.get_active_symbols()
        return [s for s in symbols if not is_reference_only(s)]

    def screen_stocks(self, criteria: Optional[ScreeningCriteria] = None) -> List[ScreenedStock]:
        """Screen stocks using FMP API.

        Args:
            criteria: Screening criteria (uses config defaults if None)

        Returns:
            List of ScreenedStock meeting criteria
        """
        if criteria is None and self.config:
            criteria = ScreeningCriteria(
                min_market_cap=self.config.universe.screening.min_market_cap,
                min_avg_volume=self.config.universe.screening.min_avg_volume,
                min_dollar_volume=self.config.universe.screening.min_dollar_volume,
                min_price=self.config.universe.screening.min_price,
                max_price=self.config.universe.screening.max_price,
                max_stocks=self.config.universe.screening.max_stocks
            )

        return self.screener.screen_for_universe(criteria, as_dataframe=False)

    def identify_new_stocks(
        self,
        screened: List[ScreenedStock]
    ) -> List[ScreenedStock]:
        """Identify stocks not currently in universe.

        Args:
            screened: List of screened stocks

        Returns:
            List of new stocks to add
        """
        current_symbols = set(self.get_active_symbols())
        new_stocks = [s for s in screened if s.symbol not in current_symbols]
        logger.info(f"Identified {len(new_stocks)} new stocks to add")
        return new_stocks

    def identify_removed_stocks(
        self,
        screened: List[ScreenedStock]
    ) -> List[str]:
        """Identify stocks to remove (no longer meet criteria).

        Permanent symbols are never removed.

        Args:
            screened: List of screened stocks

        Returns:
            List of symbols to remove
        """
        screened_symbols = set(s.symbol for s in screened)
        current_symbols = set(self.get_active_symbols())

        # Find symbols no longer in screened list
        to_remove = current_symbols - screened_symbols

        # Never remove permanent symbols
        to_remove = [s for s in to_remove if not is_permanent(s)]

        logger.info(f"Identified {len(to_remove)} stocks to remove")
        return to_remove

    def add_stocks(
        self,
        stocks: List[ScreenedStock],
        calculate_volatility: bool = True
    ) -> int:
        """Add new stocks to universe.

        Args:
            stocks: List of ScreenedStock to add
            calculate_volatility: Calculate volatility metrics for new stocks

        Returns:
            Number of stocks added
        """
        if not stocks:
            return 0

        cursor = self.db_conn.cursor()
        added_count = 0
        today = date.today()

        for stock in stocks:
            try:
                # Insert into stock_universe
                cursor.execute("""
                    INSERT INTO gap_trading.stock_universe
                    (symbol, company_name, sector, market_cap,
                     avg_volume_20d, current_price, is_permanent, is_active, added_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        sector = EXCLUDED.sector,
                        market_cap = EXCLUDED.market_cap,
                        avg_volume_20d = EXCLUDED.avg_volume_20d,
                        current_price = EXCLUDED.current_price,
                        is_active = true,
                        deactivated_date = NULL,
                        deactivation_reason = NULL,
                        last_updated = CURRENT_TIMESTAMP
                """, (
                    stock.symbol,
                    stock.company_name,
                    stock.sector,
                    stock.market_cap,
                    stock.avg_volume,
                    stock.price,
                    is_permanent(stock.symbol),
                    True,
                    today
                ))
                added_count += 1

            except Exception as e:
                logger.error(f"Error adding {stock.symbol}: {e}")
                continue

        self.db_conn.commit()
        logger.info(f"Added {added_count} stocks to universe")

        # Calculate volatility for new stocks
        if calculate_volatility and stocks:
            symbols = [s.symbol for s in stocks]
            self._calculate_and_store_volatility(symbols)

        return added_count

    def remove_stocks(
        self,
        symbols: List[str],
        reason: str = "No longer meets screening criteria"
    ) -> int:
        """Remove stocks from active universe.

        Permanent symbols cannot be removed.

        Args:
            symbols: List of symbols to remove
            reason: Reason for removal

        Returns:
            Number of stocks removed
        """
        if not symbols:
            return 0

        # Filter out permanent symbols
        symbols = [s for s in symbols if not is_permanent(s)]

        if not symbols:
            return 0

        cursor = self.db_conn.cursor()
        today = date.today()

        cursor.execute("""
            UPDATE gap_trading.stock_universe
            SET is_active = false,
                deactivated_date = %s,
                deactivation_reason = %s,
                last_updated = CURRENT_TIMESTAMP
            WHERE symbol = ANY(%s) AND is_permanent = false
        """, (today, reason, symbols))

        removed_count = cursor.rowcount
        self.db_conn.commit()

        logger.info(f"Removed {removed_count} stocks from universe")
        return removed_count

    def ensure_permanent_symbols(self) -> int:
        """Ensure all permanent symbols are in universe.

        Returns:
            Number of symbols added/updated
        """
        logger.info("Ensuring permanent symbols are in universe")

        # Get ETF data from FMP
        screened = self.screener.screen_etfs(
            symbols=PERMANENT_SYMBOLS + REFERENCE_SYMBOLS,
            as_dataframe=False
        )

        cursor = self.db_conn.cursor()
        today = date.today()
        count = 0

        for symbol in PERMANENT_SYMBOLS + REFERENCE_SYMBOLS:
            # Find in screened data or use defaults
            stock_data = next((s for s in screened if s.symbol == symbol), None)

            info = PERMANENT_SYMBOL_INFO.get(symbol, {})
            company_name = stock_data.company_name if stock_data else info.get('name', symbol)
            sector = info.get('sector', 'ETF')
            market_cap = stock_data.market_cap if stock_data else None
            avg_volume = stock_data.avg_volume if stock_data else None
            price = stock_data.price if stock_data else None

            try:
                cursor.execute("""
                    INSERT INTO gap_trading.stock_universe
                    (symbol, company_name, sector, market_cap, avg_volume_20d,
                     current_price, is_permanent, is_active, added_date)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        sector = EXCLUDED.sector,
                        market_cap = COALESCE(EXCLUDED.market_cap, gap_trading.stock_universe.market_cap),
                        avg_volume_20d = COALESCE(EXCLUDED.avg_volume_20d, gap_trading.stock_universe.avg_volume_20d),
                        current_price = COALESCE(EXCLUDED.current_price, gap_trading.stock_universe.current_price),
                        is_permanent = EXCLUDED.is_permanent,
                        is_active = true,
                        last_updated = CURRENT_TIMESTAMP
                """, (
                    symbol,
                    company_name,
                    sector,
                    market_cap,
                    avg_volume,
                    price,
                    True,  # is_permanent
                    True,  # is_active
                    today
                ))
                count += 1

            except Exception as e:
                logger.error(f"Error ensuring permanent symbol {symbol}: {e}")

        self.db_conn.commit()
        logger.info(f"Ensured {count} permanent/reference symbols in universe")
        return count

    def update_universe(
        self,
        criteria: Optional[ScreeningCriteria] = None
    ) -> Dict[str, Any]:
        """Run full universe update cycle.

        1. Ensure permanent symbols
        2. Screen stocks
        3. Identify new/removed
        4. Update database
        5. Calculate volatility for new stocks

        Args:
            criteria: Screening criteria

        Returns:
            Summary dict with counts
        """
        logger.info("Starting universe update")

        # Step 1: Ensure permanent symbols
        permanent_count = self.ensure_permanent_symbols()

        # Step 2: Screen stocks
        screened = self.screen_stocks(criteria)
        logger.info(f"Screened {len(screened)} stocks")

        # Step 3: Identify changes
        new_stocks = self.identify_new_stocks(screened)
        removed_symbols = self.identify_removed_stocks(screened)

        # Step 4: Apply changes
        added_count = self.add_stocks(new_stocks, calculate_volatility=True)
        removed_count = self.remove_stocks(removed_symbols)

        # Get final count
        final_count = len(self.get_active_symbols())

        summary = {
            'timestamp': datetime.now().isoformat(),
            'screened_count': len(screened),
            'permanent_ensured': permanent_count,
            'stocks_added': added_count,
            'stocks_removed': removed_count,
            'total_active': final_count,
            'new_symbols': [s.symbol for s in new_stocks],
            'removed_symbols': removed_symbols
        }

        logger.info(f"Universe update complete: {summary}")
        return summary

    def _calculate_and_store_volatility(self, symbols: List[str]) -> int:
        """Calculate volatility metrics and store in database.

        Args:
            symbols: List of symbols to calculate

        Returns:
            Number of symbols processed
        """
        logger.info(f"Calculating volatility for {len(symbols)} symbols")

        metrics = self.historical.calculate_batch_volatility(
            symbols,
            calculate_beta=True
        )

        cursor = self.db_conn.cursor()
        count = 0

        for symbol, m in metrics.items():
            try:
                # Determine risk tier
                if m.atr_pct < 2.0:
                    risk_tier = 'LOW'
                elif m.atr_pct < 4.0:
                    risk_tier = 'MEDIUM'
                else:
                    risk_tier = 'HIGH'

                # Update stock_universe with volatility metrics
                cursor.execute("""
                    UPDATE gap_trading.stock_universe
                    SET atr_14 = %s,
                        atr_pct = %s,
                        adr_pct = %s,
                        volatility_20d = %s,
                        volatility_60d = %s,
                        beta = %s,
                        risk_tier = %s,
                        last_updated = CURRENT_TIMESTAMP
                    WHERE symbol = %s
                """, (
                    m.atr_14,
                    m.atr_pct,
                    m.adr_20,
                    m.volatility_20,
                    m.volatility_60,
                    m.beta,
                    risk_tier,
                    symbol
                ))

                # Also insert into volatility_history
                cursor.execute("""
                    INSERT INTO gap_trading.volatility_history
                    (symbol, calc_date, atr_14, atr_pct, adr_pct,
                     volatility_20d, volatility_60d, beta, risk_tier)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (symbol, calc_date) DO UPDATE SET
                        atr_14 = EXCLUDED.atr_14,
                        atr_pct = EXCLUDED.atr_pct,
                        adr_pct = EXCLUDED.adr_pct,
                        volatility_20d = EXCLUDED.volatility_20d,
                        volatility_60d = EXCLUDED.volatility_60d,
                        beta = EXCLUDED.beta,
                        risk_tier = EXCLUDED.risk_tier
                """, (
                    symbol,
                    m.calculation_date or date.today(),
                    m.atr_14,
                    m.atr_pct,
                    m.adr_20,
                    m.volatility_20,
                    m.volatility_60,
                    m.beta,
                    risk_tier
                ))

                count += 1

            except Exception as e:
                logger.error(f"Error storing volatility for {symbol}: {e}")

        self.db_conn.commit()
        logger.info(f"Stored volatility for {count} symbols")
        return count

    def get_universe_stats(self) -> Dict[str, Any]:
        """Get statistics about current universe.

        Returns:
            Dict with universe statistics
        """
        cursor = self.db_conn.cursor()

        # Total counts
        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_permanent THEN 1 ELSE 0 END) as permanent,
                SUM(CASE WHEN is_active THEN 1 ELSE 0 END) as active
            FROM gap_trading.stock_universe
        """)
        row = cursor.fetchone()
        total, permanent, active = row

        # Sector distribution
        cursor.execute("""
            SELECT sector, COUNT(*) as count
            FROM gap_trading.stock_universe
            WHERE is_active = true
            GROUP BY sector
            ORDER BY count DESC
        """)
        sectors = {row[0]: row[1] for row in cursor.fetchall()}

        # Recent additions
        cursor.execute("""
            SELECT symbol, added_date
            FROM gap_trading.stock_universe
            WHERE is_active = true AND added_date IS NOT NULL
            ORDER BY added_date DESC
            LIMIT 10
        """)
        recent = [(row[0], row[1].isoformat() if row[1] else None) for row in cursor.fetchall()]

        return {
            'total_stocks': total,
            'permanent_stocks': permanent,
            'active_stocks': active,
            'sector_distribution': sectors,
            'recent_additions': recent
        }
