#!/usr/bin/env python3
"""Build Initial Gap Trading Universe.

This script populates the initial stock universe for gap trading:
1. Ensures permanent symbols (SPY, QQQ, etc.) are in database
2. Screens ~100 liquid stocks meeting criteria
3. Calculates volatility metrics for all stocks
4. Stores everything in gap_trading.stock_universe table

Usage:
    python build_initial_universe.py

Environment:
    - FMP_API_KEY must be set
    - Database must be accessible (localhost:5432 or timescaledb:5432)
"""

import os
import sys
import logging
from datetime import datetime, date

import psycopg2
from psycopg2.extras import execute_values

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection."""
    # Try localhost first, then docker container name
    hosts = ['localhost', 'timescaledb']

    for host in hosts:
        try:
            conn = psycopg2.connect(
                host=host,
                port=5432,
                database='timescaledb',
                user='postgres',
                password='password'
            )
            logger.info(f"Connected to database at {host}")
            return conn
        except Exception as e:
            logger.debug(f"Could not connect to {host}: {e}")
            continue

    raise ConnectionError("Could not connect to database")


def build_universe():
    """Build the initial stock universe."""
    from fmp import FMPClient
    from fmp.gap_trading import GapTradingScreener, GapTradingHistorical
    from fmp.gap_trading.screener import ScreeningCriteria
    from strategy_builder.strategies.gap_trading import (
        PERMANENT_SYMBOLS,
        REFERENCE_SYMBOLS,
        is_permanent
    )
    from strategy_builder.strategies.gap_trading.universe import PERMANENT_SYMBOL_INFO

    logger.info("=" * 60)
    logger.info("Building Initial Gap Trading Universe")
    logger.info("=" * 60)

    # Get database connection
    conn = get_db_connection()
    cursor = conn.cursor()

    # Create FMP client
    fmp_client = FMPClient()

    try:
        # Step 1: Ensure permanent symbols
        logger.info("\n[Step 1] Ensuring permanent symbols...")

        screener = GapTradingScreener(fmp_client)
        historical = GapTradingHistorical(fmp_client)

        # Get ETF data
        etf_data = screener.screen_etfs(
            symbols=PERMANENT_SYMBOLS + REFERENCE_SYMBOLS,
            as_dataframe=False
        )
        etf_dict = {s.symbol: s for s in etf_data}

        today = date.today()
        permanent_count = 0

        for symbol in PERMANENT_SYMBOLS + REFERENCE_SYMBOLS:
            etf = etf_dict.get(symbol)
            info = PERMANENT_SYMBOL_INFO.get(symbol, {})

            # Set is_reference_only for VIX
            is_ref_only = symbol in REFERENCE_SYMBOLS

            cursor.execute("""
                INSERT INTO gap_trading.stock_universe
                (symbol, company_name, sector, market_cap, avg_volume_20d,
                 current_price, is_permanent, is_reference_only, is_active, added_date)
                VALUES (%s, %s, %s, %s, %s, %s, true, %s, true, %s)
                ON CONFLICT (symbol) DO UPDATE SET
                    company_name = EXCLUDED.company_name,
                    sector = EXCLUDED.sector,
                    market_cap = COALESCE(EXCLUDED.market_cap, gap_trading.stock_universe.market_cap),
                    avg_volume_20d = COALESCE(EXCLUDED.avg_volume_20d, gap_trading.stock_universe.avg_volume_20d),
                    current_price = COALESCE(EXCLUDED.current_price, gap_trading.stock_universe.current_price),
                    is_permanent = true,
                    is_reference_only = EXCLUDED.is_reference_only,
                    is_active = true,
                    last_updated = CURRENT_TIMESTAMP
            """, (
                symbol,
                etf.company_name if etf else info.get('name', symbol),
                info.get('sector', 'ETF'),
                etf.market_cap if etf else None,
                etf.avg_volume if etf else None,
                etf.price if etf else None,
                is_ref_only,
                today
            ))
            permanent_count += 1

        conn.commit()
        logger.info(f"  Ensured {permanent_count} permanent/reference symbols")

        # Step 2: Screen liquid stocks
        logger.info("\n[Step 2] Screening liquid stocks...")

        criteria = ScreeningCriteria(
            min_market_cap=1_000_000_000,      # $1B
            min_avg_volume=100_000,             # 100K shares
            min_dollar_volume=20_000_000,       # $20M
            min_price=10.0,
            max_price=500.0,
            max_stocks=100
        )

        screened = screener.screen_for_universe(criteria, as_dataframe=False)
        logger.info(f"  Screened {len(screened)} stocks meeting criteria")

        # Step 3: Add screened stocks to universe
        logger.info("\n[Step 3] Adding screened stocks to universe...")

        added_count = 0
        for stock in screened:
            # Skip permanent symbols (already added)
            if is_permanent(stock.symbol):
                continue

            try:
                cursor.execute("""
                    INSERT INTO gap_trading.stock_universe
                    (symbol, company_name, sector, market_cap,
                     avg_volume_20d, current_price, is_permanent, is_active, added_date)
                    VALUES (%s, %s, %s, %s, %s, %s, false, true, %s)
                    ON CONFLICT (symbol) DO UPDATE SET
                        company_name = EXCLUDED.company_name,
                        sector = EXCLUDED.sector,
                        market_cap = EXCLUDED.market_cap,
                        avg_volume_20d = EXCLUDED.avg_volume_20d,
                        current_price = EXCLUDED.current_price,
                        is_active = true,
                        last_updated = CURRENT_TIMESTAMP
                """, (
                    stock.symbol,
                    stock.company_name,
                    stock.sector,
                    stock.market_cap,
                    stock.avg_volume,
                    stock.price,
                    today
                ))
                added_count += 1
            except Exception as e:
                logger.error(f"  Error adding {stock.symbol}: {e}")

        conn.commit()
        logger.info(f"  Added {added_count} stocks to universe")

        # Step 4: Calculate volatility for all stocks
        logger.info("\n[Step 4] Calculating volatility metrics...")

        cursor.execute("SELECT symbol FROM gap_trading.stock_universe WHERE is_active = true")
        all_symbols = [row[0] for row in cursor.fetchall()]

        # Skip VIX for volatility calculation (it's a volatility index itself)
        calc_symbols = [s for s in all_symbols if s != 'VIX']

        logger.info(f"  Calculating volatility for {len(calc_symbols)} symbols...")

        # Process in batches to avoid API limits
        batch_size = 20
        vol_count = 0

        for i in range(0, len(calc_symbols), batch_size):
            batch = calc_symbols[i:i + batch_size]
            logger.info(f"  Processing batch {i//batch_size + 1}/{(len(calc_symbols)-1)//batch_size + 1}...")

            metrics = historical.calculate_batch_volatility(batch, calculate_beta=True)

            for symbol, m in metrics.items():
                try:
                    # Convert numpy types to Python floats
                    atr_14 = float(m.atr_14) if m.atr_14 is not None else None
                    atr_pct = float(m.atr_pct) if m.atr_pct is not None else None
                    adr_pct = float(m.adr_20) if m.adr_20 is not None else None
                    vol_20 = float(m.volatility_20) if m.volatility_20 is not None else None
                    vol_60 = float(m.volatility_60) if m.volatility_60 is not None else None
                    beta = float(m.beta) if m.beta is not None else None

                    # Determine risk tier
                    if atr_pct is not None and atr_pct < 2.0:
                        risk_tier = 'LOW'
                    elif atr_pct is not None and atr_pct < 4.0:
                        risk_tier = 'MEDIUM'
                    else:
                        risk_tier = 'HIGH'

                    # Update stock_universe
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
                        atr_14,
                        atr_pct,
                        adr_pct,
                        vol_20,
                        vol_60,
                        beta,
                        risk_tier,
                        symbol
                    ))

                    # Insert into volatility_history
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
                        symbol, today, atr_14, atr_pct, adr_pct,
                        vol_20, vol_60, beta, risk_tier
                    ))

                    vol_count += 1
                except Exception as e:
                    logger.error(f"  Error storing volatility for {symbol}: {e}")

            conn.commit()

        logger.info(f"  Calculated volatility for {vol_count} symbols")

        # Step 5: Print summary
        logger.info("\n" + "=" * 60)
        logger.info("UNIVERSE BUILD COMPLETE")
        logger.info("=" * 60)

        cursor.execute("""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN is_permanent THEN 1 ELSE 0 END) as permanent,
                SUM(CASE WHEN risk_tier = 'LOW' THEN 1 ELSE 0 END) as low_risk,
                SUM(CASE WHEN risk_tier = 'MEDIUM' THEN 1 ELSE 0 END) as med_risk,
                SUM(CASE WHEN risk_tier = 'HIGH' THEN 1 ELSE 0 END) as high_risk
            FROM gap_trading.stock_universe
            WHERE is_active = true
        """)
        row = cursor.fetchone()
        total, permanent, low, med, high = row

        logger.info(f"\nTotal Active Stocks: {total}")
        logger.info(f"  - Permanent: {permanent}")
        logger.info(f"  - Screened: {total - permanent}")
        logger.info(f"\nRisk Tier Distribution:")
        logger.info(f"  - LOW (ATR <2%): {low or 0}")
        logger.info(f"  - MEDIUM (ATR 2-4%): {med or 0}")
        logger.info(f"  - HIGH (ATR >4%): {high or 0}")

        # Sector distribution
        cursor.execute("""
            SELECT sector, COUNT(*) as cnt
            FROM gap_trading.stock_universe
            WHERE is_active = true AND sector IS NOT NULL
            GROUP BY sector
            ORDER BY cnt DESC
            LIMIT 10
        """)
        logger.info(f"\nTop Sectors:")
        for row in cursor.fetchall():
            logger.info(f"  - {row[0]}: {row[1]}")

        return True

    except Exception as e:
        logger.error(f"Error building universe: {e}")
        conn.rollback()
        raise

    finally:
        fmp_client.close()
        conn.close()


if __name__ == '__main__':
    success = build_universe()
    sys.exit(0 if success else 1)
