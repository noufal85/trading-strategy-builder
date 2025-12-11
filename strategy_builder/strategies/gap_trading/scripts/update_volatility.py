#!/usr/bin/env python3
"""Update Volatility Metrics for Stock Universe.

This script calculates and updates volatility metrics for all stocks
in the gap_trading.stock_universe table.

Usage:
    python update_volatility.py
"""

import os
import sys
import logging
from datetime import date

import psycopg2

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_db_connection():
    """Get database connection."""
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


def update_volatility():
    """Update volatility metrics for all active stocks."""
    from fmp import FMPClient
    from fmp.gap_trading import GapTradingHistorical

    logger.info("=" * 60)
    logger.info("Updating Volatility Metrics")
    logger.info("=" * 60)

    conn = get_db_connection()
    cursor = conn.cursor()
    fmp_client = FMPClient()

    try:
        # Get all active symbols (except VIX)
        cursor.execute("""
            SELECT symbol FROM gap_trading.stock_universe
            WHERE is_active = true AND symbol != 'VIX'
        """)
        symbols = [row[0] for row in cursor.fetchall()]

        logger.info(f"Found {len(symbols)} symbols to process")

        historical = GapTradingHistorical(fmp_client)
        today = date.today()

        # Process in batches
        batch_size = 20
        vol_count = 0

        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(symbols) - 1) // batch_size + 1
            logger.info(f"Processing batch {batch_num}/{total_batches}: {batch}")

            try:
                metrics = historical.calculate_batch_volatility(batch, calculate_beta=True)
            except Exception as e:
                logger.error(f"Error calculating batch volatility: {e}")
                continue

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
                        atr_14, atr_pct, adr_pct, vol_20, vol_60, beta, risk_tier, symbol
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
                        symbol, today, atr_14, atr_pct, adr_pct, vol_20, vol_60, beta, risk_tier
                    ))

                    vol_count += 1
                    logger.info(f"  {symbol}: ATR%={atr_pct:.2f}%, Beta={beta:.2f}, Risk={risk_tier}")

                except Exception as e:
                    logger.error(f"  Error storing volatility for {symbol}: {e}")

            conn.commit()

        # Print summary
        logger.info("=" * 60)
        logger.info(f"VOLATILITY UPDATE COMPLETE: {vol_count} symbols updated")
        logger.info("=" * 60)

        # Show risk tier distribution
        cursor.execute("""
            SELECT risk_tier, COUNT(*) as cnt
            FROM gap_trading.stock_universe
            WHERE is_active = true AND risk_tier IS NOT NULL
            GROUP BY risk_tier
            ORDER BY risk_tier
        """)
        logger.info("Risk Tier Distribution:")
        for row in cursor.fetchall():
            logger.info(f"  {row[0]}: {row[1]} stocks")

        return True

    except Exception as e:
        logger.error(f"Error updating volatility: {e}")
        conn.rollback()
        raise

    finally:
        fmp_client.close()
        conn.close()


if __name__ == '__main__':
    success = update_volatility()
    sys.exit(0 if success else 1)
