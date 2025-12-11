#!/usr/bin/env python3
"""Paper Trading Validation Script.

Validates the gap trading strategy in paper trading mode before going live.

Performs:
1. Connectivity checks (Tradier sandbox, FMP API, Database)
2. Signal generation test
3. Order placement test (paper)
4. Position tracking test
5. Stop-loss monitoring test
6. Report generation test

Run this script daily during paper trading validation period.

Usage:
    python validate_paper_trading.py --full    # Run all tests
    python validate_paper_trading.py --quick   # Quick connectivity check
    python validate_paper_trading.py --report  # Generate validation report
"""

import os
import sys
import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('paper_trading_validation')


@dataclass
class ValidationResult:
    """Result of a validation check."""
    name: str
    passed: bool
    message: str
    details: Optional[Dict[str, Any]] = None
    duration_ms: float = 0


@dataclass
class ValidationReport:
    """Full validation report."""
    run_date: datetime
    environment: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    results: List[ValidationResult]
    ready_for_live: bool

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'run_date': self.run_date.isoformat(),
            'environment': self.environment,
            'total_checks': self.total_checks,
            'passed_checks': self.passed_checks,
            'failed_checks': self.failed_checks,
            'pass_rate': f"{(self.passed_checks / self.total_checks * 100):.1f}%" if self.total_checks > 0 else "0%",
            'ready_for_live': self.ready_for_live,
            'results': [asdict(r) for r in self.results]
        }

    def summary(self) -> str:
        """Generate text summary."""
        status = "✅ READY" if self.ready_for_live else "❌ NOT READY"
        summary = f"""
========================================
Paper Trading Validation Report
========================================
Run Date: {self.run_date.strftime('%Y-%m-%d %H:%M:%S')}
Environment: {self.environment}

Overall Status: {status}

Results: {self.passed_checks}/{self.total_checks} checks passed
"""
        for r in self.results:
            icon = "✅" if r.passed else "❌"
            summary += f"\n{icon} {r.name}: {r.message}"

        return summary


class PaperTradingValidator:
    """Validates paper trading setup and functionality."""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = datetime.now()

    def run_all_validations(self) -> ValidationReport:
        """Run all validation checks.

        Returns:
            ValidationReport with all results
        """
        logger.info("Starting full paper trading validation...")

        # Connectivity checks
        self._check_environment_variables()
        self._check_database_connection()
        self._check_tradier_connection()
        self._check_fmp_connection()

        # Functional checks
        self._check_universe_populated()
        self._check_signal_generation()
        self._check_order_placement()
        self._check_position_tracking()
        self._check_reporting()

        # Generate report
        passed = len([r for r in self.results if r.passed])
        failed = len([r for r in self.results if not r.passed])

        # Ready for live if all critical checks pass
        critical_checks = ['environment_variables', 'database', 'tradier_connection',
                          'universe', 'signal_generation']
        critical_passed = all(
            r.passed for r in self.results
            if any(c in r.name.lower() for c in critical_checks)
        )

        return ValidationReport(
            run_date=datetime.now(),
            environment='sandbox' if os.environ.get('TRADIER_SANDBOX', 'true').lower() == 'true' else 'production',
            total_checks=len(self.results),
            passed_checks=passed,
            failed_checks=failed,
            results=self.results,
            ready_for_live=critical_passed and failed == 0
        )

    def run_quick_check(self) -> ValidationReport:
        """Run quick connectivity checks only.

        Returns:
            ValidationReport with connectivity results
        """
        logger.info("Running quick connectivity check...")

        self._check_environment_variables()
        self._check_database_connection()
        self._check_tradier_connection()

        passed = len([r for r in self.results if r.passed])
        failed = len([r for r in self.results if not r.passed])

        return ValidationReport(
            run_date=datetime.now(),
            environment='quick_check',
            total_checks=len(self.results),
            passed_checks=passed,
            failed_checks=failed,
            results=self.results,
            ready_for_live=failed == 0
        )

    def _check_environment_variables(self):
        """Check required environment variables are set."""
        start = datetime.now()

        required_vars = [
            'TRADIER_API_KEY',
            'TRADIER_ACCOUNT_ID',
            'FMP_API_KEY',
        ]

        optional_vars = [
            'TELEGRAM_BOT_TOKEN',
            'TELEGRAM_CHAT_ID',
        ]

        missing_required = []
        missing_optional = []

        for var in required_vars:
            if not os.environ.get(var):
                missing_required.append(var)

        for var in optional_vars:
            if not os.environ.get(var):
                missing_optional.append(var)

        duration = (datetime.now() - start).total_seconds() * 1000

        if missing_required:
            self.results.append(ValidationResult(
                name='Environment Variables',
                passed=False,
                message=f"Missing required: {', '.join(missing_required)}",
                details={'missing_required': missing_required, 'missing_optional': missing_optional},
                duration_ms=duration
            ))
        else:
            self.results.append(ValidationResult(
                name='Environment Variables',
                passed=True,
                message=f"All required set. Optional missing: {len(missing_optional)}",
                details={'missing_optional': missing_optional},
                duration_ms=duration
            ))

    def _check_database_connection(self):
        """Check database connectivity."""
        start = datetime.now()

        try:
            import psycopg2

            conn_string = os.environ.get(
                'DATABASE_URL',
                'postgresql://postgres:password@localhost:5432/timescaledb'
            )

            conn = psycopg2.connect(conn_string, connect_timeout=5)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()

            # Check gap_trading schema exists
            cursor.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.schemata
                    WHERE schema_name = 'gap_trading'
                )
            """)
            schema_exists = cursor.fetchone()[0]

            conn.close()

            duration = (datetime.now() - start).total_seconds() * 1000

            if schema_exists:
                self.results.append(ValidationResult(
                    name='Database Connection',
                    passed=True,
                    message="Connected, gap_trading schema exists",
                    duration_ms=duration
                ))
            else:
                self.results.append(ValidationResult(
                    name='Database Connection',
                    passed=False,
                    message="Connected but gap_trading schema missing",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='Database Connection',
                passed=False,
                message=f"Connection failed: {str(e)[:100]}",
                duration_ms=duration
            ))

    def _check_tradier_connection(self):
        """Check Tradier API connectivity."""
        start = datetime.now()

        try:
            from stock_data_web.tradier import TradierClient

            client = TradierClient(sandbox=True)

            # Get balance
            balance = client.get_balance()

            # Get market status
            clock = client.get_market_clock()

            client.close()

            duration = (datetime.now() - start).total_seconds() * 1000

            self.results.append(ValidationResult(
                name='Tradier Connection (Sandbox)',
                passed=True,
                message=f"Connected. Buying power: ${balance.buying_power:,.2f}",
                details={
                    'account_id': balance.account_number,
                    'buying_power': balance.buying_power,
                    'market_state': clock.get('state', 'unknown')
                },
                duration_ms=duration
            ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='Tradier Connection (Sandbox)',
                passed=False,
                message=f"Connection failed: {str(e)[:100]}",
                duration_ms=duration
            ))

    def _check_fmp_connection(self):
        """Check FMP API connectivity."""
        start = datetime.now()

        try:
            from fmp import FMPClient

            client = FMPClient()

            # Test with a simple quote
            quote = client.get_quote('SPY')

            client.close()

            duration = (datetime.now() - start).total_seconds() * 1000

            self.results.append(ValidationResult(
                name='FMP API Connection',
                passed=True,
                message=f"Connected. SPY price: ${quote.price:.2f}",
                details={'spy_price': quote.price},
                duration_ms=duration
            ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='FMP API Connection',
                passed=False,
                message=f"Connection failed: {str(e)[:100]}",
                duration_ms=duration
            ))

    def _check_universe_populated(self):
        """Check that stock universe is populated."""
        start = datetime.now()

        try:
            import psycopg2

            conn_string = os.environ.get(
                'DATABASE_URL',
                'postgresql://postgres:password@localhost:5432/timescaledb'
            )

            conn = psycopg2.connect(conn_string)
            cursor = conn.cursor()

            # Count active stocks
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    COUNT(*) FILTER (WHERE is_permanent) as permanent,
                    COUNT(*) FILTER (WHERE atr_14 IS NOT NULL) as with_atr
                FROM gap_trading.stock_universe
                WHERE is_active = true
            """)

            total, permanent, with_atr = cursor.fetchone()
            conn.close()

            duration = (datetime.now() - start).total_seconds() * 1000

            if total >= 50 and with_atr >= 50:
                self.results.append(ValidationResult(
                    name='Stock Universe',
                    passed=True,
                    message=f"{total} stocks ({permanent} permanent, {with_atr} with ATR)",
                    details={'total': total, 'permanent': permanent, 'with_atr': with_atr},
                    duration_ms=duration
                ))
            else:
                self.results.append(ValidationResult(
                    name='Stock Universe',
                    passed=False,
                    message=f"Insufficient stocks: {total} total, {with_atr} with ATR",
                    details={'total': total, 'with_atr': with_atr},
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='Stock Universe',
                passed=False,
                message=f"Check failed: {str(e)[:100]}",
                duration_ms=duration
            ))

    def _check_signal_generation(self):
        """Test signal generation logic."""
        start = datetime.now()

        try:
            from strategy_builder.strategies.gap_trading.signals import (
                GapDetector, SignalGenerator, GapInfo, GapDirection
            )
            from strategy_builder.strategies.gap_trading.position_sizer import PositionSizer

            # Create test gap
            detector = GapDetector(min_gap_pct=1.5)
            gap = detector.detect_gap('TEST', 100.0, 102.0, datetime.now())

            if not gap.is_significant:
                raise ValueError("Gap detection failed for 2% gap")

            # Test confirmation
            confirmation = detector.confirm_gap(gap, 102.5, datetime.now())

            if not confirmation.is_confirmed:
                raise ValueError("Gap confirmation failed")

            # Test signal generation
            generator = SignalGenerator()
            signal = generator.generate_signal(confirmation, atr=2.0, atr_pct=2.0)

            if not signal.is_tradeable:
                raise ValueError("Signal generation failed")

            # Test position sizing
            sizer = PositionSizer(account_value=100000, risk_per_trade_pct=1.0)
            position = sizer.calculate_position(signal)

            duration = (datetime.now() - start).total_seconds() * 1000

            self.results.append(ValidationResult(
                name='Signal Generation',
                passed=True,
                message=f"Generated {signal.signal_type.value} signal, {position.shares} shares",
                details={
                    'signal_type': signal.signal_type.value,
                    'entry_price': signal.entry_price,
                    'stop_loss': signal.stop_loss,
                    'shares': position.shares
                },
                duration_ms=duration
            ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='Signal Generation',
                passed=False,
                message=f"Test failed: {str(e)[:100]}",
                duration_ms=duration
            ))

    def _check_order_placement(self):
        """Test order placement in sandbox."""
        start = datetime.now()

        try:
            from stock_data_web.tradier import TradierClient, OrderSide

            client = TradierClient(sandbox=True)

            # Check we can get quotes
            quote = client.get_quote('SPY')

            # Note: We don't actually place an order in validation
            # Just verify the API is accessible and we have buying power

            balance = client.get_balance()
            can_trade = balance.buying_power > 1000

            client.close()

            duration = (datetime.now() - start).total_seconds() * 1000

            if can_trade:
                self.results.append(ValidationResult(
                    name='Order Placement (Sandbox)',
                    passed=True,
                    message=f"API accessible, ${balance.buying_power:,.2f} buying power",
                    details={
                        'spy_price': quote.last,
                        'buying_power': balance.buying_power
                    },
                    duration_ms=duration
                ))
            else:
                self.results.append(ValidationResult(
                    name='Order Placement (Sandbox)',
                    passed=False,
                    message=f"Insufficient buying power: ${balance.buying_power:,.2f}",
                    duration_ms=duration
                ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='Order Placement (Sandbox)',
                passed=False,
                message=f"Test failed: {str(e)[:100]}",
                duration_ms=duration
            ))

    def _check_position_tracking(self):
        """Test position tracking functionality."""
        start = datetime.now()

        try:
            from strategy_builder.strategies.gap_trading.position_manager import (
                PositionManager, Position, PositionSide, PositionStatus
            )

            # Test Position class
            position = Position(
                symbol='TEST',
                side=PositionSide.LONG,
                quantity=100,
                entry_price=100.0,
                stop_price=98.0
            )

            # Test P&L calculation
            unrealized = position.calculate_unrealized_pnl(102.0)
            expected_pnl = 200.0  # (102 - 100) * 100

            if abs(unrealized - expected_pnl) > 0.01:
                raise ValueError(f"P&L calculation error: {unrealized} != {expected_pnl}")

            # Test stop hit detection
            if not position.is_stop_hit(97.0):
                raise ValueError("Stop hit detection failed for LONG")

            if position.is_stop_hit(99.0):
                raise ValueError("False stop hit for LONG")

            duration = (datetime.now() - start).total_seconds() * 1000

            self.results.append(ValidationResult(
                name='Position Tracking',
                passed=True,
                message="P&L and stop detection working correctly",
                details={'test_unrealized_pnl': unrealized},
                duration_ms=duration
            ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='Position Tracking',
                passed=False,
                message=f"Test failed: {str(e)[:100]}",
                duration_ms=duration
            ))

    def _check_reporting(self):
        """Test reporting functionality."""
        start = datetime.now()

        try:
            from strategy_builder.strategies.gap_trading.reporting import (
                ReportGenerator, TradeMetrics, DailyReport
            )

            # Test TradeMetrics
            metrics = TradeMetrics(
                total_trades=10,
                winning_trades=6,
                losing_trades=4,
                gross_profit=1200,
                gross_loss=800
            )

            if metrics.to_dict()['total_trades'] != 10:
                raise ValueError("TradeMetrics serialization failed")

            # Test DailyReport
            report = DailyReport(
                report_date=date.today(),
                trade_metrics=metrics
            )

            report_dict = report.to_dict()
            if 'trade_metrics' not in report_dict:
                raise ValueError("DailyReport serialization failed")

            duration = (datetime.now() - start).total_seconds() * 1000

            self.results.append(ValidationResult(
                name='Reporting',
                passed=True,
                message="Report generation working correctly",
                duration_ms=duration
            ))

        except Exception as e:
            duration = (datetime.now() - start).total_seconds() * 1000
            self.results.append(ValidationResult(
                name='Reporting',
                passed=False,
                message=f"Test failed: {str(e)[:100]}",
                duration_ms=duration
            ))


def main():
    """Run validation based on command line args."""
    import argparse

    parser = argparse.ArgumentParser(description='Paper Trading Validation')
    parser.add_argument('--full', action='store_true', help='Run full validation')
    parser.add_argument('--quick', action='store_true', help='Quick connectivity check')
    parser.add_argument('--report', action='store_true', help='Output JSON report')
    parser.add_argument('--output', type=str, help='Output file path')

    args = parser.parse_args()

    validator = PaperTradingValidator()

    if args.quick:
        report = validator.run_quick_check()
    else:
        report = validator.run_all_validations()

    # Output
    if args.report:
        output = json.dumps(report.to_dict(), indent=2)
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report saved to {args.output}")
        else:
            print(output)
    else:
        print(report.summary())

    # Exit code
    sys.exit(0 if report.ready_for_live else 1)


if __name__ == '__main__':
    main()
