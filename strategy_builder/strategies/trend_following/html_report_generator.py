"""
HTML Report Generator for Trend Following Strategy.

This module generates comprehensive HTML reports with individual stock analysis,
charts, and detailed failure reasons for the trend following scanner.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path


class HTMLReportGenerator:
    """
    Generates timestamped HTML reports for trend following analysis.
    """
    
    def __init__(self, base_reports_folder: str = "reports"):
        """
        Initialize the HTML report generator.
        
        Args:
            base_reports_folder: Base folder for all reports
        """
        self.base_reports_folder = base_reports_folder
        self.timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.report_folder = os.path.join(base_reports_folder, self.timestamp)
        
        # Create folder structure
        self._create_folder_structure()
    
    def _create_folder_structure(self):
        """Create the necessary folder structure for the report."""
        folders = [
            self.report_folder,
            os.path.join(self.report_folder, "stocks"),
            os.path.join(self.report_folder, "charts"),
            os.path.join(self.report_folder, "assets")
        ]
        
        for folder in folders:
            Path(folder).mkdir(parents=True, exist_ok=True)
        
        # Create symlink to latest report
        latest_link = os.path.join(self.base_reports_folder, "latest")
        if os.path.islink(latest_link):
            os.unlink(latest_link)
        elif os.path.exists(latest_link):
            os.remove(latest_link)
        
        try:
            os.symlink(self.timestamp, latest_link)
        except OSError:
            # Windows may not support symlinks, create a text file instead
            with open(latest_link + ".txt", "w", encoding="utf-8") as f:
                f.write(f"Latest report: {self.timestamp}")
    
    def generate_report(
        self,
        all_results: List[Dict[str, Any]],
        config: Dict[str, Any],
        summary: Dict[str, Any]
    ):
        """
        Generate the complete HTML report.
        
        Args:
            all_results: List of all stock analysis results
            config: Strategy configuration
            summary: Summary statistics
        """
        print(f"Generating HTML report in: {self.report_folder}")
        
        # Generate CSS and JS assets
        self._generate_assets()
        
        # Generate charts for each stock
        self._generate_charts(all_results)
        
        # Generate individual stock pages
        self._generate_stock_pages(all_results, config)
        
        # Generate main dashboard
        self._generate_dashboard(all_results, config, summary)
        
        print(f"✅ HTML report generated: {os.path.abspath(self.report_folder)}/index.html")
    
    def _generate_assets(self):
        """Generate CSS and JavaScript assets."""
        # Generate CSS
        css_content = """
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
        }
        
        .header .timestamp {
            font-size: 1.1rem;
            opacity: 0.9;
        }
        
        .summary-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 1.5rem;
            margin-bottom: 2rem;
        }
        
        .summary-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .summary-card h3 {
            color: #667eea;
            margin-bottom: 0.5rem;
        }
        
        .summary-card .number {
            font-size: 2rem;
            font-weight: bold;
            color: #333;
        }
        
        .stocks-table {
            background: white;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 2rem;
        }
        
        .stocks-table h2 {
            background: #667eea;
            color: white;
            padding: 1rem;
            margin: 0;
        }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }
        
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        
        tr:hover {
            background-color: #f5f5f5;
        }
        
        .status-qualified {
            background-color: #d4edda;
            color: #155724;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        .status-failed {
            background-color: #f8d7da;
            color: #721c24;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
        }
        
        .criteria-pass {
            color: #28a745;
            font-weight: bold;
        }
        
        .criteria-fail {
            color: #dc3545;
            font-weight: bold;
        }
        
        .stock-link {
            color: #667eea;
            text-decoration: none;
            font-weight: bold;
        }
        
        .stock-link:hover {
            text-decoration: underline;
        }
        
        .chart-container {
            text-align: center;
            margin: 2rem 0;
        }
        
        .chart-container img {
            max-width: 100%;
            height: auto;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .criteria-details {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1rem;
            margin: 1.5rem 0;
        }
        
        .criteria-item {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            border-left: 4px solid #ddd;
        }
        
        .criteria-item.pass {
            border-left-color: #28a745;
        }
        
        .criteria-item.fail {
            border-left-color: #dc3545;
        }
        
        .back-link {
            display: inline-block;
            background: #667eea;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 5px;
            margin-bottom: 2rem;
        }
        
        .back-link:hover {
            background: #5a6fd8;
        }
        
        @media (max-width: 768px) {
            .container {
                padding: 10px;
            }
            
            .header h1 {
                font-size: 2rem;
            }
            
            .summary-grid {
                grid-template-columns: 1fr;
            }
            
            table {
                font-size: 0.9rem;
            }
        }
        """
        
        with open(os.path.join(self.report_folder, "assets", "style.css"), "w", encoding="utf-8") as f:
            f.write(css_content)
        
        # Generate JavaScript
        js_content = """
        document.addEventListener('DOMContentLoaded', function() {
            // Add any interactive functionality here
            console.log('Trend Following Report loaded');
        });
        """
        
        with open(os.path.join(self.report_folder, "assets", "script.js"), "w", encoding="utf-8") as f:
            f.write(js_content)
    
    def _generate_charts(self, all_results: List[Dict[str, Any]]):
        """Generate price and volume charts for each stock."""
        print("Generating charts...")
        
        for result in all_results:
            symbol = result['symbol']
            try:
                if 'historical_data' in result:
                    data = result['historical_data']
                    self._create_stock_charts(symbol, data)
            except Exception as e:
                print(f"Warning: Could not generate charts for {symbol}: {e}")
    
    def _create_stock_charts(self, symbol: str, data: pd.DataFrame):
        """Create price and volume charts for a stock."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Price chart
        ax1.plot(data.index, data['close'], linewidth=2, color='#667eea', label='Close Price')
        
        # Add moving average if available
        if len(data) >= 5:
            ma5 = data['close'].rolling(window=5).mean()
            ax1.plot(data.index, ma5, linewidth=1, color='orange', alpha=0.7, label='5-day MA')
        
        ax1.set_title(f'{symbol} - Price Analysis', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Price ($)')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Format x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        # Volume chart
        colors = ['red' if data['close'].iloc[i] < data['close'].iloc[i-1] else 'green' 
                 for i in range(1, len(data))]
        colors.insert(0, 'gray')  # First bar color
        
        ax2.bar(data.index, data['volume'], color=colors, alpha=0.7)
        ax2.set_title('Volume Analysis', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Volume')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        
        # Save chart
        chart_path = os.path.join(self.report_folder, "charts", f"{symbol}_analysis.png")
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_stock_pages(self, all_results: List[Dict[str, Any]], config: Dict[str, Any]):
        """Generate individual HTML pages for each stock."""
        print("Generating individual stock pages...")
        
        for result in all_results:
            symbol = result['symbol']
            self._create_stock_page(result, config)
    
    def _create_stock_page(self, result: Dict[str, Any], config: Dict[str, Any]):
        """Create an individual stock analysis page."""
        symbol = result['symbol']
        
        # Determine failure reasons
        failure_reasons = self._get_failure_reasons(result, config)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{symbol} - Trend Following Analysis</title>
            <link rel="stylesheet" href="../assets/style.css">
        </head>
        <body>
            <div class="container">
                <a href="../index.html" class="back-link">← Back to Dashboard</a>
                
                <div class="header">
                    <h1>{symbol} Analysis</h1>
                    <div class="timestamp">Price: ${result['current_price']:.2f} | Status: {'QUALIFIED' if result['qualifies'] else 'FAILED'}</div>
                </div>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Current Price</h3>
                        <div class="number">${result['current_price']:.2f}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Trend Strength</h3>
                        <div class="number">{result['trend_strength']:.2f}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Volume Strength</h3>
                        <div class="number">{result['volume_strength']:.2f}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Overall Status</h3>
                        <div class="number {'status-qualified' if result['qualifies'] else 'status-failed'}">
                            {'PASS' if result['qualifies'] else 'FAIL'}
                        </div>
                    </div>
                </div>
                
                <div class="chart-container">
                    <img src="../charts/{symbol}_analysis.png" alt="{symbol} Price and Volume Analysis">
                </div>
                
                <div class="stocks-table">
                    <h2>Detailed Criteria Analysis</h2>
                    <div style="padding: 1rem;">
                        <div class="criteria-details">
                            {self._generate_criteria_html(result, config)}
                        </div>
                        
                        {self._generate_failure_reasons_html(failure_reasons)}
                    </div>
                </div>
                
                <div class="stocks-table">
                    <h2>Technical Details</h2>
                    <div style="padding: 1rem;">
                        {self._generate_technical_details_html(result)}
                    </div>
                </div>
            </div>
            <script src="../assets/script.js"></script>
        </body>
        </html>
        """
        
        with open(os.path.join(self.report_folder, "stocks", f"{symbol}.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def _generate_criteria_html(self, result: Dict[str, Any], config: Dict[str, Any]) -> str:
        """Generate HTML for criteria analysis."""
        trend_details = result.get('trend_details', {})
        volume_details = result.get('volume_details', {})
        
        # Trend criteria
        slope_pass = trend_details.get('relative_slope', 0) >= config['min_slope']
        positive_days_pass = trend_details.get('positive_days', 0) >= config['min_positive_days']
        ma_pass = trend_details.get('above_moving_average', False)
        
        # Volume criteria
        volume_pass = volume_details.get('volume_ratio', 0) >= config['volume_threshold']
        
        criteria_html = f"""
        <div class="criteria-item {'pass' if slope_pass else 'fail'}">
            <h4>Price Slope {'✅' if slope_pass else '❌'}</h4>
            <p><strong>Current:</strong> {trend_details.get('relative_slope', 0):.3f}</p>
            <p><strong>Required:</strong> ≥{config['min_slope']:.3f}</p>
        </div>
        
        <div class="criteria-item {'pass' if positive_days_pass else 'fail'}">
            <h4>Positive Days {'✅' if positive_days_pass else '❌'}</h4>
            <p><strong>Current:</strong> {trend_details.get('positive_days', 0)}/{trend_details.get('total_days', 0)}</p>
            <p><strong>Required:</strong> ≥{config['min_positive_days']}</p>
        </div>
        
        <div class="criteria-item {'pass' if ma_pass else 'fail'}">
            <h4>Above Moving Average {'✅' if ma_pass else '❌'}</h4>
            <p><strong>Current Price:</strong> ${result['current_price']:.2f}</p>
            <p><strong>Moving Average:</strong> ${trend_details.get('moving_average', 0):.2f}</p>
        </div>
        
        <div class="criteria-item {'pass' if volume_pass else 'fail'}">
            <h4>Volume Threshold {'✅' if volume_pass else '❌'}</h4>
            <p><strong>Current:</strong> {volume_details.get('volume_ratio', 0):.2f}x</p>
            <p><strong>Required:</strong> ≥{config['volume_threshold']:.2f}x</p>
        </div>
        """
        
        return criteria_html
    
    def _generate_failure_reasons_html(self, failure_reasons: List[str]) -> str:
        """Generate HTML for failure reasons."""
        if not failure_reasons:
            return '<div style="color: green; font-weight: bold; margin-top: 1rem;">✅ All criteria passed!</div>'
        
        reasons_html = '<div style="margin-top: 1.5rem;"><h4 style="color: #dc3545;">Failure Reasons:</h4><ul>'
        for reason in failure_reasons:
            reasons_html += f'<li style="color: #dc3545; margin: 0.5rem 0;">{reason}</li>'
        reasons_html += '</ul></div>'
        
        return reasons_html
    
    def _generate_technical_details_html(self, result: Dict[str, Any]) -> str:
        """Generate HTML for technical details."""
        trend_details = result.get('trend_details', {})
        volume_details = result.get('volume_details', {})
        
        details_html = f"""
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem;">
            <div>
                <h4>Price Analysis</h4>
                <p><strong>Price Change:</strong> {trend_details.get('price_change_pct', 0):.2f}%</p>
                <p><strong>Analysis Period:</strong> {trend_details.get('analysis_period', 0)} days</p>
                <p><strong>Data Points:</strong> {result.get('data_points', 0)}</p>
            </div>
            <div>
                <h4>Volume Analysis</h4>
                <p><strong>Current Volume:</strong> {result['current_volume']:,}</p>
                <p><strong>Average Volume:</strong> {volume_details.get('average_volume', 0):,.0f}</p>
                <p><strong>Volume Period:</strong> {volume_details.get('volume_avg_days', 0)} days</p>
            </div>
        </div>
        """
        
        return details_html
    
    def _get_failure_reasons(self, result: Dict[str, Any], config: Dict[str, Any]) -> List[str]:
        """Get specific failure reasons for a stock."""
        if result['qualifies']:
            return []
        
        reasons = []
        trend_details = result.get('trend_details', {})
        volume_details = result.get('volume_details', {})
        
        # Check trend criteria
        if trend_details.get('relative_slope', 0) < config['min_slope']:
            reasons.append(f"Price slope too low: {trend_details.get('relative_slope', 0):.3f} < {config['min_slope']:.3f}")
        
        if trend_details.get('positive_days', 0) < config['min_positive_days']:
            reasons.append(f"Not enough positive days: {trend_details.get('positive_days', 0)} < {config['min_positive_days']}")
        
        if not trend_details.get('above_moving_average', False):
            reasons.append("Price below moving average")
        
        # Check volume criteria
        if volume_details.get('volume_ratio', 0) < config['volume_threshold']:
            reasons.append(f"Volume too low: {volume_details.get('volume_ratio', 0):.2f}x < {config['volume_threshold']:.2f}x")
        
        return reasons
    
    def _generate_dashboard(
        self,
        all_results: List[Dict[str, Any]],
        config: Dict[str, Any],
        summary: Dict[str, Any]
    ):
        """Generate the main dashboard HTML."""
        qualified_count = len([r for r in all_results if r['qualifies']])
        total_count = len(all_results)
        
        # Generate failure breakdown
        failure_breakdown = self._analyze_failure_patterns(all_results, config)
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Trend Following Analysis - {self.timestamp}</title>
            <link rel="stylesheet" href="assets/style.css">
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>📈 Trend Following Analysis</h1>
                    <div class="timestamp">{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
                </div>
                
                <div class="summary-grid">
                    <div class="summary-card">
                        <h3>Total Analyzed</h3>
                        <div class="number">{total_count}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Qualified</h3>
                        <div class="number" style="color: #28a745;">{qualified_count}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Failed</h3>
                        <div class="number" style="color: #dc3545;">{total_count - qualified_count}</div>
                    </div>
                    <div class="summary-card">
                        <h3>Success Rate</h3>
                        <div class="number">{(qualified_count/total_count*100) if total_count > 0 else 0:.1f}%</div>
                    </div>
                </div>
                
                <div class="stocks-table">
                    <h2>Configuration Summary</h2>
                    <div style="padding: 1rem;">
                        <div class="criteria-details">
                            <div class="criteria-item">
                                <h4>Analysis Period</h4>
                                <p>{config['analysis_days']} days</p>
                            </div>
                            <div class="criteria-item">
                                <h4>Min Price Slope</h4>
                                <p>{config['min_slope']:.3f}</p>
                            </div>
                            <div class="criteria-item">
                                <h4>Min Positive Days</h4>
                                <p>{config['min_positive_days']}</p>
                            </div>
                            <div class="criteria-item">
                                <h4>Volume Threshold</h4>
                                <p>{config['volume_threshold']:.1f}x average</p>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="stocks-table">
                    <h2>Failure Analysis</h2>
                    <div style="padding: 1rem;">
                        {self._generate_failure_breakdown_html(failure_breakdown, total_count)}
                    </div>
                </div>
                
                <div class="stocks-table">
                    <h2>All Stocks Analysis</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Symbol</th>
                                <th>Price</th>
                                <th>Status</th>
                                <th>Trend Score</th>
                                <th>Volume Score</th>
                                <th>Slope</th>
                                <th>Pos Days</th>
                                <th>Vol Ratio</th>
                                <th>Details</th>
                            </tr>
                        </thead>
                        <tbody>
                            {self._generate_stocks_table_html(all_results)}
                        </tbody>
                    </table>
                </div>
            </div>
            <script src="assets/script.js"></script>
        </body>
        </html>
        """
        
        with open(os.path.join(self.report_folder, "index.html"), "w", encoding="utf-8") as f:
            f.write(html_content)
    
    def _analyze_failure_patterns(self, all_results: List[Dict[str, Any]], config: Dict[str, Any]) -> Dict[str, int]:
        """Analyze common failure patterns."""
        failure_counts = {
            'slope_too_low': 0,
            'insufficient_positive_days': 0,
            'below_moving_average': 0,
            'low_volume': 0
        }
        
        for result in all_results:
            if not result['qualifies']:
                trend_details = result.get('trend_details', {})
                volume_details = result.get('volume_details', {})
                
                if trend_details.get('relative_slope', 0) < config['min_slope']:
                    failure_counts['slope_too_low'] += 1
                
                if trend_details.get('positive_days', 0) < config['min_positive_days']:
                    failure_counts['insufficient_positive_days'] += 1
                
                if not trend_details.get('above_moving_average', False):
                    failure_counts['below_moving_average'] += 1
                
                if volume_details.get('volume_ratio', 0) < config['volume_threshold']:
                    failure_counts['low_volume'] += 1
        
        return failure_counts
    
    def _generate_failure_breakdown_html(self, failure_breakdown: Dict[str, int], total_count: int) -> str:
        """Generate HTML for failure breakdown."""
        html = '<div class="criteria-details">'
        
        failure_labels = {
            'slope_too_low': 'Price Slope Too Low',
            'insufficient_positive_days': 'Not Enough Positive Days',
            'below_moving_average': 'Below Moving Average',
            'low_volume': 'Volume Too Low'
        }
        
        for key, count in failure_breakdown.items():
            percentage = (count / total_count * 100) if total_count > 0 else 0
            html += f"""
            <div class="criteria-item">
                <h4>{failure_labels[key]}</h4>
                <p><strong>{count}</strong> stocks ({percentage:.1f}%)</p>
            </div>
            """
        
        html += '</div>'
        return html
    
    def _generate_stocks_table_html(self, all_results: List[Dict[str, Any]]) -> str:
        """Generate HTML for the stocks table."""
        html = ""
        
        # Sort by trend strength descending
        sorted_results = sorted(all_results, key=lambda x: x['trend_strength'], reverse=True)
        
        for result in sorted_results:
            symbol = result['symbol']
            trend_details = result.get('trend_details', {})
            volume_details = result.get('volume_details', {})
            
            status_class = 'status-qualified' if result['qualifies'] else 'status-failed'
            status_text = 'QUALIFIED' if result['qualifies'] else 'FAILED'
            
            html += f"""
            <tr>
                <td><a href="stocks/{symbol}.html" class="stock-link">{symbol}</a></td>
                <td>${result['current_price']:.2f}</td>
                <td><span class="{status_class}">{status_text}</span></td>
                <td>{result['trend_strength']:.2f}</td>
                <td>{result['volume_strength']:.2f}</td>
                <td>{trend_details.get('relative_slope', 0):.3f}</td>
                <td>{trend_details.get('positive_days', 0)}/{trend_details.get('total_days', 0)}</td>
                <td>{volume_details.get('volume_ratio', 0):.2f}x</td>
                <td><a href="stocks/{symbol}.html" class="stock-link">View →</a></td>
            </tr>
            """
        
        return html
