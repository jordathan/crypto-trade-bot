"""
Trading bot logging system - tracks all trades, decisions, and performance.
Saves verbose logs for analysis and model improvement.
"""

import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import os


class TradeLogger:
    """Logs all trading activity and decisions."""
    
    def __init__(self, log_dir: str = 'logs'):
        """
        Initialize trade logger.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Setup file logging
        self.logger = logging.getLogger('TradeBot')
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = self.log_dir / f'bot_{datetime.now().strftime("%Y%m%d")}.log'
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.DEBUG)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        
        # Trades file
        self.trades_file = self.log_dir / f'trades_{datetime.now().strftime("%Y%m%d")}.json'
        self.trades_data: List[Dict] = []
        
        # Load existing trades if any
        if self.trades_file.exists():
            try:
                with open(self.trades_file, 'r') as f:
                    self.trades_data = json.load(f)
            except:
                self.trades_data = []
    
    def log_signal(
        self,
        symbol: str,
        signal: str,
        confidence: float,
        price: float,
        indicators: Dict[str, float],
        sentiment_data: Dict = None
    ):
        """Log trading signal generation."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'SIGNAL_GENERATED',
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'price': price,
            'indicators': indicators,
            'sentiment_data': sentiment_data or {}
        }
        
        self.trades_data.append(log_entry)
        self.logger.info(
            f"Signal: {signal} for {symbol} @ ${price} (Conf: {confidence:.2%})"
        )
        self._save_trades()
    
    def log_trade(
        self,
        trade_type: str,  # BUY or SELL
        symbol: str,
        quantity: float,
        price: float,
        reason: str,
        portfolio_value: float,
        position_size_pct: float
    ):
        """Log executed trade."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'TRADE_EXECUTED',
            'type': trade_type,
            'symbol': symbol,
            'quantity': quantity,
            'price': price,
            'total_value': quantity * price,
            'reason': reason,
            'portfolio_value': portfolio_value,
            'position_size_pct': position_size_pct
        }
        
        self.trades_data.append(log_entry)
        self.logger.info(
            f"TRADE: {trade_type} {quantity} {symbol} @ ${price} - {reason}"
        )
        self._save_trades()
    
    def log_decision_skip(
        self,
        symbol: str,
        reason: str,
        signal: str = None,
        confidence: float = None
    ):
        """Log skipped trade decision."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'DECISION_SKIPPED',
            'symbol': symbol,
            'signal': signal,
            'confidence': confidence,
            'reason': reason
        }
        
        self.trades_data.append(log_entry)
        self.logger.debug(f"Skipped {symbol}: {reason}")
        self._save_trades()
    
    def log_error(self, symbol: str, error: str, traceback: str = None):
        """Log errors during trading."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': 'ERROR',
            'symbol': symbol,
            'error': error,
            'traceback': traceback
        }
        
        self.trades_data.append(log_entry)
        self.logger.error(f"Error for {symbol}: {error}\n{traceback or ''}")
        self._save_trades()
    
    def _save_trades(self):
        """Save trades to JSON file."""
        try:
            with open(self.trades_file, 'w') as f:
                json.dump(self.trades_data, f, indent=2, default=str)
        except Exception as e:
            self.logger.error(f"Error saving trades: {e}")
    
    def get_trades(self) -> List[Dict]:
        """Get all logged trades."""
        return self.trades_data.copy()


class PerformanceTracker:
    """Tracks portfolio performance metrics."""
    
    def __init__(self, log_dir: str = 'logs'):
        """
        Initialize performance tracker.
        
        Args:
            log_dir: Directory for log files
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.perf_file = self.log_dir / f'performance_{datetime.now().strftime("%Y%m%d")}.csv'
        self.performance_data: List[Dict] = []
        
        # Load existing performance data if any
        if self.perf_file.exists():
            try:
                df = pd.read_csv(self.perf_file)
                self.performance_data = df.to_dict('records')
            except:
                self.performance_data = []
    
    def record_snapshot(
        self,
        symbol: str,
        portfolio_value: float,
        cash_available: float,
        num_positions: int,
        num_trades: int,
        daily_return_pct: float,
        win_rate: float,
        sharpe_ratio: float = None,
        max_drawdown_pct: float = None
    ):
        """Record performance snapshot."""
        snapshot = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'portfolio_value': portfolio_value,
            'cash_available': cash_available,
            'num_positions': num_positions,
            'num_trades': num_trades,
            'daily_return_pct': daily_return_pct,
            'win_rate': win_rate,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown_pct
        }
        
        self.performance_data.append(snapshot)
        self._save_performance()
    
    def _save_performance(self):
        """Save performance data to CSV."""
        try:
            df = pd.DataFrame(self.performance_data)
            df.to_csv(self.perf_file, index=False)
        except Exception as e:
            logging.error(f"Error saving performance: {e}")
    
    def get_daily_stats(self) -> Dict:
        """Get today's performance statistics."""
        if not self.performance_data:
            return {}
        
        df = pd.DataFrame(self.performance_data)
        
        return {
            'avg_portfolio_value': df['portfolio_value'].mean(),
            'max_portfolio_value': df['portfolio_value'].max(),
            'min_portfolio_value': df['portfolio_value'].min(),
            'total_trades': df['num_trades'].max() or 0,
            'avg_win_rate': df['win_rate'].mean(),
            'total_daily_return_pct': df['daily_return_pct'].sum()
        }
    
    def get_performance_df(self) -> pd.DataFrame:
        """Get performance data as DataFrame."""
        return pd.DataFrame(self.performance_data)


class TradeAnalyzer:
    """Analyzes trading performance and generates insights."""
    
    def __init__(self, log_dir: str = 'logs'):
        """Initialize analyzer."""
        self.log_dir = Path(log_dir)
    
    def analyze_trade_quality(self, trades: List[Dict]) -> Dict:
        """
        Analyze quality of trades executed.
        
        Args:
            trades: List of trades
            
        Returns:
            Analysis metrics
        """
        buy_trades = [t for t in trades if t.get('type') == 'BUY']
        sell_trades = [t for t in trades if t.get('type') == 'SELL']
        
        win_trades = len([t for t in trades if t.get('pnl', 0) > 0])
        total_trades = len(trades)
        
        return {
            'total_trades': total_trades,
            'buy_trades': len(buy_trades),
            'sell_trades': len(sell_trades),
            'win_rate': win_trades / total_trades if total_trades > 0 else 0,
            'avg_trade_value': np.mean([t.get('total_value', 0) for t in trades]) if trades else 0
        }
    
    def get_top_performers(self, trades: List[Dict], top_n: int = 5) -> List[Dict]:
        """Get best performing trades."""
        profitable = sorted(
            [t for t in trades if t.get('pnl', 0) > 0],
            key=lambda x: x.get('pnl', 0),
            reverse=True
        )
        return profitable[:top_n]
    
    def get_biggest_losses(self, trades: List[Dict], top_n: int = 5) -> List[Dict]:
        """Get biggest losing trades."""
        losing = sorted(
            [t for t in trades if t.get('pnl', 0) < 0],
            key=lambda x: x.get('pnl', 0)
        )
        return losing[:top_n]


import numpy as np
