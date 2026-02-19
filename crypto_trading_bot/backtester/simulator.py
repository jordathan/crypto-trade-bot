"""
Backtesting engine - simulates trading strategy on historical data.
Used for strategy validation and parameter optimization.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class BacktestEngine:
    """Run backtest simulations on historical data."""
    
    def __init__(self, initial_capital: float = 1000.0):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting capital
        """
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
    
    def run_backtest(
        self,
        symbol: str,
        price_data: pd.DataFrame,
        signal_generator,
        sentiment_data: Dict = None,
        ml_predictor = None
    ) -> Dict:
        """
        Run backtest on price data.
        
        Args:
            symbol: Crypto symbol
            price_data: Historical OHLCV data
            signal_generator: Function to generate trading signals
            sentiment_data: Historical sentiment data
            ml_predictor: ML model for predictions
            
        Returns:
            Backtest results
        """
        if price_data.empty or len(price_data) < 50:
            logger.warning("Insufficient data for backtest")
            return {}
        
        # Reset
        self.capital = self.initial_capital
        self.cash = self.initial_capital
        self.positions = {}
        self.trades = []
        self.equity_curve = []
        self.daily_returns = []
        
        # Run simulation
        for i in range(50, len(price_data)):
            date = price_data.index[i]
            current_row = price_data.iloc[i]
            current_price = current_row['Close']
            
            # Get sentiment if available
            sentiment = 0.0
            if sentiment_data and date in sentiment_data:
                sentiment = sentiment_data[date].get('sentiment_score', 0)
            
            # Get ML prediction if available
            ml_prob = 0.5
            if ml_predictor:
                try:
                    features = self._extract_features(price_data, i)
                    _, ml_prob = ml_predictor(features)
                except:
                    ml_prob = 0.5
            
            # Generate signal
            signal, confidence = signal_generator(
                price_data.iloc[:i+1],
                sentiment,
                ml_prob
            )
            
            # Execute trades based on signal
            self._execute_signal(
                symbol,
                signal,
                confidence,
                current_price,
                date,
                current_row
            )
            
            # Record equity
            portfolio_value = self._calculate_portfolio_value(current_price)
            self.equity_curve.append({
                'date': date,
                'value': portfolio_value,
                'cash': self.cash,
                'price': current_price
            })
        
        # Calculate results
        return self._calculate_results()
    
    def _execute_signal(
        self,
        symbol: str,
        signal,
        confidence: float,
        price: float,
        date,
        row: pd.Series
    ):
        """Execute trades based on signal."""
        # Size calculation
        position_size = self._calculate_position_size(confidence)
        
        # BUY signal
        if signal.value > 1 and confidence > 0.55:
            if self.cash >= position_size:
                qty = position_size / price
                self.cash -= position_size
                self.positions[symbol] = {
                    'qty': qty,
                    'entry_price': price,
                    'entry_date': date,
                    'entry_signal': signal.name
                }
                self.trades.append({
                    'date': date,
                    'type': 'BUY',
                    'symbol': symbol,
                    'price': price,
                    'quantity': qty,
                    'reason': signal.name,
                    'confidence': confidence
                })
        
        # SELL signal
        elif signal.value < -1 and confidence > 0.55:
            if symbol in self.positions:
                position = self.positions[symbol]
                proceeds = position['qty'] * price
                
                pnl = proceeds - (position['qty'] * position['entry_price'])
                pnl_pct = (pnl / (position['qty'] * position['entry_price'])) * 100
                
                self.cash += proceeds
                self.trades.append({
                    'date': date,
                    'type': 'SELL',
                    'symbol': symbol,
                    'price': price,
                    'quantity': position['qty'],
                    'entry_price': position['entry_price'],
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'hold_days': (date - position['entry_date']).days,
                    'reason': signal.name,
                    'confidence': confidence
                })
                del self.positions[symbol]
    
    def _calculate_position_size(self, confidence: float) -> float:
        """Calculate position size based on confidence."""
        base_size = self.capital * 0.1
        return base_size * min(confidence, 1.0)
    
    def _calculate_portfolio_value(self, current_price: float) -> float:
        """Calculate total portfolio value."""
        positions_value = 0.0
        for symbol, position in self.positions.items():
            positions_value += position['qty'] * current_price
        
        return self.cash + positions_value
    
    def _extract_features(self, data: pd.DataFrame, index: int) -> np.ndarray:
        """Extract features for ML prediction."""
        row = data.iloc[index]
        features = []
        
        for col in ['SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
                    'MACD', 'MACD_Signal', 'RSI', 'ATR']:
            if col in data.columns:
                features.append(row[col] if pd.notna(row[col]) else 0.0)
        
        return np.array(features)
    
    def _calculate_results(self) -> Dict:
        """Calculate backtest results."""
        if not self.equity_curve:
            return {}
        
        df_equity = pd.DataFrame(self.equity_curve)
        df_equity['return'] = df_equity['value'].pct_change()
        
        # Close any open positions
        final_value = self._calculate_portfolio_value(df_equity.iloc[-1]['price'])
        
        # Calculate metrics
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Win rate
        closed_trades = [t for t in self.trades if 'pnl' in t]
        win_rate = len([t for t in closed_trades if t.get('pnl', 0) > 0]) / len(closed_trades) if closed_trades else 0
        
        # Sharpe Ratio
        daily_returns = df_equity['return'].dropna()
        sharpe = daily_returns.mean() / daily_returns.std() * np.sqrt(252) if len(daily_returns) > 0 else 0
        
        # Max Drawdown
        cummax = df_equity['value'].cummax()
        drawdown = (df_equity['value'] - cummax) / cummax
        max_drawdown = drawdown.min()
        
        # Profit Factor
        gross_profit = sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) > 0])
        gross_loss = abs(sum([t.get('pnl', 0) for t in closed_trades if t.get('pnl', 0) < 0]))
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else 0
        
        return {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'total_return_pct': total_return * 100,
            'daily_return_pct': (total_return / len(df_equity)) * 100,
            'num_trades': len(self.trades),
            'num_closed_trades': len(closed_trades),
            'win_rate': win_rate,
            'sharpe_ratio': sharpe,
            'max_drawdown_pct': max_drawdown * 100,
            'profit_factor': profit_factor,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'equity_curve': df_equity,
            'trades': self.trades
        }


class MultiSymbolBacktester:
    """Run backtest on multiple symbols."""
    
    def __init__(self, initial_capital: float = 1000.0):
        """Initialize."""
        self.initial_capital = initial_capital
        self.engine = BacktestEngine(initial_capital)
    
    def backtest_symbols(
        self,
        symbols: List[str],
        data: Dict[str, pd.DataFrame],
        signal_generator,
        sentiment_data: Dict = None
    ) -> Dict[str, Dict]:
        """
        Backtest multiple symbols.
        
        Args:
            symbols: List of symbols to test
            data: Dict of symbol -> price data
            signal_generator: Signal generation function
            sentiment_data: Historical sentiment data
            
        Returns:
            Dict of symbol -> backtest results
        """
        results = {}
        
        for symbol in symbols:
            if symbol in data and not data[symbol].empty:
                logger.info(f"Backtesting {symbol}")
                
                symbol_sentiment = None
                if sentiment_data and symbol in sentiment_data:
                    symbol_sentiment = sentiment_data[symbol]
                
                results[symbol] = self.engine.run_backtest(
                    symbol,
                    data[symbol],
                    signal_generator,
                    symbol_sentiment
                )
        
        return results
