"""
Trading strategy engine - core logic for buy/sell decisions.
Uses technical analysis, sentiment, and ML predictions.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class TradeSignal(Enum):
    """Trade action signals."""
    STRONG_BUY = 3
    BUY = 2
    HOLD = 1
    SELL = -2
    STRONG_SELL = -3
    NO_SIGNAL = 0


class TradingLogic:
    """Core trading logic with technical and sentiment analysis."""
    
    def __init__(
        self,
        target_return: float = 0.02,
        max_loss: float = 0.01,
        min_confidence: float = 0.55
    ):
        """
        Initialize trading logic.
        
        Args:
            target_return: Target daily return (default 2%)
            max_loss: Maximum acceptable loss per trade
            min_confidence: Minimum confidence threshold for trades
        """
        self.target_return = target_return
        self.max_loss = max_loss
        self.min_confidence = min_confidence
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        sentiment_score: float = 0.0,
        ml_prediction: float = 0.5
    ) -> Tuple[TradeSignal, float]:
        """
        Generate trading signal based on technical + sentiment analysis.
        
        Args:
            data: Price data with technical indicators
            sentiment_score: Sentiment score (-1 to 1)
            ml_prediction: ML confidence (0 to 1)
            
        Returns:
            Tuple of (TradeSignal, confidence_score)
        """
        if data.empty or len(data) < 50:
            return TradeSignal.NO_SIGNAL, 0.0
        
        latest = data.iloc[-1]
        
        # Get technical signals
        tech_signal, tech_conf = self._get_technical_signal(data)
        
        # Combine all indicators
        combined_signal = self._combine_signals(
            tech_signal,
            sentiment_score,
            ml_prediction,
            tech_conf
        )
        
        return combined_signal
    
    def _get_technical_signal(self, data: pd.DataFrame) -> Tuple[float, float]:
        """
        Generate signal from technical indicators.
        
        Returns:
            Tuple of (signal_score, confidence)
        """
        try:
            latest = data.iloc[-1]
            signal_score = 0.0
            confidence = 0.5
            
            # 1. EMA Crossover
            try:
                if 'EMA_12' in data.columns and 'EMA_26' in data.columns:
                    ema12_val = latest['EMA_12']
                    ema26_val = latest['EMA_26']
                    if not np.isnan(ema12_val) and not np.isnan(ema26_val):
                        if ema12_val > ema26_val:
                            signal_score += 0.2
                        else:
                            signal_score -= 0.2
            except (ValueError, TypeError, KeyError):
                pass
            
            # 2. RSI Signal
            try:
                if 'RSI' in data.columns:
                    rsi_val = latest['RSI']
                    if not np.isnan(rsi_val):
                        if rsi_val < 30:
                            signal_score += 0.15
                            confidence += 0.1
                        elif rsi_val > 70:
                            signal_score -= 0.15
                            confidence += 0.1
            except (ValueError, TypeError, KeyError):
                pass
            
            # 3. MACD Signal
            try:
                if 'MACD' in data.columns and 'MACD_Signal' in data.columns:
                    macd_val = latest['MACD']
                    macd_sig = latest['MACD_Signal']
                    if not np.isnan(macd_val) and not np.isnan(macd_sig):
                        if macd_val > macd_sig:
                            signal_score += 0.15
                        else:
                            signal_score -= 0.15
            except (ValueError, TypeError, KeyError):
                pass
            
            # 4. Bollinger Bands
            try:
                if 'BB_Lower' in data.columns and 'BB_Upper' in data.columns and 'Close' in data.columns:
                    bb_lower = latest['BB_Lower']
                    bb_upper = latest['BB_Upper']
                    price = latest['Close']
                    if not np.isnan(bb_lower) and not np.isnan(bb_upper) and not np.isnan(price):
                        if price < bb_lower:
                            signal_score += 0.15
                            confidence += 0.1
                        elif price > bb_upper:
                            signal_score -= 0.15
                            confidence += 0.1
            except (ValueError, TypeError, KeyError):
                pass
            
            # 5. Moving Average Trend
            try:
                if 'SMA_10' in data.columns and 'SMA_20' in data.columns and 'SMA_50' in data.columns:
                    sma10 = latest['SMA_10']
                    sma20 = latest['SMA_20']
                    sma50 = latest['SMA_50']
                    if not np.isnan(sma10) and not np.isnan(sma20) and not np.isnan(sma50):
                        if sma10 > sma20 and sma20 > sma50:
                            signal_score += 0.1
                        elif sma10 < sma20 and sma20 < sma50:
                            signal_score -= 0.1
            except (ValueError, TypeError, KeyError):
                pass
            
            # 6. Momentum
            try:
                if len(data) >= 20 and 'Close' in data.columns:
                    recent_close = latest['Close']
                    past_close = data.iloc[-20]['Close']
                    if not np.isnan(recent_close) and not np.isnan(past_close) and past_close != 0:
                        recent_return = (recent_close - past_close) / past_close
                        if recent_return > 0.05:
                            signal_score += 0.1
                        elif recent_return < -0.05:
                            signal_score -= 0.1
            except (ValueError, TypeError, KeyError):
                pass
            
            return signal_score, min(confidence, 1.0)
        except Exception as e:
            logger.warning(f"Error calculating technical signal: {e}")
            return 0.0, 0.5
    
    def _combine_signals(
        self,
        tech_signal: float,
        sentiment: float,
        ml_prediction: float,
        tech_conf: float
    ) -> Tuple[TradeSignal, float]:
        """
        Combine technical, sentiment, and ML signals.
        
        Returns:
            Tuple of (TradeSignal, confidence)
        """
        # Weighted combination
        weights = {
            'technical': 0.5,
            'sentiment': 0.3,
            'ml': 0.2
        }
        
        # Normalize ML prediction to -1 to 1
        ml_signal = (ml_prediction - 0.5) * 2
        
        combined = (
            tech_signal * weights['technical'] +
            sentiment * weights['sentiment'] +
            ml_signal * weights['ml']
        )
        
        # Calculate confidence
        confidence = (tech_conf + abs(sentiment) + (ml_prediction if ml_prediction > 0.5 else 1 - ml_prediction)) / 3
        
        # Convert to signal
        if combined > 0.3 and confidence > self.min_confidence:
            signal = TradeSignal.STRONG_BUY if combined > 0.5 else TradeSignal.BUY
        elif combined < -0.3 and confidence > self.min_confidence:
            signal = TradeSignal.STRONG_SELL if combined < -0.5 else TradeSignal.SELL
        else:
            signal = TradeSignal.HOLD
        
        return signal, min(confidence, 1.0)
    
    def calculate_position_size(
        self,
        capital: float,
        signal_confidence: float,
        volatility: float
    ) -> float:
        """
        Calculate position size based on confidence and volatility.
        
        Args:
            capital: Available capital
            signal_confidence: Signal confidence (0-1)
            volatility: Price volatility
            
        Returns:
            Position size in dollars
        """
        # Risk-based position sizing
        base_position = capital * 0.1  # 10% of capital
        
        # Adjust for confidence
        position = base_position * signal_confidence
        
        # Adjust for volatility (lower position on high volatility)
        volatility_factor = max(0.5, 1 - (volatility * 2))
        position = position * volatility_factor
        
        return max(position, capital * 0.01)  # Min 1% of capital
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        signal: TradeSignal,
        volatility: float
    ) -> float:
        """
        Calculate stop loss price.
        
        Args:
            entry_price: Trade entry price
            signal: Trade signal
            volatility: Price volatility
            
        Returns:
            Stop loss price
        """
        # Use volatility-adjusted stop loss
        stop_distance = entry_price * (self.max_loss + volatility)
        
        if signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY]:
            return entry_price - stop_distance
        else:
            return entry_price + stop_distance
    
    def calculate_take_profit(
        self,
        entry_price: float,
        signal: TradeSignal,
        volatility: float
    ) -> float:
        """
        Calculate take profit price.
        
        Args:
            entry_price: Trade entry price
            signal: Trade signal
            volatility: Price volatility
            
        Returns:
            Take profit price
        """
        tp_distance = entry_price * (self.target_return + volatility * 0.5)
        
        if signal in [TradeSignal.BUY, TradeSignal.STRONG_BUY]:
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance


class SimulatedPortfolio:
    """Simulates a trading portfolio for backtesting."""
    
    def __init__(self, initial_capital: float = 1000.0):
        """Initialize portfolio."""
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}  # symbol -> {qty, entry_price, entry_time}
        self.trade_history: List[Dict] = []
        self.balance_history: List[Dict] = []
    
    def buy(
        self,
        symbol: str,
        price: float,
        amount: float,
        timestamp: datetime
    ) -> bool:
        """Execute buy order."""
        cost = amount / price
        
        if cost > self.cash:
            logger.warning(f"Insufficient cash for buy: {symbol}")
            return False
        
        self.cash -= cost
        
        if symbol in self.positions:
            self.positions[symbol]['qty'] += amount
        else:
            self.positions[symbol] = {
                'qty': amount,
                'entry_price': price,
                'entry_time': timestamp
            }
        
        self.trade_history.append({
            'timestamp': timestamp,
            'action': 'BUY',
            'symbol': symbol,
            'price': price,
            'amount': amount,
            'cash_before': self.cash + cost,
            'cash_after': self.cash
        })
        
        return True
    
    def sell(
        self,
        symbol: str,
        price: float,
        amount: float,
        timestamp: datetime
    ) -> bool:
        """Execute sell order."""
        if symbol not in self.positions or self.positions[symbol]['qty'] < amount:
            logger.warning(f"Insufficient position for sell: {symbol}")
            return False
        
        proceeds = amount * price
        self.cash += proceeds
        self.positions[symbol]['qty'] -= amount
        
        if self.positions[symbol]['qty'] == 0:
            entry_price = self.positions[symbol]['entry_price']
            entry_time = self.positions[symbol]['entry_time']
            profit_pct = ((price - entry_price) / entry_price) * 100
            
            del self.positions[symbol]
            
            logger.info(
                f"Position closed: {symbol} | "
                f"Profit: {profit_pct:.2f}% | "
                f"Duration: {(timestamp - entry_time).days}d"
            )
        
        self.trade_history.append({
            'timestamp': timestamp,
            'action': 'SELL',
            'symbol': symbol,
            'price': price,
            'amount': amount,
            'cash_before': self.cash - proceeds,
            'cash_after': self.cash
        })
        
        return True
    
    def get_portfolio_value(self, current_prices: Dict[str, float]) -> float:
        """
        Get total portfolio value.
        
        Args:
            current_prices: Dict of symbol -> current price
            
        Returns:
            Total portfolio value in USD
        """
        positions_value = 0.0
        for symbol, position in self.positions.items():
            if symbol in current_prices:
                positions_value += position['qty'] * current_prices[symbol]
        
        return self.cash + positions_value
    
    def get_portfolio_stats(self, current_prices: Dict[str, float]) -> Dict:
        """Get portfolio statistics."""
        total_value = self.get_portfolio_value(current_prices)
        return_pct = ((total_value - self.initial_capital) / self.initial_capital) * 100
        
        return {
            'initial_capital': self.initial_capital,
            'current_value': total_value,
            'cash': self.cash,
            'return_pct': return_pct,
            'num_positions': len(self.positions),
            'num_trades': len(self.trade_history)
        }
