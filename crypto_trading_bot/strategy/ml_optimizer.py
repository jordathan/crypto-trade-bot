"""
Machine Learning optimizer - learns from trading history to improve predictions.
Uses scikit-learn models to predict price direction and optimize strategy.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
import logging
from datetime import datetime
from typing import Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class MLModelTrainer:
    """Trains ML models on historical trading data."""
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize ML trainer.
        
        Args:
            model_type: Type of model ('random_forest', 'gradient_boost')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
    
    def prepare_training_data(
        self,
        price_data: pd.DataFrame,
        trade_history: List[Dict],
        lookback: int = 20
    ) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data from price data and trade history.
        
        Args:
            price_data: DataFrame with OHLCV and technical indicators
            trade_history: List of past trades
            lookback: Number of periods to look back
            
        Returns:
            Tuple of (features, labels, feature_names)
        """
        features_list = []
        labels_list = []
        
        # Define feature columns
        feature_cols = [
            'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR'
        ]
        
        # Add custom features
        if 'Momentum' not in price_data.columns:
            price_data['Momentum'] = (price_data['Close'] - price_data['Close'].shift(20)) / price_data['Close'].shift(20)
        if 'Volatility' not in price_data.columns:
            price_data['Volatility'] = price_data['Close'].pct_change().rolling(window=20).std()
        
        feature_cols.extend(['Momentum', 'Volatility', 'Returns'])
        
        # Create samples
        for i in range(lookback, len(price_data) - 1):
            try:
                # Extract features
                features = []
                for col in feature_cols:
                    if col in price_data.columns:
                        val = price_data.iloc[i][col]
                        features.append(0.0 if pd.isna(val) else val)
                
                # Determine label based on next day's return
                current_price = price_data.iloc[i]['Close']
                next_price = price_data.iloc[i + 1]['Close']
                future_return = (next_price - current_price) / current_price
                
                # Label: 1 for up, 0 for down, -1 for unclear
                if future_return > 0.01:
                    label = 1
                elif future_return < -0.01:
                    label = 0
                else:
                    continue  # Skip unclear signals
                
                features_list.append(features)
                labels_list.append(label)
                
            except Exception as e:
                logger.warning(f"Error preparing training sample: {e}")
                continue
        
        # Ensure we have feature list
        valid_cols = [col for col in feature_cols if col in price_data.columns]
        self.feature_names = valid_cols
        
        return (
            np.array(features_list),
            np.array(labels_list),
            valid_cols
        )
    
    def train(
        self,
        features: np.ndarray,
        labels: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Dict:
        """
        Train ML model.
        
        Args:
            features: Training features
            labels: Training labels
            test_size: Test set size
            random_state: Random seed
            
        Returns:
            Training metrics dictionary
        """
        if len(features) < 20:
            logger.warning("Insufficient training data")
            return {}
        
        try:
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                features, labels,
                test_size=test_size,
                random_state=random_state
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Create and train model
            if self.model_type == 'random_forest':
                self.model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=15,
                    min_samples_split=5,
                    random_state=random_state,
                    n_jobs=-1
                )
            else:
                self.model = GradientBoostingClassifier(
                    n_estimators=100,
                    max_depth=5,
                    learning_rate=0.1,
                    random_state=random_state
                )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            self.is_trained = True
            
            metrics = {
                'train_accuracy': train_score,
                'test_accuracy': test_score,
                'training_samples': len(X_train),
                'test_samples': len(X_test),
                'model_type': self.model_type,
                'trained_at': datetime.now()
            }
            
            logger.info(
                f"Model trained - Train Acc: {train_score:.3f}, "
                f"Test Acc: {test_score:.3f}"
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return {}
    
    def predict(self, features: np.ndarray) -> Tuple[int, float]:
        """
        Make prediction on new data.
        
        Args:
            features: Feature array (1D or 2D)
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained or self.model is None:
            return 0, 0.5  # Default to neutral
        
        try:
            # Ensure 2D array
            if features.ndim == 1:
                features = features.reshape(1, -1)
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.model.predict(features_scaled)[0]
            probabilities = self.model.predict_proba(features_scaled)[0]
            confidence = max(probabilities)
            
            return prediction, confidence
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            return 0, 0.5
    
    def save_model(self, filepath: str):
        """Save trained model to disk."""
        try:
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'model_type': self.model_type
            }
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            logger.info(f"Model saved to {filepath}")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self, filepath: str):
        """Load trained model from disk."""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.model_type = model_data['model_type']
            self.is_trained = True
            logger.info(f"Model loaded from {filepath}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")


class StrategyOptimizer:
    """Optimizes trading parameters based on historical performance."""
    
    def __init__(self):
        """Initialize optimizer."""
        self.optimal_params = {}
        self.param_history: List[Dict] = []
    
    def analyze_trade_profitability(
        self,
        trade_history: List[Dict],
        price_data: Dict[str, pd.DataFrame]
    ) -> Dict:
        """
        Analyze which trading conditions lead to profits.
        
        Args:
            trade_history: List of past trades
            price_data: Price data for each symbol
            
        Returns:
            Analysis results dictionary
        """
        profitable_trades = []
        losing_trades = []
        
        for trade in trade_history:
            try:
                symbol = trade['symbol']
                if symbol not in price_data:
                    continue
                
                data = price_data[symbol]
                
                # Calculate PnL
                entry_price = trade['entry_price']
                exit_price = trade.get('exit_price', entry_price)
                qty = trade['quantity']
                
                pnl = (exit_price - entry_price) * qty
                
                if pnl > 0:
                    profitable_trades.append({
                        **trade,
                        'pnl': pnl,
                        'return_pct': ((exit_price - entry_price) / entry_price) * 100
                    })
                else:
                    losing_trades.append({
                        **trade,
                        'pnl': pnl,
                        'return_pct': ((exit_price - entry_price) / entry_price) * 100
                    })
            except Exception as e:
                logger.warning(f"Error analyzing trade: {e}")
        
        # Calculate win rate
        total_trades = len(profitable_trades) + len(losing_trades)
        win_rate = len(profitable_trades) / total_trades if total_trades > 0 else 0
        
        # Calculate average returns
        avg_profit = np.mean([t['return_pct'] for t in profitable_trades]) if profitable_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        return {
            'win_rate': win_rate,
            'total_trades': total_trades,
            'profitable_trades': len(profitable_trades),
            'losing_trades': len(losing_trades),
            'avg_profit_pct': avg_profit,
            'avg_loss_pct': avg_loss,
            'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else 0,
            'profitable_conditions': self._extract_winning_conditions(profitable_trades),
            'losing_conditions': self._extract_winning_conditions(losing_trades)
        }
    
    @staticmethod
    def _extract_winning_conditions(trades: List[Dict]) -> Dict:
        """Extract common conditions from winning trades."""
        if not trades:
            return {}
        
        conditions = {}
        
        # Find most common indicators for winning trades
        sentiments = [t.get('sentiment_score', 0) for t in trades if 'sentiment_score' in t]
        conditions['avg_sentiment'] = np.mean(sentiments) if sentiments else 0
        
        return conditions
