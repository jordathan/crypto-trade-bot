"""
Data collection module for historical crypto prices and market data.
Handles fetching data from yfinance and market cap rankings.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)

# Top 100 cryptocurrencies by market cap (ticker symbols for yfinance)
TOP_100_CRYPTOS = [
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'XRP-USD', 'SOL-USD', 'ADA-USD', 'DOGE-USD',
    'AVAX-USD', 'LINK-USD', 'MATIC-USD', 'LTC-USD', 'ATOM-USD', 'FIL-USD', 'UNI-USD',
    'NEAR-USD', 'ARB-USD', 'HBAR-USD', 'STX-USD', 'SUI-USD', 'OP-USD', 'PEPE-USD',
    'APE-USD', 'GALA-USD', 'SAND-USD', 'MANA-USD', 'RENDER-USD', 'CANTO-USD', 'FLOW-USD',
    'GMT-USD', 'FTT-USD', 'KAVA-USD', 'BLUR-USD', 'IMX-USD', 'OKB-USD', 'TRX-USD',
    'XLM-USD', 'THETA-USD', 'ICP-USD', 'DYM-USD', 'TAO-USD', 'ETC-USD', 'CRO-USD',
    'VET-USD', 'FLOKI-USD', 'JTO-USD', 'XEC-USD', 'ORDI-USD', 'INJ-USD', 'BRETT-USD',
    'WIF-USD', 'JUP-USD', 'BONK-USD', 'MEW-USD', 'RUNE-USD', 'TON-USD', 'GRT-USD',
    'AEVO-USD', 'POPCAT-USD', 'PIXEL-USD', 'CHZ-USD', 'ROSE-USD', 'IOTA-USD',
]


class CryptoDataCollector:
    """Collects historical crypto price data from yfinance."""
    
    def __init__(self, lookback_days: int = 90):
        """
        Initialize the data collector.
        
        Args:
            lookback_days: Number of historical days to fetch
        """
        self.lookback_days = lookback_days
        self.end_date = datetime.now()
        self.start_date = self.end_date - timedelta(days=lookback_days)
    
    def fetch_crypto_data(self, symbol: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol (e.g., 'BTC-USD')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            logger.info(f"Fetching data for {symbol}")
            data = yf.download(
                symbol,
                start=self.start_date,
                end=self.end_date,
                progress=False
            )
            
            if data.empty:
                logger.warning(f"No data found for {symbol}")
                return pd.DataFrame()
            
            # Add technical indicators
            data = self._add_technical_indicators(data)
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_cryptos(self, symbols: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple cryptocurrencies.
        
        Args:
            symbols: List of crypto symbols
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        data = {}
        for symbol in symbols:
            data[symbol] = self.fetch_crypto_data(symbol)
        return data
    
    def get_top_100_cryptos(self) -> List[str]:
        """Get list of top 100 cryptocurrencies."""
        return TOP_100_CRYPTOS.copy()
    
    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for a cryptocurrency.
        
        Args:
            symbol: Crypto symbol
            
        Returns:
            Current price in USD
        """
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period='1d')
            if not data.empty:
                return data['Close'].iloc[-1]
            return 0.0
        except Exception as e:
            logger.error(f"Error fetching current price for {symbol}: {e}")
            return 0.0
    
    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicators to price data.
        
        Args:
            data: OHLCV DataFrame
            
        Returns:
            DataFrame with added indicators
        """
        # Simple Moving Averages
        data['SMA_10'] = data['Close'].rolling(window=10).mean()
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Exponential Moving Averages
        data['EMA_12'] = data['Close'].ewm(span=12).mean()
        data['EMA_26'] = data['Close'].ewm(span=26).mean()
        
        # MACD
        data['MACD'] = data['EMA_12'] - data['EMA_26']
        data['MACD_Signal'] = data['MACD'].ewm(span=9).mean()
        data['MACD_Histogram'] = data['MACD'] - data['MACD_Signal']
        
        # RSI (Relative Strength Index)
        data['RSI'] = self._calculate_rsi(data['Close'])
        
        # Bollinger Bands
        data['BB_Middle'] = data['Close'].rolling(window=20).mean()
        data['BB_Std'] = data['Close'].rolling(window=20).std()
        data['BB_Upper'] = data['BB_Middle'] + (data['BB_Std'] * 2)
        data['BB_Lower'] = data['BB_Middle'] - (data['BB_Std'] * 2)
        
        # ATR (Average True Range)
        data['ATR'] = self._calculate_atr(data)
        
        # Daily returns
        data['Returns'] = data['Close'].pct_change()
        
        return data
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _calculate_atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate ATR indicator."""
        data['TR'] = np.maximum(
            data['High'] - data['Low'],
            np.maximum(
                abs(data['High'] - data['Close'].shift()),
                abs(data['Low'] - data['Close'].shift())
            )
        )
        return data['TR'].rolling(window=period).mean()


class MarketDataProcessor:
    """Process and normalize market data for trading."""
    
    @staticmethod
    def normalize_data(data: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize price data to 0-1 range.
        
        Args:
            data: Raw OHLCV data
            
        Returns:
            Normalized DataFrame
        """
        normalized = data.copy()
        
        # Normalize price columns
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in normalized.columns:
                normalized[col] = (normalized[col] - normalized[col].min()) / (
                    normalized[col].max() - normalized[col].min()
                )
        
        return normalized
    
    @staticmethod
    def get_feature_matrix(data: pd.DataFrame) -> Tuple[np.ndarray, pd.Index]:
        """
        Extract feature matrix for ML models.
        
        Args:
            data: Price data with technical indicators
            
        Returns:
            Tuple of (feature matrix, feature names)
        """
        features = [
            'Returns', 'SMA_10', 'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Histogram', 'RSI',
            'BB_Upper', 'BB_Middle', 'BB_Lower', 'ATR'
        ]
        
        available_features = [f for f in features if f in data.columns]
        feature_data = data[available_features].dropna()
        
        return feature_data.values, available_features
    
    @staticmethod
    def calculate_price_momentum(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate price momentum."""
        return (data['Close'] - data['Close'].shift(period)) / data['Close'].shift(period)
    
    @staticmethod
    def calculate_volatility(data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate rolling volatility."""
        return data['Returns'].rolling(window=period).std()
