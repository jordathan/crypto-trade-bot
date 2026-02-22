#!/usr/bin/env python3
"""Debug script to check signal generation."""

from data.collectors import CryptoDataCollector
from strategy.trading_engine import TradingLogic, TradeSignal
import numpy as np
import pandas as pd

# Fetch sample data
collector = CryptoDataCollector(lookback_days=90)
data = collector.fetch_crypto_data('BTC-USD')

print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")

if not data.empty and len(data) >= 50:
    latest = data.iloc[-1]
    print(f"\nLatest data:")
    print(f"Close: {latest.get('Close')}")
    print(f"RSI: {latest.get('RSI')}")
    print(f"EMA_12: {latest.get('EMA_12')}")
    print(f"EMA_26: {latest.get('EMA_26')}")
    print(f"MACD: {latest.get('MACD')}")
    print(f"MACD_Signal: {latest.get('MACD_Signal')}")
    
    # Test signal generation
    logic = TradingLogic(min_confidence=0.3)
    signal, conf = logic.generate_signal(data, 0.0, 0.5)
    print(f"\nTrading logic signal: {signal.name} (value={signal.value}), confidence={conf}")
    
    # Test simplified signal
    signal_score = 0.0
    try:
        rsi = float(latest.get('RSI')) if pd.notna(latest.get('RSI')) else np.nan
        print(f"\nRSI check: {rsi}")
        if not np.isnan(rsi):
            if rsi < 35:
                signal_score += 0.4
                print(f"  RSI < 35: signal_score += 0.4")
            elif rsi > 65:
                signal_score -= 0.4
                print(f"  RSI > 65: signal_score -= 0.4")
    except Exception as e:
        print(f"Error checking RSI: {e}")
    
    print(f"\nFinal signal_score: {signal_score}")
    
    if signal_score > 0.5:
        print(f"Result: STRONG_BUY")
    elif signal_score > 0.2:
        print(f"Result: BUY")
    else:
        print(f"Result: HOLD")
