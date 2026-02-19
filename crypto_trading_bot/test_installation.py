"""
Quick test script to verify all components are working.
Run this to validate the installation.
"""

import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    try:
        from data.collectors import CryptoDataCollector, MarketDataProcessor
        print("✓ Data collectors")
    except Exception as e:
        print(f"✗ Data collectors: {e}")
        return False
    
    try:
        from data.sentiment import SentimentAggregator
        print("✓ Sentiment analysis")
    except Exception as e:
        print(f"✗ Sentiment analysis: {e}")
        return False
    
    try:
        from strategy.trading_engine import TradingLogic, SimulatedPortfolio
        print("✓ Trading engine")
    except Exception as e:
        print(f"✗ Trading engine: {e}")
        return False
    
    try:
        from strategy.ml_optimizer import MLModelTrainer
        print("✓ ML optimizer")
    except Exception as e:
        print(f"✗ ML optimizer: {e}")
        return False
    
    try:
        from strategy.logging import TradeLogger, PerformanceTracker
        print("✓ Logging system")
    except Exception as e:
        print(f"✗ Logging system: {e}")
        return False
    
    try:
        from backtester.simulator import BacktestEngine
        print("✓ Backtest engine")
    except Exception as e:
        print(f"✗ Backtest engine: {e}")
        return False
    
    return True


def test_data_fetching():
    """Test fetching crypto data."""
    print("\nTesting data fetching...")
    
    try:
        from data.collectors import CryptoDataCollector
        
        collector = CryptoDataCollector(lookback_days=30)
        print("Fetching BTC-USD data...")
        
        data = collector.fetch_crypto_data('BTC-USD')
        
        if data is not None and len(data) > 0:
            print(f"✓ Successfully fetched {len(data)} days of data")
            print(f"  Latest close: ${data.iloc[-1]['Close']:.2f}")
            return True
        else:
            print("✗ No data fetched")
            return False
    
    except Exception as e:
        print(f"✗ Error fetching data: {e}")
        return False


def test_signal_generation():
    """Test signal generation."""
    print("\nTesting signal generation...")
    
    try:
        from data.collectors import CryptoDataCollector
        from strategy.trading_engine import TradingLogic
        
        collector = CryptoDataCollector(lookback_days=30)
        data = collector.fetch_crypto_data('BTC-USD')
        
        if data is None or len(data) == 0:
            print("✗ No data available")
            return False
        
        trading_logic = TradingLogic()
        signal, confidence = trading_logic.generate_signal(data)
        
        print(f"✓ Signal generated: {signal.name} (Confidence: {confidence:.2f})")
        return True
    
    except Exception as e:
        print(f"✗ Error generating signal: {e}")
        return False


def test_config():
    """Test configuration loading."""
    print("\nTesting configuration...")
    
    try:
        import json
        
        with open('config.json', 'r') as f:
            config = json.load(f)
        
        required_keys = ['trading', 'data', 'sentiment', 'ml', 'logging', 'scheduler']
        missing_keys = [k for k in required_keys if k not in config]
        
        if missing_keys:
            print(f"✗ Missing config keys: {missing_keys}")
            return False
        
        print(f"✓ Configuration loaded successfully")
        print(f"  Initial Capital: ${config['trading']['initial_capital']}")
        print(f"  Target Daily Return: {config['trading']['target_daily_return']*100:.1f}%")
        return True
    
    except FileNotFoundError:
        print("✗ config.json not found")
        return False
    except Exception as e:
        print(f"✗ Error loading config: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 50)
    print("Crypto Trading Bot - Installation Test")
    print("=" * 50)
    
    results = {
        'Imports': test_imports(),
        'Config': test_config(),
        'Data Fetching': test_data_fetching(),
        'Signal Generation': test_signal_generation(),
    }
    
    print("\n" + "=" * 50)
    print("Test Results:")
    print("=" * 50)
    
    all_passed = True
    for test_name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ All tests passed! Bot is ready to use.")
        print("\nNext steps:")
        print("1. Configure API keys in config.json")
        print("2. Run: python main.py --mode run")
        print("3. Or launch GUI: streamlit run gui/app.py")
    else:
        print("\n✗ Some tests failed. Check the errors above.")
        print("Installation may be incomplete.")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
