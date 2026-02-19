# 🤖 Crypto Trading Bot - AI-Powered Automated Trading System

A sophisticated cryptocurrency trading bot with machine learning, sentiment analysis, and automated daily trading cycles. 

## Features

✅ **Automated Trading** - Runs daily trading cycles with technical analysis  
✅ **Machine Learning** - Learns from historical trades to improve predictions  
✅ **Sentiment Analysis** - Analyzes X.com (Twitter) and news sentiment  
✅ **Technical Indicators** - RSI, MACD, EMA, Bollinger Bands, ATR  
✅ **Top 100 Cryptos** - Trade any of the top 100 cryptocurrencies  
✅ **Backtesting** - Validate strategies on historical data  
✅ **Interactive GUI** - Streamlit dashboard with charts and trade logs  
✅ **Detailed Logging** - Comprehensive trade history and performance tracking  
✅ **Paper Trading** - Simulation mode for risk-free testing  

## System Architecture

```
crypto_trading_bot/
├── data/                          # Data collection & sentiment
│   ├── collectors.py             # yfinance data fetcher
│   └── sentiment.py              # X.com & news sentiment analysis
├── strategy/                      # Trading logic & optimization
│   ├── trading_engine.py         # Core trading logic
│   ├── ml_optimizer.py           # ML model training
│   └── logging.py                # Trade logging & tracking
├── gui/                           # Web interface
│   └── app.py                    # Streamlit dashboard
├── backtester/                    # Strategy testing
│   └── simulator.py              # Backtest engine
├── main.py                        # Bot orchestrator & scheduler
├── config.json                    # Configuration file
└── requirements.txt               # Python dependencies
```

## Quick Start

### 1. Installation

```bash
# Clone or navigate to project directory
cd crypto_trading_bot

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config.json` to customize settings:

```json
{
  "trading": {
    "initial_capital": 1000.0,
    "target_daily_return": 0.02,
    "max_loss_per_trade": 0.01
  },
  "sentiment": {
    "x_bearer_token": "YOUR_X_API_TOKEN",
    "x_api_key": "YOUR_API_KEY",
    "x_api_secret": "YOUR_API_SECRET"
  },
  "scheduler": {
    "run_time": "09:00",
    "timezone": "UTC"
  }
}
```

#### Getting X.com API Credentials:
1. Go to https://developer.twitter.com/
2. Create a developer account
3. Create a project and app
4. Generate API keys and bearer token
5. Add to config.json

### 3. Running the Bot

#### Option A: Run Single Cycle
```bash
python main.py --mode run
```

#### Option B: Schedule Daily Runs
```bash
python main.py --mode schedule --config config.json
```
Bot will run daily at the time specified in `config.json`

#### Option C: Run Backtest
```bash
python main.py --mode backtest --symbol BTC-USD --days 90
```

#### Option D: Launch GUI Dashboard
```bash
streamlit run gui/app.py
```

The dashboard opens at http://localhost:8501

## GUI Dashboard

### 📊 Chart & Analysis Tab
- Real-time price charts with candlesticks
- Technical indicators (RSI, MACD, Bollinger Bands)
- 90-day historical data

### 🤖 Trading Signals Tab
- Current trading signal (STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL)
- Signal confidence score
- Technical breakdown
- Recommended action

### 📈 Performance Tab
- Daily returns distribution
- Cumulative performance chart
- Portfolio statistics
- Win rate and Sharpe ratio

### 📋 Trade Logs Tab
- Complete trade history
- Filterable by symbol, date, event type
- Export to CSV/Excel
- Trade-by-trade analysis

### ⚡ Backtest Tab
- Run historical backtests
- Test different parameters
- View equity curve
- Performance metrics

## Trading Strategy

### Signal Generation
The bot generates trading signals using a weighted combination of:

1. **Technical Analysis (50%)**
   - EMA Crossovers
   - RSI (Relative Strength Index)
   - MACD
   - Bollinger Bands
   - Price momentum
   - Moving average trends

2. **Sentiment Analysis (30%)**
   - X.com/Twitter mentions
   - News sentiment from Yahoo Finance & Bloomberg
   - Engagement metrics (likes, retweets)

3. **Machine Learning (20%)**
   - Random Forest classifier
   - Learns from historical trade results
   - Predicts price direction
   - Improves with more trading cycles

### Entry Rules
- **BUY** if signal > 0.3 AND confidence > 55%
- **SELL** if signal < -0.3 AND confidence > 55%
- **Position Size** = Capital × Confidence × Volatility Factor

### Risk Management
- Max 10% of capital per position
- Stop-loss = Entry - (Max Loss % × Entry Price)
- Take-profit = Entry + (Target Return × Entry Price)
- Dynamic position sizing based on volatility

## Performance Metrics

The bot tracks:
- **Win Rate** - % of profitable trades
- **Profit Factor** - Gross Profit / Gross Loss
- **Sharpe Ratio** - Risk-adjusted returns
- **Max Drawdown** - Largest portfolio decline
- **Daily Returns** - Daily P&L %
- **Trade Duration** - Avg holding period

## Logging

All activity is logged to:
- `logs/bot_YYYYMMDD.log` - Detailed activity log
- `logs/trades_YYYYMMDD.json` - All trades in JSON format
- `logs/performance_YYYYMMDD.csv` - Daily performance metrics

Example trade log:
```json
{
  "timestamp": "2024-02-18T09:00:15",
  "event": "TRADE_EXECUTED",
  "type": "BUY",
  "symbol": "BTC-USD",
  "quantity": 0.0125,
  "price": 80000.50,
  "reason": "STRONG_BUY",
  "confidence": 0.82
}
```

## Machine Learning

The bot improves over time by:

1. **Collecting Trade Data** - Logs all trades with outcomes
2. **Feature Engineering** - Extracts technical indicators
3. **Model Training** - Random Forest / Gradient Boosting
4. **Performance Analysis** - Identifies patterns in winning trades
5. **Strategy Refinement** - Adjusts thresholds based on performance

Model retrains every 50 trades with:
- 80% training data, 20% test data
- Cross-validation
- Feature importance scoring

## Supported Cryptocurrencies

The bot can trade any of the top 100 cryptocurrencies by market cap:

**Top tier**: BTC, ETH, BNB, XRP, SOL, ADA, DOGE, AVAX, LINK, MATIC

**Popular**: LTC, ATOM, FIL, UNI, NEAR, ARB, JUP, RENDER, TAO, WIF

**Full list**: See `get_top_100_cryptos()` in `data/collectors.py`

### Switch Cryptocurrencies

In the GUI:
1. Use search box: Type "ETH" → Select from results
2. Or use dropdown: Select from top 100 list
3. Settings apply immediately

Command line:
```bash
python main.py --backtest --symbol ETH-USD --days 90
```

## Advanced Configuration

### Tuning Trading Parameters

Edit `config.json`:

```json
{
  "trading": {
    "target_daily_return": 0.02,      // 2% daily - ADJUST THIS
    "max_loss_per_trade": 0.01,       // 1% max loss
    "position_size": 0.1              // 10% of capital per trade
  },
  "ml": {
    "min_historical_trades": 20,      // Min trades to train
    "retrain_interval": 50            // Retrain every 50 trades
  }
}
```

### Sentiment Weighting

In `strategy/trading_engine.py`:
```python
weights = {
    'technical': 0.5,    # 50% technical
    'sentiment': 0.3,    # 30% sentiment
    'ml': 0.2            # 20% ML
}
```

## Backtesting Examples

### Test BTC with 3 months data:
```bash
python main.py --mode backtest --symbol BTC-USD --days 90
```

### Test ETH with 1 year data:
```bash
python main.py --mode backtest --symbol ETH-USD --days 365
```

### GUI Backtest:
1. Open dashboard: `streamlit run gui/app.py`
2. Go to "⚡ Backtest" tab
3. Select symbol & period
4. Click "🚀 Run Backtest"
5. View equity curve and results

## Troubleshooting

### No data fetched
- Check internet connection
- Verify yfinance is working: `python -c "import yfinance; yf.download('BTC-USD', period='1d')"`

### X.com API errors
- Verify bearer token is valid
- Check API rate limits (v2 API: 300 requests/15 min)
- Use free fallback without sentiment if API not available

### Poor trading performance
- Backtest strategy first with historical data
- Adjust `target_daily_return` to realistic levels (2-3% is good)
- Check `min_confidence` threshold
- Verify technical indicators are calculating correctly

### Out of memory
- Reduce `lookback_days` from 90 to 30
- Trade fewer symbols at once
- Run backtest on smaller timeframes

## Performance Tips

1. **Optimize Indicators** - Use only needed indicators
2. **Reduce Update Frequency** - Change `update_interval_minutes` to 120+
3. **Limit Symbols** - Trade top 10-20 instead of 100
4. **Backtesting First** - Always test before running live
5. **Monitor Logs** - Check logs for errors/warnings

## Safety Warnings

⚠️ **This is a simulator** - No real money is used
⚠️ **Test thoroughly** - Backtest extensively before adjusting settings
⚠️ **Start small** - Use $1000-$5000 virtual capital
⚠️ **Monitor closely** - Check daily logs and performance
⚠️ **Adjust gradually** - Don't change multiple settings at once

## Future Enhancements

- [ ] Live trading integration (Coinbase, Kraken API)
- [ ] Advanced portfolio optimization
- [ ] Deep learning models (LSTM, Transformers)
- [ ] Multi-timeframe analysis
- [ ] Risk parity calculations
- [ ] PDF/Excel report generation
- [ ] Email/SMS alerts
- [ ] REST API for remote control

## Support

For issues or questions:
1. Check the logs: `logs/bot_YYYYMMDD.log`
2. Review backtest results in GUI
3. Verify config.json syntax
4. Check API credentials

## License

Open source - modify and use freely

## Disclaimer

This bot is for educational and research purposes. Cryptocurrency trading carries substantial risk of loss. Past performance does not guarantee future results. Always trade responsibly and never risk capital you can't afford to lose.

---

**Happy Trading! 📈**
