# Quick Start Guide

## 5-Minute Setup

### Step 1: Install Python Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Configure API Keys (Optional)
Edit `config.json` and add your X.com credentials:
```json
"x_bearer_token": "YOUR_TOKEN_HERE",
"x_api_key": "YOUR_KEY_HERE",
"x_api_secret": "YOUR_SECRET_HERE"
```

### Step 3: Choose Your Mode

#### **Option A: Web Dashboard (Recommended for Beginners)**
```bash
streamlit run gui/app.py
```
Opens http://localhost:8501 with interactive charts and controls

#### **Option B: Single Trading Cycle**
```bash
python main.py --mode run
```
Runs one complete trading analysis

#### **Option C: Scheduled Daily Runs**
```bash
python main.py --mode schedule
```
Runs at 09:00 UTC daily (configurable in config.json)

#### **Option D: Backtest Strategy**
```bash
python main.py --mode backtest --symbol BTC-USD --days 90
```
Tests strategy on historical data

#### **Option E: Test Installation**
```bash
python test_installation.py
```
Verifies everything is working correctly

## Common Operations

### Change Trading Cryptocurrency
In GUI: Use search box → Select from dropdown
In CLI: `python main.py --backtest --symbol ETH-USD --days 90`

### Adjust Trading Aggression
Edit `config.json`:
```json
"target_daily_return": 0.02    // Change 0.02 (2%) to desired %
"max_loss_per_trade": 0.01     // Risk tolerance
```

### View Trade History
1. Open GUI dashboard
2. Click "📋 Trade Logs" tab
3. Filter and download as CSV

### Check Performance Metrics
1. Open GUI dashboard
2. Click "📈 Performance" tab
3. View charts and statistics

## Troubleshooting

**Problem: "No module named 'streamlit'"**
→ Run: `pip install -r requirements.txt`

**Problem: "Could not fetch data"**
→ Check internet connection, try another symbol

**Problem: No signals generated**
→ May need 50+ days of training data, try increasing `lookback_days`

## Next Steps

1. ✓ Run test_installation.py to verify setup
2. ✓ Launch GUI to explore features
3. ✓ Run backtest on your chosen crypto
4. ✓ Adjust settings based on backtest results
5. ✓ Schedule daily runs when confident

## Documentation

- Full guide: `README.md`
- Config options: `config.json`
- API setup: See README.md section "Getting X.com API Credentials"

## Support Files

- `logs/` - Daily trade logs and performance
- `models/` - Trained ML models
- `config.json` - Configuration
- `.env.example` - Environment variables template

Ready to trade? Launch the GUI! 🚀
