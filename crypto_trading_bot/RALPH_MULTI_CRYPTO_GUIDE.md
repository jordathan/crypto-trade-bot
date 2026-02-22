# Ralph Multi-Crypto Backtest Guide

## Overview
Ralph can now run backtests on multiple cryptocurrencies (default: top 10) and save detailed results for analysis in the Streamlit dashboard.

## How to Use

### Method 1: Ralph Terminal UI (TUI)
Run Ralph's interactive terminal:
```bash
python main.py --mode ralph
```

Select option **4) Multi-crypto backtest (Top 10)** from the menu.

Ralph will:
- Test the top 10 cryptos: BTC, ETH, BNB, SOL, ADA, XRP, DOGE, DOT, MATIC, AVAX
- Use your current strategy settings from config.json
- Show a summary table in the terminal
- Save detailed results to `logs/ralph_multi_backtest_YYYYMMDD_HHMMSS.json`

### Method 2: Telegram Bot
If you have the Telegram bot set up:
```
/multitest YOUR_SECRET 90
```

Parameters:
- `YOUR_SECRET`: Your shared secret from .env
- `90`: Number of days to backtest (optional, default: 90)

Ralph will run the backtest and send you a summary with:
- Return % for each crypto
- Win rate
- Number of trades
- Overall averages

### Method 3: Direct Python
```python
from manager.ralph_manager import RalphManager

manager = RalphManager()

# Run on top 10 (default)
results = manager.run_multi_crypto_backtest(days=90)

# Or specify custom symbols
custom_symbols = ["BTC-USD", "ETH-USD", "LINK-USD", "UNI-USD"]
results = manager.run_multi_crypto_backtest(symbols=custom_symbols, days=60)
```

## Viewing Results

### In Streamlit Dashboard
1. Start the dashboard:
   ```bash
   streamlit run gui/app.py
   ```

2. Click the **"🎯 Ralph Multi-Crypto"** tab

3. The dashboard shows:
   - **Summary metrics**: Average return, drawdown, win rate, total trades
   - **Detailed results table**: All cryptos with color-coded returns
   - **Visualizations**:
     - Returns bar chart (sorted by performance)
     - Risk vs Return scatter plot
     - Sharpe ratio comparison
   - **Individual crypto drill-down**: 
     - Select any crypto to see its equity curve
     - View all trades for that crypto
   - **Download option**: Export full results as JSON

### Results File Format
The saved JSON file contains:
```json
{
  "timestamp": "2026-02-21T10:30:00",
  "config": {
    "days": 90,
    "initial_capital": 1000.0,
    "strategy": { ... }
  },
  "results": {
    "BTC-USD": {
      "symbol": "BTC-USD",
      "final_value": 1025.50,
      "total_return_pct": 2.55,
      "max_drawdown_pct": 5.2,
      "win_rate": 0.625,
      "num_trades": 12,
      "sharpe_ratio": 1.85,
      "trades": [ ... ],
      "equity_curve": [ ... ]
    },
    "ETH-USD": { ... }
  }
}
```

## Understanding the Results

### Key Metrics
- **Return (%)**: Total profit/loss percentage
- **Drawdown (%)**: Maximum peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Trades**: Number of completed trades

### What to Look For
✅ **Good signs:**
- Positive returns across multiple cryptos
- Win rate > 50%
- Sharpe ratio > 1.0
- Consistent performance (not just one lucky trade)

⚠️ **Warning signs:**
- High drawdown (>20%)
- Very low trade count (insufficient data)
- Negative returns on most cryptos
- Win rate < 40%

## Use Cases

### 1. Strategy Validation
Test if your current strategy works across different cryptocurrencies:
```bash
# Run Ralph TUI and select option 4
python main.py --mode ralph
```

### 2. Find Best Performers
Identify which cryptos work best with your strategy, then focus on those:
- Look for highest Sharpe ratios
- Check consistency in returns
- Verify sufficient trade volume

### 3. Before Live Trading
Run multi-crypto backtest before deploying to ensure strategy is robust:
```bash
# Via Telegram
/multitest YOUR_SECRET 180
```
(Test with 180 days for more confidence)

### 4. Compare Strategy Changes
1. Run multi-crypto backtest with current settings
2. Modify strategy parameters in config.json
3. Run again
4. Compare results in Streamlit to see improvements

## Tips

🔹 **Customize Symbols**: Edit the default list in `ralph_manager.py` line ~374 or pass custom symbols
🔹 **Longer Backtests**: Use 180+ days for more reliable results
🔹 **Regular Testing**: Run weekly to ensure strategy remains effective
🔹 **Parallel Analysis**: Use multiple strategy configs and compare results
🔹 **Export Data**: Download JSON results for external analysis (Excel, Python notebooks)

## Example Workflow

1. **Test current strategy across top 10 cryptos:**
   ```bash
   python main.py --mode ralph
   # Select option 4
   ```

2. **View results in dashboard:**
   ```bash
   streamlit run gui/app.py
   # Go to "🎯 Ralph Multi-Crypto" tab
   ```

3. **Identify best performers:**
   - Sort by Sharpe ratio
   - Look for cryptos with good return AND low drawdown

4. **Focus optimization on top performers:**
   - Update `config.json` ralph.symbols to only include winners
   - Run parameter sweep on that subset

5. **Monitor remotely via Telegram:**
   ```
   /multitest YOUR_SECRET 90
   ```

---

**Ready to find your best trading opportunities!** 🎯📈
