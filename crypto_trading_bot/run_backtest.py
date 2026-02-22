#!/usr/bin/env python3
"""Quick script to run multi-crypto backtest."""

from manager.ralph_manager import RalphManager
import logging

logging.basicConfig(level=logging.INFO)

manager = RalphManager()
print('Starting multi-crypto backtest on top 10 cryptos...')
results = manager.run_multi_crypto_backtest(days=90)
print(f'\nBacktest complete! Generated {len(results)} results.')

if results:
    print('\nResults summary:')
    for symbol in list(results.keys())[:5]:
        r = results[symbol]
        print(f'  {symbol}: {r.get("total_return_pct", 0):+.2f}% | Trades: {r.get("num_trades", 0)} | Sharpe: {r.get("sharpe_ratio", 0):.2f}')
else:
    print('No results generated - check logs for errors.')
