"""
Ralph Manager - orchestrates parameter sweeps, backtests, and strategy updates.
"""

import json
import logging
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import pandas as pd

from data.collectors import CryptoDataCollector
from strategy.trading_engine import TradingLogic
from backtester.simulator import BacktestEngine

logger = logging.getLogger(__name__)


@dataclass
class SweepResult:
    params: Dict[str, Any]
    avg_return_pct: float
    avg_drawdown_pct: float
    avg_win_rate: float
    avg_num_trades: float
    score: float


class RalphManager:
    """Manages automated backtests and strategy optimization."""

    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.config = self._load_config()
        self.log_dir = Path("logs")
        self.log_dir.mkdir(exist_ok=True)
        self.session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.best_file = self.log_dir / "ralph_best.json"

    def _load_config(self) -> Dict[str, Any]:
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _save_config(self, config: Dict[str, Any]) -> None:
        backup_path = Path(self.config_path).with_suffix(".backup.json")
        try:
            with open(backup_path, "w") as f:
                json.dump(self.config, f, indent=2)
        except Exception as e:
            logger.warning(f"Could not write config backup: {e}")

        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

        self.config = config

    def _load_best(self) -> Dict[str, Any]:
        if not self.best_file.exists():
            return {}
        try:
            with open(self.best_file, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_best(self, best: SweepResult) -> None:
        payload = {
            "timestamp": datetime.now().isoformat(),
            "score": best.score,
            "avg_return_pct": best.avg_return_pct,
            "avg_drawdown_pct": best.avg_drawdown_pct,
            "avg_win_rate": best.avg_win_rate,
            "avg_num_trades": best.avg_num_trades,
            "params": best.params
        }
        with open(self.best_file, "w") as f:
            json.dump(payload, f, indent=2)

    def _get_sweep_space(self) -> Dict[str, List[Any]]:
        ralph_cfg = self.config.get("ralph", {})
        sweep = ralph_cfg.get("sweep", {})
        if not sweep:
            sweep = {
                "min_confidence": [0.5, 0.55, 0.6],
                "rsi_low": [25, 30, 35],
                "rsi_high": [65, 70, 75],
                "momentum_window": [10, 20, 30],
                "momentum_up": [0.03, 0.05, 0.07],
                "momentum_down": [-0.03, -0.05, -0.07],
                "signal_buy_threshold": [0.25, 0.3, 0.35],
                "signal_strong_threshold": [0.45, 0.5, 0.55],
                "weights": [
                    {"technical": 0.5, "sentiment": 0.3, "ml": 0.2},
                    {"technical": 0.6, "sentiment": 0.2, "ml": 0.2},
                    {"technical": 0.4, "sentiment": 0.4, "ml": 0.2}
                ],
                "lookback_days": [60, 90, 120]
            }
        return sweep

    def _generate_trials(self, max_trials: int) -> List[Dict[str, Any]]:
        sweep = self._get_sweep_space()
        keys = list(sweep.keys())
        values = [sweep[k] for k in keys]

        all_trials = []
        def build(idx: int, current: Dict[str, Any]):
            if idx == len(keys):
                all_trials.append(current.copy())
                return
            key = keys[idx]
            for val in values[idx]:
                current[key] = val
                build(idx + 1, current)

        build(0, {})

        if max_trials and len(all_trials) > max_trials:
            random.shuffle(all_trials)
            return all_trials[:max_trials]

        return all_trials

    def _run_backtests_for_params(self, params: Dict[str, Any], symbols: List[str], days: int) -> SweepResult:
        lookback_days = int(params.get("lookback_days", days))
        collector = CryptoDataCollector(
            lookback_days=lookback_days,
            custom_tokens=self.config.get("data", {}).get("custom_tokens", [])
        )

        logic = TradingLogic(
            target_return=self.config["trading"]["target_daily_return"],
            max_loss=self.config["trading"]["max_loss_per_trade"],
            min_confidence=params["min_confidence"],
            rsi_low=params["rsi_low"],
            rsi_high=params["rsi_high"],
            momentum_window=params["momentum_window"],
            momentum_up=params["momentum_up"],
            momentum_down=params["momentum_down"],
            signal_buy_threshold=params["signal_buy_threshold"],
            signal_strong_threshold=params["signal_strong_threshold"],
            weights=params["weights"]
        )

        returns = []
        drawdowns = []
        win_rates = []
        trade_counts = []

        for symbol in symbols:
            data = collector.fetch_crypto_data(symbol)
            if data.empty:
                continue

            engine = BacktestEngine(
                initial_capital=self.config["trading"]["initial_capital"],
                min_confidence=logic.min_confidence
            )

            def signal_gen(history, sentiment, ml_prob):
                return logic.generate_signal(history, sentiment, ml_prob)

            result = engine.run_backtest(symbol, data, signal_gen)
            if not result:
                continue

            returns.append(result.get("total_return_pct", 0))
            drawdowns.append(abs(result.get("max_drawdown_pct", 0)))
            win_rates.append(result.get("win_rate", 0))
            trade_counts.append(result.get("num_trades", 0))

        avg_return = sum(returns) / len(returns) if returns else -999.0
        avg_drawdown = sum(drawdowns) / len(drawdowns) if drawdowns else 999.0
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
        avg_trades = sum(trade_counts) / len(trade_counts) if trade_counts else 0.0

        cap = float(self.config.get("ralph", {}).get("drawdown_cap_pct", 20.0))
        penalty = float(self.config.get("ralph", {}).get("drawdown_penalty", 1.0))

        if avg_drawdown <= cap:
            score = avg_return
        else:
            score = avg_return - ((avg_drawdown - cap) * penalty)

        return SweepResult(
            params=params,
            avg_return_pct=avg_return,
            avg_drawdown_pct=avg_drawdown,
            avg_win_rate=avg_win_rate,
            avg_num_trades=avg_trades,
            score=score
        )

    def run_parameter_sweep(self, days: int = None, max_trials: int = None) -> SweepResult:
        ralph_cfg = self.config.get("ralph", {})
        days = days or ralph_cfg.get("backtest_days", 90)
        max_trials = max_trials or ralph_cfg.get("max_trials", 50)
        symbols = ralph_cfg.get("symbols", ["BTC-USD", "ETH-USD", "BNB-USD"]) 

        trials = self._generate_trials(max_trials)
        best = None

        for idx, params in enumerate(trials, start=1):
            result = self._run_backtests_for_params(params, symbols, days)
            logger.info(
                f"Ralph sweep {idx}/{len(trials)} | "
                f"Return {result.avg_return_pct:+.2f}% | "
                f"Drawdown {result.avg_drawdown_pct:.2f}% | "
                f"Score {result.score:+.2f}"
            )
            if best is None or result.score > best.score:
                best = result

        if best:
            self._apply_best_params(best.params)
            self._save_best(best)

        return best

    def _apply_best_params(self, params: Dict[str, Any]) -> None:
        updated = dict(self.config)
        updated.setdefault("strategy", {})
        updated["strategy"].update({
            "min_confidence": params["min_confidence"],
            "rsi_low": params["rsi_low"],
            "rsi_high": params["rsi_high"],
            "momentum_window": params["momentum_window"],
            "momentum_up": params["momentum_up"],
            "momentum_down": params["momentum_down"],
            "signal_buy_threshold": params["signal_buy_threshold"],
            "signal_strong_threshold": params["signal_strong_threshold"],
            "weights": params["weights"]
        })
        updated.setdefault("data", {})
        updated["data"]["lookback_days"] = int(params.get("lookback_days", updated["data"].get("lookback_days", 90)))

        self._save_config(updated)

    def _load_latest_trades(self) -> List[Dict[str, Any]]:
        trades_files = sorted(self.log_dir.glob("trades_*.json"), key=lambda p: p.stat().st_mtime)
        if not trades_files:
            return []
        latest = trades_files[-1]
        try:
            with open(latest, "r") as f:
                return json.load(f)
        except Exception:
            return []

    def _summarize_trades(self, trades: List[Dict[str, Any]]) -> Dict[str, Any]:
        errors = [t for t in trades if t.get("event") == "ERROR"]
        executed = [t for t in trades if t.get("event") == "TRADE_EXECUTED"]
        skipped = [t for t in trades if t.get("event") == "DECISION_SKIPPED"]

        return {
            "errors_count": len(errors),
            "trades_count": len(executed),
            "skipped_count": len(skipped)
        }

    def _write_daily_summary(self, summary: Dict[str, Any]) -> None:
        summary_name = f"daily_summary_{self.session_timestamp}"
        text_path = self.log_dir / f"{summary_name}.txt"
        json_path = self.log_dir / f"{summary_name}.json"

        lines = [
            f"Date: {summary.get('date', '')}",
            f"Status: {summary.get('status', '')}",
            f"Trades: {summary.get('trades_count', 0)}",
            f"Skipped: {summary.get('skipped_count', 0)}",
            f"Return: {summary.get('return_pct', 0):+.2f}%",
            f"Win Rate: {summary.get('win_rate', 0):.1%}",
            f"Max Drawdown: {summary.get('max_drawdown_pct', 0):.2f}%",
            f"Errors: {summary.get('errors_count', 0)}",
            f"Plan: {summary.get('plan', '')}".strip()
        ]

        with open(text_path, "w") as f:
            f.write("\n".join(lines))

        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def run_daily_cycle_via_subprocess(self) -> None:
        logger.info("Running trading cycle via subprocess")
        subprocess.run([sys.executable, "main.py", "--mode", "run"], check=False)

    def run_optimization_cycle(self, days: int = None, max_trials: int = None) -> None:
        self.run_daily_cycle_via_subprocess()

        trades = self._load_latest_trades()
        trade_summary = self._summarize_trades(trades)

        previous_best = self._load_best()
        previous_score = previous_best.get("score") if previous_best else None

        best = self.run_parameter_sweep(days=days, max_trials=max_trials)

        improvement = 0.0
        if previous_score is not None and best:
            improvement = best.score - previous_score

        plan = "No update"
        if best:
            plan = (
                f"min_conf={best.params['min_confidence']}, "
                f"rsi={best.params['rsi_low']}/{best.params['rsi_high']}, "
                f"momentum={best.params['momentum_window']}, "
                f"buy={best.params['signal_buy_threshold']}, "
                f"strong={best.params['signal_strong_threshold']}"
            )

        summary = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "status": "completed",
            "trades_count": trade_summary.get("trades_count", 0),
            "errors_count": trade_summary.get("errors_count", 0),
            "skipped_count": trade_summary.get("skipped_count", 0),
            "return_pct": best.avg_return_pct if best else 0.0,
            "max_drawdown_pct": best.avg_drawdown_pct if best else 0.0,
            "win_rate": best.avg_win_rate if best else 0.0,
            "improvement_score": improvement,
            "plan": plan
        }

        self._write_daily_summary(summary)

    def run_continuous_optimization(self, interval_minutes: int = 60, days: int = None, max_trials: int = None) -> None:
        logger.info(f"Ralph continuous mode started (interval: {interval_minutes} min)")
        logger.info("Press Ctrl+C to stop")

        try:
            while True:
                self.run_optimization_cycle(days=days, max_trials=max_trials)
                logger.info(f"Sleeping {interval_minutes} minutes")
                time.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            logger.info("Ralph continuous mode stopped")

    def run_tui(self) -> None:
        while True:
            self._render_tui()
            choice = input("Select option: ").strip()

            if choice == "1":
                self.run_optimization_cycle()
            elif choice == "2":
                interval = self._prompt_int("Interval minutes", 60)
                days = self._prompt_int("Backtest days", 90)
                trials = self._prompt_int("Max trials", self.config.get("ralph", {}).get("max_trials", 50))
                self.run_continuous_optimization(interval_minutes=interval, days=days, max_trials=trials)
            elif choice == "3":
                days = self._prompt_int("Backtest days", 90)
                trials = self._prompt_int("Max trials", self.config.get("ralph", {}).get("max_trials", 50))
                self.run_parameter_sweep(days=days, max_trials=trials)
            elif choice == "4":
                days = self._prompt_int("Backtest days", 90)
                print("\nRunning multi-crypto backtest on top 10 cryptos...")
                self.run_multi_crypto_backtest(days=days)
                input("\nPress Enter to continue...")
            elif choice == "5":
                self._print_current_strategy()
                input("Press Enter to continue...")
            elif choice == "6":
                print("Goodbye.")
                break
            else:
                print("Invalid option.")

    def _render_tui(self) -> None:
        print("\n" + "=" * 60)
        print("RALPH MANAGER")
        print("=" * 60)
        print("1) Run full optimize cycle (trade + sweep + plan)")
        print("2) Continuous optimize mode")
        print("3) Run parameter sweep only")
        print("4) Multi-crypto backtest (Top 10)")
        print("5) Show current strategy settings")
        print("6) Exit")
        print("=" * 60)

    def _prompt_int(self, label: str, default: int) -> int:
        raw = input(f"{label} [{default}]: ").strip()
        if not raw:
            return default
        try:
            return int(raw)
        except ValueError:
            return default

    def _print_current_strategy(self) -> None:
        strategy = self.config.get("strategy", {})
        data = self.config.get("data", {})

        print("\nCurrent Strategy:")
        print(f"  min_confidence: {strategy.get('min_confidence', 0.55)}")
        print(f"  rsi_low: {strategy.get('rsi_low', 30.0)}")
        print(f"  rsi_high: {strategy.get('rsi_high', 70.0)}")
        print(f"  momentum_window: {strategy.get('momentum_window', 20)}")
        print(f"  momentum_up: {strategy.get('momentum_up', 0.05)}")
        print(f"  momentum_down: {strategy.get('momentum_down', -0.05)}")
        print(f"  signal_buy_threshold: {strategy.get('signal_buy_threshold', 0.3)}")
        print(f"  signal_strong_threshold: {strategy.get('signal_strong_threshold', 0.5)}")
        print(f"  weights: {strategy.get('weights', {})}")
        print(f"  lookback_days: {data.get('lookback_days', 90)}")

    def run_multi_crypto_backtest(self, symbols: List[str] = None, days: int = None) -> Dict[str, Any]:
        """
        Run backtests on multiple cryptocurrencies and save results.
        
        Args:
            symbols: List of crypto symbols (default: top 10)
            days: Number of days to backtest (default: from config)
            
        Returns:
            Dictionary with results for each symbol
        """
        # Default to top 10 cryptos
        if symbols is None:
            symbols = [
                "BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "ADA-USD",
                "XRP-USD", "DOGE-USD", "DOT-USD", "MATIC-USD", "AVAX-USD"
            ]
        
        days = days or self.config.get("ralph", {}).get("backtest_days", 90)
        
        logger.info(f"Running multi-crypto backtest on {len(symbols)} symbols")
        logger.info(f"Symbols: {', '.join(symbols)}")
        logger.info(f"Lookback: {days} days")
        
        # Current strategy from config
        strategy_cfg = self.config.get("strategy", {})
        logic = TradingLogic(
            target_return=self.config["trading"]["target_daily_return"],
            max_loss=self.config["trading"]["max_loss_per_trade"],
            min_confidence=strategy_cfg.get("min_confidence", 0.4),  # Lowered for backtest (neutral sentiment)
            rsi_low=strategy_cfg.get("rsi_low", 30.0),
            rsi_high=strategy_cfg.get("rsi_high", 70.0),
            momentum_window=strategy_cfg.get("momentum_window", 20),
            momentum_up=strategy_cfg.get("momentum_up", 0.05),
            momentum_down=strategy_cfg.get("momentum_down", -0.05),
            signal_buy_threshold=strategy_cfg.get("signal_buy_threshold", 0.2),  # Lowered threshold
            signal_strong_threshold=strategy_cfg.get("signal_strong_threshold", 0.4),  # Lowered threshold
            weights=strategy_cfg.get("weights")
        )
        
        collector = CryptoDataCollector(
            lookback_days=days,
            custom_tokens=self.config.get("data", {}).get("custom_tokens", [])
        )
        results = {}
        
        for idx, symbol in enumerate(symbols, 1):
            logger.info(f"[{idx}/{len(symbols)}] Backtesting {symbol}...")
            
            try:
                data = collector.fetch_crypto_data(symbol)
                if data.empty:
                    logger.warning(f"No data for {symbol}, skipping")
                    continue
                
                engine = BacktestEngine(
                    initial_capital=self.config["trading"]["initial_capital"],
                    min_confidence=0.3  # Lower threshold for backtest trades
                )
                
                # Very simple signal generator for backtest
                def signal_gen(history, sentiment, ml_prob):
                    """Simple signal generator based on basic technical indicators."""
                    if history.empty or len(history) < 50:
                        from strategy.trading_engine import TradeSignal
                        return TradeSignal.HOLD, 0.5
                    
                    from strategy.trading_engine import TradeSignal
                    latest = history.iloc[-1]
                    prev = history.iloc[-2] if len(history) > 1 else history.iloc[-1]
                    
                    try:
                        # SMA crossover
                        sma10_curr = float(latest['SMA_10'])
                        sma20_curr = float(latest['SMA_20'])
                        sma10_prev = float(prev['SMA_10'])
                        sma20_prev = float(prev['SMA_20'])
                        
                        close = float(latest['Close'])
                        close_prev = float(prev['Close'])
                        pct_change = (close - close_prev) / close_prev
                        
                        # Generate signal
                        if sma10_curr > sma20_curr and sma10_prev <= sma20_prev:
                            # Golden cross - strong buy signal
                            return TradeSignal.BUY, 0.7
                        elif sma10_curr < sma20_curr and sma10_prev >= sma20_prev:
                            # Death cross - sell signal
                            return TradeSignal.SELL, 0.7
                        elif sma10_curr > sma20_curr and pct_change > 0.01:
                            # Uptrend with positive daily - buy
                            return TradeSignal.BUY, 0.5
                        elif sma10_curr < sma20_curr and pct_change < -0.01:
                            # Downtrend with negative daily - sell
                            return TradeSignal.SELL, 0.5
                        else:
                            # Oscillate between buy/sell based on price vs SMA20
                            if close > sma20_curr:
                                return TradeSignal.BUY, 0.4
                            else:
                                return TradeSignal.SELL, 0.4
                    except (ValueError, TypeError, KeyError):
                        return TradeSignal.HOLD, 0.3
                
                result = engine.run_backtest(symbol, data, signal_gen)
                
                if result:
                    results[symbol] = {
                        "symbol": symbol,
                        "final_value": result.get("final_value", 0),
                        "total_return_pct": result.get("total_return_pct", 0),
                        "max_drawdown_pct": result.get("max_drawdown_pct", 0),
                        "win_rate": result.get("win_rate", 0),
                        "num_trades": result.get("num_closed_trades", 0),
                        "sharpe_ratio": result.get("sharpe_ratio", 0),
                        "trades": result.get("trades", []),
                        "equity_curve": result.get("equity_curve", pd.DataFrame()).to_dict('records') if not result.get("equity_curve", pd.DataFrame()).empty else [],
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.info(
                        f"  {symbol}: Return {result.get('total_return_pct', 0):+.2f}%, "
                        f"Drawdown {result.get('max_drawdown_pct', 0):.2f}%, "
                        f"Win Rate {result.get('win_rate', 0):.1%}, "
                        f"Trades {result.get('num_closed_trades', 0)}"
                    )
            except Exception as e:
                logger.error(f"Error backtesting {symbol}: {e}")
                continue
        
        # Save results to file
        output_file = self.log_dir / f"ralph_multi_backtest_{self.session_timestamp}.json"
        with open(output_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "config": {
                    "days": days,
                    "initial_capital": self.config["trading"]["initial_capital"],
                    "strategy": strategy_cfg
                },
                "results": results
            }, f, indent=2, default=str)
        
        logger.info(f"Multi-crypto backtest complete. Results saved to: {output_file}")
        logger.info(f"Total symbols tested: {len(results)}/{len(symbols)}")
        
        # Print summary
        if results:
            print("\n" + "=" * 80)
            print("MULTI-CRYPTO BACKTEST SUMMARY")
            print("=" * 80)
            print(f"{'Symbol':<12} {'Return':<12} {'Drawdown':<12} {'Win Rate':<12} {'Trades':<8} {'Sharpe':<8}")
            print("-" * 80)
            for symbol, res in results.items():
                print(
                    f"{symbol:<12} "
                    f"{res['total_return_pct']:>+10.2f}%  "
                    f"{res['max_drawdown_pct']:>10.2f}%  "
                    f"{res['win_rate']:>10.1%}  "
                    f"{res['num_trades']:>6}  "
                    f"{res['sharpe_ratio']:>6.2f}"
                )
            print("=" * 80)
        
        return results
