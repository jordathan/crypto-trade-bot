"""
Main trading bot orchestrator with daily scheduling.
Runs the complete trading workflow every day.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List
import sys
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from data.collectors import CryptoDataCollector, MarketDataProcessor
from data.sentiment import SentimentAggregator
from strategy.trading_engine import TradingLogic, SimulatedPortfolio
from strategy.ml_optimizer import MLModelTrainer, StrategyOptimizer
from strategy.logging import TradeLogger, PerformanceTracker
from backtester.simulator import BacktestEngine


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TradingBot:
    """Main trading bot class."""
    
    def __init__(self, config_path: str = 'config.json'):
        """
        Initialize trading bot.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        
        # Initialize components
        self.collector = CryptoDataCollector(
            lookback_days=self.config['data']['lookback_days']
        )
        self.processor = MarketDataProcessor()
        self.trading_logic = TradingLogic(
            target_return=self.config['trading']['target_daily_return'],
            max_loss=self.config['trading']['max_loss_per_trade']
        )
        self.portfolio = SimulatedPortfolio(
            initial_capital=self.config['trading']['initial_capital']
        )
        
        # Initialize sentiment analyzer
        self.sentiment_agg = SentimentAggregator(
            x_api_key=self.config['sentiment']['x_api_key'],
            x_api_secret=self.config['sentiment']['x_api_secret'],
            x_bearer_token=self.config['sentiment']['x_bearer_token']
        )
        
        # Initialize ML components
        self.ml_trainer = MLModelTrainer(
            model_type=self.config['ml']['model_type']
        )
        self.strategy_optimizer = StrategyOptimizer()
        
        # Initialize logging
        self.trade_logger = TradeLogger()
        self.perf_tracker = PerformanceTracker()
        
        # Trading history
        self.daily_trades: List[Dict] = []
        self.top_cryptos = self.collector.get_top_100_cryptos()
        
        logger.info("Trading Bot initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from file."""
        try:
            with open(self.config_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
    
    def run_daily_cycle(self):
        """Run one complete daily trading cycle."""
        logger.info("=" * 60)
        logger.info("Starting daily trading cycle")
        logger.info("=" * 60)
        
        cycle_start = datetime.now()
        
        try:
            # 1. Fetch market data
            logger.info("Fetching market data...")
            market_data = self.collector.fetch_multiple_cryptos(
                self.top_cryptos[:20]  # Trade top 20 for now
            )
            
            if not market_data:
                logger.error("Failed to fetch market data")
                return
            
            logger.info(f"Fetched data for {len(market_data)} cryptocurrencies")
            
            # 2. Get sentiment data
            logger.info("Analyzing sentiment...")
            sentiment_data = {}
            if self.config['sentiment']['enabled']:
                for symbol in list(market_data.keys())[:5]:  # Sentiment for top 5
                    try:
                        sentiment = self.sentiment_agg.get_combined_sentiment(symbol)
                        sentiment_data[symbol] = sentiment
                        logger.info(
                            f"{symbol} sentiment: {sentiment['combined_sentiment']:.2f} "
                            f"({sentiment['x_mentions']} mentions)"
                        )
                    except Exception as e:
                        logger.warning(f"Error getting sentiment for {symbol}: {e}")
            
            # 3. Trade each symbol
            logger.info("Generating trading signals...")
            
            for symbol, data in market_data.items():
                if data.empty:
                    continue
                
                try:
                    self._trade_symbol(symbol, data, sentiment_data)
                except Exception as e:
                    logger.error(f"Error trading {symbol}: {e}")
                    self.trade_logger.log_error(symbol, str(e))
            
            # 4. Train ML model
            logger.info("Retraining ML model...")
            self._retrain_ml_model()
            
            # 5. Record performance
            logger.info("Recording performance metrics...")
            self._record_performance()
            
            # 6. Generate report
            cycle_duration = (datetime.now() - cycle_start).total_seconds()
            logger.info(f"Daily cycle completed in {cycle_duration:.1f} seconds")
            logger.info("=" * 60)
            
            return {
                'status': 'success',
                'duration_seconds': cycle_duration,
                'trades_count': len(self.daily_trades),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error in daily cycle: {e}", exc_info=True)
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now()
            }
    
    def _trade_symbol(self, symbol: str, data, sentiment_data: Dict):
        """Execute trades for a single symbol."""
        
        # Get sentiment if available
        sentiment = sentiment_data.get(symbol, {}).get('combined_sentiment', 0.0)
        
        # Generate signal
        signal, confidence = self.trading_logic.generate_signal(
            data,
            sentiment_score=sentiment,
            ml_prediction=0.5  # Default if ML model not trained
        )
        
        # Log signal
        self.trade_logger.log_signal(
            symbol=symbol,
            signal=signal.name,
            confidence=confidence,
            price=data.iloc[-1]['Close'],
            indicators={
                'RSI': data.iloc[-1].get('RSI', 0),
                'MACD': data.iloc[-1].get('MACD', 0),
                'EMA_Cross': 1 if data.iloc[-1]['EMA_12'] > data.iloc[-1]['EMA_26'] else -1
            },
            sentiment_data=sentiment_data.get(symbol)
        )
        
        # Execute trades based on signal
        current_price = data.iloc[-1]['Close']
        latest_row = data.iloc[-1]
        
        # Calculate volatility
        volatility = data['Returns'].std() if 'Returns' in data.columns else 0.02
        
        # BUY signal
        if signal.value > 1 and confidence > self.trading_logic.min_confidence:
            position_size = self.trading_logic.calculate_position_size(
                self.portfolio.cash,
                confidence,
                volatility
            )
            
            if position_size > 0 and self.portfolio.cash >= position_size:
                quantity = position_size / current_price
                
                self.portfolio.buy(
                    symbol,
                    current_price,
                    position_size,
                    datetime.now()
                )
                
                self.trade_logger.log_trade(
                    trade_type='BUY',
                    symbol=symbol,
                    quantity=quantity,
                    price=current_price,
                    reason=signal.name,
                    portfolio_value=self.portfolio.get_portfolio_value({}),
                    position_size_pct=(position_size / self.portfolio.initial_capital) * 100
                )
                
                self.daily_trades.append({
                    'type': 'BUY',
                    'symbol': symbol,
                    'quantity': quantity,
                    'price': current_price,
                    'timestamp': datetime.now()
                })
                
                logger.info(
                    f"BUY {symbol}: {quantity:.4f} @ ${current_price:.2f} "
                    f"(Signal: {signal.name}, Conf: {confidence:.1%})"
                )
            else:
                reason = "Insufficient capital" if position_size > self.portfolio.cash else "Position size too small"
                self.trade_logger.log_decision_skip(
                    symbol,
                    reason,
                    signal.name,
                    confidence
                )
        
        # SELL signal
        elif signal.value < -1 and confidence > self.trading_logic.min_confidence:
            if symbol in self.portfolio.positions:
                position = self.portfolio.positions[symbol]
                
                self.portfolio.sell(
                    symbol,
                    current_price,
                    position['qty'],
                    datetime.now()
                )
                
                pnl_pct = ((current_price - position['entry_price']) / position['entry_price']) * 100
                
                self.trade_logger.log_trade(
                    trade_type='SELL',
                    symbol=symbol,
                    quantity=position['qty'],
                    price=current_price,
                    reason=signal.name,
                    portfolio_value=self.portfolio.get_portfolio_value({}),
                    position_size_pct=0
                )
                
                self.daily_trades.append({
                    'type': 'SELL',
                    'symbol': symbol,
                    'quantity': position['qty'],
                    'price': current_price,
                    'entry_price': position['entry_price'],
                    'pnl_pct': pnl_pct,
                    'timestamp': datetime.now()
                })
                
                logger.info(
                    f"SELL {symbol}: {position['qty']:.4f} @ ${current_price:.2f} "
                    f"(PnL: {pnl_pct:+.2f}%, Signal: {signal.name})"
                )
        
        else:
            # Hold or skip
            self.trade_logger.log_decision_skip(
                symbol,
                "Insufficient signal strength or low confidence",
                signal.name,
                confidence
            )
    
    def _retrain_ml_model(self):
        """Retrain ML model if we have enough trade history."""
        trades = self.trade_logger.get_trades()
        
        if len(trades) < self.config['ml']['min_historical_trades']:
            logger.info(
                f"Insufficient trade history for ML training "
                f"({len(trades)}/{self.config['ml']['min_historical_trades']})"
            )
            return
        
        try:
            # Get market data for training
            sample_data = self.collector.fetch_crypto_data('BTC-USD')
            
            if sample_data.empty:
                logger.warning("Could not fetch training data")
                return
            
            # Prepare training data
            features, labels, feature_names = self.ml_trainer.prepare_training_data(
                sample_data,
                trades
            )
            
            if len(features) < 20:
                logger.warning("Insufficient training samples")
                return
            
            # Train model
            metrics = self.ml_trainer.train(features, labels)
            
            if metrics:
                logger.info(
                    f"ML Model trained - Train Acc: {metrics.get('train_accuracy', 0):.3f}, "
                    f"Test Acc: {metrics.get('test_accuracy', 0):.3f}"
                )
                
                # Save model
                model_path = Path('models') / f'model_{datetime.now().strftime("%Y%m%d")}.pkl'
                model_path.parent.mkdir(exist_ok=True)
                self.ml_trainer.save_model(str(model_path))
        
        except Exception as e:
            logger.error(f"Error retraining ML model: {e}")
    
    def _record_performance(self):
        """Record daily performance metrics."""
        try:
            # Calculate stats
            stats = self.portfolio.get_portfolio_stats({})
            
            # Calculate win rate from trades
            executed_trades = [t for t in self.daily_trades if 'pnl_pct' in t]
            win_rate = len([t for t in executed_trades if t['pnl_pct'] > 0]) / len(executed_trades) if executed_trades else 0
            
            # Record
            self.perf_tracker.record_snapshot(
                symbol='PORTFOLIO',
                portfolio_value=stats['current_value'],
                cash_available=stats['cash'],
                num_positions=stats['num_positions'],
                num_trades=stats['num_trades'],
                daily_return_pct=stats['return_pct'],
                win_rate=win_rate,
                sharpe_ratio=None,
                max_drawdown_pct=None
            )
            
            logger.info(f"Portfolio Value: ${stats['current_value']:.2f} " 
                       f"(Return: {stats['return_pct']:+.2f}%)")
            
        except Exception as e:
            logger.error(f"Error recording performance: {e}")
    
    def schedule_daily_run(self):
        """Schedule bot to run daily."""
        scheduler = BlockingScheduler()
        
        # Parse run time from config
        run_time = self.config.get('scheduler', {}).get('run_time', '09:00')
        hour, minute = map(int, run_time.split(':'))
        
        # Get timezone
        tz = pytz.timezone(
            self.config.get('scheduler', {}).get('timezone', 'UTC')
        )
        
        # Schedule job
        scheduler.add_job(
            self.run_daily_cycle,
            CronTrigger(hour=hour, minute=minute, timezone=tz),
            id='daily_trading_run',
            name='Daily Trading Cycle',
            replace_existing=True
        )
        
        logger.info(f"Bot scheduled to run daily at {run_time} {tz}")
        logger.info("Starting scheduler...")
        
        try:
            scheduler.start()
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user")
    
    def run_once(self):
        """Run one cycle immediately."""
        return self.run_daily_cycle()
    
    def run_backtest(self, symbol: str, days: int = 90):
        """Run backtest for a symbol."""
        logger.info(f"Starting backtest for {symbol} ({days} days)")
        
        # Fetch historical data
        self.collector.lookback_days = days
        data = self.collector.fetch_crypto_data(symbol)
        
        if data.empty:
            logger.error(f"Could not fetch data for {symbol}")
            return None
        
        # Run backtest
        engine = BacktestEngine(
            initial_capital=self.config['trading']['initial_capital']
        )
        
        def signal_gen(data, sentiment, ml_prob):
            return self.trading_logic.generate_signal(data, sentiment, ml_prob)
        
        results = engine.run_backtest(
            symbol,
            data,
            signal_gen
        )
        
        if results:
            logger.info(f"Backtest Results for {symbol}:")
            logger.info(f"  Final Value: ${results.get('final_value', 0):.2f}")
            logger.info(f"  Return: {results.get('total_return_pct', 0):+.2f}%")
            logger.info(f"  Win Rate: {results.get('win_rate', 0):.1%}")
            logger.info(f"  Sharpe Ratio: {results.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max Drawdown: {results.get('max_drawdown_pct', 0):.2f}%")
        
        return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Crypto Trading Bot')
    parser.add_argument('--mode', choices=['run', 'schedule', 'backtest'], 
                       default='run', help='Running mode')
    parser.add_argument('--symbol', default='BTC-USD', help='Symbol for backtest')
    parser.add_argument('--days', type=int, default=90, help='Backtest days')
    parser.add_argument('--config', default='config.json', help='Config file path')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = TradingBot(config_path=args.config)
    
    if args.mode == 'run':
        logger.info("Running single trading cycle")
        result = bot.run_once()
        logger.info(f"Cycle result: {result}")
    
    elif args.mode == 'schedule':
        logger.info("Starting scheduled bot")
        bot.schedule_daily_run()
    
    elif args.mode == 'backtest':
        logger.info(f"Running backtest for {args.symbol}")
        bot.run_backtest(args.symbol, args.days)


if __name__ == "__main__":
    main()
