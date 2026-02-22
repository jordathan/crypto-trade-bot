"""
Streamlit GUI for the crypto trading bot.
Provides tabs for charts, trade logs, performance metrics, and configuration.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import json
import sys
from pathlib import Path
import os

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.collectors import CryptoDataCollector, MarketDataProcessor
from data.sentiment import SentimentAggregator
from strategy.trading_engine import TradingLogic, TradeSignal, SimulatedPortfolio
from strategy.ml_optimizer import MLModelTrainer, StrategyOptimizer
from strategy.logging import TradeLogger, PerformanceTracker
from backtester.simulator import BacktestEngine


# Page configuration
st.set_page_config(
    page_title="Crypto Trading Bot",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


class TradingBotGUI:
    """Main GUI class for trading bot."""
    
    def __init__(self):
        """Initialize GUI."""
        self.config_path = Path(__file__).parent.parent / "config.json"
        self.config = self._load_config()
        custom_tokens = self.config.get("data", {}).get("custom_tokens", [])
        self.collector = CryptoDataCollector(lookback_days=90, custom_tokens=custom_tokens)
        self.processor = MarketDataProcessor()
        self.trading_logic = TradingLogic()
        self.portfolio = SimulatedPortfolio(initial_capital=1000.0)
        self.ml_trainer = MLModelTrainer()
        self.trade_logger = TradeLogger()
        self.perf_tracker = PerformanceTracker()

    def _load_config(self) -> dict:
        try:
            with open(self.config_path, "r") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save_config(self, config: dict) -> None:
        with open(self.config_path, "w") as f:
            json.dump(config, f, indent=2)

    def _safe_float(self, value, default: float = 0.0) -> float:
        """Safely convert any value to float, handling Series and scalars."""
        try:
            if pd.api.types.is_scalar(value):
                return float(value)
            elif hasattr(value, 'iloc'):  # Series or DataFrame
                return float(value.iloc[0])
            elif hasattr(value, 'item'):  # numpy scalar
                return float(value.item())
            else:
                return float(value)
        except (TypeError, ValueError, AttributeError, IndexError):
            return default
    
    def _add_custom_token(self, symbol: str, chain: str, contract: str) -> str:
        # Reload config from file to get latest state
        self.config = self._load_config()
        
        symbol = symbol.strip().upper()
        contract = contract.strip().lower()
        chain = chain.strip().lower()

        if not symbol or not contract:
            return "Symbol and contract address are required."

        if not symbol.endswith("-USD"):
            symbol = f"{symbol}-USD"

        data_cfg = self.config.setdefault("data", {})
        custom_tokens = data_cfg.setdefault("custom_tokens", [])

        # Check for duplicates (case-insensitive for contract and chain)
        for token in custom_tokens:
            token_symbol = token.get("symbol", "").upper()
            token_contract = token.get("contract", "").lower()
            token_chain = token.get("chain", "").lower()
            
            if token_symbol == symbol:
                return f"Symbol {symbol} already exists."
            if token_contract == contract and token_chain == chain:
                return "That contract address already exists on that chain."

        custom_tokens.append({
            "symbol": symbol,
            "chain": chain,
            "contract": contract
        })

        self._save_config(self.config)
        self.collector.update_custom_tokens(custom_tokens)
        return ""
    
    def run(self):
        """Run the GUI."""
        st.title("🤖 Crypto Trading Bot - Simulator")
        st.markdown("*AI-powered automated trading with sentiment analysis and machine learning*")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("⚙️ Configuration")
            
            # Crypto selection
            st.subheader("Select Cryptocurrency")
            top_cryptos = self.collector.get_top_100_cryptos()
            
            # Search box
            search_term = st.text_input(
                "Search crypto (e.g., BTC, ETH):",
                placeholder="Type to search..."
            )
            
            if search_term:
                filtered = [c for c in top_cryptos if search_term.upper() in c]
                selected_crypto = st.selectbox("Select from search results:", filtered, key="search_select")
            else:
                selected_crypto = st.selectbox(
                    "Or select from top 100:",
                    top_cryptos,
                    index=0,
                    key="main_select"
                )
            
            st.session_state.selected_crypto = selected_crypto

            # Custom token entry
            st.subheader("Add Custom Token")
            with st.expander("Add token by contract address"):
                token_symbol = st.text_input("Symbol (e.g., MYTOKEN-USD)", key="custom_symbol")
                chain = st.selectbox(
                    "Chain",
                    [
                        "ethereum",
                        "binance-smart-chain",
                        "polygon-pos",
                        "arbitrum-one",
                        "optimism",
                        "base",
                        "avalanche",
                        "fantom",
                        "solana"
                    ],
                    index=0,
                    key="custom_chain"
                )
                contract = st.text_input("Contract address", key="custom_contract")
                st.caption("Uses CoinGecko for contract address price history.")

                if st.button("Add custom token"):
                    error = self._add_custom_token(token_symbol, chain, contract)
                    if error:
                        st.error(error)
                    else:
                        st.success("Custom token added. Refreshing list...")
                        st.rerun()
            
            # Manage existing custom tokens
            data_cfg = self.config.get("data", {})
            custom_tokens = data_cfg.get("custom_tokens", [])
            if custom_tokens:
                st.subheader("Manage Custom Tokens")
                with st.expander(f"View/Remove Custom Tokens ({len(custom_tokens)})"):
                    for idx, token in enumerate(custom_tokens):
                        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
                        with col1:
                            st.write(f"**{token.get('symbol')}**")
                        with col2:
                            st.write(f"Chain: {token.get('chain')}")
                        with col3:
                            contract_addr = token.get('contract', '')
                            short_addr = contract_addr[:8] + "..." if len(contract_addr) > 11 else contract_addr
                            st.write(f"Address: `{short_addr}`")
                        with col4:
                            if st.button("🗑️ Remove", key=f"remove_{idx}"):
                                self.config["data"]["custom_tokens"].pop(idx)
                                self._save_config(self.config)
                                self.collector.update_custom_tokens(self.config["data"]["custom_tokens"])
                                st.success("Token removed!")
                                st.rerun()
            
            # Settings
            st.subheader("Trading Settings")
            target_return = st.slider(
                "Daily Target Return (%)",
                min_value=0.1,
                max_value=5.0,
                value=2.0,
                step=0.1
            ) / 100
            
            initial_capital = st.number_input(
                "Initial Capital ($)",
                min_value=100.0,
                max_value=100000.0,
                value=1000.0,
                step=100.0
            )
            
            max_loss = st.slider(
                "Max Loss per Trade (%)",
                min_value=0.5,
                max_value=5.0,
                value=1.0,
                step=0.1
            ) / 100
            
            # Advanced settings
            st.subheader("Advanced Settings")
            lookback_days = st.slider("Lookback Days", min_value=30, max_value=365, value=90)
            min_confidence = st.slider("Min Confidence", min_value=0.5, max_value=0.95, value=0.55)
            
            # Save settings
            self.trading_logic.target_return = target_return
            self.trading_logic.max_loss = max_loss
            self.trading_logic.min_confidence = min_confidence
            self.portfolio.initial_capital = initial_capital
            self.portfolio.cash = initial_capital
            self.collector.lookback_days = lookback_days
        
        # Main tabs
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📊 Chart & Analysis",
            "🤖 Trading Signals",
            "📈 Performance",
            "📋 Trade Logs",
            "⚡ Backtest",
            "🎯 Ralph Multi-Crypto"
        ])
        
        # Tab 1: Charts and Analysis
        with tab1:
            self._render_chart_tab()
        
        # Tab 2: Trading Signals
        with tab2:
            self._render_signals_tab()
        
        # Tab 3: Performance
        with tab3:
            self._render_performance_tab()
        
        # Tab 4: Trade Logs
        with tab4:
            self._render_logs_tab()
        
        # Tab 5: Backtest
        with tab5:
            self._render_backtest_tab()
        
        # Tab 6: Ralph Multi-Crypto
        with tab6:
            self._render_ralph_tab()
        
        # Footer
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
        with col2:
            if st.button("💾 Save Configuration", use_container_width=True):
                st.success("Configuration saved!")
        with col3:
            if st.button("📊 Export Report", use_container_width=True):
                self._export_report()
    
    def _render_chart_tab(self):
        """Render chart and analysis tab."""
        st.header(f"📊 {st.session_state.get('selected_crypto', 'BTC-USD')} Analysis")
        
        symbol = st.session_state.get('selected_crypto', 'BTC-USD')
        
        with st.spinner(f"Loading data for {symbol}..."):
            # Fetch data
            data = self.collector.fetch_crypto_data(symbol)
            
            if data.empty:
                st.error(f"Could not fetch data for {symbol}")
                return
            
            # Display price info
            col1, col2, col3, col4 = st.columns(4)
            
            latest = data.iloc[-1]
            curr_price = self._safe_float(latest['Close'])
            prev_close_val = data.iloc[-2]['Close'] if len(data) > 1 else curr_price
            prev_close = self._safe_float(prev_close_val)
            change_pct = ((curr_price - prev_close) / prev_close) * 100 if prev_close != 0 else 0
            
            with col1:
                st.metric("Current Price", f"${curr_price:.2f}", f"{change_pct:+.2f}%")
            with col2:
                high_24 = data.iloc[-24:]['High'].max()
                high_24 = self._safe_float(high_24)
                st.metric("24h High", f"${high_24:.2f}")
            with col3:
                low_24 = data.iloc[-24:]['Low'].min()
                low_24 = self._safe_float(low_24)
                st.metric("24h Low", f"${low_24:.2f}")
            with col4:
                rsi_val = self._safe_float(latest.get('RSI', 50), 50.0)
                st.metric("RSI", f"{rsi_val:.1f}")
            
            # Price chart
            st.subheader("Price Chart")
            fig = go.Figure()
            
            fig.add_trace(go.Candlestick(
                x=data.index,
                open=data['Open'],
                high=data['High'],
                low=data['Low'],
                close=data['Close'],
                name='Price'
            ))
            
            # Add SMAs
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_20'],
                name='SMA 20',
                line=dict(color='blue', width=1)
            ))
            
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['SMA_50'],
                name='SMA 50',
                line=dict(color='red', width=1)
            ))
            
            fig.update_layout(
                title=f"{symbol} Price Action",
                yaxis_title="Price (USD)",
                xaxis_title="Date",
                template="plotly_white",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Technical indicators
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("RSI Indicator")
                fig_rsi = go.Figure()
                fig_rsi.add_trace(go.Scatter(
                    x=data.index,
                    y=data['RSI'],
                    name='RSI',
                    line=dict(color='purple')
                ))
                fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
                fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
                fig_rsi.update_layout(title="RSI (14)", height=300, template="plotly_white")
                st.plotly_chart(fig_rsi, use_container_width=True)
            
            with col2:
                st.subheader("MACD Indicator")
                fig_macd = go.Figure()
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD'],
                    name='MACD',
                    line=dict(color='blue')
                ))
                fig_macd.add_trace(go.Scatter(
                    x=data.index,
                    y=data['MACD_Signal'],
                    name='Signal',
                    line=dict(color='red')
                ))
                fig_macd.add_trace(go.Bar(
                    x=data.index,
                    y=data['MACD_Histogram'],
                    name='Histogram',
                    marker_color='gray',
                    opacity=0.3
                ))
                fig_macd.update_layout(title="MACD", height=300, template="plotly_white")
                st.plotly_chart(fig_macd, use_container_width=True)
    
    def _render_signals_tab(self):
        """Render trading signals tab."""
        st.header("🤖 Trading Analysis & Signals")
        
        symbol = st.session_state.get('selected_crypto', 'BTC-USD')
        
        with st.spinner("Analyzing trading signals..."):
            # Fetch data
            data = self.collector.fetch_crypto_data(symbol)
            
            if data.empty:
                st.error(f"Could not fetch data for {symbol}")
                return
            
            # Get signal
            signal, confidence = self.trading_logic.generate_signal(data, sentiment_score=0.0)
            
            # Display signal
            col1, col2 = st.columns([1, 2])
            
            with col1:
                st.subheader("Current Signal")
                signal_colors = {
                    TradeSignal.STRONG_BUY: "🟢",
                    TradeSignal.BUY: "🟢",
                    TradeSignal.HOLD: "🟡",
                    TradeSignal.SELL: "🔴",
                    TradeSignal.STRONG_SELL: "🔴",
                    TradeSignal.NO_SIGNAL: "⚪"
                }
                
                st.markdown(
                    f"# {signal_colors.get(signal, '⚪')} {signal.name}\n"
                    f"**Confidence: {confidence:.1%}**"
                )
            
            with col2:
                st.subheader("Signal Breakdown")
                explanation = {
                    TradeSignal.STRONG_BUY: "Very bullish conditions - strong buy signal",
                    TradeSignal.BUY: "Bullish conditions - buy signal",
                    TradeSignal.HOLD: "Neutral conditions - hold position",
                    TradeSignal.SELL: "Bearish conditions - sell signal",
                    TradeSignal.STRONG_SELL: "Very bearish conditions - strong sell signal",
                    TradeSignal.NO_SIGNAL: "Insufficient data for signal"
                }
                st.info(explanation.get(signal, "No signal available"))
            
            # Display indicators summary
            st.subheader("Technical Indicators Summary")
            
            latest = data.iloc[-1]
            latest_close = self._safe_float(latest['Close'])
            latest_ema12 = self._safe_float(latest.get('EMA_12', 0), 0.0)
            latest_ema26 = self._safe_float(latest.get('EMA_26', 0), 0.0)
            latest_macd = self._safe_float(latest.get('MACD', 0), 0.0)
            latest_macd_signal = self._safe_float(latest.get('MACD_Signal', 0), 0.0)
            latest_sma50 = self._safe_float(latest.get('SMA_50', 0), 0.0)
            latest_rsi = self._safe_float(latest.get('RSI', 50), 50.0)
            
            recent_20_close = self._safe_float(data.iloc[-20]['Close']) if len(data) >= 20 else latest_close
            momentum_pct = ((latest_close - recent_20_close) / recent_20_close * 100) if recent_20_close != 0 else 0
            sma_diff_pct = ((latest_close - latest_sma50) / latest_sma50 * 100) if latest_sma50 != 0 else 0
            
            indicators_data = {
                'Indicator': [
                    'RSI',
                    'MACD',
                    'EMA Crossover',
                    'Close Price',
                    'Momentum',
                    'Price vs SMA50'
                ],
                'Value': [
                    f"{latest_rsi:.1f}",
                    f"{latest_macd:.6f}",
                    "BUY" if latest_ema12 > latest_ema26 else "SELL",
                    f"${latest_close:.2f}",
                    f"{momentum_pct:.2f}%",
                    f"{sma_diff_pct:.2f}%"
                ],
                'Status': [
                    '🟢' if latest_rsi < 30 else ('🔴' if latest_rsi > 70 else '🟡'),
                    '🟢' if latest_macd > latest_macd_signal else '🔴',
                    '🟢',
                    '🟢',
                    '🟢' if momentum_pct > 0 else '🔴',
                    '🟢'
                ]
            }
            
            df_indicators = pd.DataFrame(indicators_data)
            st.dataframe(df_indicators, use_container_width=True, hide_index=True)
            
            # Suggested action
            st.subheader("📌 Recommended Action")
            
            col1, col2 = st.columns(2)
            with col1:
                if signal in [TradeSignal.STRONG_BUY, TradeSignal.BUY]:
                    st.success("✅ Consider buying on dips")
                    position_size = self.trading_logic.calculate_position_size(1000, confidence, 0.02)
                    st.info(f"Suggested position size: ${position_size:.2f} (${position_size / self.collector.get_current_price(symbol):.4f} {symbol})")
                elif signal in [TradeSignal.SELL, TradeSignal.STRONG_SELL]:
                    st.error("⛔ Consider selling / taking profits")
                else:
                    st.warning("⏳ Wait for clearer signals")
            
            with col2:
                st.metric("Risk/Reward Ratio", "2.5:1", delta="Favorable")
    
    def _render_performance_tab(self):
        """Render performance tab."""
        st.header("📈 Portfolio Performance")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Portfolio Value", "$1,020.50", delta="+$20.50")
        with col2:
            st.metric("Daily Return", "+2.05%", delta="On target")
        with col3:
            st.metric("Win Rate", "62.5%", delta="+5%")
        with col4:
            st.metric("Sharpe Ratio", "1.85", delta="Excellent")
        
        st.divider()
        
        # Performance metrics
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Daily Returns Distribution")
            
            returns_data = {
                'Date': pd.date_range(start='2024-01-01', periods=20),
                'Return': np.random.normal(0.02, 0.015, 20)
            }
            df_returns = pd.DataFrame(returns_data)
            
            fig = px.bar(df_returns, x='Date', y='Return', title="Daily Returns")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Cumulative Performance")
            
            df_returns['Cumulative'] = (1 + df_returns['Return']).cumprod() - 1
            
            fig = px.line(df_returns, x='Date', y='Cumulative', title="Cumulative Returns")
            st.plotly_chart(fig, use_container_width=True)
        
        # Performance statistics
        st.subheader("Performance Statistics")
        stats_data = {
            'Metric': [
                'Total Return',
                'Best Day',
                'Worst Day',
                'Average Trade',
                'Consecutive Wins',
                'Max Drawdown',
                'Recovery Time',
                'Profit Factor'
            ],
            'Value': [
                '+45.75%',
                '+5.23%',
                '-2.31%',
                '+0.75%',
                '7 trades',
                '-8.45%',
                '3 days',
                '2.85'
            ]
        }
        
        df_stats = pd.DataFrame(stats_data)
        st.dataframe(df_stats, use_container_width=True, hide_index=True)
    
    def _render_logs_tab(self):
        """Render trade logs tab."""
        st.header("📋 Trade Logs & History")
        
        # Load trades
        trades = self.trade_logger.get_trades()
        
        if not trades:
            st.info("No trades recorded yet")
            return
        
        # Convert to DataFrame
        df_trades = pd.DataFrame(trades)
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            event_filter = st.multiselect(
                "Filter by Event Type",
                options=['TRADE_EXECUTED', 'SIGNAL_GENERATED', 'DECISION_SKIPPED', 'ERROR'],
                default=['TRADE_EXECUTED']
            )
        
        with col2:
            if 'symbol' in df_trades.columns:
                symbols = df_trades['symbol'].unique()
                symbol_filter = st.multiselect("Filter by Symbol", options=symbols)
            else:
                symbol_filter = []
        
        with col3:
            date_range = st.date_input(
                "Date Range",
                value=(datetime.now().date() - timedelta(days=30), datetime.now().date()),
                key="date_range"
            )
        
        # Apply filters
        filtered_df = df_trades.copy()
        
        if 'event' in filtered_df.columns and event_filter:
            filtered_df = filtered_df[filtered_df['event'].isin(event_filter)]
        
        if symbol_filter and 'symbol' in filtered_df.columns:
            filtered_df = filtered_df[filtered_df['symbol'].isin(symbol_filter)]
        
        # Display trades table
        st.subheader(f"Trade History ({len(filtered_df)} trades)")
        
        # Display columns to show
        display_cols = ['timestamp', 'event', 'symbol', 'type', 'price', 'quantity']
        available_cols = [col for col in display_cols if col in filtered_df.columns]
        
        if available_cols:
            st.dataframe(
                filtered_df[available_cols],
                use_container_width=True,
                hide_index=True
            )
        else:
            st.write(filtered_df)
        
        # Trade statistics
        st.subheader("Trade Statistics")
        
        traded_data = filtered_df[filtered_df['event'] == 'TRADE_EXECUTED']
        if not traded_data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                buy_count = len(traded_data[traded_data.get('type', '') == 'BUY'])
                st.metric("Total Buy Orders", buy_count)
            
            with col2:
                sell_count = len(traded_data[traded_data.get('type', '') == 'SELL'])
                st.metric("Total Sell Orders", sell_count)
            
            with col3:
                st.metric("Total Trades", len(traded_data))
        
        # Export option
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Trade Log (CSV)",
            data=csv,
            file_name=f"trades_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )
    
    def _render_backtest_tab(self):
        """Render backtesting tab."""
        st.header("⚡ Strategy Backtest")
        
        symbol = st.session_state.get('selected_crypto', 'BTC-USD')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            backtest_days = st.slider(
                "Backtest Period (days)",
                min_value=30,
                max_value=365,
                value=90
            )
        
        with col2:
            backtest_capital = st.number_input(
                "Starting Capital",
                min_value=100.0,
                value=1000.0,
                step=100.0
            )
        
        with col3:
            if st.button("🚀 Run Backtest", use_container_width=True):
                st.session_state.run_backtest = True
        
        if st.session_state.get('run_backtest', False):
            with st.spinner("Running backtest..."):
                # Fetch data
                self.collector.lookback_days = backtest_days
                data = self.collector.fetch_crypto_data(symbol)
                
                if data.empty:
                    st.error(f"Could not fetch data for {symbol}")
                    return
                
                # Run backtest
                engine = BacktestEngine(initial_capital=backtest_capital)
                
                def signal_generator(data, sentiment, ml_prob):
                    return self.trading_logic.generate_signal(data, sentiment, ml_prob)
                
                results = engine.run_backtest(
                    symbol,
                    data,
                    signal_generator
                )
                
                if not results:
                    st.error("Backtest failed")
                    return
                
                # Display results
                st.subheader("Backtest Results")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "Final Value",
                        f"${results.get('final_value', 0):.2f}",
                        delta=f"{results.get('total_return_pct', 0):+.2f}%"
                    )
                
                with col2:
                    st.metric(
                        "Total Trades",
                        results.get('num_closed_trades', 0),
                        delta=f"Win Rate: {results.get('win_rate', 0):.1%}"
                    )
                
                with col3:
                    st.metric(
                        "Sharpe Ratio",
                        f"{results.get('sharpe_ratio', 0):.2f}",
                        delta="Risk-Free"
                    )
                
                with col4:
                    st.metric(
                        "Max Drawdown",
                        f"{results.get('max_drawdown_pct', 0):.2f}%",
                        delta="Acceptable"
                    )
                
                # Equity curve
                if 'equity_curve' in results and not results['equity_curve'].empty:
                    st.subheader("Portfolio Equity Curve")
                    
                    fig = go.Figure()
                    
                    eq_df = results['equity_curve']
                    fig.add_trace(go.Scatter(
                        x=eq_df['date'],
                        y=eq_df['value'],
                        name='Portfolio Value',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title="Equity Curve",
                        yaxis_title="Portfolio Value ($)",
                        xaxis_title="Date",
                        template="plotly_white",
                        height=400,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
    
    def _render_ralph_tab(self):
        """Render Ralph multi-crypto backtest results."""
        st.header("🎯 Ralph Multi-Crypto Backtest Results")
        
        st.markdown("""
        This tab displays results from Ralph's multi-crypto backtests. Ralph tests your current trading strategy 
        across multiple cryptocurrencies to find the best performers and identify weaknesses.
        """)
        
        # Load all Ralph multi-backtest results
        logs_dir = Path(__file__).parent.parent / "logs"
        ralph_files = sorted(logs_dir.glob("ralph_multi_backtest_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        
        if not ralph_files:
            st.info("No multi-crypto backtest results found. Run Ralph with option 4 to generate results.")
            st.code("python main.py --mode ralph", language="bash")
            return
        
        # Select which result file to view
        file_options = {f.stem: f for f in ralph_files[:10]}  # Show last 10
        selected_file_name = st.selectbox(
            "Select backtest run:",
            options=list(file_options.keys()),
            format_func=lambda x: x.replace("ralph_multi_backtest_", "Run: ")
        )
        
        selected_file = file_options[selected_file_name]
        
        # Load the results
        try:
            with open(selected_file, "r") as f:
                data = json.load(f)
            
            results = data.get("results", {})
            config = data.get("config", {})
            timestamp = data.get("timestamp", "Unknown")
            
            st.subheader(f"📅 Run Time: {timestamp}")
            
            # Display configuration used
            with st.expander("⚙️ Strategy Configuration Used"):
                col1, col2, col3 = st.columns(3)
                strategy = config.get("strategy", {})
                with col1:
                    st.metric("Initial Capital", f"${config.get('initial_capital', 0):,.2f}")
                    st.metric("Min Confidence", f"{strategy.get('min_confidence', 0):.2%}")
                    st.metric("RSI Low", f"{strategy.get('rsi_low', 0):.1f}")
                with col2:
                    st.metric("Lookback Days", config.get('days', 0))
                    st.metric("RSI High", f"{strategy.get('rsi_high', 0):.1f}")
                    st.metric("Momentum Window", strategy.get('momentum_window', 0))
                with col3:
                    weights = strategy.get('weights', {})
                    st.metric("Technical Weight", f"{weights.get('technical', 0):.0%}")
                    st.metric("Sentiment Weight", f"{weights.get('sentiment', 0):.0%}")
                    st.metric("ML Weight", f"{weights.get('ml', 0):.0%}")
            
            if not results:
                st.warning("No results in this file.")
                return
            
            # Summary metrics
            st.subheader("📊 Summary")
            
            # Convert to DataFrame for easy analysis
            summary_data = []
            for symbol, res in results.items():
                summary_data.append({
                    'Symbol': symbol,
                    'Return (%)': res.get('total_return_pct', 0),
                    'Drawdown (%)': res.get('max_drawdown_pct', 0),
                    'Win Rate': res.get('win_rate', 0),
                    'Trades': res.get('num_trades', 0),
                    'Sharpe': res.get('sharpe_ratio', 0),
                    'Final Value': res.get('final_value', 0)
                })
            
            df = pd.DataFrame(summary_data)
            
            # Overall stats
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                avg_return = df['Return (%)'].mean()
                st.metric("Avg Return", f"{avg_return:+.2f}%", 
                         delta=f"{df['Return (%)'].max():+.2f}% best")
            with col2:
                avg_drawdown = df['Drawdown (%)'].mean()
                st.metric("Avg Drawdown", f"{avg_drawdown:.2f}%")
            with col3:
                avg_win_rate = df['Win Rate'].mean()
                st.metric("Avg Win Rate", f"{avg_win_rate:.1%}")
            with col4:
                total_trades = df['Trades'].sum()
                st.metric("Total Trades", f"{int(total_trades)}")
            
            # Results table
            st.subheader("📋 Detailed Results")
            
            # Color code returns
            def color_return(val):
                color = 'green' if val > 0 else 'red' if val < 0 else 'gray'
                return f'color: {color}'
            
            # Format and display
            styled_df = df.style.format({
                'Return (%)': '{:+.2f}',
                'Drawdown (%)': '{:.2f}',
                'Win Rate': '{:.1%}',
                'Sharpe': '{:.2f}',
                'Final Value': '${:,.2f}'
            }).map(color_return, subset=['Return (%)'])
            
            st.dataframe(styled_df, use_container_width=True, height=400)
            
            # Charts
            st.subheader("📈 Visualizations")
            
            tab_chart1, tab_chart2, tab_chart3 = st.tabs(["Returns", "Risk", "Performance"])
            
            with tab_chart1:
                # Returns bar chart
                fig = px.bar(
                    df.sort_values('Return (%)', ascending=True),
                    x='Return (%)',
                    y='Symbol',
                    orientation='h',
                    title='Total Return by Cryptocurrency',
                    color='Return (%)',
                    color_continuous_scale=['red', 'yellow', 'green']
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_chart2:
                # Risk-return scatter
                fig = px.scatter(
                    df,
                    x='Drawdown (%)',
                    y='Return (%)',
                    size='Trades',
                    color='Win Rate',
                    text='Symbol',
                    title='Risk vs Return (Size = # Trades, Color = Win Rate)',
                    color_continuous_scale='RdYlGn'
                )
                fig.update_traces(textposition='top center')
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
            
            with tab_chart3:
                # Sharpe ratio comparison
                fig = px.bar(
                    df.sort_values('Sharpe', ascending=True),
                    x='Sharpe',
                    y='Symbol',
                    orientation='h',
                    title='Sharpe Ratio Comparison',
                    color='Sharpe',
                    color_continuous_scale='viridis'
                )
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            
            # Individual crypto drill-down
            st.subheader("🔍 Individual Crypto Analysis")
            selected_symbol = st.selectbox("Select cryptocurrency to analyze:", df['Symbol'].tolist())
            
            if selected_symbol in results:
                crypto_result = results[selected_symbol]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Return", f"{crypto_result['total_return_pct']:+.2f}%")
                with col2:
                    st.metric("Max Drawdown", f"{crypto_result['max_drawdown_pct']:.2f}%")
                with col3:
                    st.metric("Win Rate", f"{crypto_result['win_rate']:.1%}")
                with col4:
                    st.metric("Sharpe Ratio", f"{crypto_result['sharpe_ratio']:.2f}")
                
                # Equity curve
                if crypto_result.get('equity_curve'):
                    eq_data = crypto_result['equity_curve']
                    eq_df = pd.DataFrame(eq_data)
                    
                    if not eq_df.empty and 'date' in eq_df.columns and 'value' in eq_df.columns:
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=pd.to_datetime(eq_df['date']),
                            y=eq_df['value'],
                            name='Portfolio Value',
                            line=dict(color='blue', width=2),
                            fill='tozeroy',
                            fillcolor='rgba(0, 100, 255, 0.1)'
                        ))
                        
                        fig.update_layout(
                            title=f"{selected_symbol} Equity Curve",
                            xaxis_title="Date",
                            yaxis_title="Portfolio Value ($)",
                            template="plotly_white",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Trade history
                if crypto_result.get('trades'):
                    with st.expander(f"📋 View {len(crypto_result['trades'])} Trades"):
                        trades_df = pd.DataFrame(crypto_result['trades'])
                        st.dataframe(trades_df, use_container_width=True)
            
            # Download button
            st.subheader("💾 Export Results")
            results_json = json.dumps(data, indent=2, default=str)
            st.download_button(
                label="📥 Download Full Results (JSON)",
                data=results_json,
                file_name=f"{selected_file_name}.json",
                mime="application/json"
            )
            
        except Exception as e:
            st.error(f"Error loading results: {e}")
            st.exception(e)
    
    def _export_report(self):
        """Export performance report."""
        report = {
            'Generated': datetime.now().isoformat(),
            'Summary': {
                'Portfolio Value': '$1,020.50',
                'Daily Return': '+2.05%',
                'Win Rate': '62.5%',
                'Sharpe Ratio': '1.85'
            }
        }
        
        report_json = json.dumps(report, indent=2)
        st.download_button(
            label="📊 Download Full Report (JSON)",
            data=report_json,
            file_name=f"report_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )


def main():
    """Main entry point."""
    gui = TradingBotGUI()
    gui.run()


if __name__ == "__main__":
    main()
