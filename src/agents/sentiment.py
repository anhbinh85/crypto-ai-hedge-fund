from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import pandas as pd
import numpy as np
import json
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig
import logging

from src.tools.api import get_insider_trades, get_company_news


##### Sentiment Agent #####
def sentiment_agent(state: AgentState):
    """Analyzes market sentiment and generates trading signals for multiple tickers."""
    data = state.get("data", {})
    end_date = data.get("end_date")
    tickers = data.get("tickers")

    # Initialize sentiment analysis for each ticker
    sentiment_analysis = {}

    for ticker in tickers:
        progress.update_status("sentiment_agent", ticker, "Fetching insider trades")

        # Get the insider trades
        insider_trades = get_insider_trades(
            ticker=ticker,
            end_date=end_date,
            limit=1000,
        )

        progress.update_status("sentiment_agent", ticker, "Analyzing trading patterns")

        # Get the signals from the insider trades
        transaction_shares = pd.Series([t.transaction_shares for t in insider_trades]).dropna()
        insider_signals = np.where(transaction_shares < 0, "bearish", "bullish").tolist()

        progress.update_status("sentiment_agent", ticker, "Fetching company news")

        # Get the company news
        company_news = get_company_news(ticker, end_date, limit=100)

        # Get the sentiment from the company news
        sentiment = pd.Series([n.sentiment for n in company_news]).dropna()
        news_signals = np.where(sentiment == "negative", "bearish", 
                              np.where(sentiment == "positive", "bullish", "neutral")).tolist()
        
        progress.update_status("sentiment_agent", ticker, "Combining signals")
        # Combine signals from both sources with weights
        insider_weight = 0.3
        news_weight = 0.7
        
        # Calculate weighted signal counts
        bullish_signals = (
            insider_signals.count("bullish") * insider_weight +
            news_signals.count("bullish") * news_weight
        )
        bearish_signals = (
            insider_signals.count("bearish") * insider_weight +
            news_signals.count("bearish") * news_weight
        )

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level based on the weighted proportion
        total_weighted_signals = len(insider_signals) * insider_weight + len(news_signals) * news_weight
        confidence = 0  # Default confidence when there are no signals
        if total_weighted_signals > 0:
            confidence = round(max(bullish_signals, bearish_signals) / total_weighted_signals, 2) * 100
        reasoning = f"Weighted Bullish signals: {bullish_signals:.1f}, Weighted Bearish signals: {bearish_signals:.1f}"

        sentiment_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("sentiment_agent", ticker, "Done")

    # Create the sentiment message
    message = HumanMessage(
        content=json.dumps(sentiment_analysis),
        name="sentiment_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(sentiment_analysis, "Sentiment Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["sentiment_agent"] = sentiment_analysis

    return {
        "messages": [message],
        "data": data,
    }

class SentimentCryptoAgent:
    def __init__(self, config: BinanceConfig, data_provider: BinanceDataProvider, executor: BinanceExecutor):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        self.positions = {}

    def is_sentiment_positive(self, symbol: str) -> bool:
        # Placeholder: Replace with real sentiment logic (e.g., news, social, on-chain sentiment)
        fundamentals = self.data_provider.get_crypto_fundamentals(symbol) if hasattr(self.data_provider, 'get_crypto_fundamentals') else {}
        sentiment_score = fundamentals.get('sentiment_score', 7)  # Example: use a default value
        return sentiment_score > 6

    def run_strategy(self, symbol: str):
        price = self.data_provider.get_ticker_price(symbol)
        if self.is_sentiment_positive(symbol):
            quantity = self.executor.calculate_position_size(symbol, price) if hasattr(self.executor, 'calculate_position_size') else 1
            if quantity > 0:
                self.executor.create_order(symbol, 'BUY', 'MARKET', quantity)
        # Add sell/exit logic as needed

    def run_all_symbols(self):
        for symbol in self.config.trading_pairs:
            self.run_strategy(symbol)

class SentimentAgent:
    def __init__(self, config, data_provider, executor):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor

    def consult_crypto(self, symbol, timeframe, model_used="sentiment"):
        df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=30)
        latest_close = df['close'].iloc[-1]
        # For demonstration, use price change as a proxy for sentiment
        price_change = (latest_close - df['close'].iloc[0]) / df['close'].iloc[0]
        if price_change > 0.05:
            signal = "BULLISH"
            action = "LONG"
            confidence = 75
            reasoning = "Market sentiment is positive (price up >5%)."
            entry_condition = "Buy if sentiment is bullish."
            exit_condition = "Sell if sentiment turns bearish."
        elif price_change < -0.05:
            signal = "BEARISH"
            action = "SHORT"
            confidence = 75
            reasoning = "Market sentiment is negative (price down >5%)."
            entry_condition = "Short if sentiment is bearish."
            exit_condition = "Cover if sentiment turns bullish."
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "Market sentiment is neutral."
            entry_condition = "Wait for clear sentiment."
            exit_condition = "N/A"

        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: {self.format_price(latest_close)}. "
            f"Price change over period: {self.format_price(price_change*100)}%."
        )

        return {
            "agent": "sentiment",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "Sentiment-based position size.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If sentiment changes, reconsider.",
            "suggested_duration": "Hold while sentiment is positive.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: {self.format_price(latest_close)} | Price Change: {self.format_price(price_change*100)}%`\n"
                f"**Signal:** {signal}\n"
                f"**Action:** {action} | **Quantity:** 1\n"
                f"**Entry:** {entry_condition}\n"
                f"**Exit:** {exit_condition}\n"
                f"**Confidence:** {confidence}%\n"
                f"**Reasoning:** {reasoning}\n"
                f"**Model Used:** {model_used}\n"
            )
        }

    def format_price(self, price):
        if price == 0:
            return "$0.00"
        abs_price = abs(price)
        if abs_price >= 1:
            return f"${price:.2f}"
        elif abs_price >= 0.01:
            return f"${price:.4f}"
        elif abs_price >= 0.0001:
            return f"${price:.6f}"
        elif abs_price >= 0.00000001:
            return f"${price:.8f}"
        else:
            return f"${price:.2e}"
