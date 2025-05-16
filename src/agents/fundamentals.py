from langchain_core.messages import HumanMessage
from src.graph.state import AgentState, show_agent_reasoning
from src.utils.progress import progress
import json
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig
import logging

from src.tools.api import get_financial_metrics


##### Fundamental Agent #####
def fundamentals_agent(state: AgentState):
    """Analyzes fundamental data and generates trading signals for multiple tickers."""
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    # Initialize fundamental analysis for each ticker
    fundamental_analysis = {}

    for ticker in tickers:
        progress.update_status("fundamentals_agent", ticker, "Fetching financial metrics")

        # Get the financial metrics
        financial_metrics = get_financial_metrics(
            ticker=ticker,
            end_date=end_date,
            period="ttm",
            limit=10,
        )

        if not financial_metrics:
            progress.update_status("fundamentals_agent", ticker, "Failed: No financial metrics found")
            continue

        # Pull the most recent financial metrics
        metrics = financial_metrics[0]

        # Initialize signals list for different fundamental aspects
        signals = []
        reasoning = {}

        progress.update_status("fundamentals_agent", ticker, "Analyzing profitability")
        # 1. Profitability Analysis
        return_on_equity = metrics.return_on_equity
        net_margin = metrics.net_margin
        operating_margin = metrics.operating_margin

        thresholds = [
            (return_on_equity, 0.15),  # Strong ROE above 15%
            (net_margin, 0.20),  # Healthy profit margins
            (operating_margin, 0.15),  # Strong operating efficiency
        ]
        profitability_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if profitability_score >= 2 else "bearish" if profitability_score == 0 else "neutral")
        reasoning["profitability_signal"] = {
            "signal": signals[0],
            "details": (f"ROE: {return_on_equity:.2%}" if return_on_equity else "ROE: N/A") + ", " + (f"Net Margin: {net_margin:.2%}" if net_margin else "Net Margin: N/A") + ", " + (f"Op Margin: {operating_margin:.2%}" if operating_margin else "Op Margin: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing growth")
        # 2. Growth Analysis
        revenue_growth = metrics.revenue_growth
        earnings_growth = metrics.earnings_growth
        book_value_growth = metrics.book_value_growth

        thresholds = [
            (revenue_growth, 0.10),  # 10% revenue growth
            (earnings_growth, 0.10),  # 10% earnings growth
            (book_value_growth, 0.10),  # 10% book value growth
        ]
        growth_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bullish" if growth_score >= 2 else "bearish" if growth_score == 0 else "neutral")
        reasoning["growth_signal"] = {
            "signal": signals[1],
            "details": (f"Revenue Growth: {revenue_growth:.2%}" if revenue_growth else "Revenue Growth: N/A") + ", " + (f"Earnings Growth: {earnings_growth:.2%}" if earnings_growth else "Earnings Growth: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing financial health")
        # 3. Financial Health
        current_ratio = metrics.current_ratio
        debt_to_equity = metrics.debt_to_equity
        free_cash_flow_per_share = metrics.free_cash_flow_per_share
        earnings_per_share = metrics.earnings_per_share

        health_score = 0
        if current_ratio and current_ratio > 1.5:  # Strong liquidity
            health_score += 1
        if debt_to_equity and debt_to_equity < 0.5:  # Conservative debt levels
            health_score += 1
        if free_cash_flow_per_share and earnings_per_share and free_cash_flow_per_share > earnings_per_share * 0.8:  # Strong FCF conversion
            health_score += 1

        signals.append("bullish" if health_score >= 2 else "bearish" if health_score == 0 else "neutral")
        reasoning["financial_health_signal"] = {
            "signal": signals[2],
            "details": (f"Current Ratio: {current_ratio:.2f}" if current_ratio else "Current Ratio: N/A") + ", " + (f"D/E: {debt_to_equity:.2f}" if debt_to_equity else "D/E: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Analyzing valuation ratios")
        # 4. Price to X ratios
        pe_ratio = metrics.price_to_earnings_ratio
        pb_ratio = metrics.price_to_book_ratio
        ps_ratio = metrics.price_to_sales_ratio

        thresholds = [
            (pe_ratio, 25),  # Reasonable P/E ratio
            (pb_ratio, 3),  # Reasonable P/B ratio
            (ps_ratio, 5),  # Reasonable P/S ratio
        ]
        price_ratio_score = sum(metric is not None and metric > threshold for metric, threshold in thresholds)

        signals.append("bearish" if price_ratio_score >= 2 else "bullish" if price_ratio_score == 0 else "neutral")
        reasoning["price_ratios_signal"] = {
            "signal": signals[3],
            "details": (f"P/E: {pe_ratio:.2f}" if pe_ratio else "P/E: N/A") + ", " + (f"P/B: {pb_ratio:.2f}" if pb_ratio else "P/B: N/A") + ", " + (f"P/S: {ps_ratio:.2f}" if ps_ratio else "P/S: N/A"),
        }

        progress.update_status("fundamentals_agent", ticker, "Calculating final signal")
        # Determine overall signal
        bullish_signals = signals.count("bullish")
        bearish_signals = signals.count("bearish")

        if bullish_signals > bearish_signals:
            overall_signal = "bullish"
        elif bearish_signals > bullish_signals:
            overall_signal = "bearish"
        else:
            overall_signal = "neutral"

        # Calculate confidence level
        total_signals = len(signals)
        confidence = round(max(bullish_signals, bearish_signals) / total_signals, 2) * 100

        fundamental_analysis[ticker] = {
            "signal": overall_signal,
            "confidence": confidence,
            "reasoning": reasoning,
        }

        progress.update_status("fundamentals_agent", ticker, "Done")

    # Create the fundamental analysis message
    message = HumanMessage(
        content=json.dumps(fundamental_analysis),
        name="fundamentals_agent",
    )

    # Print the reasoning if the flag is set
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(fundamental_analysis, "Fundamental Analysis Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["fundamentals_agent"] = fundamental_analysis

    return {
        "messages": [message],
        "data": data,
    }

class FundamentalsCryptoAgent:
    def __init__(self, config: BinanceConfig, data_provider: BinanceDataProvider, executor: BinanceExecutor):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        self.positions = {}

    def has_strong_fundamentals(self, symbol: str) -> bool:
        # Placeholder: Replace with real on-chain/project fundamentals
        fundamentals = self.data_provider.get_crypto_fundamentals(symbol) if hasattr(self.data_provider, 'get_crypto_fundamentals') else {}
        liquidity_score = fundamentals.get('liquidity_score', 8)
        decentralization_score = fundamentals.get('decentralization_score', 8)
        tokenomics_score = fundamentals.get('tokenomics_score', 8)
        return liquidity_score > 7 and decentralization_score > 7 and tokenomics_score > 7

    def run_strategy(self, symbol: str):
        price = self.data_provider.get_ticker_price(symbol)
        if self.has_strong_fundamentals(symbol):
            quantity = self.executor.calculate_position_size(symbol, price) if hasattr(self.executor, 'calculate_position_size') else 1
            if quantity > 0:
                self.executor.create_order(symbol, 'BUY', 'MARKET', quantity)
        # Add sell/exit logic as needed

    def run_all_symbols(self):
        for symbol in self.config.trading_pairs:
            self.run_strategy(symbol)

class FundamentalsAgent:
    def __init__(self, config, data_provider, executor):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor

    def consult_crypto(self, symbol, timeframe, model_used="fundamentals"):
        df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=90)
        latest_close = df['close'].iloc[-1]
        avg_close = df['close'].rolling(window=90).mean().iloc[-1]
        # For demonstration, treat above-average price as strong fundamentals
        if latest_close > avg_close:
            signal = "BULLISH"
            action = "LONG"
            confidence = 80
            reasoning = "Strong fundamentals detected (price above 90-period average)."
            entry_condition = f"Buy if fundamentals are strong."
            exit_condition = f"Sell if fundamentals weaken."
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "Fundamentals are not strong."
            entry_condition = "Wait for strong fundamentals."
            exit_condition = "N/A"

        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: {self.format_price(latest_close)}. "
            f"90-period MA: {self.format_price(avg_close)}."
        )

        return {
            "agent": "fundamentals",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action == "LONG" else 0,
            "quantity_explanation": "Fundamentals-based position size.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If fundamentals change, reconsider.",
            "suggested_duration": "Hold while fundamentals are strong.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: {self.format_price(latest_close)} | 90-MA: {self.format_price(avg_close)}`\n"
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
