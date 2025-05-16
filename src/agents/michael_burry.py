from __future__ import annotations

from datetime import datetime, timedelta
import json
from typing_extensions import Literal

from src.graph.state import AgentState, show_agent_reasoning
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.tools.api import (
    get_company_news,
    get_financial_metrics,
    get_insider_trades,
    get_market_cap,
    search_line_items,
)
from src.utils.llm import call_llm
from src.utils.progress import progress
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig
import logging

__all__ = [
    "MichaelBurrySignal",
    "michael_burry_agent",
    "MichaelBurryCryptoAgent",
]

###############################################################################
# Pydantic output model
###############################################################################


class MichaelBurrySignal(BaseModel):
    """Schema returned by the LLM."""

    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0–100
    reasoning: str


###############################################################################
# Core agent
###############################################################################


def michael_burry_agent(state: AgentState):  # noqa: C901  (complexity is fine here)
    """Analyse stocks using Michael Burry's deep‑value, contrarian framework."""

    data = state["data"]
    end_date: str = data["end_date"]  # YYYY‑MM‑DD
    tickers: list[str] = data["tickers"]

    # We look one year back for insider trades / news flow
    start_date = (datetime.fromisoformat(end_date) - timedelta(days=365)).date().isoformat()

    analysis_data: dict[str, dict] = {}
    burry_analysis: dict[str, dict] = {}

    for ticker in tickers:
        # ------------------------------------------------------------------
        # Fetch raw data
        # ------------------------------------------------------------------
        progress.update_status("michael_burry_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="ttm", limit=5)

        progress.update_status("michael_burry_agent", ticker, "Fetching line items")
        line_items = search_line_items(
            ticker,
            [
                "free_cash_flow",
                "net_income",
                "total_debt",
                "cash_and_equivalents",
                "total_assets",
                "total_liabilities",
                "outstanding_shares",
                "issuance_or_purchase_of_equity_shares",
            ],
            end_date,
        )

        progress.update_status("michael_burry_agent", ticker, "Fetching insider trades")
        insider_trades = get_insider_trades(ticker, end_date=end_date, start_date=start_date)

        progress.update_status("michael_burry_agent", ticker, "Fetching company news")
        news = get_company_news(ticker, end_date=end_date, start_date=start_date, limit=250)

        progress.update_status("michael_burry_agent", ticker, "Fetching market cap")
        market_cap = get_market_cap(ticker, end_date)

        # ------------------------------------------------------------------
        # Run sub‑analyses
        # ------------------------------------------------------------------
        progress.update_status("michael_burry_agent", ticker, "Analyzing value")
        value_analysis = _analyze_value(metrics, line_items, market_cap)

        progress.update_status("michael_burry_agent", ticker, "Analyzing balance sheet")
        balance_sheet_analysis = _analyze_balance_sheet(metrics, line_items)

        progress.update_status("michael_burry_agent", ticker, "Analyzing insider activity")
        insider_analysis = _analyze_insider_activity(insider_trades)

        progress.update_status("michael_burry_agent", ticker, "Analyzing contrarian sentiment")
        contrarian_analysis = _analyze_contrarian_sentiment(news)

        # ------------------------------------------------------------------
        # Aggregate score & derive preliminary signal
        # ------------------------------------------------------------------
        total_score = (
            value_analysis["score"]
            + balance_sheet_analysis["score"]
            + insider_analysis["score"]
            + contrarian_analysis["score"]
        )
        max_score = (
            value_analysis["max_score"]
            + balance_sheet_analysis["max_score"]
            + insider_analysis["max_score"]
            + contrarian_analysis["max_score"]
        )

        if total_score >= 0.7 * max_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_score:
            signal = "bearish"
        else:
            signal = "neutral"

        # ------------------------------------------------------------------
        # Collect data for LLM reasoning & output
        # ------------------------------------------------------------------
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_score,
            "value_analysis": value_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "insider_analysis": insider_analysis,
            "contrarian_analysis": contrarian_analysis,
            "market_cap": market_cap,
        }

        progress.update_status("michael_burry_agent", ticker, "Generating LLM output")
        burry_output = _generate_burry_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        burry_analysis[ticker] = {
            "signal": burry_output.signal,
            "confidence": burry_output.confidence,
            "reasoning": burry_output.reasoning,
        }

        progress.update_status("michael_burry_agent", ticker, "Done")

    # ----------------------------------------------------------------------
    # Return to the graph
    # ----------------------------------------------------------------------
    message = HumanMessage(content=json.dumps(burry_analysis), name="michael_burry_agent")

    if state["metadata"].get("show_reasoning"):
        show_agent_reasoning(burry_analysis, "Michael Burry Agent")

    state["data"]["analyst_signals"]["michael_burry_agent"] = burry_analysis

    return {"messages": [message], "data": state["data"]}


###############################################################################
# Sub‑analysis helpers
###############################################################################


def _latest_line_item(line_items: list):
    """Return the most recent line‑item object or *None*."""
    return line_items[0] if line_items else None


# ----- Value ----------------------------------------------------------------

def _analyze_value(metrics, line_items, market_cap):
    """Free cash‑flow yield, EV/EBIT, other classic deep‑value metrics."""

    max_score = 6  # 4 pts for FCF‑yield, 2 pts for EV/EBIT
    score = 0
    details: list[str] = []

    # Free‑cash‑flow yield
    latest_item = _latest_line_item(line_items)
    fcf = getattr(latest_item, "free_cash_flow", None) if latest_item else None
    if fcf is not None and market_cap:
        fcf_yield = fcf / market_cap
        if fcf_yield >= 0.15:
            score += 4
            details.append(f"Extraordinary FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.12:
            score += 3
            details.append(f"Very high FCF yield {fcf_yield:.1%}")
        elif fcf_yield >= 0.08:
            score += 2
            details.append(f"Respectable FCF yield {fcf_yield:.1%}")
        else:
            details.append(f"Low FCF yield {fcf_yield:.1%}")
    else:
        details.append("FCF data unavailable")

    # EV/EBIT (from financial metrics)
    if metrics:
        ev_ebit = getattr(metrics[0], "ev_to_ebit", None)
        if ev_ebit is not None:
            if ev_ebit < 6:
                score += 2
                details.append(f"EV/EBIT {ev_ebit:.1f} (<6)")
            elif ev_ebit < 10:
                score += 1
                details.append(f"EV/EBIT {ev_ebit:.1f} (<10)")
            else:
                details.append(f"High EV/EBIT {ev_ebit:.1f}")
        else:
            details.append("EV/EBIT data unavailable")
    else:
        details.append("Financial metrics unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Balance sheet --------------------------------------------------------

def _analyze_balance_sheet(metrics, line_items):
    """Leverage and liquidity checks."""

    max_score = 3
    score = 0
    details: list[str] = []

    latest_metrics = metrics[0] if metrics else None
    latest_item = _latest_line_item(line_items)

    debt_to_equity = getattr(latest_metrics, "debt_to_equity", None) if latest_metrics else None
    if debt_to_equity is not None:
        if debt_to_equity < 0.5:
            score += 2
            details.append(f"Low D/E {debt_to_equity:.2f}")
        elif debt_to_equity < 1:
            score += 1
            details.append(f"Moderate D/E {debt_to_equity:.2f}")
        else:
            details.append(f"High leverage D/E {debt_to_equity:.2f}")
    else:
        details.append("Debt‑to‑equity data unavailable")

    # Quick liquidity sanity check (cash vs total debt)
    if latest_item is not None:
        cash = getattr(latest_item, "cash_and_equivalents", None)
        total_debt = getattr(latest_item, "total_debt", None)
        if cash is not None and total_debt is not None:
            if cash > total_debt:
                score += 1
                details.append("Net cash position")
            else:
                details.append("Net debt position")
        else:
            details.append("Cash/debt data unavailable")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Insider activity -----------------------------------------------------

def _analyze_insider_activity(insider_trades):
    """Net insider buying over the last 12 months acts as a hard catalyst."""

    max_score = 2
    score = 0
    details: list[str] = []

    if not insider_trades:
        details.append("No insider trade data")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    shares_bought = sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) > 0)
    shares_sold = abs(sum(t.transaction_shares or 0 for t in insider_trades if (t.transaction_shares or 0) < 0))
    net = shares_bought - shares_sold
    if net > 0:
        score += 2 if net / max(shares_sold, 1) > 1 else 1
        details.append(f"Net insider buying of {net:,} shares")
    else:
        details.append("Net insider selling")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


# ----- Contrarian sentiment -------------------------------------------------

def _analyze_contrarian_sentiment(news):
    """Very rough gauge: a wall of recent negative headlines can be a *positive* for a contrarian."""

    max_score = 1
    score = 0
    details: list[str] = []

    if not news:
        details.append("No recent news")
        return {"score": score, "max_score": max_score, "details": "; ".join(details)}

    # Count negative sentiment articles
    sentiment_negative_count = sum(
        1 for n in news if n.sentiment and n.sentiment.lower() in ["negative", "bearish"]
    )
    
    if sentiment_negative_count >= 5:
        score += 1  # The more hated, the better (assuming fundamentals hold up)
        details.append(f"{sentiment_negative_count} negative headlines (contrarian opportunity)")
    else:
        details.append("Limited negative press")

    return {"score": score, "max_score": max_score, "details": "; ".join(details)}


###############################################################################
# LLM generation
###############################################################################

def _generate_burry_output(
    ticker: str,
    analysis_data: dict,
    *,
    model_name: str,
    model_provider: str,
) -> MichaelBurrySignal:
    """Call the LLM to craft the final trading signal in Burry's voice."""

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are an AI agent emulating Dr. Michael J. Burry. Your mandate:
                - Hunt for deep value in US equities using hard numbers (free cash flow, EV/EBIT, balance sheet)
                - Be contrarian: hatred in the press can be your friend if fundamentals are solid
                - Focus on downside first – avoid leveraged balance sheets
                - Look for hard catalysts such as insider buying, buybacks, or asset sales
                - Communicate in Burry's terse, data‑driven style

                When providing your reasoning, be thorough and specific by:
                1. Start with the key metric(s) that drove your decision
                2. Cite concrete numbers (e.g. "FCF yield 14.7%", "EV/EBIT 5.3")
                3. Highlight risk factors and why they are acceptable (or not)
                4. Mention relevant insider activity or contrarian opportunities
                5. Use Burry's direct, number-focused communication style with minimal words
                
                For example, if bullish: "FCF yield 12.8%. EV/EBIT 6.2. Debt-to-equity 0.4. Net insider buying 25k shares. Market missing value due to overreaction to recent litigation. Strong buy."
                For example, if bearish: "FCF yield only 2.1%. Debt-to-equity concerning at 2.3. Management diluting shareholders. Pass."
                """,
            ),
            (
                "human",
                """Based on the following data, create the investment signal as Michael Burry would:

                Analysis Data for {ticker}:
                {analysis_data}

                Return the trading signal in the following JSON format exactly:
                {{
                  "signal": "bullish" | "bearish" | "neutral",
                  "confidence": float between 0 and 100,
                  "reasoning": "string"
                }}
                """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    # Default fallback signal in case parsing fails
    def create_default_michael_burry_signal():
        return MichaelBurrySignal(signal="neutral", confidence=0.0, reasoning="Parsing error – defaulting to neutral")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=MichaelBurrySignal,
        agent_name="michael_burry_agent",
        default_factory=create_default_michael_burry_signal,
    )


class MichaelBurryCryptoAgent:
    def __init__(self, config: BinanceConfig, data_provider: BinanceDataProvider, executor: BinanceExecutor):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor
        self.logger = logging.getLogger(__name__)
        self.positions = {}

    def _calculate_rsi(self, series, period=14):
        delta = series.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.isnull().all() else 0

    def _calculate_macd(self, series, fast=12, slow=26, signal=9):
        exp1 = series.ewm(span=fast, adjust=False).mean()
        exp2 = series.ewm(span=slow, adjust=False).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        return macd.iloc[-1], macd_signal.iloc[-1]

    def consult_crypto(self, symbol, timeframe, model_used="rule-based"):
        df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=90)
        latest_close = df['close'].iloc[-1]
        avg_close = df['close'].rolling(window=90).mean().iloc[-1]
        rsi = self._calculate_rsi(df['close'])
        macd, macd_signal = self._calculate_macd(df['close'])

        if latest_close > avg_close * 1.10 and rsi > 70:
            signal = "BEARISH"
            action = "SHORT"
            confidence = 90
            reasoning = "Market is euphoric and overvalued. Burry would short the bubble."
            entry_condition = f"Open SHORT if price drops below {self.format_price(latest_close * 0.98)}"
            exit_condition = f"Cover SHORT if price rises above {self.format_price(avg_close)}"
        elif latest_close < avg_close * 0.90 and rsi < 30:
            signal = "BULLISH"
            action = "LONG"
            confidence = 85
            reasoning = "Deep value opportunity, market is oversold."
            entry_condition = f"Open LONG if price rises above {self.format_price(latest_close * 1.02)}"
            exit_condition = f"Close LONG if price drops below {self.format_price(avg_close)}"
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "No deep value or bubble detected."
            entry_condition = "Wait for extreme conditions."
            exit_condition = "N/A"

        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: {self.format_price(latest_close)}. "
            f"90-period MA: {self.format_price(avg_close)}. RSI(14): {rsi:.2f}. MACD: {self.format_price(macd)}, Signal: {self.format_price(macd_signal)}."
        )

        return {
            "agent": "michael_burry",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "Contrarian position size.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If price moves 5% against position, consider stop-loss.",
            "stop_loss_pct": 0.05,  # 5% stop-loss
            "take_profit_pct": 0.10,  # 10% take-profit
            "suggested_duration": "Hold until value is restored.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: {self.format_price(latest_close)} | 90-MA: {self.format_price(avg_close)} | RSI(14): {rsi:.2f} | MACD: {self.format_price(macd)}`\n"
                f"**Signal:** {signal}\n"
                f"**Action:** {action} | **Quantity:** 1\n"
                f"**Entry:** {entry_condition}\n"
                f"**Exit:** {exit_condition}\n"
                f"**Reversal:** If price moves 5% against position, consider stop-loss.\n"
                f"**Confidence:** {confidence}%\n"
                f"**Reasoning:** {reasoning}\n"
                f"**Model Used:** {model_used}\n"
            )
        }

    def is_contrarian_and_undervalued(self, symbol: str) -> bool:
        # Placeholder: Replace with real contrarian and value logic
        fundamentals = self.data_provider.get_crypto_fundamentals(symbol) if hasattr(self.data_provider, 'get_crypto_fundamentals') else {}
        sentiment_score = fundamentals.get('sentiment_score', 3)  # Example: low sentiment (contrarian)
        value_score = fundamentals.get('value_score', 8)  # Example: high value
        return sentiment_score < 4 and value_score > 7

    def run_strategy(self, symbol: str):
        price = self.data_provider.get_ticker_price(symbol)
        if self.is_contrarian_and_undervalued(symbol):
            quantity = self.executor.calculate_position_size(symbol, price) if hasattr(self.executor, 'calculate_position_size') else 1
            if quantity > 0:
                self.executor.create_order(symbol, 'BUY', 'MARKET', quantity)
        # Add sell/exit logic as needed

    def run_all_symbols(self):
        for symbol in self.config.trading_pairs:
            self.run_strategy(symbol)

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
