from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
import math
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig
import logging
from fastapi import HTTPException
import pandas as pd
import numpy as np


class BenGrahamSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def ben_graham_agent(state: AgentState):
    """
    Analyzes stocks using Benjamin Graham's classic value-investing principles:
    1. Earnings stability over multiple years.
    2. Solid financial strength (low debt, adequate liquidity).
    3. Discount to intrinsic value (e.g. Graham Number or net-net).
    4. Adequate margin of safety.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]

    analysis_data = {}
    graham_analysis = {}

    for ticker in tickers:
        progress.update_status("ben_graham_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, "Gathering financial line items")
        financial_line_items = search_line_items(ticker, ["earnings_per_share", "revenue", "net_income", "book_value_per_share", "total_assets", "total_liabilities", "current_assets", "current_liabilities", "dividends_and_other_cash_distributions", "outstanding_shares"], end_date, period="annual", limit=10)

        progress.update_status("ben_graham_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)

        # Perform sub-analyses
        progress.update_status("ben_graham_agent", ticker, "Analyzing earnings stability")
        earnings_analysis = analyze_earnings_stability(metrics, financial_line_items)

        progress.update_status("ben_graham_agent", ticker, "Analyzing financial strength")
        strength_analysis = analyze_financial_strength(financial_line_items)

        progress.update_status("ben_graham_agent", ticker, "Analyzing Graham valuation")
        valuation_analysis = analyze_valuation_graham(financial_line_items, market_cap)

        # Aggregate scoring
        total_score = earnings_analysis["score"] + strength_analysis["score"] + valuation_analysis["score"]
        max_possible_score = 15  # total possible from the three analysis functions

        # Map total_score to signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"

        analysis_data[ticker] = {"signal": signal, "score": total_score, "max_score": max_possible_score, "earnings_analysis": earnings_analysis, "strength_analysis": strength_analysis, "valuation_analysis": valuation_analysis}

        progress.update_status("ben_graham_agent", ticker, "Generating Ben Graham analysis")
        graham_output = generate_graham_output(
            ticker=ticker,
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )

        graham_analysis[ticker] = {"signal": graham_output.signal, "confidence": graham_output.confidence, "reasoning": graham_output.reasoning}

        progress.update_status("ben_graham_agent", ticker, "Done")

    # Wrap results in a single message for the chain
    message = HumanMessage(content=json.dumps(graham_analysis), name="ben_graham_agent")

    # Optionally display reasoning
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(graham_analysis, "Ben Graham Agent")

    # Store signals in the overall state
    state["data"]["analyst_signals"]["ben_graham_agent"] = graham_analysis

    return {"messages": [message], "data": state["data"]}


def analyze_earnings_stability(metrics: list, financial_line_items: list) -> dict:
    """
    Graham wants at least several years of consistently positive earnings (ideally 5+).
    We'll check:
    1. Number of years with positive EPS.
    2. Growth in EPS from first to last period.
    """
    score = 0
    details = []

    if not metrics or not financial_line_items:
        return {"score": score, "details": "Insufficient data for earnings stability analysis"}

    eps_vals = []
    for item in financial_line_items:
        if item.earnings_per_share is not None:
            eps_vals.append(item.earnings_per_share)

    if len(eps_vals) < 2:
        details.append("Not enough multi-year EPS data.")
        return {"score": score, "details": "; ".join(details)}

    # 1. Consistently positive EPS
    positive_eps_years = sum(1 for e in eps_vals if e > 0)
    total_eps_years = len(eps_vals)
    if positive_eps_years == total_eps_years:
        score += 3
        details.append("EPS was positive in all available periods.")
    elif positive_eps_years >= (total_eps_years * 0.8):
        score += 2
        details.append("EPS was positive in most periods.")
    else:
        details.append("EPS was negative in multiple periods.")

    # 2. EPS growth from earliest to latest
    if eps_vals[0] > eps_vals[-1]:
        score += 1
        details.append("EPS grew from earliest to latest period.")
    else:
        details.append("EPS did not grow from earliest to latest period.")

    return {"score": score, "details": "; ".join(details)}


def analyze_financial_strength(financial_line_items: list) -> dict:
    """
    Graham checks liquidity (current ratio >= 2), manageable debt,
    and dividend record (preferably some history of dividends).
    """
    score = 0
    details = []

    if not financial_line_items:
        return {"score": score, "details": "No data for financial strength analysis"}

    latest_item = financial_line_items[0]
    total_assets = latest_item.total_assets or 0
    total_liabilities = latest_item.total_liabilities or 0
    current_assets = latest_item.current_assets or 0
    current_liabilities = latest_item.current_liabilities or 0

    # 1. Current ratio
    if current_liabilities > 0:
        current_ratio = current_assets / current_liabilities
        if current_ratio >= 2.0:
            score += 2
            details.append(f"Current ratio = {current_ratio:.2f} (>=2.0: solid).")
        elif current_ratio >= 1.5:
            score += 1
            details.append(f"Current ratio = {current_ratio:.2f} (moderately strong).")
        else:
            details.append(f"Current ratio = {current_ratio:.2f} (<1.5: weaker liquidity).")
    else:
        details.append("Cannot compute current ratio (missing or zero current_liabilities).")

    # 2. Debt vs. Assets
    if total_assets > 0:
        debt_ratio = total_liabilities / total_assets
        if debt_ratio < 0.5:
            score += 2
            details.append(f"Debt ratio = {debt_ratio:.2f}, under 0.50 (conservative).")
        elif debt_ratio < 0.8:
            score += 1
            details.append(f"Debt ratio = {debt_ratio:.2f}, somewhat high but could be acceptable.")
        else:
            details.append(f"Debt ratio = {debt_ratio:.2f}, quite high by Graham standards.")
    else:
        details.append("Cannot compute debt ratio (missing total_assets).")

    # 3. Dividend track record
    div_periods = [item.dividends_and_other_cash_distributions for item in financial_line_items if item.dividends_and_other_cash_distributions is not None]
    if div_periods:
        # In many data feeds, dividend outflow is shown as a negative number
        # (money going out to shareholders). We'll consider any negative as 'paid a dividend'.
        div_paid_years = sum(1 for d in div_periods if d < 0)
        if div_paid_years > 0:
            # e.g. if at least half the periods had dividends
            if div_paid_years >= (len(div_periods) // 2 + 1):
                score += 1
                details.append("Company paid dividends in the majority of the reported years.")
            else:
                details.append("Company has some dividend payments, but not most years.")
        else:
            details.append("Company did not pay dividends in these periods.")
    else:
        details.append("No dividend data available to assess payout consistency.")

    return {"score": score, "details": "; ".join(details)}


def analyze_valuation_graham(financial_line_items: list, market_cap: float) -> dict:
    """
    Core Graham approach to valuation:
    1. Net-Net Check: (Current Assets - Total Liabilities) vs. Market Cap
    2. Graham Number: sqrt(22.5 * EPS * Book Value per Share)
    3. Compare per-share price to Graham Number => margin of safety
    """
    if not financial_line_items or not market_cap or market_cap <= 0:
        return {"score": 0, "details": "Insufficient data to perform valuation"}

    latest = financial_line_items[0]
    current_assets = latest.current_assets or 0
    total_liabilities = latest.total_liabilities or 0
    book_value_ps = latest.book_value_per_share or 0
    eps = latest.earnings_per_share or 0
    shares_outstanding = latest.outstanding_shares or 0

    details = []
    score = 0

    # 1. Net-Net Check
    #   NCAV = Current Assets - Total Liabilities
    #   If NCAV > Market Cap => historically a strong buy signal
    net_current_asset_value = current_assets - total_liabilities
    if net_current_asset_value > 0 and shares_outstanding > 0:
        net_current_asset_value_per_share = net_current_asset_value / shares_outstanding
        price_per_share = market_cap / shares_outstanding if shares_outstanding else 0

        details.append(f"Net Current Asset Value = {net_current_asset_value:,.2f}")
        details.append(f"NCAV Per Share = {net_current_asset_value_per_share:,.2f}")
        details.append(f"Price Per Share = {price_per_share:,.2f}")

        if net_current_asset_value > market_cap:
            score += 4  # Very strong Graham signal
            details.append("Net-Net: NCAV > Market Cap (classic Graham deep value).")
        else:
            # For partial net-net discount
            if net_current_asset_value_per_share >= (price_per_share * 0.67):
                score += 2
                details.append("NCAV Per Share >= 2/3 of Price Per Share (moderate net-net discount).")
    else:
        details.append("NCAV not exceeding market cap or insufficient data for net-net approach.")

    # 2. Graham Number
    #   GrahamNumber = sqrt(22.5 * EPS * BVPS).
    #   Compare the result to the current price_per_share
    #   If GrahamNumber >> price, indicates undervaluation
    graham_number = None
    if eps > 0 and book_value_ps > 0:
        graham_number = math.sqrt(22.5 * eps * book_value_ps)
        details.append(f"Graham Number = {graham_number:.2f}")
    else:
        details.append("Unable to compute Graham Number (EPS or Book Value missing/<=0).")

    # 3. Margin of Safety relative to Graham Number
    if graham_number and shares_outstanding > 0:
        current_price = market_cap / shares_outstanding
        if current_price > 0:
            margin_of_safety = (graham_number - current_price) / current_price
            details.append(f"Margin of Safety (Graham Number) = {margin_of_safety:.2%}")
            if margin_of_safety > 0.5:
                score += 3
                details.append("Price is well below Graham Number (>=50% margin).")
            elif margin_of_safety > 0.2:
                score += 1
                details.append("Some margin of safety relative to Graham Number.")
            else:
                details.append("Price close to or above Graham Number, low margin of safety.")
        else:
            details.append("Current price is zero or invalid; can't compute margin of safety.")
    # else: already appended details for missing graham_number

    return {"score": score, "details": "; ".join(details)}


def generate_graham_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BenGrahamSignal:
    """
    Generates an investment decision in the style of Benjamin Graham:
    - Value emphasis, margin of safety, net-nets, conservative balance sheet, stable earnings.
    - Return the result in a JSON structure: { signal, confidence, reasoning }.
    """

    template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """You are a Benjamin Graham AI agent, making investment decisions using his principles:
            1. Insist on a margin of safety by buying below intrinsic value (e.g., using Graham Number, net-net).
            2. Emphasize the company's financial strength (low leverage, ample current assets).
            3. Prefer stable earnings over multiple years.
            4. Consider dividend record for extra safety.
            5. Avoid speculative or high-growth assumptions; focus on proven metrics.
            
            When providing your reasoning, be thorough and specific by:
            1. Explaining the key valuation metrics that influenced your decision the most (Graham Number, NCAV, P/E, etc.)
            2. Highlighting the specific financial strength indicators (current ratio, debt levels, etc.)
            3. Referencing the stability or instability of earnings over time
            4. Providing quantitative evidence with precise numbers
            5. Comparing current metrics to Graham's specific thresholds (e.g., "Current ratio of 2.5 exceeds Graham's minimum of 2.0")
            6. Using Benjamin Graham's conservative, analytical voice and style in your explanation
            
            For example, if bullish: "The stock trades at a 35% discount to net current asset value, providing an ample margin of safety. The current ratio of 2.5 and debt-to-equity of 0.3 indicate strong financial position..."
            For example, if bearish: "Despite consistent earnings, the current price of $50 exceeds our calculated Graham Number of $35, offering no margin of safety. Additionally, the current ratio of only 1.2 falls below Graham's preferred 2.0 threshold..."
                        
            Return a rational recommendation: bullish, bearish, or neutral, with a confidence level (0-100) and thorough reasoning.
            """,
            ),
            (
                "human",
                """Based on the following analysis, create a Graham-style investment signal:

            Analysis Data for {ticker}:
            {analysis_data}

            Return JSON exactly in this format:
            {{
              "signal": "bullish" or "bearish" or "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """,
            ),
        ]
    )

    prompt = template.invoke({"analysis_data": json.dumps(analysis_data, indent=2), "ticker": ticker})

    def create_default_ben_graham_signal():
        return BenGrahamSignal(signal="neutral", confidence=0.0, reasoning="Error in generating analysis; defaulting to neutral.")

    return call_llm(
        prompt=prompt,
        model_name=model_name,
        model_provider=model_provider,
        pydantic_model=BenGrahamSignal,
        agent_name="ben_graham_agent",
        default_factory=create_default_ben_graham_signal,
    )


class BenGrahamCryptoAgent:
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

    def consult_crypto(self, symbol: str, timeframe: str = "1d", model_used: str = "rule-based") -> dict:
        df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=90)
        latest_close = df['close'].iloc[-1]
        avg_close = df['close'].rolling(window=90).mean().iloc[-1]
        rsi = self._calculate_rsi(df['close'])
        macd, macd_signal = self._calculate_macd(df['close'])

        def format_price(price):
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

        if latest_close < avg_close * 0.95:
            signal = "BULLISH"
            action = "LONG"
            confidence = 85
            reasoning = "Price is below its 90-period average, suggesting undervaluation. Ben Graham would see this as a value opportunity."
            entry_condition = f"Open LONG if price increases 10% to {format_price(latest_close * 1.1)}"
            exit_condition = f"Close LONG if price drops 10% to {format_price(latest_close * 0.9)}"
        elif latest_close > avg_close * 1.05:
            signal = "BEARISH"
            action = "SHORT"
            confidence = 85
            reasoning = "Price is above its 90-period average, suggesting overvaluation. Ben Graham would advise caution."
            entry_condition = f"Open SHORT if price drops 10% to {format_price(latest_close * 0.9)}"
            exit_condition = f"Close SHORT if price rises 10% to {format_price(latest_close * 1.1)}"
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "Price is near its 90-period average. Ben Graham would recommend patience."
            entry_condition = "Wait for price to move 5% above or below average."
            exit_condition = "N/A"

        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: {format_price(latest_close)}. "
            f"90-period MA: {format_price(avg_close)}. RSI(14): {rsi:.2f}. MACD: {macd:.2f}, Signal: {macd_signal:.2f}."
        )

        return {
            "agent": "ben_graham",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "Recommended position size in coins/contracts. Adjust based on your risk management.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If price moves 5% against position, consider stop-loss.",
            "suggested_duration": "Until price returns to or above/below the 90-period average.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"✅ **Ben Graham | {symbol} ({timeframe})**\n\n"
                f"**Signal:** `{signal}` | **Confidence:** `{confidence}%`\n"
                f"**Action:** `{action}` | **Quantity:** `1`\n\n"
                f"**Entry Condition**\n{entry_condition}\n\n"
                f"**Exit Condition**\n{exit_condition}\n\n"
                f"**Reversal Signal**\nIf price moves 5% against position, consider stop-loss.\n\n"
                f"**Suggested Duration**\nUntil price returns to or above/below the 90-period average.\n\n"
                f"**Technical Summary**\n"
                f"```\n"
                f"{symbol} on {timeframe} timeframe. Latest price: {format_price(latest_close)}. 90-period MA: {format_price(avg_close)}. RSI(14): {rsi:.2f}. MACD: {macd:.2f}, Signal: {macd_signal:.2f}.\n"
                f"```\n"
                f"**Reasoning**\n{reasoning}\n\n"
                f"**Model Used**\n{model_used}\n"
            )
        }
