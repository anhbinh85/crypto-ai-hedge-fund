from langchain_openai import ChatOpenAI
from src.graph.state import AgentState, show_agent_reasoning
from src.tools.api import get_financial_metrics, get_market_cap, search_line_items
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import BaseModel
import json
from typing_extensions import Literal
from src.utils.progress import progress
from src.utils.llm import call_llm
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor
from src.config.binance_config import BinanceConfig
import logging


class BillAckmanSignal(BaseModel):
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float
    reasoning: str


def bill_ackman_agent(state: AgentState):
    """
    Analyzes stocks using Bill Ackman's investing principles and LLM reasoning.
    Fetches multiple periods of data for a more robust long-term view.
    Incorporates brand/competitive advantage, activism potential, and other key factors.
    """
    data = state["data"]
    end_date = data["end_date"]
    tickers = data["tickers"]
    
    analysis_data = {}
    ackman_analysis = {}
    
    for ticker in tickers:
        progress.update_status("bill_ackman_agent", ticker, "Fetching financial metrics")
        metrics = get_financial_metrics(ticker, end_date, period="annual", limit=5)
        
        progress.update_status("bill_ackman_agent", ticker, "Gathering financial line items")
        # Request multiple periods of data (annual or TTM) for a more robust long-term view.
        financial_line_items = search_line_items(
            ticker,
            [
                "revenue",
                "operating_margin",
                "debt_to_equity",
                "free_cash_flow",
                "total_assets",
                "total_liabilities",
                "dividends_and_other_cash_distributions",
                "outstanding_shares",
                # Optional: intangible_assets if available
                # "intangible_assets"
            ],
            end_date,
            period="annual",
            limit=5
        )
        
        progress.update_status("bill_ackman_agent", ticker, "Getting market cap")
        market_cap = get_market_cap(ticker, end_date)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing business quality")
        quality_analysis = analyze_business_quality(metrics, financial_line_items)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing balance sheet and capital structure")
        balance_sheet_analysis = analyze_financial_discipline(metrics, financial_line_items)
        
        progress.update_status("bill_ackman_agent", ticker, "Analyzing activism potential")
        activism_analysis = analyze_activism_potential(financial_line_items)
        
        progress.update_status("bill_ackman_agent", ticker, "Calculating intrinsic value & margin of safety")
        valuation_analysis = analyze_valuation(financial_line_items, market_cap)
        
        # Combine partial scores or signals
        total_score = (
            quality_analysis["score"]
            + balance_sheet_analysis["score"]
            + activism_analysis["score"]
            + valuation_analysis["score"]
        )
        max_possible_score = 20  # Adjust weighting as desired (5 from each sub-analysis, for instance)
        
        # Generate a simple buy/hold/sell (bullish/neutral/bearish) signal
        if total_score >= 0.7 * max_possible_score:
            signal = "bullish"
        elif total_score <= 0.3 * max_possible_score:
            signal = "bearish"
        else:
            signal = "neutral"
        
        analysis_data[ticker] = {
            "signal": signal,
            "score": total_score,
            "max_score": max_possible_score,
            "quality_analysis": quality_analysis,
            "balance_sheet_analysis": balance_sheet_analysis,
            "activism_analysis": activism_analysis,
            "valuation_analysis": valuation_analysis
        }
        
        progress.update_status("bill_ackman_agent", ticker, "Generating Bill Ackman analysis")
        ackman_output = generate_ackman_output(
            ticker=ticker, 
            analysis_data=analysis_data,
            model_name=state["metadata"]["model_name"],
            model_provider=state["metadata"]["model_provider"],
        )
        
        ackman_analysis[ticker] = {
            "signal": ackman_output.signal,
            "confidence": ackman_output.confidence,
            "reasoning": ackman_output.reasoning
        }
        
        progress.update_status("bill_ackman_agent", ticker, "Done")
    
    # Wrap results in a single message for the chain
    message = HumanMessage(
        content=json.dumps(ackman_analysis),
        name="bill_ackman_agent"
    )
    
    # Show reasoning if requested
    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(ackman_analysis, "Bill Ackman Agent")
    
    # Add signals to the overall state
    state["data"]["analyst_signals"]["bill_ackman_agent"] = ackman_analysis

    return {
        "messages": [message],
        "data": state["data"]
    }


def analyze_business_quality(metrics: list, financial_line_items: list) -> dict:
    """
    Analyze whether the company has a high-quality business with stable or growing cash flows,
    durable competitive advantages (moats), and potential for long-term growth.
    Also tries to infer brand strength if intangible_assets data is present (optional).
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze business quality"
        }
    
    # 1. Multi-period revenue growth analysis
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    if len(revenues) >= 2:
        initial, final = revenues[-1], revenues[0]
        if initial and final and final > initial:
            growth_rate = (final - initial) / abs(initial)
            if growth_rate > 0.5:  # e.g., 50% cumulative growth
                score += 2
                details.append(f"Revenue grew by {(growth_rate*100):.1f}% over the full period (strong growth).")
            else:
                score += 1
                details.append(f"Revenue growth is positive but under 50% cumulatively ({(growth_rate*100):.1f}%).")
        else:
            details.append("Revenue did not grow significantly or data insufficient.")
    else:
        details.append("Not enough revenue data for multi-period trend.")
    
    # 2. Operating margin and free cash flow consistency
    fcf_vals = [item.free_cash_flow for item in financial_line_items if item.free_cash_flow is not None]
    op_margin_vals = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if op_margin_vals:
        above_15 = sum(1 for m in op_margin_vals if m > 0.15)
        if above_15 >= (len(op_margin_vals) // 2 + 1):
            score += 2
            details.append("Operating margins have often exceeded 15% (indicates good profitability).")
        else:
            details.append("Operating margin not consistently above 15%.")
    else:
        details.append("No operating margin data across periods.")
    
    if fcf_vals:
        positive_fcf_count = sum(1 for f in fcf_vals if f > 0)
        if positive_fcf_count >= (len(fcf_vals) // 2 + 1):
            score += 1
            details.append("Majority of periods show positive free cash flow.")
        else:
            details.append("Free cash flow not consistently positive.")
    else:
        details.append("No free cash flow data across periods.")
    
    # 3. Return on Equity (ROE) check from the latest metrics
    latest_metrics = metrics[0]
    if latest_metrics.return_on_equity and latest_metrics.return_on_equity > 0.15:
        score += 2
        details.append(f"High ROE of {latest_metrics.return_on_equity:.1%}, indicating a competitive advantage.")
    elif latest_metrics.return_on_equity:
        details.append(f"ROE of {latest_metrics.return_on_equity:.1%} is moderate.")
    else:
        details.append("ROE data not available.")
    
    # 4. (Optional) Brand Intangible (if intangible_assets are fetched)
    # intangible_vals = [item.intangible_assets for item in financial_line_items if item.intangible_assets]
    # if intangible_vals and sum(intangible_vals) > 0:
    #     details.append("Significant intangible assets may indicate brand value or proprietary tech.")
    #     score += 1
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_financial_discipline(metrics: list, financial_line_items: list) -> dict:
    """
    Evaluate the company's balance sheet over multiple periods:
    - Debt ratio trends
    - Capital returns to shareholders over time (dividends, buybacks)
    """
    score = 0
    details = []
    
    if not metrics or not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data to analyze financial discipline"
        }
    
    # 1. Multi-period debt ratio or debt_to_equity
    debt_to_equity_vals = [item.debt_to_equity for item in financial_line_items if item.debt_to_equity is not None]
    if debt_to_equity_vals:
        below_one_count = sum(1 for d in debt_to_equity_vals if d < 1.0)
        if below_one_count >= (len(debt_to_equity_vals) // 2 + 1):
            score += 2
            details.append("Debt-to-equity < 1.0 for the majority of periods (reasonable leverage).")
        else:
            details.append("Debt-to-equity >= 1.0 in many periods (could be high leverage).")
    else:
        # Fallback to total_liabilities / total_assets
        liab_to_assets = []
        for item in financial_line_items:
            if item.total_liabilities and item.total_assets and item.total_assets > 0:
                liab_to_assets.append(item.total_liabilities / item.total_assets)
        
        if liab_to_assets:
            below_50pct_count = sum(1 for ratio in liab_to_assets if ratio < 0.5)
            if below_50pct_count >= (len(liab_to_assets) // 2 + 1):
                score += 2
                details.append("Liabilities-to-assets < 50% for majority of periods.")
            else:
                details.append("Liabilities-to-assets >= 50% in many periods.")
        else:
            details.append("No consistent leverage ratio data available.")
    
    # 2. Capital allocation approach (dividends + share counts)
    dividends_list = [
        item.dividends_and_other_cash_distributions
        for item in financial_line_items
        if item.dividends_and_other_cash_distributions is not None
    ]
    if dividends_list:
        paying_dividends_count = sum(1 for d in dividends_list if d < 0)
        if paying_dividends_count >= (len(dividends_list) // 2 + 1):
            score += 1
            details.append("Company has a history of returning capital to shareholders (dividends).")
        else:
            details.append("Dividends not consistently paid or no data on distributions.")
    else:
        details.append("No dividend data found across periods.")
    
    # Check for decreasing share count (simple approach)
    shares = [item.outstanding_shares for item in financial_line_items if item.outstanding_shares is not None]
    if len(shares) >= 2:
        # For buybacks, the newest count should be less than the oldest count
        if shares[0] < shares[-1]:
            score += 1
            details.append("Outstanding shares have decreased over time (possible buybacks).")
        else:
            details.append("Outstanding shares have not decreased over the available periods.")
    else:
        details.append("No multi-period share count data to assess buybacks.")
    
    return {
        "score": score,
        "details": "; ".join(details)
    }


def analyze_activism_potential(financial_line_items: list) -> dict:
    """
    Bill Ackman often engages in activism if a company has a decent brand or moat
    but is underperforming operationally.
    
    We'll do a simplified approach:
    - Look for positive revenue trends but subpar margins
    - That may indicate 'activism upside' if operational improvements could unlock value.
    """
    if not financial_line_items:
        return {
            "score": 0,
            "details": "Insufficient data for activism potential"
        }
    
    # Check revenue growth vs. operating margin
    revenues = [item.revenue for item in financial_line_items if item.revenue is not None]
    op_margins = [item.operating_margin for item in financial_line_items if item.operating_margin is not None]
    
    if len(revenues) < 2 or not op_margins:
        return {
            "score": 0,
            "details": "Not enough data to assess activism potential (need multi-year revenue + margins)."
        }
    
    initial, final = revenues[-1], revenues[0]
    revenue_growth = (final - initial) / abs(initial) if initial else 0
    avg_margin = sum(op_margins) / len(op_margins)
    
    score = 0
    details = []
    
    # Suppose if there's decent revenue growth but margins are below 10%, Ackman might see activism potential.
    if revenue_growth > 0.15 and avg_margin < 0.10:
        score += 2
        details.append(
            f"Revenue growth is healthy (~{revenue_growth*100:.1f}%), but margins are low (avg {avg_margin*100:.1f}%). "
            "Activism could unlock margin improvements."
        )
    else:
        details.append("No clear sign of activism opportunity (either margins are already decent or growth is weak).")
    
    return {"score": score, "details": "; ".join(details)}


def analyze_valuation(financial_line_items: list, market_cap: float) -> dict:
    """
    Ackman invests in companies trading at a discount to intrinsic value.
    Uses a simplified DCF with FCF as a proxy, plus margin of safety analysis.
    """
    if not financial_line_items or market_cap is None:
        return {
            "score": 0,
            "details": "Insufficient data to perform valuation"
        }
    
    # Since financial_line_items are in descending order (newest first),
    # the most recent period is the first element
    latest = financial_line_items[0]
    fcf = latest.free_cash_flow if latest.free_cash_flow else 0
    
    if fcf <= 0:
        return {
            "score": 0,
            "details": f"No positive FCF for valuation; FCF = {fcf}",
            "intrinsic_value": None
        }
    
    # Basic DCF assumptions
    growth_rate = 0.06
    discount_rate = 0.10
    terminal_multiple = 15
    projection_years = 5
    
    present_value = 0
    for year in range(1, projection_years + 1):
        future_fcf = fcf * (1 + growth_rate) ** year
        pv = future_fcf / ((1 + discount_rate) ** year)
        present_value += pv
    
    # Terminal Value
    terminal_value = (
        fcf * (1 + growth_rate) ** projection_years * terminal_multiple
    ) / ((1 + discount_rate) ** projection_years)
    
    intrinsic_value = present_value + terminal_value
    margin_of_safety = (intrinsic_value - market_cap) / market_cap
    
    score = 0
    # Simple scoring
    if margin_of_safety > 0.3:
        score += 3
    elif margin_of_safety > 0.1:
        score += 1
    
    details = [
        f"Calculated intrinsic value: ~{intrinsic_value:,.2f}",
        f"Market cap: ~{market_cap:,.2f}",
        f"Margin of safety: {margin_of_safety:.2%}"
    ]
    
    return {
        "score": score,
        "details": "; ".join(details),
        "intrinsic_value": intrinsic_value,
        "margin_of_safety": margin_of_safety
    }


def generate_ackman_output(
    ticker: str,
    analysis_data: dict[str, any],
    model_name: str,
    model_provider: str,
) -> BillAckmanSignal:
    """
    Generates investment decisions in the style of Bill Ackman.
    Includes more explicit references to brand strength, activism potential, 
    catalysts, and management changes in the system prompt.
    """
    template = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a Bill Ackman AI agent, making investment decisions using his principles:

            1. Seek high-quality businesses with durable competitive advantages (moats), often in well-known consumer or service brands.
            2. Prioritize consistent free cash flow and growth potential over the long term.
            3. Advocate for strong financial discipline (reasonable leverage, efficient capital allocation).
            4. Valuation matters: target intrinsic value with a margin of safety.
            5. Consider activism where management or operational improvements can unlock substantial upside.
            6. Concentrate on a few high-conviction investments.

            In your reasoning:
            - Emphasize brand strength, moat, or unique market positioning.
            - Review free cash flow generation and margin trends as key signals.
            - Analyze leverage, share buybacks, and dividends as capital discipline metrics.
            - Provide a valuation assessment with numerical backup (DCF, multiples, etc.).
            - Identify any catalysts for activism or value creation (e.g., cost cuts, better capital allocation).
            - Use a confident, analytic, and sometimes confrontational tone when discussing weaknesses or opportunities.

            Return your final recommendation (signal: bullish, neutral, or bearish) with a 0-100 confidence and a thorough reasoning section.
            """
        ),
        (
            "human",
            """Based on the following analysis, create an Ackman-style investment signal.

            Analysis Data for {ticker}:
            {analysis_data}

            Return your output in strictly valid JSON:
            {{
              "signal": "bullish" | "bearish" | "neutral",
              "confidence": float (0-100),
              "reasoning": "string"
            }}
            """
        )
    ])

    prompt = template.invoke({
        "analysis_data": json.dumps(analysis_data, indent=2),
        "ticker": ticker
    })

    def create_default_bill_ackman_signal():
        return BillAckmanSignal(
            signal="neutral",
            confidence=0.0,
            reasoning="Error in analysis, defaulting to neutral"
        )

    return call_llm(
        prompt=prompt, 
        model_name=model_name, 
        model_provider=model_provider, 
        pydantic_model=BillAckmanSignal, 
        agent_name="bill_ackman_agent", 
        default_factory=create_default_bill_ackman_signal,
    )


class BillAckmanCryptoAgent:
    def __init__(self, config, data_provider, executor):
        self.config = config
        self.data_provider = data_provider
        self.executor = executor

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
        df = self.data_provider.get_historical_klines(symbol, interval=timeframe, limit=60)
        latest_close = df['close'].iloc[-1]
        avg_close = df['close'].rolling(window=60).mean().iloc[-1]
        rsi = self._calculate_rsi(df['close'])
        macd, macd_signal = self._calculate_macd(df['close'])

        if rsi > 70 and latest_close > avg_close:
            signal = "BULLISH"
            action = "LONG"
            confidence = 80
            reasoning = "Strong momentum and overbought, Ackman would take a bold position."
            entry_condition = f"Open LONG if price breaks above {self.format_price(latest_close * 1.02)}"
            exit_condition = f"Close LONG if price drops below {self.format_price(avg_close)}"
        elif rsi < 30 and latest_close < avg_close:
            signal = "BEARISH"
            action = "SHORT"
            confidence = 80
            reasoning = "Oversold and below average, Ackman would short and push for change."
            entry_condition = f"Open SHORT if price drops below {self.format_price(latest_close * 0.98)}"
            exit_condition = f"Close SHORT if price rises above {self.format_price(avg_close)}"
        else:
            signal = "NEUTRAL"
            action = "HOLD"
            confidence = 60
            reasoning = "No clear activist opportunity."
            entry_condition = "Wait for a catalyst."
            exit_condition = "N/A"

        data_summary = (
            f"{symbol} on {timeframe} timeframe. Latest price: {self.format_price(latest_close)}. "
            f"60-period MA: {self.format_price(avg_close)}. RSI(14): {rsi:.2f}. MACD: {macd:.2f}, Signal: {macd_signal:.2f}."
        )

        return {
            "agent": "bill_ackman",
            "signal": signal,
            "confidence": confidence,
            "reasoning": reasoning,
            "action": action,
            "quantity": 1 if action in ["LONG", "SHORT"] else 0,
            "quantity_explanation": "Aggressive position size for activist moves.",
            "entry_condition": entry_condition,
            "exit_condition": exit_condition,
            "reversal_signal": "If price moves 5% against position, consider stop-loss.",
            "stop_loss_pct": 0.05,  # 5% stop-loss
            "take_profit_pct": 0.10,  # 10% take-profit
            "suggested_duration": "Until catalyst plays out.",
            "model_used": model_used,
            "data_summary": data_summary,
            "discord_message": (
                f"**{symbol} ({timeframe})**\n"
                f"**Technical Summary:**\n"
                f"`Price: {self.format_price(latest_close)} | 60-MA: {self.format_price(avg_close)} | RSI(14): {rsi:.2f} | MACD: {macd:.2f}`\n"
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

    def has_catalyst_and_risk_control(self, symbol: str) -> bool:
        # Placeholder: Replace with real catalyst and risk logic
        fundamentals = self.data_provider.get_crypto_fundamentals(symbol) if hasattr(self.data_provider, 'get_crypto_fundamentals') else {}
        catalyst_score = fundamentals.get('catalyst_score', 8)  # Example: use a default value
        risk_score = fundamentals.get('risk_score', 7)  # Example: use a default value
        return catalyst_score > 7 and risk_score > 6

    def run_strategy(self, symbol: str):
        price = self.data_provider.get_ticker_price(symbol)
        if self.has_catalyst_and_risk_control(symbol):
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
