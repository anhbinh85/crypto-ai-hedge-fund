from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.agents.ben_graham import BenGrahamCryptoAgent
from src.agents.bill_ackman import BillAckmanCryptoAgent
from src.agents.binance_strategy import BinanceStrategy
from src.agents.mean_reversion_agent import MeanReversionAgent
from src.agents.michael_burry import MichaelBurryCryptoAgent
from src.agents.sentiment import SentimentAgent
from src.agents.trend_following_agent import TrendFollowingAgent
from src.agents.warren_buffett import WarrenBuffettCryptoAgent
from src.agents.cathie_wood import CathieWoodCryptoAgent
from src.agents.charlie_munger import CharlieMungerCryptoAgent
from src.agents.peter_lynch import PeterLynchCryptoAgent
from src.agents.stanley_druckenmiller import StanleyDruckenmillerCryptoAgent
from src.agents.phil_fisher import PhilFisherCryptoAgent
from src.agents.valuation import ValuationAgent
from src.agents.fundamentals import FundamentalsAgent
from src.agents.technicals import TechnicalsAgent
from src.agents.risk_manager import RiskManagerAgent
from src.agents.portfolio_manager import PortfolioManagerAgent

from src.config.binance_config import BinanceConfig
from src.data.binance_provider import BinanceDataProvider
from src.tools.binance_executor import BinanceExecutor

app = FastAPI()

AGENT_REGISTRY = {
    "ben_graham": BenGrahamCryptoAgent,
    "bill_ackman": BillAckmanCryptoAgent,
    "binance_strategy": BinanceStrategy,
    "mean_reversion": MeanReversionAgent,
    "michael_burry": MichaelBurryCryptoAgent,
    "sentiment": SentimentAgent,
    "trend_following": TrendFollowingAgent,
    "warren_buffett": WarrenBuffettCryptoAgent,
    "cathie_wood": CathieWoodCryptoAgent,
    "charlie_munger": CharlieMungerCryptoAgent,
    "peter_lynch": PeterLynchCryptoAgent,
    "stanley_druckenmiller": StanleyDruckenmillerCryptoAgent,
    "phil_fisher": PhilFisherCryptoAgent,
    "valuation": ValuationAgent,
    "fundamentals": FundamentalsAgent,
    "technicals": TechnicalsAgent,
    "risk_manager": RiskManagerAgent,
    "portfolio_manager": PortfolioManagerAgent,
}

BINANCE_INTERVALS = [
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M"
]

class HedgeFundRequest(BaseModel):
    ticker: str
    timeframe: str
    selected_agents: list

@app.post("/hedge_fund_consult")
def hedge_fund_consult(request: HedgeFundRequest):
    ticker = request.ticker.upper()
    timeframe = request.timeframe
    selected_agents = request.selected_agents

    # Validate agents
    for agent_name in selected_agents:
        if agent_name not in AGENT_REGISTRY:
            return JSONResponse(
                status_code=400,
                content={
                    "error": f"Invalid agent '{agent_name}'. Allowed: {', '.join(AGENT_REGISTRY.keys())}"
                }
            )

    # Validate timeframe
    if timeframe not in BINANCE_INTERVALS:
        return JSONResponse(
            status_code=400,
            content={
                "error": f"Invalid timeframe '{timeframe}'. Allowed: {', '.join(BINANCE_INTERVALS)}"
            }
        )

    config = BinanceConfig(api_key="", api_secret="")
    data_provider = BinanceDataProvider(config.api_key, config.api_secret)
    executor = BinanceExecutor(config.api_key, config.api_secret)

    # 1. Collect signals from selected agents
    signals = []
    for agent_name in selected_agents:
        agent_class = AGENT_REGISTRY[agent_name]
        agent = agent_class(config, data_provider, executor)
        signals.append(agent.consult_crypto(ticker, timeframe))

    # 2. Add core signals
    for core_agent in ["valuation", "sentiment", "fundamentals", "technicals"]:
        agent_class = AGENT_REGISTRY[core_agent]
        agent = agent_class(config, data_provider, executor)
        signals.append(agent.consult_crypto(ticker, timeframe))

    # 3. Risk Manager
    risk_manager = AGENT_REGISTRY["risk_manager"](config, data_provider, executor)
    risk_report = risk_manager.evaluate(signals)

    # 4. Portfolio Manager
    portfolio_manager = AGENT_REGISTRY["portfolio_manager"](config, data_provider, executor)
    final_decision = portfolio_manager.make_decision(risk_report)

    return final_decision

@app.get("/agents")
def list_agents():
    return {"available_agents": list(AGENT_REGISTRY.keys())}