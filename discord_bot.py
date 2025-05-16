import os
import requests
import nextcord
from nextcord.ext import commands
from dotenv import load_dotenv
from src.backtest.backtest_engine import run_backtest, save_backtest_results
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
from src.agents.techs_analysis import get_techs_embed
from src.data.binance_provider import BinanceDataProvider
import io
from datetime import datetime
import logging

# Load environment variables
load_dotenv()
TOKEN = os.getenv("DISCORD_BOT_TOKEN")
ALLOWED_GUILD_ID = int(os.getenv("ALLOWED_GUILD_ID"))  # Read and convert to int

intents = nextcord.Intents.default()
bot = commands.Bot(command_prefix="!", intents=intents)

FASTAPI_URL = "http://127.0.0.1:8000/hedge_fund_consult"

# Define AGENT_REGISTRY at the module level for reuse in autocomplete
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
}

# Agent descriptions for /agents command
AGENT_DESCRIPTIONS = {
    "ben_graham": "Value investing legend. Focuses on undervalued assets and margin of safety.",
    "bill_ackman": "Activist investor. Looks for catalysts and deep value opportunities.",
    "binance_strategy": "Technical strategy using RSI and MACD on Binance data.",
    "mean_reversion": "Buys low, sells high. Expects prices to revert to the mean.",
    "michael_burry": "Contrarian investor. Famous for spotting bubbles and market mispricings.",
    "sentiment": "Analyzes market sentiment from news and social data.",
    "trend_following": "Follows price trends. Rides momentum up or down.",
    "warren_buffett": "Long-term value investor. Focuses on quality businesses and compounding.",
    "cathie_wood": "Growth and innovation investor. Focuses on disruptive tech.",
    "charlie_munger": "Buffett's partner. Focuses on mental models and rational investing.",
    "peter_lynch": "Growth at a reasonable price. Invests in what he understands.",
    "stanley_druckenmiller": "Macro trader. Focuses on big economic trends and timing.",
    "phil_fisher": "Scuttlebutt and qualitative analysis. Looks for outstanding companies.",
    "valuation": "Estimates fair value using financial models.",
    "fundamentals": "Analyzes company financials and health.",
    "technicals": "Uses technical indicators and price action."
}

@bot.event
async def on_ready():
    print(f"Bot is ready! Logged in as {bot.user}")

@bot.slash_command(
    description="Get crypto trading advice from multiple AI agents"
    #,guild_ids=[ALLOWED_GUILD_ID]
)
async def consult(
    interaction: nextcord.Interaction,
    ticker: str = nextcord.SlashOption(description="Ticker symbol (e.g. BTCUSDT, ETHUSDT)"),
    timeframe: str = nextcord.SlashOption(description="Timeframe (e.g. 1m, 1h, 1d)", required=False, default="1h"),
    agents: str = nextcord.SlashOption(
        description="Comma-separated agent names (e.g. ben_graham,bill_ackman)",
        required=False,
        default="ben_graham"
    )
):
    await interaction.response.defer()

    # Parse agent names into a list
    selected_agents = [a.strip() for a in agents.split(",") if a.strip()]

    payload = {
        "ticker": ticker,
        "timeframe": timeframe,
        "selected_agents": selected_agents
    }

    try:
        response = requests.post(FASTAPI_URL, json=payload, timeout=20)
        if response.status_code == 200:
            data = response.json()
            final_action = data.get("final_action", "N/A")
            reason = data.get("reason", "N/A")
            risk_score = data.get("risk_score", "N/A")
            agent_signals = data.get("agent_signals", [])

            # 1. Technical indicators (from the first agent or a dedicated field)
            tech_info = ""
            for signal in agent_signals:
                if "data_summary" in signal and signal["data_summary"]:
                    tech_info = f"📊 Technical Indicators\n{signal['data_summary']}\n"
                    break

            # 2. Build agent blocks
            agent_blocks = []
            action_counts = {"LONG": 0, "SHORT": 0, "HOLD": 0}
            bullish_count = 0
            bearish_count = 0
            neutral_count = 0
            for signal in agent_signals:
                agent = signal.get("agent", "Agent").replace("_", " ").title()
                sig = signal.get("signal", "N/A")
                act = signal.get("action", "N/A")
                conf = signal.get("confidence", "N/A")
                reasoning = signal.get("reasoning", "N/A")
                entry = signal.get("entry_condition", "N/A")
                exit_cond = signal.get("exit_condition", None)
                reversal = signal.get("reversal_signal", None)
                duration = signal.get("suggested_duration", None)

                # Count actions and signals
                if act in action_counts:
                    action_counts[act] += 1
                if str(sig).upper() == "BULLISH":
                    bullish_count += 1
                elif str(sig).upper() == "BEARISH":
                    bearish_count += 1
                else:
                    neutral_count += 1

                # Emoji for agent
                agent_emoji = {
                    "Warren Buffet": "🧑‍💼",
                    "Valuation": "💰",
                    "Sentiment": "🗣️",
                    "Fundamentals": "📈",
                    "Technicals": "🛠️",
                    "Ben Graham": "🤖",
                    "Bill Ackman": "🦈",
                    "Michael Burry": "🦉",
                    "Cathie Wood": "🦄",
                    "Charlie Munger": "🧓",
                    "Peter Lynch": "🧢",
                    "Stanley Druckenmiller": "🐂",
                    "Phil Fisher": "🔬"
                }.get(agent, "🤖")

                # Emoji for signal
                if str(sig).upper() == "BULLISH":
                    sig_emoji = "😃"
                elif str(sig).upper() == "BEARISH":
                    sig_emoji = "😢"
                else:
                    sig_emoji = "😐"

                block = (
                    f"{agent_emoji} **{agent}**\n"
                    f"Signal: {sig} | Action: {act} | Confidence: {conf}%\n"
                    f"Reasoning: {sig_emoji} {reasoning}\n"
                    f"Entry: {entry}"
                )
                if exit_cond:
                    block += f"\nExit: {exit_cond}"
                if reversal:
                    block += f"\nReversal Signal: {reversal}"
                if duration:
                    block += f"\nSuggested Duration: {duration}"
                agent_blocks.append(block)

            # 3. Final decision block (dynamic)
            total_agents = sum(action_counts.values())
            majority_action = max(action_counts, key=action_counts.get)
            majority_count = action_counts[majority_action]
            # Reason and conclusion
            if majority_action == "LONG":
                reason = "Majority agent action is LONG."
                conclusion = (
                    f"> Most agents recommend going LONG. Fundamentals, technicals, sentiment, and several famous investors are bullish (📈🐂). "
                    f"A few value-oriented agents are cautious or bearish."
                )
                action_line = "Action: 🟢 Consider a LONG position, but monitor risk."
            elif majority_action == "SHORT":
                reason = "Majority agent action is SHORT."
                conclusion = (
                    f"> Most agents recommend going SHORT. Several agents see overvaluation or negative momentum (😢). "
                    f"Some agents are neutral or bullish, so use caution."
                )
                action_line = "Action: 🔴 Consider a SHORT position, but monitor risk."
            else:
                reason = "Majority agent action is HOLD."
                conclusion = (
                    f"> Most agents recommend waiting. No strong buy or sell signal detected. "
                    f"Monitor for a clearer opportunity."
                )
                action_line = "Action: ⏸️ Hold and monitor for a clearer opportunity."

            # Risk assessment (unchanged)
            if isinstance(risk_score, (int, float)) or (isinstance(risk_score, str) and risk_score.lstrip('-').isdigit()):
                try:
                    score_val = float(risk_score)
                except Exception:
                    score_val = None
            else:
                score_val = None
            if score_val is not None:
                if score_val < 0:
                    risk_emoji = "🔴"
                    risk_msg = "Cautious, not a good time to trade"
                elif score_val == 0:
                    risk_emoji = "🟡"
                    risk_msg = "Neutral risk environment"
                else:
                    risk_emoji = "🟢"
                    risk_msg = "Trading opportunity with calculated risk"
            else:
                risk_emoji = "❓"
                risk_msg = "Unknown risk"

            final_block = (
                f"{risk_emoji} **Final Decision: {majority_action}**\n"
                f"Reason: {reason}\n"
                f"Risk Assessment: {risk_emoji} {risk_msg} (Risk Score: {risk_score})\n"
                f"Conclusion:\n{conclusion}\n"
                f"{action_line}"
            )

            # 4. Compose the full message
            header = f"🟢 {ticker.upper()} Market Analysis ({timeframe})\n"
            full_message = (
                f"{header}"
                f"{tech_info}\n"
                + "\n".join(agent_blocks)
                + "\n"
                + final_block
            )

            embed = nextcord.Embed(
                description=full_message,
                color=nextcord.Color.green() if final_action in ["BUY", "LONG"] else nextcord.Color.red()
            )
            embed.set_footer(text="Powered by CryptoConsult AI")
        else:
            embed = nextcord.Embed(
                title="❌ CryptoConsult Error",
                description=f"Backend error: {response.status_code} – {response.text}",
                color=nextcord.Color.red()
            )
    except Exception as e:
        embed = nextcord.Embed(
            title="❌ CryptoConsult Error",
            description=f"Error contacting backend: {e}",
            color=nextcord.Color.red()
        )

    # Send the embed as a DM to the user, and send an ephemeral confirmation in the channel
    user = interaction.user
    try:
        await user.send(embed=embed)
        await interaction.followup.send("✅ Check your DMs for the analysis!", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"❌ Could not send DM: {e}", ephemeral=True)

# Autocomplete callback for agent_name
async def agent_name_autocomplete(interaction: nextcord.Interaction, current: str):
    current = current.lower()
    return [
        {"name": key.replace('_', ' ').title(), "value": key}
        for key in AGENT_REGISTRY.keys() if current in key.lower()
    ][:25]

@bot.slash_command(name="agents", description="List all available agents and their descriptions")
async def agents(interaction: nextcord.Interaction):
    embed = nextcord.Embed(title="Available Agents", color=nextcord.Color.blue())
    for agent_id, desc in AGENT_DESCRIPTIONS.items():
        embed.add_field(name=f"`{agent_id}`", value=desc, inline=False)
    await interaction.response.send_message(embed=embed, ephemeral=True)

@bot.slash_command(name="backtest", description="Run a backtest simulation")
async def backtest(
    interaction: nextcord.Interaction,
    ticker: str = nextcord.SlashOption(description="Trading pair (e.g., BTCUSDT)"),
    agent_name: str = nextcord.SlashOption(
        description="Agent id (see /agents for list)"
    ),
    candle: str = nextcord.SlashOption(description="Candle interval (e.g., 1m, 5m, 1h, 1d)", default="1h"),
    observations: int = nextcord.SlashOption(description="Number of candles to backtest (default: 100)", default=100),
    leverage: float = nextcord.SlashOption(description="Leverage (default: 1.0)", default=1.0)
):
    await interaction.response.defer()
    # Send confirmation immediately
    await interaction.followup.send("✅ Backtest started! Results will be sent to your DM.", ephemeral=True)
    try:
        agent = get_agent_instance(agent_name)
        if agent is None:
            await interaction.user.send(f"❌ Agent '{agent_name}' not found!")
            return
        result, _ = run_backtest(
            ticker=ticker,
            timeframe=candle,
            agents=agent_name,
            observations=observations,
            leverage=leverage
        )
        if result is None:
            await interaction.user.send("❌ Failed to run backtest. Please check your parameters.")
            return
        action_counts = {}
        for action in result['actions_log']:
            action_counts[action['action']] = action_counts.get(action['action'], 0) + 1
        # Only show the summary in the embed
        agent_display = agent_name.replace('_', ' ').title()
        stats_block = (
            f"**Total Return:** `{result['total_return']*100:.2f}%`\n"
            f"**Max Drawdown:** `{result['max_drawdown']*100:.2f}%`\n"
            f"**Win Rate:** `{result['win_rate']*100:.2f}%`\n"
            f"**Number of Trades:** `{result['num_trades']}`\n"
            f"**Final Balance:** `${result['final_balance']:.2f}`\n"
            f"**Best Trade:** `{result['best_trade']*100:.2f}%`\n"
            f"**Worst Trade:** `{result['worst_trade']*100:.2f}%`\n"
        )
        action_summary = " | ".join(f"{k}: {v}" for k, v in action_counts.items())
        embed = nextcord.Embed(
            title=f"Backtest Results for {ticker.upper()} ({candle}) - {agent_display}",
            description=(
                f"**{agent_display}** agent returned **{result['total_return']*100:.2f}%** with a **{result['win_rate']*100:.2f}% win rate** over {observations} candles ({leverage}x leverage).\n\n"
                + stats_block
                + f"**Agent Action Summary:** `{action_summary}`"
            ),
            color=nextcord.Color.green() if result['total_return'] > 0 else nextcord.Color.red()
        )
        embed.set_footer(text="Powered by CryptoConsult AI Backtest")
        await interaction.user.send(embed=embed)
        obs_log = result.get('observation_log', [])
        if obs_log:
            output = io.StringIO()
            fieldnames = list(obs_log[0].keys())
            if 'notes' not in fieldnames:
                fieldnames.append('notes')
            import csv
            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()
            for row in obs_log:
                if 'notes' not in row:
                    row['notes'] = ''
                writer.writerow(row)
            output.seek(0)
            csv_file = nextcord.File(fp=output, filename=f"backtest_{ticker}_{candle}_{agent_name}.csv")
            await interaction.user.send("📊 Detailed backtest CSV attached:", file=csv_file)
    except Exception as e:
        await interaction.user.send(f"❌ Error running backtest: {str(e)}")

def get_agent_instance(agent_name):
    if agent_name not in AGENT_REGISTRY:
        return None
    from src.config.binance_config import BinanceConfig
    from src.data.binance_provider import BinanceDataProvider
    from src.tools.binance_executor import BinanceExecutor
    config = BinanceConfig(api_key="", api_secret="")
    data_provider = BinanceDataProvider(config.api_key, config.api_secret)
    executor = BinanceExecutor(config.api_key, config.api_secret)
    return AGENT_REGISTRY[agent_name](config, data_provider, executor)

@bot.slash_command(name="techs", description="Show technical analysis for a ticker and candle interval")
async def techs(
    interaction: nextcord.Interaction,
    ticker: str = nextcord.SlashOption(description="Ticker symbol (e.g. BTCUSDT)"),
    candle: str = nextcord.SlashOption(description="Candle interval (e.g. 1m, 5m, 15m, 1h, 1d)")
):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"/techs called with ticker={ticker}, candle={candle}, user={interaction.user}")
    await interaction.response.defer(ephemeral=True)
    try:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        provider = BinanceDataProvider(api_key, api_secret)
        result = get_techs_embed(ticker, candle, provider)
        user = interaction.user
        if "error" in result:
            logging.error(f"/techs error: {result['error']}")
            await user.send(f"❌ {result['error']}")
        else:
            embed_dict = result["embeds"][0]
            embed = nextcord.Embed(
                title=embed_dict.get("title"),
                description=embed_dict.get("description"),
                color=embed_dict.get("color", 0x808080)
            )
            for field in embed_dict.get("fields", []):
                embed.add_field(name=field["name"], value=field["value"], inline=field.get("inline", False))
            if "footer" in embed_dict:
                embed.set_footer(text=embed_dict["footer"].get("text", ""))
            if "timestamp" in embed_dict:
                embed.timestamp = datetime.now()
            await user.send(embed=embed)
            logging.info(f"/techs sent embed to user {user}")
        await interaction.followup.send("✅ Check your DMs for the technical analysis!", ephemeral=True)
    except Exception as e:
        logging.exception(f"Exception in /techs command: {e}")
        await interaction.followup.send(f"❌ Error: {e}", ephemeral=True)

bot.run(TOKEN)