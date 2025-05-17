import os
import requests
import nextcord
from nextcord.ext import commands
from dotenv import load_dotenv
load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
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
from src.agents.deepseekAIHelper import analyze_indicators_with_llm
from src.mongodb_db.cex_btc import get_cex_btc_data

# Load environment variables
# print("DEEPSEEK_API_KEY:", os.getenv("DEEPSEEK_API_KEY"))
# print("GEMINI_API_KEY:", os.getenv("GEMINI_API_KEY"))
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
    "ben_graham": "🏦 Value investing legend. Focuses on undervalued assets and margin of safety.",
    "bill_ackman": "🦈 Activist investor. Looks for catalysts and deep value opportunities.",
    "binance_strategy": "🤖 Binance Strategy: Multi-indicator technical strategy using RSI, MACD, SMA(20), Ichimoku, ADX, Stochastic, Stoch RSI, ATR, VWAP, and Volume. Combines trend, momentum, and volatility for robust signals!",
    "mean_reversion": "🔄 Buys low, sells high. Expects prices to revert to the mean.",
    "michael_burry": "💣 Contrarian investor. Famous for spotting bubbles and market mispricings.",
    "sentiment": "📰 Analyzes market sentiment from news and social data.",
    "trend_following": "📈 Follows price trends. Rides momentum up or down.",
    "warren_buffett": "🧑‍💼 Long-term value investor. Focuses on quality businesses and compounding.",
    "cathie_wood": "🚀 Growth and innovation investor. Focuses on disruptive tech.",
    "charlie_munger": "🧠 Buffett's partner. Focuses on mental models and rational investing.",
    "peter_lynch": "🧢 Growth at a reasonable price. Invests in what he understands.",
    "stanley_druckenmiller": "🌎 Macro trader. Focuses on big economic trends and timing.",
    "phil_fisher": "🔬 Scuttlebutt and qualitative analysis. Looks for outstanding companies.",
    "valuation": "💰 Estimates fair value using financial models.",
    "fundamentals": "📊 Analyzes company financials and health.",
    "technicals": "📊 Uses technical indicators and price action."
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
    ticker = ticker.upper()
    timeframe = timeframe.lower()
    # Check for uppercase 'M' in timeframe
    if 'M' in timeframe:
        await interaction.followup.send("❌ Please use lowercase 'm' for minutes (e.g., '1m', '15m'). Uppercase 'M' means 'month' on Binance.", ephemeral=True)
        return
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
            # Friendly error for user input
            try:
                err_json = response.json()
                err_msg = err_json.get("error", "Unknown error")
            except Exception:
                err_msg = response.text
            if "Invalid agent" in err_msg:
                err_msg += "\n➡️ Please use `/agents` to see the correct agent IDs."
            if "Invalid timeframe" in err_msg:
                err_msg += "\n➡️ Please check the supported timeframes (e.g., 1m, 5m, 15m, 1h, 1d)."
            embed = nextcord.Embed(
                title="❌ Input Error",
                description=err_msg,
                color=nextcord.Color.red()
            )
    except Exception as e:
        embed = nextcord.Embed(
            title="❌ CryptoConsult Error",
            description="An error occurred. Please check your input (ticker, agent, timeframe) and try again. Use `/agents` for agent IDs.",
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
    embed = nextcord.Embed(title="🧠 Available Agents", color=nextcord.Color.gold())
    for agent_id, desc in AGENT_DESCRIPTIONS.items():
        # Extract emoji and description
        parts = desc.split(' ', 1)
        emoji = parts[0] if len(parts) > 1 else ''
        description = parts[1] if len(parts) > 1 else desc
        display_name = agent_id.replace('_', ' ').title()
        field_title = f"{emoji} **`{agent_id}` {display_name}**"
        embed.add_field(name=field_title, value=description, inline=False)
    embed.set_footer(text="Tip: Use /consult with any agent for a detailed, actionable analysis!")
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
    ticker = ticker.upper()
    candle = candle.lower()
    if 'M' in candle:
        await interaction.followup.send("❌ Please use lowercase 'm' for minutes (e.g., '1m', '15m'). Uppercase 'M' means 'month' on Binance.", ephemeral=True)
        return
    if observations > 200:
        await interaction.followup.send("❌ Observations cannot exceed 200. Please choose a value between 1 and 200.", ephemeral=True)
        return
    # Send confirmation immediately
    await interaction.followup.send("✅ Backtest started! Results will be sent to your DM.", ephemeral=True)
    try:
        agent = get_agent_instance(agent_name)
        if agent is None:
            await interaction.user.send(f"❌ Agent '{agent_name}' not found! Use `/agents` to see valid agent IDs.")
            return
        try:
            result, _ = run_backtest(
                ticker=ticker,
                timeframe=candle,
                agents=agent_name,
                observations=observations,
                leverage=leverage
            )
        except Exception as e:
            err_msg = str(e)
            if "Not enough historical data" in err_msg:
                await interaction.user.send(f"❌ Not enough historical data for your request. Please try a lower number of observations or a more common symbol/timeframe.")
                return
            else:
                await interaction.user.send(f"❌ Error running backtest: {err_msg}")
                return
        if result is None:
            await interaction.user.send("❌ Failed to run backtest. Please check your parameters.")
            return
        action_counts = {}
        for action in result['actions_log']:
            action_counts[action['action']] = action_counts.get(action['action'], 0) + 1
        # Only show the summary in the embed
        agent_display = agent_name.replace('_', ' ').title()
        action_emoji = {
            'HOLD': ':pause_button:',
            'LONG': ':arrow_up:',
            'SHORT': ':arrow_down:',
            'BUY': ':shopping_cart:',
            'SELL': ':money_with_wings:',
            'STOP_LOSS': ':octagonal_sign:',
            'TAKE_PROFIT': ':tada:',
            'REVERSE': ':arrows_counterclockwise:',
            'CLOSE': ':stop_button:',
            'LIQUIDATION': ':skull_crossbones:'
        }
        actions_summary = ' '.join(
            f"{action_emoji.get(action, '')} `{action}` × {count}" for action, count in action_counts.items()
        )
        color = nextcord.Color.green() if result['total_return'] > 0 else nextcord.Color.red() if result['total_return'] < 0 else nextcord.Color.blue()
        embed = nextcord.Embed(
            title=f":chart_with_upwards_trend: Backtest Results for {ticker.upper()} ({candle}) — {agent_display}",
            description=(
                f":robot: **{agent_display} agent returned `{result['total_return']*100:.2f}%` with a `{result['win_rate']*100:.2f}%` win rate over `{observations}` candles (`{leverage}x` leverage).**\n"
                f"---\n"
                f"**:moneybag: Total Return:** `{result['total_return']*100:.2f}%`\n"
                f"**:vertical_traffic_light: Max Drawdown:** `{result['max_drawdown']*100:.2f}%`\n"
                f"**:dart: Win Rate:** `{result['win_rate']*100:.2f}%`\n"
                f"**:repeat: Number of Trades:** `{result['num_trades']}`\n"
                f"**:bank: Final Balance:** `${result['final_balance']:.2f}`\n"
                f"**:trophy: Best Trade:** `{result['best_trade']:.2f}%`\n"
                f"**:warning: Worst Trade:** `{result['worst_trade']:.2f}%`\n"
                f"---\n"
                f"**:clipboard: Agent Action Summary:**\n{actions_summary}\n"
                f"---\n"
                f":page_facing_up: **Detailed backtest CSV attached!**\n"
                f"_Powered by CryptoConsult AI Backtest_"
            ),
            color=color
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
        await interaction.user.send("❌ Error running backtest. Please check your input (ticker, agent, timeframe). Use `/agents` for agent IDs.")

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
    candle: str = nextcord.SlashOption(description="Candle interval (e.g., 1m, 5m, 15m, 1h, 1d)")
):
    logging.basicConfig(level=logging.INFO)
    logging.info(f"/techs called with ticker={ticker}, candle={candle}, user={interaction.user}")
    await interaction.response.defer(ephemeral=True)
    ticker = ticker.upper()
    candle = candle.lower()
    if 'M' in candle:
        await interaction.followup.send("❌ Please use lowercase 'm' for minutes (e.g., '1m', '15m'). Uppercase 'M' means 'month' on Binance.", ephemeral=True)
        return
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

@bot.slash_command(name="tickers_binance", description="List all Binance tickers and their support for Spot, Margin, and Futures trading (CSV export)")
async def tickers_binance(interaction: nextcord.Interaction):
    await interaction.response.defer(ephemeral=True)
    margin_error = None
    futures_error = None
    try:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        provider = BinanceDataProvider(api_key, api_secret)
        spot_info = provider.client.get_exchange_info()
        spot_symbols = {s['symbol']: s for s in spot_info['symbols']}
        # Margin support
        margin_pairs = set()
        try:
            margin_info = provider.client.get_margin_all_pairs()
            margin_pairs = set(pair['symbol'] for pair in margin_info)
        except Exception as e:
            margin_error = str(e)
            print(f"[ERROR] Could not fetch margin pairs: {margin_error}")
        # Futures support
        futures_symbols = set()
        try:
            from binance.um_futures import UMFutures  # This is correct for binance-futures-connector
            futures_client = UMFutures(key=api_key, secret=api_secret)
            fut_info = futures_client.exchange_info()
            for s in fut_info['symbols']:
                futures_symbols.add(s['symbol'])
        except Exception as e:
            futures_error = str(e)
            print(f"[ERROR] Could not fetch futures pairs: {futures_error}")
        # Compose CSV
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["Symbol", "Base Asset", "Quote Asset", "Spot", "Margin", "Futures", "Status"])
        for symbol, info in spot_symbols.items():
            base = info.get('baseAsset', '')
            quote = info.get('quoteAsset', '')
            spot = 'Yes' if info.get('status', '') == 'TRADING' else 'No'
            margin = 'Yes' if symbol in margin_pairs else 'No'
            futures = 'Yes' if symbol in futures_symbols else 'No'
            status = info.get('status', '')
            writer.writerow([symbol, base, quote, spot, margin, futures, status])
        output.seek(0)
        csv_file = nextcord.File(fp=output, filename="binance_tickers.csv")
        msg = "📄 Binance Tickers (Spot, Margin, Futures):"
        if margin_error or futures_error:
            msg += "\n⚠️ Some data could not be fetched:\n"
            if margin_error:
                msg += f"• Margin: {margin_error}\n"
            if futures_error:
                msg += f"• Futures: {futures_error}\n"
        await interaction.user.send(msg, file=csv_file)
        await interaction.followup.send("✅ Check your DMs for the full Binance tickers list!", ephemeral=True)
    except Exception as e:
        await interaction.followup.send(f"❌ Error fetching tickers: {e}", ephemeral=True)

@bot.slash_command(
    name="whales",
    description="Show the biggest whale orders for a Binance ticker",
    guild_ids=[1214331987666542642]
)
async def whales(
    interaction: nextcord.Interaction,
    ticker: str = nextcord.SlashOption(description="Ticker symbol (e.g. BTCUSDT)")
):
    await interaction.response.defer(ephemeral=True)
    ticker = ticker.upper()
    try:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        provider = BinanceDataProvider(api_key, api_secret)
        # Fetch order book (depth)
        depth = provider.client.get_order_book(symbol=ticker, limit=100)
        bids = depth.get('bids', [])
        asks = depth.get('asks', [])
        if not bids or not asks:
            await interaction.user.send(f"❌ No order book data found for {ticker}.")
            await interaction.followup.send("❌ No whale data found. Check your DM for details.", ephemeral=True)
            return
        # Find biggest bid and ask by quantity
        biggest_bid = max(bids, key=lambda x: float(x[1]))
        biggest_ask = max(asks, key=lambda x: float(x[1]))
        bid_price, bid_qty = float(biggest_bid[0]), float(biggest_bid[1])
        ask_price, ask_qty = float(biggest_ask[0]), float(biggest_ask[1])
        bid_value = bid_price * bid_qty
        ask_value = ask_price * ask_qty
        embed = nextcord.Embed(
            title=f":whale: Whale Orders for {ticker}",
            description=(
                f"**:money_with_wings: Biggest Buy (Bid):**\n"
                f"> **Price:** `${bid_price:,.8f}`\n"
                f"> **Quantity:** `{bid_qty:,.4f}`\n"
                f"> **Value:** `${bid_value:,.2f}`\n\n"
                f"**:moneybag: Biggest Sell (Ask):**\n"
                f"> **Price:** `${ask_price:,.8f}`\n"
                f"> **Quantity:** `{ask_qty:,.4f}`\n"
                f"> **Value:** `${ask_value:,.2f}`"
            ),
            color=nextcord.Color.purple()
        )
        embed.set_footer(text="Powered by CryptoConsult AI Whales")
        await interaction.user.send(embed=embed)
        await interaction.followup.send("✅ Whale order details sent to your DM!", ephemeral=True)
    except Exception as e:
        await interaction.user.send(f"❌ Error fetching whale orders: {e}")
        await interaction.followup.send("❌ Error fetching whale orders. Check your DM for details.", ephemeral=True)

@bot.slash_command(
    name="techs_analysis",
    description="AI analysis of technical indicators for a ticker and candle interval",
    guild_ids=[1214331987666542642]  # Remove or adjust for global use
)
async def techs_analysis(
    interaction: nextcord.Interaction,
    ticker: str = nextcord.SlashOption(description="Ticker symbol (e.g. BTCUSDT)"),
    candle: str = nextcord.SlashOption(description="Candle interval (e.g., 1m, 15m, 1h, 1d)", default="1h"),
    model: str = nextcord.SlashOption(description="LLM model (deepseek/gemini)", required=False, default="deepseek", choices=["deepseek", "gemini"])
):
    await interaction.response.defer(ephemeral=True)
    ticker = ticker.upper()
    candle = candle.lower()
    try:
        api_key = os.getenv("BINANCE_API_KEY", "")
        api_secret = os.getenv("BINANCE_API_SECRET", "")
        provider = BinanceDataProvider(api_key, api_secret)
        # Use your existing get_techs_embed to get the indicator summary
        result = get_techs_embed(ticker, candle, provider)
        if "error" in result:
            await interaction.user.send(f"❌ {result['error']}")
            await interaction.followup.send("❌ Error fetching indicators. Check your DM for details.", ephemeral=True)
            return
        # Compose a summary string for the LLM (truncate if needed)
        embed_dict = result["embeds"][0]
        indicator_summary = embed_dict.get("description", "")
        for field in embed_dict.get("fields", []):
            indicator_summary += f"\n{field['name']}: {field['value']}"
        # Truncate to ~2000 chars for LLM and Discord safety
        indicator_summary = indicator_summary[:2000]
        # Call LLM
        analysis = analyze_indicators_with_llm(indicator_summary, model_preference=model)
        # Compose Discord embed
        embed = nextcord.Embed(
            title=f"🤖 Techs Analysis for {ticker} ({candle})",
            description=f"**AI Model:** `{model}`\n\n{analysis}",
            color=nextcord.Color.teal()
        )
        embed.set_footer(text="Powered by DeepSeek/Gemini AI | Data: Binance")
        await interaction.user.send(embed=embed)
        await interaction.followup.send("✅ Techs analysis sent to your DM!", ephemeral=True)
    except Exception as e:
        await interaction.user.send(f"❌ Error in /techs_analysis: {e}")
        await interaction.followup.send("❌ Error in /techs_analysis. Check your DM for details.", ephemeral=True)

@bot.slash_command(
    name="cex_btc",
    description="Show BTC inflow/outflow to CEXs in USD for a given timeframe.",
    guild_ids=[1214331987666542642]  # Replace with your actual guild ID
)
async def cex_btc(
    interaction: nextcord.Interaction,
    last: str = nextcord.SlashOption(description="Time window: 'latest', '1hour', ..., '24hours'")
):
    await interaction.response.defer(ephemeral=True)
    data, price = get_cex_btc_data(last)
    if data is None:
        await interaction.followup.send(f"❌ {price}", ephemeral=True)
        return

    # Sort by total USD volume (inflow + outflow)
    data_sorted = sorted(data, key=lambda x: (x[2] + x[4]), reverse=True)

    # Only show top 10 CEXs
    data_sorted = data_sorted[:10]

    # Calculate total inflow/outflow for summary
    total_inflow_btc = sum(x[1] for x in data_sorted)
    total_inflow_usd = sum(x[2] for x in data_sorted)
    total_outflow_btc = sum(x[3] for x in data_sorted)
    total_outflow_usd = sum(x[4] for x in data_sorted)

    # Prepare summary for top 3 CEXs
    top3 = data_sorted[:3]
    summary_lines = []
    for cex, in_btc, in_usd, out_btc, out_usd in top3:
        summary_lines.append(f"**{cex.title()}**: ⬆️ `${in_usd:,.0f}` ⬇️ `${out_usd:,.0f}`")
    summary = "\n".join(summary_lines)

    # Eye-catching header and totals
    header = (
        f"<:btc:1214331987666542642> **BTC Inflow/Outflow to CEXs ({last}):**\n"
        f"**BTC/USD price:** `${price:,.2f}`\n"
        f"\n:star: **Top CEXs by Total Flow:**\n{summary}\n"
        f"\n💧 **Total Inflow:** `${total_inflow_btc:,.2f} BTC` (`${total_inflow_usd:,.0f}`)\n"
        f"🔥 **Total Outflow:** `${total_outflow_btc:,.2f} BTC` (`${total_outflow_usd:,.0f}`)\n"
        f"\n```\n"
        f"{'CEX':<12} {'In (BTC)':>10} {'In ($)':>12} {'Out (BTC)':>10} {'Out ($)':>12}\n"
        f"{'-'*58}\n"
    )
    # Build table rows (fit within 58 chars width)
    table_rows = []
    for cex, in_btc, in_usd, out_btc, out_usd in data_sorted:
        table_rows.append(f"{cex[:12]:<12} {in_btc:10.2f} {in_usd:12,.0f} {out_btc:10.2f} {out_usd:12,.0f}\n")
    # Split into chunks <= 2000 chars
    chunks = []
    current_chunk = header
    for row in table_rows:
        if len(current_chunk) + len(row) + 4 > 2000:  # +4 for closing code block and info
            current_chunk += "```\n"
            chunks.append(current_chunk)
            current_chunk = "```\n" + row
        else:
            current_chunk += row
    current_chunk += "```\n:information_source: **Top 10 CEXs by (Inflow $ + Outflow $)**\n"
    chunks.append(current_chunk)
    # Send each chunk as a DM
    for i, chunk in enumerate(chunks):
        if i == 0:
            await interaction.user.send(chunk)
        else:
            await interaction.user.send(f"(cont.)\n" + chunk)
    await interaction.followup.send("✅ Check your DMs for the CEX BTC inflow/outflow report!", ephemeral=True)

bot.run(TOKEN)