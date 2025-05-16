# AI Hedge Fund: Multi-Agent Crypto Trading Bot

A modular, multi-agent cryptocurrency trading bot inspired by legendary investors (e.g., Warren Buffett, Ben Graham, Cathie Wood, Bill Ackman, Michael Burry, and more). Features real-time trading, advanced technical/fundamental analysis, backtesting, and a beautiful Discord bot interface for actionable, detailed insights.

---

## 🚀 Features

- **Multi-Agent System**: Each agent mimics a famous investor or strategy (value, growth, macro, technical, sentiment, etc.).
- **Discord Bot**: Get detailed, actionable, and visually appealing trading analysis via Discord DMs and slash commands.
- **Backtesting Engine**: Simulate strategies and agents on historical data with detailed performance metrics.
- **Technical & Fundamental Analysis**: Combines TA (RSI, MACD, etc.) and FA (valuation, financials, sentiment).
- **Risk Management**: Position sizing, stop-loss, take-profit, and risk scoring.
- **Modular & Extensible**: Easily add new agents, strategies, or data sources.
- **Logging**: All trades, signals, and errors are logged for transparency.

---

## 🏗️ Architecture Overview

- `src/agents/`: Investor-inspired agents (Buffett, Graham, Ackman, Burry, Wood, Munger, Lynch, Druckenmiller, Fisher, etc.)
- `src/backtest/`: Backtesting engine and utilities
- `src/tools/`: Exchange executors, API helpers
- `src/data/`: Data providers (Binance, cache, models)
- `src/config/`: Configuration management
- `src/llm/`: LLM/AI integration (optional)
- `discord_bot.py`: Discord bot interface (slash commands, DMs, agent registry)
- `src/main.py`: CLI entry point for live trading

---

## 🧑‍💼 Available Agents

- **ben_graham**: Value investing, margin of safety
- **warren_buffett**: Quality, long-term compounding
- **bill_ackman**: Activist, catalyst-driven
- **michael_burry**: Contrarian, bubble-spotting
- **cathie_wood**: Growth, innovation
- **charlie_munger**: Mental models, rationality
- **peter_lynch**: Growth at reasonable price
- **stanley_druckenmiller**: Macro, trend timing
- **phil_fisher**: Qualitative, scuttlebutt
- **binance_strategy**: Classic TA (RSI, MACD)
- **mean_reversion**: Buys low, sells high
- **trend_following**: Momentum/trend
- **sentiment**: News/social sentiment
- **valuation**: Fair value models
- **fundamentals**: Financial health
- **technicals**: Price action, indicators

---

## 🤖 Discord Bot Usage

- **Slash Commands** (work in DMs and servers):
  - `/consult ticker:BTCUSDT timeframe:1h agents:ben_graham,bill_ackman,...`
    - Get a unified, detailed analysis from multiple agents (DM only, with channel confirmation).
  - `/agents` — List all available agents and their descriptions.
  - `/backtest ticker:BTCUSDT agent_name:ben_graham candle:1h observations:100 leverage:1.0`
    - Run a backtest simulation for any agent.
  - `/techs ticker:BTCUSDT candle:1h` — Show technical analysis summary.

- **Response Format**: Each agent's block includes:
  - Technical indicators
  - Signal, action, confidence
  - Reasoning
  - Entry/exit/reversal/suggested duration
  - Technical summary
- **Multi-agent summary**: Final decision, risk assessment, and actionable recommendation (with emojis and Markdown for clarity).
- **All analysis is sent via DM for privacy.**

---

## 🧪 Backtesting

- Simulate any agent or ensemble on historical data.
- Supports custom timeframes, leverage, and candle counts.
- Results include PnL, win/loss, drawdown, and more.
- Example:
  ```bash
  # From Discord:
  /backtest ticker:BTCUSDT agent_name:ben_graham candle:1h observations:100
  # Or from CLI:
  python src/backtest/backtest_engine.py --ticker BTCUSDT --agent ben_graham --timeframe 1h --observations 100
  ```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/anhbinh85/crypto-ai-hedge-fund.git
   cd crypto-ai-hedge-fund
   ```
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables:**
   - Create a `.env` file in the root directory:
     ```env
     BINANCE_API_KEY=your_api_key_here
     BINANCE_API_SECRET=your_api_secret_here
     DISCORD_BOT_TOKEN=your_discord_token_here
     # Optional: ALLOWED_GUILD_ID=your_guild_id
     ```

---

## 🔧 Configuration

- Edit `.env` for API keys and trading parameters (see sample in repo).
- Advanced config in `src/config/binance_config.py`.

---

## 🏃 Usage

- **Start the trading bot (CLI):**
  ```bash
  python src/main.py
  ```
- **Start the Discord bot:**
  ```bash
  python discord_bot.py
  ```
- **Interact via Discord DMs using slash commands.**

---

## 📦 Dependencies

- Python 3.8+
- numpy, pandas, talib, pandas_ta, python-binance, nextcord, python-dotenv, requests, ratelimit

---

## 📜 License

MIT License. See [LICENSE](LICENSE).

---

## ⚠️ Disclaimer

This bot is for educational purposes only. Crypto trading is risky. Use at your own risk. No guarantees of profit or safety.
