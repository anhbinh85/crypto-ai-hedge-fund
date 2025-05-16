# Binance Trading Bot

A cryptocurrency trading bot that uses technical analysis to make trading decisions on Binance. The bot implements a strategy based on RSI and MACD indicators, with configurable risk management parameters.

## Features

- Real-time trading on Binance
- Technical analysis using RSI and MACD indicators
- Configurable trading pairs and parameters
- Risk management with stop-loss and take-profit
- Position sizing based on portfolio percentage
- Detailed logging of all trading activities

## Prerequisites

- Python 3.8 or higher
- Binance account with API access
- API key and secret from Binance

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file in the project root with your Binance API credentials:
```env
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

## Configuration

You can configure the trading bot by setting environment variables in the `.env` file:

```env
# API Configuration
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here

# Trading Parameters
TRADING_PAIRS=BTCUSDT,ETHUSDT,BNBUSDT  # Comma-separated list of trading pairs
BASE_CURRENCY=USDT
QUOTE_CURRENCY=USDT

# Order Parameters
DEFAULT_ORDER_TYPE=LIMIT
DEFAULT_TIME_IN_FORCE=GTC

# Risk Management
MAX_POSITION_SIZE=0.1  # Maximum position size as fraction of portfolio
STOP_LOSS_PCT=0.02    # Stop loss percentage
TAKE_PROFIT_PCT=0.04  # Take profit percentage

# Time Intervals
CANDLE_INTERVAL=1h    # Candle interval for technical analysis
UPDATE_INTERVAL=300   # Update interval in seconds (5 minutes)

# Technical Analysis Parameters
RSI_PERIOD=14
RSI_OVERBOUGHT=70
RSI_OVERSOLD=30
MACD_FAST=12
MACD_SLOW=26
MACD_SIGNAL=9
```

## Usage

1. Start the trading bot:
```bash
python src/main.py
```

The bot will:
- Connect to Binance using your API credentials
- Monitor the configured trading pairs
- Execute trades based on technical analysis signals
- Manage positions with stop-loss and take-profit
- Log all activities to `trading_bot.log`

## Trading Strategy

The bot uses a combination of RSI and MACD indicators to generate trading signals:

### Buy Signal
- RSI is below the oversold threshold (default: 30)
- MACD line crosses above the signal line

### Sell Signal
- RSI is above the overbought threshold (default: 70)
- MACD line crosses below the signal line

### Risk Management
- Stop-loss: Automatically sells when price drops by the configured percentage
- Take-profit: Automatically sells when price increases by the configured percentage
- Position sizing: Limits each position to a maximum percentage of the portfolio

## Logging

The bot logs all activities to `trading_bot.log`, including:
- Trading signals
- Order executions
- Position updates
- Error messages
- Account balance changes

## Disclaimer

This trading bot is for educational purposes only. Use it at your own risk. Cryptocurrency trading involves significant risk of loss and is not suitable for all investors. Past performance is not indicative of future results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
