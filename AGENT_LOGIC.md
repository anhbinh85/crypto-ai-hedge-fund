# Agent Logic Overview

This document summarizes the core logic and decision rules for each agent in the CryptoConsult AI system.

---

## 🤖 Ben Graham
- **Philosophy:** Value investing, margin of safety, undervaluation.
- **Key Indicators:** Price vs. 90-period MA, deep value, over/undervaluation.
- **Entry:** Buy if price is significantly below 90-MA; short if above.
- **Exit:** Opposite threshold or if price returns to average.

## 🦈 Bill Ackman
- **Philosophy:** Activist investing, momentum, bold moves on strong signals.
- **Key Indicators:** RSI, price vs. MA, momentum.
- **Entry:** Buy on strong momentum and overbought; short on oversold.
- **Exit:** Opposite momentum or price crossing MA.

## 🏦 Binance Strategy
- **Philosophy:** Exchange-specific strategies (details depend on implementation).
- **Key Indicators:** May use volume, volatility, or proprietary signals.
- **Entry/Exit:** Exchange-driven logic.

## 🔄 Mean Reversion
- **Philosophy:** Prices revert to the mean over time.
- **Key Indicators:** Price deviation from moving average.
- **Entry:** Buy when price is far below mean; sell/short when above.
- **Exit:** When price returns to mean.

## 🦉 Michael Burry
- **Philosophy:** Contrarian, deep value, bubble detection.
- **Key Indicators:** Price vs. 90-MA, RSI extremes, market euphoria/panic.
- **Entry:** Short if overvalued/euphoric; buy if deeply undervalued/oversold.
- **Exit:** When value is restored or bubble bursts.

## 🗣️ Sentiment
- **Philosophy:** Market mood and news drive price.
- **Key Indicators:** News sentiment, price change %, social signals.
- **Entry:** Buy if sentiment is strongly positive; sell if negative.
- **Exit:** Sentiment reversal.

## 📈 Trend Following
- **Philosophy:** Ride established trends.
- **Key Indicators:** Moving averages, breakouts, trend strength.
- **Entry:** Buy on uptrend confirmation; sell/short on downtrend.
- **Exit:** Trend reversal or loss of momentum.

## 🧑‍💼 Warren Buffett
- **Philosophy:** Quality at a fair price, economic moat, patience.
- **Key Indicators:** Price vs. 90-MA, RSI, quality metrics.
- **Entry:** Buy if price is near lows and quality is high; sell if overvalued.
- **Exit:** When price exceeds fair value or quality deteriorates.

## 🦄 Cathie Wood
- **Philosophy:** Growth, innovation, disruptive tech.
- **Key Indicators:** Price momentum, innovation trends, growth signals.
- **Entry:** Buy on strong growth/innovation signals.
- **Exit:** Growth slows or reverses.

## 🧓 Charlie Munger
- **Philosophy:** Quality business at a fair price, simplicity, patience.
- **Key Indicators:** Price vs. 90-MA, quality metrics.
- **Entry:** Buy if price is near lows and business is high quality.
- **Exit:** Price exceeds fair value or quality drops.

## 🧢 Peter Lynch
- **Philosophy:** Growth at a reasonable price, "ten-baggers".
- **Key Indicators:** Price vs. 60-MA, RSI, growth signals.
- **Entry:** Buy on strong uptrend and momentum.
- **Exit:** Growth slows or price drops below average.

## 🐂 Stanley Druckenmiller
- **Philosophy:** Asymmetric risk-reward, momentum, conviction.
- **Key Indicators:** Price vs. 90-MA, RSI, MACD, growth, sentiment, insider activity.
- **Entry:** Buy on strong macro trend and momentum; short on reversal.
- **Exit:** Trend reversal or stop-loss.

## 🔬 Phil Fisher
- **Philosophy:** Quality growth, deep research, long-term holding.
- **Key Indicators:** Price vs. 60-MA, RSI, growth signals.
- **Entry:** Buy on strong growth and quality signals.
- **Exit:** Growth slows or reverses.

## 💰 Valuation
- **Philosophy:** Buy undervalued, sell overvalued assets.
- **Key Indicators:** Price vs. intrinsic value, valuation ratios.
- **Entry:** Buy if price is below intrinsic value by a margin.
- **Exit:** Price approaches or exceeds fair value.

## 📊 Fundamentals
- **Philosophy:** Strong fundamentals drive long-term value.
- **Key Indicators:** Revenue, earnings, NVT ratio, growth rates.
- **Entry:** Buy if fundamentals are strong.
- **Exit:** Fundamentals weaken.

## 🛠️ Technicals
- **Philosophy:** Price action and indicators predict moves.
- **Key Indicators:** MA crossovers, RSI, MACD, breakouts.
- **Entry:** Buy on bullish technical signals; sell on bearish.
- **Exit:** Opposite technical signal.

## 🛡️ Risk Manager
- **Philosophy:** Control risk, avoid large losses.
- **Key Indicators:** Volatility, drawdown, position sizing.
- **Entry/Exit:** Adjusts or closes positions if risk exceeds threshold.

## 📁 Portfolio Manager
- **Philosophy:** Diversification, allocation, rebalancing.
- **Key Indicators:** Asset weights, correlations, performance.
- **Entry/Exit:** Rebalances or reallocates based on portfolio rules.

---

*For more details, see the agent source code in `src/agents/`.* 